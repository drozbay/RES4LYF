import torch
from torch import Tensor
from tqdm.auto import trange
from tqdm import tqdm
import math
import latent_preview
from typing import Callable
import re


from tqdm import trange

import torch.nn.functional as F
import torchvision.transforms as T


import comfy.model_patcher
import comfy.sample
import comfy.samplers
from comfy.samplers import CFGGuider, sampling_function


from .noise_sigmas_timesteps_scaling import get_res4lyf_step_with_model, get_res4lyf_half_step3, get_alpha_ratio_from_sigma_down

from ..beta.noise_classes import NOISE_GENERATOR_CLASSES, NOISE_GENERATOR_NAMES, prepare_noise

from ..beta.phi_functions import phi, Phi

from ..beta.rk_coefficients_beta import rk_coeff

from ..beta.rk_method_beta import RK_Method_Beta

from ..helper import initialize_or_scale, get_extra_options_kv, extra_options_flag
from ..latents import get_collinear, get_orthogonal, slerp, lagrange_interpolation, normalize_zscore, slerp_tensor, line_intersection


from . import res



class ExtraOptions():
    def __init__(self, extra_options):
        self.extra_options = extra_options
        
    def __call__(self, option, default=None, ret_type=None, match_all_flags=False):
        if isinstance(option, (tuple, list)):
            if match_all_flags:
                return all(self(single_option, default, ret_type) for single_option in option)
            else:
                return any(self(single_option, default, ret_type) for single_option in option)

        if default is None: # get flag
            pattern = rf"^(?:{re.escape(option)}\s*$|{re.escape(option)}=)"
            return bool(re.search(pattern, self.extra_options, flags=re.MULTILINE))
        elif ret_type is None:
            ret_type = type(default)
        
            if ret_type.__module__ != "builtins":
                mod = __import__(default.__module__)
                ret_type = lambda v: getattr(mod, v, None)
        
        if ret_type == list:
            pattern = rf"^{re.escape(option)}\s*=\s*([a-zA-Z0-9_.,+-]+)\s*$"
            match   = re.search(pattern, self.extra_options, flags=re.MULTILINE)
            
            if match:
                value = match.group(1)
            else:
                value = default
                
            if type(value) == str:
                value = value.split(',')
            
                if type(default[0]) == type:
                    ret_type = default[0]
                else:
                    ret_type = type(default[0])
                
                value = [ret_type(value[_]) for _ in range(len(value))]
        
        else:
            pattern = rf"^{re.escape(option)}\s*=\s*([a-zA-Z0-9_.+-]+)\s*$"
            match = re.search(pattern, self.extra_options, flags=re.MULTILINE)
            if match:
                value = ret_type(match.group(1))
            else:
                value = default
        
        return value

def phi_remainder(j, neg_h):
    remainder = torch.zeros_like(neg_h)
    
    for k in range(j): 
        remainder += (neg_h)**k / math.factorial(k)
    phi_j_h = ((neg_h).exp() - remainder) / (neg_h)**j
    
    return phi_j_h


def calculate_gamma(c2, c3):
    return (3*(c3**3) - 2*c3) / (c2*(2 - 3*c2))

def epsilon(model, x, sigma, **extra_args):
    s_in = x.new_ones([x.shape[0]])
    data = model(x, sigma * s_in, **extra_args)
    eps = (x - data) / (sigma * s_in) 
    return eps, data


def epsilon_res(model, x, sigma, **extra_args):
    s_in = x.new_ones([x.shape[0]])
    data = model(x, sigma * s_in, **extra_args)
    eps = data - x
    return eps, data


def epsilon_lin(model, x, sigma, **extra_args):
    s_in = x.new_ones([x.shape[0]])
    data = model(x, sigma * s_in, **extra_args)
    eps = (x - data) / (sigma * s_in) 
    return eps, data

def epsilon_log(model, x, sigma, **extra_args):
    s_in = x.new_ones([x.shape[0]])
    data = model(x, sigma * s_in, **extra_args)
    eps = data - x
    return eps, data


def res_2m_predictor_step(x_0, denoised_1, denoised_2, sigma_prev, sigma, sigma_down,):
    
    h_prev = -torch.log(sigma/sigma_prev)
    h = -torch.log(sigma_down/sigma)

    c1 = 0
    c2 = (-h_prev / h).item()

    ci = [c1,c2]
    φ = Phi(h, ci, analytic_solution=True)

    b2 = φ(2)/c2
    b1 = φ(1) - b2
    
    eps_2 = denoised_1 - x_0
    eps_1 = denoised_2 - x_0

    h_a_k_sum = h * (b1 * eps_1 + b2 * eps_2)
    
    x = torch.exp(-h) * x_0 + h_a_k_sum
    
    denoised = x_0 + (sigma / (sigma - sigma_down)) * h_a_k_sum

    return x, denoised


#extra_args.setdefault("model_options", {}).setdefault("transformer_options", {}).update(transformer_options)
#RK.update_transformer_options({'model_call_type': 'base'})

class ModelCall():
    def __init__(self, model, x, sigmas, step=0, extra_args=None, EXPONENTIAL=False, RET_DATA=False):
        self.model       = model
        self.s_in        = x.new_ones([x.shape[0]])
        self.step        = step
        self.sigmas      = sigmas.clone()
        self.extra_args  = {} if extra_args is None else extra_args
        self.EXPONENTIAL = EXPONENTIAL
        self.RET_DATA    = RET_DATA
        
    def __call__(self, x, sigma=None, transformer_options=None, extra_args=None, step=None, EXPONENTIAL=None, RET_DATA=None):
        extra_args  = self.extra_args   if extra_args  is None else extra_args
        step        = self.step         if step        is None else step
        sigma       = self.sigmas[step] if sigma       is None else sigma
        RET_DATA    = self.RET_DATA     if RET_DATA    is None else RET_DATA
        EXPONENTIAL = self.EXPONENTIAL  if EXPONENTIAL is None else EXPONENTIAL
        
        if transformer_options is not None:
            extra_args.setdefault("model_options", {}).setdefault("transformer_options", {}).update(transformer_options)
        
        data = self.model(x, sigma * self.s_in, **extra_args)
        eps  = data - x if EXPONENTIAL else (x - data) / sigma

        if RET_DATA:
            return eps, data
        else:
            return eps
        
    def h_fn(self, step=None, sigma=None, sigma_next=None, EXPONENTIAL=None):
        step        = self.step           if step        is None else step
        sigma       = self.sigmas[step]   if sigma       is None else sigma
        sigma_next  = self.sigmas[step+1] if sigma_next  is None else sigma_next
        EXPONENTIAL = self.EXPONENTIAL    if EXPONENTIAL is None else EXPONENTIAL
        
        if EXPONENTIAL:
            return -torch.log(sigma_next/sigma)
        else:
            return sigma_next - sigma

    def sigma_to_t(self, sigma, EXPONENTIAL=None):
        EXPONENTIAL = self.EXPONENTIAL  if EXPONENTIAL is None else EXPONENTIAL

        t = -torch.log(sigma) if EXPONENTIAL else sigma
        return t

    def t_to_sigma(self, t, EXPONENTIAL=None):
        EXPONENTIAL = self.EXPONENTIAL  if EXPONENTIAL is None else EXPONENTIAL

        sigma = torch.exp(-t) if EXPONENTIAL else t
        return sigma
    
    def get_sigma_substep(self, ci, step=None, sigma=None, sigma_next=None, EXPONENTIAL=None):
        step        = self.step           if step        is None else step
        sigma       = self.sigmas[step]   if sigma       is None else sigma
        sigma_next  = self.sigmas[step+1] if sigma_next  is None else sigma_next
        EXPONENTIAL = self.EXPONENTIAL    if EXPONENTIAL is None else EXPONENTIAL
        
        h = self.h_fn(sigma=sigma, sigma_next=sigma_next, EXPONENTIAL=EXPONENTIAL)
        
        t_substep     = self.sigma_to_t(sigma,     EXPONENTIAL) + h*ci
        sigma_substep = self.t_to_sigma(t_substep, EXPONENTIAL)
        
        return sigma_substep
        
    def end_step(self):
        self.step += 1
    


class NoiseGen():
    def __init__(self, model, x, sigmas, seed=42, step=0, noise_sampler_type="gaussian", eta=0.5, noise_mode="hard"):
        self.model         = model
        self.sigmas        = sigmas.clone()
        self.sigma_min     = model.inner_model.inner_model.model_sampling.sigma_min
        self.sigma_max     = model.inner_model.inner_model.model_sampling.sigma_max
        self.step          = step
        self.noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=seed, sigma_min=self.sigma_min, sigma_max=self.sigma_max)
        self.noise_init    = normalize_zscore(self.noise_sampler(sigma=self.sigma_max, sigma_next=self.sigma_min), channelwise=True, inplace=True)
        self.noise_mode    = noise_mode
        self.eta           = eta
        
    def __call__(self, sigma=None, sigma_next=None, step=None):
        step       = self.step           if step       is None else step
        sigma      = self.sigmas[step]   if sigma      is None else sigma
        sigma_next = self.sigmas[step+1] if sigma_next is None else sigma_next
        
        return normalize_zscore(self.noise_sampler(sigma=sigma, sigma_next=sigma_next), channelwise=True, inplace=True)

    def linear_noise(self, x, sigma=None, sigma_next=None, step=None):
        step       = self.step           if step       is None else step
        sigma      = self.sigmas[step]   if sigma      is None else sigma
        sigma_next = self.sigmas[step+1] if sigma_next is None else sigma_next
        
        #noise = normalize_zscore(self.noise_sampler(sigma=sigma, sigma_next=sigma_next), channelwise=True, inplace=True)
        return (1-sigma)*x + sigma*self.noise_init
    
    def hard_noise(self, sigma, eta, sigma_max=1.0):
        sigma_up       = sigma * eta
        sigma_signal   = sigma_max - sigma
        sigma_residual = torch.sqrt(sigma**2 - sigma_up**2)
        alpha_ratio    = sigma_signal + sigma_residual
        sigma_down     = sigma_residual / alpha_ratio
        return alpha_ratio, sigma_down, sigma_up
    
    def swap_noise_xt_next(self, xt, yx0, eps_yx, sigma, eta, noise=None, noise_mode=None):
        eta        = self.eta            if eta        is None else eta
        noise_mode = self.noise_mode     if noise_mode is None else noise_mode
        
        alpha_ratio, sigma_down, sigma_up = self.hard_noise(sigma, eta)
        
        data_yx     = yx0 - sigma * eps_yx
        eps_xt_yx   = (xt - data_yx) / sigma
        
        xt_noised = alpha_ratio * (data_yx + sigma_down * eps_xt_yx) + sigma_up * noise
        return xt_noised
    
    def swap_noise_xt(self, xt, data_yx, sigma, sigma_next=None, step=None, eta=None, noise=None, noise_mode=None):
        eta        = self.eta            if eta        is None else eta
        noise_mode = self.noise_mode     if noise_mode is None else noise_mode
        step       = self.step           if step       is None else step
        sigma      = self.sigmas[step]   if sigma      is None else sigma
        sigma_next = self.sigmas[step+1] if sigma_next is None else sigma_next
        
        alpha_ratio, sigma_down, sigma_up = self.hard_noise(sigma, eta)
        
        eps_xt_yx  = (xt - data_yx) / sigma
        
        if noise is None:
            noise = normalize_zscore(self.noise_sampler(sigma=sigma, sigma_next=sigma_next), channelwise=True, inplace=True)
        
        xt_noised = alpha_ratio * (data_yx + sigma_down * eps_xt_yx) + sigma_up * noise
        return xt_noised

    def swap_noise_from_step(self, x_0, x_next, sigma, sigma_next, eta=None, noise_mode=None):
        eta        = self.eta            if eta        is None else eta
        noise_mode = self.noise_mode     if noise_mode is None else noise_mode
        
        sigma_up, sigma, sigma_down, alpha_ratio = get_res4lyf_step_with_model(self.model, sigma, sigma_next, eta, noise_mode)
        noise_eps_next = (xt - xt_2) / (sigma - sigma_2)
        noise_denoised_next = xt - sigma * noise_eps_next
        xt_2 = alpha_ratio * (noise_denoised_next + sigma_down * noise_eps_next) + sigma_up * noise

    def end_step(self):
        self.step += 1

def expand_dims(v, dims):
    """
    Expand the tensor `v` to the dim `dims`.

    Args:
        `v`: a PyTorch tensor with shape [N].
        `dim`: a `int`.
    Returns:
        a PyTorch tensor with shape [N, 1, 1, ..., 1] and the total dimension is `dims`.
    """
    return v[(...,) + (None,)*(dims - 1)]


class SigmaConvert:
    schedule = ""
    def marginal_log_mean_coeff(self, sigma):
        return 0.5 * torch.log(1 / ((sigma * sigma) + 1))              # (1/2) * torch.log(1/(1+sigma**2))

    def marginal_alpha(self, t):
        return torch.exp(self.marginal_log_mean_coeff(t))              # t(0.0)->1.0  t(1.0)->0.707  t(2.0)->0.447

    def marginal_std(self, t):
        return torch.sqrt(1. - torch.exp(2. * self.marginal_log_mean_coeff(t)))

    def marginal_lambda(self, t):
        """
        Compute lambda_t = log(alpha_t) - log(sigma_t) of a given continuous-time label t in [0, T].
        """
        log_mean_coeff = self.marginal_log_mean_coeff(t)
        log_std = 0.5 * torch.log(1. - torch.exp(2. * log_mean_coeff))
        return log_mean_coeff - log_std

"""

def sample_unipc(model, noise, sigmas, extra_args=None, callback=None, disable=False, variant='bh1'):
        timesteps = sigmas.clone()
        if sigmas[-1] == 0:
            timesteps = sigmas[:]
            timesteps[-1] = 0.001
        else:
            timesteps = sigmas.clone()
        ns = SigmaConvert()

        noise = noise / torch.sqrt(1.0 + timesteps[0] ** 2.0)
        model_type = "noise"

        #model_fn = model_wrapper(
        #    lambda input, sigma, **kwargs: predict_eps_sigma(model, input, sigma, **kwargs),
        #    ns,
        #    model_type=model_type,
        #    guidance_type="uncond",
        #    model_kwargs=extra_args,
        #)

        #order = min(3, len(timesteps) - 2)
        #uni_pc = UniPC(model_fn, ns, predict_x0=True, thresholding=False, variant=variant)
        x = uni_pc.sample(noise, timesteps=timesteps, skip_type="time_uniform", method="multistep", order=order, lower_order_final=True, callback=callback, disable_pbar=disable)
        x /= ns.marginal_alpha(timesteps[-1])   # renormalization?
        return x



    def sample(self, x, timesteps, t_start=None, t_end=None, order=3, skip_type='time_uniform',
        method='singlestep', lower_order_final=True, denoise_to_zero=False, solver_type='dpm_solver',
        atol=0.0078, rtol=0.05, corrector=False, callback=None, disable_pbar=False
    ):
        # t_0 = 1. / self.noise_schedule.total_N if t_end is None else t_end
        # t_T = self.noise_schedule.T if t_start is None else t_start
        steps = len(timesteps) - 1
        if method == 'multistep':
            assert steps >= order
            # timesteps = self.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=steps, device=device)
            assert timesteps.shape[0] - 1 == steps
            # with torch.no_grad():
            for step_index in trange(steps, disable=disable_pbar):
                if step_index == 0:
                    vec_t = timesteps[0].expand((x.shape[0]))
                    model_prev_list = [self.model_fn(x, vec_t)]
                    t_prev_list = [vec_t]
                elif step_index < order:
                    init_order = step_index
                # Init the first `order` values by lower order multistep DPM-Solver.
                # for init_order in range(1, order):
                    vec_t = timesteps[init_order].expand(x.shape[0])
                    x, model_x = self.multistep_uni_pc_update(x, model_prev_list, t_prev_list, vec_t, init_order, use_corrector=True)
                    if model_x is None:
                        model_x = self.model_fn(x, vec_t)
                    model_prev_list.append(model_x)
                    t_prev_list.append(vec_t)
                else:
                    extra_final_step = 0
                    if step_index == (steps - 1):
                        extra_final_step = 1
                    for step in range(step_index, step_index + 1 + extra_final_step):
                        vec_t = timesteps[step].expand(x.shape[0])
                        if lower_order_final:
                            step_order = min(order, steps + 1 - step)
                        else:
                            step_order = order
                        # print('this step order:', step_order)
                        if step == steps:
                            # print('do not run corrector at the last step')
                            use_corrector = False
                        else:
                            use_corrector = True
                        x, model_x =  self.multistep_uni_pc_update(x, model_prev_list, t_prev_list, vec_t, step_order, use_corrector=use_corrector)
                        for i in range(order - 1):
                            t_prev_list[i] = t_prev_list[i + 1]
                            model_prev_list[i] = model_prev_list[i + 1]
                        t_prev_list[-1] = vec_t
                        # We do not need to evaluate the final model value.
                        if step < steps:
                            if model_x is None:
                                model_x = self.model_fn(x, vec_t)
                            model_prev_list[-1] = model_x
                if callback is not None:
                    callback({'x': x, 'i': step_index, 'denoised': model_prev_list[-1]})
        else:
            raise NotImplementedError()
        # if denoise_to_zero:
        #     x = self.denoise_to_zero_fn(x, torch.ones((x.shape[0],)).to(device) * t_0)
        return x
"""

def sample_rk_uniuni(
    model, x, sigmas, extra_args=None, callback=None, disable=None,
    noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard",
    rk_type="dormand-prince", sigma_fn_formula="", t_fn_formula="",
    eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1,
    c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0,
    reverse_weight=0.0, extra_options="", cfg1=0, cfg2=0,
    cfg_cw=1.0, latent_guide=None,
    p=3, r_list=None, 
    order: int = 3,                # ← new, backward‐compatible
    lower_order_final: bool = True,
):
    """
    Multistep BH1 sampler in half-logSNR (λ=−logσ) space,
    with dynamic r_k computed from the last `order` steps.
    """
    extra_args = {} if extra_args is None else extra_args
    device, dtype = x.device, x.dtype
    s_in = x.new_ones([x.shape[0]])
    # precompute all lambdas = -log(sigmas)
    lambdas = -torch.log(sigmas)

    # history buffers
    lambda_hist = []  # most recent λ_n, λ_{n-1}, ...
    D_hist      = []  # corresponding drifts D0,D1,...

    for n in trange(len(sigmas)-1, disable=disable):
        σ_n      = sigmas[n]
        σ_np1    = sigmas[n+1]
        λ_n      = lambdas[n]
        λ_np1    = lambdas[n+1]
        h        = λ_np1 - λ_n

        # 1) compute the “zero‐th” drift at time σ_n
        α_n      = torch.sqrt(1 - σ_n**2)
        denoised = model(x, σ_n * s_in, **extra_args)
        eps_n    = (x - denoised) / σ_n
        D0       = - (α_n / σ_n) * eps_n

        # push into history
        lambda_hist.append(λ_n)
        D_hist.append(D0)

        # decide how many steps we have so far
        K = min(order, len(D_hist))

        if n == 0:
            # pure Euler for the very first step
            update = D0

        else:
            # 2) build the dynamic node ratios r_k = (λ_{n-k} - λ_n)/h, last one = 1
            rks = []
            for k in range(1, K):
                r = ((lambda_hist[-k-1] - λ_n) / h).item()
                rks.append(r)
            rks.append(1.0)
            rks = torch.tensor(rks, device=device, dtype=dtype)  # shape (K,)

            # 3) stack the last K drifts into a tensor of shape (K, *x.shape)
            D1s = torch.stack(D_hist[-K:], dim=0)  # (K, batch, ...)

            # 4) build the Vandermonde R[k,m] = (r_m * h)**k
            R = torch.stack([ (rks*h)**k for k in range(K) ], dim=1)  # (K,K)

            # 5) build the φ→b vector:
            #    φ₁ = exp(h) - 1*(?). Actually hφ₁ = expm1(h), φ_{k+1} = (φ_k - 1/k!)/h
            hh       = h
            hφ1      = torch.expm1(hh)
            φk       = hφ1 / hh - 1.0
            b_list   = []
            factorial = 1
            B_h       = hh  # BH1
            for k in range(1, K+1):
                b_list.append((φk * factorial) / B_h)
                factorial *= (k+1)
                φk = φk / hh - 1.0 / factorial
            b = torch.tensor(b_list, device=device, dtype=dtype)  # (K,)

            # 6) solve R · ρ = b  for the coefficients ρ
            if K == 1:
                ρ = torch.tensor([1.0], device=device, dtype=dtype)
            else:
                ρ = torch.linalg.solve(R, b)  # (K,)

            # 7) form the total update = sum_m ρ[m] * D1s[m]
            #    then x_{n+1} = x_n + σ_{n+1} * update
            update = torch.tensordot(D1s, ρ, dims=([0], [0]))  # shape of x

        # apply the step
        x = x + σ_np1 * update

        # if history too long, pop the oldest
        if len(D_hist) > order:
            D_hist.pop(0)
            lambda_hist.pop(0)

        # callback
        if callback is not None:
            callback({
                'x': x,
                'i': n,
                'sigma': σ_n,
                'sigma_next': σ_np1,
                'denoised': denoised
            })

    return x



def sample_rk_fedit_euler(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None, latent_guide_inv=None,mask=None):
    extra_args = {} if extra_args is None else extra_args
    
    model_options = extra_args.get('model_options', {})
    transformer_options = model_options.get('transformer_options', {})
    transformer_options = {**transformer_options}
    model_options['transformer_options'] = transformer_options
    extra_args['model_options'] = model_options
    source_extra_args = {**extra_args, 'model_options': { 'transformer_options': { **transformer_options,'latent_type': 'xt '} }}

    
    s_in = x.new_ones([x.shape[0]])
    seed = get_extra_options_kv("seed", 42, extra_options)
    generator = torch.manual_seed(seed)
    if latent_guide is not None:
        y0 = model.inner_model.inner_model.process_latent_in(latent_guide['samples']).clone().to(x.device)
    y0 = y0.to(torch.float64)
    sigmas = sigmas.to(torch.float64)
    if y0.ndim == 5:
        x = y0 = y0.repeat(1,1,x.shape[-3],1,1).clone()
    else:
        x = y0.clone()

    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
    noise_sampler2 = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+10000, sigma_min=0.0, sigma_max=1.0)
    
    weight = get_extra_options_kv("weight", 1.0, extra_options)
    stop_step = get_extra_options_kv("stop_step", 10000, extra_options)

    sigma, sigma_next = sigmas[0], sigmas[1]
    noise = noise_sampler(sigma=sigma, sigma_next=sigma_next)
    noise = (noise - noise.mean()) / noise.std()
    
    #z_tar = noise.clone()
    
    #t_mask = torch.ones_like(x)
    #t_mask_inv = 1-t_mask
    
    #x = x + t_mask_inv * (noise - x)
    refine_steps = stop_step

    x_init = x  
    x_tgt = x_init.clone()
    N = len(sigmas)-1
    s_in = x_init.new_ones([x_init.shape[0]])

    skip_steps = get_extra_options_kv("skip_steps", 0, extra_options)
    sigmas = sigmas[skip_steps:]

    x_tgt = x_init.clone()
    N = len(sigmas)-1
    s_in = x_init.new_ones([x_init.shape[0]])
    noise_mask = extra_args.get('denoise_mask', None)
    if noise_mask is None:
        noise_mask = torch.ones_like(x_init)
    else:
        extra_args['denoise_mask'] = None
        source_extra_args['denoise_mask'] = None

    for i in trange(N, disable=disable):
        sigma = sigmas[i]
        noise = torch.randn(x_init.shape, generator=generator).to(x_init.device)

        zt_src = (1-sigma)*x_init + sigma*noise
        
        if i < N-refine_steps:
            zt_tgt = x_tgt + zt_src - x_init
            vt_src = model(zt_src, sigma*s_in, **source_extra_args)
        else:
            if i == N-refine_steps:
                zt_tgt = x_tgt + (zt_src - x_init)
                x_tgt = x_tgt + (zt_src - x_init) * noise_mask
            else:
                zt_tgt = x_tgt * (noise_mask) + (1-noise_mask) * ( (1-sigma)*x_tgt + sigma*noise )
            vt_src = 0
            
        extra_args['model_options']['transformer_options']['latent_type'] = 'yt'
        vt_tgt = model(zt_tgt, sigma*s_in, **extra_args)
        
        v_delta = vt_tgt - vt_src
        x_tgt += (sigmas[i+1] - sigmas[i]) * v_delta * noise_mask
        
        if callback is not None:
            callback({'x': x_tgt, 'denoised': x_tgt, 'i': i+skip_steps, 'sigma': sigmas[i], 'sigma_hat': sigmas[i]})

    return x_tgt
    
    
    
    
            
"""        display=None
        if extra_options_flag("z_tar", extra_options):
            display = zt_tgt
        if extra_options_flag("y0_noised", extra_options):
            display = y0_noised
        if extra_options_flag("x_tgt", extra_options):
            display = x_tgt
        if extra_options_flag("denoised_src", extra_options):
            display = denoised_src
        if extra_options_flag("denoised_tar", extra_options):
            display = denoised_tar
        if extra_options_flag("eps_src", extra_options):
            display = eps_src
        if extra_options_flag("eps_tar", extra_options):
            display = eps_tar
            
        if extra_options_flag("denoised", extra_options):
            display = denoised
        if extra_options_flag("eps", extra_options):
            display = eps
        if display is None:
            display = x_tgt
            
        if callback is not None:
            callback({'x': x_tgt, 'denoised': display, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i]})

    return x_tgt"""






def sample_rk_flow_ralston_2s(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
                            sigma_fn_formula="", t_fn_formula="",
                            eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, step=0, extra_options="", 
                            cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None, latent_guide_inv=None, mask=None):
    
    extra_args = {} if extra_args is None else extra_args
    s_in       = x.new_ones([x.shape[0]])
    MAX_STEPS  = 10000
    
    EO         = ExtraOptions(extra_options)

    y0         = model.inner_model.inner_model.process_latent_in(latent_guide    ['samples']).clone().to(x.device).to(torch.float64) if latent_guide     is not None else None
    y0_inv     = model.inner_model.inner_model.process_latent_in(latent_guide_inv['samples']).clone().to(x.device).to(torch.float64) if latent_guide_inv is not None else None

    sigmas     = sigmas.to(torch.float64)

    if mask is not None:
        mask = mask.unsqueeze(1)
        mask = F.interpolate(mask, size=(x.shape[-2], x.shape[-1]), mode='bilinear', align_corners=False)
        mask = mask.expand_as(x).to(x.device).to(x.dtype)
    else:
        mask = torch.ones_like(x)

    NG         = NoiseGen (model, x, sigmas, seed=EO("seed", torch.initial_seed()+1), noise_sampler_type=noise_sampler_type, eta=eta, noise_mode=noise_mode)
    MC         = ModelCall(model, x, sigmas, extra_args=extra_args, EXPONENTIAL=False, RET_DATA=True)
    
    a2_1 = 2/3
    b1   = 1/4
    b2   = 3/4
    c2   = 2/3
    
    weight0    = EO("weight0", 0.5)
    weight1    = EO("weight1", 0.0)
    weight2    = EO("weight2", 0.0)
    start_step = EO("start_step", 0)
    end_step   = EO("end_step",   MAX_STEPS)
    
    data_x = None
    
    base_noise = NG.noise_init.clone()
    
    #first_step = EO("step", 0)
    x = NG()
    while step < start_step:
        sigma      = sigmas[step]
        sigma_next = sigmas[step+1]
        h          = MC.h_fn(step)
        
        eps, data  = MC(x, sigma, {'latent_type': 'xt'})
        x         += h * eps

        if callback is not None:
            callback({'x': x, 'denoised': data, 'i': step, 'sigma': sigma, 'sigma_hat': sigma})
            
        step += 1
        MC.end_step()
        NG.end_step()
    
    if step > 0:
        y0  = (1-weight0) * data + weight0 * y0
    yx0 = y0.clone()
    
    while step < end_step:
        sigma      = sigmas[step]
        sigma_next = sigmas[step+1]
        h          = MC.h_fn(step)
        sigma_2    = MC.get_sigma_substep(c2, step)
        
        noise = NG()
        #yt = NG.linear_noise(y0, sigma)
        yt = (1-sigma) * y0 + sigma * base_noise
        yt = NG.swap_noise_xt(yt, y0,  sigma, noise=noise)
        xt = yx0 + yt - y0                                          # denoised xt....       xt = yx0 + sigma * (noise - y0)     x = data + sigma * eps      xt = (yx0 - sigma*y0) + sigma*noise
        xt = NG.swap_noise_xt(xt, yx0 - sigma*y0, sigma, noise=noise)
        
        eps_y, data_y  = MC(yt, sigma, {'latent_type': 'yt'})
        eps_x, data_x  = MC(xt, sigma, {'latent_type': 'xt'})
        #eps_yx      = (eps_x - eps_y) * mask 
        eps_yx      = (eps_y - eps_x) * mask 
        data_yx     = yx0 - sigma * eps_yx
        eps_yx_next = (xt - data_yx) / sigma
        
        # UPDATE
        yx0_2 = yx0 + h * (a2_1 * eps_yx)


        # NOISE ADD
        noise = NG()
        #yt_2 = NG.linear_noise(y0, sigma_2)
        yt_2 = (1-sigma_2) * y0 + sigma_2 * base_noise
        yt_2 = NG.swap_noise_xt(yt_2, y0,      sigma_2, noise=noise)

        xt_2 = yx0_2 + yt_2 - y0
        #xt_2 = yx0_2 + ((1-sigma_2) * y0 + sigma_2 * noise) - y0
        noisy_noise = (yt_2 - (1-sigma_2)*y0) / sigma_2
        xt_2 = NG.swap_noise_xt(xt_2, yx0_2 - sigma_2*y0, sigma_2, noise=noise)
        #xt_2 = NG.swap_noise_xt(xt_2, data_yx, sigma_2, noise=noise)
        
        
        yx0_2 = xt_2 + y0 - yt_2  
        yt_2  = xt_2 + y0 - yx0_2
        
        
        if EO("use_bong2_update"):
            yx0 = yx0_2 - h * (a2_1 * eps_yx)

            """eps_x = (xt_2 - data_x) / (sigma + h * a2_1)
            #xt = data_x + sigma * eps_x
            xt = data_x + (sigma*(xt_2 - data_x)) / (h*a2_1 + sigma)
            
            eps_yx = eps_x - eps_y
            
            #xt = yx0 + yt - y0
            yx0 = y0 + xt - yt"""

        if EO("use_bong1_update"):

            #yx0_2 = yx0 + h * (a2_1 * eps_yx)
            #yx0 = yx0_2 - h * (a2_1 * eps_yx) 
            
            #data_yx = yx0 - sigma * eps_yx
            #eps_yx = (yx0_2 - data_yx) / (sigma + h * a2_1)

            #eps_y = (yt_2 - data_y) / (sigma + h * a2_1)
            #yt = data_y + sigma * eps_y
            #yt = data_y + (sigma*(yt_2 - data_y)) / (h*a2_1 + sigma)
            
            eps_x = (xt_2 - data_x) / (sigma + h * a2_1)
            #xt = data_x + sigma * eps_x
            xt = data_x + (sigma*(xt_2 - data_x)) / (h*a2_1 + sigma)
            
            #eps_yx = eps_x - eps_y
            eps_yx = eps_y - eps_x
            
            #xt = yx0 + yt - y0
            yx0 = y0 + xt - yt
            
            #eps = (x_2 - denoised) / (sigma + h * a2_1)
            #x_0 = x = denoised + sigma * eps
            #_0 = x = denoised + (sigma*(x_2 - denoised)) / (h*a2_1 + sigma)
        
        
        eps_y_2, data_y_2  = MC(yt_2, sigma_2, {'latent_type': 'yt'})
        eps_x_2, data_x_2  = MC(xt_2, sigma_2, {'latent_type': 'xt'})
        
        eps_y_2 = (yx0 - data_y_2) / sigma
        eps_x_2 = (yx0 - data_x_2) / sigma
        
        eps_yx_2  = (eps_x_2 - eps_y_2) * mask 
        data_yx_2 = yx0_2 - sigma_2 * eps_yx_2
        
        
        
        yx0_next = yx0 + h * (b1 * eps_yx + b2 * eps_yx_2 )
        
        
        yx0 = yx0_next

        if callback is not None:
            callback({'x': yx0, 'denoised': data_x, 'i': step, 'sigma': sigma, 'sigma_hat': sigma})
            
        step += 1
        MC.end_step()
        NG.end_step()
    
    #yt = NG.linear_noise(y0, sigma_next)
    yt = (1-sigma_next) * y0 + sigma_next * base_noise
    x = yx0 + yt - y0
    
    while step < len(sigmas)-1:
        sigma      = sigmas[step]
        sigma_next = sigmas[step+1]
        h          = MC.h_fn(step)
        
        eps, data  = MC(x, sigma, {'latent_type': 'xt'})
        
        x += h * eps

        if callback is not None:
            callback({'x': x, 'denoised': data, 'i': step, 'sigma': sigma, 'sigma_hat': sigma})
        
        step += 1
        MC.end_step()
        NG.end_step()
    
    return x





def sample_rk_flow_ralston_2s_redo(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
                            sigma_fn_formula="", t_fn_formula="",
                            eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, step=0, extra_options="", 
                            cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None, latent_guide_inv=None, mask=None):
    
    extra_args = {} if extra_args is None else extra_args
    s_in       = x.new_ones([x.shape[0]])
    MAX_STEPS  = 10000
    
    EO         = ExtraOptions(extra_options)

    y0         = model.inner_model.inner_model.process_latent_in(latent_guide    ['samples']).clone().to(x.device).to(torch.float64) if latent_guide     is not None else None
    y0_inv     = model.inner_model.inner_model.process_latent_in(latent_guide_inv['samples']).clone().to(x.device).to(torch.float64) if latent_guide_inv is not None else None

    sigmas     = sigmas.to(torch.float64)

    if mask is not None:
        mask = mask.unsqueeze(1)
        mask = F.interpolate(mask, size=(x.shape[-2], x.shape[-1]), mode='bilinear', align_corners=False)
        mask = mask.expand_as(x).to(x.device).to(x.dtype)
    else:
        mask = torch.ones_like(x)

    NG         = NoiseGen (model, x, sigmas, seed=EO("seed", torch.initial_seed()+1), noise_sampler_type=noise_sampler_type, eta=eta, noise_mode=noise_mode)
    MC         = ModelCall(model, x, sigmas, extra_args=extra_args, EXPONENTIAL=False, RET_DATA=True)
    
    a2_1 = 2/3
    b1   = 1/4
    b2   = 3/4
    c2   = 2/3
    
    weight0    = EO("weight0", 0.5)
    weight1    = EO("weight1", 0.0)
    weight2    = EO("weight2", 0.0)
    start_step = EO("start_step", 0)
    end_step   = EO("end_step",   MAX_STEPS)
    
    flowiter = EO("flowiter", 1)
    
    data_x = None
    
    base_noise = NG.noise_init.clone()
    
    #first_step = EO("step", 0)
    x = NG()
    while step < start_step:
        sigma      = sigmas[step]
        sigma_next = sigmas[step+1]
        h          = MC.h_fn(step)
        
        eps, data  = MC(x, sigma, {'latent_type': 'xt'})
        x         += h * eps

        if callback is not None:
            callback({'x': x, 'denoised': data, 'i': step, 'sigma': sigma, 'sigma_hat': sigma})
            
        step += 1
        MC.end_step()
        NG.end_step()
    
    #if step > 0:
    #    y0  = (1-weight0) * data + weight0 * y0
    #yx0 = y0.clone()
    noise_yt = NG.noise_init.clone()
    noise_xt = NG.noise_init.clone()

    y0 = y0.clone()
    yx0 = y0.clone()
    
    yt = (1-sigmas[step]) * y0  + sigmas[step] * noise_yt
    xt = (1-sigmas[step]) * yx0 + sigmas[step] * noise_xt
    
    while step < end_step:
        sigma      = sigmas[step]
        sigma_next = sigmas[step+1]
        h          = MC.h_fn(step)
        sigma_2    = MC.get_sigma_substep(c2, step)

        eps_xy = torch.zeros_like(y0)
        
        for iter in range(flowiter):
            
            noise_xt_new = noise_yt_new = NG()
            if EO("separate_noise"):
                noise_yt_new = NG()
                
            xt = xt + sigma * eta * (noise_xt_new - noise_xt)
            yt = yt + sigma * eta * (noise_yt_new - noise_yt)
            
            noise_xt = noise_xt + eta * (noise_xt_new - noise_xt)
            noise_yt = noise_yt + eta * (noise_yt_new - noise_yt)
        
            eps_x, data_x = MC(xt, sigma, {'latent_type': 'xt'})
            eps_y, data_y = MC(yt, sigma, {'latent_type': 'yt'})
            
            eps_xy += (1/flowiter) * (eps_x - eps_y)
            #eps_xy = eps_x - eps_y

        eps_flow    = eps_xy + noise_xt - y0
        eps_flow_y0 =          noise_yt - y0
            
        data_flow_x = xt   +   h * eps_xy   -   sigma * (noise_xt - y0)     #######
        data_flow_y = yt                    -   sigma * (noise_yt - y0)

        xt_2 = xt + h * a2_1 * eps_flow
        yt_2 = yt + h * a2_1 * eps_flow_y0
        
        noise_xt_new = noise_yt_new = NG()
        if EO("separate_noise"):
            noise_yt_new = NG()
            
        xt_2 = xt_2 + sigma_2 * eta * (noise_xt_new - noise_xt)
        yt_2 = yt_2 + sigma_2 * eta * (noise_yt_new - noise_yt)
        
        noise_xt = noise_xt + eta * (noise_xt_new - noise_xt)
        noise_yt = noise_yt + eta * (noise_yt_new - noise_yt)


        if EO("x_bong_only"):
            yt   = y0 + sigma   * (noise_yt - y0)
            yt_2 = y0 + sigma_2 * (noise_yt - y0)
            eps_y = (yt - data_y) / sigma
            
            for i in range(100):
                xt = xt_2 - h * a2_1 * (eps_x - eps_y + noise_xt - y0)
                xt_2 = xt + h * a2_1 * (eps_x - eps_y + noise_xt - y0)
                eps_x = (xt - data_x) / sigma
        elif EO("altbong"):
            yt   = y0 + sigma   * (noise_yt - y0)
            yt_2 = y0 + sigma_2 * (noise_yt - y0)
            eps_y = (yt - data_y) / sigma
            
            for i in range(100):
                eps_x = (xt - data_x) / sigma
                xt  = data_flow_x - (h * (eps_x - eps_y)   -   sigma * (noise_xt - y0) )
                
                
            
        else:
            for i in range(100):

                xt = xt_2 - h * a2_1 * (eps_x - eps_y + noise_xt - y0)
                xt_2 = xt + h * a2_1 * (eps_x - eps_y + noise_xt - y0)

                yt = yt_2 - h * a2_1 * (noise_yt - y0)
                yt_2 = yt + h * a2_1 * (noise_yt - y0)

                eps_x = (xt - data_x) / sigma
                eps_y = (yt - data_y) / sigma

        eps_xy_2 = torch.zeros_like(eps_xy)
        for iter in range(flowiter):
            
            noise_xt_new = noise_yt_new = NG()
            if EO("separate_noise"):
                noise_yt_new = NG()
                
            xt_2 = xt_2 + sigma_2 * eta * (noise_xt_new - noise_xt)
            yt_2 = yt_2 + sigma_2 * eta * (noise_yt_new - noise_yt)
            
            noise_xt = noise_xt + eta * (noise_xt_new - noise_xt)
            noise_yt = noise_yt + eta * (noise_yt_new - noise_yt)
            
            yt   = y0 + sigma   * (noise_yt - y0)
            yt_2 = y0 + sigma_2 * (noise_yt - y0)
            eps_y = (yt - data_y) / sigma
            
            for i in range(100):
                xt = xt_2 - h * a2_1 * (eps_x - eps_y + noise_xt - y0)
                xt_2 = xt + h * a2_1 * (eps_x - eps_y + noise_xt - y0)
                eps_x = (xt - data_x) / sigma
                
            
        

            eps_x_2, data_x_2 = MC(xt_2, sigma_2, {'latent_type': 'xt'})
            eps_y_2, data_y_2 = MC(yt_2, sigma_2, {'latent_type': 'yt'})
            
            if EO("epssync"):
                eps_x_2 = (xt - data_x_2) / sigma
                eps_y_2 = (yt - data_y_2) / sigma
            
            eps_xy_2 += (1/flowiter) * (eps_x_2 - eps_y_2)

        eps_flow_2    = eps_xy_2 + noise_xt - y0
        eps_flow_y0_2 =            noise_yt - y0
        

        xt_next = xt + h * (b1 * eps_flow    + b2 * eps_flow_2)
        yt_next = yt + h * (b1 * eps_flow_y0 + b2 * eps_flow_y0_2)
        
        noise_xt_new = noise_yt_new = NG()
        if EO("separate_noise"):
            noise_yt_new = NG()
        
        xt_next = xt_next + sigma_next * eta_var * (noise_xt_new - noise_xt)
        yt_next = yt_next + sigma_next * eta_var * (noise_yt_new - noise_yt)
        
        noise_xt = noise_xt + eta_var * (noise_xt_new - noise_xt)
        noise_yt = noise_yt + eta_var * (noise_yt_new - noise_yt)

        xt = xt_next
        yt = yt_next
        
        #yx0 = yx0_next

        display=None
        if EO("yx0"):
            display = yx0
        if EO("data_x"):
            display = data_x
        if EO("data_y"):
            display = data_y
        if EO("xt"):
            display = xt
        if EO("yt"):
            display = yt
        if EO("eps_yx"):
            display = (eps_y - eps_x)
        if EO("data_xy"):
            display = (data_x - data_y)

        if display is None:
            display = data_flow_x # data_x
            
        if callback is not None:
            callback({'x': yx0, 'denoised': display, 'i': step, 'sigma': sigma, 'sigma_hat': sigma})
            
        step += 1
        MC.end_step()
        NG.end_step()
        
    if step >= len(sigmas)-1:
        return xt
    
    x = xt
    
    while step < len(sigmas)-1:
        sigma      = sigmas[step]
        sigma_next = sigmas[step+1]
        h          = MC.h_fn(step)
        
        eps, data  = MC(x, sigma, {'latent_type': 'xt'})
        
        x += h * eps

        if callback is not None:
            callback({'x': x, 'denoised': data, 'i': step, 'sigma': sigma, 'sigma_hat': sigma})
        
        step += 1
        MC.end_step()
        NG.end_step()
    
    return x








def sample_rk_flow_ralston_4s_redo(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
                            sigma_fn_formula="", t_fn_formula="",
                            eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, step=0, extra_options="", 
                            cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None, latent_guide_inv=None, mask=None):
    
    extra_args = {} if extra_args is None else extra_args
    s_in       = x.new_ones([x.shape[0]])
    MAX_STEPS  = 10000
    
    EO         = ExtraOptions(extra_options)

    y0         = model.inner_model.inner_model.process_latent_in(latent_guide    ['samples']).clone().to(x.device).to(torch.float64) if latent_guide     is not None else None
    y0_inv     = model.inner_model.inner_model.process_latent_in(latent_guide_inv['samples']).clone().to(x.device).to(torch.float64) if latent_guide_inv is not None else None

    sigmas     = sigmas.to(torch.float64)

    if mask is not None:
        mask = mask.unsqueeze(1)
        mask = F.interpolate(mask, size=(x.shape[-2], x.shape[-1]), mode='bilinear', align_corners=False)
        mask = mask.expand_as(x).to(x.device).to(x.dtype)
    else:
        mask = torch.ones_like(x)

    NG         = NoiseGen (model, x, sigmas, seed=EO("seed", torch.initial_seed()+1), noise_sampler_type=noise_sampler_type, eta=eta, noise_mode=noise_mode)
    MC         = ModelCall(model, x, sigmas, extra_args=extra_args, EXPONENTIAL=False, RET_DATA=True)
    
    a2_1 = 2/5
    
    a3_1 = (-2889+1428 * 5**0.5)/1024
    a3_2 = (3785-1620 * 5**0.5)/1024
    
    a4_1 = (-3365+2094 * 5**0.5)/6040
    a4_2 = (-975-3046 * 5**0.5)/2552
    a4_3 = (467040+203968*5**0.5)/240845
    
    b1   = (263+24*5**0.5)/1812
    b2   = (125-1000*5**0.5)/3828
    b3   = (3426304+1661952*5**0.5)/5924787
    b4   = (30-4*5**0.5)/123
    
    c2   = 2/5
    c3   = (14-3 * 5**0.5)/16
    c4   = 1.
    

    a2_1 = 1/2
    
    a3_1 = 0
    a3_2 = 1/2
    
    a4_1 = 0
    a4_2 = 0
    a4_3 = 1.
    
    b1   = 1/6
    b2   = 1/3
    b3   = 1/3
    b4   = 1/6
    
    c2   = 1/2
    c3   = 1/2
    c4   = 1.
    
    
    weight0    = EO("weight0", 0.5)
    weight1    = EO("weight1", 0.0)
    weight2    = EO("weight2", 0.0)
    start_step = EO("start_step", 0)
    end_step   = EO("end_step",   MAX_STEPS)
    
    flowiter = EO("flowiter", 1)
    
    data_x = None
    
    base_noise = NG.noise_init.clone()
    
    #first_step = EO("step", 0)
    x = NG()
    while step < start_step:
        sigma      = sigmas[step]
        sigma_next = sigmas[step+1]
        h          = MC.h_fn(step)
        
        eps, data  = MC(x, sigma, {'latent_type': 'xt'})
        x         += h * eps

        if callback is not None:
            callback({'x': x, 'denoised': data, 'i': step, 'sigma': sigma, 'sigma_hat': sigma})
            
        step += 1
        MC.end_step()
        NG.end_step()

    noise_yt = NG.noise_init.clone()
    noise_xt = NG.noise_init.clone()

    y0 = y0.clone()
    yx0 = y0.clone()
    
    yt = (1-sigmas[step]) * y0  + sigmas[step] * noise_yt
    xt = (1-sigmas[step]) * yx0 + sigmas[step] * noise_xt
    
    while step < end_step:
        sigma      = sigmas[step]
        sigma_next = sigmas[step+1]
        h          = MC.h_fn(step)
        sigma_2    = MC.get_sigma_substep(c2, step)
        sigma_3    = MC.get_sigma_substep(c3, step)
        sigma_4    = MC.get_sigma_substep(c4, step)

        eps_xy = torch.zeros_like(y0)
        
        eps_x, data_x = MC(xt, sigma, {'latent_type': 'xt'})
        eps_y, data_y = MC(yt, sigma, {'latent_type': 'yt'})
        
        eps_xy = eps_x - eps_y

        eps_flow    = eps_xy + noise_xt - y0
        eps_flow_y0 =          noise_yt - y0
        
        data_flow_x = xt   +   h * eps_xy   -   sigma * (noise_xt - y0)     #######

        xt_2 = xt + h * a2_1 * eps_flow
        yt_2 = yt + h * a2_1 * eps_flow_y0
        

        eps_xy_2 = torch.zeros_like(eps_xy)

        eps_x_2, data_x_2 = MC(xt_2, sigma_2, {'latent_type': 'xt'})
        eps_y_2, data_y_2 = MC(yt_2, sigma_2, {'latent_type': 'yt'})
        
        eps_x_2 = (xt - data_x_2) / sigma
        eps_y_2 = (yt - data_y_2) / sigma
            
        eps_xy_2 = eps_x_2 - eps_y_2

        eps_flow_2    = eps_xy_2 + noise_xt - y0
        eps_flow_y0_2 =            noise_yt - y0
        
        xt_3 = xt + h * (a3_1 * eps_flow    + a3_2 * eps_flow_2 )
        yt_3 = yt + h * (a3_1 * eps_flow_y0 + a3_2 * eps_flow_y0_2)
        
        
        
        eps_x_3, data_x_3 = MC(xt_3, sigma_3, {'latent_type': 'xt'})
        eps_y_3, data_y_3 = MC(yt_3, sigma_3, {'latent_type': 'yt'})
        
        eps_x_3 = (xt - data_x_3) / sigma
        eps_y_3 = (yt - data_y_3) / sigma
            
        eps_xy_3 = eps_x_3 - eps_y_3

        eps_flow_3    = eps_xy_3 + noise_xt - y0
        eps_flow_y0_3 =            noise_yt - y0
        
        xt_4 = xt + h * (a4_1 * eps_flow    + a4_2 * eps_flow_2    + a4_3 * eps_flow_3)
        yt_4 = yt + h * (a4_1 * eps_flow_y0 + a4_2 * eps_flow_y0_2 + a4_3 * eps_flow_y0_3)
        
        
        
        eps_x_4, data_x_4 = MC(xt_4, sigma_4, {'latent_type': 'xt'})
        eps_y_4, data_y_4 = MC(yt_4, sigma_4, {'latent_type': 'yt'})
        
        eps_x_4 = (xt - data_x_4) / sigma
        eps_y_4 = (yt - data_y_4) / sigma
        
        eps_xy_4 = eps_x_4 - eps_y_4

        eps_flow_4    = eps_xy_4 + noise_xt - y0
        eps_flow_y0_4 =            noise_yt - y0

        xt_next = xt + h * (b1 * eps_flow    + b2 * eps_flow_2    + b3 * eps_flow_3    + b4 * eps_flow_4)
        yt_next = yt + h * (b1 * eps_flow_y0 + b2 * eps_flow_y0_2 + b3 * eps_flow_y0_3 + b4 * eps_flow_y0_4)
        
        xt = xt_next
        yt = yt_next
        
        #yx0 = yx0_next

        display=None
        if EO("yx0"):
            display = yx0
        if EO("data_x"):
            display = data_x
        if EO("data_y"):
            display = data_y
        if EO("xt"):
            display = xt
        if EO("yt"):
            display = yt
        if EO("eps_yx"):
            display = (eps_y - eps_x)
        if EO("data_xy"):
            display = (data_x - data_y)

        if display is None:
            display = data_flow_x # data_x
            
        if callback is not None:
            callback({'x': yx0, 'denoised': display, 'i': step, 'sigma': sigma, 'sigma_hat': sigma})
            
        step += 1
        MC.end_step()
        NG.end_step()
        
    if step >= len(sigmas)-1:
        return xt
    
    x = xt
    
    while step < len(sigmas)-1:
        sigma      = sigmas[step]
        sigma_next = sigmas[step+1]
        h          = MC.h_fn(step)
        
        eps, data  = MC(x, sigma, {'latent_type': 'xt'})
        
        x += h * eps

        if callback is not None:
            callback({'x': x, 'denoised': data, 'i': step, 'sigma': sigma, 'sigma_hat': sigma})
        
        step += 1
        MC.end_step()
        NG.end_step()
    
    return x





def sample_rk_flow_ralston_3s_redo(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
                            sigma_fn_formula="", t_fn_formula="",
                            eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, step=0, extra_options="", 
                            cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None, latent_guide_inv=None, mask=None):
    
    extra_args = {} if extra_args is None else extra_args
    s_in       = x.new_ones([x.shape[0]])
    MAX_STEPS  = 10000
    
    x_init = x.clone()
    
    EO         = ExtraOptions(extra_options)

    y0         = model.inner_model.inner_model.process_latent_in(latent_guide    ['samples']).clone().to(x.device).to(torch.float64) if latent_guide     is not None else None
    y0_inv     = model.inner_model.inner_model.process_latent_in(latent_guide_inv['samples']).clone().to(x.device).to(torch.float64) if latent_guide_inv is not None else None

    sigmas     = sigmas.to(torch.float64)

    if mask is not None:
        mask = mask.unsqueeze(1)
        mask = F.interpolate(mask, size=(x.shape[-2], x.shape[-1]), mode='bilinear', align_corners=False)
        mask = mask.expand_as(x).to(x.device).to(x.dtype)
    else:
        mask = torch.ones_like(x)

    NG         = NoiseGen (model, x, sigmas, seed=EO("seed", torch.initial_seed()+1), noise_sampler_type=noise_sampler_type, eta=eta, noise_mode=noise_mode)
    MC         = ModelCall(model, x, sigmas, extra_args=extra_args, EXPONENTIAL=False, RET_DATA=True)
    

    a2_1 = 1/2
    
    a3_1 = 0
    a3_2 = 3/4
    
    b1   = 2/9
    b2   = 1/3
    b3   = 4/9
    
    c2   = 1/2
    c3   = 3/4
    
    
    weight0    = EO("weight0", 0.5)
    weight1    = EO("weight1", 0.0)
    weight2    = EO("weight2", 0.0)
    start_step = EO("start_step", 0)
    end_step   = EO("end_step",   MAX_STEPS)
    
    flowiter = EO("flowiter", 1)
    
    data_x = None
    
    base_noise = NG.noise_init.clone()
    
    #first_step = EO("step", 0)
    x = NG()
    while step < start_step:
        sigma      = sigmas[step]
        sigma_next = sigmas[step+1]
        h          = MC.h_fn(step)
        
        eps, data  = MC(x, sigma, {'latent_type': 'xt'})
        x         += h * eps

        if callback is not None:
            callback({'x': x, 'denoised': data, 'i': step, 'sigma': sigma, 'sigma_hat': sigma})
            
        step += 1
        MC.end_step()
        NG.end_step()

    noise_yt = NG.noise_init.clone()
    noise_xt = NG.noise_init.clone()
    
    if EO("use_x_init"):
        noise_xt = x_init.clone()
        noise_yt = x_init.clone()

    y0 = y0.clone()
    yx0 = y0.clone()
    
    yt = (1-sigmas[step]) * y0  + sigmas[step] * noise_yt
    xt = (1-sigmas[step]) * yx0 + sigmas[step] * noise_xt
    
    while step < end_step:
        sigma      = sigmas[step]
        sigma_next = sigmas[step+1]
        h          = MC.h_fn(step)
        sigma_2    = MC.get_sigma_substep(c2, step)
        sigma_3    = MC.get_sigma_substep(c3, step)

        eps_xy = torch.zeros_like(y0)
        
        eps_x, data_x = MC(xt, sigma, {'latent_type': 'xt'})
        eps_y, data_y = MC(yt, sigma, {'latent_type': 'yt'})
        
        eps_xy = eps_x - eps_y

        eps_flow    = eps_xy + noise_xt - y0
        eps_flow_y0 =          noise_yt - y0
        
        data_flow_x = xt   +   h * eps_xy   -   sigma * (noise_xt - y0)     #######

        xt_2 = xt + h * a2_1 * eps_flow
        yt_2 = yt + h * a2_1 * eps_flow_y0
        

        eps_xy_2 = torch.zeros_like(eps_xy)

        eps_x_2, data_x_2 = MC(xt_2, sigma_2, {'latent_type': 'xt'})
        eps_y_2, data_y_2 = MC(yt_2, sigma_2, {'latent_type': 'yt'})
        
        #eps_x_2 = (xt - data_x_2) / sigma
        #eps_y_2 = (yt - data_y_2) / sigma
            
        eps_xy_2 = eps_x_2 - eps_y_2

        eps_flow_2    = eps_xy_2 + noise_xt - y0
        eps_flow_y0_2 =            noise_yt - y0
        
        xt_3 = xt + h * (a3_1 * eps_flow    + a3_2 * eps_flow_2 )
        yt_3 = yt + h * (a3_1 * eps_flow_y0 + a3_2 * eps_flow_y0_2)
        
        
        
        eps_x_3, data_x_3 = MC(xt_3, sigma_3, {'latent_type': 'xt'})
        eps_y_3, data_y_3 = MC(yt_3, sigma_3, {'latent_type': 'yt'})
        
        #eps_x_3 = (xt - data_x_3) / sigma
        #eps_y_3 = (yt - data_y_3) / sigma
            
        eps_xy_3 = eps_x_3 - eps_y_3

        eps_flow_3    = eps_xy_3 + noise_xt - y0
        eps_flow_y0_3 =            noise_yt - y0

        xt_next = xt + h * (b1 * eps_flow    + b2 * eps_flow_2    + b3 * eps_flow_3   )
        yt_next = yt + h * (b1 * eps_flow_y0 + b2 * eps_flow_y0_2 + b3 * eps_flow_y0_3)
        
        xt = xt_next
        yt = yt_next
        
        noise_xt_new = noise_yt_new = NG()
        
        xt = xt + sigma * eta * (noise_xt_new - noise_xt)
        yt = yt + sigma * eta * (noise_yt_new - noise_yt)
        
        noise_xt = noise_xt + eta * (noise_xt_new - noise_xt)
        noise_yt = noise_yt + eta * (noise_yt_new - noise_yt)

        
        #yx0 = yx0_next

        display=None
        if EO("yx0"):
            display = yx0
        if EO("data_x"):
            display = data_x
        if EO("data_y"):
            display = data_y
        if EO("xt"):
            display = xt
        if EO("yt"):
            display = yt
        if EO("eps_yx"):
            display = (eps_y - eps_x)
        if EO("data_xy"):
            display = (data_x - data_y)

        if display is None:
            display = data_flow_x # data_x
            
        if callback is not None:
            callback({'x': yx0, 'denoised': display, 'i': step, 'sigma': sigma, 'sigma_hat': sigma})
            
        step += 1
        MC.end_step()
        NG.end_step()
        
    if step >= len(sigmas)-1:
        return xt
    
    x = xt
    
    while step < len(sigmas)-1:
        sigma      = sigmas[step]
        sigma_next = sigmas[step+1]
        h          = MC.h_fn(step)
        
        eps, data  = MC(x, sigma, {'latent_type': 'xt'})
        
        x += h * eps

        if callback is not None:
            callback({'x': x, 'denoised': data, 'i': step, 'sigma': sigma, 'sigma_hat': sigma})
        
        step += 1
        MC.end_step()
        NG.end_step()
    
    return x








def sample_rk_flow_gauss_2s(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
                            sigma_fn_formula="", t_fn_formula="",
                            eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, step=0, extra_options="", 
                            cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None, latent_guide_inv=None, mask=None):
    
    extra_args = {} if extra_args is None else extra_args
    s_in       = x.new_ones([x.shape[0]])
    MAX_STEPS  = 10000
    
    x_init = x.clone()
    
    EO         = ExtraOptions(extra_options)

    y0         = model.inner_model.inner_model.process_latent_in(latent_guide    ['samples']).clone().to(x.device).to(torch.float64) if latent_guide     is not None else None
    y0_inv     = model.inner_model.inner_model.process_latent_in(latent_guide_inv['samples']).clone().to(x.device).to(torch.float64) if latent_guide_inv is not None else None

    sigmas     = sigmas.to(torch.float64)

    if mask is not None:
        mask = mask.unsqueeze(1)
        mask = F.interpolate(mask, size=(x.shape[-2], x.shape[-1]), mode='bilinear', align_corners=False)
        mask = mask.expand_as(x).to(x.device).to(x.dtype)
    else:
        mask = torch.ones_like(x)

    NG         = NoiseGen (model, x, sigmas, seed=EO("seed", torch.initial_seed()+1), noise_sampler_type=noise_sampler_type, eta=eta, noise_mode=noise_mode)
    MC         = ModelCall(model, x, sigmas, extra_args=extra_args, EXPONENTIAL=False, RET_DATA=True)
    
    a1_1 = 1/4
    a1_2 = 1/4 - 3**0.5 / 6
    
    a2_1 = 1/4 + 3**0.5 / 6
    a2_2 = 1/4
    
    b1   = 1/2
    b2   = 1/2
    
    c1   = 1/2 - 3**0.5 / 6
    c2   = 1/2 + 3**0.5 / 6
    
    
    weight0    = EO("weight0", 0.5)
    weight1    = EO("weight1", 0.0)
    weight2    = EO("weight2", 0.0)
    start_step = EO("start_step", 0)
    end_step   = EO("end_step",   MAX_STEPS)
    
    flowiter = EO("flowiter", 1)
    
    data_x = None
    
    base_noise = NG.noise_init.clone()
    
    #first_step = EO("step", 0)
    x = NG()
    while step < start_step:
        sigma      = sigmas[step]
        sigma_next = sigmas[step+1]
        h          = MC.h_fn(step)
        
        eps, data  = MC(x, sigma, {'latent_type': 'xt'})
        x         += h * eps

        if callback is not None:
            callback({'x': x, 'denoised': data, 'i': step, 'sigma': sigma, 'sigma_hat': sigma})
            
        step += 1
        MC.end_step()
        NG.end_step()

    noise_yt = NG.noise_init.clone()
    noise_xt = NG.noise_init.clone()
    
    if EO("use_x_init"):
        noise_xt = x_init.clone()
        noise_yt = x_init.clone()

    y0 = y0.clone()
    yx0 = y0.clone()
    
    yt = (1-sigmas[step]) * y0  + sigmas[step] * noise_yt
    xt = (1-sigmas[step]) * yx0 + sigmas[step] * noise_xt
    
    y_scale = EO("y_scale", 1.0)
    
    while step < end_step:
        sigma      = sigmas[step]
        sigma_next = sigmas[step+1]
        h          = MC.h_fn(step)
        
        sigma_1    = MC.get_sigma_substep(c1, step)
        sigma_2    = MC.get_sigma_substep(c2, step)

        eps_xy = torch.zeros_like(y0)
        
        eps_x, data_x = MC(xt, sigma, {'latent_type': 'xt'})
        eps_y, data_y = MC(yt, sigma, {'latent_type': 'yt'})
        
        eps_xy = eps_x - y_scale * eps_y

        eps_flow    = eps_xy + noise_xt - y_scale * y0
        eps_flow_y0 =          noise_yt - y_scale * y0
        
        data_flow_x = xt   +   h * eps_xy   -   sigma * (noise_xt - y_scale * y0)     #######

        xt_1 = xt + (sigma_1 - sigma) * eps_flow
        yt_1 = yt + (sigma_1 - sigma) * eps_flow_y0
        
        xt_2 = xt + (sigma_2 - sigma) * eps_flow
        yt_2 = yt + (sigma_2 - sigma) * eps_flow_y0
        
        for iter in range(flowiter):
            eps_x_1, data_x_1 = MC(xt_1, sigma_1, {'latent_type': 'xt'})
            eps_x_2, data_x_2 = MC(xt_2, sigma_2, {'latent_type': 'xt'})

            if iter == 0:
                eps_y_1, data_y_1 = MC(yt_1, sigma_1, {'latent_type': 'yt'})
                eps_y_2, data_y_2 = MC(yt_2, sigma_2, {'latent_type': 'yt'})
                
                eps_flow_y0_1 = noise_yt - y0
                eps_flow_y0_2 = noise_yt - y0
            
            eps_xy_1 = eps_x_1 - y_scale * eps_y_1
            eps_xy_2 = eps_x_2 - y_scale * eps_y_2
            
            eps_flow_1 = eps_xy_1 + noise_xt - y_scale * y0
            eps_flow_2 = eps_xy_2 + noise_xt - y_scale * y0
            
            xt_1 = xt + h * (a1_1 * eps_flow_1 + a1_2 * eps_flow_2)
            xt_2 = xt + h * (a2_1 * eps_flow_1 + a2_2 * eps_flow_2)
            
            if iter == 0:
                yt_1 = yt + h * (a1_1 * eps_flow_y0_1 + a1_2 * eps_flow_y0_2)
                yt_2 = yt + h * (a2_1 * eps_flow_y0_1 + a2_2 * eps_flow_y0_2)

        xt_next = xt + h * (b1 * eps_flow    + b2 * eps_flow_2   )
        yt_next = yt + h * (b1 * eps_flow_y0 + b2 * eps_flow_y0_2)
        
        xt = xt_next
        yt = yt_next
        
        noise_xt_new = noise_yt_new = NG()
        
        xt = xt + sigma_next * eta * (noise_xt_new - noise_xt)            # previously was sigma...
        yt = yt + sigma_next * eta * (noise_yt_new - noise_yt)
        
        noise_xt = noise_xt + eta * (noise_xt_new - noise_xt)
        noise_yt = noise_yt + eta * (noise_yt_new - noise_yt)

        
        #yx0 = yx0_next

        display=None
        if EO("yx0"):
            display = yx0
        if EO("data_x"):
            display = data_x
        if EO("data_y"):
            display = data_y
        if EO("xt"):
            display = xt
        if EO("yt"):
            display = yt
        if EO("eps_yx"):
            display = (eps_y - eps_x)
        if EO("data_xy"):
            display = (data_x - data_y)

        if display is None:
            display = data_flow_x # data_x
            
        if callback is not None:
            callback({'x': yx0, 'denoised': display, 'i': step, 'sigma': sigma, 'sigma_hat': sigma})
            
        step += 1
        MC.end_step()
        NG.end_step()
        
    if step >= len(sigmas)-1:
        return xt
    
    x = xt
    
    while step < len(sigmas)-1:
        sigma      = sigmas[step]
        sigma_next = sigmas[step+1]
        h          = MC.h_fn(step)
        
        eps, data  = MC(x, sigma, {'latent_type': 'xt'})
        
        x += h * eps

        if callback is not None:
            callback({'x': x, 'denoised': data, 'i': step, 'sigma': sigma, 'sigma_hat': sigma})
        
        step += 1
        MC.end_step()
        NG.end_step()
    
    return x










def sample_rk_flow(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, step=0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None, latent_guide_inv=None, mask=None):
    
    extra_args = {} if extra_args is None else extra_args
    s_in       = x.new_ones([x.shape[0]])
    MAX_STEPS  = 10000
    
    EO         = ExtraOptions(extra_options)

    y0         = model.inner_model.inner_model.process_latent_in(latent_guide    ['samples']).clone().to(x.device).to(torch.float64) if latent_guide     is not None else None
    y0_inv     = model.inner_model.inner_model.process_latent_in(latent_guide_inv['samples']).clone().to(x.device).to(torch.float64) if latent_guide_inv is not None else None
    
    y0 = y0.expand_as(x).to(x.device).to(x.dtype)

    sigmas     = sigmas.to(torch.float64)

    if mask is not None:
        mask = mask.unsqueeze(1)
        mask = F.interpolate(mask, size=(x.shape[-2], x.shape[-1]), mode='bilinear', align_corners=False)
        mask = mask.expand_as(x).to(x.device).to(x.dtype)
        extra_args['denoise_mask'] = mask
    else:
        mask = torch.ones_like(x)
    NG         = NoiseGen (model, x, sigmas, seed=EO("seed", torch.initial_seed()+1))
    MC         = ModelCall(model, x, sigmas, extra_args=extra_args, RET_DATA=True)
    
    weight0    = EO("weight0", 0.5)
    weight1    = EO("weight1", 0.0)
    weight2    = EO("weight2", 0.0)
    start_step = EO("start_step", 0)
    end_step   = EO("end_step",   MAX_STEPS)
    
    data_x = None
    schedule = None
    
    #first_step = EO("step", 0)
    x = NG()
    while step < start_step:
        sigma      = sigmas[step]
        sigma_next = sigmas[step+1]
        h          = MC.h_fn(step)
        
        eps, data  = MC(x, sigma, {'latent_type': 'xt'})
        x         += h * eps

        if callback is not None:
            callback({'x': x, 'denoised': data, 'i': step, 'sigma': sigma, 'sigma_hat': sigma})
            
        step += 1
        MC.end_step()
        NG.end_step()
    
    if step > 0:
        y0  = (1-weight0) * data + weight0 * y0
    yx0 = y0.clone()
    if y0_inv is not None:
        yx0 = y0_inv.clone()
        #yx0 = slerp_tensor(0.5, y0.clone(), y0_inv.clone())
    x_latent = None
    while step < end_step:
        sigma      = sigmas[step]
        sigma_next = sigmas[step+1]
        h          = MC.h_fn(step)

        #yt = NG.linear_noise(y0, sigma)
        #xt = yx0 + yt - y0
        
        #yt = y0  + sigma * (NG.noise_init - y0)
        #xt = yx0 + sigma * (NG.noise_init - y0)
        
        if schedule is not None:
            mask = schedule[step]
        
        for i in range(EO("iter", 1)):
            sigma = torch.full_like(sigma, EO("sigma_set", sigma.item()))
            sigma_next = torch.full_like(sigma_next, EO("sigma_next_set", sigma_next.item()))
            if EO("new_noise"):
                noise = NG()
                yt = y0  + sigma * (noise - y0)
                xt = yx0 + sigma * (noise - y0)
                if y0_inv is not None:
                    xt = yx0 + sigma * (noise - y0_inv)
            elif EO("x_latent"):
                yt = y0  + sigma * (NG.noise_init - y0)
                if x_latent is None:
                    xt = yx0 + sigma * (NG.noise_init - y0)
                    x_latent = xt.clone()
                else:
                    xt = x_latent #yx0 + sigma * (NG.noise_init - y0)
                if y0_inv is not None:
                    xt = yx0 + sigma * (NG.noise_init - y0_inv)
            else:
                yt = y0  + sigma * (NG.noise_init - y0)
                xt = yx0 + sigma * (NG.noise_init - y0)
                if y0_inv is not None:
                    xt = yx0 + sigma * (NG.noise_init - y0_inv)
            
            eps_y, data_y  = MC(yt, sigma, {'latent_type': 'yt'})
            eps_x, data_x  = MC(xt, sigma, {'latent_type': 'xt'})
            
            eps_y_alt = (yx0 - data_y) / sigma
            eps_x_alt = (yx0 - data_x) / sigma
            
            h *= EO("h_mult", 1.0)
            h = torch.full_like(h, EO("h_set", h.item()))
            yx0_prev = yx0.clone()
            
            if   EO("eps_x_minus_y"):
                yx0 += h * (eps_x - eps_y) * mask 
            
            elif EO("data_y_minus_x"):
                yx0 = yx0 + h * (data_y - data_x)
            elif EO("data_x_minus_y"):
                yx0 = yx0 + h * (data_x - data_y)
            
            elif EO("data_y_minus_x_scaled"):
                yx0 = yx0 + h * (data_y - data_x) / sigma
            elif EO("data_x_minus_y_scaled"):                #WORKS
                yx0 = yx0 + h * (data_x - data_y) / sigma
            
            elif EO("alt_eps_x_minus_y"):
                yx0 = yx0 + h * (eps_x_alt - eps_y_alt)
            elif EO("alt_eps_y_minus_x"):
                yx0 = yx0 + h * (eps_y_alt - eps_x_alt)
            
            elif EO("x_latent"):
                #x_latent_yx0 = yx0 + sigma_next * (NG.noise_init - y0) * mask
                #x_latent += h * (eps_y - eps_x) * mask
                yx0 += h * (eps_y - eps_x) * mask
                #x_latent = (x_latent + (h * eps_x)) * (1-mask) + x_latent_yx0 * mask
                
                #x_latent = (1-(sigma_next/sigma)) * yx0 + (sigma_next/sigma) * NG.noise_init
                
                x_latent = yx0 + sigma_next * (NG.noise_init - y0)
            
            
            else:
                yx0 += h * (eps_y - eps_x) * mask            #WORKS
            
            h_prev = h

            #yx0 += weight2 * (data_x - yx0)
            #y0  += weight2 * (data_x - y0)

            #if callback is not None:
            #    callback({'x': yx0, 'denoised': data_x, 'i': step, 'sigma': sigma, 'sigma_hat': sigma})
            
            display=None
            if EO("yx0"):
                display = yx0
            if EO("data_x"):
                display = data_x
            if EO("data_y"):
                display = data_y
            if EO("xt"):
                display = xt
            if EO("yt"):
                display = yt
            if EO("eps_yx"):
                display = (eps_y - eps_x)
            if EO("data_xy"):
                display = (data_x - data_y)

            if display is None:
                display = data_x
                
            if callback is not None:
                callback({'x': yx0, 'denoised': display, 'i': step, 'sigma': sigma, 'sigma_hat': sigma})
                
            print("step, iter, yx0-y0 norm, yx0-yx0_prev norm: ", step, i, torch.norm(yx0 - y0).item(), torch.norm(yx0 - yx0_prev).item(), flush=True)

        
        step += 1
        MC.end_step()
        NG.end_step()
    #yt = (1 - sigma_next)*y0 + sigma_next * noise             # x = yx0 + (1 - sigma_next)*y0 + sigma_next * noise - y0           x = 
    #yt = NG.linear_noise(y0, sigma_next)
    #x = yx0 + yt - y0
    if EO("x_latent"):
        x = x_latent
    else:
        x = yx0 + sigma_next * (NG.noise_init - y0) * mask
        if y0_inv is not None:
            x = yx0 + sigma_next * (NG.noise_init - y0_inv) * mask
    
    while step < len(sigmas)-1:
        sigma      = sigmas[step]
        sigma_next = sigmas[step+1]
        h          = MC.h_fn(step)
        
        eps, data  = MC(x, sigma, {'latent_type': 'xt'})
        
        x += h * eps * mask

        if callback is not None:
            callback({'x': x, 'denoised': data, 'i': step, 'sigma': sigma, 'sigma_hat': sigma})
        
        step += 1
        MC.end_step()
        NG.end_step()
    
    return x





def sample_rk_flow2(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, step=0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None, latent_guide_inv=None, mask=None):
    
    extra_args = {} if extra_args is None else extra_args
    s_in       = x.new_ones([x.shape[0]])
    MAX_STEPS  = 10000
    
    EO         = ExtraOptions(extra_options)

    y0         = model.inner_model.inner_model.process_latent_in(latent_guide    ['samples']).clone().to(x.device).to(torch.float64) if latent_guide     is not None else None
    y0_inv     = model.inner_model.inner_model.process_latent_in(latent_guide_inv['samples']).clone().to(x.device).to(torch.float64) if latent_guide_inv is not None else None
    
    y0 = y0.expand_as(x).to(x.device).to(x.dtype)

    sigmas     = sigmas.to(torch.float64)

    if mask is not None:
        mask = mask.unsqueeze(1)
        mask = F.interpolate(mask, size=(x.shape[-2], x.shape[-1]), mode='bilinear', align_corners=False)
        mask = mask.expand_as(x).to(x.device).to(x.dtype)
        extra_args['denoise_mask'] = mask
    else:
        mask = torch.ones_like(x)
    NG         = NoiseGen (model, x, sigmas, seed=EO("seed", torch.initial_seed()+1))
    MC         = ModelCall(model, x, sigmas, extra_args=extra_args, RET_DATA=True)
    
    weight0    = EO("weight0", 0.5)
    weight1    = EO("weight1", 0.0)
    weight2    = EO("weight2", 0.0)
    start_step = EO("start_step", 0)
    end_step   = EO("end_step",   MAX_STEPS)
    
    data_x = None
    schedule = None
    
    #first_step = EO("step", 0)
    x = NG()
    while step < start_step:
        sigma      = sigmas[step]
        sigma_next = sigmas[step+1]
        h          = MC.h_fn(step)
        
        eps, data  = MC(x, sigma, {'latent_type': 'xt'})
        x         += h * eps

        if callback is not None:
            callback({'x': x, 'denoised': data, 'i': step, 'sigma': sigma, 'sigma_hat': sigma})
            
        step += 1
        MC.end_step()
        NG.end_step()

    y0 = y0.clone()
    yx0 = y0.clone()
    
    while step < end_step:
        sigma      = sigmas[step]
        sigma_next = sigmas[step+1]
        h          = MC.h_fn(step)

        if schedule is not None:
            mask = schedule[step]

        eps_yx = torch.zeros_like(y0)
        for i in range(EO("iter", 1)):
            #y0_bleed = EO("y0_bleed", 0.0)
            #y0 = (1-y0_bleed) * y0 + y0_bleed * yx0
            
            if EO("noise_init_yt"):
                noise_yt = NG.noise_init
            else:
                noise_yt = NG()
            if EO("noise_init_xt"):
                noise_xt = NG.noise_init
            elif EO("noise_separate_xt"):
                noise_xt = NG()
            else:
                noise_xt = noise_yt

            yt = y0  + sigma * (noise_yt - y0)
            xt = yx0 + sigma * (noise_xt - y0)
            
            eps_x, data_x = MC(xt, sigma, {'latent_type': 'xt'})
            eps_y, data_y = MC(yt, sigma, {'latent_type': 'yt'})
            
            eps_yx += (1/EO("iter",1)) * (eps_x - eps_y)
            
            if EO("implicit_iter"):
                yx0 = yx0 + h * (1/EO("iter",1)) * (eps_x - eps_y)
            if EO("implicit_iter_large"):
                yx0 = yx0 + h * (eps_x - eps_y)
        
        if not EO("implicit_iter") and not EO("implicit_iter_large"):
            yx0 = yx0 + h * eps_yx
        
        y0_eps_y = EO("y0_eps_y", 0.0)
        y0_eps_x = EO("y0_eps_x", 0.0)
        y0_eps_xy = EO("y0_eps_xy", 0.0)
        y0 = y0 + y0_eps_y * h * eps_y
        y0 = y0 + y0_eps_x * h * eps_x
        y0 = y0 + y0_eps_xy * h * eps_yx
        
        

        display=None
        if EO("yx0"):
            display = yx0
        if EO("data_x"):
            display = data_x
        if EO("data_y"):
            display = data_y
        if EO("xt"):
            display = xt
        if EO("yt"):
            display = yt
        if EO("eps_yx"):
            display = (eps_y - eps_x)
        if EO("data_xy"):
            display = (data_x - data_y)

        if display is None:
            display = yx0
            
        if callback is not None:
            callback({'x': yx0, 'denoised': display, 'i': step, 'sigma': sigma, 'sigma_hat': sigma})
            
        step += 1
        MC.end_step()
        NG.end_step()
        
    if step == len(sigmas):
        return yx0

    #yt = (1 - sigma_next)*y0 + sigma_next * noise             # x = yx0 + (1 - sigma_next)*y0 + sigma_next * noise - y0           x = 
    #yt = NG.linear_noise(y0, sigma_next)
    #x = yx0 + yt - y0
    
    if EO("x_latent"):
        x = x_latent
    else:
        x = yx0 + sigma_next * (NG.noise_init - y0) * mask
        if y0_inv is not None:
            x = yx0 + sigma_next * (NG.noise_init - y0_inv) * mask
    
    while step < len(sigmas)-1:
        sigma      = sigmas[step]
        sigma_next = sigmas[step+1]
        h          = MC.h_fn(step)
        
        eps, data  = MC(x, sigma, {'latent_type': 'xt'})
        
        x += h * eps * mask

        if callback is not None:
            callback({'x': x, 'denoised': data, 'i': step, 'sigma': sigma, 'sigma_hat': sigma})
        
        step += 1
        MC.end_step()
        NG.end_step()
    
    return x







def sample_rk_flow3(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, step=0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None, latent_guide_inv=None, mask=None):
    
    extra_args = {} if extra_args is None else extra_args
    s_in       = x.new_ones([x.shape[0]])
    MAX_STEPS  = 10000
    
    EO         = ExtraOptions(extra_options)

    y0         = model.inner_model.inner_model.process_latent_in(latent_guide    ['samples']).clone().to(x.device).to(torch.float64) if latent_guide     is not None else None
    y0_inv     = model.inner_model.inner_model.process_latent_in(latent_guide_inv['samples']).clone().to(x.device).to(torch.float64) if latent_guide_inv is not None else None
    
    y0 = y0.expand_as(x).to(x.device).to(x.dtype)

    sigmas     = sigmas.to(torch.float64)

    if mask is not None:
        mask = mask.unsqueeze(1)
        mask = F.interpolate(mask, size=(x.shape[-2], x.shape[-1]), mode='bilinear', align_corners=False)
        mask = mask.expand_as(x).to(x.device).to(x.dtype)
        extra_args['denoise_mask'] = mask
    else:
        mask = torch.ones_like(x)
    NG         = NoiseGen (model, x, sigmas, seed=EO("seed", torch.initial_seed()+1))
    MC         = ModelCall(model, x, sigmas, extra_args=extra_args, RET_DATA=True)
    
    weight0    = EO("weight0", 0.5)
    weight1    = EO("weight1", 0.0)
    weight2    = EO("weight2", 0.0)
    start_step = EO("start_step", 0)
    end_step   = EO("end_step",   MAX_STEPS)
    
    data_x = None
    schedule = None
    
    #first_step = EO("step", 0)
    x = NG()
    while step < start_step:
        sigma      = sigmas[step]
        sigma_next = sigmas[step+1]
        h          = MC.h_fn(step)
        
        eps, data  = MC(x, sigma, {'latent_type': 'xt'})
        x         += h * eps

        if callback is not None:
            callback({'x': x, 'denoised': data, 'i': step, 'sigma': sigma, 'sigma_hat': sigma})
            
        step += 1
        MC.end_step()
        NG.end_step()

    noise_yt = NG.noise_init.clone()
    noise_xt = NG.noise_init.clone()

    y0 = y0.clone()
    yx0 = y0.clone()
    
    yt = (1-sigmas[step]) * y0  + sigmas[step] * noise_yt
    xt = (1-sigmas[step]) * yx0 + sigmas[step] * noise_xt
    
    while step < end_step:
        sigma      = sigmas[step]
        sigma_next = sigmas[step+1]
        h          = MC.h_fn(step)

        eps_xy = torch.zeros_like(y0)
        
        eps_x, data_x = MC(xt, sigma, {'latent_type': 'xt'})
        eps_y, data_y = MC(yt, sigma, {'latent_type': 'yt'})
        
        eps_xy = eps_x - eps_y

        eps_flow    = eps_xy + noise_xt - y0
        eps_flow_y0 =          noise_yt - y0
        
        noise = NG()

        xt = xt + h * eps_flow
        yt = yt + h * eps_flow_y0
        
        xt = xt + sigma_next * eta * (noise - noise_xt)
        yt = yt + sigma_next * eta * (noise - noise_yt)
        
        noise_xt = noise_xt + eta * (noise - noise_xt)
        noise_yt = noise_yt + eta * (noise - noise_yt)
        
        for i in range(EO("iter", 0)):
            eps_x, data_x = MC(xt, sigma_next, {'latent_type': 'xt'})
            eps_y, data_y = MC(yt, sigma_next, {'latent_type': 'yt'})
            
            eps_xy = eps_x - eps_y

            eps_flow    = eps_xy + noise_xt - y0
            eps_flow_y0 =          noise_yt - y0
            
            noise = NG()

            xt = xt + h * eps_flow
            yt = yt + h * eps_flow_y0
            
            xt = xt + sigma_next * eta * (noise - noise_xt)
            yt = yt + sigma_next * eta * (noise - noise_yt)
            
            noise_xt = noise_xt + eta * (noise - noise_xt)
            noise_yt = noise_yt + eta * (noise - noise_yt)
        

        display=None
        if EO("yx0"):
            display = yx0
        if EO("data_x"):
            display = data_x
        if EO("data_y"):
            display = data_y
        if EO("xt"):
            display = xt
        if EO("yt"):
            display = yt
        if EO("eps_yx"):
            display = (eps_y - eps_x)
        if EO("eps_xy"):
            display = (eps_x - eps_y)
        if EO("data_xy"):
            display = (data_x - data_y)
        if EO("data_yx"):
            display = (data_y - data_x)

        if display is None:
            display = xt - sigma * eps_flow
            
        if callback is not None:
            callback({'x': xt, 'denoised': display, 'i': step, 'sigma': sigma, 'sigma_hat': sigma})
            
        step += 1
        MC.end_step()
        NG.end_step()
        
    if step >= len(sigmas) -1:
        return xt

    #yt = (1 - sigma_next)*y0 + sigma_next * noise             # x = yx0 + (1 - sigma_next)*y0 + sigma_next * noise - y0           x = 
    #yt = NG.linear_noise(y0, sigma_next)
    #x = yx0 + yt - y0
    
    #if EO("x_latent"):
    #    x = x_latent
    #else:
    #    x = yx0 + sigma_next * (NG.noise_init - y0) * mask
    #    if y0_inv is not None:
    #        x = yx0 + sigma_next * (NG.noise_init - y0_inv) * mask
    
    x = xt
    
    while step < len(sigmas)-1:
        sigma      = sigmas[step]
        sigma_next = sigmas[step+1]
        h          = MC.h_fn(step)
        
        eps, data  = MC(x, sigma, {'latent_type': 'xt'})
        
        x += h * eps * mask

        if callback is not None:
            callback({'x': x, 'denoised': data, 'i': step, 'sigma': sigma, 'sigma_hat': sigma})
        
        step += 1
        MC.end_step()
        NG.end_step()
    
    return x





def sample_rk_triflow(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, step=0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None, latent_guide_inv=None, mask=None):
    
    extra_args = {} if extra_args is None else extra_args
    s_in       = x.new_ones([x.shape[0]])
    MAX_STEPS  = 10000
    
    EO         = ExtraOptions(extra_options)

    y0         = model.inner_model.inner_model.process_latent_in(latent_guide    ['samples']).clone().to(x.device).to(torch.float64) if latent_guide     is not None else None
    y0_inv     = model.inner_model.inner_model.process_latent_in(latent_guide_inv['samples']).clone().to(x.device).to(torch.float64) if latent_guide_inv is not None else None
    
    y0 = y0.expand_as(x).to(x.device).to(x.dtype)
    y0_inv = y0_inv.expand_as(x).to(x.device).to(x.dtype) if y0_inv is not None else None

    sigmas     = sigmas.to(torch.float64)

    if mask is not None:
        mask = mask.unsqueeze(1)
        mask = F.interpolate(mask, size=(x.shape[-2], x.shape[-1]), mode='bilinear', align_corners=False)
        mask = mask.expand_as(x).to(x.device).to(x.dtype)
    else:
        mask = torch.ones_like(x)

    NG         = NoiseGen (model, x, sigmas, seed=EO("seed", torch.initial_seed()+1))
    MC         = ModelCall(model, x, sigmas, extra_args=extra_args, RET_DATA=True)
        
    weight0    = EO("weight0", 0.5)
    weight1    = EO("weight1", 0.0)
    weight2    = EO("weight2", 0.0)
    start_step = EO("start_step", 0)
    end_step   = EO("end_step",   MAX_STEPS)
    
    data_x = None
    
    
    #first_step = EO("step", 0)
    x = NG()
    while step < start_step:
        sigma      = sigmas[step]
        sigma_next = sigmas[step+1]
        h          = MC.h_fn(step)
        
        eps, data  = MC(x, sigma, {'latent_type': 'xt'})
        x         += h * eps

        if callback is not None:
            callback({'x': x, 'denoised': data, 'i': step, 'sigma': sigma, 'sigma_hat': sigma})
            
        step += 1
        MC.end_step()
        NG.end_step()
    
    if step > 0:
        y0  = (1-weight0) * data + weight0 * y0
    
    if step > 0:
        y0_inv  = (1-weight0) * data + weight0 * y0_inv
    
    y_slerp = slerp_tensor(0.5, y0.clone(), y0_inv.clone())
    yx0 = y_slerp.clone()
    
    
    while step < end_step:
        sigma      = sigmas[step]
        sigma_next = sigmas[step+1]
        h          = MC.h_fn(step)

        #yt = NG.linear_noise(y0, sigma)
        #xt = yx0 + yt - y0
        
        #yt = y0  + sigma * (NG.noise_init - y0)
        #xt = yx0 + sigma * (NG.noise_init - y0)
        
        for i in range(EO("iter", 1)):
            noise = NG()
            yt      = y0      + sigma * (noise - y0)
            yt_inv  = y0_inv  + sigma * (noise - y0_inv )
            
            xt      = yx0     + sigma * (noise - y_slerp)

            eps_y    , data_y      = MC(yt,     sigma, {'latent_type': 'yt'})
            eps_y_inv, data_y_inv  = MC(yt_inv, sigma, {'latent_type': 'yt'})
            
            eps_x    , data_x      = MC(xt,     sigma, {'latent_type': 'xt'})
            
            eps_y_alt     = (yx0 - data_y)     / sigma
            eps_y_alt_inv = (yx0 - data_y_inv) / sigma

            eps_x_alt     = (yx0 - data_x)     / sigma
            
            
            if   EO("eps_x_minus_y"):
                yx0 += h * (eps_x - eps_y) * mask 
            
            elif EO("data_y_minus_x"):
                yx0 = yx0 + h * (data_y - data_x)
            elif EO("data_x_minus_y"):
                yx0 = yx0 + h * (data_x - data_y)
            
            elif EO("data_y_minus_x_scaled"):
                yx0 = yx0 + h * (data_y - data_x) / sigma
            elif EO("data_x_minus_y_scaled"):                #WORKS
                yx0 = yx0 + h * (data_x - data_y) / sigma
            
            elif EO("alt_eps_x_minus_y"):
                yx0 = yx0 + h * (eps_x_alt - eps_y_alt)
            elif EO("alt_eps_y_minus_x"):
                yx0 = yx0 + h * (eps_y_alt - eps_x_alt)
            
            elif EO("oppo"):
                yx0     += h * (eps_y_inv - eps_x)    
                yx0_inv += h * (eps_y     - eps_x_inv) 
                
            elif EO("oppo0"):
                yx0     += h * (eps_y - eps_x_inv)     
                y0      += h * (eps_y - eps_x_inv)     
                
            elif EO("oppo2"):
                yx0     += h * (eps_y_inv - eps_x)     
                y0      += h * (eps_y_inv - eps_x)     
                
            elif EO("oppo2a"):
                yx0     += h * (eps_y_inv - eps_x)    
                y0      += h * (data_x_inv- data_y) / sigma   
            elif EO("oppo2b"):
                yx0     += h * (eps_y_inv - eps_x)    
                y0      += h * (data_x    - data_y_inv)  / sigma   
            elif EO("oppo3"):
                yx0     += h * (eps_y_inv - eps_x)     
                y0      += h * (eps_y_inv - eps_x)     
                
                yx0_inv += h * (eps_y - eps_x_inv)     
                y0_inv  += h * (eps_y - eps_x_inv)     
            elif EO("prev_default"):
                yx0     += h * (eps_y     - eps_x)     
                yx0_inv += h * (eps_y_inv - eps_x_inv)         
                
                y0      += h * (eps_y_inv - eps_x_inv) 
                y0_inv  += h * (eps_y     - eps_x)    

            elif EO("intersect"):
                yx0 = line_intersection((eps_y - eps_x), y0, (eps_y_inv - eps_x), y0_inv  )


            else:
                eps_y_slerp = slerp_tensor(0.5, eps_y, eps_y_inv)
                yx0     += h * (eps_y_slerp     - eps_x)     
                

                

        #yx0 += weight2 * (data_x - yx0)
        #y0  += weight2 * (data_x - y0)

        if callback is not None:
            callback({'x': yx0, 'denoised': data_x, 'i': step, 'sigma': sigma, 'sigma_hat': sigma})
        
        step += 1
        MC.end_step()
        NG.end_step()
    #yt = (1 - sigma_next)*y0 + sigma_next * noise             # x = yx0 + (1 - sigma_next)*y0 + sigma_next * noise - y0           x = 
    #yt = NG.linear_noise(y0, sigma_next)
    #x = yx0 + yt - y0
    x     = yx0     + sigma_next * (NG.noise_init - y_slerp)
    #x_inv = yx0_inv + sigma_next * (NG.noise_init - y0_inv)
    
    #x = (x + x_inv) / 2
    
    while step < len(sigmas)-1:
        sigma      = sigmas[step]
        sigma_next = sigmas[step+1]
        h          = MC.h_fn(step)
        
        eps, data  = MC(x, sigma, {'latent_type': 'xt'})
        
        x += h * eps

        if callback is not None:
            callback({'x': x, 'denoised': data, 'i': step, 'sigma': sigma, 'sigma_hat': sigma})
        
        step += 1
        MC.end_step()
        NG.end_step()
    
    return x




def sample_rk_triflow_intersection(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, step=0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None, latent_guide_inv=None, mask=None):
    
    extra_args = {} if extra_args is None else extra_args
    s_in       = x.new_ones([x.shape[0]])
    MAX_STEPS  = 10000
    
    EO         = ExtraOptions(extra_options)

    y0         = model.inner_model.inner_model.process_latent_in(latent_guide    ['samples']).clone().to(x.device).to(torch.float64) if latent_guide     is not None else None
    y0_inv     = model.inner_model.inner_model.process_latent_in(latent_guide_inv['samples']).clone().to(x.device).to(torch.float64) if latent_guide_inv is not None else None
    
    y0 = y0.expand_as(x).to(x.device).to(x.dtype)
    y0_inv = y0_inv.expand_as(x).to(x.device).to(x.dtype) if y0_inv is not None else None

    sigmas     = sigmas.to(torch.float64)

    if mask is not None:
        mask = mask.unsqueeze(1)
        mask = F.interpolate(mask, size=(x.shape[-2], x.shape[-1]), mode='bilinear', align_corners=False)
        mask = mask.expand_as(x).to(x.device).to(x.dtype)
    else:
        mask = torch.ones_like(x)

    NG         = NoiseGen (model, x, sigmas, seed=EO("seed", torch.initial_seed()+1))
    MC         = ModelCall(model, x, sigmas, extra_args=extra_args, RET_DATA=True)
        
    weight0    = EO("weight0", 0.5)
    weight1    = EO("weight1", 0.0)
    weight2    = EO("weight2", 0.0)
    start_step = EO("start_step", 0)
    end_step   = EO("end_step",   MAX_STEPS)
    
    data_x = None
    
    
    #first_step = EO("step", 0)
    x = NG()
    while step < start_step:
        sigma      = sigmas[step]
        sigma_next = sigmas[step+1]
        h          = MC.h_fn(step)
        
        eps, data  = MC(x, sigma, {'latent_type': 'xt'})
        x         += h * eps

        if callback is not None:
            callback({'x': x, 'denoised': data, 'i': step, 'sigma': sigma, 'sigma_hat': sigma})
            
        step += 1
        MC.end_step()
        NG.end_step()
    
    if step > 0:
        y0  = (1-weight0) * data + weight0 * y0
    yx0 = y0.clone()
    
    if step > 0:
        y0_inv  = (1-weight0) * data + weight0 * y0_inv
    yx0_inv = y0_inv.clone()
    
    #yx0 = slerp_tensor(0.5, y0.clone(), y0_inv.clone())
    
    
    while step < end_step:
        sigma      = sigmas[step]
        sigma_next = sigmas[step+1]
        h          = MC.h_fn(step)

        #yt = NG.linear_noise(y0, sigma)
        #xt = yx0 + yt - y0
        
        #yt = y0  + sigma * (NG.noise_init - y0)
        #xt = yx0 + sigma * (NG.noise_init - y0)
        
        for i in range(EO("iter", 1)):
            noise = NG()
            yt      = y0      + sigma * (noise - y0)
            yt_inv  = y0_inv  + sigma * (noise - y0_inv )
            
            #xt      = yx0     + sigma * (noise - y0)
            #xt_inv  = yx0_inv + sigma * (noise - y0_inv )
            xt      = yx0     + sigma * (noise - yx0)
            xt_inv  = yx0_inv + sigma * (noise - yx0_inv )

            eps_y    , data_y      = MC(yt,     sigma, {'latent_type': 'yt'})
            eps_y_inv, data_y_inv  = MC(yt_inv, sigma, {'latent_type': 'yt'})
            
            eps_x    , data_x      = MC(xt,     sigma, {'latent_type': 'xt'})
            eps_x_inv, data_x_inv  = MC(xt_inv, sigma, {'latent_type': 'xt'})
            
            eps_y_alt     = (yx0     - data_y)     / sigma
            eps_y_alt_inv = (yx0_inv - data_y_inv) / sigma

            eps_x_alt     = (yx0     - data_x)     / sigma
            eps_x_alt_inv = (yx0_inv - data_x_inv) / sigma
            
            
            if   EO("eps_x_minus_y"):
                yx0 += h * (eps_x - eps_y) * mask 
            
            elif EO("data_y_minus_x"):
                yx0 = yx0 + h * (data_y - data_x)
            elif EO("data_x_minus_y"):
                yx0 = yx0 + h * (data_x - data_y)
            
            elif EO("data_y_minus_x_scaled"):
                yx0 = yx0 + h * (data_y - data_x) / sigma
            elif EO("data_x_minus_y_scaled"):                #WORKS
                yx0 = yx0 + h * (data_x - data_y) / sigma
            
            elif EO("alt_eps_x_minus_y"):
                yx0 = yx0 + h * (eps_x_alt - eps_y_alt)
            elif EO("alt_eps_y_minus_x"):
                yx0 = yx0 + h * (eps_y_alt - eps_x_alt)
            
            elif EO("oppo"):
                yx0     += h * (eps_y_inv - eps_x)     #* mask            #WORKS
                yx0_inv += h * (eps_y     - eps_x_inv) #* mask            #WORKS
                
            elif EO("oppo0"):
                yx0     += h * (eps_y - eps_x_inv)     #* mask            #WORKS
                y0      += h * (eps_y - eps_x_inv)     #* mask            #WORKS
                
            elif EO("oppo2"):
                yx0     += h * (eps_y_inv - eps_x)     #* mask            #WORKS
                y0      += h * (eps_y_inv - eps_x)     #* mask            #WORKS
                
            elif EO("oppo2a"):
                yx0     += h * (eps_y_inv - eps_x)     #* mask            #WORKS
                y0      += h * (data_x_inv- data_y) / sigma    #* mask            #WORKS
            elif EO("oppo2b"):
                yx0     += h * (eps_y_inv - eps_x)     #* mask            #WORKS
                y0      += h * (data_x    - data_y_inv)  / sigma   #* mask            #WORKS
            elif EO("oppo3"):
                yx0     += h * (eps_y_inv - eps_x)     #* mask            #WORKS
                y0      += h * (eps_y_inv - eps_x)     #* mask            #WORKS
                
                yx0_inv += h * (eps_y - eps_x_inv)     #* mask            #WORKS
                y0_inv  += h * (eps_y - eps_x_inv)     #* mask            #WORKS
                
            elif EO("intersect_data_x_minus_y"):                #WORKS
                yx0 = line_intersection((eps_y - eps_x_inv), y0, (eps_y_inv - eps_x), y0_inv  )
                yx0_inv = line_intersection((eps_y_inv - eps_x), y0_inv, (eps_y - eps_x_inv), y0  )
                
                #yx0 = yx0 + h * (data_x - data_y) / sigma
                #yx0_inv = yx0_inv + h * (data_x_inv - data_y_inv) / sigma
                
            elif EO("intersect_data_x_minus_y_2"):                #WORKS
                yx0 = line_intersection((eps_y_inv - eps_x), y0, (eps_y - eps_x_inv), y0_inv  )
                yx0_inv = line_intersection((eps_y - eps_x_inv), y0_inv, (eps_y_inv - eps_x), y0  )
                
                #yx0 = yx0 + h * (data_x - data_y) / sigma
                #yx0_inv = yx0_inv + h * (data_x_inv - data_y_inv) / sigma
                
            elif EO("intersect_data_x_minus_y_scaled"):                #WORKS
                yx0_target = line_intersection((eps_y - eps_x_inv), y0, (eps_y_inv - eps_x), y0_inv  )
                yx0_inv_target = line_intersection((eps_y_inv - eps_x), y0_inv, (eps_y - eps_x_inv), y0  )
                
                yx0 = yx0 + h * (yx0_target - yx0)
                yx0_inv = yx0_inv + h * (yx0_inv_target - yx0_inv)
                
                #yx0 = yx0 + h * (data_x - data_y) / sigma
                #yx0_inv = yx0_inv + h * (data_x_inv - data_y_inv) / sigma
                
            elif EO("intersect_data_x_minus_y_2_scaled"):                #WORKS
                yx0_target = line_intersection((eps_y_inv - eps_x), y0, (eps_y - eps_x_inv), y0_inv  )
                yx0_inv_target = line_intersection((eps_y - eps_x_inv), y0_inv, (eps_y_inv - eps_x), y0  )
                
                yx0 = yx0 + h * (yx0_target - yx0)
                yx0_inv = yx0_inv + h * (yx0_inv_target - yx0_inv)
                
                #yx0 = yx0 + h * (data_x - data_y) / sigma
                #yx0_inv = yx0_inv + h * (data_x_inv - data_y_inv) / sigma
                
                
                
            elif EO("lagrange_data_x_minus_y_scaled"):                #WORKS
                yx0 = yx0 + h * (data_x - data_y) / sigma
                yx0_inv = yx0_inv + h * (data_x_inv - data_y_inv) / sigma
                
            else:
                yx0     += h * (eps_y     - eps_x)     #* mask            #WORKS
                yx0_inv += h * (eps_y_inv - eps_x_inv) #* mask            #WORKS            
                
                y0      += h * (eps_y_inv - eps_x_inv)     #* mask            #WORKS
                y0_inv  += h * (eps_y     - eps_x) #* mask            #WORKS            

        #yx0 += weight2 * (data_x - yx0)
        #y0  += weight2 * (data_x - y0)

        if callback is not None:
            callback({'x': yx0, 'denoised': data_x, 'i': step, 'sigma': sigma, 'sigma_hat': sigma})
        
        step += 1
        MC.end_step()
        NG.end_step()
    #yt = (1 - sigma_next)*y0 + sigma_next * noise             # x = yx0 + (1 - sigma_next)*y0 + sigma_next * noise - y0           x = 
    #yt = NG.linear_noise(y0, sigma_next)
    #x = yx0 + yt - y0
    #x     = yx0     + sigma_next * (NG.noise_init - y0)
    x     = yx0     + sigma_next * (NG.noise_init - yx0)
    #x_inv = yx0_inv + sigma_next * (NG.noise_init - y0_inv)
    
    #x = (x + x_inv) / 2
    
    while step < len(sigmas)-1:
        sigma      = sigmas[step]
        sigma_next = sigmas[step+1]
        h          = MC.h_fn(step)
        
        eps, data  = MC(x, sigma, {'latent_type': 'xt'})
        
        x += h * eps

        if callback is not None:
            callback({'x': x, 'denoised': data, 'i': step, 'sigma': sigma, 'sigma_hat': sigma})
        
        step += 1
        MC.end_step()
        NG.end_step()
    
    return x







def sample_rk_triflow_backup(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, step=0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None, latent_guide_inv=None, mask=None):
    
    extra_args = {} if extra_args is None else extra_args
    s_in       = x.new_ones([x.shape[0]])
    MAX_STEPS  = 10000
    
    EO         = ExtraOptions(extra_options)

    y0         = model.inner_model.inner_model.process_latent_in(latent_guide    ['samples']).clone().to(x.device).to(torch.float64) if latent_guide     is not None else None
    y0_inv     = model.inner_model.inner_model.process_latent_in(latent_guide_inv['samples']).clone().to(x.device).to(torch.float64) if latent_guide_inv is not None else None
    
    y0 = y0.expand_as(x).to(x.device).to(x.dtype)
    y0_inv = y0_inv.expand_as(x).to(x.device).to(x.dtype) if y0_inv is not None else None

    sigmas     = sigmas.to(torch.float64)

    if mask is not None:
        mask = mask.unsqueeze(1)
        mask = F.interpolate(mask, size=(x.shape[-2], x.shape[-1]), mode='bilinear', align_corners=False)
        mask = mask.expand_as(x).to(x.device).to(x.dtype)
    else:
        mask = torch.ones_like(x)

    NG         = NoiseGen (model, x, sigmas, seed=EO("seed", torch.initial_seed()+1))
    MC         = ModelCall(model, x, sigmas, extra_args=extra_args, RET_DATA=True)
        
    weight0    = EO("weight0", 0.5)
    weight1    = EO("weight1", 0.0)
    weight2    = EO("weight2", 0.0)
    start_step = EO("start_step", 0)
    end_step   = EO("end_step",   MAX_STEPS)
    
    data_x = None
    
    
    #first_step = EO("step", 0)
    x = NG()
    while step < start_step:
        sigma      = sigmas[step]
        sigma_next = sigmas[step+1]
        h          = MC.h_fn(step)
        
        eps, data  = MC(x, sigma, {'latent_type': 'xt'})
        x         += h * eps

        if callback is not None:
            callback({'x': x, 'denoised': data, 'i': step, 'sigma': sigma, 'sigma_hat': sigma})
            
        step += 1
        MC.end_step()
        NG.end_step()
    
    if step > 0:
        y0  = (1-weight0) * data + weight0 * y0
    yx0 = y0.clone()
    
    if step > 0:
        y0_inv  = (1-weight0) * data + weight0 * y0_inv
    yx0_inv = y0_inv.clone()
    
    
    while step < end_step:
        sigma      = sigmas[step]
        sigma_next = sigmas[step+1]
        h          = MC.h_fn(step)

        #yt = NG.linear_noise(y0, sigma)
        #xt = yx0 + yt - y0
        
        #yt = y0  + sigma * (NG.noise_init - y0)
        #xt = yx0 + sigma * (NG.noise_init - y0)
        
        for i in range(EO("iter", 1)):
            noise = NG()
            yt      = y0      + sigma * (noise - y0)
            yt_inv  = y0_inv  + sigma * (noise - y0_inv )
            
            xt      = yx0     + sigma * (noise - y0)
            xt_inv  = yx0_inv + sigma * (noise - y0_inv )

            eps_y    , data_y      = MC(yt,     sigma, {'latent_type': 'yt'})
            eps_y_inv, data_y_inv  = MC(yt_inv, sigma, {'latent_type': 'yt'})
            
            eps_x    , data_x      = MC(xt,     sigma, {'latent_type': 'xt'})
            eps_x_inv, data_x_inv  = MC(xt_inv, sigma, {'latent_type': 'xt'})
            
            eps_y_alt     = (yx0     - data_y)     / sigma
            eps_y_alt_inv = (yx0_inv - data_y_inv) / sigma

            eps_x_alt     = (yx0     - data_x)     / sigma
            eps_x_alt_inv = (yx0_inv - data_x_inv) / sigma
            
            
            if   EO("eps_x_minus_y"):
                yx0 += h * (eps_x - eps_y) * mask 
            
            elif EO("data_y_minus_x"):
                yx0 = yx0 + h * (data_y - data_x)
            elif EO("data_x_minus_y"):
                yx0 = yx0 + h * (data_x - data_y)
            
            elif EO("data_y_minus_x_scaled"):
                yx0 = yx0 + h * (data_y - data_x) / sigma
            elif EO("data_x_minus_y_scaled"):                #WORKS
                yx0 = yx0 + h * (data_x - data_y) / sigma
            
            elif EO("alt_eps_x_minus_y"):
                yx0 = yx0 + h * (eps_x_alt - eps_y_alt)
            elif EO("alt_eps_y_minus_x"):
                yx0 = yx0 + h * (eps_y_alt - eps_x_alt)
            
            else:
                yx0     += h * (eps_y_inv - eps_x)     #* mask            #WORKS
                yx0_inv += h * (eps_y     - eps_x_inv) #* mask            #WORKS
            

        #yx0 += weight2 * (data_x - yx0)
        #y0  += weight2 * (data_x - y0)

        if callback is not None:
            callback({'x': yx0, 'denoised': data_x, 'i': step, 'sigma': sigma, 'sigma_hat': sigma})
        
        step += 1
        MC.end_step()
        NG.end_step()
    #yt = (1 - sigma_next)*y0 + sigma_next * noise             # x = yx0 + (1 - sigma_next)*y0 + sigma_next * noise - y0           x = 
    #yt = NG.linear_noise(y0, sigma_next)
    #x = yx0 + yt - y0
    x     = yx0     + sigma_next * (NG.noise_init - y0)
    x_inv = yx0_inv + sigma_next * (NG.noise_init - y0_inv)
    
    x = (x + x_inv) / 2
    
    while step < len(sigmas)-1:
        sigma      = sigmas[step]
        sigma_next = sigmas[step+1]
        h          = MC.h_fn(step)
        
        eps, data  = MC(x, sigma, {'latent_type': 'xt'})
        
        x += h * eps

        if callback is not None:
            callback({'x': x, 'denoised': data, 'i': step, 'sigma': sigma, 'sigma_hat': sigma})
        
        step += 1
        MC.end_step()
        NG.end_step()
    
    return x



def sample_rk_triflow_backup2(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
            sigma_fn_formula="", t_fn_formula="",
                eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, step=0, extra_options="", 
                cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None, latent_guide_inv=None, mask=None):
    
    extra_args = {} if extra_args is None else extra_args
    s_in       = x.new_ones([x.shape[0]])
    MAX_STEPS  = 10000
    
    EO         = ExtraOptions(extra_options)

    y0         = model.inner_model.inner_model.process_latent_in(latent_guide    ['samples']).clone().to(x.device).to(torch.float64) if latent_guide     is not None else None
    y0_inv     = model.inner_model.inner_model.process_latent_in(latent_guide_inv['samples']).clone().to(x.device).to(torch.float64) if latent_guide_inv is not None else None
    
    y0 = y0.expand_as(x).to(x.device).to(x.dtype)
    y0_inv = y0_inv.expand_as(x).to(x.device).to(x.dtype) if y0_inv is not None else None

    sigmas     = sigmas.to(torch.float64)

    if mask is not None:
        mask = mask.unsqueeze(1)
        mask = F.interpolate(mask, size=(x.shape[-2], x.shape[-1]), mode='bilinear', align_corners=False)
        mask = mask.expand_as(x).to(x.device).to(x.dtype)
    else:
        mask = torch.ones_like(x)

    NG         = NoiseGen (model, x, sigmas, seed=EO("seed", torch.initial_seed()+1))
    MC         = ModelCall(model, x, sigmas, extra_args=extra_args, RET_DATA=True)
        
    weight0    = EO("weight0", 0.5)
    weight1    = EO("weight1", 0.0)
    weight2    = EO("weight2", 0.0)
    start_step = EO("start_step", 0)
    end_step   = EO("end_step",   MAX_STEPS)
    
    data_x = None
    
    
    #first_step = EO("step", 0)
    x = NG()
    while step < start_step:
        sigma      = sigmas[step]
        sigma_next = sigmas[step+1]
        h          = MC.h_fn(step)
        
        eps, data  = MC(x, sigma, {'latent_type': 'xt'})
        x         += h * eps

        if callback is not None:
            callback({'x': x, 'denoised': data, 'i': step, 'sigma': sigma, 'sigma_hat': sigma})
            
        step += 1
        MC.end_step()
        NG.end_step()
    
    if step > 0:
        y0  = (1-weight0) * data + weight0 * y0
    yx0 = y0.clone()
    
    if step > 0:
        y0_inv  = (1-weight0) * data + weight0 * y0_inv
    yx0_inv = y0_inv.clone()
    
    yx0 = slerp_tensor(0.5, y0.clone(), y0_inv.clone())
    
    
    while step < end_step:
        sigma      = sigmas[step]
        sigma_next = sigmas[step+1]
        h          = MC.h_fn(step)

        #yt = NG.linear_noise(y0, sigma)
        #xt = yx0 + yt - y0
        
        #yt = y0  + sigma * (NG.noise_init - y0)
        #xt = yx0 + sigma * (NG.noise_init - y0)
        
        for i in range(EO("iter", 1)):
            noise = NG()
            yt      = y0      + sigma * (noise - y0)
            yt_inv  = y0_inv  + sigma * (noise - y0_inv )
            
            xt      = yx0     + sigma * (noise - y0)
            xt_inv  = yx0_inv + sigma * (noise - y0_inv )

            eps_y    , data_y      = MC(yt,     sigma, {'latent_type': 'yt'})
            eps_y_inv, data_y_inv  = MC(yt_inv, sigma, {'latent_type': 'yt'})
            
            eps_x    , data_x      = MC(xt,     sigma, {'latent_type': 'xt'})
            eps_x_inv, data_x_inv  = MC(xt_inv, sigma, {'latent_type': 'xt'})
            
            eps_y_alt     = (yx0     - data_y)     / sigma
            eps_y_alt_inv = (yx0_inv - data_y_inv) / sigma

            eps_x_alt     = (yx0     - data_x)     / sigma
            eps_x_alt_inv = (yx0_inv - data_x_inv) / sigma
            
            
            if   EO("eps_x_minus_y"):
                yx0 += h * (eps_x - eps_y) * mask 
            
            elif EO("data_y_minus_x"):
                yx0 = yx0 + h * (data_y - data_x)
            elif EO("data_x_minus_y"):
                yx0 = yx0 + h * (data_x - data_y)
            
            elif EO("data_y_minus_x_scaled"):
                yx0 = yx0 + h * (data_y - data_x) / sigma
            elif EO("data_x_minus_y_scaled"):                #WORKS
                yx0 = yx0 + h * (data_x - data_y) / sigma
            
            elif EO("alt_eps_x_minus_y"):
                yx0 = yx0 + h * (eps_x_alt - eps_y_alt)
            elif EO("alt_eps_y_minus_x"):
                yx0 = yx0 + h * (eps_y_alt - eps_x_alt)
            
            elif EO("oppo"):
                yx0     += h * (eps_y_inv - eps_x)     #* mask            #WORKS
                yx0_inv += h * (eps_y     - eps_x_inv) #* mask            #WORKS
                
            elif EO("oppo0"):
                yx0     += h * (eps_y - eps_x_inv)     #* mask            #WORKS
                y0      += h * (eps_y - eps_x_inv)     #* mask            #WORKS
                
            elif EO("oppo2"):
                yx0     += h * (eps_y_inv - eps_x)     #* mask            #WORKS
                y0      += h * (eps_y_inv - eps_x)     #* mask            #WORKS
                
            elif EO("oppo2a"):
                yx0     += h * (eps_y_inv - eps_x)     #* mask            #WORKS
                y0      += h * (data_x_inv- data_y) / sigma    #* mask            #WORKS
            elif EO("oppo2b"):
                yx0     += h * (eps_y_inv - eps_x)     #* mask            #WORKS
                y0      += h * (data_x    - data_y_inv)  / sigma   #* mask            #WORKS
            elif EO("oppo3"):
                yx0     += h * (eps_y_inv - eps_x)     #* mask            #WORKS
                y0      += h * (eps_y_inv - eps_x)     #* mask            #WORKS
                
                yx0_inv += h * (eps_y - eps_x_inv)     #* mask            #WORKS
                y0_inv  += h * (eps_y - eps_x_inv)     #* mask            #WORKS
            else:
                yx0     += h * (eps_y     - eps_x)     #* mask            #WORKS
                yx0_inv += h * (eps_y_inv - eps_x_inv) #* mask            #WORKS            
                
                y0      += h * (eps_y_inv - eps_x_inv)     #* mask            #WORKS
                y0_inv  += h * (eps_y     - eps_x) #* mask            #WORKS            

        #yx0 += weight2 * (data_x - yx0)
        #y0  += weight2 * (data_x - y0)

        if callback is not None:
            callback({'x': yx0, 'denoised': data_x, 'i': step, 'sigma': sigma, 'sigma_hat': sigma})
        
        step += 1
        MC.end_step()
        NG.end_step()
    #yt = (1 - sigma_next)*y0 + sigma_next * noise             # x = yx0 + (1 - sigma_next)*y0 + sigma_next * noise - y0           x = 
    #yt = NG.linear_noise(y0, sigma_next)
    #x = yx0 + yt - y0
    x     = yx0     + sigma_next * (NG.noise_init - y0)
    #x_inv = yx0_inv + sigma_next * (NG.noise_init - y0_inv)
    
    #x = (x + x_inv) / 2
    
    while step < len(sigmas)-1:
        sigma      = sigmas[step]
        sigma_next = sigmas[step+1]
        h          = MC.h_fn(step)
        
        eps, data  = MC(x, sigma, {'latent_type': 'xt'})
        
        x += h * eps

        if callback is not None:
            callback({'x': x, 'denoised': data, 'i': step, 'sigma': sigma, 'sigma_hat': sigma})
        
        step += 1
        MC.end_step()
        NG.end_step()
    
    return x








def bagel(cream_cheese):
    while step < len(sigmas)-1:
        sigma      = sigmas[step]
        sigma_next = sigmas[step+1]
        h          = MC.h_fn(step)

        yt = NG.linear_noise(y0, sigma)
        
        if step < end_step:
            xt    = yx0 + yt - y0
            eps_y, data_y = MC(yt, sigma, {'latent_type': 'yt'})

        else:
            if step == end_step:
                yx0 += yt - y0

            xt    = yx0
            eps_y = 0
        
        eps_x, data_x  = MC(xt, sigma, {'latent_type': 'xt'})

        eps_yx = eps_x - eps_y
        
        if step < end_step:
            yx0   += h * eps_yx * mask 
        else:
            yx0   += h * eps_yx
        
        if step < end_step:
            yx0    = yx0 + weight2 * (data_x - yx0)
            y0     = y0  + weight2 * (data_x - y0)

        if callback is not None:
            callback({'x': yx0, 'denoised': data_x, 'i': step, 'sigma': sigma, 'sigma_hat': sigma})
            
        step += 1
        MC.end_step()
        NG.end_step()

    return yx0





def sample_rk_flow_works(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, step=0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None, latent_guide_inv=None, mask=None):
    
    extra_args = {} if extra_args is None else extra_args
    s_in       = x.new_ones([x.shape[0]])
    MAX_STEPS  = 10000
    
    EO         = ExtraOptions(extra_options)

    y0         = model.inner_model.inner_model.process_latent_in(latent_guide    ['samples']).clone().to(x.device).to(torch.float64) if latent_guide     is not None else None
    y0_inv     = model.inner_model.inner_model.process_latent_in(latent_guide_inv['samples']).clone().to(x.device).to(torch.float64) if latent_guide_inv is not None else None

    sigmas     = sigmas.to(torch.float64)

    if mask is not None:
        mask = mask.unsqueeze(1)
        mask = F.interpolate(mask, size=(x.shape[-2], x.shape[-1]), mode='bilinear', align_corners=False)
        mask = mask.expand_as(x).to(x.device).to(x.dtype)
    else:
        mask = torch.ones_like(x)

    NG         = NoiseGen (model, x, sigmas, seed=EO("seed", torch.initial_seed()+1))
    MC         = ModelCall(model, x, sigmas, extra_args=extra_args, RET_DATA=True)
    
    noise = NG()
    
    weight0    = EO("weight0", 1.0)
    weight1    = EO("weight1", 1.0)
    weight2    = EO("weight2", 1.0)
    start_step = EO("start_step", 0)
    end_step   = EO("end_step",   MAX_STEPS)
    
    data_x = None
    
    y0_orig = y0.clone()
    yx0     = y0.clone()
    
    step = EO("step", 0)
    while step < len(sigmas)-1:
        sigma      = sigmas[step]
        sigma_next = sigmas[step+1]
        h          = MC.h_fn(step)

        #yt = NG.linear_noise(y0, sigma)
        noise = NG(step)
        yt = (1-sigma) * y0 + sigma * noise
        
        if step < end_step:
            xt    = yx0 + yt - y0
            eps_y, data_y = MC(yt, sigma, {'latent_type': 'yt'})

        else:
            if step == end_step:
                yx0 += yt - y0

            xt    = yx0
            eps_y = 0
        
        eps_x, data_x  = MC(xt, sigma, {'latent_type': 'xt'})

        eps_yx = eps_x - eps_y
        
        if step < end_step:
            yx0   += h * eps_yx * mask 
        else:
            yx0   += h * eps_yx
        
        if step < end_step:
            yx0    = yx0 + weight2 * (data_x - yx0)
            y0     = y0  + weight2 * (data_x - y0)

        if callback is not None:
            callback({'x': yx0, 'denoised': data_x, 'i': step, 'sigma': sigma, 'sigma_hat': sigma})
            
        step += 1
        MC.end_step()
        NG.end_step()

    return yx0






def sample_rk_fedit(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None, latent_guide_inv=None, mask=None):
    extra_args = {} if extra_args is None else extra_args
    
    model_options = extra_args.get('model_options', {})
    transformer_options = model_options.get('transformer_options', {})
    transformer_options = {**transformer_options}
    model_options['transformer_options'] = transformer_options
    extra_args['model_options'] = model_options
    source_extra_args = {**extra_args, 'model_options': { 'transformer_options': { **transformer_options,'latent_type': 'source '} }}

    if mask is not None:
        mask = mask.unsqueeze(1)
        mask = F.interpolate(mask, size=(x.shape[-2], x.shape[-1]), mode='bilinear', align_corners=False)
        mask = mask.expand_as(x).to(x.device).to(x.dtype)
        extra_args['denoise_mask'] = mask
        source_extra_args['denoise_mask'] = mask
    else:
        mask = torch.ones_like(x)
    
    s_in = x.new_ones([x.shape[0]])
    seed = get_extra_options_kv("seed", 42, extra_options)
    generator = torch.manual_seed(seed)
    if latent_guide is not None:
        y0 = model.inner_model.inner_model.process_latent_in(latent_guide['samples']).clone().to(x.device)
    y0 = y0.to(torch.float64)
    sigmas = sigmas.to(torch.float64)
    if y0.ndim == 5:
        x = y0 = y0.repeat(1,1,x.shape[-3],1,1).clone()
    else:
        x = y0.clone()

    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
    noise_sampler2 = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+10000, sigma_min=0.0, sigma_max=1.0)
    
    weight = get_extra_options_kv("weight", 1.0, extra_options)
    stop_step = get_extra_options_kv("stop_step", 10000, extra_options)

    sigma, sigma_next = sigmas[0], sigmas[1]
    noise = noise_sampler(sigma=sigma, sigma_next=sigma_next)
    noise = (noise - noise.mean()) / noise.std()
    
    #z_tar = noise.clone()
    
    #t_mask = torch.ones_like(x)
    #t_mask_inv = 1-t_mask
    
    #x = x + t_mask_inv * (noise - x)
    refine_steps = stop_step

    x_init = x  
    x_tgt = x_init.clone()
    N = len(sigmas)-1
    s_in = x_init.new_ones([x_init.shape[0]])

    for i in trange(N, disable=disable):
        sigma = sigmas[i]
        noise = torch.randn(x_init.shape, generator=generator).to(x_init.device)

        zt_src = (1-sigma)*x_init + sigma*noise
        
        if i < N-refine_steps:
            zt_tgt = x_tgt + zt_src - x_init 
            transformer_options['latent_type'] = 'source'    # yt
            source_extra_args['model_options']['transformer_options']['latent_type'] = 'source'
            vt_src = model(zt_src, sigma*s_in, **source_extra_args)
            vt_src = (zt_src - vt_src) / sigma
        else:
            if i == N-refine_steps:
                #x_tgt = x_tgt + zt_src - x_init
                zt_tgt = x_tgt + (zt_src - x_init)
                x_tgt = x_tgt + (zt_src - x_init) * mask
            #zt_tgt = x_tgt
            zt_tgt = x_tgt * (mask) + (1-mask) * ( (1-sigma)*x_tgt + sigma*noise )   # zt_tgt = yx0
            vt_src = 0
            
        transformer_options['latent_type'] = 'target'         # xt
        vt_tgt = model(zt_tgt, sigma*s_in, **extra_args)
        vt_tgt = (zt_tgt - vt_tgt) / sigma
        
        v_delta = vt_tgt - vt_src
        x_tgt += (sigmas[i+1] - sigmas[i]) * v_delta * mask
    
            
        display=None
        if extra_options_flag("z_tar", extra_options):
            display = zt_tgt
        if extra_options_flag("y0_noised", extra_options):
            display = y0_noised
        if extra_options_flag("x_tgt", extra_options):
            display = x_tgt
        if extra_options_flag("denoised_src", extra_options):
            display = denoised_src
        if extra_options_flag("denoised_tar", extra_options):
            display = denoised_tar
        if extra_options_flag("eps_src", extra_options):
            display = eps_src
        if extra_options_flag("eps_tar", extra_options):
            display = eps_tar
            
        if extra_options_flag("denoised", extra_options):
            display = denoised
        if extra_options_flag("eps", extra_options):
            display = eps
        if display is None:
            display = x_tgt
            
        if callback is not None:
            callback({'x': x_tgt, 'denoised': display, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i]})

    return x_tgt





def sample_rk_fedit_meh(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
            sigma_fn_formula="", t_fn_formula="",
                eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    seed = get_extra_options_kv("seed", 42, extra_options)
    generator = torch.manual_seed(seed)
    if latent_guide is not None:
        y0 = model.inner_model.inner_model.process_latent_in(latent_guide['samples']).clone().to(x.device)
    y0 = y0.to(torch.float64)
    sigmas = sigmas.to(torch.float64)
    if y0.ndim == 5:
        x = y0 = y0.repeat(1,1,x.shape[-3],1,1).clone()
    else:
        x = y0.clone()

    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
    noise_sampler2 = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+10000, sigma_min=0.0, sigma_max=1.0)
    
    weight = get_extra_options_kv("weight", 1.0, extra_options)
    stop_step = get_extra_options_kv("stop_step", 10000, extra_options)

    sigma, sigma_next = sigmas[0], sigmas[1]
    noise = noise_sampler(sigma=sigma, sigma_next=sigma_next)
    noise = (noise - noise.mean()) / noise.std()
    
    #z_tar = noise.clone()
    
    #t_mask = torch.ones_like(x)
    #t_mask_inv = 1-t_mask
    
    #x = x + t_mask_inv * (noise - x)
    refine_steps = stop_step

    x_init = x  
    x_tgt = x_init.clone()
    N = len(sigmas)-1
    s_in = x_init.new_ones([x_init.shape[0]])

    for i in trange(N, disable=disable):
        sigma = sigmas[i]
        noise = torch.randn(x_init.shape, generator=generator).to(x_init.device)

        zt_src = (1-sigma)*x_init + sigma*noise
        
        if i < N-refine_steps:
            zt_tgt = x_tgt + zt_src - x_init
            vt_src = model(zt_src, sigma*s_in, **extra_args)
        else:
            if i == N-refine_steps:
                x_tgt = x_tgt + zt_src - x_init
            zt_tgt = x_tgt
            vt_src = 0
            
        vt_tgt = model(zt_tgt, sigma*s_in, **extra_args)
        
        v_delta = vt_tgt - vt_src
        x_tgt += (sigmas[i+1] - sigmas[i]) * v_delta
        
            
        display=None
        if extra_options_flag("z_tar", extra_options):
            display = z_tar
        if extra_options_flag("y0_noised", extra_options):
            display = y0_noised
        if extra_options_flag("x_tgt", extra_options):
            display = x_tgt
        if extra_options_flag("denoised_src", extra_options):
            display = denoised_src
        if extra_options_flag("denoised_tar", extra_options):
            display = denoised_tar
        if extra_options_flag("eps_src", extra_options):
            display = eps_src
        if extra_options_flag("eps_tar", extra_options):
            display = eps_tar
            
        if extra_options_flag("denoised", extra_options):
            display = denoised
        if extra_options_flag("eps", extra_options):
            display = eps
        if display is None:
            display = x_tgt
            
        if callback is not None:
            callback({'x': x_tgt, 'denoised': display, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i]})

    return x_tgt





def sample_rk_fedit_x_is_y0(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    if latent_guide is not None:
        y0 = model.inner_model.inner_model.process_latent_in(latent_guide['samples']).clone().to(x.device)
    y0 = y0.to(torch.float64)
    sigmas = sigmas.to(torch.float64)
    if y0.ndim == 5:
        x = y0 = y0.repeat(1,1,x.shape[-3],1,1).clone()
    else:
        x = y0.clone()

    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
    noise_sampler2 = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+10000, sigma_min=0.0, sigma_max=1.0)
    
    weight = get_extra_options_kv("weight", 1.0, extra_options)
    stop_step = get_extra_options_kv("stop_step", 10000, extra_options)

    sigma, sigma_next = sigmas[0], sigmas[1]
    noise = noise_sampler(sigma=sigma, sigma_next=sigma_next)
    noise = (noise - noise.mean()) / noise.std()
    
    #z_tar = noise.clone()
    
    #t_mask = torch.ones_like(x)
    #t_mask_inv = 1-t_mask
    
    #x = x + t_mask_inv * (noise - x)

    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]

        noise = noise_sampler(sigma=sigma, sigma_next=sigma_next)
        noise = (noise - noise.mean()) / noise.std()

        h = sigma_next - sigma
        
        if sigma_next == 0:
            x = denoised = model(x, sigma * s_in, **extra_args)
            
        elif step < stop_step:
            y0_noised    = (1-sigma) * y0 + sigma * noise            #UNSAMPLE
            denoised_src = model(y0_noised, sigma * s_in, **extra_args)
            eps_src      = (y0_noised - denoised_src) / sigma
            #eps_src      = (denoised_src - y0_noised) / sigma
            
            #z_tar        = x + (y0_noised - y0)                #ISNT THIS JUST MY GUIDE ADDITION?
            z_tar        = x + sigma * (noise - y0)
            #z_tar        = x + (-sigma) * y0 + sigma * noise
            """if step == 0:
                z_tar        = x + (-sigma) * y0 + sigma * noise
            else:
                z_tar        = (denoised + sigma * eps) + (-sigma) * y0 + sigma * noise"""
            denoised_tar = model(z_tar, sigma * s_in, **extra_args)
            eps_tar      = (z_tar - denoised_tar) / sigma
            
            x = x + h * (eps_tar - eps_src)
            #x = x + h * -eps_src
            #x = x + h * eps_tar

            if step == stop_step - 1:
                x = z_tar + h * eps_tar
            #else:
            #    x = x + h * (eps_tar - t_mask * eps_src)

            denoised = denoised_tar
            eps = (x - denoised) / sigma
            
        else:
            denoised = model(x, sigma * s_in, **extra_args)
            eps      = (x - denoised) / sigma
            x        = x + h * eps
            
        if extra_options_flag("z_tar", extra_options):
            display = z_tar
        if extra_options_flag("y0_noised", extra_options):
            display = y0_noised
        if extra_options_flag("x", extra_options):
            display = x
        if extra_options_flag("denoised_src", extra_options):
            display = denoised_src
        if extra_options_flag("denoised_tar", extra_options):
            display = denoised_tar
        if extra_options_flag("eps_src", extra_options):
            display = eps_src
        if extra_options_flag("eps_tar", extra_options):
            display = eps_tar
            
        if extra_options_flag("denoised", extra_options):
            display = denoised
        if extra_options_flag("eps", extra_options):
            display = eps
            
        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': display})

    return x





def sample_rk_fedit_works(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    if latent_guide is not None:
        y0 = model.inner_model.inner_model.process_latent_in(latent_guide['samples']).clone().to(x.device)

    z_fe = y0.clone()

    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
    noise_sampler2 = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+10000, sigma_min=0.0, sigma_max=1.0)
    
    weight = get_extra_options_kv("weight", 1.0, extra_options)
    stop_step = get_extra_options_kv("stop_step", 10000, extra_options)

    sigma, sigma_next = sigmas[0], sigmas[1]
    noise = noise_sampler(sigma=sigma, sigma_next=sigma_next)
    noise = (noise - noise.mean()) / noise.std()

    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]

        h = sigma_next - sigma
        
        if sigma_next == 0:
            x = denoised = model(x, sigma * s_in, **extra_args)
            
        elif step < stop_step:
            
            z_src = (1-sigma) * y0 + sigma * noise

            z_tar = z_fe + z_src - y0
            
            denoised_tar = denoised = model(z_tar, sigma * s_in, **extra_args)
            eps_tar = (z_tar - denoised_tar) / sigma

            denoised_src            = model(z_src, sigma * s_in, **extra_args)
            eps_src = (z_src - denoised_src) / sigma
            
            v_delta = eps_tar - eps_src
            
            if step == stop_step - 1:
                x = z_tar + h * eps_tar
                #x = z_fe + h * v_delta
            else:
                z_fe = z_fe + h * v_delta
                x    = z_fe
            
        else:
            denoised = model(x, sigma * s_in, **extra_args)
            eps = (x - denoised) / sigma
            x = x + h * eps
            
        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised})

    return x






def sample_rk_fedit2(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    if latent_guide is not None:
        y0 = model.inner_model.inner_model.process_latent_in(latent_guide['samples']).clone().to(x.device)

    z_fe = y0.clone()

    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
    noise_sampler2 = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+10000, sigma_min=0.0, sigma_max=1.0)
    
    weight = get_extra_options_kv("weight", 1.0, extra_options)
    stop_step = get_extra_options_kv("stop_step", 10000, extra_options)


    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
        
        #sigma_up, sigma, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, eta, noise_mode)
        
        #x_0 = x.clone()
    
        h = sigma_next - sigma
        
        #z_fe = x.clone()
        
        if sigma_next == 0:
            x = denoised = model(x, sigma * s_in, **extra_args)
        elif step < stop_step:
            

            noise = noise_sampler(sigma=sigma, sigma_next=sigma_next)
            noise = (noise - noise.mean()) / noise.std()
            
            z_src = (1-sigma) * y0 + sigma * noise
            
            if step == stop_step:
                z_fe = z_fe + z_src - y0
                z_tar = z_fe
            else:
                z_tar = z_fe + z_src - y0
            
            denoised_tar = denoised = model(z_tar, sigma * s_in, **extra_args)
            eps_tar = (z_tar - denoised_tar) / sigma

            if step < stop_step:
                denoised_src            = model(z_src, sigma * s_in, **extra_args)
                eps_src = (z_src - denoised_src) / sigma
            else: 
                eps_src = 0
            
            v_delta = eps_tar - eps_src
            z_fe    = z_fe + h * v_delta
            
            x = z_fe
        else:
            denoised = model(x, sigma * s_in, **extra_args)
            eps = (x - denoised) / sigma
            x = x + h * eps
            

        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised})

    return x





def sample_rk_fedit3(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    if latent_guide is not None:
        y0 = model.inner_model.inner_model.process_latent_in(latent_guide['samples']).clone().to(x.device)
    y0 = y0.repeat(1,1,x.shape[-3],1,1)
    z_fe = y0.clone()

    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
    noise_sampler2 = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+10000, sigma_min=0.0, sigma_max=1.0)
    
    weight = get_extra_options_kv("weight", 1.0, extra_options)
    stop_step = get_extra_options_kv("stop_step", 10000, extra_options)


    x_tgt = y0.clone()
    N = len(sigmas)-1
    s_in = y0.new_ones([y0.shape[0]])

    for i in trange(N, disable=disable):
        sigma = sigmas[i]
        sigma_next = sigmas[i+1]
        noise = noise_sampler(sigma=sigma, sigma_next=sigma_next)
        #noise = torch.randn(y0.shape, generator=generator).to(y0.device)

        zt_src = (1-sigma)*y0 + sigma*noise
        
        if i < N-stop_step:
            zt_tgt = x_tgt + zt_src - y0
            vt_src = model(zt_src, sigma*s_in, **extra_args)
        else:
            if i == N-stop_step:
                x_tgt = x_tgt + zt_src - y0
            zt_tgt = x_tgt
            vt_src = 0
        
        vt_tgt = model(zt_tgt, sigma*s_in, **extra_args)
        
        v_delta = vt_tgt - vt_src
        x_tgt += (sigmas[i+1] - sigmas[i]) * v_delta
        
        if callback is not None:
            callback({'x': x_tgt, 'denoised': x_tgt, 'i': i+0, 'sigma': sigmas[i], 'sigma_hat': sigmas[i]})

    return x_tgt
    









@torch.no_grad()
def sample_er_sde_comfy(model, x, sigmas, extra_args=None, callback=None, disable=None, s_noise=1., noise_sampler=None, noise_scaler=None, max_stage=3):
    """
    Extended Reverse-Time SDE solver (VE ER-SDE-Solver-3). Arxiv: https://arxiv.org/abs/2309.06169.
    Code reference: https://github.com/QinpengCui/ER-SDE-Solver/blob/main/er_sde_solver.py.
    """
    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)
    noise_sampler = default_noise_sampler(x, seed=seed) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])

    def default_noise_scaler(sigma):
        return sigma * ((sigma ** 0.3).exp() + 10.0)
    noise_scaler = default_noise_scaler if noise_scaler is None else noise_scaler
    num_integration_points = 200.0
    point_indice = torch.arange(0, num_integration_points, dtype=torch.float32, device=x.device)

    old_denoised = None
    old_denoised_d = None

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        stage_used = min(max_stage, i + 1)
        if sigmas[i + 1] == 0:
            x = denoised
        elif stage_used == 1:
            r = noise_scaler(sigmas[i + 1]) / noise_scaler(sigmas[i])
            x = r * x + (1 - r) * denoised
        else:
            r = noise_scaler(sigmas[i + 1]) / noise_scaler(sigmas[i])
            x = r * x + (1 - r) * denoised

            dt = sigmas[i + 1] - sigmas[i]
            sigma_step_size = -dt / num_integration_points
            sigma_pos = sigmas[i + 1] + point_indice * sigma_step_size
            scaled_pos = noise_scaler(sigma_pos)

            # Stage 2
            s = torch.sum(1 / scaled_pos) * sigma_step_size
            denoised_d = (denoised - old_denoised) / (sigmas[i] - sigmas[i - 1])
            x = x + (dt + s * noise_scaler(sigmas[i + 1])) * denoised_d

            if stage_used >= 3:
                # Stage 3
                s_u = torch.sum((sigma_pos - sigmas[i]) / scaled_pos) * sigma_step_size
                denoised_u = (denoised_d - old_denoised_d) / ((sigmas[i] - sigmas[i - 2]) / 2)
                x = x + ((dt ** 2) / 2 + s_u * noise_scaler(sigmas[i + 1])) * denoised_u
            old_denoised_d = denoised_d

        if s_noise != 0 and sigmas[i + 1] > 0:
            x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * (sigmas[i + 1] ** 2 - sigmas[i] ** 2 * r ** 2).sqrt()
        old_denoised = denoised
    return x





@torch.no_grad()
def sample_er_sde(model, x, sigmas, extra_args=None, callback=None, disable=None, s_noise=1., eta=0.05, noise_sampler=None, noise_scaler=None, max_stage=3, **kwargs):
    """
    Extended Reverse-Time SDE solver (VE ER-SDE-Solver-3). Arxiv: https://arxiv.org/abs/2309.06169.
    Code reference: https://github.com/QinpengCui/ER-SDE-Solver/blob/main/er_sde_solver.py.
    """
    extra_args    = {} if extra_args is None else extra_args
    seed          = extra_args.get("seed", None)
    noise_sampler = default_noise_sampler(x, seed=seed) if noise_sampler is None else noise_sampler
    s_in          = x.new_ones([x.shape[0]])

    def default_noise_scaler(sigma):
        return sigma * ((sigma ** 0.3).exp() + 10.0)
    
    noise_scaler = default_noise_scaler if noise_scaler is None else noise_scaler
    num_integration_points = 200.0
    point_indice = torch.arange(0, num_integration_points, dtype=torch.float32, device=x.device)

    old_denoised = None
    old_denoised_d = None

    for i in trange(len(sigmas)-1, disable=disable):
        sigma_next, sigma = sigmas[i+1], sigmas[i]
        
        sigma_prev  = sigmas[i-1] if i > 0 else None
        sigma_prev2 = sigmas[i-2] if i > 1 else None
        
        h       = sigma_next - sigma
        h_prev  = sigma - sigma_prev if sigma_prev is not None else None
        h_prev2 = sigma - sigma_prev2 if sigma_prev2 is not None else None
        
        sigma_up = sigma_next * eta
        alpha_ratio = (  (sigma_next**2 - sigma_up**2) / sigma**2) ** 0.5
        
        #alpha_ratio = noise_scaler(sigma_next) / noise_scaler(sigma)
        #sigma_up    = (sigma_next ** 2 - sigma ** 2 * alpha_ratio ** 2) ** 0.5
        
        denoised = model(x, sigma * s_in, **extra_args)
        
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigma, 'sigma_hat': sigma, 'denoised': denoised})
            
        stage_used = min(max_stage, i + 1)
        
        if sigma_next == 0:
            x = denoised
        else:
            #alpha_ratio = noise_scaler(sigma_next) / noise_scaler(sigma)
            x   =   alpha_ratio * x   +   (1-alpha_ratio) * denoised

            if stage_used >= 2:    
                sigma_step_size = -h / num_integration_points
                sigma_pos       = sigma_next + point_indice * sigma_step_size
                scaled_pos      = noise_scaler(sigma_pos)

                # Stage 2
                s          = torch.sum(1 / scaled_pos) * sigma_step_size
                denoised_d = (denoised - old_denoised) / h_prev
                x          = x + (h + s * noise_scaler(sigma_next)) * denoised_d

                if stage_used >= 3:
                    # Stage 3
                    s_u        = torch.sum((sigma_pos - sigma) / scaled_pos) * sigma_step_size
                    denoised_u = (denoised_d - old_denoised_d) / (h_prev2 / 2)
                    x          = x + ((h ** 2) / 2 + s_u * noise_scaler(sigma_next)) * denoised_u
                
                old_denoised_d = denoised_d

        if s_noise != 0 and sigma_next > 0:
            noise    = noise_sampler(sigma, sigma_next) * s_noise
            #sigma_up = (sigma_next ** 2 - sigma ** 2 * alpha_ratio ** 2) ** 0.5
            x        = x + noise * sigma_up
        
        old_denoised = denoised
        
    return x







def sample_rk_gausscycle(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
    noise_sampler2 = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+10000, sigma_min=0.0, sigma_max=1.0)
    
    x_next_pred = None
    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
        
        sigma_up, sigma, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, eta, noise_mode)
        
        x_0 = x.clone()
        
        a1_1, a1_2 = 1/4, 1/4 - 3**0.5 / 6
        a2_1, a2_2 = 1/4 + 3**0.5 / 6, 1/4
        b1, b2 = 1/2, 1/2
        c1, c2 = 1/2 - 3**0.5 / 6, 1/2 + 3**0.5 / 6
        
        alpha_ratio_3 = 1
        sigma_up_3 = 0
        sigma_down_3 = sigma_down
        if sigma_next > 0:
            sigma_down_3 = sigma_next + (sigmas[step+2] - sigmas[step+1]) * c1
            alpha_ratio_3, sigma_up_3, sigma_down_3 = get_alpha_ratio_from_sigma_down(sigma_down_3, sigma_next, eta)
            h = sigma_down_3 - sigma
        else:
            h_no_eta = sigma_next - sigma
            h = sigma_down - sigma
        
        sigma_1 = sigma + h * c1
        sigma_2 = sigma + h * c2
        
        if sigma_next == 0:
            return denoised_2
        
        if x_next_pred == None:
            denoised = model(x, sigma * s_in, **extra_args)
            if sigma_next == 0:
                return denoised
            eps = (x - denoised)/sigma
            
            eps_1 = eps * sigma_1
            eps_2 = eps * sigma_2 
            
            x_1 = x_0 + h * (a1_1 * eps_1 + a1_2 * eps_2)
        else:
            x_1 = x_next_pred
            if extra_options_flag("new_guess", extra_options):
                eps_1 = (x_0 - denoised_prev) / sigma_1   
                eps_2 = (x_0 - denoised_prev) / sigma_2   
                x_1 = x_0 + h * (a1_1 * eps_1 + a1_2 * eps_2)

        denoised_1 = model(x_1, sigma_1 * s_in, **extra_args)
        
        #if x_next_pred is not None:
        #    eps_2 = (x_0 - denoised_1) / sigma_2
        #eps_1 = (x_0 - denoised_1) / sigma_1                       # will be for c1 for radau ia 
        
        if x_next_pred is not None:
            eps_2 = (x_0 - denoised_1) / sigma_2
        eps_1 = (x_0 - denoised_1) / sigma_1                      # will be for c1 for radau ia 
        
        sub_sigma_next = sigma_2
        sub_sigma_up, sub_sigma, sub_sigma_down, sub_alpha_ratio = get_res4lyf_step_with_model(model, sigma, sub_sigma_next, eta_var, noise_mode)
        h_new = h * (sub_sigma_down - sigma) / (sub_sigma_next - sigma) 
        
        x_2 = x_0 + h_new * (a2_1 * eps_1 + a2_2 * eps_2)
        
        noise = noise_sampler2(sigma=sigma, sigma_next=sigma_next)
        noise = (noise - noise.mean()) / noise.std()
        x_2 = sub_alpha_ratio * x_2 + noise * sub_sigma_up
        
        
        
        
        denoised_2 = model(x_2, sigma_2 * s_in, **extra_args) 
        eps_2 = (x_0 - denoised_2) / sigma_2                       # c = 1 maps to 2/3 for radau ia, as in, c2 as well



        #sub_sigma_next = sigma_down_3
        #sub_sigma_up, sub_sigma, sub_sigma_down, sub_alpha_ratio = get_res4lyf_step_with_model(model, sigma, sub_sigma_next, eta_var, noise_mode)
        #h_new = h * (sub_sigma_down - sigma) / (sub_sigma_next - sigma) 
        
        x_next = x_next_pred = x_0 + h * (b1 * eps_1 + b2 * eps_2)
        
        #x_next = x_next_pred = x_0 + h_new * (b1 * eps_1 + b2 * eps_2)
        #noise = noise_sampler2(sigma=sigma, sigma_next=sigma_next)
        #noise = (noise - noise.mean()) / noise.std()
        #x_next = sub_alpha_ratio * x_next + noise * sub_sigma_up
        
        noise1 = noise_sampler(sigma=sigma, sigma_next=sigma_next)
        noise1 = (noise1 - noise1.mean()) / noise1.std()
        
        x_next = x_next_pred = alpha_ratio * x_next + noise * sigma_up

        x = x_next
        
        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised_2})

        x = alpha_ratio_3 * x + noise * s_noise * sigma_up_3   
         
        #if extra_options_flag("noise1", extra_options):
        #    x = alpha_ratio_3 * x + noise1 * s_noise * sigma_up_3   
        #else:
        #    x = alpha_ratio_3 * x + noise * s_noise * sigma_up_3     
        
        denoised_prev = denoised_2   #THIS MIGHT BE COMPLETELY WRONG!!!!!!!!!!!!!!!!!!!!! just hiding pylance error

    return denoised



def lagrange_interpolation(x_values, y_values, x_new):
    dtype = y_values[0].dtype
    device = y_values[0].device

    if not isinstance(x_values, torch.Tensor):
        x_values = torch.tensor(x_values, dtype=dtype, device=device)
    if x_values.ndim != 1:
        raise ValueError("x_values must be a 1D tensor or a list of scalars.")

    if not isinstance(x_new, torch.Tensor):
        x_new = torch.tensor(x_new, dtype=dtype, device=device)
    if x_new.ndim == 0:
        x_new = x_new.unsqueeze(0)

    if isinstance(y_values, list):
        y_values = torch.stack(y_values, dim=0)
    if y_values.ndim < 1:
        raise ValueError("y_values must have at least one dimension (the sample dimension).")

    n = x_values.shape[0]
    if y_values.shape[0] != n:
        raise ValueError(f"Mismatch: x_values has length {n} but y_values has {y_values.shape[0]} samples.")

    m = x_new.shape[0]
    result_shape = (m,) + y_values.shape[1:]
    result = torch.zeros(result_shape, dtype=dtype, device=device)

    for i in range(n):
        Li = torch.ones_like(x_new, dtype=dtype, device=device)
        xi = x_values[i]
        for j in range(n):
            if i == j:
                continue
            xj = x_values[j]
            Li = Li * ((x_new - xj) / (xi - xj))
        extra_dims = (1,) * (y_values.ndim - 1)
        Li = Li.view(m, *extra_dims)
        result = result + Li * y_values[i]

    return result






def sample_rk_radau_ia_2s_lang_full(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
    noise_sampler2 = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+10000, sigma_min=0.0, sigma_max=1.0)
    
    A_tableau = torch.tensor([[1/4, 1/4 - 3**0.5 / 6], [1/4 + 3**0.5 / 6, 1/4],], dtype=x.dtype, device=x.device)
    B_row = torch.tensor([1/2, 1/2], dtype=x.dtype, device=x.device)
    A_inv = torch.linalg.inv(A_tableau)
    D_row = B_row @ A_inv
    
    denoised_prev, denoised_prev2 = [torch.zeros_like(x) for _ in range(2)]
    
    x_next_pred = None
    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
        
        sigma_up, sigma, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, eta, noise_mode)
        
        x_0 = x.clone()
    
        
        a1_1, a1_2 = 1/4, -1/4
        a2_1, a2_2 = 1/4, 5/12
        b1, b2 = 1/4, 3/4
        c1, c2 = 0, 2/3
        
        h = sigma_next - sigma
        
        sigma_1 = sigma + h * c1
        sigma_2 = sigma + h * c2
        
        if sigma_next == 0:
            return denoised_2
        
        if step == 0:
            denoised = model(x, sigma * s_in, **extra_args)
            if sigma_next == 0:
                return denoised
            eps = (x - denoised)/sigma
            
            eps_1 = eps * sigma_1
            eps_2 = eps * sigma_2 
            
            x_1 = x_0 + h * (a1_1 * eps_1 + a1_2 * eps_2)
            x_2 = x_0 + h * (a2_1 * eps_1 + a2_2 * eps_2)
            
        else:
            sigma_prev = sigmas[step-1]
            h_prev = sigma - sigma_prev
            
            w = h / h_prev

            #zi_1 = lagrange_interpolation([c1,c2,1], [z_1_prev, z_2_prev, z_next_prev], 1 + w*c1).squeeze(0) + x_prev - x_0
            
            #zi_1 = lagrange_interpolation([c1,c2,1], [z_1_prev, z_2_prev, z_next_prev], 1 + w*c1).squeeze(0) + x_prev - x_0
            #zi_2 = lagrange_interpolation([c1,c2,1], [z_1_prev, z_2_prev, z_next_prev], 1 + w*c2).squeeze(0) + x_prev - x_0
            
            zi_1 = lagrange_interpolation([c1,c2], [z_1_prev, z_2_prev], 1 + w*c1).squeeze(0) + x_prev - x_0
            zi_2 = lagrange_interpolation([c1,c2], [z_1_prev, z_2_prev], 1 + w*c2).squeeze(0) + x_prev - x_0
            
            #x_1 = x_0 + zi_1
            x_1 = x_0
            x_2 = x_0 + zi_2
        
        for full_iter in range(sub_iter):
            denoised_1 = model(x_1, sigma_1 * s_in, **extra_args)
            
            eps_1 = (x_0 - denoised_1) / sigma_1
            eps_2 = (x_0 - denoised_1) / sigma_2              

            z_1 = h * (a1_1 * eps_1 + a1_2 * eps_2)
            x_1 = x_1 + z_1
            
            z_2 = h * (a2_1 * eps_1 + a2_2 * eps_2)
            x_2 = x_0 + z_2
                        
            denoised_2 = model(x_2, sigma_2 * s_in, **extra_args) 
            
            eps_1 = (x_0 - denoised_1) / sigma_1
            eps_2 = (x_0 - denoised_2) / sigma_2  
            
            z_2 = h * (a2_1 * eps_1 + a2_2 * eps_2)
            x_2 = x_0 + z_2
            
            z_1 = h * (a1_1 * eps_1 + a1_2 * eps_2)
            x_1 = x_0 + z_1


        
        for full_iter in range(iter):
            denoised_1 = model(x_1, sigma_1 * s_in, **extra_args)
            denoised_2 = model(x_2, sigma_2 * s_in, **extra_args) 
            
            eps_1 = (x_1 - denoised_1) / sigma_1
            eps_2 = (x_2 - denoised_2) / sigma_2              

            z_1 = h * (a1_1 * eps_1 + a1_2 * eps_2)
            z_2 = h * (a2_1 * eps_1 + a2_2 * eps_2)
            
            x_1 = x_0 + z_1
            x_2 = x_0 + z_2
        
        
        #eps_1 = (x_0 - denoised_1) / sigma_1
        #eps_2 = (x_0 - denoised_2) / sigma_2              

        z_1 = h * (a1_1 * eps_1 + a1_2 * eps_2)
        z_2 = h * (a2_1 * eps_1 + a2_2 * eps_2)
        
        
        
        z_next = D_row[0]*z_1 + D_row[1]*z_2
        x_next = x_0 + z_next
        
        x_prev = x_0.clone()
        x = x_next
        
        z_1_prev = z_1
        z_2_prev = z_2
        #z_next_prev = h * (b1 * eps_1 + b2 * eps_2)
        z_next_prev = D_row[0]*z_1 + D_row[1]*z_2
        
        
        denoised_prev2 = denoised_prev
        denoised_prev = denoised
        
        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised_2})

    return denoised











def sample_rk_radau_iia_2s_lang_full(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
    noise_sampler2 = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+10000, sigma_min=0.0, sigma_max=1.0)
    
    A_tableau = torch.tensor([[1/4, 1/4 - 3**0.5 / 6], [1/4 + 3**0.5 / 6, 1/4],], dtype=x.dtype, device=x.device)
    B_row = torch.tensor([1/2, 1/2], dtype=x.dtype, device=x.device)
    A_inv = torch.linalg.inv(A_tableau)
    D_row = B_row @ A_inv
    
    denoised_prev, denoised_prev2 = [torch.zeros_like(x) for _ in range(2)]
    
    x_next_pred = None
    #for step in trange(len(sigmas)-1, disable=disable):
        
    step = 0
    pbar = tqdm(total=len(sigmas) - 1)  # Initialize progress bar with total steps
    step_increment = 1
    step_increment_prev = 1

    while step < len(sigmas) - 1:
        
        sigma, sigma_next = sigmas[step], sigmas[step+step_increment]
        
        sigma_up, sigma, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, eta, noise_mode)
        
        x_0 = x.clone()
    
        
        a1_1, a1_2 = 5/12, -1/12
        a2_1, a2_2 = 3/4, 1/4
        b1, b2 = 3/4, 1/4
        c1, c2 = 1/3, 1
        
        h = sigma_next - sigma
        
        sigma_1 = sigma + h * c1
        sigma_2 = sigma + h * c2
        
        if sigma_next == 0:
            return denoised_2
        
        if step == 0:
            denoised = model(x, sigma * s_in, **extra_args)
            if sigma_next == 0:
                return denoised
            eps = (x - denoised)/sigma
            
            eps_1 = eps * sigma_1
            eps_2 = eps * sigma_2 
            
            x_1 = x_0 + h * (a1_1 * eps_1 + a1_2 * eps_2)
            x_2 = x_0 + h * (a2_1 * eps_1 + a2_2 * eps_2)
            
        else:
            sigma_prev = sigmas[step-step_increment_prev]
            h_prev = sigma - sigma_prev
            
            w = h / h_prev

            #zi_1 = lagrange_interpolation([c1,c2,1], [z_1_prev, z_2_prev, z_next_prev], 1 + w*c1).squeeze(0) + x_prev - x_0
            
            #zi_1 = lagrange_interpolation([c1,c2,1], [z_1_prev, z_2_prev, z_next_prev], 1 + w*c1).squeeze(0) + x_prev - x_0
            #zi_2 = lagrange_interpolation([c1,c2,1], [z_1_prev, z_2_prev, z_next_prev], 1 + w*c2).squeeze(0) + x_prev - x_0
            
            zi_1 = lagrange_interpolation([c1,c2], [z_1_prev, z_2_prev], 1 + w*c1).squeeze(0) + x_prev - x_0
            zi_2 = lagrange_interpolation([c1,c2], [z_1_prev, z_2_prev], 1 + w*c2).squeeze(0) + x_prev - x_0
            
            x_1 = x_0 + zi_1
            x_2 = x_0 + zi_2
        
        for full_iter in range(sub_iter):
            denoised_1 = model(x_1, sigma_1 * s_in, **extra_args)
            
            eps_1 = (x_0 - denoised_1) / sigma_1
            eps_2 = (x_0 - denoised_1) / sigma_2              

            z_1 = h * (a1_1 * eps_1 + a1_2 * eps_2)
            x_1 = x_1 + z_1
            
            z_2 = h * (a2_1 * eps_1 + a2_2 * eps_2)
            x_2 = x_0 + z_2
                        
            denoised_2 = model(x_2, sigma_2 * s_in, **extra_args) 
            
            eps_1 = (x_0 - denoised_1) / sigma_1
            eps_2 = (x_0 - denoised_2) / sigma_2  
            
            z_2 = h * (a2_1 * eps_1 + a2_2 * eps_2)
            x_2 = x_0 + z_2
            
            z_1 = h * (a1_1 * eps_1 + a1_2 * eps_2)
            x_1 = x_0 + z_1


        
        for full_iter in range(iter):
            denoised_1 = model(x_1, sigma_1 * s_in, **extra_args)
            denoised_2 = model(x_2, sigma_2 * s_in, **extra_args) 
            
            eps_1 = (x_1 - denoised_1) / sigma_1
            eps_2 = (x_2 - denoised_2) / sigma_2              

            z_1 = h * (a1_1 * eps_1 + a1_2 * eps_2)
            z_2 = h * (a2_1 * eps_1 + a2_2 * eps_2)
            
            x_1 = x_0 + z_1
            x_2 = x_0 + z_2
        
        
        #eps_1 = (x_0 - denoised_1) / sigma_1
        #eps_2 = (x_0 - denoised_2) / sigma_2              

        z_1 = h * (a1_1 * eps_1 + a1_2 * eps_2)
        z_2 = h * (a2_1 * eps_1 + a2_2 * eps_2)
        
        
        
        z_next = D_row[0]*z_1 + D_row[1]*z_2
        x_next = x_0 + z_next
        
        x_prev = x_0.clone()
        x = x_next
        
        z_1_prev = z_1
        z_2_prev = z_2
        #z_next_prev = h * (b1 * eps_1 + b2 * eps_2)
        z_next_prev = D_row[0]*z_1 + D_row[1]*z_2
        
        
        denoised_prev2 = denoised_prev
        denoised_prev = denoised
        
        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised_2})
        pbar.update(1) 
        
        step_increment_prev = step_increment
        
        step += step_increment
    
    
    pbar.close()
    return denoised










def sample_rk_gausslang_full(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
    noise_sampler2 = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+10000, sigma_min=0.0, sigma_max=1.0)
    
    A_tableau = torch.tensor([[1/4, 1/4 - 3**0.5 / 6], [1/4 + 3**0.5 / 6, 1/4],], dtype=x.dtype, device=x.device)
    B_row = torch.tensor([1/2, 1/2], dtype=x.dtype, device=x.device)
    A_inv = torch.linalg.inv(A_tableau)
    D_row = B_row @ A_inv
    
    denoised_prev, denoised_prev2 = [torch.zeros_like(x) for _ in range(2)]
    
    x_next_pred = None
    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
        
        sigma_up, sigma, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, eta, noise_mode)
        
        x_0 = x.clone()
    
        
        a1_1, a1_2 = 1/4, 1/4 - 3**0.5 / 6
        a2_1, a2_2 = 1/4 + 3**0.5 / 6, 1/4
        b1, b2 = 1/2, 1/2
        c1, c2 = 1/2 - 3**0.5 / 6, 1/2 + 3**0.5 / 6
        
        h = sigma_next - sigma
        
        sigma_1 = sigma + h * c1
        sigma_2 = sigma + h * c2
        
        if sigma_next == 0:
            return denoised_2
        
        if step == 0:
            denoised = model(x, sigma * s_in, **extra_args)
            if sigma_next == 0:
                return denoised
            eps = (x - denoised)/sigma
            
            eps_1 = eps * sigma_1
            eps_2 = eps * sigma_2 
            
            x_1 = x_0 + h * (a1_1 * eps_1 + a1_2 * eps_2)
            x_2 = x_0 + h * (a2_1 * eps_1 + a2_2 * eps_2)
            
        else:
            sigma_prev = sigmas[step-1]
            h_prev = sigma - sigma_prev
            
            w = h / h_prev

            #zi_1 = lagrange_interpolation([c1,c2,1], [z_1_prev, z_2_prev, z_next_prev], 1 + w*c1).squeeze(0) + x_prev - x_0
            
            #zi_1 = lagrange_interpolation([c1,c2,1], [z_1_prev, z_2_prev, z_next_prev], 1 + w*c1).squeeze(0) + x_prev - x_0
            #zi_2 = lagrange_interpolation([c1,c2,1], [z_1_prev, z_2_prev, z_next_prev], 1 + w*c2).squeeze(0) + x_prev - x_0
            
            zi_1 = lagrange_interpolation([c1,c2], [z_1_prev, z_2_prev], 1 + w*c1).squeeze(0) + x_prev - x_0
            zi_2 = lagrange_interpolation([c1,c2], [z_1_prev, z_2_prev], 1 + w*c2).squeeze(0) + x_prev - x_0
            
            x_1 = x_0 + zi_1
            x_2 = x_0 + zi_2
        
        for full_iter in range(sub_iter):
            denoised_1 = model(x_1, sigma_1 * s_in, **extra_args)
            
            eps_1 = (x_1 - denoised_1) / sigma_1
            eps_2 = (x_2 - denoised_1) / sigma_2              

            z_1 = h * (a1_1 * eps_1 + a1_2 * eps_2)
            x_1 = x_1 + z_1
            
            z_2 = h * (a2_1 * eps_1 + a2_2 * eps_2)
            x_2 = x_0 + z_2
                        
            denoised_2 = model(x_2, sigma_2 * s_in, **extra_args) 
            
            eps_1 = (x_1 - denoised_1) / sigma_1
            eps_2 = (x_2 - denoised_2) / sigma_2  
            
            z_2 = h * (a2_1 * eps_1 + a2_2 * eps_2)
            x_2 = x_0 + z_2
            
            z_1 = h * (a1_1 * eps_1 + a1_2 * eps_2)
            x_1 = x_0 + z_1


        
        for full_iter in range(iter):
            denoised_1 = model(x_1, sigma_1 * s_in, **extra_args)
            denoised_2 = model(x_2, sigma_2 * s_in, **extra_args) 
            
            eps_1 = (x_1 - denoised_1) / sigma_1
            eps_2 = (x_2 - denoised_2) / sigma_2              

            z_1 = h * (a1_1 * eps_1 + a1_2 * eps_2)
            z_2 = h * (a2_1 * eps_1 + a2_2 * eps_2)
            
            x_1 = x_0 + z_1
            x_2 = x_0 + z_2
        
        
        #eps_1 = (x_0 - denoised_1) / sigma_1
        #eps_2 = (x_0 - denoised_2) / sigma_2              

        z_1 = h * (a1_1 * eps_1 + a1_2 * eps_2)
        z_2 = h * (a2_1 * eps_1 + a2_2 * eps_2)
        
        
        
        z_next = D_row[0]*z_1 + D_row[1]*z_2
        x_next = x_0 + z_next
        
        x_prev = x_0.clone()
        x = x_next
        
        z_1_prev = z_1
        z_2_prev = z_2
        #z_next_prev = h * (b1 * eps_1 + b2 * eps_2)
        z_next_prev = D_row[0]*z_1 + D_row[1]*z_2
        
        
        denoised_prev2 = denoised_prev
        denoised_prev = denoised
        
        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised_2})

    return denoised






def sample_rk_gausslang_3s_full_guide(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    if latent_guide is not None:
        y0 = model.inner_model.inner_model.process_latent_in(latent_guide['samples']).clone().to(x.device)

    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
    noise_sampler2 = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+10000, sigma_min=0.0, sigma_max=1.0)
    
    A_tableau = torch.tensor([[5/36, 2/9 - 15**0.5 / 15, 5/36 - 15**0.5 / 30],            [5/36 + 15**0.5 / 24, 2/9, 5/36 - 15**0.5 / 24],            [5/36 + 15**0.5 / 30, 2/9 + 15**0.5 / 15, 5/36],], dtype=x.dtype, device=x.device)
    B_row = torch.tensor([5/18, 4/9, 5/18], dtype=x.dtype, device=x.device)
    A_inv = torch.linalg.inv(A_tableau)
    D_row = B_row @ A_inv
    
    denoised_prev, denoised_prev2 = [torch.zeros_like(x) for _ in range(2)]
    
    x_next_pred = None
    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
        
        sigma_up, sigma, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, eta, noise_mode)
        
        x_0 = x.clone()
    
        
        a1_1, a1_2, a1_3 = 5/36, 2/9 - 15**0.5 / 15, 5/36 - 15**0.5 / 30
        a2_1, a2_2, a2_3 = 5/36 + 15**0.5 / 24, 2/9, 5/36 - 15**0.5 / 24
        a3_1, a3_2, a3_3 = 5/36 + 15**0.5 / 30, 2/9 + 15**0.5 / 15, 5/36
        b1, b2, b3 = 5/18, 4/9, 5/18
        c1, c2, c3 = 1/2 - 15**0.5 / 10, 1/2, 1/2 + 15**0.5 / 10
        
        h = sigma_next - sigma
        
        sigma_1 = sigma + h * c1
        sigma_2 = sigma + h * c2
        sigma_3 = sigma + h * c3
        
        if sigma_next == 0:
            return denoised_2
        
        if step == 0:
            denoised = model(x, sigma * s_in, **extra_args)
            if sigma_next == 0:
                return denoised
            eps = (x - denoised)/sigma
            
            eps_1 = eps * sigma_1
            eps_2 = eps * sigma_2 
            eps_3 = eps * sigma_3
            
            x_1 = x_0 + h * (a1_1 * eps_1 + a1_2 * eps_2 + a1_3 * eps_3)
            x_2 = x_0 + h * (a2_1 * eps_1 + a2_2 * eps_2 + a2_3 * eps_3)
            x_3 = x_0 + h * (a3_1 * eps_1 + a3_2 * eps_2 + a3_3 * eps_3)
            
        else:
            sigma_prev = sigmas[step-1]
            h_prev = sigma - sigma_prev
            
            w = h / h_prev

            #zi_1 = lagrange_interpolation([c1,c2,1], [z_1_prev, z_2_prev, z_next_prev], 1 + w*c1).squeeze(0) + x_prev - x_0
            
            #zi_1 = lagrange_interpolation([c1,c2,1], [z_1_prev, z_2_prev, z_next_prev], 1 + w*c1).squeeze(0) + x_prev - x_0
            #zi_2 = lagrange_interpolation([c1,c2,1], [z_1_prev, z_2_prev, z_next_prev], 1 + w*c2).squeeze(0) + x_prev - x_0
            
            zi_1 = lagrange_interpolation([c1,c2,c3], [z_1_prev, z_2_prev, z_3_prev], 1 + w*c1).squeeze(0) + x_prev - x_0
            zi_2 = lagrange_interpolation([c1,c2,c3], [z_1_prev, z_2_prev, z_3_prev], 1 + w*c2).squeeze(0) + x_prev - x_0
            zi_3 = lagrange_interpolation([c1,c2,c3], [z_1_prev, z_2_prev, z_3_prev], 1 + w*c3).squeeze(0) + x_prev - x_0
            
            x_1 = x_0 + zi_1
            x_2 = x_0 + zi_2
            x_3 = x_0 + zi_3
        
        for full_iter in range(sub_iter):
            denoised_1 = model(x_1, sigma_1 * s_in, **extra_args)
            
            eps_1 = (x_1 - denoised_1) / sigma_1
            eps_2 = (x_2 - denoised_1) / sigma_2     
            eps_3 = (x_3 - denoised_1) / sigma_3            

            z_1 = h * (a1_1 * eps_1 + a1_2 * eps_2 + a1_3 * eps_3)
            x_1 = x_1 + z_1
            
            z_2 = h * (a2_1 * eps_1 + a2_2 * eps_2 + a2_3 * eps_3)
            x_2 = x_0 + z_2
            
            z_3 = h * (a3_1 * eps_1 + a3_2 * eps_2 + a3_3 * eps_3)
            x_3 = x_0 + z_3
                        
            denoised_2 = model(x_2, sigma_2 * s_in, **extra_args) 
            
            eps_1 = (x_1 - denoised_1) / sigma_1
            eps_2 = (x_2 - denoised_2) / sigma_2     
            eps_3 = (x_3 - denoised_2) / sigma_3            

            z_1 = h * (a1_1 * eps_1 + a1_2 * eps_2 + a1_3 * eps_3)
            x_1 = x_1 + z_1
            
            z_2 = h * (a2_1 * eps_1 + a2_2 * eps_2 + a2_3 * eps_3)
            x_2 = x_0 + z_2
            
            z_3 = h * (a3_1 * eps_1 + a3_2 * eps_2 + a3_3 * eps_3)
            x_3 = x_0 + z_3
            
            denoised_3 = model(x_3, sigma_3 * s_in, **extra_args) 

            eps_1 = (x_1 - denoised_1) / sigma_1
            eps_2 = (x_2 - denoised_2) / sigma_2     
            eps_3 = (x_3 - denoised_3) / sigma_3            

            z_1 = h * (a1_1 * eps_1 + a1_2 * eps_2 + a1_3 * eps_3)
            x_1 = x_1 + z_1
            
            z_2 = h * (a2_1 * eps_1 + a2_2 * eps_2 + a2_3 * eps_3)
            x_2 = x_0 + z_2
            
            z_3 = h * (a3_1 * eps_1 + a3_2 * eps_2 + a3_3 * eps_3)
            x_3 = x_0 + z_3
        
        

            
        for full_iter in range(iter):
            
            if latent_guide is not None:
                yps_1 = (x_1 - y0) / sigma_1
                yps_2 = (x_2 - y0) / sigma_2   
                yps_3 = (x_3 - y0) / sigma_3 

            if extra_options_flag("fullguide_eps1", extra_options) and int(get_extra_options_kv("guidestop", "10000", extra_options)) < step:
                eps_1 = yps_1
            else:
                denoised_1 = model(x_1, sigma_1 * s_in, **extra_args)
                eps_1 = (x_1 - denoised_1) / sigma_1
            
            if extra_options_flag("fullguide_eps2", extra_options) and int(get_extra_options_kv("guidestop", "10000", extra_options)) < step:
                eps_2 = yps_2
            else:
                denoised_2 = model(x_2, sigma_2 * s_in, **extra_args) 
                eps_2 = (x_2 - denoised_2) / sigma_2   
                
            if extra_options_flag("fullguide_eps3", extra_options) and int(get_extra_options_kv("guidestop", "10000", extra_options)) < step:
                eps_3 = yps_3
            else:
                denoised_3 = model(x_3, sigma_3 * s_in, **extra_args) 
                eps_3 = (x_3 - denoised_3) / sigma_3     

            z_1 = h * (a1_1 * eps_1 + a1_2 * eps_2 + a1_3 * eps_3)
            x_1 = x_1 + z_1
            
            z_2 = h * (a2_1 * eps_1 + a2_2 * eps_2 + a2_3 * eps_3)
            x_2 = x_0 + z_2
            
            z_3 = h * (a3_1 * eps_1 + a3_2 * eps_2 + a3_3 * eps_3)
            x_3 = x_0 + z_3
        
        
        #eps_1 = (x_0 - denoised_1) / sigma_1
        #eps_2 = (x_0 - denoised_2) / sigma_2        
    
        if latent_guide is not None:
            yps_1 = (x_1 - y0) / sigma_1
            yps_2 = (x_2 - y0) / sigma_2   
            yps_3 = (x_3 - y0) / sigma_3 

        if int(get_extra_options_kv("guidestop", "10000", extra_options)) < step:
            if extra_options_flag("postguide_eps1", extra_options):
                eps_1 = yps_1
            if extra_options_flag("postguide_eps2", extra_options):
                eps_2 = yps_2
            if extra_options_flag("postguide_eps3", extra_options):
                eps_3 = yps_3

        z_1 = h * (a1_1 * eps_1 + a1_2 * eps_2 + a1_3 * eps_3)
        z_2 = h * (a2_1 * eps_1 + a2_2 * eps_2 + a2_3 * eps_3)
        z_3 = h * (a3_1 * eps_1 + a3_2 * eps_2 + a3_3 * eps_3)
        
        z_next = D_row[0]*z_1 + D_row[1]*z_2 + D_row[2]*z_3
        #z_next = h * (b1 * eps_1 + b2 * eps_2 + b3 * eps_3)
        x_next = x_0 + z_next
        
        x_prev = x_0.clone()
        x = x_next
        
        z_next_prev = D_row[0]*z_1 + D_row[1]*z_2 + D_row[2]*z_3
        
        
        if int(get_extra_options_kv("guidestop", "10000", extra_options)) < step:
            if latent_guide is not None:
                yps_1 = (x_1 - y0) / sigma_1
                yps_2 = (x_2 - y0) / sigma_2   
                yps_3 = (x_3 - y0) / sigma_3 
            if extra_options_flag("preguide_eps1", extra_options):
                eps_1 = yps_1
            if extra_options_flag("preguide_eps2", extra_options):
                eps_2 = yps_2
            if extra_options_flag("preguide_eps3", extra_options):
                eps_3 = yps_3

        z_1 = h * (a1_1 * eps_1 + a1_2 * eps_2 + a1_3 * eps_3)
        z_2 = h * (a2_1 * eps_1 + a2_2 * eps_2 + a2_3 * eps_3)
        z_3 = h * (a3_1 * eps_1 + a3_2 * eps_2 + a3_3 * eps_3)
        
        z_1_prev = z_1
        z_2_prev = z_2
        z_3_prev = z_3
        #z_next_prev = h * (b1 * eps_1 + b2 * eps_2)
        #z_next_prev = h * (b1 * eps_1 + b2 * eps_2 + b3 * eps_3)
        
        
        
        denoised_prev2 = denoised_prev
        denoised_prev = denoised
        
        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised_2})

    return denoised





def sample_rk_radau_iia_alt_lang_3s_full(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
    noise_sampler2 = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+10000, sigma_min=0.0, sigma_max=1.0)
    
    A_tableau = torch.tensor([[(88 - 7*6**0.5) / 360, (296 - 169*6**0.5) / 1800, (-2 + 3 * 6**0.5) / 225],            [(296 + 169*6**0.5) / 1800, (88 + 7*6**0.5) / 360, (-2 - 3*6**0.5) / 225],            [(16 - 6**0.5) / 36, (16 + 6**0.5) / 36, 1/9],], dtype=x.dtype, device=x.device)
    B_row = torch.tensor([(16 - 6**0.5) / 36,              (16 + 6**0.5) / 36,              1/9], dtype=x.dtype, device=x.device)
    A_inv = torch.linalg.inv(A_tableau)
    D_row = B_row @ A_inv
    
    denoised_prev, denoised_prev2 = [torch.zeros_like(x) for _ in range(2)]
    
    x_next_pred = None
    #for step in trange(len(sigmas)-1, disable=disable):
        
    step = 0
    pbar = tqdm(total=len(sigmas) - 1)  # Initialize progress bar with total steps
    step_increment = 1
    step_increment_prev = 1

    while step < len(sigmas) - 1:
        
        sigma, sigma_next = sigmas[step], sigmas[step+step_increment]
        #sigma, sigma_next = sigmas[step], sigmas[step+1]
        
        sigma_up, sigma, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, eta, noise_mode)
        
        x_0 = x.clone()
    
        
        a1_1, a1_2, a1_3 = (88 - 7*6**0.5) / 360, (296 - 169*6**0.5) / 1800, (-2 + 3 * 6**0.5) / 225
        a2_1, a2_2, a2_3 = (296 + 169*6**0.5) / 1800, (88 + 7*6**0.5) / 360, (-2 - 3*6**0.5) / 225
        a3_1, a3_2, a3_3 = (16 - 6**0.5) / 36, (16 + 6**0.5) / 36, 1/9
        b1, b2, b3 = (16 - 6**0.5) / 36,              (16 + 6**0.5) / 36,              1/9
        c1, c2, c3 = (4 - 6**0.5) / 10,          (4 + 6**0.5) / 10,          1.
        
        h = sigma_down - sigma
        
        sigma_1 = sigma + h * c1
        sigma_2 = sigma + h * c2
        sigma_3 = sigma + h * c3
        
        if sigma_next == 0:
            return denoised_2
        
        if step == 0:
            denoised = model(x, sigma * s_in, **extra_args)
            if sigma_next == 0:
                return denoised
            eps = (x - denoised)/sigma
            
            eps_1 = eps * sigma_1
            eps_2 = eps * sigma_2 
            eps_3 = eps * sigma_3
            
            x_1 = x_0 + h * (a1_1 * eps_1 + a1_2 * eps_2 + a1_3 * eps_3)
            x_2 = x_0 + h * (a2_1 * eps_1 + a2_2 * eps_2 + a2_3 * eps_3)
            x_3 = x_0 + h * (a3_1 * eps_1 + a3_2 * eps_2 + a3_3 * eps_3)
            
        else:
            sigma_prev = sigmas[step-1]
            h_prev = sigma - sigma_prev
            
            w = h / h_prev

            zi_1 = lagrange_interpolation([c1,c2,c3], [z_1_prev, z_2_prev, z_3_prev], 1 + w*c1).squeeze(0) + x_prev - x_0
            zi_2 = lagrange_interpolation([c1,c2,c3], [z_1_prev, z_2_prev, z_3_prev], 1 + w*c2).squeeze(0) + x_prev - x_0
            zi_3 = lagrange_interpolation([c1,c2,c3], [z_1_prev, z_2_prev, z_3_prev], 1 + w*c3).squeeze(0) + x_prev - x_0
            
            x_1 = x_0 + zi_1
            x_2 = x_0 + zi_2
            x_3 = x_0 + zi_3
        
        for full_iter in range(sub_iter):
            denoised_1 = model(x_1, sigma_1 * s_in, **extra_args)
            
            eps_1 = (x_1 - denoised_1) / sigma_1
            eps_2 = (x_2 - denoised_1) / sigma_2     
            eps_3 = (x_3 - denoised_1) / sigma_3            

            z_1 = h * (a1_1 * eps_1 + a1_2 * eps_2 + a1_3 * eps_3)
            z_2 = h * (a2_1 * eps_1 + a2_2 * eps_2 + a2_3 * eps_3)
            z_3 = h * (a3_1 * eps_1 + a3_2 * eps_2 + a3_3 * eps_3)
            
            x_1 = x_0 + z_1
            x_2 = x_0 + z_2
            x_3 = x_0 + z_3
            
            denoised_2 = model(x_2, sigma_2 * s_in, **extra_args) 
            
            eps_1 = (x_1 - denoised_1) / sigma_1
            eps_2 = (x_2 - denoised_2) / sigma_2     
            eps_3 = (x_3 - denoised_2) / sigma_3            

            z_1 = h * (a1_1 * eps_1 + a1_2 * eps_2 + a1_3 * eps_3)
            z_2 = h * (a2_1 * eps_1 + a2_2 * eps_2 + a2_3 * eps_3)
            z_3 = h * (a3_1 * eps_1 + a3_2 * eps_2 + a3_3 * eps_3)
            
            x_1 = x_0 + z_1
            x_2 = x_0 + z_2
            x_3 = x_0 + z_3
            
            denoised_3 = model(x_3, sigma_3 * s_in, **extra_args) 

            eps_1 = (x_1 - denoised_1) / sigma_1
            eps_2 = (x_2 - denoised_2) / sigma_2     
            eps_3 = (x_3 - denoised_3) / sigma_3            

            z_1 = h * (a1_1 * eps_1 + a1_2 * eps_2 + a1_3 * eps_3)
            z_2 = h * (a2_1 * eps_1 + a2_2 * eps_2 + a2_3 * eps_3)
            z_3 = h * (a3_1 * eps_1 + a3_2 * eps_2 + a3_3 * eps_3)
            
            x_1 = x_0 + z_1
            x_2 = x_0 + z_2
            x_3 = x_0 + z_3
            
            if full_iter > 0 and extra_options_flag("rtol", extra_options):
                theta_k_1 = (torch.log(torch.norm(z_1_prev) + 1e-8) - torch.log(torch.norm(z_1) + 1e-8)).abs().item()
                theta_k_2 = (torch.log(torch.norm(z_2_prev) + 1e-8) - torch.log(torch.norm(z_2) + 1e-8)).abs().item()
                theta_k_3 = (torch.log(torch.norm(z_3_prev) + 1e-8) - torch.log(torch.norm(z_3) + 1e-8)).abs().item()
                
                print("iter:", full_iter, "theta:", theta_k_1, theta_k_2, theta_k_3)
                
                if rtol > 0:
                    if max(theta_k_1, theta_k_2, theta_k_3) < rtol:
                        break
                    #else:
                    #    full_iter = 0
                    
            elif full_iter > 0 and extra_options_flag("ztol", extra_options):
                theta_k_1 = torch.norm(z_1) / torch.norm(z_1_prev)
                theta_k_2 = torch.norm(z_2) / torch.norm(z_2_prev)
                theta_k_3 = torch.norm(z_3) / torch.norm(z_3_prev)
                
                eta_k_1 = theta_k_1 / (1 - theta_k_1)
                eta_k_2 = theta_k_2 / (1 - theta_k_2)
                eta_k_3 = theta_k_3 / (1 - theta_k_3)
                
                eta_k_delta_z_1 = (eta_k_1 * torch.norm(z_1)).item()
                eta_k_delta_z_2 = (eta_k_2 * torch.norm(z_2)).item()
                eta_k_delta_z_3 = (eta_k_3 * torch.norm(z_3)).item()
                
                print("iter:", full_iter, "theta:", eta_k_delta_z_1, eta_k_delta_z_2, eta_k_delta_z_3)
                
                if eta_k_delta_z_1 <= kappa_tol and eta_k_delta_z_2 <= kappa_tol and eta_k_delta_z_3 <= kappa_tol:
                    break
                
            
            
            
        h_new1 = h_new2 = h_new3 = h
        
        z_1, z_2, z_3 = None, None, None
        
        rtol = float(get_extra_options_kv("rtol", "0", extra_options))
        
        ztol = float(get_extra_options_kv("ztol", "0", extra_options))
        kappa = float(get_extra_options_kv("kappa", "1e-2", extra_options))
        #max = float(get_extra_options_kv("rtol", "0", extra_options))
        
        kappa_tol = ztol * kappa
        
        
        full_iter = 0
        while full_iter < iter:
            #for full_iter in range(iter):
            
            z_1_prev = z_1
            z_2_prev = z_2
            z_3_prev = z_3
            
            sub_sigma_next1 = sigma_1
            sub_sigma_up1, sub_sigma1, sub_sigma_down1, sub_alpha_ratio1 = get_res4lyf_step_with_model(model, sigma, sub_sigma_next1, eta_var, noise_mode)
            h_new1 = h * (sub_sigma_down1 - sigma) / (sub_sigma_next1 - sigma) 
                        
            sub_sigma_next2 = sigma_2
            sub_sigma_up2, sub_sigma2, sub_sigma_down2, sub_alpha_ratio2 = get_res4lyf_step_with_model(model, sigma, sub_sigma_next2, eta_var, noise_mode)
            h_new2 = h * (sub_sigma_down2 - sigma) / (sub_sigma_next2 - sigma) 
            
            sub_sigma_next3 = sigma_3
            sub_sigma_up3, sub_sigma3, sub_sigma_down3, sub_alpha_ratio3 = get_res4lyf_step_with_model(model, sigma, sub_sigma_next3, eta_var, noise_mode)
            h_new3 = h * (sub_sigma_down3 - sigma) / (sub_sigma_next3 - sigma) 
            
            denoised_1 = model(x_1, sigma_1 * s_in, **extra_args)
            denoised_2 = model(x_2, sigma_2 * s_in, **extra_args) 
            denoised_3 = model(x_3, sigma_3 * s_in, **extra_args) 
            
            eps_1 = (x_1 - denoised_1) / sigma_1
            eps_2 = (x_2 - denoised_2) / sigma_2     
            eps_3 = (x_3 - denoised_3) / sigma_3            

            z_1 = h_new1 * (a1_1 * eps_1 + a1_2 * eps_2 + a1_3 * eps_3)
            z_2 = h_new2 * (a2_1 * eps_1 + a2_2 * eps_2 + a2_3 * eps_3)
            z_3 = h_new3 * (a3_1 * eps_1 + a3_2 * eps_2 + a3_3 * eps_3)
            
            x_1 = x_0 + z_1
            x_2 = x_0 + z_2
            x_3 = x_0 + z_3
            
            noise1 = noise_sampler2(sigma=sigma, sigma_next=sigma_1)
            noise2 = noise_sampler2(sigma=sigma, sigma_next=sigma_2)
            noise3 = noise_sampler2(sigma=sigma, sigma_next=sigma_3)
            
            x_1 = sub_alpha_ratio1 * x_1 + sub_sigma_up1 * noise1
            x_2 = sub_alpha_ratio2 * x_2 + sub_sigma_up2 * noise2
            x_3 = sub_alpha_ratio3 * x_3 + sub_sigma_up3 * noise3
            
            
            if full_iter > 0 and extra_options_flag("rtol", extra_options):
                theta_k_1 = (torch.log(torch.norm(z_1_prev) + 1e-8) - torch.log(torch.norm(z_1) + 1e-8)).abs().item()
                theta_k_2 = (torch.log(torch.norm(z_2_prev) + 1e-8) - torch.log(torch.norm(z_2) + 1e-8)).abs().item()
                theta_k_3 = (torch.log(torch.norm(z_3_prev) + 1e-8) - torch.log(torch.norm(z_3) + 1e-8)).abs().item()
                
                print("iter:", full_iter, "theta:", theta_k_1, theta_k_2, theta_k_3)
                
                if rtol > 0:
                    if max(theta_k_1, theta_k_2, theta_k_3) < rtol:
                        if full_iter == 1  and extra_options_flag("use_autostep", extra_options):
                            print("step increment:", step_increment)
                            step_increment = 2 #+= 1
                        break
                    #else:
                    #    full_iter = 0
                    
            elif full_iter > 0 and extra_options_flag("ztol", extra_options):
                theta_k_1 = torch.norm(z_1) / torch.norm(z_1_prev)
                theta_k_2 = torch.norm(z_2) / torch.norm(z_2_prev)
                theta_k_3 = torch.norm(z_3) / torch.norm(z_3_prev)
                
                eta_k_1 = theta_k_1 / (1 - theta_k_1)
                eta_k_2 = theta_k_2 / (1 - theta_k_2)
                eta_k_3 = theta_k_3 / (1 - theta_k_3)
                
                eta_k_delta_z_1 = (eta_k_1 * torch.norm(z_1)).item()
                eta_k_delta_z_2 = (eta_k_2 * torch.norm(z_2)).item()
                eta_k_delta_z_3 = (eta_k_3 * torch.norm(z_3)).item()
                
                print("iter:", full_iter, "theta:", eta_k_delta_z_1, eta_k_delta_z_2, eta_k_delta_z_3)
                
                if eta_k_delta_z_1 <= kappa_tol and eta_k_delta_z_2 <= kappa_tol and eta_k_delta_z_3 <= kappa_tol:
                    if full_iter == 1 and extra_options_flag("use_autostep", extra_options):
                        print("step increment:", step_increment)
                        step_increment += 1
                    break
                
            
            full_iter += 1



        #z_1 = h_new1 * (a1_1 * eps_1 + a1_2 * eps_2 + a1_3 * eps_3)
        #z_2 = h_new2 * (a2_1 * eps_1 + a2_2 * eps_2 + a2_3 * eps_3)
        #z_3 = h_new3 * (a3_1 * eps_1 + a3_2 * eps_2 + a3_3 * eps_3)
        
        z_1 = h * (a1_1 * eps_1 + a1_2 * eps_2 + a1_3 * eps_3)
        z_2 = h * (a2_1 * eps_1 + a2_2 * eps_2 + a2_3 * eps_3)
        z_3 = h * (a3_1 * eps_1 + a3_2 * eps_2 + a3_3 * eps_3)
        
        z_next = D_row[0]*z_1 + D_row[1]*z_2 + D_row[2]*z_3
        #z_next = h * (b1 * eps_1 + b2 * eps_2 + b3 * eps_3)
        x_next = x_0 + z_next
        
        x_prev = x_0.clone()
        x = x_next
        
        z_next_prev = D_row[0]*z_1 + D_row[1]*z_2 + D_row[2]*z_3
        
        z_1 = h * (a1_1 * eps_1 + a1_2 * eps_2 + a1_3 * eps_3)
        z_2 = h * (a2_1 * eps_1 + a2_2 * eps_2 + a2_3 * eps_3)
        z_3 = h * (a3_1 * eps_1 + a3_2 * eps_2 + a3_3 * eps_3)
        
        z_1_prev = z_1
        z_2_prev = z_2
        z_3_prev = z_3
        #z_next_prev = h * (b1 * eps_1 + b2 * eps_2)
        #z_next_prev = h * (b1 * eps_1 + b2 * eps_2 + b3 * eps_3)
        
        noise = noise_sampler(sigma=sigma, sigma_next=sigma_next)
        x = alpha_ratio * x + sigma_up * noise
        
        denoised_prev2 = denoised_prev
        denoised_prev = denoised
        
        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised_2})
        pbar.update(step_increment) 
        
        step_increment_prev = step_increment
        
        if step_increment + step >= len(sigmas-4):
            print("patch step")
            step_increment = 1 #len(sigmas-1) - step - 1
        
        step += step_increment
        
    pbar.close()
    return denoised






def sample_rk_implicit(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
    noise_sampler2 = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+10000, sigma_min=0.0, sigma_max=1.0)
    
    rk_type = get_extra_options_kv("rk_type", "radau_iia_3s_alt", extra_options)
    diverging_tol = float(get_extra_options_kv("diverging_tol", "10000", extra_options))
    RST_STEP_MAX = int(get_extra_options_kv("RST_STEP_MAX", "2", extra_options))
    ITER_MIN = int(get_extra_options_kv("ITER_MIN", "0", extra_options))
    
    A_tableau = torch.tensor(rk_coeff[rk_type][0],    dtype=x.dtype, device=x.device)
    B_row     = torch.tensor(rk_coeff[rk_type][1][0], dtype=x.dtype, device=x.device)
    C_nodes   = torch.tensor(rk_coeff[rk_type][2],    dtype=x.dtype, device=x.device)
    
    A_inv = torch.linalg.inv(A_tableau)
    D_row = B_row @ A_inv
    
    RK_rows   = len(C_nodes)
    
    x_, eps_, data_, z_, z_prev_  = [torch.zeros((RK_rows, *x.shape), dtype=x.dtype, device=x.device) for _ in range(5)]
    
    RST_CONT = False
    RST_STEP = 0 #False
    step = 0
    step_increment_prev = step_increment = 1
    pbar = tqdm(total=len(sigmas) - 1, dynamic_ncols=True)
    while step < len(sigmas) - 1:
        
        sigma, sigma_next = sigmas[step], sigmas[step+step_increment]        
        sigma_up, sigma, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, eta, noise_mode)
        
        x_0 = x.clone()

        h = sigma_down - sigma
        s_ = (sigma + h * C_nodes)
        
        m_sigma_up, m_sigma, m_sigma_down, m_alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_down, eta_var, noise_mode)
        h_m = m_sigma_down - sigma
        s_m_ = (sigma + h_m * C_nodes)
        

        
        if sigma_next == 0:
            denoised = model(x_0, sigma * s_in, **extra_args)
            x = denoised
            break
        
        elif step == 0:
            denoised = model(x_0, sigma * s_in, **extra_args)
            eps = (x_0 - denoised)/sigma
            
            eps_ = eps.unsqueeze(0).repeat(RK_rows, *[1] * eps.ndim)
            eps_ = torch.einsum("i,i...->i...", s_, eps_)
            
            z_ = torch.einsum('ij, j... -> i...', A_tableau, eps_)
            x_ = x_0 + h * z_
            
        elif extra_options_flag("alt_init", extra_options):
            denoised = model(x_0, sigma * s_in, **extra_args)
            eps = (x_0 - denoised)/sigma
            eps_ = eps.unsqueeze(0).repeat(RK_rows, *[1] * eps.ndim)
            eps_ = torch.einsum("i,i...->i...", s_, eps_)
            z_ = torch.einsum('ij, j... -> i...', A_tableau, eps_)
            x_ = x_0 + h * z_
            
        else:
            h_prev = sigma - sigmas[step-1]
            w = h / h_prev

            for r in range(RK_rows):
                z_[r] = lagrange_interpolation(C_nodes, h_prev * z_prev_step_, 1 + w*C_nodes[r]).squeeze(0) + x_prev - x_0
            
            x_ = x_0 + z_
            
            if C_nodes[0] == 0 and not extra_options_flag("disable_lefthand", extra_options):
                x_[0] = x_0
                
        if extra_options_flag("newton_iter", extra_options):
            for r in range (RK_rows):
                eps_[r] = (x_[r] - denoised) / s_[r]
            
            eps_anchor = eps_[0].clone()
            for i in range(100):
                if extra_options_flag("newton_iter_anchor", extra_options):
                    eps_[0] = eps_anchor.clone()
                #for r in range (RK_rows):
                #eps_ = torch.einsum("i,i...->i...", s_, eps_)
                z_ = torch.einsum('ij, j... -> i...', A_tableau, eps_)
                x_ = x_0 + h * z_
                #eps_ = (x_ - denoised.view(-1, *([1] * (x_.ndim - 1)))) / s_.view(-1, *([1] * (x_.ndim - 1)))
                for r in range (RK_rows):
                    eps_[r] = (x_[r] - denoised) / s_[r]

        z_D = torch.einsum("i,  i... ->  ...", D_row, z_)
        x_next = x_0 + h * z_D
        for diag_iter in range(sub_iter):
            x_prev = x_next
            z_prev_ = z_.clone()

            for r in range(RK_rows):
                data_[r] = model(x_[r], s_[r] * s_in, **extra_args)
            
                eps_ = (x_ - data_) / s_.view(-1, *([1] * (x_.ndim - 1)))
                z_ = torch.einsum('ij, j... -> i...', A_tableau, eps_)
                x_ = x_0 + h * z_
            
            z_D = torch.einsum("i,  i... ->  ...", D_row, z_)
            x_next = x_0 + h * z_D
            
            stop, step_increment, diverging = check_convergence(diag_iter, x_0, z_, z_prev_, x_next, x_prev, h, step_increment, D_row, extra_options)
            if diverging > diverging_tol: #RST_STEP == False:
                if RST_STEP >= RST_STEP_MAX: # 0:
                    print("skipping restart...")
                else:
                    sigma_mid = (sigma + sigma_next) / 2
                    sigmas = torch.cat([sigmas[:step+1], sigma_mid.unsqueeze(0), sigmas[step+1:]])
                    print("restarting due to diverging:", diverging, sigma.item(), sigma_mid.item(), sigma_next.item())
                    pbar.total = len(sigmas) - 1
                    pbar.refresh()
                    RST_STEP += 1 #= RST_STEP_MAX #True
                    RST_CONT = True
                    break
            if stop and full_iter >= ITER_MIN:
                break
            
            
            if C_nodes[-1] == 1 and not extra_options_flag("disable_righthand", extra_options):
                #z_D = torch.einsum("i,  i... ->  ...", D_row, z_)
                x_[-1]   = x_next # x_0 +     h * z_D
        if RST_CONT == True:
            RST_CONT = False
            continue
                    
        z_D = torch.einsum("i,  i... ->  ...", D_row, z_)
        
        x_prev2 = x_0
        x_prev = x_0 + h * z_D
        
        z_prev2_, z_prev3_ = None, None
        z_prev_ = z_.clone()
        
        for full_iter in range(iter):

            sub_sigma_next, sub_sigma_up, sub_sigma, sub_sigma_down, sub_alpha_ratio = [torch.zeros_like(s_) for _ in range(5)]
            for r in range(len(s_)):
                sub_sigma_next[r] = s_[r]
                sub_sigma_up[r], sub_sigma[r], sub_sigma_down[r], sub_alpha_ratio[r] = get_res4lyf_step_with_model(model, s_[r], s_[r], eta_var, noise_mode)
                print(sub_sigma_up[r], sub_sigma[r], sub_sigma_down[r], sub_alpha_ratio[r])
            
            if extra_options_flag("newton_pre_iter", extra_options) and full_iter > 0:
                eps_anchor = eps_[0].clone()
                for i in range(100):
                    for r in range (RK_rows):
                        eps_[r] = (x_[r] - data_[r]) / s_[r]
                
                    if extra_options_flag("newton_pre_iter_anchor", extra_options):
                        eps_[0] = eps_anchor.clone()
                        
                    z_ = torch.einsum('ij, j... -> i...', A_tableau, eps_)
                    x_ = x_0 + h * z_
            
            for r in range(RK_rows):
                data_[r] = model(x_[r], s_[r] * s_in, **extra_args)
            
            if extra_options_flag("newton_post_iter", extra_options) and full_iter > 0:
                eps_anchor = eps_[0].clone()
                for i in range(100):
                    for r in range (RK_rows):
                        eps_[r] = (x_[r] - data_[r]) / s_[r]
                
                    if extra_options_flag("newton_post_iter_anchor", extra_options):
                        eps_[0] = eps_anchor.clone()
                        
                    z_ = torch.einsum('ij, j... -> i...', A_tableau, eps_)
                    x_ = x_0 + h * z_
            
            if extra_options_flag("eps_against_x_0", extra_options):
                #eps_ = (x_0 - data_) / s_.view(-1, *([1] * (x_.ndim - 1)))
                eps_ = ((x_0 - data_) / sigma) #* s_.view(-1, *([1] * (x_.ndim - 1)))
                
                """elif extra_options_flag("substep_down", extra_options):
                    eps_down_ = (x_ - data_) / sub_sigma_down.view(-1, *([1] * (x_.ndim - 1)))
                    z_down_ = torch.einsum('ij, j... -> i...', A_tableau, eps_down_)"""
                #for r in range(RK_rows):
                #    eps_[r] *= 
            else:
                eps_ = (x_ - data_) / s_.view(-1, *([1] * (x_.ndim - 1)))
                
                
            z_ = torch.einsum('ij, j... -> i...', A_tableau, eps_)
            
            if extra_options_flag("substep_down", extra_options):
                
                eps_down_ = (x_ - data_) / sub_sigma_down.view(-1, *([1] * (x_.ndim - 1)))
                z_down_ = torch.einsum('ij, j... -> i...', A_tableau, eps_down_)
                #x_ = x_0 + h * z_down_
                eps_x_0_ = ((x_0 - data_) / sigma)
                eps_down_guess_ = ((x_ - data_) / sigma)
                z_x_0_ = torch.einsum('ij, j... -> i...', A_tableau, eps_x_0_)
                z_down_guess_ = torch.einsum('ij, j... -> i...', A_tableau, eps_down_guess_)
                
                for r in range(RK_rows):
                    h_new      = h * (sub_sigma_down[r] - sigma) / (sub_sigma_next[r] - sigma) 
                    #x_[r] = x_0 + h_new * z_down_guess_[r]
                    
                    #x_[r] = x_0 + h_new * z_down_[r]
                    
                    #x_[r] = x_0 + h * (sub_sigma_next[r] / sub_sigma_down[r]) * eps_[r]
                    x_[r] = x_0 + h * (1 - sub_sigma_down[r]) / (1 - sub_sigma_next[r]) * eps_[r]
                    
                    #x_[r] = x_0 + h_new * z_[r]
                    #x_[r] = x_[r] + (sub_sigma_down[r] - sub_sigma_next[r]) * eps_x_0_[r] #z_x_0_[r]
                    
                    #x_[r] = x_[r] + (sub_sigma_down[r] - s_[r]) * z_down_[r] #eps_[r]
                    noise = noise_sampler2(sigma=sigma, sigma_next=sigma_next)
                    x_[r] = sub_alpha_ratio[r] * x_[r] + sub_sigma_up[r] * noise
            else:
                x_ = x_0 + h * z_
            
            
            z_D = torch.einsum("i,  i... ->  ...", D_row, z_)
            x_next = x_0 + h * z_D
            
            stop, step_increment, diverging = check_convergence(full_iter, x_0, z_, z_prev_, z_prev2_, x_next, x_prev, x_prev2, h, step_increment, D_row, extra_options)
            if diverging > diverging_tol and full_iter >= ITER_MIN: #RST_STEP == False:
                if RST_STEP >= RST_STEP_MAX: # > 0:
                    print("skipping restart...")
                else:
                    sigma_mid = (sigma + sigma_next) / 2
                    sigmas = torch.cat([sigmas[:step+1], sigma_mid.unsqueeze(0), sigmas[step+1:]])
                    print("restarting due to diverging:", diverging, sigma.item(), sigma_mid.item(), sigma_next.item())
                    pbar.total = len(sigmas) - 1
                    pbar.refresh()
                    RST_STEP += 1 # = RST_STEP_MAX #True
                    RST_CONT = True
                    break
            if stop and full_iter >= ITER_MIN:
                break
            
            if C_nodes[-1] == 1 and not extra_options_flag("disable_righthand", extra_options):
                #z_D = torch.einsum("i,  i... ->  ...", D_row, z_)
                x_[-1]   = x_next #x_0 +     h * z_D
            
            DIFF_NEWT_START = int(get_extra_options_kv("DIFF_NEWT_START", "0", extra_options))
            
            
            if z_prev3_ is not None and extra_options_flag("diff_newt34", extra_options) and full_iter >= DIFF_NEWT_START+1:
                z_D_prev  = torch.einsum("i,  i... ->  ...", D_row, z_prev_)
                z_D_prev2 = torch.einsum("i,  i... ->  ...", D_row, z_prev2_)
                z_D_prev3 = torch.einsum("i,  i... ->  ...", D_row, z_prev3_)
                
                x_ = x_0 + h * ((1/8) * z_ + (3/8) * z_prev_ + (3/8) * z_prev2_ + (1/8) * z_prev3_)
                
                #z_D = ((1/8) * z_D + (3/8) * z_D_prev + (3/8) * z_D_prev2 + (1/8) * z_D_prev3)
                #x_next = x_0 + h * z_D
                
            if z_prev2_ is not None and extra_options_flag("diff_newt3", extra_options) and full_iter >= DIFF_NEWT_START+1:
                z_D_prev  = torch.einsum("i,  i... ->  ...", D_row, z_prev_)
                z_D_prev2 = torch.einsum("i,  i... ->  ...", D_row, z_prev2_)
                
                x_ = x_0 + h * ((1/6) * z_ + (2/3) * z_prev_ + (1/6) * z_prev2_)
                
                #z_D = ((1/6) * z_D + (2/3) * z_D_prev + (1/6) * z_D_prev2)
                #x_next = x_0 + h * z_D
                
            elif z_prev_ is not None and extra_options_flag("diff_newt", extra_options) and full_iter >= DIFF_NEWT_START:
                ci = [0, c2]
                φ = Phi(h, ci)
                
                a2_1 = c2 * φ(1,2)
                b2 = φ(2)/c2
                b1 = φ(1) - b2
                
                z_D_prev  = torch.einsum("i,  i... ->  ...", D_row, z_prev_)

                x_ = x_0 + h * (0.5 * z_ + 0.5 * z_prev_)
                
                #z_D = (0.5 * z_D + 0.5 * z_D_prev)
                #x_next = x_0 + h * z_D            
            
            
            
            x_prev2 = x_prev
            x_prev = x_next
            
            if z_prev2_ is not None:
                z_prev3_ = z_prev2_.clone()
            z_prev2_ = z_prev_.clone()
            z_prev_ = z_.clone()
            
            
        if RST_CONT == True:
            RST_CONT = False
            continue

        #z_D = torch.einsum("i,  i... ->  ...", D_row, z_)
        
        if extra_options_flag("x_next_eps_vs_x_0", extra_options):
            #eps_ = (x_0 - data_) / s_.view(-1, *([1] * (x_.ndim - 1)))
            eps_ = ((x_0 - data_) / sigma) #* s_.view(-1, *([1] * (x_.ndim - 1)))
            z_tmp_ = torch.einsum('ij, j... -> i...', A_tableau, eps_)
            z_D = torch.einsum("i,  i... ->  ...", D_row, z_tmp_)

        
        
        
        x_next   = x_0 +     h * z_D
        denoised = x_0 - sigma * z_D
        
        x_prev  = x_0.clone()
        z_prev_step_ = z_.clone()


        noise = noise_sampler(sigma=sigma, sigma_next=sigma_next)
        x_next = alpha_ratio * x_next + sigma_up * noise

        if callback is not None:
            callback({'x': x_next, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised})
        pbar.update(step_increment) 
        
        step_increment_prev = step_increment
        
        if step_increment + step >= len(sigmas-4):
            print("patch step")
            step_increment = 1 #len(sigmas-1) - step - 1
        
        x = x_next
        step += step_increment
        if RST_STEP > 0:
            RST_STEP -= 1
        
    pbar.close()
    return x


def check_convergence(full_iter, x_0, z_, z_prev_, z_prev2_, x_next, x_prev, x_prev2, h, step_increment, D_row, extra_options):
    RK_rows = z_.shape[0]
    R_tol  = float(get_extra_options_kv("R_tol",  "0",    extra_options))
    A_tol  = float(get_extra_options_kv("A_tol",  "1e-8",    extra_options))
    
    Rtol  = float(get_extra_options_kv("Rtol",  "0",    extra_options))
    rtol  = float(get_extra_options_kv("rtol",  "0",    extra_options))
    xtol  = float(get_extra_options_kv("xtol",  "0",    extra_options))
    x0tol  = float(get_extra_options_kv("x0tol",  "0",    extra_options))
    ztol  = float(get_extra_options_kv("ztol",  "0",    extra_options))
    kappa = float(get_extra_options_kv("kappa", "1e-2", extra_options))
    kappa_tol = ztol * kappa
    z_D = torch.einsum("i,  i... ->  ...", D_row, z_)
    z_D_prev = torch.einsum("i,  i... ->  ...", D_row, z_prev_)
    
    theta_k_D = torch.norm(z_D) / torch.norm(z_D_prev)
    eta_k_D = theta_k_D / (1 - theta_k_D)
    eta_k_delta_z_D = eta_k_D * torch.norm(z_D)
    diverging = 0
    
    if x_prev2 is not None and extra_options_flag("x_prev", extra_options):
        z_D = x_next - x_prev
        z_D_prev = x_prev - x_prev2
            
        theta_k_D = torch.norm(z_D) / torch.norm(z_D_prev)
        eta_k_D = theta_k_D / (1 - theta_k_D)
        eta_k_delta_z_D = eta_k_D * torch.norm(z_D)

        print("realD:", full_iter, theta_k_D.item(), eta_k_D.item(), eta_k_delta_z_D.item())
        #if eta_k_delta_z_D <= kappa_tol:
        #    return True, step_increment, theta_k_D
    
    if z_prev2_ is not None and extra_options_flag("z_prev", extra_options):
        z_D_prev2 = torch.einsum("i,  i... ->  ...", D_row, z_prev2_)
        
        delta_z_D2 = z_D_prev - z_D_prev2
        delta_z_D  = z_D - z_D_prev
        
        theta_k_D = torch.norm(delta_z_D) / torch.norm(delta_z_D2)
        eta_k_D = theta_k_D / (1 - theta_k_D)
        eta_k_delta_z_D = eta_k_D * torch.norm(delta_z_D)
        
        print("realZdeltaD:", full_iter, theta_k_D.item(), eta_k_D.item(), eta_k_delta_z_D.item())
        #if eta_k_delta_z_D <= kappa_tol:
        #    return True, step_increment, theta_k_D

    
    if full_iter > 0:
        print("norms:", torch.norm(z_D).item(), torch.norm(z_D_prev).item())
        print("blah:", theta_k_D.item(), eta_k_D.item(), eta_k_delta_z_D.item())
        diverging = theta_k_D
    
    
    if full_iter > 0 and extra_options_flag("R_tol", extra_options):
        theta_k = torch.norm(z_D) / (torch.norm(x_0) + A_tol)

        print(f"rdtol iter: {full_iter} theta: {f'{theta_k:.4f}'}")

        if R_tol > 0 and theta_k < R_tol:
            if full_iter == 1 and extra_options_flag("use_autostep", extra_options):
                print("step increment:", step_increment)
                step_increment = 2
            return True, step_increment, diverging
    
    
    if full_iter > 0 and extra_options_flag("rdtol", extra_options):
        theta_k = (torch.log(torch.norm(z_D_prev) )    -     torch.log(torch.norm(z_D)).abs().item())

        print(f"rdtol iter: {full_iter} theta: {f'{theta_k:.4f}'}")

        if rtol > 0 and theta_k < rtol:
            if full_iter == 1 and extra_options_flag("use_autostep", extra_options):
                print("step increment:", step_increment)
                step_increment = 2
            return True, step_increment, diverging
        
    if full_iter > 0 and extra_options_flag("x0tol", extra_options):
        diff_prev = (torch.norm(x_next - x_prev).item())
        diff_full = (torch.norm(x_next - x_0).item())
        theta_k = diff_prev / diff_full

        print(f"xtol iter: {full_iter} theta: {f'{theta_k:.4f}'}")

        if x0tol > 0 and theta_k < x0tol:
            if full_iter == 1 and extra_options_flag("use_autostep", extra_options):
                print("step increment:", step_increment)
                step_increment = 2
            return True, step_increment, diverging
        
        
    if full_iter > 0 and extra_options_flag("xtol", extra_options):
        theta_k = (torch.norm(x_next - x_prev).item())

        print(f"xtol iter: {full_iter} theta: {f'{theta_k:.4f}'}")

        if xtol > 0 and theta_k < xtol:
            if full_iter == 1 and extra_options_flag("use_autostep", extra_options):
                print("step increment:", step_increment)
                step_increment = 2
            return True, step_increment, diverging
        
    if full_iter > 0 and extra_options_flag("rtol", extra_options):
        #theta_k_ = torch.zeros((RK_rows), dtype=z_.dtype, device=z_.device)
        theta_k_ = torch.log(torch.norm(z_D - z_D_prev)).abs().item()
        print(f"rtol iter: {full_iter} theta: {theta_k_:.4f}")
        #print(f"iter: {full_iter} theta: {f'{theta_k_:.4f}'}")

        if rtol > 0 and theta_k_ < rtol:
            if full_iter == 1 and extra_options_flag("use_autostep", extra_options):
                print("step increment:", step_increment)
                step_increment = 2
            return True, step_increment, diverging
        
    if full_iter > 0 and extra_options_flag("Rtol", extra_options):
        theta_k_ = torch.zeros((RK_rows), dtype=z_.dtype, device=z_.device)
        for r in range(RK_rows):
            if extra_options_flag("Rtol_discrete", extra_options):
                theta_k_[r] = torch.log(torch.norm(z_prev_[r] - z_[r]) + 1e-8).abs().item()
            else:
                theta_k_[r] = (torch.log(torch.norm(z_prev_[r]) + 1e-8)     -     torch.log(torch.norm(z_[r]) + 1e-8)).abs().item()

        print(f"Rtol iter: {full_iter} theta: {[f'{theta:.4f}' for theta in theta_k_.tolist()]}")

        if Rtol > 0 and theta_k_.max() < Rtol:
            if full_iter == 1 and extra_options_flag("use_autostep", extra_options):
                print("step increment:", step_increment)
                step_increment = 2
            return True, step_increment, diverging
        
        
        
    elif full_iter > 0 and extra_options_flag("ztol", extra_options):
        theta_k_, eta_k_, eta_k_delta_z_ = [torch.zeros((RK_rows), dtype=z_.dtype, device=z_.device) for _ in range(3)]
        x_0_norm = torch.norm(x_0)
        
        for r in range(RK_rows):
            theta_k_[r] = torch.norm(z_[r]) / torch.norm(z_prev_[r])
            eta_k_[r] = theta_k_[r] / (1 - theta_k_[r])
            eta_k_delta_z_[r] = (eta_k_[r] * torch.norm(z_[r]) / x_0_norm)
        
            
        """theta_k_1 = torch.norm(z_1) / torch.norm(z_1_prev)
        theta_k_2 = torch.norm(z_2) / torch.norm(z_2_prev)
        theta_k_3 = torch.norm(z_3) / torch.norm(z_3_prev)
        
        eta_k_1 = theta_k_1 / (1 - theta_k_1)
        eta_k_2 = theta_k_2 / (1 - theta_k_2)
        eta_k_3 = theta_k_3 / (1 - theta_k_3)
        
        eta_k_delta_z_1 = (eta_k_1 * torch.norm(z_1)).item()
        eta_k_delta_z_2 = (eta_k_2 * torch.norm(z_2)).item()
        eta_k_delta_z_3 = (eta_k_3 * torch.norm(z_3)).item()"""
        
        #print("iter:", full_iter, "theta:", {[f'{theta:.4f}' for theta in theta_k_.tolist()]}, {[f'{theta:.4f}' for theta in eta_k_.tolist()]}, {[f'{theta:.4f}' for theta in eta_k_delta_z_.tolist()]},)
        print("ztol iter:", full_iter, "theta:", 
            [f'{theta:.4f}' for theta in theta_k_.tolist()], 
            [f'{theta:.4f}' for theta in eta_k_.tolist()], 
            [f'{theta:.4f}' for theta in eta_k_delta_z_.tolist()])

        #if eta_k_delta_z_1 <= kappa_tol and eta_k_delta_z_2 <= kappa_tol and eta_k_delta_z_3 <= kappa_tol:
        if eta_k_delta_z_.max() <= kappa_tol:
            if full_iter == 1 and extra_options_flag("use_autostep", extra_options):
                print("step increment:", step_increment)
                step_increment = 2
            return True, step_increment, diverging

    return False, step_increment, diverging




def sample_rk_gausslang_3s_full(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
    noise_sampler2 = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+10000, sigma_min=0.0, sigma_max=1.0)
    
    A_tableau = torch.tensor([[5/36, 2/9 - 15**0.5 / 15, 5/36 - 15**0.5 / 30],            [5/36 + 15**0.5 / 24, 2/9, 5/36 - 15**0.5 / 24],            [5/36 + 15**0.5 / 30, 2/9 + 15**0.5 / 15, 5/36],], dtype=x.dtype, device=x.device)
    B_row = torch.tensor([5/18, 4/9, 5/18], dtype=x.dtype, device=x.device)
    A_inv = torch.linalg.inv(A_tableau)
    D_row = B_row @ A_inv
    
    denoised_prev, denoised_prev2 = [torch.zeros_like(x) for _ in range(2)]
    
    x_next_pred = None
    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
        
        sigma_up, sigma, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, eta, noise_mode)
        
        x_0 = x.clone()
    
        
        a1_1, a1_2, a1_3 = 5/36, 2/9 - 15**0.5 / 15, 5/36 - 15**0.5 / 30
        a2_1, a2_2, a2_3 = 5/36 + 15**0.5 / 24, 2/9, 5/36 - 15**0.5 / 24
        a3_1, a3_2, a3_3 = 5/36 + 15**0.5 / 30, 2/9 + 15**0.5 / 15, 5/36
        b1, b2, b3 = 5/18, 4/9, 5/18
        c1, c2, c3 = 1/2 - 15**0.5 / 10, 1/2, 1/2 + 15**0.5 / 10
        
        h = sigma_next - sigma
        
        sigma_1 = sigma + h * c1
        sigma_2 = sigma + h * c2
        sigma_3 = sigma + h * c3
        
        if sigma_next == 0:
            return denoised_2
        
        if step == 0:
            denoised = model(x, sigma * s_in, **extra_args)
            if sigma_next == 0:
                return denoised
            eps = (x - denoised)/sigma
            
            eps_1 = eps * sigma_1
            eps_2 = eps * sigma_2 
            eps_3 = eps * sigma_3
            
            x_1 = x_0 + h * (a1_1 * eps_1 + a1_2 * eps_2 + a1_3 * eps_3)
            x_2 = x_0 + h * (a2_1 * eps_1 + a2_2 * eps_2 + a2_3 * eps_3)
            x_3 = x_0 + h * (a3_1 * eps_1 + a3_2 * eps_2 + a3_3 * eps_3)
            
        else:
            sigma_prev = sigmas[step-1]
            h_prev = sigma - sigma_prev
            
            w = h / h_prev

            zi_1 = lagrange_interpolation([c1,c2,c3], [z_1_prev, z_2_prev, z_3_prev], 1 + w*c1).squeeze(0) + x_prev - x_0
            zi_2 = lagrange_interpolation([c1,c2,c3], [z_1_prev, z_2_prev, z_3_prev], 1 + w*c2).squeeze(0) + x_prev - x_0
            zi_3 = lagrange_interpolation([c1,c2,c3], [z_1_prev, z_2_prev, z_3_prev], 1 + w*c3).squeeze(0) + x_prev - x_0
            
            x_1 = x_0 + zi_1
            x_2 = x_0 + zi_2
            x_3 = x_0 + zi_3
        
        for full_iter in range(sub_iter):
            denoised_1 = model(x_1, sigma_1 * s_in, **extra_args)
            
            eps_1 = (x_1 - denoised_1) / sigma_1
            eps_2 = (x_2 - denoised_1) / sigma_2     
            eps_3 = (x_3 - denoised_1) / sigma_3            

            z_1 = h * (a1_1 * eps_1 + a1_2 * eps_2 + a1_3 * eps_3)
            z_2 = h * (a2_1 * eps_1 + a2_2 * eps_2 + a2_3 * eps_3)
            z_3 = h * (a3_1 * eps_1 + a3_2 * eps_2 + a3_3 * eps_3)
            
            x_1 = x_0 + z_1
            x_2 = x_0 + z_2
            x_3 = x_0 + z_3
            
            denoised_2 = model(x_2, sigma_2 * s_in, **extra_args) 
            
            eps_1 = (x_1 - denoised_1) / sigma_1
            eps_2 = (x_2 - denoised_2) / sigma_2     
            eps_3 = (x_3 - denoised_2) / sigma_3            

            z_1 = h * (a1_1 * eps_1 + a1_2 * eps_2 + a1_3 * eps_3)
            z_2 = h * (a2_1 * eps_1 + a2_2 * eps_2 + a2_3 * eps_3)
            z_3 = h * (a3_1 * eps_1 + a3_2 * eps_2 + a3_3 * eps_3)
            
            x_1 = x_0 + z_1
            x_2 = x_0 + z_2
            x_3 = x_0 + z_3
            
            denoised_3 = model(x_3, sigma_3 * s_in, **extra_args) 

            eps_1 = (x_1 - denoised_1) / sigma_1
            eps_2 = (x_2 - denoised_2) / sigma_2     
            eps_3 = (x_3 - denoised_3) / sigma_3            

            z_1 = h * (a1_1 * eps_1 + a1_2 * eps_2 + a1_3 * eps_3)
            z_2 = h * (a2_1 * eps_1 + a2_2 * eps_2 + a2_3 * eps_3)
            z_3 = h * (a3_1 * eps_1 + a3_2 * eps_2 + a3_3 * eps_3)
            
            x_1 = x_0 + z_1
            x_2 = x_0 + z_2
            x_3 = x_0 + z_3
            
        h_new1 = h_new2 = h_new3 = h
            
        for full_iter in range(iter):
            
            sub_sigma_next1 = sigma_1
            sub_sigma_up1, sub_sigma1, sub_sigma_down1, sub_alpha_ratio1 = get_res4lyf_step_with_model(model, sigma, sub_sigma_next1, eta_var, noise_mode)
            h_new1 = h * (sub_sigma_down1 - sigma) / (sub_sigma_next1 - sigma) 
                        
            sub_sigma_next2 = sigma_2
            sub_sigma_up2, sub_sigma2, sub_sigma_down2, sub_alpha_ratio2 = get_res4lyf_step_with_model(model, sigma, sub_sigma_next2, eta_var, noise_mode)
            h_new2 = h * (sub_sigma_down2 - sigma) / (sub_sigma_next2 - sigma) 
            
            sub_sigma_next3 = sigma_3
            sub_sigma_up3, sub_sigma3, sub_sigma_down3, sub_alpha_ratio3 = get_res4lyf_step_with_model(model, sigma, sub_sigma_next3, eta_var, noise_mode)
            h_new3 = h * (sub_sigma_down3 - sigma) / (sub_sigma_next3 - sigma) 
            
            
            denoised_1 = model(x_1, sigma_1 * s_in, **extra_args)
            denoised_2 = model(x_2, sigma_2 * s_in, **extra_args) 
            denoised_3 = model(x_3, sigma_3 * s_in, **extra_args) 
            
            eps_1 = (x_1 - denoised_1) / sigma_1
            eps_2 = (x_2 - denoised_2) / sigma_2     
            eps_3 = (x_3 - denoised_3) / sigma_3            

            z_1 = h_new1 * (a1_1 * eps_1 + a1_2 * eps_2 + a1_3 * eps_3)
            z_2 = h_new2 * (a2_1 * eps_1 + a2_2 * eps_2 + a2_3 * eps_3)
            z_3 = h_new3 * (a3_1 * eps_1 + a3_2 * eps_2 + a3_3 * eps_3)
            
            x_1 = x_0 + z_1
            x_2 = x_0 + z_2
            x_3 = x_0 + z_3
            
            noise1 = noise_sampler2(sigma=sigma, sigma_next=sigma_1)
            noise2 = noise_sampler2(sigma=sigma, sigma_next=sigma_2)
            noise3 = noise_sampler2(sigma=sigma, sigma_next=sigma_3)
            
            x_1 = sub_alpha_ratio1 * x_1 + sub_sigma_up1 * noise1
            x_2 = sub_alpha_ratio2 * x_2 + sub_sigma_up2 * noise2
            x_3 = sub_alpha_ratio3 * x_3 + sub_sigma_up3 * noise3

        

        #z_1 = h_new1 * (a1_1 * eps_1 + a1_2 * eps_2 + a1_3 * eps_3)
        #z_2 = h_new2 * (a2_1 * eps_1 + a2_2 * eps_2 + a2_3 * eps_3)
        #z_3 = h_new3 * (a3_1 * eps_1 + a3_2 * eps_2 + a3_3 * eps_3)
        
        z_1 = h * (a1_1 * eps_1 + a1_2 * eps_2 + a1_3 * eps_3)
        z_2 = h * (a2_1 * eps_1 + a2_2 * eps_2 + a2_3 * eps_3)
        z_3 = h * (a3_1 * eps_1 + a3_2 * eps_2 + a3_3 * eps_3)
        
        z_next = D_row[0]*z_1 + D_row[1]*z_2 + D_row[2]*z_3
        #z_next = h * (b1 * eps_1 + b2 * eps_2 + b3 * eps_3)
        x_next = x_0 + z_next
        
        x_prev = x_0.clone()
        x = x_next
        
        z_next_prev = D_row[0]*z_1 + D_row[1]*z_2 + D_row[2]*z_3
        
        z_1 = h * (a1_1 * eps_1 + a1_2 * eps_2 + a1_3 * eps_3)
        z_2 = h * (a2_1 * eps_1 + a2_2 * eps_2 + a2_3 * eps_3)
        z_3 = h * (a3_1 * eps_1 + a3_2 * eps_2 + a3_3 * eps_3)
        
        z_1_prev = z_1
        z_2_prev = z_2
        z_3_prev = z_3
        #z_next_prev = h * (b1 * eps_1 + b2 * eps_2)
        #z_next_prev = h * (b1 * eps_1 + b2 * eps_2 + b3 * eps_3)
        
        
        
        denoised_prev2 = denoised_prev
        denoised_prev = denoised
        
        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised_2})

    return denoised












def sample_rk_gausslang(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
    noise_sampler2 = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+10000, sigma_min=0.0, sigma_max=1.0)
    
    A_tableau = torch.tensor([[1/4, 1/4 - 3**0.5 / 6], [1/4 + 3**0.5 / 6, 1/4],], dtype=x.dtype, device=x.device)
    B_row = torch.tensor([1/2, 1/2], dtype=x.dtype, device=x.device)
    A_inv = torch.linalg.inv(A_tableau)
    D_row = B_row @ A_inv
    
    x_next_pred = None
    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
        
        sigma_up, sigma, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, eta, noise_mode)
        
        x_0 = x.clone()
    
        
        a1_1, a1_2 = 1/4, 1/4 - 3**0.5 / 6
        a2_1, a2_2 = 1/4 + 3**0.5 / 6, 1/4
        b1, b2 = 1/2, 1/2
        c1, c2 = 1/2 - 3**0.5 / 6, 1/2 + 3**0.5 / 6
        
        h = sigma_next - sigma
        
        sigma_1 = sigma + h * c1
        sigma_2 = sigma + h * c2
        
        if sigma_next == 0:
            return denoised_2
        
        if step == 0:
            denoised = model(x, sigma * s_in, **extra_args)
            if sigma_next == 0:
                return denoised
            eps = (x - denoised)/sigma
            
            eps_1 = eps * sigma_1
            eps_2 = eps * sigma_2 
            
            x_1 = x_0 + h * (a1_1 * eps_1 + a1_2 * eps_2)
        else:
            sigma_prev = sigmas[step-1]
            h_prev = sigma - sigma_prev
            
            w = h / h_prev

            #zi_1 = lagrange_interpolation([c1,c2,1], [z_1_prev, z_2_prev, z_next_prev], 1 + w*c1).squeeze(0) + x_prev - x_0
            
            zi_1 = lagrange_interpolation([c1,c2,1], [z_1_prev, z_2_prev,z_next_prev], 1 + w*c1).squeeze(0) + x_prev - x_0
            x_1 = x_0 + zi_1
            
            #zi_2 = lagrange_interpolation([c1,c2,1], [z_1_prev, z_2_prev, z_next_prev], 1 + w*c2).squeeze(0) + x_prev - x_0
            #x_2 = x_0 + zi_2

        

        denoised_1 = model(x_1, sigma_1 * s_in, **extra_args)
        
        eps_2 = (x_0 - denoised_1) / sigma_2
        eps_1 = (x_0 - denoised_1) / sigma_1                      # will be for c1 for radau ia 
        
        #z_1 = h * (a1_1 * eps_1 + a1_2 * eps_2)

        #z_2 = h * (a2_1 * eps_1 + a2_2 * eps_2)
        #x_2 = x_0 + z_2

        if True:# step == 0:
            z_2 = h * (a2_1 * eps_1 + a2_2 * eps_2)
            x_2 = x_0 + z_2
        else:
            zi_2 = lagrange_interpolation([c1,c2,1], [z_1_prev, z_2_prev,z_next_prev], 1 + w*c2).squeeze(0) + x_prev - x_0
            x_2 = x_0 + zi_2
        
        denoised_2 = model(x_2, sigma_2 * s_in, **extra_args) 
        eps_2 = (x_0 - denoised_2) / sigma_2              

        z_next = h * (b1 * eps_1 + b2 * eps_2)
        x_next = x_0 + z_next
        
        
        

        
        z_1 = h * (a1_1 * eps_1 + a1_2 * eps_2)
        z_2 = h * (a2_1 * eps_1 + a2_2 * eps_2)
        
        z_next = D_row[0]*z_1 + D_row[1]*z_2
        x_next = x_0 + z_next
        
        x_prev = x_0.clone()
        x = x_next
        
        z_1_prev = z_1
        z_2_prev = z_2
        #z_next_prev = h * (b1 * eps_1 + b2 * eps_2)
        z_next_prev = D_row[0]*z_1 + D_row[1]*z_2
        
        
        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised_2})

    return denoised







def sample_rk_gausslangeps(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
    noise_sampler2 = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+10000, sigma_min=0.0, sigma_max=1.0)
    
    A_tableau = torch.tensor([[1/4, 1/4 - 3**0.5 / 6], [1/4 + 3**0.5 / 6, 1/4],], dtype=x.dtype, device=x.device)
    B_row = torch.tensor([1/2, 1/2], dtype=x.dtype, device=x.device)
    A_inv = torch.linalg.inv(A_tableau)
    D_row = B_row @ A_inv
    
    x_next_pred = None
    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
        
        sigma_up, sigma, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, eta, noise_mode)
        
        x_0 = x.clone()
    
        
        a1_1, a1_2 = 1/4, 1/4 - 3**0.5 / 6
        a2_1, a2_2 = 1/4 + 3**0.5 / 6, 1/4
        b1, b2 = 1/2, 1/2
        c1, c2 = 1/2 - 3**0.5 / 6, 1/2 + 3**0.5 / 6
        
        h = sigma_next - sigma
        
        sigma_1 = sigma + h * c1
        sigma_2 = sigma + h * c2
        
        if sigma_next == 0:
            return denoised_2
        
        if step == 0:
            denoised = model(x, sigma * s_in, **extra_args)
            if sigma_next == 0:
                return denoised
            eps = (x - denoised)/sigma
            
            eps_1 = eps * sigma_1
            eps_2 = eps * sigma_2 
            
            x_1 = x_0 + h * (a1_1 * eps_1 + a1_2 * eps_2)
        else:
            sigma_prev = sigmas[step-1]
            h_prev = sigma - sigma_prev
            
            w = h / h_prev

            #zi_1 = lagrange_interpolation([c1,c2,1], [z_1_prev, z_2_prev, z_next_prev], 1 + w*c1).squeeze(0) + x_prev - x_0
            
            #eps_1_guess = lagrange_interpolation([c1,c2], [eps_1_prev, eps_2_prev], 1 + w*c1).squeeze(0) + x_prev - x_0   # page 120 (pdf) of butcher book, stiff equations
            #eps_2_guess = lagrange_interpolation([c1,c2], [eps_1_prev, eps_2_prev], 1 + w*c2).squeeze(0) + x_prev - x_0
            
            eps_1_guess = lagrange_interpolation([c1,c2,1], [eps_1_prev, eps_2_prev, eps_full_prev], 1 + w*c1).squeeze(0) #+ x_prev - x_0
            eps_2_guess = lagrange_interpolation([c1,c2,1], [eps_1_prev, eps_2_prev, eps_full_prev], 1 + w*c2).squeeze(0) #+ x_prev - x_0
            
            z_1_guess = h * (a1_1 * eps_1_guess + a1_2 * eps_2_guess)
            x_1 = x_0 + z_1_guess
            
            #zi_2 = lagrange_interpolation([c1,c2,1], [z_1_prev, z_2_prev, z_next_prev], 1 + w*c2).squeeze(0) + x_prev - x_0
            #x_2 = x_0 + zi_2

        

        denoised_1 = model(x_1, sigma_1 * s_in, **extra_args)
        
        #eps_2 = (x_0 - denoised_1) / sigma_2
        eps_1 = (x_0 - denoised_1) / sigma_1                      # will be for c1 for radau ia 
        
        #z_1 = h * (a1_1 * eps_1 + a1_2 * eps_2)

        #z_2 = h * (a2_1 * eps_1 + a2_2 * eps_2)
        #x_2 = x_0 + z_2

        if step == 0:
            eps_2 = (x_0 - denoised_1) / sigma_2
            
            z_2 = h * (a2_1 * eps_1 + a2_2 * eps_2)
            x_2 = x_0 + z_2
        else:
            #zi_2 = lagrange_interpolation([c1,c2,1], [z_1_prev, z_2_prev,z_next_prev], 1 + w*c2).squeeze(0) + x_prev - x_0
            
            if extra_options_flag("eps2guess", extra_options):
                #eps_2_guess = lagrange_interpolation([c1,c2,1, 1+w*c1], [eps_1_prev, eps_2_prev, eps_full_prev, eps_2], 1 + w*c2).squeeze(0) + x_prev - x_0
                eps_2_guess = lagrange_interpolation([c1,c2,1, 1+w*c1], [eps_1_prev, eps_2_prev, eps_full_prev, eps_2], 1 + w*c2).squeeze(0)
            z_2_semiguess = h * (a2_1 * eps_1 + a2_2 * eps_2_guess)
            x_2 = x_0 + z_2_semiguess
        
        denoised_2 = model(x_2, sigma_2 * s_in, **extra_args) 
        eps_2 = (x_0 - denoised_2) / sigma_2              

        


        #z_next = h * (b1 * eps_1 + b2 * eps_2)
        #x_next = x_0 + z_next
        
        
        

        
        z_1 = h * (a1_1 * eps_1 + a1_2 * eps_2)
        z_2 = h * (a2_1 * eps_1 + a2_2 * eps_2)
        
        z_next = D_row[0]*z_1 + D_row[1]*z_2
        x_next = x_0 + z_next
        
        x_prev = x_0.clone()
        x = x_next
        
        z_1_prev = z_1
        z_2_prev = z_2
        #z_next_prev = h * (b1 * eps_1 + b2 * eps_2)
        z_next_prev = D_row[0]*z_1 + D_row[1]*z_2
        
        eps_1_prev = eps_1
        eps_2_prev = eps_2
        
        
        z_1_full = (-sigma) * (a1_1 * eps_1 + a1_2 * eps_2)
        z_2_full = (-sigma) * (a2_1 * eps_1 + a2_2 * eps_2)
        z_next_full_prev = D_row[0]*z_1_full + D_row[1]*z_2_full
        
        denoised_full = x_0 + z_next_full_prev
        eps_full_prev = (x_0 - denoised_full) / sigma
        
        
        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised_2})

    return denoised






# NOT FINISHED IMPLEMENTING
def sample_rk_gauss_2s(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
    noise_sampler2 = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+10000, sigma_min=0.0, sigma_max=1.0)
    
    x_next_pred = None
    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
        
        sigma_up, sigma, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, eta, noise_mode)
        
        x_0 = x.clone()
        
        a1_1, a1_2 = 1/4, 1/4 - 3**0.5 / 6
        a2_1, a2_2 = 1/4 + 3**0.5 / 6, 1/4
        b1, b2 = 1/2, 1/2
        c1, c2 = 1/2 - 3**0.5 / 6, 1/2 + 3**0.5 / 6
        
        h = sigma_next - sigma
        
        sigma_1 = sigma + h * c1
        sigma_2 = sigma + h * c2
        
        if sigma_next == 0:
            return denoised_2
        
        if step == 0:
            denoised = model(x, sigma * s_in, **extra_args)
            if sigma_next == 0:
                return denoised
            eps = (x - denoised)/sigma
            
            eps_1 = eps * sigma_1
            eps_2 = eps * sigma_2 
            
            x_1 = x_0 + h * (a1_1 * eps_1 + a1_2 * eps_2)
        else:
            sigma_prev = sigmas[step-1]
            h_prev = sigma - sigma_prev
            
            w = h / h_prev

            zi_1 = lagrange_interpolation([c1,c2,1], [z_1_prev, z_2_prev, z_next_prev], 1 + w*c1).squeeze(0) + x_prev - x_0
            x_1 = x_0 + zi_1
            
            zi_2 = lagrange_interpolation([c1,c2,1], [z_1_prev, z_2_prev, z_next_prev], 1 + w*c2).squeeze(0) + x_prev - x_0
            x_2 = x_0 + zi_2

        z_1 = h * (a1_1 * eps_1 + a1_2 * eps_2)

        denoised_1 = model(x_1, sigma_1 * s_in, **extra_args)
        
        eps_2 = (x_0 - denoised_1) / sigma_2
        eps_1 = (x_0 - denoised_1) / sigma_1                      # will be for c1 for radau ia 

        if step == 0:
            z_2 = h * (a2_1 * eps_1 + a2_2 * eps_2)
            x_2 = x_0 + z_2
        
        denoised_2 = model(x_2, sigma_2 * s_in, **extra_args) 
        eps_2 = (x_0 - denoised_2) / sigma_2              

        z_next = h * (b1 * eps_1 + b2 * eps_2)
        x_next = x_0 + z_next
        
        x_prev = x.clone()
        x = x_next
        
        z_1_prev = h * (a1_1 * eps_1 + a1_2 * eps_2)
        z_2_prev = h * (a2_1 * eps_1 + a2_2 * eps_2)
        z_next_prev = h * (b1 * eps_1 + b2 * eps_2)
        
        
        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised_2})

    return denoised





def get_data_from_step(x, x_next, sigma, sigma_next):
    h = sigma_next - sigma
    return (sigma_next * x - sigma * x_next) / h

def get_epsilon_from_step(x, x_next, sigma, sigma_next):
    h = sigma_next - sigma
    return (x - x_next) / h




# NOT FINISHED IMPLEMENTING
def sample_rk_ralradau(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
    noise_sampler2 = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+10000, sigma_min=0.0, sigma_max=1.0)
    
    x_next_pred = None
    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
        
        sigma_up, sigma, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, eta, noise_mode)
        
        x_0 = x.clone()
        
        a1_1, a1_2 = 1/4, 1/4 - 3**0.5 / 6
        a2_1, a2_2 = 1/4 + 3**0.5 / 6, 1/4
        b1, b2 = 1/2, 1/2
        c1, c2 = 1/2 - 3**0.5 / 6, 1/2 + 3**0.5 / 6
        
        h = sigma_next - sigma
        
        sigma_1 = sigma + h * c1
        sigma_2 = sigma + h * c2
        
        if sigma_next == 0:
            return denoised_2
        
        denoised = model(x, sigma * s_in, **extra_args)
        eps_1 = (x - denoised)/sigma

        if sigma_next == 0:
            return denoised
        
        x_2 = x_0 + h * ((2/3) * eps_1) # ralston_2s explicit
        
        """x_2 = x_0 + h*(a2_1*eps_1) + h*(a2_2*eps_2)
        
        x_2 - x_0 + h*(a2_1*eps_1) = h*(a2_2*eps_2)
        
        h*(a2_2*eps_2) = x_2 - x_0 + h*(a2_1*eps_1)"""

        eps_2 = (x_2 - x_0 + h*(a2_1*eps_1)) / (h*a2_2) # find eps_2 to solve radau ia 2s
        
        eps_2 = get_data_from_step(x_0, x_2, sigma, sigma_2)
        
        """eps_2 = get_epsilon_from_step(x_0, x_2, sigma, sigma_2)


        x_1 = x_0 + h * (a1_1 * eps_1 + a1_2 * eps_2)
        
        denoised_1 = model(x_1, sigma_1 * s_in, **extra_args)
        eps_1 = (x_0 - denoised_1)/sigma_1"""
        
        x_2 = x_0 + h * (a2_1 * eps_1 + a2_2 * eps_2)
        
        denoised_2 = model(x_2, sigma_2 * s_in, **extra_args)
        eps_2 = (x_0 - denoised_2)/sigma_2
        
        x_next = x_0 + h * (b1 * eps_1 + b2 * eps_2)


        """z_1 = h * (a1_1 * eps_1 + a1_2 * eps_2)

        denoised_1 = model(x_1, sigma_1 * s_in, **extra_args)
        
        eps_2 = (x_0 - denoised_1) / sigma_2
        eps_1 = (x_0 - denoised_1) / sigma_1                      # will be for c1 for radau ia 

        if step == 0:
            z_2 = h * (a2_1 * eps_1 + a2_2 * eps_2)
            x_2 = x_0 + z_2
        
        denoised_2 = model(x_2, sigma_2 * s_in, **extra_args) 
        eps_2 = (x_0 - denoised_2) / sigma_2              

        z_next = h * (b1 * eps_1 + b2 * eps_2)
        x_next = x_0 + z_next
        
        x_prev = x.clone()
        x = x_next
        
        z_1_prev = h * (a1_1 * eps_1 + a1_2 * eps_2)
        z_2_prev = h * (a2_1 * eps_1 + a2_2 * eps_2)
        z_next_prev = h * (b1 * eps_1 + b2 * eps_2)"""
        
        x = x_next
        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised_2})

    return denoised










def sample_rk_gausscycle2(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
    noise_sampler2 = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+10000, sigma_min=0.0, sigma_max=1.0)
    
    x_next_pred = None
    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
        
        sigma_up, sigma, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, eta, noise_mode)
        
        x_0 = x.clone()
        
        a1_1, a1_2 = 1/4, 1/4 - 3**0.5 / 6
        a2_1, a2_2 = 1/4 + 3**0.5 / 6, 1/4
        b1, b2 = 1/2, 1/2
        c1, c2 = 1/2 - 3**0.5 / 6, 1/2 + 3**0.5 / 6
        
        alpha_ratio_3 = 1
        sigma_up_3 = 0
        sigma_down_3 = sigma_down
        if sigma_next > 0:
            sigma_down_3 = sigma_next + (sigmas[step+2] - sigmas[step+1]) * c1
            alpha_ratio_3, sigma_up_3, sigma_down_3 = get_alpha_ratio_from_sigma_down(sigma_down_3, sigma_next, eta)
            h_down_3 = sigma_down_3 - sigma

        h_no_eta = sigma_next - sigma
        h = sigma_down - sigma
        
        sigma_1 = sigma + h * c1
        sigma_2 = sigma + h * c2
        
        if sigma_next == 0:
            return denoised_2
        
        if x_next_pred == None:
            denoised = model(x, sigma * s_in, **extra_args)
            if sigma_next == 0:
                return denoised
            eps = (x - denoised)/sigma
            
            eps_1 = eps * sigma_1
            eps_2 = eps * sigma_2 
            
            x_1 = x_0 + h * (a1_1 * eps_1 + a1_2 * eps_2)
        else:
            x_1 = x_next_pred
            if extra_options_flag("new_guess", extra_options):
                eps_1 = (x_0 - denoised_prev) / sigma_1   
                eps_2 = (x_0 - denoised_prev) / sigma_2   
                x_1 = x_0 + h * (a1_1 * eps_1 + a1_2 * eps_2)

        denoised_1 = model(x_1, sigma_1 * s_in, **extra_args)
        
        eps_2 = (x_0 - denoised_1) / sigma_2
        eps_1 = (x_0 - denoised_1) / sigma_1                      # will be for c1 for radau ia 
        
        sub_sigma_next = sigma_2
        sub_sigma_up, sub_sigma, sub_sigma_down, sub_alpha_ratio = get_res4lyf_step_with_model(model, sigma, sub_sigma_next, eta_var, noise_mode)
        h_new = h * (sub_sigma_down - sigma) / (sub_sigma_next - sigma) 
        
        x_2 = x_0 + h_new * (a2_1 * eps_1 + a2_2 * eps_2)
        
        noise = noise_sampler2(sigma=sigma, sigma_next=sigma_next)
        noise = (noise - noise.mean()) / noise.std()
        x_2 = sub_alpha_ratio * x_2 + noise * sub_sigma_up
        
        
        
        
        denoised_2 = model(x_2, sigma_2 * s_in, **extra_args) 
        eps_2 = (x_0 - denoised_2) / sigma_2                       # c = 1 maps to 2/3 for radau ia, as in, c2 as well



        #sub_sigma_next = sigma_down_3
        #sub_sigma_up, sub_sigma, sub_sigma_down, sub_alpha_ratio = get_res4lyf_step_with_model(model, sigma, sub_sigma_next, eta_var, noise_mode)
        #h_new = h * (sub_sigma_down - sigma) / (sub_sigma_next - sigma) 
        
        x_next = x_0 + h * (b1 * eps_1 + b2 * eps_2)
        
        x_next_pred = x_0 + h_down_3 * (b1 * eps_1 + b2 * eps_2)
        
        #x_next = x_next_pred = x_0 + h_new * (b1 * eps_1 + b2 * eps_2)
        #noise = noise_sampler2(sigma=sigma, sigma_next=sigma_next)
        #noise = (noise - noise.mean()) / noise.std()
        #x_next = sub_alpha_ratio * x_next + noise * sub_sigma_up
        
        #noise1 = noise_sampler(sigma=sigma, sigma_next=sigma_next)
        #noise1 = (noise1 - noise1.mean()) / noise1.std()
        
        #x_next = x_next_pred = alpha_ratio * x_next + noise * sigma_up

        x = x_next
        
        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised_2})

        #x = alpha_ratio_3 * x + noise * s_noise * sigma_up_3   
         
        #if extra_options_flag("noise1", extra_options):
        #    x = alpha_ratio_3 * x + noise1 * s_noise * sigma_up_3   
        #else:
        #    x = alpha_ratio_3 * x + noise * s_noise * sigma_up_3     
        
        denoised_prev = denoised_2   #THIS MIGHT BE COMPLETELY WRONG!!!!!!!!!!!!!!!!!!!!! just hiding pylance error

    return denoised







def sample_rk_radaucycle_3s(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
    noise_sampler2 = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+10000, sigma_min=0.0, sigma_max=1.0)
    
    x_next_pred = None
    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
        
        sigma_up, sigma, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, eta, noise_mode)
        
        x_0 = x.clone()
        
        a1_1, a1_2, a1_3 = 11/45 - 7*6**0.5 / 360, 37/225 - 169*6**0.5 / 1800, -2/225 + 6**0.5 / 75
        a2_1, a2_2, a2_3 = 37/225 + 169*6**0.5 / 1800, 11/45 + 7*6**0.5 / 360, -2/225 - 6**0.5 / 75
        a3_1, a3_2, a3_3 = 4/9 - 6**0.5 / 36, 4/9 + 6**0.5 / 36, 1/9
        b1, b2, b3 = 4/9 - 6**0.5 / 36, 4/9 + 6**0.5 / 36, 1/9
        c1, c2, c3 = 2/5 - 6**0.5 / 10, 2/5 + 6**0.5 / 10, 1.
                
        h_no_eta = sigma_next - sigma
        h = sigma_down - sigma
        sigma_1 = sigma + h * c1
        sigma_2 = sigma + h * c2
        sigma_3 = sigma + h * c3
        
        # radau iia stage
        
        if sigma_next == 0:
            return denoised_2
        
        if x_next_pred == None:
            denoised = model(x, sigma * s_in, **extra_args)
            if sigma_next == 0:
                return denoised
            eps = (x - denoised)/sigma
            
            eps_1 = eps * sigma_1
            eps_2 = eps * sigma_2 
            eps_3 = eps * sigma_3
            
            x_1 = x_0 + h * (a1_1 * eps_1 + a1_2 * eps_2 + a1_3 * eps_3)
        else:
            x_1 = x_next_pred
            if extra_options_flag("new_guess", extra_options):
                eps_1 = (x_0 - denoised_prev) / sigma_1   
                eps_2 = (x_0 - denoised_prev) / sigma_2   
                eps_3 = (x_0 - denoised_prev) / sigma_3
                x_1 = x_0 + h * (a1_1 * eps_1 + a1_2 * eps_2 + a1_3 * eps_3)
                
        denoised_1 = model(x_1, sigma_1 * s_in, **extra_args)
        
        if x_next_pred is not None:
            eps_2 = (x_0 - denoised_1) / sigma_2
            eps_3 = (x_0 - denoised_1) / sigma_3
        eps_1 = (x_0 - denoised_1) / sigma_1                       # will be for c1 for radau ia 
        
        sub_sigma_next = sigma_2
        sub_sigma_up, sub_sigma, sub_sigma_down, sub_alpha_ratio = get_res4lyf_step_with_model(model, sigma, sub_sigma_next, eta_var, noise_mode)
        h_new = h * (sub_sigma_down - sigma) / (sub_sigma_next - sigma) 
        
        x_2 = x_0 + h_new * (a2_1 * eps_1 + a2_2 * eps_2 + a2_3 * eps_3)
        
        noise = noise_sampler2(sigma=sigma, sigma_next=sigma_next)
        noise = (noise - noise.mean()) / noise.std()
        x_2 = sub_alpha_ratio * x_2 + noise * sub_sigma_up
        
        
        denoised_2 = model(x_2, sigma_2 * s_in, **extra_args) 
        eps_2 = (x_0 - denoised_2) / sigma_2                       # c = 1 maps to 2/3 for radau ia, as in, c2 as well






        sub_sigma_next = sigma_3
        sub_sigma_up, sub_sigma, sub_sigma_down, sub_alpha_ratio = get_res4lyf_step_with_model(model, sigma, sub_sigma_next, eta_var, noise_mode)
        h_new = h * (sub_sigma_down - sigma) / (sub_sigma_next - sigma) 
        
        x_3 = x_0 + h_new * (a3_1 * eps_1 + a3_2 * eps_2 + a3_3 * eps_3)
        
        noise = noise_sampler2(sigma=sigma, sigma_next=sigma_next)
        noise = (noise - noise.mean()) / noise.std()
        x_3 = sub_alpha_ratio * x_3 + noise * sub_sigma_up
        
        
        denoised_3 = model(x_3, sigma_3 * s_in, **extra_args) 
        eps_3 = (x_0 - denoised_3) / sigma_3                       # c = 1 maps to 2/3 for radau ia, as in, c2 as well








        x_next = x_0 + h * (b1 * eps_1 + b2 * eps_2 + b3 * eps_3)
        
        noise = noise_sampler(sigma=sigma, sigma_next=sigma_next)
        noise = (noise - noise.mean()) / noise.std()
        
        # LAST STAGE
        if sigma_next > 0:
            b1, b2, b3 = 1/9, 4/9 + 6**0.5/36, 4/9 - 6**0.5/36
            
            sigma_4 = sigma_next + (sigmas[step+2] - sigmas[step+1]) * c1
            sigma_4_down = sigma + h * (1 + c1)
            
            if sigma_4_down >= sigma_4:
                h = sigma_4_down - sigma
                h = h / (1 + c1)
            
            x_next_pred = x_0 + h * (b1 * eps_1 + b2 * eps_2 + b3 * eps_3)
            
            denoised_prev = x_0 - sigma * (b1 * eps_1 + b2 * eps_2 + b3 * eps_3)
            
            if sigma_4_down < sigma_4:
                alpha_ratio_3, sigma_up_3, sigma_down_3 = get_alpha_ratio_from_sigma_down(sigma_4_down, sigma_4, eta)
                
                #if extra_options_flag("solve_2m", extra_options):
                #    x_next_pred, denoised_prev = res_2m_predictor_step(x_0, denoised_1, denoised_2, sigma_1, sigma_2, sigma_4_down)

                x_next_pred = alpha_ratio_3 * x_next_pred + noise * sigma_up_3

        
        
        x = x_next
        
        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised_3})

        x = alpha_ratio * x + noise * s_noise * sigma_up
        
        
    return denoised








def sample_rk_radaucycle(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
    noise_sampler2 = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+10000, sigma_min=0.0, sigma_max=1.0)
    
    x_next_pred = None
    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
        
        sigma_up, sigma, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, eta, noise_mode)
        
        x_0 = x.clone()
        
        a1_1, a1_2 = 5/12, -1/12
        a2_1, a2_2 = 3/4, 1/4
        b1, b2 = 3/4, 1/4
        c1, c2 = 1/3, 1
                
        h_no_eta = sigma_next - sigma
        h = sigma_down - sigma
        sigma_1 = sigma + h * c1
        sigma_2 = sigma + h * c2
        
        # radau iia stage
        
        if sigma_next == 0:
            return denoised_2
        
        if x_next_pred == None:
            denoised = model(x, sigma * s_in, **extra_args)
            if sigma_next == 0:
                return denoised
            eps = (x - denoised)/sigma
            
            eps_1 = eps * sigma_1
            eps_2 = eps * sigma_2 
            
            x_1 = x_0 + h * (a1_1 * eps_1 + a1_2 * eps_2)
        else:
            x_1 = x_next_pred
        
        # K1    MODEL AND UPDATE
        denoised_1 = model(x_1, sigma_1 * s_in, **extra_args)
        eps_1 = (x_0 - denoised_1) / sigma_1   # will be for c1 for radau ia 
        eps_2 = (x_0 - denoised_1) / sigma_2   #if x_next_pred is not None:
        
        
        

        
        
        
        sub_sigma_next = sigma_2
        sub_sigma_up, sub_sigma, sub_sigma_down, sub_alpha_ratio = get_res4lyf_step_with_model(model, sigma, sub_sigma_next, eta_var, noise_mode)
        h_new = h * (sub_sigma_down - sigma) / (sub_sigma_next - sigma) 
        
        
        
        if step > 0 and extra_options_flag("lagrange_eps2", extra_options):
            #b1_crazy, b2_crazy = 3/4, 1/4
            #x_2 = x_0 + h_new * (b1_crazy * eps_2_prev + b2_crazy * eps_1)
            sigma_prev = sigmas[step-1]
            h_prev = sigma - sigma_prev
            
            w = h / h_prev
            #eps_2 = lagrange_interpolation([c1,c2,1], [eps_1_prev, eps_2_prev, eps_full_prev], 1 + w*c2).squeeze(0)
            #eps_2 = lagrange_interpolation([c1,c2, 1], [eps_1_prev, eps_2_prev, eps_full_prev], 1 + w*c2).squeeze(0)
            #eps_2 = lagrange_interpolation([c1,1], [eps_1_prev, eps_full_prev], 1 + w*c2).squeeze(0)
            
            #eps_2 = lagrange_interpolation([0, c1], [eps_full_prev, eps_1], c2).squeeze(0)
            
            eps_2 = lagrange_interpolation([0, c1], [eps_2_prev, eps_1], c2).squeeze(0)
            
            x_2 = x_0 + h_new * (a2_1 * eps_1 + a2_2 * eps_2)
        else:
            x_2 = x_0 + h_new * (a2_1 * eps_1 + a2_2 * eps_2)
        
        noise = noise_sampler2(sigma=sigma, sigma_next=sigma_next)
        noise = (noise - noise.mean()) / noise.std()
        x_2 = sub_alpha_ratio * x_2 + noise * sub_sigma_up
        
        # K2    MODEL AND UPDATE
        denoised_2 = model(x_2, sigma_2 * s_in, **extra_args) 
        eps_2 = (x_0 - denoised_2) / sigma_2                       # c = 1 maps to 2/3 for radau ia, as in, c2 as well

        x_next = x_0 + h * (b1 * eps_1 + b2 * eps_2)
        
        noise = noise_sampler(sigma=sigma, sigma_next=sigma_next)
        noise = (noise - noise.mean()) / noise.std()
        
        # radau ia stage
        if sigma_next > 0:
            b1, b2 = 1/4, 3/4
            
            h_next = sigmas[step+2] - sigmas[step+1]
            sigma_3 = sigma_next + h_next * c1
            sigma_3_down = sigma + h      * (1+c1)
            
            if sigma_3_down >= sigma_3:
                h_3 = sigma_3_down - sigma
                h_3 = h_3 / (1+c1)
            
            x_next_pred   = x_0 + h_3   * (b1 * eps_1 + b2 * eps_2)
            denoised_prev = x_0 - sigma * (b1 * eps_1 + b2 * eps_2)
            
            if sigma_3_down < sigma_3:
                alpha_ratio_3, sigma_up_3, sigma_down_3 = get_alpha_ratio_from_sigma_down(sigma_3_down, sigma_3, eta)
                
                if extra_options_flag("solve_2m", extra_options):
                    x_next_pred, denoised_prev = res_2m_predictor_step(x_0, denoised_1, denoised_2, sigma_1, sigma_2, sigma_3_down)

                x_next_pred = alpha_ratio_3 * x_next_pred + noise * sigma_up_3

        x = x_next
        
        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised_2})

        x = alpha_ratio * x + noise * s_noise * sigma_up
        
        eps_1_prev = eps_1
        eps_2_prev = eps_2
        denoised_full_prev = x_0 -sigma * (b1 * eps_1 + b2 * eps_2)
        eps_full_prev = (x_0 - denoised_full_prev) / sigma
        #eps_full_prev = (b1 * eps_1 + b2 * eps_2)
        
    return denoised









def sample_rk_radaucycle_retry(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
    noise_sampler2 = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+10000, sigma_min=0.0, sigma_max=1.0)
    
    x_next_pred = None
    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
        
        sigma_up, sigma, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, eta, noise_mode)
        
        x_0 = x.clone()
        
        a1_1, a1_2 = 5/12, -1/12
        a2_1, a2_2 = 3/4, 1/4
        b1, b2 = 3/4, 1/4
        c1, c2 = 1/3, 1
                
        h_no_eta = sigma_next - sigma
        h = sigma_down - sigma
        
        sigma_1 = sigma + h * c1
        sigma_2 = sigma + h * c2
        
        # radau iia stage
        
        if sigma_next == 0:
            return denoised_2
        
        if x_next_pred == None:
            denoised = model(x, sigma * s_in, **extra_args)
            if sigma_next == 0:
                return denoised
            eps = (x - denoised)/sigma
            
            eps_1 = eps * sigma_1
            eps_2 = eps * sigma_2 
            
            x_1 = x_0 + h * (a1_1 * eps_1 + a1_2 * eps_2)
        else:
            x_1 = x_next_pred
        
        # K1    MODEL AND UPDATE
        denoised_1 = model(x_1, sigma_1 * s_in, **extra_args)                   # x_1, sigma_1 = starting point for ralston-radau update
        eps_1 = (x_0 - denoised_1) / sigma_1   # will be for c1 for radau ia 
        eps_2 = (x_0 - denoised_1) / sigma_2   #if x_next_pred is not None:
        
        
        
        sub_sigma_next = sigma_2
        sub_sigma_up, sub_sigma, sub_sigma_down, sub_alpha_ratio = get_res4lyf_step_with_model(model, sigma, sub_sigma_next, eta_var, noise_mode)
        h_new = h * (sub_sigma_down - sigma) / (sub_sigma_next - sigma) 
        
        x_2 = x_0 + h_new * (a2_1 * eps_1 + a2_2 * eps_2)

        noise = noise_sampler2(sigma=sigma, sigma_next=sigma_next)
        noise = (noise - noise.mean()) / noise.std()
        x_2 = sub_alpha_ratio * x_2 + noise * sub_sigma_up
        
        # K2    MODEL AND UPDATE
        denoised_2 = model(x_2, sigma_2 * s_in, **extra_args) 
        eps_2 = (x_0 - denoised_2) / sigma_2                       # c = 1 maps to 2/3 for radau ia, as in, c2 as well

        x = x_0 + h * (b1 * eps_1 + b2 * eps_2)       #this goes to c = 1
        
        noise = noise_sampler(sigma=sigma, sigma_next=sigma_next)
        noise = (noise - noise.mean()) / noise.std()
        x = alpha_ratio * x + noise * s_noise * sigma_up
        
        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised_2})

        x_next_pred = x_1 + h * ((1/4) * eps_1 + (3/4) * eps_2)

        eps_1_prev = eps_1
        eps_2_prev = eps_2
        denoised_full_prev = x_0 -sigma * (b1 * eps_1 + b2 * eps_2)
        eps_full_prev = (x_0 - denoised_full_prev) / sigma
        #eps_full_prev = (b1 * eps_1 + b2 * eps_2)
        
    return denoised




def sample_rk_radaucycle_staggered(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
    noise_sampler2 = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+10000, sigma_min=0.0, sigma_max=1.0)
    
    x_next_pred = None
    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
        
        sigma_up, sigma, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, eta, noise_mode)
        
        x_0 = x.clone()
        
        a1_1, a1_2 = 5/12, -1/12
        a2_1, a2_2 = 3/4, 1/4
        b1, b2 = 3/4, 1/4
        c1, c2 = 1/3, 1
                
        h_no_eta = sigma_next - sigma
        h = sigma_down - sigma
        
        sigma_1 = sigma + h * c1
        sigma_2 = sigma + h * c2
        
        # radau iia stage
        
        if sigma_next == 0:
            return denoised_2
        
        if x_next_pred == None:
            denoised = model(x, sigma * s_in, **extra_args)
            if sigma_next == 0:
                return denoised
            eps = (x - denoised)/sigma
            
            eps_1 = eps * sigma_1
            eps_2 = eps * sigma_2 
            
            x_1 = x_0 + h * (a1_1 * eps_1 + a1_2 * eps_2)
        else:
            x_1 = x_next_pred
        
        # K1    MODEL AND UPDATE
        denoised_1 = model(x_1, sigma_1 * s_in, **extra_args)                   # x_1, sigma_1 = starting point for ralston-radau update
        eps_1 = (x_0 - denoised_1) / sigma_1   # will be for c1 for radau ia 
        eps_2 = (x_0 - denoised_1) / sigma_2   #if x_next_pred is not None:
        
        
        
        sub_sigma_next = sigma_2
        sub_sigma_up, sub_sigma, sub_sigma_down, sub_alpha_ratio = get_res4lyf_step_with_model(model, sigma, sub_sigma_next, eta_var, noise_mode)
        h_new = h * (sub_sigma_down - sigma) / (sub_sigma_next - sigma) 
        
        x_2 = x_0 + h_new * (a2_1 * eps_1 + a2_2 * eps_2)

        noise = noise_sampler2(sigma=sigma, sigma_next=sigma_next)
        noise = (noise - noise.mean()) / noise.std()
        x_2 = sub_alpha_ratio * x_2 + noise * sub_sigma_up
        
        # K2    MODEL AND UPDATE
        denoised_2 = model(x_2, sigma_2 * s_in, **extra_args) 
        eps_2 = (x_0 - denoised_2) / sigma_2                       # c = 1 maps to 2/3 for radau ia, as in, c2 as well

        x = x_0 + h * (b1 * eps_1 + b2 * eps_2)       #this goes to c = 1
        
        noise = noise_sampler(sigma=sigma, sigma_next=sigma_next)
        noise = (noise - noise.mean()) / noise.std()
        x = alpha_ratio * x + noise * s_noise * sigma_up
        
        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised_2})

        # RADAU IIA 2S COMPLETE

        x_0_orig = x_0.clone()
        x_2_orig = x_2.clone()
        
        x_0 = x_1.clone()
        x_2 = x

        a1_1, a1_2 = 1/4, -1/4
        a2_1, a2_2 = 1/4, 5/12
        b1, b2 = 1/4, 3/4
        c1, c2 = 0, 2/3
        
        sigma = sigma_1
        
        h = (sigma_2 - sigma) * 1.5
        
        sigma_next = sigma + h
        
        sigma_up, sigma, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, eta, noise_mode)
        
        h_no_eta = sigma_next - sigma
        h = sigma_down - sigma
        
        #sigma_1 = sigma + h * c1
        #sigma_2 = sigma + h * c2
        
        # radau ia stage
        
        
        x_1 = x_0 + h_no_eta * (a1_1 * eps_1 + a1_2 * eps_2)
        
        
        # K1    MODEL AND UPDATE
        denoised_1 = model(x_1, sigma_1 * s_in, **extra_args)                   # x_1, sigma_1 = starting point for ralston-radau update
        eps_1 = (x_0 - denoised_1) / sigma_1   # will be for c1 for radau ia 
        #eps_2 = (x_0 - denoised_1) / sigma_2   #if x_next_pred is not None:
        
        
        
        sub_sigma_next = sigma_2
        sub_sigma_up, sub_sigma, sub_sigma_down, sub_alpha_ratio = get_res4lyf_step_with_model(model, sigma, sub_sigma_next, eta_var, noise_mode)
        h_new = h * (sub_sigma_down - sigma) / (sub_sigma_next - sigma) 
        
        x_2 = x_0 + h_new * (a2_1 * eps_1 + a2_2 * eps_2)

        noise = noise_sampler2(sigma=sigma, sigma_next=sigma_next)
        noise = (noise - noise.mean()) / noise.std()
        x_2 = sub_alpha_ratio * x_2 + noise * sub_sigma_up
        
        # K2    MODEL AND UPDATE
        denoised_2 = model(x_2, sigma_2 * s_in, **extra_args) 
        eps_2 = (x_0 - denoised_2) / sigma_2                       # c = 1 maps to 2/3 for radau ia, as in, c2 as well

        x_next_pred = x_0 + h * (b1 * eps_1 + b2 * eps_2)       #this goes to c = 1
        
        noise = noise_sampler(sigma=sigma, sigma_next=sigma_next)
        noise = (noise - noise.mean()) / noise.std()
        x_next_pred = alpha_ratio * x_next_pred + noise * s_noise * sigma_up
        
        x = x_2
        
        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised_2})






        #x_next_pred = x_1 + h * ((1/4) * eps_1 + (3/4) * eps_2)

        eps_1_prev = eps_1
        eps_2_prev = eps_2
        denoised_full_prev = x_0 -sigma * (b1 * eps_1 + b2 * eps_2)
        eps_full_prev = (x_0 - denoised_full_prev) / sigma
        #eps_full_prev = (b1 * eps_1 + b2 * eps_2)
        
    return denoised














def sample_rk_radaucycle_ia(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
    noise_sampler2 = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+10000, sigma_min=0.0, sigma_max=1.0)
    
    x_next_pred = None
    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
        
        sigma_up, sigma, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, eta, noise_mode)
        
        x_0 = x.clone()
        

        
        a1_1, a1_2 = 1/4, -1/4
        a2_1, a2_2 = 1/4, 5/12
        b1, b2 = 1/4, 3/4
        c1, c2 = 0, 2/3
                
        h_no_eta = sigma_next - sigma
        h = sigma_down - sigma
        sigma_1 = sigma + h * c1
        sigma_2 = sigma + h * c2
        
        # radau iia stage
        
        if sigma_next == 0:
            return denoised_2
        
        if x_next_pred == None:
            denoised = model(x, sigma * s_in, **extra_args)
            if sigma_next == 0:
                return denoised
            eps = (x - denoised)/sigma
            
            eps_1 = eps * sigma_1
            eps_2 = eps * sigma_2 
            
            x_1 = x_0 + h * (a1_1 * eps_1 + a1_2 * eps_2)
        else:
            x_1 = x_next_pred
            if extra_options_flag("new_guess", extra_options):
                eps_1 = (x_0 - denoised_prev) / sigma_1   
                eps_2 = (x_0 - denoised_prev) / sigma_2   
                x_1 = x_0 + h * (a1_1 * eps_1 + a1_2 * eps_2)
                
        denoised_1 = model(x_1, sigma_1 * s_in, **extra_args)
        
        if x_next_pred is not None:
            eps_2 = (x_0 - denoised_1) / sigma_2
        eps_1 = (x_0 - denoised_1) / sigma_1                       # will be for c1 for radau ia 
        
        sub_sigma_next = sigma_2
        sub_sigma_up, sub_sigma, sub_sigma_down, sub_alpha_ratio = get_res4lyf_step_with_model(model, sigma, sub_sigma_next, eta_var, noise_mode)
        h_new = h * (sub_sigma_down - sigma) / (sub_sigma_next - sigma) 
        
        x_2 = x_0 + h_new * (a2_1 * eps_1 + a2_2 * eps_2)
        
        noise = noise_sampler2(sigma=sigma, sigma_next=sigma_next)
        noise = (noise - noise.mean()) / noise.std()
        x_2 = sub_alpha_ratio * x_2 + noise * sub_sigma_up
        
        
        denoised_2 = model(x_2, sigma_2 * s_in, **extra_args) 
        eps_2 = (x_0 - denoised_2) / sigma_2                       # c = 1 maps to 2/3 for radau ia, as in, c2 as well

        x_next = x_0 + h * (b1 * eps_1 + b2 * eps_2)
        
        noise = noise_sampler(sigma=sigma, sigma_next=sigma_next)
        noise = (noise - noise.mean()) / noise.std()
        
        # radau ia stage
        if sigma_next > 0:
            a1_1, a1_2 = 5/12, -1/12
            a2_1, a2_2 = 3/4, 1/4
            b1, b2 = 3/4, 1/4
            #c1, c2 = 0, 2/3
            c1, c2 = 0, 2/3
            
            sigma_3 = sigma_next + (sigmas[step+2] - sigma_next) * (1/3)
            sigma_3_down = sigma + h        * (1 + 1/3)
            
            if sigma_3_down >= sigma_3:
                h = sigma_3_down - sigma
                h = h / (1 + 1/3)
            
            x_next_pred = x_0 + h * (b1 * eps_1 + b2 * eps_2)
            
            denoised_prev = x_0 - sigma * (b1 * eps_1 + b2 * eps_2)
            
            if sigma_3_down < sigma_3:
                alpha_ratio_3, sigma_up_3, sigma_down_3 = get_alpha_ratio_from_sigma_down(sigma_3_down, sigma_3, eta)
                
                if extra_options_flag("solve_2m", extra_options):
                    x_next_pred, denoised_prev = res_2m_predictor_step(x_0, denoised_1, denoised_2, sigma_1, sigma_2, sigma_3_down)

                x_next_pred = alpha_ratio_3 * x_next_pred + noise * sigma_up_3

        
        
        x = x_next
        
        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised_2})

        x = alpha_ratio * x + noise * s_noise * sigma_up
        
        

    return denoised








def sample_rk_radau_iia_2s_BS(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    
    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
    noise_sampler2 = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+10000, sigma_min=0.0, sigma_max=1.0)
    
    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
        
        sigma_up, sigma, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, eta, noise_mode)
        
        denoised = model(x, sigma * s_in, **extra_args)
        if sigma_next == 0:
            return denoised
        
        eps = (x - denoised)/sigma
        
        x_0 = x.clone()
                
        h = sigma_down - sigma
        
        a1_1, a1_2 = 5/12, -1/12
        a2_1, a2_2 = 3/4, 1/4
        b1, b2 = 3/4, 1/4
        c1, c2 = 1/3, 1
        
        sigma_1 = sigma + h * c1
        sigma_2 = sigma + h * c2
        sigma_2_BS = sigma + h * (2/3)
        
        eps_1 = eps * sigma_1
        eps_2 = eps * sigma_2 
        
        x_1 = x_0 + h * (a1_1 * eps_1 + a1_2 * eps_2)
                
        denoised_1 = model(x_1, sigma_1 * s_in, **extra_args)
        eps_1 = (x_0 - denoised_1) / sigma_1

        sub_sigma_next = sigma_2_BS
        sub_sigma_up, sub_sigma, sub_sigma_down, sub_alpha_ratio = get_res4lyf_step_with_model(model, sigma, sub_sigma_next, eta_var, noise_mode)
        h_new = h * (sub_sigma_down - sigma) / (sub_sigma_next - sigma) 
        
        #x_2 = x_0 + h_new * (a2_1 * eps_1 + a2_2 * eps_2)
        
        x_2_BS = x_0 + h_new * ((1/4) * eps_1 + (5/12) * eps_2) #coefficients for radau ia 2s, for node 2/3 instead of 1

        noise = noise_sampler2(sigma=sigma, sigma_next=sigma_next)
        noise = (noise - noise.mean()) / noise.std()
        x_2_BS = sub_alpha_ratio * x_2_BS + noise * sub_sigma_up

        #x_2 = x_0 + h * (a2_1 * eps_1 + a2_2 * eps_2)
        
        denoised_2_BS = model(x_2_BS, sigma_2_BS * s_in, **extra_args)
        eps_2 = (x_0 - denoised_2_BS) / sigma_2
        #eps_2 = (x_0 - denoised_2_BS) / sigma_2_BS
        
        #denoised_2 = model(x_2, sigma_2 * s_in, **extra_args)
        #eps_2 = (x_0 - denoised_2) / sigma_2

        x = x_0 + h * (b1 * eps_1 + b2 * eps_2)
        
        noise = noise_sampler(sigma=sigma, sigma_next=sigma_next)
        noise = (noise - noise.mean()) / noise.std()
        x = alpha_ratio * x + noise * s_noise * sigma_up
        
        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised})
            
    return denoised








def sample_rk_radau_iia_2s(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    
    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
    noise_sampler2 = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+10000, sigma_min=0.0, sigma_max=1.0)
    
    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
        
        sigma_up, sigma, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, eta, noise_mode)
        
        denoised = model(x, sigma * s_in, **extra_args)
        if sigma_next == 0:
            return denoised
        
        eps = (x - denoised)/sigma
        
        x_0 = x.clone()
                
        h = sigma_down - sigma
        
        a1_1, a1_2 = 5/12, -1/12
        a2_1, a2_2 = 3/4, 1/4
        b1, b2 = 3/4, 1/4
        c1, c2 = 1/3, 1
        
        sigma_1 = sigma + h * c1
        sigma_2 = sigma + h * c2
        
        eps_1 = eps * sigma_1
        eps_2 = eps * sigma_2 
        
        x_1 = x_0 + h * (a1_1 * eps_1 + a1_2 * eps_2)
                
        denoised_1 = model(x_1, sigma_1 * s_in, **extra_args)
        eps_1 = (x_0 - denoised_1) / sigma_1

        sub_sigma_next = sigma_2
        sub_sigma_up, sub_sigma, sub_sigma_down, sub_alpha_ratio = get_res4lyf_step_with_model(model, sigma, sub_sigma_next, eta_var, noise_mode)
        h_new = h * (sub_sigma_down - sigma) / (sub_sigma_next - sigma) 
        
        x_2 = x_0 + h_new * (a2_1 * eps_1 + a2_2 * eps_2)

        noise = noise_sampler2(sigma=sigma, sigma_next=sigma_next)
        noise = (noise - noise.mean()) / noise.std()
        x_2 = sub_alpha_ratio * x_2 + noise * sub_sigma_up

        #x_2 = x_0 + h * (a2_1 * eps_1 + a2_2 * eps_2)
        
        denoised_2 = model(x_2, sigma_2 * s_in, **extra_args)
        eps_2 = (x_0 - denoised_2) / sigma_2

        x = x_0 + h * (b1 * eps_1 + b2 * eps_2)
        
        noise = noise_sampler(sigma=sigma, sigma_next=sigma_next)
        noise = (noise - noise.mean()) / noise.std()
        x = alpha_ratio * x + noise * s_noise * sigma_up
        
        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised})
            
    return denoised












def sample_rk_radau_iia_3s(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    
    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
    noise_sampler2 = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+10000, sigma_min=0.0, sigma_max=1.0)
    
    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
        
        sigma_up, sigma, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, eta, noise_mode)
        
        denoised = model(x, sigma * s_in, **extra_args)
        if sigma_next == 0:
            return denoised
        
        eps = (x - denoised)/sigma
        
        x_0 = x.clone()
                
        h = sigma_down - sigma
        
        a1_1, a1_2, a1_3 = 11/45 - 7*6**0.5 / 360, 37/225 - 169*6**0.5 / 1800, -2/225 + 6**0.5 / 75
        a2_1, a2_2, a2_3 = 37/225 + 169*6**0.5 / 1800, 11/45 + 7*6**0.5 / 360, -2/225 - 6**0.5 / 75
        a3_1, a3_2, a3_3 = 4/9 - 6**0.5 / 36, 4/9 + 6**0.5 / 36, 1/9
        b1, b2, b3 = 4/9 - 6**0.5 / 36, 4/9 + 6**0.5 / 36, 1/9
        c1, c2, c3 = 2/5 - 6**0.5 / 10, 2/5 + 6**0.5 / 10, 1.
        
        sigma_1 = sigma + h * c1
        sigma_2 = sigma + h * c2
        sigma_3 = sigma + h * c3
        
        eps_1 = eps * sigma_1
        eps_2 = eps * sigma_2 
        eps_3 = eps * sigma_3
        
        x_1 = x_0 + h * (a1_1 * eps_1 + a1_2 * eps_2 + a1_3 * eps_3)
                
        denoised_1 = model(x_1, sigma_1 * s_in, **extra_args)
        eps_1 = (x_0 - denoised_1) / sigma_1

        sub_sigma_next = sigma_2
        sub_sigma_up, sub_sigma, sub_sigma_down, sub_alpha_ratio = get_res4lyf_step_with_model(model, sigma, sub_sigma_next, eta_var, noise_mode)
        h_new = h * (sub_sigma_down - sigma) / (sub_sigma_next - sigma) 
        
        x_2 = x_0 + h_new * (a2_1 * eps_1 + a2_2 * eps_2 + a2_3 * eps_3)

        noise = noise_sampler2(sigma=sigma, sigma_next=sigma_next)
        noise = (noise - noise.mean()) / noise.std()
        x_2 = sub_alpha_ratio * x_2 + noise * sub_sigma_up

        denoised_2 = model(x_2, sigma_2 * s_in, **extra_args)
        eps_2 = (x_0 - denoised_2) / sigma_2




        sub_sigma_next = sigma_3
        sub_sigma_up, sub_sigma, sub_sigma_down, sub_alpha_ratio = get_res4lyf_step_with_model(model, sigma, sub_sigma_next, eta_var, noise_mode)
        h_new = h * (sub_sigma_down - sigma) / (sub_sigma_next - sigma) 
        
        x_3 = x_0 + h_new * (a3_1 * eps_1 + a3_2 * eps_2 + a3_3 * eps_3)

        noise = noise_sampler2(sigma=sigma, sigma_next=sigma_next)
        noise = (noise - noise.mean()) / noise.std()
        x_3 = sub_alpha_ratio * x_3 + noise * sub_sigma_up

        denoised_3 = model(x_3, sigma_3 * s_in, **extra_args) 
        eps_3 = (x_0 - denoised_3) / sigma_3 




        x = x_0 + h * (b1 * eps_1 + b2 * eps_2 + b3 * eps_3)
        
        noise = noise_sampler(sigma=sigma, sigma_next=sigma_next)
        noise = (noise - noise.mean()) / noise.std()
        x = alpha_ratio * x + noise * s_noise * sigma_up
        
        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised})
            
    return denoised











def sample_rk_sphere(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                  cfg1=0, cfg2=0, latent_guide=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    
    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
        
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()
    
    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
        
        if sigma == 1.0:
            sigma = torch.full_like(sigma, 0.9999)
        
        t, t_next = t_fn(torch.clamp(sigma, max=0.9999)), t_fn(sigma_next)
        h = t_next - t
        
        sigma_up, sigma, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, eta, eta_var, noise_mode, h)
        
        t_down = t_fn(sigma_down)
        
        h = t_down - t
        
        sigma_s = sigma_fn(t + h*c2)
        
        h = -torch.log(sigma_down/sigma)
        
        a2_1 = c2 * phi(1, -h*c2)
        b1 =        phi(1, -h) - phi(2, -h)/c2
        b2 =        phi(2, -h)/c2
                
        denoised = model(x, sigma * s_in, **extra_args)
        if sigma_down > 0:
            x_2 = torch.exp(-h * c2) * x + h * (a2_1 * denoised)
            
            denoised_2 = model(x_2, sigma_s * s_in, **extra_args)
            
            x = torch.exp(-h) * x + h * (b1 * denoised + b2 * denoised_2)
            
            x = slerp(denoised_2, x, (sigma_next/sigma))
            #x = slerp(h*(b1 * denoised + b2 * denoised_2), x, (sigma_next/sigma)**c2)
            #x = slerp((sigma_next-sigma)*(b1 * denoised + b2 * denoised_2), x, (sigma_next/sigma))
            #x = slerp(h*(b1 * denoised + b2 * denoised_2), x, h**c2)
            
            if callback is not None:
                callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised})

            noise = noise_sampler(sigma=sigma, sigma_next=sigma_next)
            noise = (noise - noise.mean()) / noise.std()
            x = alpha_ratio * x + noise * s_noise * sigma_up

    return denoised





def sample_rk_momentum(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    
    eps_prev, m_t, v_t, m_t_prev, v_t_prev = [torch.zeros_like(x) for _ in range(5)]

    momentum = float(get_extra_options_kv("momentum", "0.0", extra_options))
    beta1    = float(get_extra_options_kv("beta1", "0.0", extra_options))
    beta2    = float(get_extra_options_kv("beta2", "0.0", extra_options))

    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
    
    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
        
        sigma_up, sigma, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, eta, noise_mode)
        
        h = sigma_down - sigma

        denoised = model(x, sigma * s_in, **extra_args)

        eps = (x - denoised) / sigma
        s_theta = -eps  

        if step == 0:
            eps_prev = eps.clone()

        
        eps_collin = get_collinear (eps, eps_prev)     
        eps_ortho  = get_orthogonal(eps, eps_prev)

        #eps_ortho  = get_orthogonal(eps, eps_collin)
        
        eps_prev_collin = get_collinear (eps_prev, eps)     
        eps_prev_ortho  = get_orthogonal(eps_prev, eps)

        #eps_ortho  = get_orthogonal(eps_prev, eps_collin)

        
        #
        # eps = eps + beta2 * (eps_collin - eps_ortho)
        
        #eps = beta2 * eps + (1-beta2) * (eps_collin + eps_prev_ortho)
        
        eps = (1-beta1) * eps + beta1 * eps_prev

        eps = (1-beta2) * eps + beta2 * (eps_prev_collin + eps_ortho)

            
        x = x + h * eps

        
        noise = noise_sampler(sigma=sigma, sigma_next=sigma_next)
        noise = (noise - noise.mean()) / (noise.std())
        
        x = alpha_ratio * x + sigma_up * noise
        
        eps_prev = eps
        
        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised})

    return denoised





def sample_rk_ralston_2s(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
    

    
    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
        
        sigma_up, sigma, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, eta, noise_mode)
        
        h = sigma_down - sigma
        
        a2_1 = 2/3
        b1, b2 = 1/4, 3/4
        c1, c2 = 0.0, 2/3
        
        a2_1 = 1.0
        b1, b2 = 1/2, 1/2
        c1, c2 = 0.0, 1.0
        
        sigma2 = sigma + h * c2
        
        #h2 = sigma2 - sigma
        
        sub_sigma_up, sub_sigma, sub_sigma_down, sub_alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma2, eta, noise_mode)

        h2 = sub_sigma_down - sigma
        #h_new = h * 

        denoised = model(x, sigma * s_in, **extra_args)

        eps = (x - denoised) / sigma

        x2_eps = x + h2 * (a2_1 * eps)
        
        #x2 = (sub_sigma_down/sigma) * x + (1 - sub_sigma_down/sigma) * denoised
        
        #print("x2_eps - x2", torch.norm(x2 - x2_eps).item())
        
        noise = noise_sampler(sigma=sigma, sigma_next=sigma2)
        noise = (noise - noise.mean()) / (noise.std())
        
        x2 = x2_eps
        x2 = sub_alpha_ratio * x2 + sub_sigma_up * noise
        
        

        denoised2 = model(x2, sigma2 * s_in, **extra_args)
        eps2 = (x - denoised2) / sigma
        if extra_options_flag("use_x2_for_eps2", extra_options):
            eps2 = (x2 - denoised2) / sigma2
        
        x_eps = x + h * (b1 * eps + b2 * eps2)
        
        #x = (sigma_down/sigma) * x + (1 - sigma_down/sigma) * (0.5 * denoised + 0.5 * denoised2)
        
        #print("x_eps - x", torch.norm(x - x_eps).item())
        
        
        noise = noise_sampler(sigma=sigma, sigma_next=sigma_next)
        noise = (noise - noise.mean()) / (noise.std())
        
        x = x_eps
        x = alpha_ratio * x + sigma_up * noise
        
        
        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised})

    return denoised




def sample_rk_implicit_res_2s(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
    
    reverse_weight = float(get_extra_options_kv("reverse_weight", str(0.0), extra_options))
    
    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
        
        sigma_up, sigma, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, eta, noise_mode)
                
        h = -torch.log(sigma_down/sigma)
        
        #s = t + h * c2
        #sigma_s = sigma_fn_x(s)
        s2 = -torch.log(sigma) + h * c2
        sigma_2 = torch.exp(-s2)
        #sigma_2 = torch.exp(-h * c2)
        
        a2_1 = c2 * phi(1, -h*c2)
        b1 =        phi(1, -h) - phi(2, -h)/c2
        b2 =        phi(2, -h)/c2
        
        denoised = model(x, sigma * s_in, **extra_args)
        if sigma_down == 0:
            return denoised
            
        #denoised = model(x, sigma * s_in, **extra_args)
        eps = (denoised - x)
        x_2 = x + h * (a2_1 * eps)
        
        for i in range(iter):
            denoised = model(x_2, sigma_2 * s_in, **extra_args)
            eps = (denoised - x)
            x_2 = x + h * (a2_1 * eps)

        denoised_2 = model(x_2, sigma_2 * s_in, **extra_args)
        eps_2 = (denoised_2 - x)
        x_next = x + h * (b1 * eps + b2 * eps_2)
        
        for i in range(iter):
            denoised_2 = model(x_next, sigma_2 * s_in, **extra_args)
            eps_2 = (denoised_2 - x)
            x_next = x + h * (b1 * eps + b2 * eps_2)

        x = x_next
        
        
        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised})

    return denoised





def sample_rk_res_2m(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
        
    denoised_prev = model(x, sigmas[0] * s_in, **extra_args)
    x = (sigmas[1] / sigmas[0]) * x + (1-sigmas[1] / sigmas[0]) * denoised_prev
    
    x_prev = x
    h_prev = h = -torch.log(sigmas[1] / sigmas[0])
    
    for step in trange(1, len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
        x_0 = x.clone()
        
        h = -torch.log(sigma_next/sigma)
        
        c2 = -h_prev / h

        b1 =        phi(1, -h) - phi(2, -h)/c2
        b2 =        phi(2, -h)/c2

        denoised = model(x, sigma * s_in, **extra_args)
        if sigma_next == 0:
            return denoised
            
        #x = torch.exp(-h) * x + h * (b1 * denoised + b2 * denoised_prev)
        
        eps = denoised - x
        
        if extra_options_flag("against_x_prev", extra_options):
            eps_prev = denoised_prev - x_prev
        else:
            eps_prev = denoised_prev - x
        
        #x = torch.exp(-h) * x + h * (b1 * denoised + b2 * denoised_prev)
        
        x = x + h * (b1 * eps + b2 * eps_prev)
        
        print(step, h.item(), c2.item(), b1.item(), b2.item())
                
        h_prev = h
        denoised_prev = denoised
        x_prev = x_0
        
        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised})

    return denoised





def sample_rk_res_2m_prenoise_alt(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
        
    brownian_main_start = get_extra_options_kv("brownian_main_start", "sigma",    extra_options)
    brownian_main_stop  = get_extra_options_kv("brownian_main_stop",  "sigma_next", extra_options)
        
    denoised_prev = model(x, sigmas[0] * s_in, **extra_args)
    x = (sigmas[1] / sigmas[0]) * x + (1-sigmas[1] / sigmas[0]) * denoised_prev
    
    x_prev = x
    h_prev = h = -torch.log(sigmas[1] / sigmas[0])
    sigma_prev = sigmas[0]
    eps_prev = denoised_prev - x_prev
    eps_prev2 = None
    c2_prev = 1/2
    b1_prev, b2_prev = None, None
    
    for step in trange(1, len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
        sigma_up, sigma, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, eta, noise_mode)  
        x_0 = x.clone()
           
        h = -torch.log(sigma_next/sigma)
        
        c2 = -h_prev / h

        b1 =        phi(1, -h) - phi(2, -h)/c2
        b2 =        phi(2, -h)/c2

        denoised = model(x_0, sigma * s_in, **extra_args)
        if sigma_next == 0:
            return denoised
                    
        eps = denoised - x_0
        eps_prev = denoised_prev - x_0 #x_prev
        
        if extra_options_flag("use_bong1_update0a0", extra_options):
            φ = Phi(h_prev, [0.,c2_prev])
            a2_1 = c2_prev * φ(1,2)
            for i in range(100):
                x_prev = x_0 - h_prev * (a2_1 * eps)
                eps = denoised - x_prev
        
        if extra_options_flag("use_bong1_update0a1", extra_options):
            φ = Phi(h_prev, [0.,c2_prev])
            a2_1 = c2_prev * φ(1,2)
            for i in range(100):
                x_prev = x_0 - h_prev * (a2_1 * eps_prev)
                eps_prev = denoised_prev - x_prev
                
        if extra_options_flag("use_bong1_update0a2", extra_options):
            φ = Phi(h, [0.,c2])
            a2_1 = c2 * φ(1,2)
            for i in range(100):
                x_prev = x_0 - h * (a2_1 * eps_prev)
                eps_prev = denoised_prev - x_prev
                
        if extra_options_flag("use_bong1_update0a3", extra_options):
            φ = Phi(h, [0.,c2])
            a2_1 = c2 * φ(1,2)
            for i in range(100):
                x_prev = x_0 - h * (a2_1 * eps_prev)
                eps_prev =((denoised_prev - x_prev) + (denoised_prev - x_0)) / 2
                
        if extra_options_flag("use_bong1_update0b", extra_options) and eps_prev2 is not None:
            φ = Phi(h_prev, [0.,0.5])
            a2_1 = 0.5 * φ(1,2)
            for i in range(100):
                x_prev = x_0 - h_prev * (b1_prev * eps_prev + b2_prev * eps_prev2)
                eps_prev = denoised_prev - x_prev
                
        if extra_options_flag("use_bong1_update0c", extra_options) and eps_prev2 is not None:
            φ = Phi(h_prev, [0.,0.5])
            a2_1 = 0.5 * φ(1,2)
            for i in range(100):
                x_prev2 = x_prev - h_prev * (b1_prev * eps_prev + b2_prev * eps_prev2)
                eps_prev = denoised_prev - x_prev
                
        if extra_options_flag("use_bong1_update0d", extra_options): # and eps_prev2 is not None:
            for i in range(100):
                x_prev = x_0 - h * (b1 * eps + b2 * eps_prev)
                eps_prev = denoised_prev - x_prev
                
        if extra_options_flag("use_bong1_update0e", extra_options): # and eps_prev2 is not None:
            for i in range(100):
                x_prev = x_0 - h_prev * (b1_prev * eps_prev + b2_prev * eps_prev2)
                eps_prev = denoised_prev - x_prev
                
        if extra_options_flag("use_bong1_update0f", extra_options): 
            eps_prev = (x_0 - denoised_prev) / (sigma + torch.exp(-( h * a2_1)))
            x_prev = denoised_prev + sigma*eps_prev
        
        if extra_options_flag("use_bong1_update1", extra_options):
            φ = Phi(h_prev, [0.,0.5])
            a2_1 = 0.5 * φ(1,2)
            for i in range(100):
                x_0 = x_next - h * (b1 * eps + b2 * eps_prev)
                eps_prev = denoised_prev - x_prev
        
        x_next = x_0 + h * (b1 * eps + b2 * eps_prev)
        
        eps_next = (x_0 - x_next) / (sigma - sigma_next)
        denoised_next = x_0 - sigma * eps_next
        s_dict = {"sigma": sigma, "sigma_next": sigma_next, "sigma_prev": sigma_prev, "sigma_down": sigma_down, "sigma_up": sigma_up}
        noise = noise_sampler(sigma=s_dict[brownian_main_start], sigma_next=s_dict[brownian_main_stop])
        x_next = alpha_ratio * (denoised_next + sigma_down * eps_next) + sigma_up * noise   
        
        
        if extra_options_flag("use_bong1_update2", extra_options):
            #φ = Phi(h_prev, [0.,0.5])
            #a2_1 = 0.5 * φ(1,2)
            for i in range(100):
                #x_0 = x_next - h * (b1 * eps + b2 * eps_prev)
                x_prev = x_0 - h * (b1 * eps + b2 * eps_prev)
                eps_prev = denoised_prev - x_prev
            x_next = x_0 + h * (b1 * eps + b2 * eps_prev)
        
        if extra_options_flag("use_bong1_update3", extra_options) and eps_prev2 is not None:
            #φ = Phi(h_prev, [0.,0.5])
            #a2_1 = 0.5 * φ(1,2)
            for i in range(100):
                #x_0 = x_next - h * (b1 * eps + b2 * eps_prev)
                x_prev = x_0 - h_prev * (b1 * eps_prev + b2 * eps_prev2)
                eps_prev = denoised_prev - x_prev
            x_next = x_0 + h * (b1 * eps + b2 * eps_prev)
            
        if extra_options_flag("use_bong1_update4", extra_options) and eps_prev2 is not None:
            #φ = Phi(h_prev, [0.,0.5])
            #a2_1 = 0.5 * φ(1,2)
            for i in range(100):
                #x_0 = x_next - h * (b1 * eps + b2 * eps_prev)
                x_prev = x_0 - h_prev * (b1_prev * eps_prev + b2_prev * eps_prev2)
                eps_prev = denoised_prev - x_prev
            x_next = x_0 + h * (b1 * eps + b2 * eps_prev)
        
        
        
        if extra_options_flag("use_bong1_update5", extra_options) and eps_prev2 is not None:
            #φ = Phi(h_prev, [0.,0.5])
            #a2_1 = 0.5 * φ(1,2)
            for i in range(100):
                #x_0 = x_next - h * (b1 * eps + b2 * eps_prev)
                x_0 = x_next - h * (b1 * eps + b2 * eps_prev)
                eps = denoised - x_0
            #x_next = x_0 + h * (b1 * eps + b2 * eps_prev)
        
        h_prev2 = h_prev
        h_prev = h
        
        eps_prev2 = eps_prev
        eps_prev = eps
        
        denoised_prev2 = denoised_prev
        denoised_prev = denoised
        
        x_prev2 = x_prev
        x_prev = x_0
        
        b1_prev2, b2_prev2 = b1_prev, b2_prev
        b1_prev, b2_prev = b1, b2
        
        c2_prev2 = c2_prev
        c2_prev = c2
        
        x = x_next
        
        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised})

    return denoised









def sample_rk_res_2m_prenoise_alt2(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
        
    brownian_main_start = get_extra_options_kv("brownian_main_start", "sigma",    extra_options)
    brownian_main_stop  = get_extra_options_kv("brownian_main_stop",  "sigma_next", extra_options)
        
    x_0 = x.clone()
    denoised_prev = model(x, sigmas[0] * s_in, **extra_args)
    x = (sigmas[1] / sigmas[0]) * x + (1-sigmas[1] / sigmas[0]) * denoised_prev
    
    x_prev = x
    h_prev = h = -torch.log(sigmas[1] / sigmas[0])
    sigma_prev = sigmas[0]
    eps_prev = denoised_prev - x_prev
    eps_prev2 = None
    c2_prev = 1/2
    b1_prev, b2_prev = None, None
    
    denoised_2 = denoised_prev
    
    
    for step in trange(1, len(sigmas)-1, disable=disable):
        sigma, sigma_2, sigma_next = sigmas[step-1], sigmas[step], sigmas[step+1]
        #sigma_up, sigma, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, eta, noise_mode)  
        x_2 = x.clone()
        denoised = denoised_2
           
        h = -torch.log(sigma_next/sigma)
        
        t  = -torch.log(sigma)
        t2 = -torch.log(sigma_2)
        
        c2 = (t2 - t) / h
        
        
        s2 = torch.exp(-  (-torch.log(sigma) + h * c2))
        
        
        
        #c2 = -h_prev / h

        b1 =        phi(1, -h) - phi(2, -h)/c2
        b2 =        phi(2, -h)/c2

        denoised_2 = model(x_2, sigma_2 * s_in, **extra_args)
        if sigma_next == 0:
            return denoised_2

                    
        eps = denoised - x_0
        eps_2 = denoised_2 - x_0 #x_prev

        x_next = x_0 + h * (b1 * eps + b2 * eps_2)
        
        #eps_next = (x_0 - x_next) / (sigma - sigma_next)
        #denoised_next = x_0 - sigma * eps_next
        #x_next = alpha_ratio * (denoised_next + sigma_down * eps_next) + sigma_up * noise   
        
        
        h_prev2 = h_prev
        h_prev = h
        
        eps_prev2 = eps_prev
        eps_prev = eps
        
        denoised_prev2 = denoised_prev
        denoised_prev = denoised
        
        x_prev2 = x_prev
        x_prev = x_0
        
        b1_prev2, b2_prev2 = b1_prev, b2_prev
        b1_prev, b2_prev = b1, b2
        
        c2_prev2 = c2_prev
        c2_prev = c2
        
        x_0 = x
        x = x_next
        
        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised})

    return denoised














def sample_rk_res_2m_nonstandard_prenoise_alt(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
        
    brownian_main_start = get_extra_options_kv("brownian_main_start", "sigma",    extra_options)
    brownian_main_stop  = get_extra_options_kv("brownian_main_stop",  "sigma_next", extra_options)
        
    denoised_prev = model(x, sigmas[0] * s_in, **extra_args)
    x = (sigmas[1] / sigmas[0]) * x + (1-sigmas[1] / sigmas[0]) * denoised_prev
    
    x_prev = x
    h_prev = h = -torch.log(sigmas[1] / sigmas[0])
    sigma_prev = sigmas[0]
    eps_prev = denoised_prev - x_prev
    eps_prev2 = None
    c2_prev = 1/2
    b1_prev, b2_prev = None, None
    
    for step in trange(1, len(sigmas)-1, disable=disable):
        sigma_prev, sigma, sigma_next = sigmas[step-1], sigmas[step], sigmas[step+1]
        sigma_up, sigma_prev, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma_prev, sigma_next, eta, noise_mode)  
        x_0 = x.clone()
        
        if sigma_next == 0:
            denoised = model(x_0, sigma * s_in, **extra_args)
            x = denoised
            break
        
        h = -torch.log(sigma_next/sigma_prev)
        
        t2 = -torch.log(sigma)
        t0 = -torch.log(sigma_prev)
        c2 = (t2 - t0) / h
        
        #c2 = -h_prev / h

        φ = Phi(h, [0.,c2])
        a2_1 = c2*φ(1,2)
        b2   = φ(2)/c2
        b1   = φ(1)-b2
        
        #x_0 = x_prev + h * (a2_1 * eps_prev)
        eps_prev = denoised_prev - x_prev
        
        if extra_options_flag("use_bong1_update4", extra_options):
            for i in range(100):
                x_prev = x_0 - h * (a2_1 * eps_prev)
                eps_prev = denoised_prev - x_0
                
        if extra_options_flag("use_bong1_update5", extra_options):
            for i in range(100):
                x_prev = x_0 - h * (a2_1 * eps_prev)
                eps_prev = denoised_prev - x_prev
                
        if extra_options_flag("use_bong1_update0f", extra_options): 
            eps_prev = (x_0 - denoised_prev) / (sigma + torch.exp(-( h * a2_1)))
            x_prev = denoised_prev + sigma*eps_prev

        denoised = model(x_0, sigma * s_in, **extra_args)
        
        eps = denoised - x_prev

        x_next = x_prev + h * (b1 * eps_prev + b2 * eps)
        
        eps_next = (x_prev - x_next) / (sigma_prev - sigma_next)
        denoised_next = x_prev - sigma_prev * eps_next
        s_dict = {"sigma": sigma, "sigma_next": sigma_next, "sigma_prev": sigma_prev, "sigma_down": sigma_down, "sigma_up": sigma_up}
        noise = noise_sampler(sigma=s_dict[brownian_main_start], sigma_next=s_dict[brownian_main_stop])
        x_next = alpha_ratio * (denoised_next + sigma_down * eps_next) + sigma_up * noise   

        h_prev2 = h_prev
        h_prev = h
        
        eps_prev2 = eps_prev
        eps_prev = eps
        
        denoised_prev2 = denoised_prev
        denoised_prev = denoised
        
        x_prev2 = x_prev
        x_prev = x_0
        
        b1_prev2, b2_prev2 = b1_prev, b2_prev
        b1_prev, b2_prev = b1, b2
        
        c2_prev2 = c2_prev
        c2_prev = c2
        
        x = x_next
        
        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised})

    return x




def jvp_finite_difference(
    func: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    v: torch.Tensor,
    h: float = 1e-5
) -> torch.Tensor:
    """
    Compute the Jacobian-vector product (JVP) using finite differences.
    """
    f_plus = func(x + h * v)
    f_minus = func(x - h * v)
    return (f_plus - f_minus) / (2 * h)

def randomized_low_rank_jacobian(
    func: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    k: int = 2,  # Adjusted for small test case
    num_seeds: int = 1,
    h: float = 1e-5
) -> tuple:
    """
    Estimate a low-rank approximation of the Jacobian matrix using Randomized SVD.
    Returns U_mean, S_mean, Vh_mean for the low-rank approximation.
    """
    m = func(x).numel()
    n = x.numel()
    k = int(k)  # Ensure k is integer
    
    # Initialize accumulators
    U_total = torch.zeros(m, k, device=x.device)
    S_total = torch.zeros(k, device=x.device)
    V_total = torch.zeros(k, n, device=x.device)
    
    for seed in range(num_seeds):
        # Generate random projection matrix Omega (n x k)
        Omega = torch.randn(n, k, device=x.device)
        
        # Compute Y = J * Omega using finite differences
        Y = torch.zeros(m, k, device=x.device)
        for i in range(k):
            omega_i = Omega[:, i]
            Jv = jvp_finite_difference(func, x, omega_i, h=h)
            Y[:, i] = Jv
        
        # Perform SVD on Y
        U, S, Vh = torch.linalg.svd(Y, full_matrices=False)
        
        # Accumulate
        U_total += U
        S_total += S
        V_total += Vh
    
    # Average over seeds
    U_mean = U_total / num_seeds
    S_mean = S_total / num_seeds
    Vh_mean = V_total / num_seeds
    
    return U_mean, S_mean, Vh_mean

def low_rank_jvp(U: torch.Tensor, S: torch.Tensor, Vh: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Compute J * v using the low-rank approximation J ≈ U S V^H.
    """
    return U @ (S * (Vh @ v))


"""def sample_rk_randomized_svd(
    model: Callable[[torch.Tensor, float, dict], torch.Tensor],
    x: torch.Tensor,
    sigmas: torch.Tensor,
    extra_args: dict = {},
    callback=None,
    disable: bool = False,
    noise_sampler=None,
    eta: float = 0.5,
    s_noise: float = 1.0,
    number_implicit_refinement_cycles: int = 2,
    k: int = 10,  # Keep k small (e.g., 10)
    num_seeds: int = 1,
    h: float = 1e-5
) -> torch.Tensor:"""
def sample_rk_randomized_svd(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None):
    """
    High-dimensional sampler using randomized SVD-based Jacobian estimation.
    """
    if extra_args is None:
        extra_args = {}
    if callback is None:
        callback = lambda info: None  # No-op callback
    if noise_sampler is None:
        noise_sampler = torch.randn_like  # Default noise sampler
    s_in = x.new_ones([x.shape[0]])
    number_implicit_refinement_cycles = int(get_extra_options_kv("number_implicit_refinement_cycles", str("0"), extra_options))

    for step in trange(len(sigmas)-1, disable=disable, desc="Sampling"):
        sigma = sigmas[step]
        sigma_next = sigmas[step+1]
        h_step = sigma_next - sigma  # Negative step to ensure descent

        # Explicit Euler Step
        denoised = model(x, sigma * s_in, **extra_args)
        eps = (x - denoised) / sigma
        x_next = x + h_step * eps  # Explicit update

        # Implicit Refinement
        for cycle in range(number_implicit_refinement_cycles):
            # Define the residual function as a closure
            def residual_func(x_new):
                denoised_new = model(x_new, sigma_next * s_in, **extra_args)
                return (x - denoised_new) / sigma_next - (x_new - x) / h_step

            # Estimate Jacobian using Randomized SVD
            U, S, Vh = randomized_low_rank_jacobian(
                func=residual_func,
                x=x_next,
                k=k,
                num_seeds=2,
                h=1e-5
            )

            # Define a function to compute J * v using low-rank approximation
            def J_times_v(v):
                return low_rank_jvp(U, S, Vh, v)

            # Implement an iterative solver that uses J_times_v
            # Using Conjugate Gradient-like loop
            delta = torch.zeros_like(x)
            residual = residual_func(delta)
            r = -residual.clone()  # Initial residual
            p = r.clone()
            rsold = torch.dot(r, r)
            
            for i in range(50):  # max_iter=50
                J_p = J_times_v(p)
                p_dot_J_p = torch.dot(p, J_p)
                if p_dot_J_p == 0:
                    break  # Prevent division by zero
                alpha = rsold / p_dot_J_p
                delta += alpha * p
                r -= alpha * J_p
                rsnew = torch.dot(r, r)
                if torch.sqrt(rsnew) < 1e-4:
                    break
                p = r + (rsnew / rsold) * p
                rsold = rsnew

            # Update x_next
            x_next = x_next + delta

        # Add noise
        #noise = noise_sampler(x_next) * s_noise * eta
        #x_next = x_next + noise

        # Update for next iteration
        x = x_next

        # Callback for monitoring
        if callback is not None:
            callback({
                'x': x,
                'i': step,
                'sigma': sigma,
                'sigma_next': sigma_next,
                'denoised': denoised
            })

    return denoised






def sample_rk_ddim_test(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
        

    for step in trange(1, len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
           
        denoised = model(x, sigma * s_in, **extra_args)
        if sigma_next == 0:
            return denoised
                    
        eps = (x - denoised) / sigma

        x = (sigma_next/sigma) * x   +   (1 - sigma_next/sigma) * eps * torch.sqrt(1-sigma**2)        
        
        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised})

    return denoised



def extract_pred(x_before, x_after, sigma_before, sigma_after):
    alpha = sigma_after / sigma_before
    return (x_after - alpha * x_before) / (1 - alpha)



def sample_rk_res_2s_downswap(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    sigmas = sigmas.to(torch.float64)
    x = x.to(torch.float64)

    h_fn = lambda sigma_down, sigma: -torch.log(sigma_down/sigma)
    
    sigma_min = model.inner_model.inner_model.model_sampling.sigma_min.to(x.dtype)
    sigma_max = model.inner_model.inner_model.model_sampling.sigma_max.to(x.dtype)
    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=sigma_min, sigma_max=sigma_max)
    noise_sampler2 = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+10000, sigma_min=sigma_min, sigma_max=sigma_max)

    
    substep_eta = float(get_extra_options_kv("substep_eta", str(eta), extra_options))
    
    brownian_sub_start = get_extra_options_kv("brownian_sub_start", "sigma",    extra_options)
    brownian_sub_stop  = get_extra_options_kv("brownian_sub_stop",  "sigma_2", extra_options)
    brownian_main_start = get_extra_options_kv("brownian_main_start", "sigma",    extra_options)
    brownian_main_stop  = get_extra_options_kv("brownian_main_stop",  "sigma_next", extra_options)
            
    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
        sigma_up, sigma, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, eta, noise_mode)  
        
        x_0 = x.clone()
           
        c1,c2 = 0, 1/2
        ci = [c1,c2]
        h = -torch.log(sigma_down/sigma)
        φ = Phi(h, ci)
        
        a2_1 = c2 * φ(1,2)
        b2 = φ(2)/c2
        b1 = φ(1) - b2
        
        s2 = -torch.log(sigma) + h * c2
        sigma_2 = torch.exp(-s2)

        denoised = model(x, sigma * s_in, **extra_args)
        if sigma_next == 0:
            x = denoised
            break
        
        eps = denoised - x
        x_2 = x + h * (a2_1 * eps)


        sub_sigma_next = sigma_2
        sub_sigma_up, sub_sigma, sub_sigma_down, sub_alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_2, substep_eta, noise_mode)       
        s_dict = {"sigma": sigma, "sigma_next": sigma_next, "sigma_2": sigma_2, "sigma_down": sigma_down, "sigma_up": sigma_up, "sigma_up_2": sub_sigma_up, "sigma_down_2": sub_sigma_down}

        h_new = h * h_fn(sub_sigma_down, sigma) / h_fn(sub_sigma_next, sigma) 

        x_2 = x + h_new * (a2_1 * eps)
        noise = noise_sampler(sigma=s_dict[brownian_sub_start], sigma_next=s_dict[brownian_sub_stop])
        x_2 = sub_alpha_ratio * x_2 + sub_sigma_up * noise

        
        denoised_2 = model(x_2, sigma_2 * s_in, **extra_args)
        eps_2 = denoised_2 - x

        x_down = x + h * (b1 * eps + b2 * eps_2)
        
        noise = noise_sampler(sigma=s_dict[brownian_main_start], sigma_next=s_dict[brownian_main_stop])
        
        #x = alpha_ratio * x + sigma_up * noise
        
        eps_next = (x_0 - x_down) / (sigma - sigma_down)
        denoised_next = x_0 - sigma * eps_next
        x = alpha_ratio * (denoised_next + sigma_down * eps_next) + sigma_up * noise * s_noise

        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised})

    return x




def sample_rk_res_2s_scaled(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None, latent_guide_inv=None, mask=None, noise_initial=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    sigmas = sigmas.to(torch.float64)
    x = x.to(torch.float64)

    h_fn = lambda sigma_down, sigma: -torch.log(sigma_down/sigma)
    
    sigma_min = model.inner_model.inner_model.model_sampling.sigma_min.to(x.dtype)
    sigma_max = model.inner_model.inner_model.model_sampling.sigma_max.to(x.dtype)
    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=sigma_min, sigma_max=sigma_max)
    noise_sampler2 = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+10000, sigma_min=sigma_min, sigma_max=sigma_max)

    
    substep_eta = float(get_extra_options_kv("substep_eta", str(eta), extra_options))
    
    brownian_sub_start = get_extra_options_kv("brownian_sub_start", "sigma",    extra_options)
    brownian_sub_stop  = get_extra_options_kv("brownian_sub_stop",  "sigma_2", extra_options)
    brownian_main_start = get_extra_options_kv("brownian_main_start", "sigma",    extra_options)
    brownian_main_stop  = get_extra_options_kv("brownian_main_stop",  "sigma_next", extra_options)
            
    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
        #sigma_up, sigma, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, eta, noise_mode)  
        
        x_0 = x.clone()
        
        c1,c2 = 0, 1/2
        ci = [c1,c2]
        #h = -torch.log(sigma_down/sigma)
        
        h = -torch.log(sigma_next/sigma)
        φ = Phi(h, ci)
        
        a2_1 = c2 * φ(1,2)
        b2 = φ(2)/c2
        b1 = φ(1) - b2
        
        s2 = -torch.log(sigma) + h * c2
        sigma_2 = torch.exp(-s2)
        
        print("before:", h.item(), a2_1.item(), b1.item(), b2.item(), flush=True)

        
        if extra_options_flag("exp2lin", extra_options):
            a2_1 *= -sigma
            b1   *= -sigma
            b2   *= -sigma
            
            print("after:", h.item(), a2_1.item(), b1.item(), b2.item(), flush=True)
        
        
        if extra_options_flag("h2h", extra_options):
            h_exp = -torch.log(sigma_next/sigma)
            h = sigma_next - sigma
            
            a2_1 *= -sigma * h_exp/h
            b1   *= -sigma * h_exp/h
            b2   *= -sigma * h_exp/h
            
            print("after:", h.item(), a2_1.item(), b1.item(), b2.item(), flush=True)

        denoised = model(x, sigma * s_in, **extra_args)
        if sigma_next == 0:
            x = denoised
            break
        
        if extra_options_flag("exp2lin", extra_options) or extra_options_flag("h2h", extra_options):
            eps = (x - denoised) / sigma
        else:
            eps = denoised - x
        x_2 = x + h * (a2_1 * eps)

        denoised_2 = model(x_2, sigma_2 * s_in, **extra_args)
        if extra_options_flag("exp2lin", extra_options) or extra_options_flag("h2h", extra_options):
            eps_2 = (x - denoised_2) / sigma
        else:
            eps_2 = denoised_2 - x

        x = x + h * (b1 * eps + b2 * eps_2)

        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised})

    return x



def sample_rk_res_2m_scaled(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None, latent_guide_inv=None, mask=None, noise_initial=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
        
    denoised_prev = model(x, sigmas[0] * s_in, **extra_args)
    x = (sigmas[1] / sigmas[0]) * x + (1-sigmas[1] / sigmas[0]) * denoised_prev
    
    x_prev = x
    h_prev = h = -torch.log(sigmas[1] / sigmas[0])
    
    for step in trange(1, len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
        x_0 = x.clone()
        
        h = -torch.log(sigma_next/sigma)
        
        c2 = -h_prev / h

        b1 =        phi(1, -h) - phi(2, -h)/c2
        b2 =        phi(2, -h)/c2
        
        print("before:", h.item(), b1.item(), b2.item(), flush=True)
        if extra_options_flag("h2h", extra_options):
            h_exp = -torch.log(sigma_next/sigma)
            h = sigma_next - sigma
            
            b1   *= -sigma * h_exp/h
            b2   *= -sigma * h_exp/h
            
            print("after:", h.item(), b1.item(), b2.item(), flush=True)

        denoised = model(x, sigma * s_in, **extra_args)
        if sigma_next == 0:
            return denoised
            
        #x = torch.exp(-h) * x + h * (b1 * denoised + b2 * denoised_prev)
        
        if extra_options_flag("exp2lin", extra_options) or extra_options_flag("h2h", extra_options):
            eps = (x - denoised) / sigma
        else:
            eps = denoised - x
        
        if extra_options_flag("against_x_prev", extra_options):
            eps_prev = denoised_prev - x_prev
        elif extra_options_flag("exp2lin", extra_options) or extra_options_flag("h2h", extra_options):
            eps_prev = (x - denoised_prev) / sigma
        else:
            eps_prev = denoised_prev - x
        
        #x = torch.exp(-h) * x + h * (b1 * denoised + b2 * denoised_prev)
        
        x = x + h * (b1 * eps + b2 * eps_prev)
        
        #print(step, h.item(), c2.item(), b1.item(), b2.item())
                
        if extra_options_flag("exp2lin", extra_options) or extra_options_flag("h2h", extra_options):
            h_prev = h_exp
        else:
            h_prev = h
        denoised_prev = denoised
        x_prev = x_0
        
        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised})

    return denoised





def sample_rk_vptest3(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None, latent_guide_inv=None, mask=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    sigmas = sigmas.to(torch.float64)
    x = x.to(torch.float64)

    h_fn = lambda sigma_down, sigma: -torch.log(sigma_down/sigma)
    
    sigma_min = model.inner_model.inner_model.model_sampling.sigma_min.to(x.dtype)
    sigma_max = model.inner_model.inner_model.model_sampling.sigma_max.to(x.dtype)
    
    denoised = model(x, sigmas[0] * s_in, **extra_args)
    eps = (x - denoised) / sigmas[0]
    
    x = x + (sigmas[1]-sigmas[0]) * eps

    #x = x / torch.sqrt(1.0 + sigmas[1] ** 2.0)
    x = scale_in(x, sigmas[1])

    for step in trange(1, len(sigmas)-1, disable=disable):
        sigma_prev, sigma, sigma_next = sigmas[step-1], sigmas[step], sigmas[step+1]
        denoised_prev = denoised

        h = torch.log(sigma_next/sigma)
        x = (sigma_next/sigma) * x    +   (1 - sigma_next/sigma) * denoised_prev
        denoised = model(x, sigma_next * s_in, **extra_args)

        #denoised = model(x* ((sigma ** 2 + 1.0) ** 0.5)   , sigma * s_in, **extra_args)
        #denoised = model(scale_out(x, sigma), sigma * s_in, **extra_args)
        if sigma_next == 0:
            x = denoised
            break
        
        D1_t = (denoised - denoised_prev)
        #x = (sigma_next/sigma) * x +  (1 - sigma_next/sigma) * (0.0 + 0.5 * D1_t)    # (corr_res + rhos_c[-1] * D1_t)
        x = x +  (1 - sigma_next/sigma) * (0.0 + 0.5 * D1_t)    # (corr_res + rhos_c[-1] * D1_t)

        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised})

    return x


"""
def marginal_log_mean_coeff(sigma):
    return torch.log(1.0 - sigma)
def marginal_alpha(sigma): return 1 - sigma
def marginal_std(sigma):   return sigma
def marginal_lambda(sigma): return torch.log((1 - sigma) / sigma)
"""


def sample_rk_unibutt(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None, latent_guide_inv=None, mask=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    sigmas = sigmas.to(torch.float64)
    x = x.to(torch.float64)

    h_fn = lambda sigma_down, sigma: -torch.log(sigma_down/sigma)
    
    sigma_min = model.inner_model.inner_model.model_sampling.sigma_min.to(x.dtype)
    sigma_max = model.inner_model.inner_model.model_sampling.sigma_max.to(x.dtype)
    
    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=sigma_min, sigma_max=sigma_max)
    
    if sigmas[0] == 1.0:
        sigmas[0] = 0.9999
    
    denoised = model(x, sigmas[0] * s_in, **extra_args)
    
    full_order = 3
    
    model_prev_list = full_order * [denoised]
    t_prev_list     = full_order * [sigmas[0].unsqueeze(0)]

    x = scale_in(x, sigmas[0])

    for step in trange(1, len(sigmas)-1, disable=disable):
        order = min(step, full_order)
        sigma_prev, sigma, sigma_next = sigmas[step-1].unsqueeze(0), sigmas[step].unsqueeze(0), sigmas[step+1].unsqueeze(0)
        
        sigma_up, sigma_prev, sigma, alpha_ratio = get_res4lyf_step_with_model(model, sigma_prev, sigma, eta, noise_mode)
        
        D = torch.zeros((x.shape[0], order, *x.shape[1:])).to(x)
        r = torch.ones((order)).to(x)
        R = torch.zeros((order,order)).to(x)
        b = torch.zeros((order)).to(x)

        t_prev                      = t_prev_list[-1]
        lambda_prev                 = marginal_lambda(t_prev)
        lambda_t                    = marginal_lambda(sigma)
        model_prev                  = model_prev_list[-1]
        sigma_prev,         sigma_t = marginal_std(t_prev),            marginal_std(sigma)
        log_alpha_prev, log_alpha_t = marginal_log_mean_coeff(t_prev), marginal_log_mean_coeff(sigma)
        alpha_t                     = torch.exp(log_alpha_t)

        h = lambda_t - lambda_prev
        
        for i in range(1, order):
            model_prev_i  =             model_prev_list[-(i+1)]
            lambda_prev_i = marginal_lambda(t_prev_list[-(i+1)])
            r  [i-1] = ((lambda_prev_i - lambda_prev) / h)[0]
            D[:,i-1] =   (model_prev_i - model_prev)  / r[i-1]

        h_phi_1 = torch.expm1(-h) # h\phi_1(h) = e^h - 1
        h_phi_k = h_phi_1/-h - 1

        #if   self.variant == 'bh1':
        #B_h = -h
        #elif self.variant == 'bh2':
        B_h = h_phi_1

        factorial_i = 1
        for i in range(1, order+1):
            R[i-1]       = torch.pow(r, i-1)
            b[i-1]       = h_phi_k * factorial_i/B_h
            factorial_i *= (i+1)
            h_phi_k      = h_phi_k/-h - 1/factorial_i

        rhos_p, rhos_c = [torch.tensor([0.5], device=x.device) for _ in range(2)]
        if order > 2:
            rhos_p = torch.linalg.solve(R[:-1, :-1], b[:-1])
        if order > 1 and step < len(sigmas)-1:
            rhos_c = torch.linalg.solve(R, b)

        x_0 = (sigma_t/sigma_prev) * x    -    (alpha_t * h_phi_1) * model_prev
        
        #noise = noise_sampler(sigma=sigma_prev, sigma_next=sigma)
        #x_0 = alpha_ratio * x_0 + sigma_up * noise

        x = x_0 - (alpha_t * B_h) * torch.einsum('k,bkchw->bchw', rhos_p, D[:,:-1])

        #noise = noise_sampler(sigma=sigma_prev, sigma_next=sigma)
        #x = alpha_ratio * x + sigma_up * noise

        model_t = model(x, sigmas[step].unsqueeze(0), **extra_args)
        D[:,-1] = model_t - model_prev
        x = x_0 - (alpha_t * B_h) * torch.einsum('k,bkchw->bchw', rhos_c, D)

        for i in range(len(t_prev_list)-1):
            t_prev_list    [i] =     t_prev_list[i+1]
            model_prev_list[i] = model_prev_list[i+1]
        
        model_prev_list[-1] = model_t
        t_prev_list    [-1] = sigmas[step].unsqueeze(0)
        
        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': model_t})
            
        noise = noise_sampler(sigma=sigma_prev, sigma_next=sigma)
        x = alpha_ratio * x + sigma_up * noise
        
    x = scale_out(x, sigmas[-1])
    return x


def scale_in(x, sigma):
    return x / scale_sigma(sigma)
    
def scale_out(x, sigma):
    return x * scale_sigma(sigma)

def scale_sigma(sigma):
    return torch.ones_like(sigma)
    #return (sigma ** 2 + 1.0) ** 0.5

def marginal_alpha(sigma):           # 1 - sigma
    return 1. - sigma

def marginal_std(sigma):             # sigma
    return sigma

def marginal_log_mean_coeff(sigma):  # log(1 - sigma)
    return torch.log(1. - sigma)

def marginal_lambda(sigma):          # log((1 - sigma) / sigma)
    #return -torch.log(sigma)
    return torch.log((1. - sigma) / sigma)





### WE HAVE A WORKING STARTING POINT!!!

def sample_rk_vptest(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None, latent_guide_inv=None, mask=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    sigmas = sigmas.to(torch.float64)
    x = x.to(torch.float64)

    h_fn = lambda sigma_down, sigma: -torch.log(sigma_down/sigma)
    
    sigma_min = model.inner_model.inner_model.model_sampling.sigma_min.to(x.dtype)
    sigma_max = model.inner_model.inner_model.model_sampling.sigma_max.to(x.dtype)
    
    if sigmas[0] == 1.0:
        sigmas[0] = 0.9999
    
    denoised = model(x, sigmas[0] * s_in, **extra_args)

    x = scale_in(x, sigmas[0])

    for step in trange(len(sigmas)-1, disable=disable):
        sigma_prev, sigma, sigma_next = sigmas[step-1], sigmas[step], sigmas[step+1]
        sigma_prev = sigma
        sigma = sigma_next

        model_prev_0 = denoised

        lambda_prev_0                 = marginal_lambda(sigma_prev)
        lambda_t                      = marginal_lambda(sigma)
        model_prev_0                  = denoised # model_prev_list[-1]
        sigma_prev_0,         sigma_t = marginal_std(sigma_prev),            marginal_std(sigma)
        log_alpha_prev_0, log_alpha_t = marginal_log_mean_coeff(sigma_prev), marginal_log_mean_coeff(sigma)
        alpha_t                       = torch.exp(log_alpha_t)
        alpha_t = 1 - sigma_next
        h = lambda_t - lambda_prev_0

        x = (sigma_t/sigma_prev_0) * x    -    (alpha_t * torch.expm1(-h)) * model_prev_0

        denoised = model(scale_out(x, sigma)   , sigma * s_in, **extra_args)
        if sigma_next == 0:
            x = denoised
            break
        
        D1_t = (denoised - model_prev_0)
        x = x - (alpha_t * -h)  *  (0.0 + 0.5 * D1_t)    # (corr_res + rhos_c[-1] * D1_t)

        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised})
            
    x = scale_out(x, sigmas[-1])
    return x



"""
def marginal_log_mean_coeff(sigma):
    return 0.5 * torch.log(1 / scale_sigma(sigma))              # (1/2) * torch.log(1/(1+sigma**2))

def marginal_alpha(t):
    return torch.exp(marginal_log_mean_coeff(t))    # simplifies to   1 / (sigma**2 + 1)**0.5          torch.sqrt(1 / (sigma**2 + 1) )             # t(0.0)->1.0  t(1.0)->0.707  t(2.0)->0.447

def marginal_std(t):
    #return torch.ones_like(t)
    return torch.sqrt(1. - torch.exp(2. * marginal_log_mean_coeff(t)))    # simplifies to  1 - 1/(sigma**2 + 1)

def marginal_lambda(t):
    #return torch.log(t)
    #Compute lambda_t = log(alpha_t) - log(sigma_t) of a given continuous-time label t in [0, T].
    log_mean_coeff = marginal_log_mean_coeff(t)
    log_std = 0.5 * torch.log(1. - torch.exp(2. * log_mean_coeff))
    return log_mean_coeff - log_std
"""




def sample_rk_vptest2(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None, latent_guide_inv=None, mask=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    sigmas = sigmas.to(torch.float64)
    x = x.to(torch.float64)

    h_fn = lambda sigma_down, sigma: -torch.log(sigma_down/sigma)
    
    sigma_min = model.inner_model.inner_model.model_sampling.sigma_min.to(x.dtype)
    sigma_max = model.inner_model.inner_model.model_sampling.sigma_max.to(x.dtype)

    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]

        denoised = model(x, sigma * s_in, **extra_args)
        if sigma_next == 0:
            x = denoised
            break
        
        eps = (x - denoised) / sigma
        noise = eps + denoised
        
        x = x + (sigma_next - sigma) * noise

        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised})

    return x




def sample_rk_res_2s(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None, latent_guide_inv=None, mask=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    sigmas = sigmas.to(torch.float64)
    x = x.to(torch.float64)

    h_fn = lambda sigma_down, sigma: -torch.log(sigma_down/sigma)
    
    lam_fn = lambda sigma: torch.log((1.-sigma)/sigma)
    sig_fn = lambda lam: 1/(torch.exp(lam)+1)
    ham_fn = lambda sigma_next, sigma: lam_fn(sigma_next) - lam_fn(sigma)
    
    sigma_min = model.inner_model.inner_model.model_sampling.sigma_min.to(x.dtype)
    sigma_max = model.inner_model.inner_model.model_sampling.sigma_max.to(x.dtype)
    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=sigma_min, sigma_max=sigma_max)
    noise_sampler2 = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+10000, sigma_min=sigma_min, sigma_max=sigma_max)

    
    substep_eta = float(get_extra_options_kv("substep_eta", str(eta), extra_options))
    
    brownian_sub_start = get_extra_options_kv("brownian_sub_start", "sigma",    extra_options)
    brownian_sub_stop  = get_extra_options_kv("brownian_sub_stop",  "sigma_2", extra_options)
    brownian_main_start = get_extra_options_kv("brownian_main_start", "sigma",    extra_options)
    brownian_main_stop  = get_extra_options_kv("brownian_main_stop",  "sigma_next", extra_options)
            
    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
        sigma_up, sigma, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, eta, noise_mode)  
        
        x_0 = x.clone()
        
        c1,c2 = 0, 1/2
        ci = [c1,c2]
        h = -torch.log(sigma_down/sigma)
        #h = ham_fn(sigma_down, sigma)
        φ = Phi(h, ci)
        
        a2_1 = c2 * φ(1,2)
        b2 = φ(2)/c2
        b1 = φ(1) - b2
        
        s2 = -torch.log(sigma) + h * c2
        sigma_2 = torch.exp(-s2)
        
        #sigma_2 = sig_fn(lam_fn(sigma) + h * c2)

        denoised = model(x, sigma * s_in, **extra_args)
        if sigma_next == 0:
            x = denoised
            break
        
        #x_2 = (sigma_2/sigma)**c2 * x + h * a2_1 * denoised # (1 - sigma_2/sigma) * denoised
        
        #denoised_2 = model(x_2, sigma_2 * s_in, **extra_args)
        
        #x = (sigma_down/sigma) * x + h * (b1 * denoised + b2 * denoised_2) # (1 - sigma_down/sigma) * (b1 * denoised + b2 * denoised_2)
        
        eps = denoised - x
        #eps = (x - denoised) / sigma
        x_2 = x + h * (a2_1 * eps)

        
        
        #x_2 = torch.exp(-h * c2) * x + h * (a2_1 * denoised)
        
        sub_sigma_next = sigma_2
        sub_sigma_up, sub_sigma, sub_sigma_down, sub_alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_2, substep_eta, noise_mode)       
        s_dict = {"sigma": sigma, "sigma_next": sigma_next, "sigma_2": sigma_2, "sigma_down": sigma_down, "sigma_up": sigma_up, "sigma_up_2": sub_sigma_up, "sigma_down_2": sub_sigma_down}

        h_new = h * h_fn(sub_sigma_down, sigma) / h_fn(sub_sigma_next, sigma) 
        h_new = h

        x_2 = x + h_new * (a2_1 * eps)
        noise = noise_sampler(sigma=s_dict[brownian_sub_start], sigma_next=s_dict[brownian_sub_stop])
        #noise = noise_sampler2(sigma=sigma, sigma_next=sub_sigma_next)
        x_2 = sub_alpha_ratio * x_2 + sub_sigma_up * noise

        
        denoised_2 = model(x_2, sigma_2 * s_in, **extra_args)
        eps_2 = denoised_2 - x
        #eps_2 = (x_2 - denoised_2) / sigma_2

        x = x + h * (b1 * eps + b2 * eps_2)
        
        noise = noise_sampler(sigma=s_dict[brownian_main_start], sigma_next=s_dict[brownian_main_stop])
        #noise = noise_sampler(sigma=sigma, sigma_next=sigma_next)
        
        x = alpha_ratio * x + sigma_up * noise
        
        #x = torch.exp(-h) * x + h * (b1 * denoised + b2 * denoised_2)

        #denoised = x_0 + ((sigma / (sigma - sigma_next)) *  h) * (b1 * eps + b2 * eps_2)
        
        #denoised = extract_pred(x_0, x, sigma, sigma_next)"""

        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised})

    return x


def sample_rk_res_2s_overstep(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    sigmas = sigmas.to(torch.float64)
    x = x.to(torch.float64)

    h_fn = lambda sigma_down, sigma: -torch.log(sigma_down/sigma)
    
    sigma_min = model.inner_model.inner_model.model_sampling.sigma_min.to(x.dtype)
    sigma_max = model.inner_model.inner_model.model_sampling.sigma_max.to(x.dtype)

    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=sigma_min, sigma_max=sigma_max)
    
    overstep = float(get_extra_options_kv("overstep", "0.0", extra_options))
      
    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
        
        sigma_up, sigma, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, eta, noise_mode)
        #sigma_down = (1-overstep) * sigma_next
        
        x_0 = x.clone()
           
        c1,c2 = 0, 1/2
        ci = [c1,c2]
        h = -torch.log(sigma_down/sigma)
        h_no_eta = -torch.log(sigma_next/sigma)
        φ = Phi(h, ci)
        
        a2_1 = c2 * φ(1,2)
        b2 = φ(2)/c2
        b1 = φ(1) - b2
        
        s2 = -torch.log(sigma) + h * c2
        sigma_2 = torch.exp(-s2)

        denoised = model(x, sigma * s_in, **extra_args)
        if sigma_next == 0:
            x = denoised
            break
        
        eps = denoised - x
 
        x_2 = x + h * (a2_1 * eps)
        
        denoised_2 = model(x_2, sigma_2 * s_in, **extra_args)
        eps_2 = denoised_2 - x


        x_next = x + h_no_eta * (b1 * eps + b2 * eps_2)
        
        x_down = x + h * (b1 * eps + b2 * eps_2)
        
        eps_down = (x_0 - x_down) / (sigma - sigma_down)
        
        #denoised_next = x + ((sigma / (sigma - sigma_down)) *  h) * (b1 * eps + b2 * eps_2)
        
        #eps_down = denoised_next - x_0
        
        denoised_next = x_0 - sigma * eps_down
        
        x = denoised_next + sigma_next * eps_down
        
        
        sigma_up, sigma, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, eta_var, noise_mode)
        #x = alpha_ratio * (denoised_next + sigma_down * eps_down) + sigma_up * eps_down 
        
        noise = noise_sampler(sigma=sigma, sigma_next=sigma_next)
        
        eps_next = (x_0 - x) / (sigma - sigma_next)
        denoised_next = x_0 - sigma * eps_next
        x = alpha_ratio * (denoised_next + sigma_down * eps_next) + sigma_up * noise * s_noise
        
        
        
        #x = x_next

        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised_next})

    return x




def get_epsilon(x_0, denoised, sigma, rk_type):
    if RK_Method_Beta.is_exponential(rk_type):
        eps = denoised - x_0
    else:
        eps = (x_0 - denoised) / sigma
    return eps

def get_data_from_step(x, x_next, sigma, sigma_next):
    h = sigma_next - sigma
    return (sigma_next * x - sigma * x_next) / h

def get_epsilon_from_step(x, x_next, sigma, sigma_next):
    h = sigma_next - sigma
    return (x - x_next) / h


def sample_rk_res_2s_prenoise(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    sigmas = sigmas.to(torch.float64)
    x = x.to(torch.float64)

    h_fn = lambda sigma_down, sigma: -torch.log(sigma_down/sigma)
    
    sigma_min = model.inner_model.inner_model.model_sampling.sigma_min.to(x.dtype)
    sigma_max = model.inner_model.inner_model.model_sampling.sigma_max.to(x.dtype)
    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=sigma_min, sigma_max=sigma_max)
    noise_sampler2 = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+10000, sigma_min=sigma_min, sigma_max=sigma_max)

    
    substep_eta = float(get_extra_options_kv("substep_eta", str(eta), extra_options))
            
    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
                   
        c1,c2 = 0, 1/2
        ci = [c1,c2]
        h = -torch.log(sigma_next/sigma)
        φ = Phi(h, ci)
        
        a2_1 = c2 * φ(1,2)
        b2 = φ(2)/c2
        b1 = φ(1) - b2
        
        s2 = -torch.log(sigma) + h * c2
        sigma_2 = torch.exp(-s2)

        if sigma_next > 0 and step > 0 and eta > 0:
            if extra_options_flag("brownian_sigma_1", extra_options):
                noise = noise_sampler(sigma=sigma, sigma_next=sigma_2)  
            elif extra_options_flag("brownian_prev_sigma", extra_options):
                noise = noise_sampler(sigma=sigmas[step-1], sigma_next=sigmas[step])
            else:
                noise = noise_sampler(sigma=sigma, sigma_next=sigma_next)
                
            sigma_up, sigma, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma, eta, noise_mode)  
            eps_prev = (x - denoised) / sigma
            x = alpha_ratio * (denoised + sigma_down * eps_prev) + sigma_up * noise    

        denoised = model(x, sigma * s_in, **extra_args)
        if sigma_next == 0:
            x = denoised
            break
        
        eps = denoised - x
        x_2 = x + h * (a2_1 * eps)
        
        if sigma_next > 0 and step > 0 and eta_var > 0:

            if extra_options_flag("brownian_2_sigma", extra_options):
                noise = noise_sampler2(sigma=sigma, sigma_next=sigma_2)
            elif extra_options_flag("brownian_sigma_2", extra_options):
                noise = noise_sampler2(sigma=sigma_2, sigma_next=sigma_next)
            else:
                noise = noise_sampler2(sigma=sigma, sigma_next=sigma_next)
                
            sigma_up_2, sigma_2, sigma_down_2, alpha_ratio_2 = get_res4lyf_step_with_model(model, sigma_2, sigma_2, eta_var, noise_mode)  
            
            if not extra_options_flag("slope_from_start", extra_options):
                if extra_options_flag("recalc_denoised_2", extra_options):
                    denoised = get_data_from_step(x, x_2, sigma, sigma_2)
                eps_prev = (x_2 - denoised) / sigma_2
                x_2 = alpha_ratio_2 * (denoised + sigma_down_2 * eps_prev) + sigma_up_2 * noise    
                
                if not extra_options_flag("disable_k1_update", extra_options):
                    eps = (x_2 - x) / (h * a2_1)
                    
            else:
                eps_prev = (x - denoised) / sigma
                x_2 = alpha_ratio * (denoised + sigma_down * eps_prev) + sigma_up * noise    

        denoised_2 = model(x_2, sigma_2 * s_in, **extra_args)
        eps_2 = denoised_2 - x
        
        denoised = x + (sigma / (sigma - sigma_next)) * h * (b1 * eps + b2 * eps_2)
        
        x = x + h * (b1 * eps + b2 * eps_2)
        
        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised_2})

    return x



def sample_rk_ralston_2s_prenoise(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    sigmas = sigmas.to(torch.float64)
    x = x.to(torch.float64)

    h_fn = lambda sigma_down, sigma: -torch.log(sigma_down/sigma)
    
    sigma_min = model.inner_model.inner_model.model_sampling.sigma_min.to(x.dtype)
    sigma_max = model.inner_model.inner_model.model_sampling.sigma_max.to(x.dtype)
    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=sigma_min, sigma_max=sigma_max)
    noise_sampler2 = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+10000, sigma_min=sigma_min, sigma_max=sigma_max)

    
    substep_eta = float(get_extra_options_kv("substep_eta", str(eta), extra_options))
            
    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
           
        a2_1 = 2/3
        b1, b2 = 1/4, 3/4
        c1, c2 = 0.0, 2/3
        
        h = sigma_next - sigma
        sigma_2 = sigma + h * c2
        
        if sigma_next > 0 and step > 0 and eta > 0:
            if extra_options_flag("brownian_sigma_1", extra_options):
                noise = noise_sampler(sigma=sigma, sigma_next=sigma_2)  
            elif extra_options_flag("brownian_prev_sigma", extra_options):
                noise = noise_sampler(sigma=sigmas[step-1], sigma_next=sigmas[step])
            else:
                noise = noise_sampler(sigma=sigma, sigma_next=sigma_next)
                
            sigma_up, sigma, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma, eta, noise_mode)  
            eps_prev = (x - denoised) / sigma
            x = alpha_ratio * (denoised + sigma_down * eps_prev) + sigma_up * noise    
        
        denoised = model(x, sigma * s_in, **extra_args)
        if sigma_next == 0:
            x = denoised
            break
        
        eps = (x - denoised) / sigma
        x_2 = x + h * (a2_1 * eps)
        
        if sigma_next > 0 and step > 0 and eta_var > 0:

            if extra_options_flag("brownian_2_sigma", extra_options):
                noise = noise_sampler2(sigma=sigma, sigma_next=sigma_2)
            elif extra_options_flag("brownian_sigma_2", extra_options):
                noise = noise_sampler2(sigma=sigma_2, sigma_next=sigma_next)
            else:
                noise = noise_sampler2(sigma=sigma, sigma_next=sigma_next)
                
            sigma_up_2, sigma_2, sigma_down_2, alpha_ratio_2 = get_res4lyf_step_with_model(model, sigma_2, sigma_2, eta_var, noise_mode)  
            
            if not extra_options_flag("slope_from_start", extra_options):
                if extra_options_flag("recalc_denoised_2", extra_options):
                    denoised = get_data_from_step(x, x_2, sigma, sigma_2)
                eps_prev = (x_2 - denoised) / sigma_2
                x_2 = alpha_ratio_2 * (denoised + sigma_down_2 * eps_prev) + sigma_up_2 * noise    
                
                if not extra_options_flag("disable_k1_update", extra_options):
                    eps = (x_2 - x) / (h * a2_1)
                    
            else:
                eps_prev = (x - denoised) / sigma
                x_2 = alpha_ratio * (denoised + sigma_down * eps_prev) + sigma_up * noise    

        denoised_2 = model(x_2, sigma_2 * s_in, **extra_args)
        eps_2 = (x_2 - denoised_2) / sigma_2
        
        denoised = x + (sigma / (sigma - sigma_next)) * h * (b1 * eps + b2 * eps_2)

        x = x + h * (b1 * eps + b2 * eps_2)
        
        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised})

    return x


def sample_rk_res_2s_prenoise_data(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    sigmas = sigmas.to(torch.float64)
    x = x.to(torch.float64)

    h_fn = lambda sigma_down, sigma: -torch.log(sigma_down/sigma)
    
    sigma_min = model.inner_model.inner_model.model_sampling.sigma_min.to(x.dtype)
    sigma_max = model.inner_model.inner_model.model_sampling.sigma_max.to(x.dtype)
    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=sigma_min, sigma_max=sigma_max)
    noise_sampler2 = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+10000, sigma_min=sigma_min, sigma_max=sigma_max)

    
    substep_eta = float(get_extra_options_kv("substep_eta", str(eta), extra_options))
    
    brownian_sub_start = get_extra_options_kv("brownian_sub_start", "sigma_2",    extra_options)
    brownian_sub_stop  = get_extra_options_kv("brownian_sub_stop",  "sigma_next", extra_options)
    brownian_main_start = get_extra_options_kv("brownian_main_start", "sigma",    extra_options)
    brownian_main_stop  = get_extra_options_kv("brownian_main_stop",  "sigma_next", extra_options)
    
    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
        x_0 = x.clone()
           
        c1,c2 = 0, 1/2
        ci = [c1,c2]
        h = -torch.log(sigma_next/sigma)
        φ = Phi(h, ci)
        
        a2_1 = c2 * φ(1,2)
        b2 = φ(2)/c2
        b1 = φ(1) - b2
        
        s2 = -torch.log(sigma) + h * c2
        sigma_2 = torch.exp(-s2)
        
        sigma_up, sigma, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, eta, noise_mode)  
        sigma_up_2, sigma, sigma_down_2, alpha_ratio_2 = get_res4lyf_step_with_model(model, sigma, sigma_2, eta_var, noise_mode)  

        denoised = model(x, sigma * s_in, **extra_args)
        if sigma_next == 0:
            x = denoised
            break
        
        x_2 = torch.exp(-h * c2) * x_0 + h * (a2_1 * denoised)
        
        s_dict = {"sigma": sigma, "sigma_next": sigma_next, "sigma_2": sigma_2, "sigma_down": sigma_down, "sigma_up": sigma_up, "sigma_up_2": sigma_up_2, "sigma_down_2": sigma_down_2}
        
        if sigma_next > 0 and eta_var > 0: #step > 0 and 
            #x_next = x + h * (b1 * eps + b2 * eps_2)
            eps_2 = (x_0 - x_2) / (sigma - sigma_2)
            denoised_2 = x_0 - sigma * eps_2
           
            noise = noise_sampler(sigma=s_dict[brownian_sub_start], sigma_next=s_dict[brownian_sub_stop])
            #noise = noise_sampler(sigma=sigma_2, sigma_next=sigma_next)
            x_2 = alpha_ratio_2 * (denoised_2 + sigma_down_2 * eps_2) + sigma_up_2 * noise        #NOISE SWAP NOISE
            
            if extra_options_flag("use_x_update", extra_options):
                eps = denoised - x_0
                x_0 = x = x_2 - h * (a2_1 * eps)
                eps = denoised - x_0
            
            if extra_options_flag("use_k1_update", extra_options):
                eps = (x_2 - x) / (h * a2_1)
                
            if extra_options_flag("use_bong1_update", extra_options): # does not get x_2 synced with eps (wrong: x_2 = x_0 + h * (a2_1 * eps)), but does give same result for x_0 = denoised + sigma * eps, and x_0 = x_2 - h * (a2_1 * eps)
                eps = (x_2 - denoised) / (sigma + h * a2_1)
                x_0 = x = denoised + (sigma*(x_2 - denoised)) / (h*a2_1 + sigma)
                #eps = -(sigma * eps)

            #x_2 = alpha_ratio * (denoised + sigma_down_2 * eps_prev) + sigma_up_2 * noise    

        denoised_2 = model(x_2, sigma_2 * s_in, **extra_args)
        
        x_next = torch.exp(-h) * x_0 + h * (b1 * denoised + b2 * denoised_2)
        eps_next = (x_0 - x_next) / (sigma - sigma_next)
        denoised_next = x_0 - sigma * eps_next
        
        noise = noise_sampler(sigma=s_dict[brownian_main_start], sigma_next=s_dict[brownian_main_stop])
        
        #noise = noise_sampler(sigma=sigma, sigma_next=sigma_next)
        x = alpha_ratio * (denoised_next + sigma_down * eps_next) + sigma_up * noise    
    
        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised})

    return x



def sample_rk_res_2s_prenoise_alt(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    sigmas = sigmas.to(torch.float64)
    x = x.to(torch.float64)

    h_fn = lambda sigma_down, sigma: -torch.log(sigma_down/sigma)
    
    sigma_min = model.inner_model.inner_model.model_sampling.sigma_min.to(x.dtype)
    sigma_max = model.inner_model.inner_model.model_sampling.sigma_max.to(x.dtype)
    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=sigma_min, sigma_max=sigma_max)
    noise_sampler2 = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+10000, sigma_min=sigma_min, sigma_max=sigma_max)

    
    substep_eta = float(get_extra_options_kv("substep_eta", str(eta), extra_options))
    
    brownian_sub_start = get_extra_options_kv("brownian_sub_start", "sigma_2",    extra_options)
    brownian_sub_stop  = get_extra_options_kv("brownian_sub_stop",  "sigma_next", extra_options)
    brownian_main_start = get_extra_options_kv("brownian_main_start", "sigma",    extra_options)
    brownian_main_stop  = get_extra_options_kv("brownian_main_stop",  "sigma_next", extra_options)
    
    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
        x_0 = x.clone()
           
        c1,c2 = 0, 1/2
        ci = [c1,c2]
        h = -torch.log(sigma_next/sigma)
        φ = Phi(h, ci)
        
        a2_1 = c2 * φ(1,2)
        b2 = φ(2)/c2
        b1 = φ(1) - b2
        
        s2 = -torch.log(sigma) + h * c2
        sigma_2 = torch.exp(-s2)
        
        sigma_up, sigma, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, eta, noise_mode)  
        sigma_up_2, sigma, sigma_down_2, alpha_ratio_2 = get_res4lyf_step_with_model(model, sigma, sigma_2, eta_var, noise_mode)  
        #sigma_up_2, sigma_2, sigma_down_2, alpha_ratio_2 = get_res4lyf_step_with_model(model, sigma_2, sigma_next, eta_var, noise_mode)  
        
        h_down = -torch.log(sigma_down/sigma)
        
        if extra_options_flag("use_h_down", extra_options):
            h = h_down
            φ = Phi(h, ci)
        
            a2_1 = c2 * φ(1,2)
            b2 = φ(2)/c2
            b1 = φ(1) - b2
            
            s2 = -torch.log(sigma) + h * c2
            sigma_2 = torch.exp(-s2)



        denoised = model(x, sigma * s_in, **extra_args)
        if sigma_next == 0:
            x = denoised
            break
        
        eps = denoised - x_0
        x_2 = x + h * (a2_1 * eps)
        
        s_dict = {"sigma": sigma, "sigma_next": sigma_next, "sigma_2": sigma_2, "sigma_down": sigma_down, "sigma_up": sigma_up, "sigma_up_2": sigma_up_2, "sigma_down_2": sigma_down_2}
        
        if sigma_next > 0 and eta_var > 0: #step > 0 and 
            #x_next = x + h * (b1 * eps + b2 * eps_2)
            eps_2 = (x_0 - x_2) / (sigma - sigma_2)
            denoised_2 = x_0 - sigma * eps_2
           
            noise = noise_sampler(sigma=s_dict[brownian_sub_start], sigma_next=s_dict[brownian_sub_stop])    # THIS USED TO BE NOISE_SAMPLER2 !!!!!!!!!!!!!!!!!
            #noise = noise_sampler(sigma=sigma_2, sigma_next=sigma_next)
            x_2 = alpha_ratio_2 * (denoised_2 + sigma_down_2 * eps_2) + sigma_up_2 * noise    
            
            if extra_options_flag("use_x_update", extra_options):
                x_0 = x = x_2 - h * (a2_1 * eps)
                eps = denoised - x_0
            
            if extra_options_flag("use_k1_update", extra_options):
                eps = (x_2 - x) / (h * a2_1)
                
            if extra_options_flag("use_bong1_update1", extra_options): # does not get x_2 synced with eps (wrong: x_2 = x_0 + h * (a2_1 * eps)), but does give same result for x_0 = denoised + sigma * eps, and x_0 = x_2 - h * (a2_1 * eps)
                eps = (x_2 - denoised) / (sigma + h * a2_1)
                x_0 = x = denoised + (sigma*(x_2 - denoised)) / (h*a2_1 + sigma)
                #eps = -(sigma * eps)
        if extra_options_flag("use_bong1_update_fixed", extra_options): # does not get x_2 synced with eps (wrong: x_2 = x_0 + h * (a2_1 * eps)), but does give same result for x_0 = denoised + sigma * eps, and x_0 = x_2 - h * (a2_1 * eps)
            eps = (x_2 - denoised) / torch.exp(-(-torch.log(sigma) + h * a2_1))
            x_0 = x = denoised + sigma*eps
        if extra_options_flag("use_bong1_update5", extra_options):
            for i in range(100):
                x_0 = x = x_2 - h * (a2_1 * eps)
                eps = denoised - x_0
        if extra_options_flag("use_bong1_update6", extra_options):
            for i in range(100):
                eps = denoised - (x_2 - h * (a2_1 * eps))

        denoised_2 = model(x_2, sigma_2 * s_in, **extra_args)
        eps_2 = denoised_2 - x_0
        
        if extra_options_flag("use_bong2_update7", extra_options):
            x_next = x_0 + h * (b1 * eps + b2 * eps_2)
            for i in range(100):
                
                x_0 = x = x_next - h * (b1 * eps + b2 * eps_2)
                x_2 =     x_0 + h * (a2_1 * eps)
                
                eps_2 = denoised_2 - x_2
                eps   = denoised   - x_0
        
        if extra_options_flag("h_down_big", extra_options):
            φ = Phi(h_down, ci)
        
            a2_1 = c2 * φ(1,2)
            b2 = φ(2)/c2
            b1 = φ(1) - b2
            
            denoised = x + (sigma / (sigma - sigma_next)) * h_down * (b1 * eps + b2 * eps_2)

            x_down = x + h_down * (b1 * eps + b2 * eps_2)
            
            noise = noise_sampler(sigma=sigma, sigma_next=sigma_next)
            x = alpha_ratio * x_down + sigma_up * noise
        else:
            x_next = x_0 + h * (b1 * eps + b2 * eps_2)
            eps_next = (x_0 - x_next) / (sigma - sigma_next)
            denoised_next = x_0 - sigma * eps_next
            
            noise = noise_sampler(sigma=s_dict[brownian_main_start], sigma_next=s_dict[brownian_main_stop])
            
            #noise = noise_sampler(sigma=sigma, sigma_next=sigma_next)
            x = alpha_ratio * (denoised_next + sigma_down * eps_next) + sigma_up * noise    
        
        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised})

    return x

def vpsde_noise_add_old(x_0, x_next, sigma, sigma_next, sigma_down, sigma_up, alpha_ratio, noise_sampler):
    eps_next = (x_0 - x_next) / (sigma - sigma_next)
    denoised_next = x_0 - sigma * eps_next
    noise = noise_sampler(sigma=sigma, sigma_next=sigma_next)
    x = alpha_ratio * (denoised_next + sigma_down * eps_next) + sigma_up * noise    
    return x



def sample_rk_ralston_2s_prenoise_alt(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    sigmas = sigmas.to(torch.float64)
    x = x.to(torch.float64)

    h_fn = lambda sigma_down, sigma: -torch.log(sigma_down/sigma)
    
    sigma_min = model.inner_model.inner_model.model_sampling.sigma_min.to(x.dtype)
    sigma_max = model.inner_model.inner_model.model_sampling.sigma_max.to(x.dtype)
    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=sigma_min, sigma_max=sigma_max)
    noise_sampler2 = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+10000, sigma_min=sigma_min, sigma_max=sigma_max)

    a2_1 = 2/3
    b1, b2 = 1/4, 3/4
    c1, c2 = 0.0, 2/3
    substep_eta = float(get_extra_options_kv("substep_eta", str(eta), extra_options))
            
    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
        x_0 = x.clone()
           
        h = sigma_next - sigma

        sigma_2 = sigma + h * c2
        
        sigma_up, sigma, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, eta, noise_mode)  
        sigma_up_2, sigma, sigma_down_2, alpha_ratio_2 = get_res4lyf_step_with_model(model, sigma, sigma_2, eta_var, noise_mode)  
        #sigma_up_2, sigma_2, sigma_down_2, alpha_ratio_2 = get_res4lyf_step_with_model(model, sigma_2, sigma_next, eta_var, noise_mode)  
        
        h_down = sigma_down - sigma
        
        if extra_options_flag("use_h_down", extra_options):
            h = h_down
            sigma_2 = sigma + h_down * c2

        denoised = model(x, sigma * s_in, **extra_args)
        if sigma_next == 0:
            x = denoised
            break
        
        eps = (x - denoised) / sigma
        x_2 = x + h * (a2_1 * eps)
        
        if sigma_next > 0 and step > 0 and eta_var > 0:
            #x_next = x + h * (b1 * eps + b2 * eps_2)
            eps_2 = (x_0 - x_2) / (sigma - sigma_2)
            denoised_2 = x_0 - sigma * eps_2
            
            noise = noise_sampler(sigma=sigma_2, sigma_next=sigma_next)
            x_2 = alpha_ratio_2 * (denoised_2 + sigma_down_2 * eps_2) + sigma_up_2 * noise    
            
            if not extra_options_flag("disable_k1_update", extra_options):
                eps = (x_2 - x) / (h * a2_1)
            
            #x_2 = alpha_ratio * (denoised + sigma_down_2 * eps_prev) + sigma_up_2 * noise    

        denoised_2 = model(x_2, sigma_2 * s_in, **extra_args)
        eps_2 = (x_2 - denoised_2) / sigma_2
        
        if extra_options_flag("h_down_big", extra_options):

            denoised = x + (sigma / (sigma - sigma_next)) * h_down * (b1 * eps + b2 * eps_2)

            x_down = x + h_down * (b1 * eps + b2 * eps_2)
            
            noise = noise_sampler(sigma=sigma, sigma_next=sigma_next)
            x = alpha_ratio * x_down + sigma_up * noise
        else:
            x_next = x + h * (b1 * eps + b2 * eps_2)
            eps_next = (x_0 - x_next) / (sigma - sigma_next)
            denoised_next = x_0 - sigma * eps_next
            
            noise = noise_sampler(sigma=sigma, sigma_next=sigma_next)
            x = alpha_ratio * (denoised_next + sigma_down * eps_next) + sigma_up * noise    
        
        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised})

    return x






def sample_rk_ralston_2s_prenoise_alt2(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    sigmas = sigmas.to(torch.float64)
    x = x.to(torch.float64)

    h_fn = lambda sigma_down, sigma: -torch.log(sigma_down/sigma)
    
    sigma_min = model.inner_model.inner_model.model_sampling.sigma_min.to(x.dtype)
    sigma_max = model.inner_model.inner_model.model_sampling.sigma_max.to(x.dtype)
    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=sigma_min, sigma_max=sigma_max)
    noise_sampler2 = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+10000, sigma_min=sigma_min, sigma_max=sigma_max)

    a2_1 = 2/3
    b1, b2 = 1/4, 3/4
    c1, c2 = 0.0, 2/3
    substep_eta = float(get_extra_options_kv("substep_eta", str(eta), extra_options))
            
    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
        x_0 = x.clone()
        x_0_orig = x.clone()
    
        h = sigma_next - sigma

        sigma_2 = sigma + h * c2
        
        sigma_up, sigma, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, eta, noise_mode)  
        sigma_up_2, sigma, sigma_down_2, alpha_ratio_2 = get_res4lyf_step_with_model(model, sigma, sigma_2, eta_var, noise_mode)  
        
        h_down = sigma_down - sigma
        
        denoised = model(x, sigma * s_in, **extra_args)
        if sigma_next == 0:
            x = denoised
            break
        
        eps = (x - denoised) / sigma
        x_2 = x + h * (a2_1 * eps)                                                                # UPDATE
        
        if sigma_next > 0 and eta_var > 0:
            eps_2 = (x_0 - x_2) / (sigma - sigma_2)
            denoised_2 = x_0 - sigma * eps_2
            
            noise = noise_sampler(sigma=sigma_2, sigma_next=sigma_next)
            x_2 = alpha_ratio_2 * (denoised_2 + sigma_down_2 * eps_2) + sigma_up_2 * noise        # NOISE ADD analogous noising step
            
            if extra_options_flag("use_x_update", extra_options):
                x_0 = x = x_2 - h * (a2_1 * eps)
                eps = (x_0 - denoised) / sigma
            
            if extra_options_flag("use_k1_update", extra_options):
                eps = (x_2 - x) / (h * a2_1)

            if extra_options_flag("use_bong1_update", extra_options):
                eps = (x_2 - denoised) / (sigma + h * a2_1)
                x_0 = x = denoised + sigma * eps
                x_0 = x = denoised + (sigma*(x_2 - denoised)) / (h*a2_1 + sigma)

        denoised_2 = model(x_2, sigma_2 * s_in, **extra_args)
        if extra_options_flag("use_x_0_eps_2", extra_options):
            eps_2 = (x_0 - denoised_2) / sigma_2
        else:
            eps_2 = (x_2 - denoised_2) / sigma_2
        
        x_next = x_0 + h * (b1 * eps + b2 * eps_2)
        eps_next = (x_0 - x_next) / (sigma - sigma_next)
        denoised_next = x_0 - sigma * eps_next
        
        noise = noise_sampler(sigma=sigma, sigma_next=sigma_next)
        x = alpha_ratio * (denoised_next + sigma_down * eps_next) + sigma_up * noise    
        
        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised})

    return x



def vpsde_noise_add(x_0, x_next, sigma, sigma_next, sigma_down, sigma_up, alpha_ratio, noise_sampler):
    if sigma_next == 0:
        return x_next
    if sigma == sigma_next:
        sigma_next *= 0.999
    eps_next = (x_0 - x_next) / (sigma - sigma_next)
    denoised_next = x_0 - sigma * eps_next
    noise = noise_sampler(sigma=sigma, sigma_next=sigma_next)
    x = alpha_ratio * (denoised_next + sigma_down * eps_next) + sigma_up * noise    
    return x


def sample_rk_ralston_3s_prenoise_alt2(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    sigmas = sigmas.to(torch.float64)
    x = x.to(torch.float64)

    h_fn = lambda sigma_down, sigma: -torch.log(sigma_down/sigma)
    
    sigma_min = model.inner_model.inner_model.model_sampling.sigma_min.to(x.dtype)
    sigma_max = model.inner_model.inner_model.model_sampling.sigma_max.to(x.dtype)
    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=sigma_min, sigma_max=sigma_max)
    
    noise_sampler_type2  = get_extra_options_kv("noise_sampler_type2",  str(noise_sampler_type), extra_options)
    noise_sampler_type3  = get_extra_options_kv("noise_sampler_type3",  str(noise_sampler_type), extra_options)
    
    noise_sampler2 = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+10000, sigma_min=sigma_min, sigma_max=sigma_max)
    
    noise_sampler3 = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+100000, sigma_min=sigma_min, sigma_max=sigma_max)
    
    if not extra_options_flag("trisampler", extra_options):
        noise_sampler2 = noise_sampler3 = noise_sampler

    brownian_2_start = get_extra_options_kv("brownian_2_start", "sigma",    extra_options)
    brownian_2_stop  = get_extra_options_kv("brownian_2_stop",  "sigma_2", extra_options)
    
    brownian_3_start = get_extra_options_kv("brownian_3_start", "sigma_2",    extra_options)
    brownian_3_stop  = get_extra_options_kv("brownian_3_stop",  "sigma_3", extra_options)
    
    brownian_main_start = get_extra_options_kv("brownian_main_start", "sigma",    extra_options)
    brownian_main_stop  = get_extra_options_kv("brownian_main_stop",  "sigma_next", extra_options)
    
    normalize_noise_latent = extra_options_flag("normalize_noise_latent", extra_options)
    normalize_noise_channels = extra_options_flag("normalize_noise_channels", extra_options)
    
    brownian_2_fraction_start = float(get_extra_options_kv("brownian_2_fraction_start", "-1.",    extra_options))
    brownian_2_fraction_stop = float(get_extra_options_kv("brownian_2_fraction_stop", "-1.",    extra_options))

    brownian_3_fraction_start = float(get_extra_options_kv("brownian_3_fraction_start", "-1.",    extra_options))
    brownian_3_fraction_stop = float(get_extra_options_kv("brownian_3_fraction_stop", "-1.",    extra_options))    
    
    brownian_main_fraction_start = float(get_extra_options_kv("brownian_main_fraction_start", "-1.",    extra_options))
    brownian_main_fraction_stop = float(get_extra_options_kv("brownian_main_fraction_stop", "-1.",    extra_options))  
    

    a2_1 = 1/2
    a3_1, a3_2 = 0., 3/4
    b1, b2, b3 = 2/9, 1/3, 4/9
    c1, c2, c3 = 0., 1/2, 3/4
    
    if extra_options_flag("use_ssprk3", extra_options):
        a2_1 = 1.
        a3_1, a3_2 = 1/4, 1/4
        b1, b2, b3 = 1/6, 1/6, 2/3
        c1, c2, c3 = 0., 1., 1/2
    
    substep_eta = float(get_extra_options_kv("substep_eta", str(eta), extra_options))
    
    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
        x_0 = x.clone()
        x_0_orig = x.clone()
           
        h = sigma_next - sigma

        sigma_2 = sigma + h * c2
        sigma_3 = sigma + h * c3
        
        sigma_up, sigma, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, eta, noise_mode)  
        
        sigma_up_2, sigma, sigma_down_2, alpha_ratio_2 = get_res4lyf_step_with_model(model, sigma, sigma_2, eta_var, noise_mode)  
        sigma_up_3, sigma_2, sigma_down_3, alpha_ratio_3 = get_res4lyf_step_with_model(model, sigma_2, sigma_3, eta_var, noise_mode)  
        
        s_dict = {"sigma": sigma, "sigma_next": sigma_next, "sigma_down": sigma_down, "sigma_up": sigma_up, "sigma_2": sigma_2, "sigma_up_2": sigma_up_2, "sigma_down_2": sigma_down_2, \
            "sigma_3": sigma_3, "sigma_up_3": sigma_up_3, "sigma_down_3": sigma_down_3, }
        
        h_down = sigma_down - sigma
        
        denoised = model(x, sigma * s_in, **extra_args)
        if sigma_next == 0:
            x = denoised
            break
        
        eps = (x - denoised) / sigma
        x_2 = x_0 + h * (a2_1 * eps)
        
        if sigma_next > 0 and eta_var > 0:
            eps_2 = (x_0 - x_2) / (sigma - sigma_2)
            denoised_2 = x_0 - sigma * eps_2
            
            #noise2 = noise_sampler2(sigma=sigma, sigma_next=sigma_2)
            if brownian_2_fraction_start < 0.0:
                noise2 = noise_sampler2(sigma=s_dict[brownian_2_start], sigma_next=s_dict[brownian_2_stop])
            else:
                #sigma_brownian = sigma + brownian_2_fraction_start * (sigma - sigma_next)
                #sigma_next_brownian = sigma + brownian_2_fraction_stop * (sigma - sigma_next)
                sigma_brownian = sigma + brownian_2_fraction_start * (sigma_next - sigma)
                sigma_next_brownian = sigma + brownian_2_fraction_stop * (sigma_next - sigma)
                noise2 = noise_sampler2(sigma=sigma_brownian, sigma_next=sigma_next_brownian)
                
            if normalize_noise_latent:
                noise2 = (noise2 - noise2.mean()) / noise2.std()
            if normalize_noise_channels:
                for ch in range(x.shape[-3]):
                    noise2[0][ch] = (noise2[0][ch] - noise2[0][ch].mean()) / noise2[0][ch].std()
            
            x_2 = alpha_ratio_2 * (denoised_2 + sigma_down_2 * eps_2) + sigma_up_2 * noise2
            
            if extra_options_flag("use_x_update", extra_options):
                x_0 = x = x_2 - h * (a2_1 * eps)
                eps = (x_0 - denoised) / sigma
            
            if extra_options_flag("use_k1_update", extra_options):
                eps = (x_2 - x) / (h * a2_1)

            if extra_options_flag("use_bong1_update1", extra_options):
                eps = (x_2 - denoised) / (sigma + h * a2_1)
                x_0 = x = denoised + (sigma*(x_2 - denoised)) / (h*a2_1 + sigma)
            if extra_options_flag("use_bong1_update5", extra_options):
                for i in range(100):
                    x_0 = x = x_2 - h * (a2_1 * eps)
                    eps = (x_0 - denoised) / sigma
            if extra_options_flag("use_bong1_update6", extra_options):
                for i in range(100):
                    x_0_tmp = x_2 - h * (a2_1 * eps)
                    eps = (x_0_tmp - denoised) / sigma
            if extra_options_flag("use_bong1_update7", extra_options):
                for i in range(100):
                    eps = ((x_2 - x_0) / (h * a2_1)    +     (x_0 - denoised) / sigma) / 2


        denoised_2 = model(x_2, sigma_2 * s_in, **extra_args)
        if extra_options_flag("use_x_0_eps_2", extra_options):
            eps_2 = (x_0 - denoised_2) / sigma_2
        else:
            eps_2 = (x_2 - denoised_2) / sigma_2
        
        
        
        x_3 = x_0 + h * (a3_1 * eps + a3_2 * eps_2)
        
        if sigma_next > 0 and eta_var > 0:
            eps_3 = (x_0 - x_3) / (sigma - sigma_3)
            denoised_3 = x_0 - sigma * eps_3
            
            #noise3 = noise_sampler2(sigma=sigma_2, sigma_next=sigma_3)         # CHANGED TO NOISE SAMPLER 2
            #noise3 = noise_sampler3(sigma=s_dict[brownian_3_start], sigma_next=s_dict[brownian_3_stop])
            if brownian_3_fraction_start < 0.0:
                noise3 = noise_sampler3(sigma=s_dict[brownian_3_start], sigma_next=s_dict[brownian_3_stop])
            else:
                #sigma_brownian = sigma + brownian_3_fraction_start * (sigma - sigma_next)
                #sigma_next_brownian = sigma + brownian_3_fraction_stop * (sigma - sigma_next)
                sigma_brownian = sigma + brownian_3_fraction_start * (sigma_next - sigma)
                sigma_next_brownian = sigma + brownian_3_fraction_stop * (sigma_next - sigma)
                noise3 = noise_sampler3(sigma=sigma_brownian, sigma_next=sigma_next_brownian)

            if normalize_noise_latent:
                noise3 = (noise3 - noise3.mean()) / noise3.std()
            if normalize_noise_channels:
                for ch in range(x.shape[-3]):
                    noise3[0][ch] = (noise3[0][ch] - noise3[0][ch].mean()) / noise3[0][ch].std()
            
            x_3 = alpha_ratio_3 * (denoised_3 + sigma_down_3 * eps_3) + sigma_up_3 * noise3
            
            if extra_options_flag("use_x_update", extra_options):
                x_0 = x = x_3 - h * (a3_1 * eps + a3_2 * eps_2)
                eps_2 = (x_0 - denoised_2) / sigma
            
            if extra_options_flag("use_k1_update", extra_options):
                eps = (x_3 - x - h*a3_2*eps) / (h * a3_1)

            if extra_options_flag("use_bong2_update1", extra_options):
                #eps = (x_3 - denoised) / (sigma + h * a3_1)
                #x_0 = x = denoised + (sigma*(x_3 - denoised)) / (h*a3_1 + sigma)
                #eps = (x_2 - denoised) / (a2_1 * (sigma/c2 + h))
                #x_0 = x = denoised + (sigma * a2_1) / c2 * epsilon
                
                # eps = (x_2 - denoised) / (sigma + h * a2_1)
                #x_0 = x = denoised + sigma * epsilon
                #eps_2 = (denoised + sigma * epsilon - denoised_2) / sigma_2
                eps = (x_3 - denoised - (h*a3_2)/sigma_2 * (denoised - denoised_2)) / (sigma + h * a3_1 + (h*a3_2)/sigma_2 * (sigma + h * a2_1))
                x_0 = x = denoised + sigma * eps
                x_2_tmp = denoised + eps*(sigma + h * a2_1)
                eps_2 = (x_2_tmp - denoised_2) / sigma_2
                print("x_2 - x_2_tmp", torch.norm(x_2 - x_2_tmp).item())
            if extra_options_flag("use_bong2_update2", extra_options):
                eps = (x_2 - denoised) / (sigma + h * a2_1)
                x_0 = x = denoised + sigma * eps
                
                eps_2 = (x_3 - denoised - (sigma + h * a3_1) * eps) / (h * a3_2)
            if extra_options_flag("use_bong2_update3", extra_options):
                eps = (x_3 - denoised - (h*a3_2)/sigma_2 * (x_2 - denoised_2)) / (sigma + h * a3_1)
                x_0 = x = denoised + sigma * eps
                eps_2 = (x_2 - denoised_2) / sigma_2
            if extra_options_flag("use_bong2_update4", extra_options):
                eps = (x_3 - denoised - (h*a3_2)/sigma_2 * (denoised - denoised_2)) / (sigma + h * a3_1 + (h*a3_2)/sigma_2 * (sigma + h * a2_1))
                x_0 = x = denoised + sigma * eps
                eps_2 = (denoised + eps*(sigma + h * a2_1) - denoised_2) / sigma_2
                x_2 = denoised + (sigma + h * a2_1) * eps
                #print("x_2 - x_2_tmp", torch.norm(x_2 - x_2_tmp).item())
            if extra_options_flag("use_bong2_update5", extra_options):
                for i in range(100):
                    
                    x_0 = x = x_3 - h * (a3_1 * eps + a3_2 * eps_2)
                    x_2 =     x_0 + h * (a2_1 * eps)
                    
                    eps_2 = (x_2 - denoised_2) / sigma_2
                    eps   = (x_0 - denoised)   / sigma
            if extra_options_flag("use_bong2_update6a", extra_options):
                for i in range(100):
                    
                    x_0_tmp = x_3 - h * (a3_1 * eps + a3_2 * eps_2)
                    x_2_tmp =     x_0 + h * (a2_1 * eps)
                    
                    eps_2 = (x_2_tmp - denoised_2) / sigma_2
                    eps   = (x_0_tmp - denoised)   / sigma
                    
            if extra_options_flag("use_bong2_update6b", extra_options):
                for i in range(100):
                    
                    x_0_tmp = x_3 - h * (a3_1 * eps + a3_2 * eps_2)
                    x_2_tmp =     x_0_tmp + h * (a2_1 * eps)
                    
                    eps_2 = (x_2_tmp - denoised_2) / sigma_2
                    eps   = (x_0_tmp - denoised)   / sigma                #print("x_2 - x_2_tmp", torch.norm(x_2 - x_2_tmp).item())

            if extra_options_flag("use_bong2_update7", extra_options):
                for i in range(100):
                    
                    #x_0 = x_3 - h * (a3_1 * eps + a3_2 * eps_2)
                    #x_3 = x_0 + h * (a3_1 * eps + a3_2 * eps_2)
                    #x_2 =     x_0 + h * (a2_1 * eps)
                    
                    eps = ((x_2 - x_0) / (h * a2_1)                            +       (x_0 - denoised)   / sigma)      /     2
                    eps_2 = ((x_3 - x_0 - h * a3_1 * eps) / (h * a3_2)         +       (x_2 - denoised_2) / sigma_2)    /     2
                    
                    #eps = (x_3 - x_0 - h * a3_2 * eps_2) / (h * a3_1)
                    
                    #eps_2 = (x_2 - denoised_2) / sigma_2
                    #eps   = (x_0 - denoised)   / sigma

                        

        denoised_3 = model(x_3, sigma_3 * s_in, **extra_args)
        if extra_options_flag("use_x_0_eps_2", extra_options):
            eps_3 = (x_0 - denoised_3) / sigma_3
        else:
            eps_3 = (x_3 - denoised_3) / sigma_3
        
        
        
        if extra_options_flag("use_bong2_update8", extra_options):
            x_next = x_0 + h * (b1 * eps + b2 * eps_2 + b3 * eps_3)
            for i in range(100):
                
                x_0 = x = x_next - h * (b1 * eps + b2 * eps_2 + b3 * eps_3)
                x_2 = x_0 + h * (a2_1 * eps)
                x_3 = x_0 + h * (a3_1 * eps + a3_2 * eps_2)
                
                eps_3 = denoised_3 - x_3
                eps_2 = denoised_2 - x_2
                eps   = denoised   - x_0
        
        
        
        x_next = x_0 + h * (b1 * eps + b2 * eps_2 + b3 * eps_3)
        eps_next = (x_0 - x_next) / (sigma - sigma_next)
        denoised_next = x_0 - sigma * eps_next
        
        
        
        if brownian_main_fraction_start < 0.0:
            noise = noise_sampler(sigma=s_dict[brownian_main_start], sigma_next=s_dict[brownian_main_stop])
        else:
            sigma_brownian = sigma + brownian_main_fraction_start * (sigma_next - sigma)
            sigma_next_brownian = sigma + brownian_main_fraction_stop * (sigma_next - sigma)
            noise = noise_sampler(sigma=sigma_brownian, sigma_next=sigma_next_brownian)
        if normalize_noise_latent:
            noise = (noise - noise.mean()) / noise.std()
        if normalize_noise_channels:
            for ch in range(x.shape[-3]):
                noise[0][ch] = (noise[0][ch] - noise[0][ch].mean()) / noise[0][ch].std()
        
        print("noise2:", noise2.std().item(), noise2.mean().item(), noise2.sum().item(), noise2.abs().sum().item())
        print("noise3:", noise3.std().item(), noise3.mean().item(), noise3.sum().item(), noise3.abs().sum().item())
        print("noise :", noise.std().item(), noise.mean().item(), noise.sum().item(), noise.abs().sum().item())
        x = alpha_ratio * (denoised_next + sigma_down * eps_next) + sigma_up * noise    
        
        if callback is not None:
            if extra_options_flag("preview_denoised_next", extra_options):
                callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised_next})
            else:
                callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised})

    return x


"""n
denoised = x - sigma * epsilon
denoised_2 = x_2 - sigma_2 * epsilon_2
denoised_3 = x_3 - sigma_3 * epsilon_3
x_2 = x + h * (a2_1 * epsilon)
x_3 = x + h * (a3_1 * epsilon + a3_2 * epsilon_2)
x_4 = x + h * (a4_1 * epsilon + a4_2 * epsilon_2 + a4_3 * epsilon_3)
"""


def sample_rk_res_2s_orig(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    sigmas = sigmas.to(torch.float64)
    x = x.to(torch.float64)

    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
    
    sigma_min = model.inner_model.inner_model.model_sampling.sigma_min.to(x.dtype)
    sigma_max = model.inner_model.inner_model.model_sampling.sigma_max.to(x.dtype)
    #sigmas[-1] = sigma_min
            
    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
        x_0 = x.clone()
           
        h = -torch.log(sigma_next/sigma)
        #h = -torch.log(sigma_next) + torch.log(sigma)
        
        c2 =  0.5

        a2_1 = c2 * phi(1, -h*c2)
        b1 =        phi(1, -h) - phi(2, -h)/c2
        b2 =        phi(2, -h)/c2

        denoised = model(x, sigma * s_in, **extra_args)
        if sigma_next == 0:
            return denoised
        
        eps = denoised - x
        #eps = (x - denoised) / sigma
        
        x_2 = x + h * (a2_1 * eps)
        #x_2 = torch.exp(-h * c2) * x + h * (a2_1 * denoised)
        
        s2 = -torch.log(sigma) + h * c2
        sigma_2 = torch.exp(-s2)
        
        denoised_2 = model(x_2, sigma_2 * s_in, **extra_args)
        eps_2 = denoised_2 - x
        #eps_2 = (x_2 - denoised_2) / sigma_2

        x = x + h * (b1 * eps + b2 * eps_2)
        
        #x = torch.exp(-h) * x + h * (b1 * denoised + b2 * denoised_2)

        #denoised = x_0 + ((sigma / (sigma - sigma_next)) *  h) * (b1 * eps + b2 * eps_2)
        
        #denoised = extract_pred(x_0, x, sigma, sigma_next)

        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised})

    return denoised




def gen_first_col_exp(a, b, c, φ):
    for i in range(len(c)): 
        a[i][0] = c[i] * φ(1,i+1) - sum(a[i])
    for i in range(len(b)): 
        b[i][0] =         φ(1)     - sum(b[i])
    return a, b






# using GenLawson45 https://ora.ox.ac.uk/objects/uuid:cc001282-4285-4ca2-ad06-31787b540c61/files/m611df1a355ca243beb09824b70e5e774 MULTISTEP! reconsider implementation of U/V, see pg 66/183
def sample_rk_crazymod43(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    sigma_max = model.inner_model.inner_model.model_sampling.sigma_max.to(x.dtype)
    sigma_min = model.inner_model.inner_model.model_sampling.sigma_min.to(x.dtype)
    print("initial seed (will add 1): ", torch.initial_seed())
    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=sigma_min, sigma_max=sigma_max)
    
    if sigmas[-1] == 0:
        sigmas[-1] = sigma_min
    
    h_fn = lambda sigma_next, sigma: -torch.log()
    c1,c2,c3,c4 = 0, 1/2, 1/2, 1

    h_prev1, h_prev2, denoised_prev1, denoised_prev2 = None, None, None, None
    
    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
        
        sigma_up, sigma, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, eta, noise_mode)
        
        sigma1 = sigma
        
        h = -torch.log(sigma_down/sigma)
        h_no_eta = -torch.log(sigma_next/sigma)
        
        s2 = -torch.log(sigma) + h * c2
        sigma2 = torch.exp(-s2)
        
        s3 = -torch.log(sigma) + h * c3
        sigma3 = torch.exp(-s3)
        
        s4 = -torch.log(sigma) + h * c4
        sigma4 = torch.exp(-s4)

        
        x_0 = x.clone()
        
        ci = [c1,c2,c3,c4]
        φ = Phi(h, ci)

        a3_2 = 1/2
        a4_3 = φ(0,2)
        
        b3 = b2 = (1/3) * a4_3
        #b3 = b2 = (1/3) * φ(0,2)
        #b3 = (1/3) * φ(0,2)
        b4 = (1/3)*φ(2) + φ(3) + φ(4) - (5/24)*φ(0,2)

        a = [
                [0, 0,    0, 0],
                [0, 0,    0, 0],
                [0, a3_2, 0, 0],
                [0, 0, a4_3, 0],
        ]
        b = [
                [0, b2, b3, b4,],
        ]

        a, b = gen_first_col_exp(a,b,ci,φ)
        
        a2_1 = a[1][0]
        a3_1 = a[2][0]
        a4_1 = a[3][0]
        a4_2 = a[3][1]
        b1 = b[0][0]
        
        u2_1, u3_1, u4_1, u2_2, u3_2, u4_2, v1, v2, k1_prev, k2_prev = 0,0,0,0,0, 0, 0, 0, 0, 0
        
        if h_prev2 is not None:
            
            φ2 = Phi(h_prev1 * h/h_no_eta, ci)
            
            u2_1 = -2*φ2(2,2) - 2*φ2(3,2)
            u3_1 = -2*φ2(2,2) - 2*φ2(3,2) + 5/8

            u4_1 = -2*φ2(2) - 2*φ2(3) + (5/4)*φ2(0,2)
            
            v1 = -φ2(2) + φ2(3) + 3*φ2(4) + (5/24)*φ2(0,2)
            
            a2_1 -= u2_1
            a3_1 -= u3_1
            a4_1 -= u4_1
            b1 -= v1
            
            k1_prev = denoised_prev1 - x_0
                    
        if h_prev2 is not None:
            
            φ3 = Phi(h_prev2 * h/h_no_eta, ci)
            
            u2_2 = -(1/2)*φ3(2,2) + φ3(3,2)
            u3_2 = (1/2)*φ3(2,2) + φ3(3,2) - 3/16

            u4_2 = (1/2)*φ3(2) + φ3(3) - (3/8)*φ3(0,2)
            
            v2 = (1/6)*φ3(2) - φ3(4) - (1/24)*φ3(0,2)
            
            a2_1 -= u2_2
            a3_1 -= u3_2
            a4_1 -= u4_2
            b1 -= v2
            
            k2_prev = denoised_prev2 - x_0
            
        if h_prev2 is None:
            c1,c2,c3,c4 = 0, 1/2, 1/2, 1
            ci = [c1,c2,c3,c4]
            φ = Phi(h, ci)
            
            a2_1 = c2 * φ(1,2)
            a3_1 = 0
            a3_2 = c3 * φ(1,3)
            a4_1 = (1/2) * φ(1,3) * (φ(0,3) - 1)
            a4_2 = 0
            a4_3 = φ(1,3)
            b1 = φ(1) - 3*φ(2) + 4*φ(3)
            b2 = 2*φ(2) - 4*φ(3)
            b3 = 2*φ(2) - 4*φ(3)
            b4 = 4*φ(3) - φ(2)
            
        x1 = x
        
        denoised1 = model(x1, sigma1 * s_in, **extra_args)
        if sigma_next == 0:
            return denoised1
        k1 = denoised1 - x_0              
        x2 = x_0 + h * (a2_1*k1)   +   h * (u2_1*k1_prev + u2_2*k2_prev)
        
        denoised2 = model(x2, sigma2 * s_in, **extra_args)
        k2 = denoised2 - x_0
        x3 = x_0 + h * (a3_1*k1 + a3_2*k2)   +   h * (u3_1*k1_prev + u3_2*k2_prev)
        
        denoised3 = model(x3, sigma3 * s_in, **extra_args)
        k3 = denoised3 - x_0
        x4 = x_0 + h * (a4_1*k1 + a4_2*k2 + a4_3*k3)   +   h * (u4_1*k1_prev + u4_2*k2_prev)
        
        denoised4 = model(x4, sigma4 * s_in, **extra_args)
        k4 = denoised4 - x_0
        x_down = x_0 + h * (b1*k1 + b2*k2 + b3*k3 + b4*k4)   +   h * (v1*k1_prev + v2*k2_prev)
        
        #denoised = x_0 + ((sigma / (sigma - sigma_down)) *  h) * ((b1*k1 + b2*k2 + b3*k3 + b4*k4) + h*(v1*k1_prev))
        denoised = x_0 + ((sigma / (sigma - sigma_down)) *  h) * ((b1*k1 + b2*k2 + b3*k3 + b4*k4 + v1*k1_prev + v2*k2_prev))
        eps = denoised - x_0
        

        h_prev2, denoised_prev2 = h_prev1, denoised_prev1
        h_prev1, denoised_prev1 = h_no_eta, denoised
                
        noise = noise_sampler(sigma=sigma, sigma_next=sigma_next)
        noise = (noise - noise.mean()) / noise.std()
        
        x = alpha_ratio * x_down + noise * sigma_up
        
        #denoised = x_0 + ((sigma / (sigma - sigma_next)) *  h) * (b1*k1 + b2*k2 + b3*k3 + b4*k4)
        #eps = denoised - x_0

        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised})

    if sigmas[-1] > 0:
        denoised = model(x, sigmas[-1] * s_in, **extra_args)
        
    return denoised







# using GenLawson45 https://ora.ox.ac.uk/objects/uuid:cc001282-4285-4ca2-ad06-31787b540c61/files/m611df1a355ca243beb09824b70e5e774 MULTISTEP! reconsider implementation of U/V, see pg 66/183
def sample_rk_crazymod44(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
    
    h_fn = lambda sigma_next, sigma: -torch.log()
    c1,c2,c3,c4 = 0, 1/2, 1/2, 1

    h_prev1, x_prev1, denoised_prev1, eps_prev1 = None, None, None, None
    h_prev2, x_prev2, denoised_prev2, eps_prev2 = None, None, None, None
    h_prev3, x_prev3, denoised_prev3, eps_prev3 = None, None, None, None
    h_prev4, x_prev4, denoised_prev4, eps_prev4 = None, None, None, None
        
    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
        
        sigma_up, sigma, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, eta, noise_mode)
        
        sigma1 = sigma
        
        h = -torch.log(sigma_down/sigma)
        h_no_eta = -torch.log(sigma_next/sigma)
        
        s2 = -torch.log(sigma) + h * c2
        sigma2 = torch.exp(-s2)
        
        s3 = -torch.log(sigma) + h * c3
        sigma3 = torch.exp(-s3)
        
        s4 = -torch.log(sigma) + h * c4
        sigma4 = torch.exp(-s4)

        
        x_0 = x.clone()
        
        ci = [c1,c2,c3,c4]
        φ = Phi(h, ci)

        a3_2 = 1/2
        a4_3 = φ(0,2)
        
        b3 = b2 = (1/3) * a4_3
        #b3 = b2 = (1/3) * φ(0,2)
        #b3 = (1/3) * φ(0,2)
        b4 = (1/4)*φ(2) + (11/12)*φ(3) + (3/2)*φ(4) + φ(5) - (35/192)*φ(0,2)

        a = [
                [0, 0,    0, 0],
                [0, 0,    0, 0],
                [0, a3_2, 0, 0],
                [0, 0, a4_3, 0],
        ]
        b = [
                [0, b2, b3, b4,],
        ]

        a, b = gen_first_col_exp(a,b,ci,φ)
        
        a2_1 = a[1][0]
        a3_1 = a[2][0]
        a4_1 = a[3][0]
        a4_2 = a[3][1]
        b1 = b[0][0]
        
        u2_1, u3_1, u4_1, u2_2, u3_2, u4_2, v1, v2, k1_prev, k2_prev = 0,0,0,0,0, 0, 0, 0, 0, 0
        u2_3, u3_3, u4_3, v3, k3_prev = 0,0,0,0,0
        
        if h_prev3 is not None:
            
            φ1 = Phi(h_prev1 * h/h_no_eta, ci)
            
            u2_1 = -3*φ1(2,2) - 5*φ1(3,2) - 3*φ1(4,2)
            u3_1 = u2_1 + 35/32

            u4_1 = -3*φ1(2) - 5*φ1(3) - 3*φ1(4) + (35/16)*φ1(0,2)
            
            v1 = -(3/2)*φ1(2) + (1/2)*φ1(3) + 6*φ1(4) + 6*φ1(5) + (35/96)*φ1(0,2)
            
            a2_1 -= u2_1
            a3_1 -= u3_1
            a4_1 -= u4_1
            b1 -= v1
            
            k1_prev = denoised_prev1 - x_0
                    
        if h_prev3 is not None:
            
            φ2 = Phi(h_prev2 * h/h_no_eta, ci)
            
            u2_2 = (3/2)*φ2(2,2) + 4*φ2(3,2) + 3*φ2(4,2)
            u3_2 = u2_2 - 21/32

            u4_2 = (3/2)*φ2(2) + 4*φ2(3) + 3*φ2(4) - (21/16)*φ2(0,2)
            
            v2 = (1/2)*φ2(2) + (1/3)*φ2(3) - 3*φ2(4) - 4*φ2(5) - (7/48)*φ2(0,2)
            
            a2_1 -= u2_2
            a3_1 -= u3_2
            a4_1 -= u4_2
            b1 -= v2
            
            k2_prev = denoised_prev2 - x_0
                    
        if h_prev3 is not None:
            
            φ3 = Phi(h_prev3 * h/h_no_eta, ci)
            
            u2_3 = (-1/3)*φ3(2,2) - φ3(3,2) - φ3(4,2)
            u3_3 = u2_3 + 5/32

            u4_3 = -(1/3)*φ3(2) - φ3(3) - φ3(4) + (5/16)*φ3(0,2)
            
            v3 = -(1/12)*φ3(2) - (1/12)*φ3(3) + (1/2)*φ3(4) + φ3(5) + (5/192)*φ3(0,2)
            
            a2_1 -= u2_3
            a3_1 -= u3_3
            a4_1 -= u4_3
            b1 -= v3
            
            k3_prev = denoised_prev3 - x_0
            
        if h_prev3 is None:
            c1,c2,c3,c4 = 0, 1/2, 1/2, 1
            ci = [c1,c2,c3,c4]
            φ = Phi(h, ci)
            
            a2_1 = c2 * φ(1,2)
            a3_1 = 0
            a3_2 = c3 * φ(1,3)
            #a4_1 = (1/2) * φ(1,3) * φ(0,3 - 1)   # This doesn't work, but it's in the paper: j=0 leads to taking a factorial of a negative with the remainder method, and doesn't work with the analytic solution either
            #a4_1 = (1/2) * φ(1,3) * (torch.exp(-h*c3) - 1) 
            a4_1 = (1/2) * φ(1,3) * (φ(0,3) - 1)
            a4_2 = 0
            a4_3 = φ(1,3)
            b1 = φ(1) - 3*φ(2) + 4*φ(3)
            b2 = 2*φ(2) - 4*φ(3)
            b3 = 2*φ(2) - 4*φ(3)
            b4 = 4*φ(3) - φ(2)
            
        x1 = x
        
        
        denoised1 = model(x1, sigma1 * s_in, **extra_args)
        if sigma_next == 0:
            return denoised1
        k1 = denoised1 - x_0              
        x2 = x_0 + h * (a2_1*k1)   +   h * (u2_1*k1_prev + u2_2*k2_prev + u2_3*k3_prev)
        
        denoised2 = model(x2, sigma2 * s_in, **extra_args)
        k2 = denoised2 - x_0
        x3 = x_0 + h * (a3_1*k1 + a3_2*k2)   +   h * (u3_1*k1_prev + u3_2*k2_prev + u3_3*k3_prev)
        
        denoised3 = model(x3, sigma3 * s_in, **extra_args)
        k3 = denoised3 - x_0
        x4 = x_0 + h * (a4_1*k1 + a4_2*k2 + a4_3*k3)   +   h * (u4_1*k1_prev + u4_2*k2_prev + u4_3*k3_prev)
        
        denoised4 = model(x4, sigma4 * s_in, **extra_args)
        k4 = denoised4 - x_0
        x_down = x_0 + h * (b1*k1 + b2*k2 + b3*k3 + b4*k4)   +   h * (v1*k1_prev + v2*k2_prev + v3*k3_prev)
        
        #denoised = x_0 + ((sigma / (sigma - sigma_down)) *  h) * ((b1*k1 + b2*k2 + b3*k3 + b4*k4) + h*(v1*k1_prev))
        denoised = x_0 + ((sigma / (sigma - sigma_down)) *  h) * ((b1*k1 + b2*k2 + b3*k3 + b4*k4 + v1*k1_prev + v2*k2_prev + v3*k3_prev))
        eps = denoised - x_0
        
        h_prev3, denoised_prev3 = h_prev2, denoised_prev2
        h_prev2, denoised_prev2 = h_prev1, denoised_prev1
        h_prev1, denoised_prev1 = h_no_eta, denoised
                
        noise = noise_sampler(sigma=sigma, sigma_next=sigma_next)
        noise = (noise - noise.mean()) / noise.std()
        
        x = alpha_ratio * x_down + noise * sigma_up
        
        #denoised = x_0 + ((sigma / (sigma - sigma_next)) *  h) * (b1*k1 + b2*k2 + b3*k3 + b4*k4)
        #eps = denoised - x_0

        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised})

    return denoised






# using GenLawson45 https://ora.ox.ac.uk/objects/uuid:cc001282-4285-4ca2-ad06-31787b540c61/files/m611df1a355ca243beb09824b70e5e774 MULTISTEP! reconsider implementation of U/V, see pg 66/183
def sample_rk_crazy(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
    
    h_fn = lambda sigma_next, sigma: -torch.log()
    c1,c2,c3,c4 = 0, 1/2, 1/2, 1

    h_prev, x_prev, denoised_prev, eps_prev = None, None, None, None
    
    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
        
        sigma_up, sigma, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, eta, noise_mode)
        
        sigma1 = sigma
        
        h = -torch.log(sigma_down/sigma)
        h_no_eta = -torch.log(sigma_next/sigma)
        
        s2 = -torch.log(sigma) + h * c2
        sigma2 = torch.exp(-s2)
        
        s3 = -torch.log(sigma) + h * c3
        sigma3 = torch.exp(-s3)
        
        s4 = -torch.log(sigma) + h * c4
        sigma4 = torch.exp(-s4)

        
        x_0 = x.clone()
        
        ci = [c1,c2,c3,c4]
        φ = Phi(h, ci)

        a3_2 = 1/2
        a4_3 = φ(0,2)
        
        b2 = (1/3) * φ(0,2)
        b3 = (1/3) * φ(0,2)
        b4 = (1/2)*φ(2) + φ(3) - (1/4)*φ(0,2)

        a = [
                [0, 0,    0, 0],
                [0, 0,    0, 0],
                [0, a3_2, 0, 0],
                [0, 0, a4_3, 0],
        ]
        b = [
                [0, b2, b3, b4,],
        ]

        a, b = gen_first_col_exp(a,b,ci,φ)
        
        a2_1 = a[1][0]
        a3_1 = a[2][0]
        a4_1 = a[3][0]
        a4_2 = a[3][1]
        b1 = b[0][0]
        
        u2_1, u3_1, u4_1, v1, k1_prev = 0,0,0,0,0
        if h_prev is not None:
            if sigma_next == 0:
                return denoised_prev
            
            φ2 = Phi(h_prev * h/h_no_eta, ci)
            
            u2_1 = -φ2(2,2)
            u3_1 = -φ2(2,2) + 1/4

            u4_1 = -φ2(2) + (1/2)*φ2(0,2)
            
            v1 = -(1/2)*φ2(2) + φ2(3) + (1/12)*φ2(0,2)
            
            a2_1 -= u2_1
            a3_1 -= u3_1
            a4_1 -= u4_1
            b1 -= v1
            
            k1_prev = denoised_prev - x_0
            
        if h_prev is None:
            c1,c2,c3,c4 = 0, 1/2, 1/2, 1
            ci = [c1,c2,c3,c4]
            φ = Phi(h, ci)
            
            a2_1 = c2 * φ(1,2)
            a3_1 = 0
            a3_2 = c3 * φ(1,3)
            a4_1 = (1/2) * φ(1,3) * (φ(0,3) - 1)
            a4_2 = 0
            a4_3 = φ(1,3)
            b1 = φ(1) - 3*φ(2) + 4*φ(3)
            b2 = 2*φ(2) - 4*φ(3)
            b3 = 2*φ(2) - 4*φ(3)
            b4 = 4*φ(3) - φ(2)
            
        x1 = x
        
        denoised1 = model(x1, sigma1 * s_in, **extra_args)
        if sigma_next == 0:
            return denoised1
        k1 = denoised1 - x_0              
        x2 = x_0 + h * (a2_1*k1)   +   h * (u2_1 * k1_prev)
        
        denoised2 = model(x2, sigma2 * s_in, **extra_args)
        k2 = denoised2 - x_0
        x3 = x_0 + h * (a3_1*k1 + a3_2*k2)   +   h * (u3_1 * k1_prev)
        
        denoised3 = model(x3, sigma3 * s_in, **extra_args)
        k3 = denoised3 - x_0
        x4 = x_0 + h * (a4_1*k1 + a4_2*k2 + a4_3*k3)   +   h * (u4_1 * k1_prev)
        
        denoised4 = model(x4, sigma4 * s_in, **extra_args)
        k4 = denoised4 - x_0
        x_down = x_0 + h * (b1*k1 + b2*k2 + b3*k3 + b4*k4)   +   h * (v1 * k1_prev)
        
        #denoised = x_0 + ((sigma / (sigma - sigma_down)) *  h) * ((b1*k1 + b2*k2 + b3*k3 + b4*k4) + h*(v1*k1_prev))
        denoised = x_0 + ((sigma / (sigma - sigma_down)) *  h) * ((b1*k1 + b2*k2 + b3*k3 + b4*k4 + v1*k1_prev))
        eps = denoised - x_0
        
        
        
        h_prev, denoised_prev = h_no_eta, denoised
        
        noise = noise_sampler(sigma=sigma, sigma_next=sigma_next)
        noise = (noise - noise.mean()) / noise.std()
        
        x = alpha_ratio * x_down + noise * sigma_up
        
        #denoised = x_0 + ((sigma / (sigma - sigma_next)) *  h) * (b1*k1 + b2*k2 + b3*k3 + b4*k4)
        #eps = denoised - x_0

        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised})

    return denoised






# using PEC423 https://ora.ox.ac.uk/objects/uuid:cc001282-4285-4ca2-ad06-31787b540c61/files/m611df1a355ca243beb09824b70e5e774 pg138/183
def sample_rk_pec423(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
    
    h_fn = lambda sigma_next, sigma: -torch.log()
    

    h_prev1, x_prev1, denoised_prev1, eps_prev1 = None, None, None, None
    h_prev2, x_prev2, denoised_prev2, eps_prev2 = None, None, None, None    
    
    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
        
        sigma_up, sigma, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, eta, noise_mode)
        
        sigma1 = sigma
        
        h = -torch.log(sigma_down/sigma)
        h_no_eta = -torch.log(sigma_next/sigma)
        
        c1,c2 = 0, 1
        ci = [c1,c2]
        φ = Phi(h, ci)
        
        s2 = -torch.log(sigma) + h * c2
        sigma2 = torch.exp(-s2)

        x_0 = x.clone()

        a2_1 = c2 * φ(1,2)
        b2 = (1/3)*φ(2) + φ(3) + φ(4)
        b1 = φ(1) - b2
        
        u2_1, u2_2, v1, v2, k1_prev, k2_prev = 0,0,0,0,0,0
        if h_prev2 is not None:

            if extra_options_flag("h_prev_h_h_no_eta", extra_options):
                φ = Phi(h_prev1 * h/h_no_eta, ci)
            
            u2_1 = -2*φ(2) - 2*φ(3)
            #u2_2 = (1/2)*φ(2) + φ(3)
            
            v1 = -φ(2) + φ(3) + 3*φ(4)
            #v2 = (1/6)*φ(2) - φ(4)
            
            a2_1 -= u2_1 
            b1 -= v1
            #b2 -= v2
            
            k1_prev = denoised_prev1 - x_0
            #k2_prev = denoised_prev2 - x_0
            
        if h_prev2 is not None:
            if extra_options_flag("h_prev_h_h_no_eta", extra_options):
                φ = Phi(h_prev2 * h/h_no_eta, ci)
            
            #u2_1 = -2*φ(2) - 2*φ(3)
            u2_2 = (1/2)*φ(2) + φ(3)
            
            #v1 = -φ(2) + φ(3) + 3*φ(4)
            v2 = (1/6)*φ(2) - φ(4)
            
            a2_1 -= u2_2 
            #b1 -= v1
            b2 -= v2
            
            #k1_prev = denoised_prev1 - x_0
            k2_prev = denoised_prev2 - x_0



        if h_prev2 is None:
            c2 = float(get_extra_options_kv("c2", str("0.5"), extra_options))

            ci = [0, c2]
            φ = Phi(h, ci)
            
            a2_1 = c2 * φ(1,2)
            b2 = φ(2)/c2
            b1 = φ(1) - b2
            
            s2 = -torch.log(sigma) + h * c2
            sigma2 = torch.exp(-s2)
            
            
        x1 = x
        
        denoised1 = model(x1, sigma1 * s_in, **extra_args)
        if sigma_next == 0:
            return denoised1
        k1 = denoised1 - x_0              
        x2 = x_0 + h * (a2_1*k1)   +   h * (u2_1 * k1_prev + u2_2 * k2_prev)
        
        denoised2 = model(x2, sigma2 * s_in, **extra_args)
        k2 = denoised2 - x_0

        x_down = x_0 + h * (b1*k1 + b2*k2)   +   h * (v1*k1_prev + v2*k2_prev)
        
        #denoised = x_0 + ((sigma / (sigma - sigma_down)) *  h) * ((b1*k1 + b2*k2 + b3*k3 + b4*k4) + h*(v1*k1_prev))
        denoised = x_0 + ((sigma / (sigma - sigma_down)) *  h) * (b1*k1 + b2*k2 + v1*k1_prev + v2*k2_prev)
        eps = denoised - x_0
        
        
        
        h_prev2, denoised_prev2 = h_prev1, denoised_prev1
        h_prev1, denoised_prev1 = h_no_eta, denoised
        
        noise = noise_sampler(sigma=sigma, sigma_next=sigma_next)
        noise = (noise - noise.mean()) / noise.std()
        
        x = alpha_ratio * x_down + noise * sigma_up
        
        #denoised = x_0 + ((sigma / (sigma - sigma_next)) *  h) * (b1*k1 + b2*k2 + b3*k3 + b4*k4)
        #eps = denoised - x_0

        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised})

    return denoised









# using PEC433 https://ora.ox.ac.uk/objects/uuid:cc001282-4285-4ca2-ad06-31787b540c61/files/m611df1a355ca243beb09824b70e5e774 pg138/183
def sample_rk_pec433(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
    
    h_fn = lambda sigma_next, sigma: -torch.log()

    h_prev1, x_prev1, denoised_prev1, eps_prev1 = None, None, None, None
    h_prev2, x_prev2, denoised_prev2, eps_prev2 = None, None, None, None
    h_prev3, x_prev3, denoised_prev3, eps_prev3 = None, None, None, None
    h_prev4, x_prev4, denoised_prev4, eps_prev4 = None, None, None, None
    
    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
        
        sigma_up, sigma, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, eta, noise_mode)
        
        sigma1 = sigma
        
        h = -torch.log(sigma_down/sigma)
        h_no_eta = -torch.log(sigma_next/sigma)
        
        
        c1,c2,c3 = 0, 1, 1
        ci = [c1,c2,c3]
        φ = Phi(h, ci)
        
        s2 = -torch.log(sigma) + h * c2
        sigma2 = torch.exp(-s2)
        
        s3 = -torch.log(sigma) + h * c3
        sigma3 = torch.exp(-s3)

        x_0 = x.clone()

        a2_1 = c2 * φ(1,2)
        a3_2 = (1/3)*φ(2) + φ(3) + φ(4)
        a3_1 = c3 * φ(1,3) - a3_2
        
        b2 = 0
        b3 = (1/3)*φ(2) + φ(3) + φ(4)
        b1 = φ(1) - b2 - b3
        
        u2_1, u2_2, u3_1, u3_2, v1, v2, k1_prev, k2_prev = 0,0,0,0,0,0,0,0
        
        if h_prev2 is not None:

            φ = Phi(h_prev1 * h/h_no_eta, ci)
            
            u2_1 = -2*φ(2) - 2*φ(3)
            u3_1 = -φ(2) + φ(3) + 3*φ(4)
            
            v1 = -φ(2) + φ(3) + 3*φ(4)
            
            a2_1 -= u2_1 
            a3_1 -= u3_1
            b1 -= v1
            
            k1_prev = denoised_prev1 - x_0
            
        if h_prev2 is not None:

            φ = Phi(h_prev2 * h/h_no_eta, ci)
            
            u2_2 = (1/2)*φ(2) + φ(3)
            u3_2 = (1/6)*φ(2) - φ(4)
            
            v2 = (1/6)*φ(2) - φ(4)
            
            a2_1 -= u2_2 
            a3_1 -= u3_2
            b2 -= v2
            
            k2_prev = denoised_prev2 - x_0


        if h_prev2 is None:
            c2 = float(get_extra_options_kv("c2", str("0.5"), extra_options))
            c3 = float(get_extra_options_kv("c3", str("1.0"), extra_options))
            
            gamma = calculate_gamma(c2, c3)
            a2_1 = c2 * phi(1, -h*c2)
            a3_2 = gamma * c2 * phi(2, -h*c2) + (c3 ** 2 / c2) * phi(2, -h*c3) #phi_2_c3_h  # a32 from k2 to k3
            a3_1 = c3 * phi(1, -h*c3) - a3_2 # a31 from k1 to k3
            b3 = (1 / (gamma * c2 + c3)) * phi(2, -h)      
            b2 = gamma * b3  #simplified version of: b2 = (gamma / (gamma * c2 + c3)) * phi_2_h  
            b1 = phi(1, -h) - b2 - b3    
            
            s2 = -torch.log(sigma) + h * c2
            sigma2 = torch.exp(-s2)
            
            s3 = -torch.log(sigma) + h * c3
            sigma3 = torch.exp(-s3)  
            
        x1 = x
        
        denoised1 = model(x1, sigma1 * s_in, **extra_args)
        if sigma_next == 0:
            return denoised1
        k1 = denoised1 - x_0            
        x2 = x_0 + h * (a2_1*k1)   +   h * (u2_1*k1_prev + u2_2*k2_prev)
        
        denoised2 = model(x2, sigma2 * s_in, **extra_args)
        k2 = denoised2 - x_0
        x3 = x_0 + h * (a3_1*k1 + a3_2*k2)   +   h * (u3_1*k1_prev + u3_2*k2_prev)
        
        denoised3 = model(x3, sigma3 * s_in, **extra_args)
        k3 = denoised3 - x_0
        x_down = x_0 + h * (b1*k1 + b2*k2 + b3*k3)   +   h * (v1*k1_prev + v2*k2_prev)  
        #x4 = x_0 + h * (a4_1*k1 + a4_2*k2 + a4_3*k3)     +   h * (u4_1*k1_prev + u4_2*k2_prev + u4_3*k3_prev + u4_4*k4_prev)
        
        
        #denoised = x_0 + ((sigma / (sigma - sigma_down)) *  h) * ((b1*k1 + b2*k2 + b3*k3 + b4*k4) + h*(v1*k1_prev))
        denoised = x_0 + ((sigma / (sigma - sigma_down)) *  h) * (b1*k1 + b2*k2 + b3*k3 + v1*k1_prev + v2*k2_prev)
        eps = denoised - x_0
        
        
        
        h_prev4, denoised_prev4 = h_prev3, denoised_prev3
        h_prev3, denoised_prev3 = h_prev2, denoised_prev2
        h_prev2, denoised_prev2 = h_prev1, denoised_prev1
        h_prev1, denoised_prev1 = h_no_eta, denoised
        
        
        noise = noise_sampler(sigma=sigma, sigma_next=sigma_next)
        noise = (noise - noise.mean()) / noise.std()
        
        x = alpha_ratio * x_down + noise * sigma_up
        
        #denoised = x_0 + ((sigma / (sigma - sigma_next)) *  h) * (b1*k1 + b2*k2 + b3*k3 + b4*k4)
        #eps = denoised - x_0

        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised})

    return denoised









# using ABNorsett4 https://ora.ox.ac.uk/objects/uuid:cc001282-4285-4ca2-ad06-31787b540c61/files/m611df1a355ca243beb09824b70e5e774 pg130/183
def sample_rk_abnorsett4(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
    
    h_fn = lambda sigma_next, sigma: -torch.log()
    c1,c2 = 0, 1

    h_prev1, x_prev1, denoised_prev1, eps_prev1 = None, None, None, None
    h_prev2, x_prev2, denoised_prev2, eps_prev2 = None, None, None, None
    h_prev3, x_prev3, denoised_prev3, eps_prev3 = None, None, None, None
    h_prev4, x_prev4, denoised_prev4, eps_prev4 = None, None, None, None
    k1,k2,k3 = 0,0,0
    
    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
        
        sigma_up, sigma, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, eta, noise_mode)
        
        sigma1 = sigma
        
        h = -torch.log(sigma_down/sigma)
        h_no_eta = -torch.log(sigma_next/sigma)

        x_0 = x.clone()
        
        ci = [c1,c2]
        φ = Phi(h, ci)
        
        b1 = phi(1, -h)
        b2 = 0
        b3 = 0

        v1, v2, v3, k1_prev, k2_prev, k3_prev = 0,0,0,0,0,0
        if h_prev3 is not None:

            φ = Phi(h_prev1 * h/h_no_eta, ci)
            v1 = -3*φ(2) - 5*φ(3) - 3*φ(4)
            
            φ = Phi(h_prev2 * h/h_no_eta, ci)
            v2 = (3/2)*φ(2) + 4*φ(3) + 3*φ(4)

            φ = Phi(h_prev2 * h/h_no_eta, ci)
            v3 = -(1/3)*φ(2) - φ(3) - φ(4)
            
            k1_prev = denoised_prev1 - x_0
            k2_prev = denoised_prev2 - x_0
            k3_prev = denoised_prev3 - x_0
        
            b1 = b1 - v1 - v2 - v3
        
        
        x1 = x
        
        denoised1 = model(x1, sigma1 * s_in, **extra_args)
        if sigma_next == 0:
            return denoised1
        k1 = denoised1 - x_0              


        x_down = x_0 + h * (b1*k1)   +   h * (v1*k1_prev + v2*k2_prev + v3*k3_prev)
        
        denoised = x_0 + ((sigma / (sigma - sigma_down)) *  h) * (b1*k1 + v1*k1_prev + v2*k2_prev + v3*k3_prev)
        eps = denoised - x_0
        
        
        
        h_prev4, denoised_prev4 = h_prev3, denoised_prev3
        h_prev3, denoised_prev3 = h_prev2, denoised_prev2
        h_prev2, denoised_prev2 = h_prev1, denoised_prev1
        h_prev1, denoised_prev1 = h_no_eta, denoised
        
        noise = noise_sampler(sigma=sigma, sigma_next=sigma_next)
        noise = (noise - noise.mean()) / noise.std()
        
        x = alpha_ratio * x_down + noise * sigma_up
        

        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised})

    return denoised




# using GenLawson45 https://ora.ox.ac.uk/objects/uuid:cc001282-4285-4ca2-ad06-31787b540c61/files/m611df1a355ca243beb09824b70e5e774 MULTISTEP! reconsider implementation of U/V, see pg 66/183
def sample_rk_crazy_old(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
    
    h_fn = lambda sigma_next, sigma: -torch.log()
    c1,c2,c3,c4 = 0, 1/2, 1/2, 1

    h_prev, x_prev, denoised_prev, eps_prev = None, None, None, None
    
    for step in trange(1, len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
        
        sigma_up, sigma, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, eta, noise_mode)
        
        sigma1 = sigma
        
        h = -torch.log(sigma_down/sigma)
        h_no_eta = -torch.log(sigma_next/sigma)
        
        s2 = -torch.log(sigma) + h * c2
        sigma2 = torch.exp(-s2)
        
        s3 = -torch.log(sigma) + h * c3
        sigma3 = torch.exp(-s3)
        
        s4 = -torch.log(sigma) + h * c4
        sigma4 = torch.exp(-s4)

        
        x_0 = x.clone()
        
        ci = [c1,c2,c3,c4]
        φ = Phi(h, ci)

        a3_2 = 1/2
        a4_3 = φ(0,2)
        
        b2 = (1/3) * φ(0,2)
        b3 = (1/3) * φ(0,2)
        b4 = (1/2)*φ(2) + φ(3) - (1/4)*φ(0,2)

        a = [
                [0, 0,    0, 0],
                [0, 0,    0, 0],
                [0, a3_2, 0, 0],
                [0, 0, a4_3, 0],
        ]
        b = [
                [0, b2, b3, b4,],
        ]

        a, b = gen_first_col_exp(a,b,ci,φ)
        
        a2_1 = a[1][0]
        a3_1 = a[2][0]
        a4_1 = a[3][0]
        a4_2 = a[3][1]
        b1 = b[0][0]
        
        x1 = x
        if h_prev is not None:
            if sigma_next == 0:
                return denoised_prev
            
            φ2 = Phi(h_prev, ci)
            
            u2_1 = -φ2(2,2)
            u3_1 = -φ2(2,2) + 1/4

            u4_1 = -φ2(2) + (1/2)*φ2(0,2)
            
            v1 = -(1/2)*φ2(2) + φ2(3) + (1/12)*φ2(0,2)
            
            a2_1 -= u2_1
            a3_1 -= u3_1
            a4_1 -= u4_1
            b1 -= v1
            
            k1 = denoised_prev - x_0
        else:
            denoised1 = model(x1, sigma1 * s_in, **extra_args)
            if sigma_next == 0:
                return denoised1
            k1 = denoised1 - x_0      
                  
        x2 = x_0 + h * (a2_1*k1)
        
        denoised2 = model(x2, sigma2 * s_in, **extra_args)
        k2 = denoised2 - x_0
        x3 = x_0 + h * (a3_1*k1 + a3_2*k2)
        
        denoised3 = model(x3, sigma3 * s_in, **extra_args)
        k3 = denoised3 - x_0
        x4 = x_0 + h * (a4_1*k1 + a4_2*k2 + a4_3*k3)
        
        denoised4 = model(x4, sigma4 * s_in, **extra_args)
        k4 = denoised4 - x_0
        x_down = x_0 + h * (b1*k1 + b2*k2 + b3*k3 + b4*k4)
        
        denoised = x_0 + ((sigma / (sigma - sigma_down)) *  h) * (b1*k1 + b2*k2 + b3*k3 + b4*k4)
        eps = denoised - x_0
        
        
        
        h_prev, denoised_prev = h_no_eta, denoised
        
        noise = noise_sampler(sigma=sigma, sigma_next=sigma_next)
        noise = (noise - noise.mean()) / noise.std()
        
        x = alpha_ratio * x_down + noise * sigma_up
        
        #denoised = x_0 + ((sigma / (sigma - sigma_next)) *  h) * (b1*k1 + b2*k2 + b3*k3 + b4*k4)
        #eps = denoised - x_0

        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised})

    return denoised





# using GenLawson45 https://ora.ox.ac.uk/objects/uuid:cc001282-4285-4ca2-ad06-31787b540c61/files/m611df1a355ca243beb09824b70e5e774 MULTISTEP! reconsider implementation of U/V, see pg 66/183
def sample_rk_crazymod45(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
    
    h_fn = lambda sigma_next, sigma: -torch.log()
    
    h_prev1, x_prev1, denoised_prev1, eps_prev1 = None, None, None, None
    h_prev2, x_prev2, denoised_prev2, eps_prev2 = None, None, None, None
    h_prev3, x_prev3, denoised_prev3, eps_prev3 = None, None, None, None
    h_prev4, x_prev4, denoised_prev4, eps_prev4 = None, None, None, None

    
    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
        sigma_up, sigma, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, eta, noise_mode)
        h = -torch.log(sigma_down/sigma)
        h_no_eta = -torch.log(sigma_next/sigma)
        
        sigma1 = sigma
        
        c1,c2,c3,c4 = 0, 1/2, 1/2, 1
        
        s2 = -torch.log(sigma) + h * c2
        sigma2 = torch.exp(-s2)
        
        s3 = -torch.log(sigma) + h * c3
        sigma3 = torch.exp(-s3)
        
        s4 = -torch.log(sigma) + h * c4
        sigma4 = torch.exp(-s4)
        
        x_0 = x.clone()
        
        ci = [c1,c2,c3,c4]
        φ = Phi(h, ci)

        a3_2 = 1/2
        a4_3 = φ(0,2)
        
        b2 = (1/3) * φ(0,2)
        b3 = (1/3) * φ(0,2)
        b4 = (12/59)*φ(2) + (50/59)*φ(3) + (105/59)*φ(4) + (120/59)*φ(5) - (60/59)*φ(6) - (157/944)*φ(0,2)

        a = [
                [0, 0,    0, 0],
                [0, 0,    0, 0],
                [0, a3_2, 0, 0],
                [0, 0, a4_3, 0],
        ]
        b = [
                [0, b2, b3, b4,],
        ]

        a, b = gen_first_col_exp(a,b,ci,φ)
        
        a2_1 = a[1][0]
        a2_2 = 0
        a2_3 = 0
        a2_4 = 0
        
        a3_1 = a[2][0]
        a3_3 = 0
        a3_4 = 0
        
        a4_1 = a[3][0]
        a4_2 = a[3][1]
        a4_4 = 0
        
        b1 = b[0][0]
        
        x1 = x
        
        u2_1 = -4*φ(2,2) - (26/3)*φ(3,2) - 9*φ(4,2) - 4*φ(5,2)
        u2_2 = 3*φ(2,2) + (19/2)*φ(3,2) + 12*φ(4,2) + 6*φ(5,2)
        u2_3 = -(4/3)*φ(2,2) - (14/3)*φ(3,2) - 7*φ(4,2) - 4*φ(5,2)
        u2_4 = (1/4)*φ(2,2) + (88/96)*φ(3,2) + (3/2)*φ(4,2) + φ(5,2)
        
        u4_1 = -4*φ(2) - (26/3)*φ(3) - 9*φ(4) - 4*φ(5) + (105/32)*φ(0,2)
        u4_2 = 3*φ(2) + (19/2)*φ(3) + 12*φ(4) + 6*φ(5) - (189/64)*φ(0,2)
        u4_3 = -(4/3)*φ(2) - (14/3)*φ(3) - 7*φ(4) - 4*φ(5) +(45/32)*φ(0,2)
        u4_4 = (1/4)*φ(2) + (11/12)*φ(3) + (3/2)*φ(4) + φ(5) - (35/128)*φ(0,2)
        
        u3_1 = u2_1 + 105/64
        u3_2 = u2_2 - 189/128
        u3_3 = u2_3 + 45/64
        u3_4 = u2_4 - 35/256
        
        v1 = -(116/59)*φ(2) -  (34/177)*φ(3) + (519/59)*φ(4) + (964/59)*φ(5) - (600/59)*φ(6) +   (495/944)*φ(0,2)
        v2 =   (57/59)*φ(2) + (121/118)*φ(3) - (342/59)*φ(4) - (846/59)*φ(5) + (600/59)*φ(6) -  (577/1888)*φ(0,2)
        v3 = -(56/177)*φ(2) -  (76/177)*φ(3) + (112/59)*φ(4) + (364/59)*φ(5) - (300/59)*φ(6) +    (25/236)*φ(0,2)
        v4 =  (11/236)*φ(2) +  (49/708)*φ(3) - (33/118)*φ(4) -  (61/59)*φ(5) + ( 60/59)*φ(6) - (181/11328)*φ(0,2)
        
        k1_prev, k2_prev, k3_prev, k4_prev = 0,0,0,0
        u2_1,u2_2,u2_3,u2_4=0,0,0,0
        u3_1,u3_2,u3_3,u3_4=0,0,0,0
        u4_1,u4_2,u4_3,u4_4=0,0,0,0
        v1,v2,v3,v4=0,0,0,0
        
        #h_prev2, x_prev2, denoised_prev2, eps_prev2 = None, None, None, None
        #h_prev3, x_prev3, denoised_prev3, eps_prev3 = None, None, None, None
        #h_prev4, x_prev4, denoised_prev4, eps_prev4 = None, None, None, None
        x1 = x
        if h_prev4 is not None:
            #φ1 = Phi(h_prev1, ci)
            φ1 = Phi(h_prev1 * h/h_no_eta, ci)
            
            u2_1 = -4*φ1(2,2) - (26/3)*φ1(3,2) - 9*φ1(4,2) - 4*φ1(5,2)
            u3_1 = u2_1 + 105/64
            u4_1 = -4*φ1(2) - (26/3)*φ1(3) - 9*φ1(4) - 4*φ1(5) + (105/32)*φ1(0,2)
            
            v1 = -(116/59)*φ1(2) -  (34/177)*φ1(3) + (519/59)*φ1(4) + (964/59)*φ1(5) - (600/59)*φ1(6) +   (495/944)*φ1(0,2)
            
            a2_1 -= u2_1
            a3_1 -= u3_1
            a4_1 -= u4_1
            b1 -= v1
            
            k1_prev = denoised_prev1 - x_0

        

        if h_prev4 is not None:
            #φ2 = Phi(h_prev2, ci)
            φ2 = Phi(h_prev2 * h/h_no_eta, ci)
            
            u2_2 = 3*φ2(2,2) + (19/2)*φ2(3,2) + 12*φ2(4,2) + 6*φ2(5,2)
            u3_2 = u2_2 - 189/128
            u4_2 = 3*φ2(2) + (19/2)*φ2(3) + 12*φ2(4) + 6*φ2(5) - (189/64)*φ2(0,2)

            v2 =  (57/59)*φ2(2) + (121/118)*φ2(3) - (342/59)*φ2(4) - (846/59)*φ2(5) + (600/59)*φ2(6) -  (577/1888)*φ2(0,2)

            a2_1 -= u2_2
            a3_1 -= u3_2
            a4_1 -= u4_2
            b1 -= v2
            
            k2_prev = denoised_prev2 - x_0
    
        

        if h_prev4 is not None:
            #φ3 = Phi(h_prev3, ci)
            φ3 = Phi(h_prev3 * h/h_no_eta, ci)
            
            u2_3 = -(4/3)*φ3(2,2) - (14/3)*φ3(3,2) - 7*φ3(4,2) - 4*φ3(5,2)
            u3_3 = u2_3 + 45/64
            u4_3 = -(4/3)*φ3(2) - (14/3)*φ3(3) - 7*φ3(4) - 4*φ3(5) +(45/32)*φ3(0,2)

            v3 = -(56/177)*φ3(2) -  (76/177)*φ3(3) + (112/59)*φ3(4) + (364/59)*φ3(5) - (300/59)*φ3(6) +    (25/236)*φ3(0,2)

            a2_1 -= u2_3
            a3_1 -= u3_3
            a4_1 -= u4_3
            b1 -= v3
            
            k3_prev = denoised_prev3 - x_0


        if h_prev4 is not None:
            #φ4 = Phi(h_prev4, ci)
            φ4 = Phi(h_prev4 * h/h_no_eta, ci)
            
            u2_4 = (1/4)*φ4(2,2) + (88/96)*φ4(3,2) + (3/2)*φ4(4,2) + φ4(5,2)
            u3_4 = u2_4 - 35/256
            u4_4 = (1/4)*φ4(2) + (11/12)*φ4(3) + (3/2)*φ4(4) + φ4(5) - (35/128)*φ4(0,2)

            v4 =  (11/236)*φ4(2) +  (49/708)*φ4(3) - (33/118)*φ4(4) -  (61/59)*φ4(5) + ( 60/59)*φ4(6) - (181/11328)*φ4(0,2)

            a2_1 -= u2_4
            a3_1 -= u3_4
            a4_1 -= u4_4
            b1 -= v4
            
            k4_prev = denoised_prev4 - x_0
        
        """denoised2 = model(x2, sigma2 * s_in, **extra_args)
        k2 = denoised2 - x_0
        x3 = x_0 + h * (a3_1*k1 + a3_2*k2)
        
        denoised3 = model(x3, sigma3 * s_in, **extra_args)
        k3 = denoised3 - x_0
        x4 = x_0 + h * (a4_1*k1 + a4_2*k2 + a4_3*k3)"""
        
        denoised1 = model(x1, sigma1 * s_in, **extra_args)
        if sigma_next == 0:
            return denoised1
        k1 = denoised1 - x_0            
        x2 = x_0 + h * (a2_1*k1)   +   h * (u2_1*k1_prev + u2_2*k2_prev + u2_3*k3_prev + u2_4*k4_prev)
        
        denoised2 = model(x2, sigma2 * s_in, **extra_args)
        k2 = denoised2 - x_0
        x3 = x_0 + h * (a3_1*k1 + a3_2*k2)   +   h * (u3_1*k1_prev + u3_2*k2_prev + u3_3*k3_prev + u3_4*k4_prev)
        
        denoised3 = model(x3, sigma3 * s_in, **extra_args)
        k3 = denoised3 - x_0
        x4 = x_0 + h * (a4_1*k1 + a4_2*k2 + a4_3*k3)     +   h * (u4_1*k1_prev + u4_2*k2_prev + u4_3*k3_prev + u4_4*k4_prev)
        
        denoised4 = model(x4, sigma4 * s_in, **extra_args)
        k4 = denoised4 - x_0
        x_down = x_0 + h * (b1*k1 + b2*k2 + b3*k3 + b4*k4)   +   h * (v1*k1_prev + v2*k2_prev + v3*k3_prev + v4*k4_prev)
        
        
        denoised = x_0   +   ((sigma / (sigma - sigma_down)) * h) * (b1*k1 + b2*k2 + b3*k3 + b4*k4 + v1*k1_prev + v2*k2_prev + v3*k3_prev + v4*k4_prev) 
        eps = denoised - x_0
        
        h_prev4, denoised_prev4 = h_prev3, denoised_prev3
        h_prev3, denoised_prev3 = h_prev2, denoised_prev2
        h_prev2, denoised_prev2 = h_prev1, denoised_prev1
        h_prev1, denoised_prev1 = h_no_eta, denoised
        
        
        
        noise = noise_sampler(sigma=sigma, sigma_next=sigma_next)
        noise = (noise - noise.mean()) / noise.std()
        
        x = alpha_ratio * x_down + noise * sigma_up
        

        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised})

    return denoised






# using GenLawson45 https://ora.ox.ac.uk/objects/uuid:cc001282-4285-4ca2-ad06-31787b540c61/files/m611df1a355ca243beb09824b70e5e774 MULTISTEP! reconsider implementation of U/V, see pg 66/183
def sample_rk_crazy2(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
    
    h_fn = lambda sigma_next, sigma: -torch.log()
    c1,c2,c3,c4 = 0, 1/2, 1/2, 1

    h_prev1, x_prev1, denoised_prev1, eps_prev1 = None, None, None, None
    h_prev2, x_prev2, denoised_prev2, eps_prev2 = None, None, None, None
    h_prev3, x_prev3, denoised_prev3, eps_prev3 = None, None, None, None
    h_prev4, x_prev4, denoised_prev4, eps_prev4 = None, None, None, None

    
    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
        sigma_up, sigma, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, eta, noise_mode)
        h = -torch.log(sigma_down/sigma)
        h_no_eta = -torch.log(sigma_next/sigma)
        
        sigma1 = sigma
        
        s2 = -torch.log(sigma) + h * c2
        sigma2 = torch.exp(-s2)
        
        s3 = -torch.log(sigma) + h * c3
        sigma3 = torch.exp(-s3)
        
        s4 = -torch.log(sigma) + h * c4
        sigma4 = torch.exp(-s4)
        
        x_0 = x.clone()
        
        ci = [c1,c2,c3,c4]
        φ = Phi(h, ci)

        a3_2 = 1/2
        a4_3 = φ(0,2)
        
        b2 = (1/3) * φ(0,2)
        b3 = (1/3) * φ(0,2)
        b4 = 1/6

        a = [
                [0, 0,    0, 0],
                [0, 0,    0, 0],
                [0, a3_2, 0, 0],
                [0, 0, a4_3, 0],
        ]
        b = [
                [0, b2, b3, b4,],
        ]

        a, b = gen_first_col_exp(a,b,ci,φ)
        
        a2_1 = a[1][0]
        a2_2 = 0
        a2_3 = 0
        a2_4 = 0
        
        a3_1 = a[2][0]
        a3_3 = 0
        a3_4 = 0
        
        a4_1 = a[3][0]
        a4_2 = a[3][1]
        a4_4 = 0
        
        b1 = b[0][0]
        
        x1 = x
        
        u2_1 = -4*φ(2,2) - (26/3)*φ(3,2) - 9*φ(4,2) - 4*φ(5,2)
        u2_2 = 3*φ(2,2) + (19/2)*φ(3,2) + 12*φ(4,2) + 6*φ(5,2)
        u2_3 = -(4/3)*φ(2,2) - (14/3)*φ(3,2) - 7*φ(4,2) - 4*φ(5,2)
        u2_4 = (1/4)*φ(2,2) + (88/96)*φ(3,2) + (3/2)*φ(4,2) + φ(5,2)
        
        u4_1 = -4*φ(2) - (26/3)*φ(3) - 9*φ(4) - 4*φ(5) + (105/32)*φ(0,2)
        u4_2 = 3*φ(2) + (19/2)*φ(3) + 12*φ(4) + 6*φ(5) - (189/64)*φ(0,2)
        u4_3 = -(4/3)*φ(2) - (14/3)*φ(3) - 7*φ(4) - 4*φ(5) +(45/32)*φ(0,2)
        u4_4 = (1/4)*φ(2) + (11/12)*φ(3) + (3/2)*φ(4) + φ(5) - (35/128)*φ(0,2)
        
        u3_1 = u2_1 + 105/64
        u3_2 = u2_2 - 189/128
        u3_3 = u2_3 + 45/64
        u3_4 = u2_4 - 35/256
        
        v1 = -4*φ(2) - (26/3)*φ(3) - 9*φ(4) - 4*φ(5) + (35/16)*φ(0,2) + 5/3
        v2 = 3*φ(2) + (19/2)*φ(3) + 12*φ(4) + 6*φ(5) - (63/32)*φ(0,2) - 5/3
        v3 = -(4/3)*φ(2) - (14/3)*φ(3) - 7*φ(4) - 4*φ(5) + (15/16)*φ(0,2) + 5/6
        v4 = (1/4)*φ(2) + (11/12)*φ(3) + (3/2)*φ(4) + φ(5) - (35/192)*φ(0,2) - 1/6
        
        k1_prev, k2_prev, k3_prev, k4_prev = 0,0,0,0
        u2_1,u2_2,u2_3,u2_4=0,0,0,0
        u3_1,u3_2,u3_3,u3_4=0,0,0,0
        u4_1,u4_2,u4_3,u4_4=0,0,0,0
        v1,v2,v3,v4=0,0,0,0
        
        #h_prev2, x_prev2, denoised_prev2, eps_prev2 = None, None, None, None
        #h_prev3, x_prev3, denoised_prev3, eps_prev3 = None, None, None, None
        #h_prev4, x_prev4, denoised_prev4, eps_prev4 = None, None, None, None
        x1 = x
        if h_prev4 is not None:
            φ1 = Phi(h_prev1, ci)
            
            u2_1 = -4*φ1(2,2) - (26/3)*φ1(3,2) - 9*φ1(4,2) - 4*φ1(5,2)
            u3_1 = u2_1 + 105/64
            u4_1 = -4*φ1(2) - (26/3)*φ1(3) - 9*φ1(4) - 4*φ1(5) + (105/32)*φ1(0,2)
            
            v1 = -4*φ1(2) - (26/3)*φ1(3) - 9*φ1(4) - 4*φ1(5) + (35/16)*φ1(0,2) + 5/3
            
            a2_1 -= u2_1
            a3_1 -= u3_1
            a4_1 -= u4_1
            b1 -= v1
            
            k1_prev = denoised_prev1 - x_0

        

        if h_prev4 is not None:
            φ2 = Phi(h_prev2, ci)
            
            u2_2 = 3*φ2(2,2) + (19/2)*φ2(3,2) + 12*φ2(4,2) + 6*φ2(5,2)
            u3_2 = u2_2 - 189/128
            u4_2 = 3*φ2(2) + (19/2)*φ2(3) + 12*φ2(4) + 6*φ2(5) - (189/64)*φ2(0,2)

            v2 = 3*φ2(2) + (19/2)*φ2(3) + 12*φ2(4) + 6*φ2(5) - (63/32)*φ2(0,2) - 5/3

            a2_1 -= u2_2
            a3_1 -= u3_2
            a4_1 -= u4_2
            b1 -= v2
            
            k2_prev = denoised_prev2 - x_0
    
        

        if h_prev4 is not None:
            φ3 = Phi(h_prev3, ci)
            
            u2_3 = -(4/3)*φ3(2,2) - (14/3)*φ3(3,2) - 7*φ3(4,2) - 4*φ3(5,2)
            u3_3 = u2_3 + 45/64
            u4_3 = -(4/3)*φ3(2) - (14/3)*φ3(3) - 7*φ3(4) - 4*φ3(5) +(45/32)*φ3(0,2)

            v3 = -(4/3)*φ3(2) - (14/3)*φ3(3) - 7*φ3(4) - 4*φ3(5) + (15/16)*φ3(0,2) + 5/6

            a2_1 -= u2_3
            a3_1 -= u3_3
            a4_1 -= u4_3
            b1 -= v3
            
            k3_prev = denoised_prev3 - x_0


        if h_prev4 is not None:
            φ4 = Phi(h_prev4, ci)
            
            u2_4 = (1/4)*φ4(2,2) + (88/96)*φ4(3,2) + (3/2)*φ4(4,2) + φ4(5,2)
            u3_4 = u2_4 - 35/256
            u4_4 = (1/4)*φ4(2) + (11/12)*φ4(3) + (3/2)*φ4(4) + φ4(5) - (35/128)*φ4(0,2)

            v4 = (1/4)*φ4(2) + (11/12)*φ4(3) + (3/2)*φ4(4) + φ4(5) - (35/192)*φ4(0,2) - 1/6

            a2_1 -= u2_4
            a3_1 -= u3_4
            a4_1 -= u4_4
            b1 -= v4
            
            k4_prev = denoised_prev4 - x_0
        
        """denoised2 = model(x2, sigma2 * s_in, **extra_args)
        k2 = denoised2 - x_0
        x3 = x_0 + h * (a3_1*k1 + a3_2*k2)
        
        denoised3 = model(x3, sigma3 * s_in, **extra_args)
        k3 = denoised3 - x_0
        x4 = x_0 + h * (a4_1*k1 + a4_2*k2 + a4_3*k3)"""
        
        denoised1 = model(x1, sigma1 * s_in, **extra_args)
        if sigma_next == 0:
            return denoised1
        k1 = denoised1 - x_0            
        x2 = x_0 + h * (a2_1*k1)   +   h * (u2_1*k1_prev + u2_2*k2_prev + u2_3*k3_prev + u2_4*k4_prev)
        
        denoised2 = model(x2, sigma2 * s_in, **extra_args)
        k2 = denoised2 - x_0
        x3 = x_0 + h * (a3_1*k1 + a3_2*k2)   +   h * (u3_1*k1_prev + u3_2*k2_prev + u3_3*k3_prev + u3_4*k4_prev)
        
        denoised3 = model(x3, sigma3 * s_in, **extra_args)
        k3 = denoised3 - x_0
        x4 = x_0 + h * (a4_1*k1 + a4_2*k2 + a4_3*k3)     +   h * (u4_1*k1_prev + u4_2*k2_prev + u4_3*k3_prev + u4_4*k4_prev)
        
        denoised4 = model(x4, sigma4 * s_in, **extra_args)
        k4 = denoised4 - x_0
        x_down = x_0 + h * (b1*k1 + b2*k2 + b3*k3 + b4*k4)   +   h * (v1*k1_prev + v2*k2_prev + v3*k3_prev + v4*k4_prev)
        
        
        denoised = x_0   +   ((sigma / (sigma - sigma_down)) * h) * (b1*k1 + b2*k2 + b3*k3 + b4*k4)   +   ((sigma / (sigma - sigma_down)) * h) + (v1*k1_prev + v2*k2_prev + v3*k3_prev + v4*k4_prev) 
        eps = denoised - x_0
        
        h_prev4, denoised_prev4 = h_prev3, denoised_prev3
        h_prev3, denoised_prev3 = h_prev2, denoised_prev2
        h_prev2, denoised_prev2 = h_prev1, denoised_prev1
        h_prev1, denoised_prev1 = h_no_eta, denoised
        
        
        
        noise = noise_sampler(sigma=sigma, sigma_next=sigma_next)
        noise = (noise - noise.mean()) / noise.std()
        
        x = alpha_ratio * x_down + noise * sigma_up
        

        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised})

    return denoised







# using GenLawson45 https://ora.ox.ac.uk/objects/uuid:cc001282-4285-4ca2-ad06-31787b540c61/files/m611df1a355ca243beb09824b70e5e774 MULTISTEP! reconsider implementation of U/V, see pg 66/183
def sample_rk_crazy2_botched(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
    
    h_fn = lambda sigma_next, sigma: -torch.log()
    c1,c2,c3,c4 = 0, 1/2, 1/2, 1

    h_prev1, x_prev1, denoised_prev1, eps_prev1 = None, None, None, None
    h_prev2, x_prev2, denoised_prev2, eps_prev2 = None, None, None, None
    h_prev3, x_prev3, denoised_prev3, eps_prev3 = None, None, None, None
    h_prev4, x_prev4, denoised_prev4, eps_prev4 = None, None, None, None

    
    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
        sigma_up, sigma, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, eta, noise_mode)
        h = -torch.log(sigma_down/sigma)
        h_no_eta = -torch.log(sigma_next/sigma)
        
        sigma1 = sigma
        
        s2 = -torch.log(sigma) + h * c2
        sigma2 = torch.exp(-s2)
        
        s3 = -torch.log(sigma) + h * c3
        sigma3 = torch.exp(-s3)
        
        s4 = -torch.log(sigma) + h * c4
        sigma4 = torch.exp(-s4)
        
        x_0 = x.clone()
        
        ci = [c1,c2,c3,c4]
        φ = Phi(h, ci)

        a3_2 = 1/2
        a4_3 = φ(0,2)
        
        b2 = (1/3) * φ(0,2)
        b3 = (1/3) * φ(0,2)
        b4 = 1/6

        a = [
                [0, 0,    0, 0],
                [0, 0,    0, 0],
                [0, a3_2, 0, 0],
                [0, 0, a4_3, 0],
        ]
        b = [
                [0, b2, b3, b4,],
        ]

        a, b = gen_first_col_exp(a,b,ci,φ)
        
        a2_1 = a[1][0]
        a3_1 = a[2][0]
        a4_1 = a[3][0]
        a4_2 = a[3][1]
        b1 = b[0][0]
        
        x1 = x
        
        u2_1 = -4*φ(2,2) - (26/3)*φ(3,2) - 9*φ(4,2) - 4*φ(5,2)
        u2_2 = 3*φ(2,2) + (19/2)*φ(3,2) + 12*φ(4,2) + 6*φ(5,2)
        u2_3 = -(4/3)*φ(2,2) - (14/3)*φ(3,2) - 7*φ(4,2) - 4*φ(5,2)
        u2_4 = (1/4)*φ(2,2) + (88/96)*φ(3,2) + (3/2)*φ(4,2) + φ(5,2)
        
        u4_1 = -4*φ(2) - (26/3)*φ(3) - 9*φ(4) - 4*φ(5) + (105/32)*φ(0,2)
        u4_2 = 3*φ(2) + (19/2)*φ(3) + 12*φ(4) + 6*φ(5) - (189/64)*φ(0,2)
        u4_3 = -(4/3)*φ(2) - (14/3)*φ(3) - 7*φ(4) - 4*φ(5) +(45/32)*φ(0,2)
        u4_4 = (1/4)*φ(2) + (11/12)*φ(3) + (3/2)*φ(4) + φ(5) - (35/128)*φ(0,2)
        
        u3_1 = u2_1 + 105/64
        u3_2 = u2_2 - 189/128
        u3_3 = u2_3 + 45/64
        u3_4 = u2_4 - 35/256
        
        v1 = -4*φ(2) - (26/3)*φ(3) - 9*φ(4) - 4*φ(5) + (35/16)*φ(0,2) + 5/3
        v2 = 3*φ(2) + (19/2)*φ(3) + 12*φ(4) + 6*φ(5) - (63/32)*φ(0,2) - 5/3
        v3 = -(4/3)*φ(2) - (14/3)*φ(3) - 7*φ(4) - 4*φ(5) + (15/16)*φ(0,2) + 5/6
        v4 = (1/4)*φ(2) + (11/12)*φ(3) + (3/2)*φ(4) + φ(5) - (35/192)*φ(0,2) - 1/6
        
        
        x1 = x
        if h_prev1 is not None:
            if sigma_next == 0:
                return denoised_prev1
            
            φ1 = Phi(h_prev1, ci)
            
            u2_1 = -4*φ1(2,2) - (26/3)*φ1(3,2) - 9*φ1(4,2) - 4*φ1(5,2)
            u3_1 = u2_1 + 105/64
            u4_1 = -4*φ1(2) - (26/3)*φ1(3) - 9*φ1(4) - 4*φ1(5) + (105/32)*φ1(0,2)
            
            v1 = -4*φ1(2) - (26/3)*φ1(3) - 9*φ1(4) - 4*φ1(5) + (35/16)*φ1(0,2) + 5/3
            
            a2_1 -= u2_1
            a3_1 -= u3_1
            a4_1 -= u4_1
            b1 -= v1
            
            k1 = denoised_prev1 - x_0

        

        if h_prev2 is not None:
            φ2 = Phi(h_prev2, ci)
            
            u2_2 = 3*φ2(2,2) + (19/2)*φ2(3,2) + 12*φ2(4,2) + 6*φ2(5,2)
            u3_2 = u2_2 - 189/128
            u4_2 = 3*φ2(2) + (19/2)*φ2(3) + 12*φ2(4) + 6*φ2(5) - (189/64)*φ2(0,2)

            v2 = 3*φ2(2) + (19/2)*φ2(3) + 12*φ2(4) + 6*φ2(5) - (63/32)*φ2(0,2) - 5/3

            a2_2 -= u2_2
            a3_2 -= u3_2
            a4_2 -= u4_2
            b2 -= v2
            
            k2 = denoised_prev2 - x_0
    
        

        if h_prev2 is not None:
            φ2 = Phi(h_prev2, ci)
            
            u2_2 = 3*φ2(2,2) + (19/2)*φ2(3,2) + 12*φ2(4,2) + 6*φ2(5,2)
            u3_2 = u2_2 - 189/128
            u4_2 = 3*φ2(2) + (19/2)*φ2(3) + 12*φ2(4) + 6*φ2(5) - (189/64)*φ2(0,2)

            v2 = 3*φ2(2) + (19/2)*φ2(3) + 12*φ2(4) + 6*φ2(5) - (63/32)*φ2(0,2) - 5/3

            a2_2 -= u2_2
            a3_2 -= u3_2
            a4_2 -= u4_2
            b2 -= v2
            
            k2 = denoised_prev2 - x_0

        
        """denoised2 = model(x2, sigma2 * s_in, **extra_args)
        k2 = denoised2 - x_0
        x3 = x_0 + h * (a3_1*k1 + a3_2*k2)
        
        denoised3 = model(x3, sigma3 * s_in, **extra_args)
        k3 = denoised3 - x_0
        x4 = x_0 + h * (a4_1*k1 + a4_2*k2 + a4_3*k3)"""
        
        if h_prev1 is None:
            denoised1 = model(x1, sigma1 * s_in, **extra_args)
            if sigma_next == 0:
                return denoised1
            k1 = denoised1 - x_0            
        x2 = x_0 + h * (a2_1*k1)
        
        if h_prev2 is None:
            denoised2 = model(x2, sigma2 * s_in, **extra_args)
            k2 = denoised2 - x_0
        x3 = x_0 + h * (a3_1*k1 + a3_2*k2)
        
        if h_prev3 is None:
            denoised3 = model(x3, sigma3 * s_in, **extra_args)
            k3 = denoised3 - x_0
        x4 = x_0 + h * (a4_1*k1 + a4_2*k2 + a4_3*k3)  
        
        denoised4 = model(x4, sigma4 * s_in, **extra_args)
        k4 = denoised4 - x_0
        x_next = x_0 + h * (b1*k1 + b2*k2 + b3*k3 + b4*k4)
        
        
        denoised = x_0 + ((sigma / (sigma - sigma_down)) *  h) * (b1*k1 + b2*k2 + b3*k3 + b4*k4)
        eps = denoised - x_0
        
        x = x_next
        
        noise = noise_sampler(sigma=sigma, sigma_next=sigma_next)
        noise = (noise - noise.mean()) / noise.std()
        
        x = alpha_ratio * x + noise * sigma_up
        
        denoised = denoised4

        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised})

    return denoised





def sample_rk_res_denoise_eps(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
    
    h_fn = lambda sigma_next, sigma: -torch.log()
        
    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
        x_0 = x.clone()
           
        #h = -torch.log(sigma_next/sigma)
        #e_h = torch.exp(sigma - sigma_next)
        #h = -torch.log(e_h)
        
        h = sigma_next - sigma
        
        c2 =  0.5

        a2_1 = c2 * phi(1, -h*c2)
        b1 =        phi(1, -h) - phi(2, -h)/c2
        b2 =        phi(2, -h)/c2

        denoised = model(x, sigma * s_in, **extra_args)
        #if sigma_next == 0:
        #    return denoised
        
        eps = (x - denoised) / sigma
        
        x_2 = x + h * (a2_1 * eps)
        
        sigma_2 = (sigma_next - sigma) * c2 + sigma
        
        denoised_2 = model(x_2, sigma_2 * s_in, **extra_args)
        eps_2 = (x_2 - denoised_2) / sigma_2
        #eps_2 = (x - denoised_2) / sigma


        x = x + h * (b1 * eps + b2 * eps_2)
        
        denoised = x_0 + ((sigma / (sigma - sigma_next)) *  h) * (b1 * eps + b2 * eps_2)


        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised})

    return denoised


def sample_rk_euler_lowranksvd(
    model, x, sigmas, extra_args=None, callback=None, disable=None,
    noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard",
    rk_type="dormand-prince", sigma_fn_formula="", t_fn_formula="",
    eta=0.5, eta_var=0.0, s_noise=1.0, alpha=-1.0, k=1, scale=0.1,
    c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0,
    reverse_weight=0.0, extra_options="", cfg1=0, cfg2=0,
    cfg_cw=1.0, latent_guide=None, jacobian_scale=0.01, epsilon=1e-4
):
    """
    Perform Euler updates with low-rank Jacobian approximation using SVD on multi-dimensional tensors.
    
    Args:
        model: Function that computes the drift term. Signature: model(x, sigma * s_in, **extra_args).
        x: Current state tensor with shape (1, channels, height, width).
        sigmas: Tensor of sigma values (step sizes).
        extra_args: Additional arguments for the model.
        callback: Optional callback function for monitoring.
        disable: Disable tqdm progress bar if True.
        noise_sampler: Noise sampler instance.
        noise_sampler_type: Type of noise sampler to use.
        noise_mode: Mode for noise sampling.
        rk_type: Runge-Kutta type (unused in this context).
        sigma_fn_formula: Placeholder for sigma function formula.
        t_fn_formula: Placeholder for time function formula.
        eta: Noise scaling factor.
        eta_var: Noise variance scaling factor.
        s_noise: Noise strength.
        alpha: Placeholder parameter.
        k: Rank for low-rank Jacobian approximation.
        scale: Scaling factor (unused in this context).
        c2: Placeholder parameter.
        c3: Placeholder parameter.
        buffer: Placeholder parameter.
        cfgpp: Placeholder parameter.
        iter: Placeholder parameter.
        sub_iter: Placeholder parameter.
        reverse_weight: Placeholder parameter.
        extra_options: Additional options (unused in this context).
        cfg1: Placeholder parameter.
        cfg2: Placeholder parameter.
        cfg_cw: Placeholder parameter.
        latent_guide: Placeholder parameter.
        jacobian_scale: Scaling factor for Jacobian correction to prevent over-denoising.
        epsilon: Small perturbation value for finite differences.
    
    Returns:
        Denoised tensor after Euler updates with the same shape as input `x`.
    """
    # Initialize extra_args if not provided
    extra_args = {} if extra_args is None else extra_args

    # Extract tensor dimensions
    batch_size, channels, height, width = x.shape
    assert batch_size == 1, "This function assumes batch_size = 1."
    x = x.to(torch.float32)
    sigmas = sigmas.to(torch.float32)
    device = x.device  # Ensure all operations are on the same device

    # Initialize s_in with ones, matching the shape of x
    # Shape: [1, channels, 1, 1] for broadcasting
    s_in = x.new_ones([x.shape[0]])


    # Initialize noise_sampler if not provided
    if noise_sampler is None:
        noise_sampler = torch.randn_like  # Default to Gaussian noise

    # Ensure k is an integer and does not exceed the total spatial dimension
    k = int(k)
    spatial_dim = height * width
    total_dim = channels * spatial_dim
    if k > total_dim:
        print(f"Rank k={k} exceeds total dimension {total_dim}. Reducing k to {total_dim}.")
        k = total_dim

    # Iterate over each sigma step
    for step in trange(len(sigmas)-1, disable=disable, desc="Sampling"):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
        sigma_up, sigma, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, eta, noise_mode)
        h = sigma_next - sigma

        # Step 1: Compute denoised output from the model
        # The model expects input with shape (1, channels, height, width)
        denoised = model(x, sigma * s_in, **extra_args)  # Shape: [1, channels, height, width]

        # Step 2: Compute epsilon (drift term)
        #eps = denoised - x  # Shape: [1, channels, height, width]
        eps = (x - denoised) / sigma

        if sigma_next == 0:
            return denoised

        # Step 3: Low-Rank Jacobian Approximation using SVD on the flattened matrix
        # Flatten the tensor to [channels, height * width]
        A = x.view(channels, spatial_dim)  # Shape: [channels, height*width]
        A_den = denoised.view(channels, spatial_dim)  # Shape: [channels, height*width]
        eps_flat = eps.view(channels, spatial_dim)  # Shape: [channels, height*width]

        # Step 3a: Generate random projection matrix Omega (k, channels * height * width)
        Omega = torch.randn(k, channels * spatial_dim, device=device)  # Shape: [k, channels*H*W]
        noise = torch.randn_like(x)

        # Step 3b: Compute Y = J @ Omega using finite differences
        Y = torch.zeros(k, channels * spatial_dim, device=device)  # Shape: [k, channels*H*W]

        for i in range(k):
            # Reshape Omega[i] to [1, channels, height, width]
            perturb = epsilon * Omega[i].view(1, channels, height, width)  # Shape: [1, C, H, W]
            x_perturbed = x + h * perturb # (eps + noise)              #perturb   # Shape: [1, C, H, W]

            # Compute denoised perturbed output
            denoised_perturbed = model(x_perturbed, sigma * s_in, **extra_args)  # Shape: [1, C, H, W]

            # Flatten perturbed denoised output
            A_den_perturbed = denoised_perturbed.view(channels, spatial_dim)  # Shape: [C, H*W]

            # Approximate J @ Omega[:, i]
            Jv = (A_den_perturbed - A_den) / epsilon  # Shape: [C, H*W]

            # Store in Y
            Y[i] = Jv.view(-1)  # Shape: [C*H*W]

        # Step 3c: QR Decomposition of Y^T to get Q (channels*H*W, k)
        Q, _ = torch.linalg.qr(Y.T)  # Shape: [channels*H*W, k]

        # Step 3d: Compute B = Q^T @ J using finite differences
        B = torch.zeros(k, channels * spatial_dim, device=device)  # Shape: [k, channels*H*W]

        for i in range(k):
            # Reshape Q[:, i] to [1, channels, height, width]
            q = Q[:, i].view(1, channels, height, width)  # Shape: [1, C, H, W]
            perturb = epsilon * q  # Shape: [1, C, H, W]
            x_perturbed = x + h * perturb #(eps + noise) # + 0.001* perturb  # Shape: [1, C, H, W]

            # Compute denoised perturbed output
            denoised_perturbed = model(x_perturbed, sigma * s_in, **extra_args)  # Shape: [1, C, H, W]

            # Flatten perturbed denoised output
            A_den_perturbed = denoised_perturbed.view(channels, spatial_dim)  # Shape: [C, H*W]

            # Approximate J^T @ Q[:, i]
            Jtq = (A_den_perturbed - A_den) / epsilon  # Shape: [C, H*W]

            # Store in B
            B[i] = Jtq.view(-1)  # Shape: [C*H*W]

        # Step 3e: Perform SVD on B
        U_B, S, Vh_B = torch.linalg.svd(B, full_matrices=False)  # U_B: [k, k], S: [k], Vh_B: [k, channels*H*W]

        # Step 3f: Reconstruct low-rank approximation of Jacobian
        # Compute U_k = Q @ U_B [channels*H*W, k] @ [k, k] = [channels*H*W, k]
        U_k = Q @ U_B  # Shape: [channels*H*W, k]

        # Compute V_k = Vh_B.T [channels*H*W, k]
        V_k = Vh_B.T  # Shape: [channels*H*W, k]

        # Step 4: Compute J @ eps using low-rank approximation
        # Flatten eps
        eps_vector = eps_flat.view(-1)  # Shape: [channels*H*W]

        # Compute V_k^T @ eps_vector [k, channels*H*W] @ [channels*H*W] = [k]
        V_k_T_eps = V_k.t() @ eps_vector  # Shape: [k]

        # Multiply by singular values S [k]
        S_V_k_T_eps = S * V_k_T_eps  # Shape: [k]

        # Multiply by U_B [k, k]
        U_B_S_V_k_T_eps = U_B @ S_V_k_T_eps  # Shape: [k]

        # Multiply by U_k [channels*H*W, k] @ [k] = [channels*H*W]
        J_eps_approx = U_k @ U_B_S_V_k_T_eps  # Shape: [channels*H*W]

        # Reshape to [1, channels, height, width]
        J_eps_approx = J_eps_approx.view(1, channels, height, width)  # Shape: [1, C, H, W]

        # Step 5: Perform Euler update with Jacobian correction
        # Scale the Jacobian correction to prevent over-denoising
        x = x + h * (eps + jacobian_scale * J_eps_approx)  # Shape: [1, C, H, W]

        # Step 6: Add noise based on noise_mode
        if noise_sampler is not None:
            if noise_mode == "hard":
                noise = noise_sampler(x) * s_noise * eta  # Shape: [1, C, H, W]
                x = x + noise
            elif noise_mode == "soft":
                noise = noise_sampler(x) * (s_noise * eta) ** 2  # Shape: [1, C, H, W]
                x = x + noise
            else:
                # No noise for undefined modes
                pass

        # Step 7: Clamp to prevent numerical issues
        #x = torch.clamp(x, min=-1e5, max=1e5)  # Shape: [1, C, H, W]

        # Callback after step (if provided)
        if callback is not None:
            denoised_reshaped = denoised.clone().detach()
            callback({
                'x': x.clone().detach(),
                'i': step,
                'sigma': sigma.item(),
                'sigma_next': sigma_next.item(),
                'denoised': denoised_reshaped
            })

    return denoised


def sample_rk_euler_lowranksvd_burns(
    model, x, sigmas, extra_args=None, callback=None, disable=None,
    noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard",
    rk_type="dormand-prince", sigma_fn_formula="", t_fn_formula="",
    eta=0.5, eta_var=0.0, s_noise=1.0, alpha=-1.0, k=1, scale=0.1,
    c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0,
    reverse_weight=0.0, extra_options="", cfg1=0, cfg2=0,
    cfg_cw=1.0, latent_guide=None
):
    """
    Perform Euler updates with low-rank Jacobian approximation using SVD on multi-dimensional tensors.
    
    Args:
        model: Function that computes the drift term. Signature: model(x, sigma * s_in, **extra_args).
        x: Current state tensor with shape (1, channels, height, width).
        sigmas: Tensor of sigma values (step sizes).
        extra_args: Additional arguments for the model.
        callback: Optional callback function for monitoring.
        disable: Disable tqdm progress bar if True.
        noise_sampler: Noise sampler instance.
        noise_sampler_type: Type of noise sampler to use.
        noise_mode: Mode for noise sampling.
        rk_type: Runge-Kutta type (unused in this context).
        sigma_fn_formula: Placeholder for sigma function formula.
        t_fn_formula: Placeholder for time function formula.
        eta: Noise scaling factor.
        eta_var: Noise variance scaling factor.
        s_noise: Noise strength.
        alpha: Placeholder parameter.
        k: Rank for low-rank Jacobian approximation.
        scale: Scaling factor (unused in this context).
        c2: Placeholder parameter.
        c3: Placeholder parameter.
        buffer: Placeholder parameter.
        cfgpp: Placeholder parameter.
        iter: Placeholder parameter.
        sub_iter: Placeholder parameter.
        reverse_weight: Placeholder parameter.
        extra_options: Additional options (unused in this context).
        cfg1: Placeholder parameter.
        cfg2: Placeholder parameter.
        cfg_cw: Placeholder parameter.
        latent_guide: Placeholder parameter.
    
    Returns:
        Denoised tensor after Euler updates with the same shape as input `x`.
    """
    # Initialize extra_args if not provided
    extra_args = {} if extra_args is None else extra_args

    # Extract tensor dimensions
    batch_size, channels, height, width = x.shape
    assert batch_size == 1, "This function assumes batch_size = 1."
    device = x.device  # Ensure all operations are on the same device

    # Initialize s_in with ones, matching the shape of x
    # Shape: [1, channels, height, width] for broadcasting
    #s_in = torch.ones(1, channels, 1, 1, device=device)
    s_in = x.new_ones([x.shape[0]])

    # Initialize noise_sampler if not provided
    if noise_sampler is None:
        noise_sampler = torch.randn_like  # Default to Gaussian noise

    # Ensure k is an integer and does not exceed the channels * spatial_dim
    k = int(k)
    spatial_dim = height * width
    total_dim = channels * spatial_dim
    if k > total_dim:
        print(f"Rank k={k} exceeds total dimension {total_dim}. Reducing k to {total_dim}.")
        k = total_dim

    epsilon = 1e-4  # Perturbation step size

    # Iterate over each sigma step
    for step in trange(len(sigmas)-1, disable=disable, desc="Sampling"):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
        h = sigma_next - sigma

        # Step 1: Compute denoised output from the model
        denoised = model(x, sigma * s_in, **extra_args)  # Shape: [1, channels, height, width]

        # Step 2: Compute epsilon (drift term)
        #eps = denoised - x  # Shape: [1, channels, height, width]
        eps = (x - denoised) / sigma

        if sigma_next == 0:
            return denoised

        # Step 3: Low-Rank Jacobian Approximation using SVD on the flattened matrix
        # Flatten the tensor to [channels, height * width]
        A = x.view(channels, spatial_dim)  # Shape: [channels, height*width]

        # Perform SVD on A
        U, S, Vh = torch.linalg.svd(A, full_matrices=False)  # U: [channels, k], S: [k], Vh: [k, height*width]

        # Reconstruct the low-rank approximation
        # Here, k is the rank, so we keep top-k singular values
        # Ensure that k does not exceed the available singular values
        k_eff = min(k, U.shape[1])
        U_k = U[:, :k_eff]  # [channels, k_eff]
        S_k = S[:k_eff]     # [k_eff]
        Vh_k = Vh[:k_eff, :]  # [k_eff, height*width]

        # Reconstruct A using low-rank approximation
        A_reconstructed = U_k @ torch.diag(S_k) @ Vh_k  # Shape: [channels, height*width]

        # Step 4: Compute Jacobian-vector product approximation
        # Assuming J ≈ U_k @ Vh_k, then J @ eps_flat ≈ U_k @ (Vh_k @ eps_flat)
        eps_flat = eps.view(channels, spatial_dim)  # Shape: [channels, height*width]

        # Compute Vh_k @ eps_flat.T -> [k_eff, 1, height*width] x [channels, height*width] -> Need to align dimensions
        # To compute (Vh_k @ eps_flat), we need to treat eps_flat as a vector. Given the dimensions, it's ambiguous.

        # Since SVD was performed on A = [channels, height*width], J is effectively the identity if A is orthogonal.
        # The low-rank approximation might not directly correspond to the Jacobian without specific context.
        # Therefore, this simplified approach assumes that the Jacobian can be approximated via SVD on the flattened matrix.

        # For demonstration, we'll proceed with reconstructing the denoised output using the low-rank approximation
        denoised_reconstructed = A_reconstructed.view(1, channels, height, width)  # Shape: [1, channels, height, width]

        # Step 5: Perform Euler update with Jacobian correction
        # Here, we can interpret the difference between denoised_reconstructed and denoised as the Jacobian correction
        J_eps_approx = denoised_reconstructed - denoised  # Shape: [1, channels, height, width]

        # Update the current state with Euler method
        x = x + h * (eps + J_eps_approx)  # Shape: [1, channels, height, width]

        # Step 6: Add noise based on noise_mode
        if noise_sampler is not None:
            if noise_mode == "hard":
                noise = noise_sampler(x) * s_noise * eta  # Shape: [1, channels, height, width]
                x = x + noise
            elif noise_mode == "soft":
                noise = noise_sampler(x) * (s_noise * eta) ** 2  # Shape: [1, channels, height, width]
                x = x + noise
            else:
                # No noise for undefined modes
                pass

        # Step 7: Clamp to prevent numerical issues

        # Callback after step (if provided)
        if callback is not None:
            callback({
                'x': x.clone().detach(),
                'i': step,
                'sigma': sigma.item(),
                'sigma_next': sigma_next.item(),
                'denoised': denoised.clone().detach()
            })

    return denoised


def sample_rk_euler_lowranksvd_channelfuckery(
    model, x, sigmas, extra_args=None, callback=None, disable=None,
    noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard",
    rk_type="dormand-prince", sigma_fn_formula="", t_fn_formula="",
    eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1, scale=0.1,
    c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0,
    reverse_weight=0.0, extra_options="", cfg1=0, cfg2=0,
    cfg_cw=1.0, latent_guide=None
):
    """
    Perform Euler updates with low-rank Jacobian approximation using SVD on multi-dimensional tensors.

    Args:
        model: Function that computes the drift term. Signature: model(x, sigma * s_in, **extra_args).
        x: Current state tensor with shape (batch_size, channels, height, width).
        sigmas: Tensor of sigma values (step sizes).
        extra_args: Additional arguments for the model.
        callback: Optional callback function for monitoring.
        disable: Disable tqdm progress bar if True.
        noise_sampler: Noise sampler instance.
        noise_sampler_type: Type of noise sampler to use.
        noise_mode: Mode for noise sampling.
        rk_type: Runge-Kutta type (unused in this context).
        sigma_fn_formula: Placeholder for sigma function formula.
        t_fn_formula: Placeholder for time function formula.
        eta: Noise scaling factor.
        eta_var: Noise variance scaling factor.
        s_noise: Noise strength.
        alpha: Placeholder parameter.
        k: Rank for low-rank Jacobian approximation.
        scale: Scaling factor (unused in this context).
        c2: Placeholder parameter.
        c3: Placeholder parameter.
        buffer: Placeholder parameter.
        cfgpp: Placeholder parameter.
        iter: Placeholder parameter.
        sub_iter: Placeholder parameter.
        reverse_weight: Placeholder parameter.
        extra_options: Additional options (unused in this context).
        cfg1: Placeholder parameter.
        cfg2: Placeholder parameter.
        cfg_cw: Placeholder parameter.
        latent_guide: Placeholder parameter.

    Returns:
        Denoised tensor after Euler updates with the same shape as input `x`.
    """
    extra_args = {} if extra_args is None else extra_args
    batch_size, channels, height, width = x.shape
    device = x.device 
    x_flat = x.view(batch_size, channels, -1) 
    s_in = x.new_ones([x.shape[0]])

    if noise_sampler is None:
        noise_sampler = torch.randn_like

    # Ensure k is an integer and does not exceed the spatial dimension
    k = int(k)
    spatial_dim = height * width
    if k > spatial_dim:
        print(f"Rank k={k} exceeds spatial dimension {spatial_dim}. Reducing k to {spatial_dim}.")
        k = spatial_dim

    epsilon = 1e-4  # Perturbation for finite differences

    for step in trange(len(sigmas)-1, disable=disable, desc="Sampling"):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
        h = sigma_next - sigma

        denoised = model(x, sigma * s_in, **extra_args) 
        eps = (x - denoised) / sigma 

        if sigma_next == 0:
            return denoised

        # Step 2: Compute epsilon (drift term)
        denoised_flat = denoised.view(batch_size, channels, -1)  
        eps_flat = denoised_flat - x_flat  # Shape: (batch_size, channels, height*width)

        # Step 3: Low-Rank Jacobian Approximation using SVD per channel
        for b in range(batch_size):
            for c in range(channels):
                x_bc = x_flat[b, c]  # Shape: (height*width,)
                denoised_bc = denoised_flat[b, c]  # Shape: (height*width,)
                eps_bc = eps[b, c]  # Shape: (height*width,)

                # Step 3a: Generate random projection matrix Omega (height*width, k)
                Omega = torch.randn(spatial_dim, k, device=device)  # Shape: (height*width, k)

                # Step 3b: Compute Y = J @ Omega using finite differences
                Y = []
                for i in range(k):
                    v = Omega[:, i]  # Shape: (height*width,)
                    perturbed_x = x_bc + epsilon * v  # Shape: (height*width,)
                    perturbed_x_reshaped = perturbed_x.view(1, channels, height, width)  # Reshape for the model
                    # Ensure the perturbed input has the same batch and channel dimensions
                    perturbed_den = model(perturbed_x_reshaped, sigma * s_in[b, c], **extra_args).view(-1)  # Shape: (height*width,)
                    Jv = (perturbed_den - denoised_bc) / epsilon  # Shape: (height*width,)
                    Y.append(Jv)
                Y = torch.stack(Y, dim=1)  # Shape: (height*width, k)

                # Step 3c: QR Decomposition of Y
                Q, _ = torch.linalg.qr(Y)  # Shape: (height*width, k)

                # Step 3d: Compute B = Q^T @ J using finite differences
                B = []
                for i in range(k):
                    q = Q[:, i]  # Shape: (height*width,)
                    perturbed_x = x_bc + epsilon * q  # Shape: (height*width,)
                    perturbed_x_reshaped = perturbed_x.view(1, channels, height, width)  # Reshape for the model
                    perturbed_den = model(perturbed_x_reshaped, sigma * s_in[b, c], **extra_args).view(-1)  # Shape: (height*width,)
                    Jtq = (perturbed_den - denoised_bc) / epsilon  # Shape: (height*width,)
                    B.append(Jtq)
                B = torch.stack(B, dim=0)  # Shape: (k, height*width)

                # Step 3e: Perform SVD on B
                U_B, S, Vh_B = torch.linalg.svd(B, full_matrices=False)  # U_B: (k, k), S: (k,), Vh_B: (k, height*width)

                # Step 3f: Reconstruct low-rank approximation of Jacobian
                U_k = Q @ U_B  # Shape: (height*width, k)
                V_k = Vh_B.T   # Shape: (height*width, k)

                # Step 4: Compute J @ eps using low-rank approximation
                # J @ eps ≈ U_k @ (S * (V_k^T @ eps))
                V_k_T_eps = V_k.t() @ eps_bc  # Shape: (k,)
                S_V_k_T_eps = S * V_k_T_eps  # Shape: (k,)
                J_eps_approx = U_k @ S_V_k_T_eps  # Shape: (height*width,)

                # Step 5: Perform Euler update with Jacobian correction
                x_bc_next = x_bc + h * (eps_bc + J_eps_approx)  # Shape: (height*width,)

                # Step 6: Add noise based on noise_mode
                if noise_sampler is not None:
                    if noise_mode == "hard":
                        noise = noise_sampler(x_bc_next) * s_noise * eta  # Shape: (height*width,)
                    elif noise_mode == "soft":
                        noise = noise_sampler(x_bc_next) * (s_noise * eta) ** 2  # Shape: (height*width,)
                    else:
                        noise = torch.zeros_like(x_bc_next)  # No noise for undefined modes
                    x_bc_next = x_bc_next + noise  # Shape: (height*width,)

                # Step 8: Update current state
                x_flat[b, c] = x_bc_next

        # Optional: Callback after all updates
        if callback is not None:
            denoised_reshaped = denoised.view(batch_size, channels, height, width)
            callback({
                'x': x.view(batch_size, channels, height, width).clone().detach(),
                'i': step,
                'sigma': sigma,
                'sigma_next': sigma_next,
                'denoised': denoised_reshaped.clone().detach()
            })

        # Update current state
        x = x_flat.view(batch_size, channels, height, width)

    return denoised


def sample_rk_euler_lowranksvd_borked(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
                  sigma_fn_formula="", t_fn_formula="",
                      eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                      cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None):
    """
    Perform Euler updates with low-rank Jacobian approximation using SVD on channel-wise tensors.

    Args:
        model: Function that computes the drift term. Signature: model(x, sigma * s_in, **extra_args).
        x: Current state tensor with shape (batch_size, channels, height, width).
        sigmas: Tensor of sigma values (step sizes).
        extra_args: Additional arguments for the model.
        callback: Optional callback function for monitoring.
        disable: Disable tqdm progress bar if True.
        noise_sampler: Noise sampler instance.
        noise_sampler_type: Type of noise sampler to use.
        noise_mode: Mode for noise sampling.
        rk_type: Runge-Kutta type (unused in this context).
        sigma_fn_formula: Placeholder for sigma function formula.
        t_fn_formula: Placeholder for time function formula.
        eta: Noise scaling factor.
        eta_var: Noise variance scaling factor.
        s_noise: Noise strength.
        alpha: Placeholder parameter.
        k: Rank for low-rank Jacobian approximation.
        scale: Scaling factor (unused in this context).
        c2: Placeholder parameter.
        c3: Placeholder parameter.
        buffer: Placeholder parameter.
        cfgpp: Placeholder parameter.
        iter: Placeholder parameter.
        sub_iter: Placeholder parameter.
        reverse_weight: Placeholder parameter.
        extra_options: Additional options (unused in this context).
        cfg1: Placeholder parameter.
        cfg2: Placeholder parameter.
        cfg_cw: Placeholder parameter.
        latent_guide: Placeholder parameter.

    Returns:
        Denoised tensor after Euler updates with the same shape as input `x`.
    """
    extra_args = {} if extra_args is None else extra_args
    batch_size, channels, height, width = x.shape
    x_flat = x.view(batch_size, channels, -1)  # Flatten spatial dimensions
    s_in = x_flat.new_ones([batch_size, channels])

    # Initialize noise_sampler based on noise_sampler_type
    if noise_sampler is None:
        noise_sampler = torch.randn_like  # Default to Gaussian noise

    # Ensure k is an integer and does not exceed output_dim per channel
    k = int(k)
    
    for step in trange(len(sigmas)-1, disable=disable, desc="Sampling"):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
           
        h = sigma_next - sigma
        
        # Step 1: Compute denoised output from the model
        denoised = model(x_flat, sigma * s_in, **extra_args)  # Shape: (batch_size, channels, output_dim)
        if sigma_next == 0:
            denoised = denoised.view(batch_size, channels, height, width)
            return denoised
        
        # Step 2: Compute epsilon (drift term)
        eps = denoised - x_flat  # Shape: (batch_size, channels, output_dim)
        
        # Step 3: Low-Rank Jacobian Approximation using SVD per channel
        for b in range(batch_size):
            for c in range(channels):
                x_bc = x_flat[b, c]  # Shape: (output_dim,)
                denoised_bc = denoised[b, c]  # Shape: (output_dim,)
                eps_bc = eps[b, c]  # Shape: (output_dim,)
                
                # Step 3a: Generate random projection matrix Omega (output_dim, rank)
                Omega = torch.randn(eps_bc.shape[0], k)
                
                # Step 3b: Compute Y = J @ Omega using finite differences
                Y = []
                for i in range(k):
                    v = Omega[:, i]  # Shape: (output_dim,)
                    perturbed_x = x_bc + h * v  # Perturb input
                    f_perturbed = model(x_flat[b, c].clone() + h * v, sigma * s_in[b, c], **extra_args)  # Shape: (output_dim,)
                    Jv = (f_perturbed - denoised_bc) / h  # Approximate J @ v
                    Y.append(Jv)
                Y = torch.stack(Y, dim=1)  # Shape: (output_dim, k)
                
                # Step 3c: QR Decomposition of Y
                Q, _ = torch.linalg.qr(Y)  # Shape: (output_dim, k)
                
                # Step 3d: Compute B = Q^T @ J using finite differences
                B = []
                for i in range(k):
                    q = Q[:, i]  # Shape: (output_dim,)
                    perturbed_x = x_bc + h * q  # Perturb input
                    f_perturbed = model(x_flat[b, c].clone() + h * q, sigma * s_in[b, c], **extra_args)  # Shape: (output_dim,)
                    Jtq = (f_perturbed - denoised_bc) / h  # Approximate J^T @ q
                    B.append(Jtq)
                B = torch.stack(B, dim=0)  # Shape: (k, output_dim)
                
                # Step 3e: Perform SVD on B
                U_B, S, Vh_B = torch.linalg.svd(B, full_matrices=False)  # U_B: (k, k), S: (k,), Vh_B: (k, output_dim)
                
                # Step 3f: Reconstruct low-rank approximation of Jacobian
                U_k = Q @ U_B  # Shape: (output_dim, k)
                V_k = Vh_B.T   # Shape: (output_dim, k)
                
                # Step 4: Compute J @ eps using low-rank approximation
                # J @ eps ≈ U_k @ (S * (V_k^T @ eps))
                V_k_T_eps = V_k.t() @ eps_bc  # Shape: (k,)
                S_V_k_T_eps = S * V_k_T_eps  # Shape: (k,)
                J_eps_approx = U_k @ S_V_k_T_eps  # Shape: (output_dim,)
                
                # Step 5: Perform Euler update with Jacobian correction
                x_bc_next = x_bc + h * (eps_bc + J_eps_approx)  # Shape: (output_dim,)
                
                # Step 6: Add noise based on noise_mode
                if noise_sampler is not None:
                    if noise_mode == "hard":
                        noise = noise_sampler(x_bc_next) * s_noise * eta
                    elif noise_mode == "soft":
                        noise = noise_sampler(x_bc_next) * (s_noise * eta) ** 2
                    else:
                        noise = torch.zeros_like(x_bc_next)  # No noise for undefined modes
                    x_bc_next = x_bc_next + noise
                
                # Step 7: Clamp to prevent numerical issues
                x_bc_next = torch.clamp(x_bc_next, min=-1e5, max=1e5)
                
                # Step 8: Update the tensor
                x_flat[b, c] = x_bc_next
        
        # Step 9: Callback for monitoring
        if callback is not None:
            # Reshape x_flat and denoised to original tensor shape for callback
            x_next_reshaped = x_flat.view(batch_size, channels, height, width)
            denoised_reshaped = denoised.view(batch_size, channels, height, width)
            callback({
                'x': x_next_reshaped.clone().detach(),
                'i': step,
                'sigma': sigma,
                'sigma_next': sigma_next,
                'denoised': denoised_reshaped.clone().detach()
            })
        
        # Step 10: Update current state
        x = x_flat.view(batch_size, channels, height, width)
    
    return denoised.view(batch_size, channels, height, width)




def sample_rk_euler_lowranksvd_(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
        
    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
           
        h = sigma_next - sigma
        
        denoised = model(x, sigma * s_in, **extra_args)
        if sigma_next == 0:
            return denoised
        
        eps = denoised - x

        x_next = x + h * eps
                
        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised})

    return denoised





def sample_rk_euler(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
    
    overstep = float(get_extra_options_kv("overstep", "0.0", extra_options))
        
    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
                
        sigma_up, sigma, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, eta, noise_mode)
        
        h = sigma_down - sigma
        
        denoised = model(x, sigma * s_in, **extra_args)
        if sigma_next == 0:
            x = denoised
            break
        
        eps = (x - denoised) / sigma

        x_next = x + h * eps
        
        x = alpha_ratio * x_next + sigma_up * noise_sampler(sigma=sigma, sigma_next=sigma_next)
                
        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised})

    return x


def sample_rk_euler_overstep(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
    
    overstep = float(get_extra_options_kv("overstep", "0.0", extra_options))
        
    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
                
        #sigma_up, sigma, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, eta, noise_mode)
        
        #h = sigma_down - sigma
        
        h = (1 - overstep) * sigma_next - sigma
        
        denoised = model(x, sigma * s_in, **extra_args)
        if sigma_next == 0:
            x = denoised
            break
        
        eps = (x - denoised) / sigma

        x_next = x + h * eps
        
        #x = alpha_ratio * x_next + sigma_up * noise_sampler(sigma=sigma, sigma_next=sigma_next)
                
        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised})

    return x




def sample_rk_euler_prenoise(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince",
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
        
    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
           
        h = sigma_next - sigma
        
        #sigma_up = h * eta
        sigma_up = eta * sigma_next 
        
        #sigma_hat = sigma * (1 + eta)
        #sigma_up = (sigma_hat ** 2 - sigma ** 2) ** .5
        
        if sigma_next > 0 and step > 0:
            noise = noise_sampler(sigma=sigma, sigma_next=sigma_next)
            """sigma_hat = sigma * (1 + eta)
            sigma_up = (sigma_hat ** 2 - sigma ** 2) ** .5
            alpha_ratio = torch.ones_like(sigma)
            sigma = torch.sqrt(sigma**2 + sigma_up**2)"""
            #alpha_ratio = torch.sqrt((sigma**2 - sigma_up**2)/sigma**2)
            
            #alpha_ratio = (1 - eta * sigma_next)
            
            

            #almost works, but just gets faded... losing the denoised component, probably
            #alpha_ratio = torch.sqrt(1 - eta * sigma**2)
            #sigma_up = sigma * torch.sqrt(1 - alpha_ratio**2)
            
            #x = alpha_ratio * x + sigma_up * noise_sampler(sigma=sigma, sigma_next=sigma_next)
            
            eps_prev = (x - denoised) / sigma
            #eps_prev_new = alpha_ratio * eps_prev.clone() + sigma_up * noise
            
            #x = sigma * eps_prev_new + denoised + (sigma_next - sigma_down) * denoised
            
            # this is correct!
            sigma_up, sigma, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma, eta, noise_mode)
            x = alpha_ratio * (denoised + sigma_down * eps_prev) + sigma_up * noise    

            # underdenoised
            #alpha_ratio = torch.sqrt(1 - eta * sigma**2)
            #sigma_up = sigma * torch.sqrt(1 - alpha_ratio**2)
            #x = alpha_ratio * (denoised + sigma * eps_prev) + sigma_up * noise
            
            print(sigma.item(), h.item(), alpha_ratio.item(), sigma_up.item())
            
            
        if False is True and sigma_next > 0 and step > 0:
            noise = noise_sampler(sigma=sigma, sigma_next=sigma_next)
            
            alpha_ratio = torch.sqrt(1 - eta * sigma**2)
            sigma_up = sigma * torch.sqrt(1 - alpha_ratio**2)
            
            eps_prev = (x - denoised) / sigma
            #x = eps_prev * sigma + denoised
            #eps_prev_new = alpha_ratio * eps_prev + sigma_up * noise
            eps_prev_new = (1-eta) * eps_prev + eta * noise
            x = sigma * eps_prev_new + denoised
        
        denoised = model(x, sigma * s_in, **extra_args)
        if sigma_next == 0:
            x = denoised
            break
        
        eps = (x - denoised) / sigma

        x_next = x + h * eps
        
        x = x_next
                
        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised})

    return x




def sample_rk_euler_hamberder(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
        
    sigma_max = 1.
    ε = 1e-3
    
    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
        
        dt = sigma - sigma_next
        
        denoised = model(x, sigma * s_in, **extra_args)
        
        if sigma_next == 0:
            x = denoised
            break
        
        eps = (x - denoised) / sigma
        sθ = (x - denoised) / sigma
        
        t = (1 - sigma) * (1-ε) + ε
        t = sigma * 0.9999
        sigma_t = eta * (1-t)
        
        #sθ = (-1/t) * x - (1-t)/t * eps
        #drift = (sθ   +   (sigma_t**2) / (2 * ((1-t)**2))   *   (t * (1-t) * sθ - (2-t)*x.clone()) / 2)    *    dt
        
        drift = ((1/t)*x.clone() + (2*(1-t))/t * sθ )    *    -dt
        
        diffusion = torch.sqrt((2*(1-t))/t) * noise_sampler(sigma=sigma, sigma_next=sigma_next)
        
        x = x.clone() + drift + diffusion
                
        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised})

    return x


        #pred_sigma = pred   +   (sigma_t**2) / (2*(sigma_max**2) * ((1.-num_t)**2))   *   (0.5 * num_t * (1.-num_t) * pred - 0.5 * (2.-num_t)*x.clone())
        #pred_sigma = pred   +   (sigma_t**2) / (2*(sigma_max**2) * ((1-t)**2))   *   (t * (1-t) * pred - (2-t)*x.clone()) / 2

        #x = x.clone() + pred_sigma * dt + sigma_t * torch.sqrt(dt) * noise



def srk_f1(delta):
    return torch.exp(2*delta)/2 - 2*torch.exp(delta) + delta + 3/2
    
def srk_f2(delta):
    return torch.exp(2*delta)/2 - 1/2
    
def srk_f3(delta):
    return torch.exp(2*delta)/2 - torch.exp(delta) + 1/2

def srk_zeta1(delta_k):
    numerator = 2 * 2**0.5 * srk_f1(delta_k) ** 0.5
    denominator = torch.exp(delta_k) - torch.exp(-delta_k)
    return numerator/denominator

def srk_zeta2(delta_k):
    numerator = 2**0.5 * srk_f3(delta_k)
    denominator = srk_f1(delta_k) ** 0.5
    return numerator/denominator

def srk_zeta3(delta_k):
    return torch.sqrt(   2 * srk_f2(delta_k) - (2 * srk_f3(delta_k)**2 / srk_f1(delta_k)))



def sample_rk_salmon(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
        
    sigma_max = 1.
    ε = 1e-3
    
    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
        
        delta_k = sigma_next - sigma
        
        noises = [noise_sampler(sigma=sigma, sigma_next=sigma_next) for _ in range(3)]
        
        alpha_k = torch.exp(-2*delta_k)
        
        x = x + srk_zeta1(delta_k) * noises[0]
        
        denoised = model(x, sigma * s_in, **extra_args)
        
        if sigma_next == 0:
            x = denoised
            break
        
        sθ = -(x - denoised) / sigma
        
        x_next = (1/torch.sqrt(alpha_k)) * (x.clone() + (1 - alpha_k) * sθ) + srk_zeta2(delta_k)*noises[1] + srk_zeta3(delta_k)*noises[2]

        x = x_next
        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised})

    return x





def sample_rk_euler_banana(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    
    sigma_max = model.inner_model.inner_model.model_sampling.sigma_max.to(x.dtype)

    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
        
    sigma_max = 1.
    ε = 1e-3
    
    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
        
        dt = sigma - sigma_next
        
        denoised = model(x, sigma * s_in, **extra_args)
        
        if sigma_next == 0:
            x = denoised
            break
        
        sθ = -(x - denoised) / sigma
        
        t = (1 - sigma) * (1-ε) + ε
        sigma_t = eta * (1-t)

        #drift = (sθ   +   (sigma_t**2) / (2 * ((1-t)**2))   *   (t * (1-t) * sθ - (2-t)*x.clone()) / 2)    *    dt
        
        #drift = (sθ   +   (sigma_t**2) / (2 * ((1-t)**2))   *   (t * (1-t) * sθ - (2-t)*x.clone()) / 2)    *    dt   # THIS ONE WAS WORKING
        
        drift = (sθ   +   (sigma_t**2) / (2 * (sigma_max**2) * ((1-t)**2))   *   (t * (1-t) * sθ - (2-t)*x.clone()) / 2)    *    dt
        
        diffusion = sigma_t * torch.sqrt(dt) * noise_sampler(sigma=sigma, sigma_next=sigma_next)
        
        x = x.clone() + drift + diffusion
                
        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised})

    return x


        #pred_sigma = pred   +   (sigma_t**2) / (2*(sigma_max**2) * ((1.-num_t)**2))   *   (0.5 * num_t * (1.-num_t) * pred - 0.5 * (2.-num_t)*x.clone())
        #pred_sigma = pred   +   (sigma_t**2) / (2*(sigma_max**2) * ((1-t)**2))   *   (t * (1-t) * pred - (2-t)*x.clone()) / 2

        #x = x.clone() + pred_sigma * dt + sigma_t * torch.sqrt(dt) * noise









def get_vpsde_step_RF(sigma, sigma_next, eta, sigma_max=1.0):
    dt = sigma - sigma_next
    sigma_up = eta * sigma * dt**0.5
    alpha_ratio = 1 - dt * (eta**2/4) * (1 + sigma)
    sigma_down = sigma_next - (eta/4)*sigma*(1-sigma)*(sigma - sigma_next)
    return sigma_up, sigma_down, alpha_ratio


def sample_rk_euler_banana_alphaupdown_test(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
        
    sigma_max = 1.
    ε = 1e-3
    
    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
                
        denoised = model(x, sigma * s_in, **extra_args)
        
        if sigma_next == 0:
            x = denoised
            break
        
        eps = (x - denoised) / sigma
        
        sigma_up, sigma_down, alpha_ratio = get_vpsde_step_RF(sigma, sigma_next, eta)
        h = sigma_down - sigma

        noise = noise_sampler(sigma=sigma, sigma_next=sigma_next)
        
        x_down = x + h * eps
        x_next = alpha_ratio * x_down + sigma_up * noise
        x = x_next
        
        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised})

    return x


        #pred_sigma = pred   +   (sigma_t**2) / (2*(sigma_max**2) * ((1.-num_t)**2))   *   (0.5 * num_t * (1.-num_t) * pred - 0.5 * (2.-num_t)*x.clone())
        #pred_sigma = pred   +   (sigma_t**2) / (2*(sigma_max**2) * ((1-t)**2))   *   (t * (1-t) * pred - (2-t)*x.clone()) / 2

        #x = x.clone() + pred_sigma * dt + sigma_t * torch.sqrt(dt) * noise




def sample_rk_res_2s_banana(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
        
    sigma_max = 1.
    ε = 1e-3
    
    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
        
        dt = sigma - sigma_next
        
        denoised = model(x, sigma * s_in, **extra_args)
        
        x_0 = x.clone()
        
        if sigma_next == 0:
            x = denoised
            break
        
        c1,c2 = 0, 1/2
        ci = [c1,c2]
        h = -torch.log(sigma_next/sigma)
        φ = Phi(h, ci)
        
        a2_1 = c2 * φ(1,2)
        b2 = φ(2)/c2
        b1 = φ(1) - b2
        
        s2 = -torch.log(sigma) + h * c2
        sigma_2 = torch.exp(-s2)
        
        eps = denoised - x_0
        
        x_2 = x_0 + h * (a2_1 * eps)
        
        denoised_2 = model(x_2, sigma_2 * s_in, **extra_args)
        eps_2 = denoised_2 - x_0
        
        denoised = x_0 +  ((sigma / (sigma - sigma_next)) *  h) * (b1 * eps + b2 * eps_2)
        
        sθ = -(x_0 - denoised) / sigma
        
        t = (1 - sigma) * (1-ε) + ε
        sigma_t = eta * (1-t)

        #drift = (sθ   +   (sigma_t**2) / (2 * ((1-t)**2))   *   (t * (1-t) * sθ - (2-t)*x.clone()) / 2)    *    dt
        
        drift = (sθ   +   (sigma_t**2) / (2 * ((1-t)**2))   *   (t * (1-t) * sθ - (2-t)*x.clone()) / 2)    *    dt
        
        diffusion = sigma_t * torch.sqrt(dt) * noise_sampler(sigma=sigma, sigma_next=sigma_next)
        
        x = x.clone() + drift + diffusion
                
        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised})

    return x




def sample_rk_res_3s_banana(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
        
    sigma_max = 1.
    ε = 1e-3
    
    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
        
        dt = sigma - sigma_next
        
        denoised = model(x, sigma * s_in, **extra_args)
        
        x_0 = x.clone()
        
        if sigma_next == 0:
            x = denoised
            break
        
        c1,c2 = 0, 1/2
        ci = [c1,c2]
        h = -torch.log(sigma_next/sigma)
        φ = Phi(h, ci)
        
        a2_1 = c2 * φ(1,2)
        b2 = φ(2)/c2
        b1 = φ(1) - b2
        
        s2 = -torch.log(sigma) + h * c2
        sigma_2 = torch.exp(-s2)
        
        eps = denoised - x_0
        
        x_2 = x_0 + h * (a2_1 * eps)
        
        denoised_2 = model(x_2, sigma_2 * s_in, **extra_args)
        eps_2 = denoised_2 - x_0
        
        denoised = x_0 +  ((sigma / (sigma - sigma_next)) *  h) * (b1 * eps + b2 * eps_2)
        
        sθ = -(x_0 - denoised) / sigma
        
        t = (1 - sigma) * (1-ε) + ε
        sigma_t = eta * (1-t)

        #drift = (sθ   +   (sigma_t**2) / (2 * ((1-t)**2))   *   (t * (1-t) * sθ - (2-t)*x.clone()) / 2)    *    dt
        
        drift = (sθ   +   (sigma_t**2) / (2 * ((1-t)**2))   *   (t * (1-t) * sθ - (2-t)*x.clone()) / 2)    *    dt
        
        diffusion = sigma_t * torch.sqrt(dt) * noise_sampler(sigma=sigma, sigma_next=sigma_next)
        
        x = x.clone() + drift + diffusion
                
        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised})

    return x






def sample_rk_exp_euler_denoise_eps(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
        
    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
           
        h = -torch.log(sigma_next/sigma)
        
        denoised = model(x, sigma * s_in, **extra_args)
        if sigma_next == 0:
            return denoised
        
        eps = denoised - x

        x_next = x + h * eps
        
        denoised_new = (sigma * x_next   -   sigma_next * x) / (sigma - sigma_next)
        
        x = x_next
        
        print(torch.norm(denoised - denoised_new))
        
        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised})

    return denoised





def sample_rk_euler_alt_sde(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
        
    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
           
        h = sigma_next - sigma
        
        denoised = model(x, sigma * s_in, **extra_args)
        
        eps = (x - denoised) / sigma

        x_next = x + h * eps
        
        noise = noise_sampler(sigma=sigma, sigma_next=sigma_next)
        noise = (noise - noise.mean()) / noise.std()
        
        x = denoised + (sigma_next - eta/sigma_next) * eps + (eta/sigma_next) * noise
        

        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised})

    return denoised



def sample_rk_implicit_cycloeuler(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
    noise_sampler2 = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+10000, sigma_min=0.0, sigma_max=1.0)

        
    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
        
        sigma_up, sigma, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, eta, noise_mode)
           
        h = sigma_down - sigma
        
        denoised = model(x, sigma * s_in, **extra_args)
        if sigma_next == 0:
            x = denoised
            break
        
        eps = (x - denoised) / sigma

        x_down = x + h * eps
        
        if extra_options_flag("add_eps_loopstart", extra_options):
            eps_prev = eps.clone()
        
        for i in range(iter):
            super_alpha_ratio, super_sigma_up, super_sigma_down = get_alpha_ratio_from_sigma_down(sigma_down, sigma, eta)
            noise = noise_sampler2(sigma=sigma, sigma_next=sigma_next)
            
            if extra_options_flag("add_eps", extra_options) and step > 0:
                x = super_alpha_ratio * x_down + super_sigma_up * eps_prev
            else:
                x = super_alpha_ratio * x_down + super_sigma_up * noise
            
            denoised = model(x, sigma * s_in, **extra_args)
            eps = (x - denoised) / sigma
            x_down = x + h * eps        
        
        noise = noise_sampler(sigma=sigma, sigma_next=sigma_next)
        x = alpha_ratio * x_down + sigma_up * noise
        
        eps_prev = eps
                
        if callback is not None: 
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised})

    return x




def sample_rk_implicit_euler(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
    noise_sampler2 = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+10000, sigma_min=0.0, sigma_max=1.0)
    
    reverse_weight = float(get_extra_options_kv("reverse_weight", str(0.0), extra_options))
    
    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
        
        sigma_up, sigma, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, eta, noise_mode)
        
        h = sigma_down - sigma
        h_no_eta = sigma_next - sigma
        
        denoised = model(x, sigma * s_in, **extra_args)
        if sigma_next == 0:
            x = denoised
            break
        eps = (x - denoised) / sigma
        x_next = x + h * eps
        denoised_orig = denoised.clone()
        
        if iter == 0:
            noise = noise_sampler(sigma=sigma, sigma_next=sigma_next)
        else:
            noise = noise_sampler2(sigma=sigma, sigma_next=sigma_next) #use substep noise to avoid changing main noise seed schedule
        x_next = alpha_ratio * x_next + sigma_up * noise
        
        eps_orig = eps.clone()
        if not extra_options_flag("nested_iter", extra_options):
            for i in range(iter):
                
                denoised = model(x_next, sigma_next * s_in, **extra_args)
                eps = (x - denoised) / sigma
                
                if extra_options_flag("stable_eps", extra_options):
                    x_next_new = x + h * (eps_orig + eps) / 2
                else:
                    x_next_new = x + h * eps
                
                x_reverse_new = (x_next - h*denoised) / (sigma_down/sigma)
                x = reverse_weight * x_reverse_new + (1-reverse_weight) * x
                
                x_next = x_next_new
                
                if i == iter-1:
                    noise = noise_sampler(sigma=sigma, sigma_next=sigma_next)
                else:
                    noise = noise_sampler2(sigma=sigma, sigma_next=sigma_next) #use substep noise to avoid changing main noise seed schedule
                x_next = alpha_ratio * x_next + sigma_up * noise


        denoised_prev = denoised_orig.clone()
        if sub_iter + step < len(sigmas)-2 and sub_iter > 0:
            for i in range(1, sub_iter+1):
                sigma2, sigma_next2 = sigmas[step+i], sigmas[step+i+1]
                h2_no_eta = sigma_next2 - sigma
                sigma_up2, sigma2, sigma_down2, alpha_ratio2 = get_res4lyf_step_with_model(model, sigma2, sigma_next2, eta, noise_mode)
                h2 = sigma_down2 - sigma2
                
                denoised = model(x_next, sigma2 * s_in, **extra_args)
                eps = (x - denoised) / sigma
                #eps_orig = (x - denoised_orig) / sigma
                eps2 = (x_next - denoised) / sigma2
                eps2_prev = (x_next - denoised_prev) / sigma2
                
                if extra_options_flag("stable_eps", extra_options):
                    x_next_main = x + h * (eps_orig + eps) / 2
                    x_next      = x_next + h2* (eps2_prev + eps2) / 2
                else:
                    x_next_main = x + h * eps
                    x_next      = x_next + h2* eps2
                denoised_prev = denoised

                
                if i == iter-1:
                    noise = noise_sampler(sigma=sigma, sigma_next=sigma_next)
                else:
                    noise = noise_sampler2(sigma=sigma, sigma_next=sigma_next) #use substep noise to avoid changing main noise seed schedule
                x_next      = alpha_ratio2 * x_next      + sigma_up2 * noise
                
                if extra_options_flag("nested_iter", extra_options):
                    x_next_next = x_next.clone()
                    for i in range(iter):
                
                        denoised = model(x_next_next, sigma_next2 * s_in, **extra_args)
                        eps = (x - denoised) / sigma
                        eps2 = (x_next - denoised) / sigma2
                        eps2_prev = (x_next - denoised_prev) / sigma2
                        
                        if extra_options_flag("stable_eps", extra_options):
                            x_next_main = x + h * (eps_orig + eps) / 2
                            x_next_next      = x_next + h2* (eps2_prev + eps2) / 2
                        else:
                            x_next_main = x + h * eps
                            x_next_next      = x_next + h2* eps2
                        
                        if i == iter-1:
                            noise = noise_sampler(sigma=sigma, sigma_next=sigma_next)
                        else:
                            noise = noise_sampler2(sigma=sigma, sigma_next=sigma_next) #use substep noise to avoid changing main noise seed schedule
                        x_next_next = alpha_ratio * x_next_next + sigma_up * noise
                        
                        denoised_prev = denoised

                
                
                
            x_next_main = alpha_ratio * x_next_main + sigma_up * noise
                
            x_next = x_next_main




        #noise = noise_sampler(sigma=sigma, sigma_next=sigma_next)
        #x = alpha_ratio * x + sigma_up * noise
        
        x = x_next
        
        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised})

    return denoised





def low_rank_fd_approx_for_svd(func, y, num_dirs=1, epsilon=1e-5):
    """
    Compute a low-rank approximation of the Jacobian of 'func' at y using central finite differences,
    without using gradient tracking.
    
    Instead of computing the full Jacobian (which is N x N for y ∈ ℝ^N),
    we sample 'num_dirs' random directions. For each direction vᵢ (sampled from a Rademacher distribution),
    we approximate:
    
         U[:, i] ≈ (func(y + ε·vᵢ) - func(y - ε·vᵢ)) / (2ε)
         and store vᵢ in the i-th column of V.
    
    Returns:
      U: a tensor of shape (N, m)
      V: a tensor of shape (N, m)
    """
    y_flat = y.view(-1)
    N = y_flat.numel()
    m = num_dirs
    U = torch.zeros(N, m, dtype=y.dtype, device=y.device)
    V = torch.zeros(N, m, dtype=y.dtype, device=y.device)
    for i in range(m):
        # Sample a Rademacher vector: entries are ±1.
        v = torch.randint(0, 2, (N,), device=y.device, dtype=y.dtype) * 2 - 1
        V[:, i] = v
        f_plus = func((y_flat + epsilon * v).view_as(y))
        f_minus = func((y_flat - epsilon * v).view_as(y))
        U[:, i] = (f_plus.view(-1) - f_minus.view(-1)) / (2 * epsilon)
    return U, V

# -------------------------------------------
# Implicit Euler Sampler Using SVD-Based Low-Rank FD Newton Correction (No Gradients)
# -------------------------------------------
def sample_rk_implicit_euler_von_svd(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None, r=1, epsilon=1e-5):
    """
    Implicit Euler sampler for diffusion models (ComfyUI version) that uses Newton corrections
    with a low-rank finite-difference SVD approximation of the Jacobian, all without gradient tracking.
    
    The implicit update equation is interpreted as solving for x_next such that:
    
         F(x_next) = x_next - ( x + h * ((x - model(x_next, sigma_next * s_in, **extra_args)) / sigma) ) = 0,
    
    where h = sigma_next - sigma and s_in = ones([x.shape[0]]).
    
    The algorithm is as follows:
    
      1. **Predictor:**  
         Compute:
             denoised = model(x, sigma * s_in, **extra_args)
             eps = (x - denoised) / sigma
             x_predict = x + h * eps
      2. **Residual:**  
         Define:
             F(x_guess) = x_guess - ( x + h * ((x - model(x_guess, sigma_next * s_in, **extra_args)) / sigma) )
      3. **Newton Correction:**  
         For a number of iterations (given by 'iter'):
           - Compute F(x_guess).
           - Compute a low-rank FD approximation of the Jacobian of F at x_guess by calling low_rank_fd_approx.
           - Compute the SVD of the FD Jacobian and truncate it to rank r.
           - Form the approximated derivative of F as:
                 F'(x) ≈ I - h * J_approx,  where J_approx = U_fixed @ diag(S_fixed) @ V_fixed^T.
           - Solve the linear system:
                 (I - h * J_approx) Δx = -F(x_guess)
             via the Woodbury formula:
                 Δx = - (I + h U_fixed (I - h V_fixed^T U_fixed)^{-1} V_fixed^T) F(x_guess)
           - Update:
                 x_guess ← x_guess + Δx.
      4. **Update:**  
         Set x = x_guess and continue with the next sigma step.
    
    All original parameters are preserved.
    
    Args:
      model      : function with signature model(x, sigma, **extra_args)
      x          : initial latent state (e.g. image tensor)
      sigmas     : 1D tensor of sigma values (e.g. noise schedule)
      extra_args : dictionary of extra arguments (default {})
      callback   : callback function (optional)
      disable    : disable progress bar (optional)
      (Other parameters are as provided; notably, iter is the number of Newton iterations per step.)
      r          : target rank for SVD truncation (default: 5)
      epsilon    : finite-difference step size (default: 1e-5)
    
    Returns:
      The final denoised output: model(x, sigma_next * s_in, **extra_args)
    """
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    
    # Loop over the sigma schedule.
    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
        h = sigma_next - sigma  # step size
        
        # Predictor: explicit Euler update using the current sigma.
        denoised = model(x, sigma * s_in, **extra_args)
        if sigma_next == 0:
            x = denoised
            break
        eps = (x - denoised) / sigma
        x_next = x + h * eps  # initial guess for x_next
        
        # Newton iterations (without using autograd; using FD and SVD)
        for i in range(iter):
            # Define the residual function F(x_guess):
            # F(x_guess) = x_guess - ( x + h * ((x - model(x_guess, sigma_next * s_in, **extra_args)) / sigma) )
            def F_func(x_guess):
                denoised_guess = model(x_guess, sigma_next * s_in, **extra_args)
                eps_guess = (x - denoised_guess) / sigma
                return x_guess - (x + h * eps_guess)
            
            F_val = F_func(x_next)
            # Compute the full FD Jacobian of F at x_next, but using a low-rank approximation:
            # Instead of computing the full Jacobian (which would be huge), we call our helper that
            # samples a small number of random directions.
            U, V = low_rank_fd_approx_for_svd(F_func, x_next, num_dirs=r, epsilon=epsilon)
            # Our low-rank approximated Jacobian is: J_approx ≈ U @ V^T.
            # Hence, the derivative of F is approximated as:
            #    F'(x) ≈ I - h * (U @ V^T)
            n = x_next.view(-1).numel()
            I_n = torch.eye(n, dtype=x_next.dtype, device=x_next.device)
            # Instead of forming I - h * (U V^T) explicitly (which is an n×n matrix),
            # we use the Woodbury identity to compute the Newton update:
            # Δx = - (I + h U (I - h V^T U)^{-1} V^T) F(x_next)
            M = torch.eye(V.shape[1], dtype=x_next.dtype, device=x_next.device) - h * (V.t() @ U)
            M_inv = torch.linalg.inv(M)
            F_val_flat = F_val.view(-1, 1)
            correction = - (I_n + h * (U @ M_inv @ V.t())) @ F_val_flat
            delta = correction.view_as(x_next)
            x_next = (x_next + delta).clone().detach()
        
        x = x_next.clone().detach()
        if callback is not None:
            denoised = model(x, sigma_next * s_in, **extra_args)
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised})
    
    denoised = model(x, sigma_next * s_in, **extra_args)
    return denoised






def low_rank_fd_approx(func, y, num_dirs=5, epsilon=1e-5):
    """
    Compute a low-rank approximation of the Jacobian of 'func' at y.
    Instead of computing the full Jacobian (which is n x n for y in R^n),
    we sample 'num_dirs' random directions and approximate:
    
         U[:, i] ≈ (func(y + ε*v_i) - func(y - ε*v_i)) / (2ε),
         where v_i is a random direction (here, using Rademacher variables).
    
    Returns:
      U: a tensor of shape (n, m)
      V: a tensor of shape (n, m) (each column is the random direction used)
    """
    y_flat = y.view(-1)
    n = y_flat.numel()
    m = num_dirs
    U = torch.zeros(n, m, dtype=y.dtype, device=y.device)
    V = torch.zeros(n, m, dtype=y.dtype, device=y.device)
    for i in range(m):
        # Rademacher vector: entries are ±1
        v = torch.randint(0, 2, (n,), device=y.device, dtype=y.dtype) * 2 - 1
        V[:, i] = v
        f_plus = func((y_flat + epsilon*v).view_as(y))
        f_minus = func((y_flat - epsilon*v).view_as(y))
        U[:, i] = (f_plus.view(-1) - f_minus.view(-1)) / (2 * epsilon)
    return U, V

# -----------------------------------------
# Sampler: Implicit Euler with Low-Rank FD Newton Correction
# -----------------------------------------
def sample_rk_implicit_euler_lowrank_fd(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None,
              newton_steps=3, recompute_every=1, num_dirs=5, epsilon=1e-5):
    """
    An implicit Euler sampler for diffusion models that uses a low-rank finite-difference
    Jacobian approximation in the Newton correction.
    
    The implicit update is defined by:
         x_next = x + h * ((x - model(x_next, sigma_next * s_in, **extra_args))/sigma)
    where h is determined from the sigma schedule.
    
    Equivalently, we want to solve for x_next:
         F(x_next) = x_next - ( x + h*((x - model(x_next, sigma_next * s_in, **extra_args))/sigma) ) = 0.
    
    Newton iterations are applied to refine an explicit Euler predictor using a low-rank FD
    approximation of the Jacobian. The low-rank approximation is computed by sampling 'num_dirs'
    random directions.
    
    All original parameters (and the sigma schedule) are preserved.
    """
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    num_dirs = int(get_extra_options_kv("num_dirs", "3", extra_options))
    epsilon = float(get_extra_options_kv("epsilon", "1e-5", extra_options))

    newton_steps = int(get_extra_options_kv("newton_steps", "3", extra_options))
    recompute_every = int(get_extra_options_kv("recompute_every", "1", extra_options))
    
    # Loop over the sigma schedule
    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]

        h = sigma_next - sigma  # step size
        
        # Predictor: Compute explicit Euler step using the current sigma.
        denoised = model(x, sigma * s_in, **extra_args)
        eps = (x - denoised) / sigma
        x_next = x + h * eps  # initial guess
        
        # Define the residual function F(x_next):
        # We wish to solve:
        #    F(x_next) = x_next - ( x + h * ((x - model(x_next, sigma_next * s_in, **extra_args))/sigma) )
        def F_func(x_guess):
            denoised_guess = model(x_guess, sigma_next * s_in, **extra_args)
            eps_guess = (x - denoised_guess) / sigma
            return x_guess - (x + h * eps_guess)
        
        # Newton iterations with low-rank FD Jacobian approximation
        U_fixed, V_fixed = None, None
        for newt in range(newton_steps):
            F_val = F_func(x_next)
            if newt % recompute_every == 0:
                U_fixed, V_fixed = low_rank_fd_approx(F_func, x_next, num_dirs=num_dirs, epsilon=epsilon)
            # Flatten F_val for linear algebra.
            F_val_flat = F_val.view(-1, 1)  # shape (n, 1)
            n = x_next.view(-1).numel()
            I_n = torch.eye(n, dtype=x_next.dtype, device=x_next.device)
            # Compute the m x m matrix: (I - h * V^T U)
            M = torch.eye(V_fixed.shape[1], dtype=x_next.dtype, device=x_next.device) - h * (V_fixed.t() @ U_fixed)
            M_inv = torch.linalg.inv(M)
            # Apply Woodbury formula:
            # (I - h*U*V^T)^{-1} = I + h * U * (I - h*V^T U)^{-1} * V^T
            correction = - (I_n + h * (U_fixed @ M_inv @ V_fixed.t())) @ F_val_flat
            delta = correction.view_as(x_next)
            x_next = (x_next + delta).clone().detach()
        
        # Update x for next step
        x = x_next.clone().detach()
        
        if callback is not None:
            denoised = model(x, sigma_next * s_in, **extra_args)
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised})
    
    # At the end, return the denoised result at the final sigma.
    denoised = model(x, sigma_next * s_in, **extra_args)
    return denoised


def finite_diff_jacobian_nograd(func, y, epsilon=1e-5):
    """
    Approximates the full Jacobian of 'func' at y using central finite differences,
    without using gradient tracking.
    
    Args:
      func    : function mapping y -> output (assumed same shape as y)
      y       : a tensor (assumed to be 1D or flattened) at which to compute the Jacobian.
      epsilon : finite-difference step size.
      
    Returns:
      A tensor of shape (n, n), where n = y.numel(), representing the Jacobian.
    """
    y_flat = y.view(-1)
    n = y_flat.numel()
    J = torch.zeros(n, n, dtype=y.dtype, device=y.device)
    for j in range(n):
        e = torch.zeros_like(y_flat)
        e[j] = epsilon
        f_plus = func((y_flat + e).view_as(y))
        f_minus = func((y_flat - e).view_as(y))
        J[:, j] = ((f_plus - f_minus).view(-1)) / (2 * epsilon)
    return J

def finite_diff_jacobian_vectorized(func, y, epsilon=1e-5):
    """
    Approximates the full Jacobian of 'func' at y using central finite differences,
    in a vectorized manner using torch.vmap (PyTorch 2.0+).
    
    The input y is first flattened, so that the returned Jacobian has shape (n, n)
    where n = y.numel().
    
    Args:
      func    : function mapping a tensor y (with the same shape as the original)
                to a tensor of the same shape.
      y       : a tensor (e.g. 2x2) at which to compute the Jacobian.
      epsilon : finite-difference step size.
      
    Returns:
      A tensor of shape (n, n) representing the Jacobian.
    """
    # Flatten y into a vector of length n.
    y_flat = y.view(-1)
    n = y_flat.numel()
    
    # Create a perturbation matrix E of shape (n, n) where each row is a perturbation vector.
    # Each row of E is epsilon times a row of the identity matrix.
    E = epsilon * torch.eye(n, dtype=y.dtype, device=y.device)  # shape: (n, n)
    
    # Define a helper function that, given a perturbation vector e (shape (n,)),
    # returns func evaluated at y + e, flattened to a vector.
    def eval_func(e):
        return func((y_flat + e).view_as(y)).view(-1)
    
    # Use torch.vmap to apply eval_func to each row of E.
    f_plus = torch.vmap(eval_func)(E)  # shape: (n, n)
    
    # Similarly for the negative perturbation.
    def eval_func_minus(e):
        return func((y_flat - e).view_as(y)).view(-1)
    f_minus = torch.vmap(eval_func_minus)(E)  # shape: (n, n)
    
    # The finite-difference Jacobian is computed column-wise:
    # For each perturbation direction e_j, we approximate the j-th column of J as:
    #     (f(y + e_j) - f(y - e_j)) / (2 * epsilon)
    J = (f_plus - f_minus) / (2 * epsilon)
    return J


def sample_rk_implicit_euler_fd(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
                  sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None, newton_steps=3, recompute_every=1, epsilon=1e-5):
    """
    This is an implicit Euler sampler adapted to use a finite-difference Newton correction
    to solve the implicit update equation.
    
    The original sampler uses:
         denoised = model(x, sigma * s_in, **extra_args)
         eps = (x - denoised) / sigma
         x_next = x + h * eps
         
    and then iterates:
         denoised = model(x_next, sigma_next * s_in, **extra_args)
         eps = (x - denoised) / sigma
         x_next = x + h * eps
         
    Here, we instead interpret the implicit update as finding x_next satisfying:
    
         x_next = x + h * ((x - model(x_next, sigma_next * s_in, **extra_args)) / sigma)
    
    Equivalently, we want to solve for x_next such that the residual
    
         F(x_next) = x_next - (x + h * ((x - model(x_next, sigma_next * s_in, **extra_args)) / sigma)) = 0.
    
    We then apply Newton’s method to refine our initial predictor (the explicit Euler step),
    using a finite-difference estimate of the full Jacobian of F (computed without autograd).
    """
    with torch.inference_mode(False), torch.enable_grad():
        
        x = x.to(torch.float32)
        sigmas = sigmas.to(torch.float32)
        
        x = x.clone().detach().requires_grad_(True)
        sigmas = sigmas.clone().detach().requires_grad_(True)
        
        newton_steps = int(get_extra_options_kv("newton_steps", "3", extra_options))
        recompute_every = int(get_extra_options_kv("recompute_every", "1", extra_options))
        
        extra_args = {} if extra_args is None else extra_args
        s_in = x.new_ones([x.shape[0]])

        # Initialize noise sampler (preserved from your code)
        noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
        
        # Loop over sigma steps.
        for step in trange(len(sigmas)-1, disable=disable):
            sigma, sigma_next = sigmas[step], sigmas[step+1]
            
            # Get additional parameters from the model (as in your code)
            sigma_up, sigma, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, eta, noise_mode)
            h = sigma_down - sigma
            
            # First, compute the "explicit" Euler predictor using the current sigma.
        
            x = x.clone().detach().requires_grad_(True)
            x = x.clone()
            
            
            denoised = model(x, sigma * s_in, **extra_args)
            
            #denoised.backward()
            
            
            eps = (x - denoised) / sigma
            
            
            x_next = x + h * eps  # initial guess
        
        
            # We now wish to refine x_next by solving:
            #     F(x_next) = x_next - ( x + h * ((x - model(x_next, sigma_next * s_in, **extra_args)) / sigma) ) = 0.
            # Note: Here, x (the starting latent) is fixed, and we want to adjust x_next.
            def F_func(x_guess):
                # x_guess has the same shape as x.
                # We evaluate the model at x_guess with sigma_next * s_in.
                denoised_guess = model(x_guess, sigma_next * s_in, **extra_args)
                eps_guess = (x - denoised_guess) / sigma  # note: sigma here is the current sigma, as in original code.
                return x_guess - (x + h * eps_guess)
            
            # Newton iterations using finite differences to approximate the Jacobian of F.
            # Optionally, we can recompute the FD Jacobian only every 'recompute_every' iterations.
            J_fixed = None
            for newt in range(newton_steps):
                F_val = F_func(x_next)
                # Recompute the Jacobian via FD if needed.
                if newt % recompute_every == 0:
                    J_fixed = finite_diff_jacobian_vectorized(F_func, x_next, epsilon=epsilon)
                # Solve for the update: J_fixed * delta = -F_val.
                # (We assume F, and hence x, is treated as a vector; if multidimensional, flattening is assumed.)
                delta = torch.linalg.solve(J_fixed, -F_val.view(-1))
                delta = delta.view_as(x_next)
                # Newton update.
                x_next = (x_next + delta).clone().detach()
            
            # Update x for the next step.
            x = x_next
            
            if callback is not None:
                # For callback, compute the current "denoised" prediction at sigma_next.
                denoised = model(x, sigma_next * s_in, **extra_args)
                callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised})
        
    # At the end, return the final denoised output.
    denoised = model(x, sigma_next * s_in, **extra_args)
    return denoised



"""with torch.inference_mode(False), torch.enable_grad():
    # Ensure requires_grad=True is set
    x2 = torch.randn(x.shape, dtype=torch.float32, device='cuda', requires_grad=True)
    x2 = x2.clone().detach().requires_grad_(True)
    print("Requires grad:", x2.requires_grad)  # Should print True

    # Perform operations
    y = model(x2, sigma * s_in, **extra_args)  # Example operation

    # Backward pass
    y.backward()
    print("Gradient of x2:", x2.grad)  # Should print gradients"""




def default_noise_sampler(x, seed=None):
    if seed is not None:
        generator = torch.Generator(device=x.device)
        generator.manual_seed(seed)
    else:
        generator = None

    return lambda sigma, sigma_next: torch.randn(x.size(), dtype=x.dtype, layout=x.layout, device=x.device, generator=generator)


@torch.no_grad()
def sample_res_multistep(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1., noise_sampler=None):
    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)
    noise_sampler = default_noise_sampler(x, seed=seed) if noise_sampler is None else noise_sampler
    x0_func = lambda x, sigma: model(x, sigma, **extra_args)
    solver_cfg = res.SolverConfig()
    solver_cfg.s_churn = s_churn
    solver_cfg.s_t_max = s_tmax
    solver_cfg.s_t_min = s_tmin
    solver_cfg.s_noise = s_noise
    x = res.differential_equation_solver(x0_func, sigmas, solver_cfg, noise_sampler, callback=callback, disable=disable)(x)
    return x




def sample_rk_implicit_euler_reverse_weight_fail(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
    
    reverse_weight = float(get_extra_options_kv("reverse_weight", str(0.0), extra_options))
    
    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
        
        sigma_up, sigma, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, eta, noise_mode)
        
        h = sigma_down - sigma
        
        denoised = model(x, sigma * s_in, **extra_args)
        eps = (x - denoised) / sigma
        x_next = x + h * eps
         
        
        for i in range(iter):
            denoised = model(x_next, sigma_next * s_in, **extra_args)
            eps = (x - denoised) / sigma
            x_next_new = x + h * eps
            
            x_reverse_new = (x_next - h*denoised) / (sigma_down/sigma)
            x = reverse_weight * x_reverse_new + (1-reverse_weight) * x
            
            x_next = x_next_new

        x = x_next
        
        
        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised})

    return denoised








def sample_rk_momentum_adam(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    
    eps_prev, m_t, v_t, m_t_prev, v_t_prev = [torch.zeros_like(x) for _ in range(5)]

    momentum = float(get_extra_options_kv("momentum", "0.0", extra_options))
    beta1    = float(get_extra_options_kv("beta1", "0.0", extra_options))
    beta2    = float(get_extra_options_kv("beta2", "0.0", extra_options))

    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
    
    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
        
        sigma_up, sigma, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, eta, noise_mode)
        
        h = sigma_down - sigma

        denoised = model(x, sigma * s_in, **extra_args)

        eps = (x - denoised) / sigma
        s_theta = -eps  

        if step == 0:
            m_t_prev = eps.clone()
            v_t_prev = eps**2
            
        m_t = beta1 * m_t_prev + (1-beta1) * eps
        v_t = beta2 * v_t_prev + (1-beta2) * eps**2
        
        v_t = torch.clamp(v_t, min=1e-8)  #this is probably nonsense for sampling. explosive noise growth

        eps = m_t / torch.sqrt(v_t)
            
        x = x + h * eps

        m_t_prev, v_t_prev = m_t, v_t
        
        noise = noise_sampler(sigma=sigma, sigma_next=sigma_next)
        noise = (noise - noise.mean()) / (noise.std())
        
        x = alpha_ratio * x + sigma_up * noise
        
        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised})

    return denoised






def sample_rk_vpsde(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    

    if extra_args is None:
        extra_args = {}

    # If you have some noise_sampler object, otherwise just do:
    if noise_sampler is None:
        def noise_sampler(sigma, sigma_next):
            # Basic Gaussian
            return torch.randn_like(x)
    
    # We'll do an Euler step for each pair (sigma_i, sigma_{i+1})
    for step in trange(len(sigmas)-1, disable=disable):
        sigma_i = sigmas[step]
        sigma_next = sigmas[step+1]

        # 1) Get the model's predicted "denoised" image
        denoised = model(x, sigma_i, **extra_args)

        # 2) Convert that to an epsilon-prediction and the score:
        #    eps = (x - denoised) / sigma_i
        #    score = - eps / sigma_i, but we often just write s_theta = - eps
        eps = (x - denoised) / sigma_i
        s_theta = -eps  # i.e. approximate ∇ log p_t(x)

        # 3) We'll define the discrete "beta_t" as the difference in sigma^2
        #    for a variance-preserving approach:
        beta_t = sigma_i**2 - sigma_next**2
        # This acts like ∫ beta(t) dt from t_i to t_{i+1} in continuous time.

        # 4) The drift has two parts:
        #    (a) -1/2 * beta_t * x    (Ornstein–Uhlenbeck shrink of x)
        #    (b) -    beta_t * s_theta (the "score" drift)
        drift = -0.5 * beta_t * x + (-beta_t) * s_theta

        # 5) The diffusion term is sqrt(beta_t)
        #    We also typically add Gaussian noise here
        noise = noise_sampler(sigma_i, sigma_next)
        noise = (noise - noise.mean()) / (noise.std() + 1e-20)  # optional "normalize"

        # Euler–Maruyama step:
        x = x + drift + torch.sqrt(torch.clamp(beta_t, min=1e-20)) * noise

        # If callback is needed
        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma_i, 'sigma_next': sigma_next, 'denoised': denoised})
            


    # after final step, we expect x to be a (nearly) denoised sample
    return x
    
    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
        
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()
    
    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
        
        #beta_t = sigma_next - sigma
        #beta_t = torch.sqrt(sigma**2 - sigma_next**2)        a2_1 = c2 * phi(1, -h*c2)

        
        

        #beta_t = sigma_next #maybe???
        
        beta_t = torch.sqrt(sigma**2 - sigma_next**2)
        
        #delta_t = sigma_next - sigma
        delta_t = sigma - sigma_next
        
        denoised = model(x, sigma * s_in, **extra_args)
        if sigma_next == 0:
            return denoised
        
        eps = (x - denoised) / sigma
        s_theta = -eps

        #s_theta = -1 * eps / sigma
        
        #s_theta = -(x - alpha_t * denoised) / (sigma**2)
        
        
        #s_theta = -1 * eps / beta_t
        

        noise = noise_sampler(sigma=sigma, sigma_next=sigma_next)
        noise = (noise - noise.mean()) / noise.std()
        
        
        beta_t = torch.sqrt(-2 * sigma * sigma - sigma**2)
        delta_t = -sigma + torch.sqrt(sigma**2 - beta_t**2)
        
        x = x + ((1/2) * beta_t * x   -   beta_t * eps) * delta_t   # +   torch.sqrt(beta_t * delta_t) * noise
        
        
        
        
        #alpha_coeff = 2 - torch.sqrt(1 - beta_t)
        #x_next = alpha_coeff * x   +  (1/2) * beta_t * s_theta   
        #x_next = x_next + torch.sqrt(beta_t) * noise
        
        #x = x_next_noised
        
        #x = x_next #ROCHEURCHOIRCHOEURCHOEURCHOERCUHROCEHURCOEHURCOEHURCOHEURCHOERCUHORECUHRCOEUHRCOHEU
        
        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised})


    return denoised



def sample_rk_vpsde_csbw(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    
    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
        
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()
    
    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
        
        denoised = model(x, sigma * s_in, **extra_args)
        if sigma_next == 0:
            return denoised
        eps = (x - denoised) / sigma
        s_theta = -1 * eps #/ sigma

        noise = noise_sampler(sigma=sigma, sigma_next=sigma_next)
        noise = (noise - noise.mean()) / noise.std()
        
        #alpha_coeff = 1 - sigma_next + torch.sqrt(sigma_next**2 - (sigma_next/2)**2)

        #x_next = alpha_coeff * (x + (sigma-(torch.sqrt(sigma_next**2 - (sigma_next/2)**2))/(1 - sigma_next + torch.sqrt(sigma_next**2 - (sigma_next/2)**2))) * eps) + (sigma_next/2) * noise
        #x = x_next
        
        #x = (sigma_next**2 / sigma**2) * x + (1 - (sigma_next**2 / sigma**2)) * denoised
        
        #x = x + (sigma**2 - sigma_next**2) * s_theta + torch.sqrt(  (sigma_next**2 * (sigma**2 - sigma_next**2))   /   sigma**2  ) * noise
        
        
        #x = x + torch.sqrt(  (sigma_next**2 * (sigma**2 - sigma_next**2))   /   sigma**2  ) * noise
        
        
        #x = x + (sigma**2 - sigma_next**2) * (-eps)   + torch.sqrt(  (sigma_next**2 * (sigma**2 - sigma_next**2))   /   sigma**2  ) * noise
        
        #h = sigma - sigma_next
        
        #alpha_t = h**2 - 2*h + 1 
        
        #x = torch.sqrt(alpha_t) * x   +   h * s_theta   +   torch.sqrt(1 - alpha_t) * noise
        
        
        #x = torch.sqrt(sigma_next**2 / sigma**2) * x   +    (1 - torch.sqrt(sigma_next**2 / sigma**2)) * s_theta  # +   torch.sqrt(sigma**2 - sigma_next**2) * noise 
        
        
        #x =  x   +    (1 - torch.sqrt(sigma_next**2 / sigma**2)) * s_theta
        
        #x = x + h * (-eps)   + torch.sqrt(2 *h ) * noise
        
        h = sigma - sigma_next

        sigma_up = sigma_next * eta 
        sigma_signal = 1 - sigma_next
        sigma_residual = torch.sqrt(sigma_next**2 - sigma_up**2)
        
        sigma_residual = sigma_next * (1 - eta**2)**0.5
        
        eta_root = (1 - eta**2)**0.5
        
        
        
        sigma_residual = sigma_next * eta_root

        alpha_ratio = sigma_signal + sigma_residual
        sigma_down = sigma_residual / alpha_ratio
            
        
        alpha_ratio = (1 - sigma_next)   +   sigma_next * eta_root
        
        
        alpha_ratio = 1  +   sigma_next * eta_root - sigma_next
        
        alpha_ratio = (1  +   sigma_next * (eta_root - 1))


        
        #x = x + (sigma - sigma_down) * s_theta
        
        #x = alpha_ratio * x + sigma_up * noise
        
        
        #x = alpha_ratio * (x + (sigma - sigma_down) * s_theta) + sigma_up * noise
        
        
        #x = x    +    (torch.sqrt(sigma_next**2 - sigma_up**2) - sigma_next) * x    +   (1 + torch.sqrt(sigma_next**2 - sigma_up**2) - sigma_next) * (sigma - sigma_residual / alpha_ratio) * s_theta    +    sigma_up * noise
        
        
        #x = x    +    (sigma_residual - sigma_next) * x    +   (1 + sigma_residual - sigma_next) * (sigma - sigma_residual / alpha_ratio) * s_theta    +    sigma_up * noise
        
        #x = x    +    (sigma_residual - sigma_next) * x    +   alpha_ratio * (sigma - sigma_residual / alpha_ratio) * s_theta    +    sigma_up * noise
        
        #x = x    +    (alpha_ratio - 1) * x    +   alpha_ratio * (sigma - sigma_residual / alpha_ratio) * s_theta    +    sigma_up * noise
        
        
        
        #x = x    +    (alpha_ratio - 1) * x    +   (sigma * alpha_ratio - sigma_residual) * s_theta    +    sigma_up * noise
        
        
        #x = x    +    (alpha_ratio - 1) * x    +   (sigma * alpha_ratio - sigma_residual) * s_theta    +    sigma_up * noise
        
        #x = x    +     sigma_next * (eta_root - 1) * x    +   (sigma *  (1  +   sigma_next * (eta_root - 1)) - sigma_next * eta_root) * s_theta    +    sigma_up * noise
        
        x = x    +     sigma_next * (eta_root - 1) * x    +   (sigma *  (1  +   sigma_next * (eta_root - 1)) - sigma_next * eta_root) * s_theta    +    sigma_next * eta * noise


        

        
        
        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised})


    return denoised





def sample_rk_vpsde_idk(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    
    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
        
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()
    
    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
        
        #beta_t = sigma_next - sigma
        #beta_t = sigma**2 - sigma_next**2
        #beta_t = sigma - sigma_next
        alpha_t = (1 - sigma) / (1 - sigma_next)
        #alpha_t = (1 - sigma**2) / (1 - sigma_next**2)
        beta_t = 1 - alpha_t
        
        denoised = model(x, sigma * s_in, **extra_args)
        if sigma_next == 0:
            return denoised
        
        eps = (x - denoised) / sigma
        s_theta = -1 * eps / sigma
        
        #s_theta = -(x - alpha_t * denoised) / (sigma**2)
        
        #s_theta = -eps
        
        #s_theta = -1 * eps / beta_t
        
        alpha_coeff = 2 - torch.sqrt(1 - beta_t)
        x_next = alpha_coeff * x   +   beta_t * s_theta
        


        noise = noise_sampler(sigma=sigma, sigma_next=sigma_next)
        noise = (noise - noise.mean()) / noise.std()
        x_next_noised = x_next + torch.sqrt(beta_t) * noise
        
        x = x_next_noised
        
        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised})


    return denoised







def sample_rk_vpsde_trivial(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None):  
      
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    
    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(
        x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0
    )
    
    for step in trange(len(sigmas) - 1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step + 1]
        
        # Model predicts noise (epsilon_theta)
        denoised = model(x, sigma * s_in, **extra_args)
        eps = (x - denoised) / sigma
        
        if sigma_next == 0:
            return denoised  # Return final denoised image

        # Compute x_t-1 using VP-SDE equation
        scale_factor = sigma_next / sigma  # Scales the update step
        x_pred = x + (sigma_next**2 - sigma**2) * eps  # Drift term
        
        # Add stochastic noise for VPSDE
        noise = noise_sampler(sigma=sigma, sigma_next=sigma_next)
        noise = (noise - noise.mean()) / noise.std()  # Normalize
        x_next = x_pred + scale_factor * noise  # Stochastic step
        
        x = x_next  # Update state

        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised})
    
    return x  # Return final state



def sample_rk_vpsde_trivial_old(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    
    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
        
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()
    
    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
        

        denoised = model(x, sigma * s_in, **extra_args)
        if sigma_next == 0:
            return denoised

        eps = (x - denoised) / sigma

        noise = noise_sampler(sigma=sigma, sigma_next=sigma_next)
        noise = (noise - noise.mean()) / noise.std()

        #x = denoised + sigma_next * noise
        x = x + (sigma_next - sigma) * eps
        
        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised})


    return denoised






def sample_rk_vpsde_ddpm(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    
    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
        
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()
    
    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
        

        denoised = model(x, sigma * s_in, **extra_args)
        if sigma_next == 0:
            return denoised

        eps = (x - denoised) / sigma

        noise = noise_sampler(sigma=sigma, sigma_next=sigma_next)
        noise = (noise - noise.mean()) / noise.std()

        x_next = torch.sqrt(sigma_next**2 + 1) / torch.sqrt(sigma**2 + 1)   *   (x - (sigma / torch.sqrt(sigma**2 + 1))*eps )  +  sigma_next * noise
        
        x = x_next
        
        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised})


    return denoised





def sample_rk_logit(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options=""):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    
    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
        
    if extra_options_flag("logit", extra_options):
        sigma_fn = lambda t: (t.exp() + 1) ** -1
        t_fn = lambda sigma: ((1-sigma)/sigma).log()
    if extra_options_flag("logsnr", extra_options):
        sigma_fn = lambda t: t.neg().exp()
        t_fn = lambda sigma: sigma.log().neg()
    
    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
        
        if sigma == 1.0:
            sigma = torch.full_like(sigma, 0.9999)
        
        t, t_next = t_fn(torch.clamp(sigma, max=0.9999)), t_fn(sigma_next)
        h = t_next - t
        
        sigma_up, sigma, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, eta, eta_var, noise_mode, h)
        
        t_down = t_fn(sigma_down)
        
        h = t_down - t
        
        sigma_s = sigma_fn(t + h*c2)
        
        h = -torch.log(sigma_down/sigma)
        
        a2_1 = c2 * phi(1, -h*c2)
        b1 =        phi(1, -h) - phi(2, -h)/c2
        b2 =        phi(2, -h)/c2
                
        denoised = model(x, sigma * s_in, **extra_args)
        if sigma_down > 0:
            x_2 = torch.exp(-h * c2) * x + h * (a2_1 * denoised)
            
            denoised_2 = model(x_2, sigma_s * s_in, **extra_args)
            
            x = torch.exp(-h) * x + h * (b1 * denoised + b2 * denoised_2)
            
            if callback is not None:
                callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised})

            noise = noise_sampler(sigma=sigma, sigma_next=sigma_next)
            noise = (noise - noise.mean()) / noise.std()
            x = alpha_ratio * x + noise * s_noise * sigma_up

    return denoised



ZAMPLER_NOISE_MODES = ["hard", "lorentzian", "hard_sq", "soft", "softer", "vpsde", "exp", "none"]

class Zampler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {#"momentum": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step":0.01, "round": False}),
                    "eta":                 ("FLOAT",               {"default": 0.25, "min": -100.0,   "max": 100.0,   "step":0.01, "round": False, "tooltip": "Calculated noise amount to be added, then removed, after each step."}),
                    "eta_var":             ("FLOAT",               {"default": 0.0,  "min": -100.0,   "max": 100.0,   "step":0.01, "round": False, "tooltip": "Calculate variance-corrected noise amount (overrides eta/noise_mode settings). Cannot be used at very low sigma values; reverts to eta/noise_mode for final steps."}),
                    "s_noise":             ("FLOAT",               {"default": 1.0,  "min": -100.0,   "max": 100.0,   "step":0.01, "round": False, "tooltip": "Ratio of calculated noise amount actually added after each step. >1.0 will leave extra noise behind, <1.0 will remove more noise than it adds."}),
                    "alpha":               ("FLOAT",               {"default": 0.0,  "min": -10000.0, "max": 10000.0, "step":0.1,  "round": False, "tooltip": "Fractal noise mode: <0 = extra high frequency noise, >0 = extra low frequency noise, 0 = white noise."}),
                    "k":                   ("FLOAT",               {"default": 1.0,  "min": -10000.0, "max": 10000.0, "step":2.0,  "round": False, "tooltip": "Fractal noise mode: all that matters is positive vs. negative. Effect unclear."}),
                    "cfg1":                ("FLOAT",               {"default": 1.0,  "min": -10000.0, "max": 10000.0, "step":0.01, "round": False, "tooltip": "Sample CFG."}),
                    "cfg2":                ("FLOAT",               {"default": 1.0,  "min": -10000.0, "max": 10000.0, "step":0.01, "round": False, "tooltip": "Unsample CFG."}),
                    "noise_sampler_type":  (NOISE_GENERATOR_NAMES, {"default": "gaussian"}),
                    "noise_mode":          (ZAMPLER_NOISE_MODES,   {"default": 'hard',                                                             "tooltip": "How noise scales with the sigma schedule. Hard is the most aggressive, the others start strong and drop rapidly."}),
                    "iter":                ("INT",                 {"default": 0,    "min": 0,        "max": 100,     "step":1,                    "tooltip": "Number of implicit refinement steps to run after each explicit step. Currently only working with CFG."}),
                    "sub_iter":            ("INT",                 {"default": 0,    "min": 0,        "max": 100,     "step":1,                    "tooltip": "Number of implicit refinement steps to run after each explicit step. Currently only working with CFG."}),
                    "extra_options":       ("STRING",              {"default": "", "multiline": True}),   
                    },
                    "optional": 
                    {
                        "latent_guide":    ("LATENT",),
                        "latent_guide_inv":("LATENT",),
                        "mask":("MASK",),

                        "options":         ("OPTIONS", ),   

                    }  
                }
    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "get_sampler"

    def get_sampler(self, eta=0.25, eta_var=0.0, s_noise=1.0, alpha=-1.0, k=1.0, cfg1=1.0, cfg2=1.0, buffer=0, extra_options="", noise_sampler_type="gaussian", noise_mode="hard",
                    rk_type="dormand-prince", t_fn_formula=None, sigma_fn_formula=None, iter=0, sub_iter=0, latent_guide=None, latent_guide_inv=None, mask=None,
                    ):
        
        #guider = SharkGuider(work_model)
        
        #options_mgr = OptionsManager(options, **kwargs)
        #flow_cond   = options_mgr.get('flow_cond', {})
        
        sampler_name = "zample"
        
        sampler_name = get_extra_options_kv("sampler_name", "zample", extra_options)

        sampler = comfy.samplers.ksampler(sampler_name, {"eta": eta, "eta_var": eta_var, "s_noise": s_noise, "alpha": alpha, "k": k, "cfg1": cfg1, "cfg2": cfg2, "noise_sampler_type": noise_sampler_type, "noise_mode": noise_mode, "rk_type": rk_type, 
                                                        "iter": iter,"sub_iter": sub_iter, "latent_guide": latent_guide, "latent_guide_inv": latent_guide_inv, "mask": mask, "extra_options": extra_options})

        #sampler = comfy.samplers.ksampler(sampler_name, {"eta": eta, "eta_var": eta_var, "s_noise": s_noise, "alpha": alpha, "k": k, "cfg1": cfg1, "cfg2": cfg2, "buffer": buffer, "noise_sampler_type": noise_sampler_type, "noise_mode": noise_mode, "rk_type": rk_type, 
        #                                                "iter": iter,"sub_iter": sub_iter, "latent_guide": latent_guide, "latent_guide_inv": latent_guide_inv, "mask": mask, "extra_options": extra_options})
        return (sampler, )








class Zampler_Test:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {#"momentum": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step":0.01, "round": False}),
                     "eta": ("FLOAT", {"default": 0.25, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Calculated noise amount to be added, then removed, after each step."}),
                     "eta_var": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Calculate variance-corrected noise amount (overrides eta/noise_mode settings). Cannot be used at very low sigma values; reverts to eta/noise_mode for final steps."}),
                     "s_noise": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Ratio of calculated noise amount actually added after each step. >1.0 will leave extra noise behind, <1.0 will remove more noise than it adds."}),
                     "alpha": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step":0.1, "round": False, "tooltip": "Fractal noise mode: <0 = extra high frequency noise, >0 = extra low frequency noise, 0 = white noise."}),
                     "k": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step":2.0, "round": False, "tooltip": "Fractal noise mode: all that matters is positive vs. negative. Effect unclear."}),
                     "cfg1": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step":0.01, "round": False, "tooltip": "Sample CFG."}),
                     "cfg2": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step":0.01, "round": False, "tooltip": "Unsample CFG."}),
                     "noise_sampler_type": (NOISE_GENERATOR_NAMES, {"default": "gaussian"}),
                     #"noise_mode": (NOISE_MODE_NAMES, {"default": 'hard', "tooltip": "How noise scales with the sigma schedule. Hard is the most aggressive, the others start strong and drop rapidly."}),
                     "noise_mode": (["hard", "lorentzian", "hard_sq", "soft", "softer", "vpsde", "exp", "none"], {"default": 'hard', "tooltip": "How noise scales with the sigma schedule. Hard is the most aggressive, the others start strong and drop rapidly."}),
                     "iter": ("INT", {"default": 0, "min": 0, "max": 100, "step":1, "tooltip": "Number of implicit refinement steps to run after each explicit step. Currently only working with CFG."}),
                     "sub_iter": ("INT", {"default": 0, "min": 0, "max": 100, "step":1, "tooltip": "Number of implicit refinement steps to run after each explicit step. Currently only working with CFG."}),
                    "extra_options": ("STRING", {"default": "", "multiline": True}),   
                    },
                    "optional": 
                    {
                        "latent_guide": ("LATENT",),
                    }  
               }
    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "get_sampler"

    def get_sampler(self, eta=0.25, eta_var=0.0, s_noise=1.0, alpha=-1.0, k=1.0, cfg1=1.0, cfg2=1.0, buffer=0, extra_options="", noise_sampler_type="gaussian", noise_mode="hard",
                    rk_type="dormand-prince", t_fn_formula=None, sigma_fn_formula=None, iter=0, sub_iter=0, latent_guide=None,
                    ):
        sampler_name = "zample"
        
        sampler_name = get_extra_options_kv("sampler_name", "zample", extra_options)

        steps = 10000

        sampler = comfy.samplers.ksampler(sampler_name, {"eta": eta, "eta_var": eta_var, "s_noise": s_noise, "alpha": alpha, "k": k, "cfg1": cfg1, "cfg2": cfg2, "buffer": buffer, "noise_sampler_type": noise_sampler_type, "noise_mode": noise_mode, "rk_type": rk_type, 
                                                         "iter": iter,"sub_iter": sub_iter, "latent_guide": latent_guide, "extra_options": extra_options})
        return (sampler, )







def sample_zample_edit(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfg1=1.0, cfg2=1.0, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="",  latent_guide=None,):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
    sigma_min = model.inner_model.inner_model.model_sampling.sigma_min.to(x.dtype)
    sigma_max = model.inner_model.inner_model.model_sampling.sigma_max.to(x.dtype)
    
    y0 = latent_guide = model.inner_model.inner_model.process_latent_in(latent_guide['samples']).clone().to(x.device)
    x  = y0.clone()
    
    noise = noise_sampler(sigma=1.0, sigma_next=sigma_min)
    noise = (noise - noise.mean()) / noise.std()
    x_noise = noise
    
    #sigma_fn = lambda t: t.neg().exp()
    #t_fn = lambda sigma: sigma.log().neg()
    
    sigma_fn = lambda t: t
    t_fn = lambda sigma: sigma

    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
        
        t, t_next = t_fn(sigma), t_fn(sigma_next)
        h = t_next - t
        sigma_up, sigma, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, eta, eta_var, noise_mode, h)
        t_down = t_fn(sigma_down)
        h = t_down - t
        
        #noise = noise_sampler(sigma=sigma, sigma_next=sigma_min)
        #noise = (noise - noise.mean()) / noise.std()
        
        #y = x + eta * (y0 - x)
        #y = y0.clone()
        
        y_adj = sigma * (noise - y0)

        y0_noised = y0 + sigma * (noise - y0)
        eps_y, data_y = epsilon(model, y0_noised, sigma, **extra_args)

        #x_hat     = x  + sigma * (noise - eta * y0 - (1-eta) * x)
        x_hat     = x  + eta * sigma * (noise - y0)
        #x_hat = (x - sigma*y0) + sigma*noise
        eps_x, data_x = epsilon(model, x_hat, sigma, **extra_args)
        
        x_next = x + h * (eps_x - eta * eps_y)

        denoised = data_x
        
        if extra_options_flag("x_hat", extra_options):
            x_next = x + h * eps_x
        elif extra_options_flag("y_noised", extra_options):
            x_next = x + h * eps_y
        if extra_options_flag("denoised_y", extra_options):
            denoised = data_y
        elif extra_options_flag("denoised_x", extra_options):
            denoised = data_x

        x = x_next
        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised})

    return denoised








def sample_zample_edit2(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfg1=1.0, cfg2=1.0, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="",  latent_guide=None,):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
    
    y0 = latent_guide['samples'].to(x.device).to(x.dtype)
    x  = y0.clone()
    #sigma_fn = lambda t: t.neg().exp()
    #t_fn = lambda sigma: sigma.log().neg()
    
    sigma_fn = lambda t: t
    t_fn = lambda sigma: sigma

    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
        
        t, t_next = t_fn(sigma), t_fn(sigma_next)
        h = t_next - t
        #sigma_up, sigma, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, eta, eta_var, noise_mode, h)
        sigma_down = sigma_next
        
        t_down = t_fn(sigma_down)
        h = t_down - t
        
        sigma_min = model.inner_model.inner_model.model_sampling.sigma_min.to(x.dtype)
        
        noise = noise_sampler(sigma=sigma, sigma_next=sigma_min)
        noise = (noise - noise.mean()) / noise.std()
        #x = alpha_ratio * x + noise * s_noise * sigma_up
        
        z_src = (1-sigma)*y0 + sigma*noise
        
        z_tar = x + (z_src - y0)
        
        denoised_tar = model(z_tar, sigma * s_in, **extra_args)
        denoised_src = model(z_src, sigma * s_in, **extra_args)
        
        eps_tar = (z_tar - denoised_tar) / sigma
        eps_src = (z_src - denoised_src) / sigma
        
        x = x + h * (eps_tar - eps_src)
        
        if extra_options_flag("denoised_tar", extra_options):
            denoised = denoised_tar
        else:
            denoised = denoised_src
            
        if sigma_next == 0:
            x = denoised

        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised})

    return x









def sample_zample_inversion(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfg1=1.0, cfg2=1.0, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="",  latent_guide=None,):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    
    uncond = [0]
    uncond[0] = torch.full_like(x, 0.0)
    def post_cfg_function(args):
        uncond[0] = args["uncond_denoised"]
        return args["denoised"]
    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = comfy.model_patcher.set_model_options_post_cfg_function(model_options, post_cfg_function, disable_cfg1_optimization=True)
    
    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
    
    #sigma_fn = lambda t: t.neg().exp()
    #t_fn = lambda sigma: sigma.log().neg()
    
    sigma_fn = lambda t: t
    t_fn = lambda sigma: sigma

    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
        
        t, t_next = t_fn(sigma), t_fn(sigma_next)
        h = t_next - t
        sigma_up, sigma, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, eta, eta_var, noise_mode, h)
        t_down = t_fn(sigma_down)
        h = t_down - t
        
        
        
        denoised = model(x, sigma * s_in, **extra_args)
        denoised = cfg_fn(denoised, uncond[0], cfg1)
        eps = (x - denoised) / sigma
        
        if sigma_down > 0:
            #x_down = torch.sqrt(sigma_down) * (x - torch.sqrt(1-sigma) * eps) / torch.sqrt(sigma)   +   torch.sqrt(1 - sigma_down) * eps
            x_down = x + h * eps
            
            denoised_down = model(x_down, sigma_down * s_in, **extra_args)
            denoised_down = cfg_fn(denoised_down, uncond[0], cfg2)
            eps_down = (x_down - denoised_down) / sigma
            
            #x_next = torch.sqrt(sigma / sigma_down) * x_down   +   torch.sqrt(sigma) * (torch.sqrt(1/sigma - 1) - torch.sqrt(1/sigma_down - 1)) * eps_down
            
            x_next = x_down + (sigma_next - sigma_down) * eps_down

            x = x_next
            
        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised})

    return denoised




def cfg_fn(cond, uncond, scale):
    return cond + scale * (cond - uncond) 


def sample_zample_paper(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfg1=1.0, cfg2=1.0, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="",  latent_guide=None,):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    
    uncond = [0]
    uncond[0] = torch.full_like(x, 0.0)
    def post_cfg_function(args):
        uncond[0] = args["uncond_denoised"]
        return args["denoised"]
    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = comfy.model_patcher.set_model_options_post_cfg_function(model_options, post_cfg_function, disable_cfg1_optimization=True)
    
    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
    
    #sigma_fn = lambda t: t.neg().exp()
    #t_fn = lambda sigma: sigma.log().neg()
    
    sigma_fn = lambda t: t
    t_fn = lambda sigma: sigma

    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
        
        t, t_next = t_fn(sigma), t_fn(sigma_next)
        h = t_next - t
        sigma_up, sigma, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, eta, eta_var, noise_mode, h)
        t_down = t_fn(sigma_down)
        h = t_down - t
        
        
        
        denoised = model(x, sigma * s_in, **extra_args)
        denoised = cfg_fn(denoised, uncond[0], cfg1)
        eps = (x - denoised) / sigma
        
        if sigma_down > 0:
            #x_down = torch.sqrt(sigma_down) * (x - torch.sqrt(1-sigma) * eps) / torch.sqrt(sigma)   +   torch.sqrt(1 - sigma_down) * eps
            x_down = torch.sqrt(sigma) * (x - torch.sqrt(1-sigma_down) * eps) / torch.sqrt(sigma_down)   +   torch.sqrt(1 - sigma) * eps

            #x_down = x + h * eps
            
            denoised_down = model(x_down, sigma_down * s_in, **extra_args)
            denoised_down = cfg_fn(denoised_down, uncond[0], cfg2)
            eps_down = (x_down - denoised_down) / sigma
            
            #x_next = torch.sqrt(sigma / sigma_down) * x_down   +   torch.sqrt(sigma) * (torch.sqrt(1/sigma - 1) - torch.sqrt(1/sigma_down - 1)) * eps_down
            x_next = torch.sqrt(sigma_down / sigma) * x_down   +   torch.sqrt(sigma_down) * (torch.sqrt(1/sigma_down - 1) - torch.sqrt(1/sigma - 1)) * eps_down

            #x_next = x_down + (sigma_next - sigma_down) * eps_down

            x = x_next
            
        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised})

    return denoised




def sample_zsample(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfg1=1.0, cfg2=1.0, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="",  latent_guide=None,):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    
    uncond = [0]
    uncond[0] = torch.full_like(x, 0.0)
    def post_cfg_function(args):
        uncond[0] = args["uncond_denoised"]
        return args["denoised"]
    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = comfy.model_patcher.set_model_options_post_cfg_function(model_options, post_cfg_function, disable_cfg1_optimization=True)
    
    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
    
    #sigma_fn = lambda t: t.neg().exp()
    #t_fn = lambda sigma: sigma.log().neg()
    
    sigma_fn = lambda t: t
    t_fn = lambda sigma: sigma

    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
        
        t, t_next = t_fn(sigma), t_fn(sigma_next)
        h = t_next - t
        sigma_up, sigma, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, eta, eta_var, noise_mode, h)
        t_down = t_fn(sigma_down)
        h = t_down - t
        
        
        
        denoised = model(x, sigma * s_in, **extra_args)
        denoised = cfg_fn(denoised, uncond[0], cfg1)
        eps = (x - denoised) / sigma
        
        if sigma_down > 0:
            #x_down = torch.sqrt(sigma_down) * (x - torch.sqrt(1-sigma) * eps) / torch.sqrt(sigma)   +   torch.sqrt(1 - sigma_down) * eps
            x_down = x + h * eps
            
            denoised_down = model(x_down, sigma_down * s_in, **extra_args)
            denoised_down = cfg_fn(denoised_down, uncond[0], cfg2)
            eps_down = (x_down - denoised_down) / sigma
            
            #x_next = torch.sqrt(sigma / sigma_down) * x_down   +   torch.sqrt(sigma) * (torch.sqrt(1/sigma - 1) - torch.sqrt(1/sigma_down - 1)) * eps_down
            
            x_next = x_down + (sigma_next - sigma_down) * eps_down

            x = x_next
            
        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised})

            #noise = noise_sampler(sigma=sigma, sigma_next=sigma_next)
            #noise = (noise - noise.mean()) / noise.std()
            #x = alpha_ratio * x + noise * s_noise * sigma_up

    return denoised







def sample_rk_test3(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options=""):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    
    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
    
    h = sigmas[1] - sigmas[0]
    k1, data1 = epsilon(model, x, sigmas[0], **extra_args)
    x_2 = x + h * (11/200) * k1
    k2, data2 = epsilon(model, x_2, sigmas[0] + h * (2/3), **extra_args)
    
    xp  =  ((1/4)*k1 + (1/4)*k2)
    xpp =  ((1/4)*k1 + (3/4)*k2)
    #xp = k1.clone() #* (sigmas[1] - sigmas[0])
    #xpp = k1.clone() * (sigmas[1] - sigmas[0])
    #xpp = k1.clone() * (sigmas[1] - sigmas[0])

    h = -torch.log(sigmas[1] / sigmas[0])
    k1, data1 = epsilon_res(model, x, sigmas[0], **extra_args)
    a2_1 = c2 * phi(1, -h*c2)
    x_2 = x + h * a2_1 * k1
    k2, data2 = epsilon_res(model, x_2, sigmas[0] + h * c2, **extra_args)

    xp, xpp = (k1.clone() for _ in range(2))
    xpp = (k2 - k1) 
    
    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]

        #sigma_up, sigma, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, eta, eta_var, noise_mode, sigma_next-sigma )
        #h = sigma_down - sigma
        
        sigma_up, sigma, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, eta, eta_var, noise_mode, -torch.log(sigma_next/sigma))
        h = -torch.log(sigma_down/sigma)
        
        a2_1 = c2 * phi(1, -h*c2)
        b1 =        phi(1, -h) - phi(2, -h)/c2
        b2 =        phi(2, -h)/c2
                
        if sigma_down == 0:
            denoised = model(x, sigma * s_in, **extra_args)

        else:
            
            
            
            #k1, data1 = epsilon(model, x, sigma, **extra_args)
            
            #x_2 = x   +   h*(2/3)*xp   +   ((h**2)/2)*((2/3)**2) * xpp   +   (h**3) * ((11/200)*k1)
            #k2, data2 = epsilon(model, x_2, sigma + h*(2/3), **extra_args)
            #x   =   x + h*xp + ((h**2)/2)*xpp + (h**3) * ((1/8)*k1 + (1/24)*k2)
            
            k1, data1 = epsilon_res(model, x, sigma, **extra_args)
            x_2 = x   +   h*c2*xp   +   ((h**2)/2)*(c2**2) * xpp   +   (h**3) * (a2_1*k1)
            k2, data2 = epsilon_res(model, x_2, sigma + h*c2, **extra_args)
            x   =   x + h*xp + ((h**2)/2)*xpp + (h**3) * (b1*k1 + b2*k2)
            
            #xp = k1.clone()
            #xpp = (k2 - k1) 
            
            xp  =  xp + h*xpp + h**2 * (b1*k1 + b2*k2)
            xpp = xpp + h * (b1*k1 + b2*k2)
            
            #xp  =  xp + h*xpp + h**2 * ((1/4)*k1 + (1/4)*k2)
            
            #xpp = (k2 - k1) / h
            #xpp = xpp + h * ((1/4)*k1 + (3/4)*k2)


            
            denoised = data2
            
        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised})

        noise = noise_sampler(sigma=sigma, sigma_next=sigma_next)
        noise = (noise - noise.mean()) / noise.std()
        x = alpha_ratio * x + noise * s_noise * sigma_up

    return denoised








def sample_rk_test6th(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options=""):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    
    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
    
    h = sigmas[1] - sigmas[0]
    a2_1 = 7/120 - (3 * 15**0.5)/200
    
    a3_1 = -1/96 + (15**0.5)/480
    a3_2 = 1/32 - (15**0.5)/480
    
    a4_1 = -1/600 + (15**0.5)/600
    a4_2 = (15**0.5)/50
    a4_3 = 3/50 - (15**0.5)/150
    
    b1 = 0
    b2 = 1/18 + (15**0.5)/72
    b3 = 1/18
    b4 = 1/18 - (15**0.5)/72
    
    b1p = 0
    b2p = 5/36 + (15**0.5)/36
    b3p = 2/9
    b4p = 5/36 - (15**0.5)/36
    
    b1pp = 0
    b2pp = 5/18
    b3pp = 4/9
    b4pp = 5/18
    
    c2 = 1/2 - (15**0.5)/10
    c3 = 1/2
    c4 = 1/2 + (15**0.5)/10
    
    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
    
    h = sigmas[1] - sigmas[0]
    
    k1, data1 = epsilon(model, x, sigmas[0], **extra_args)
    
    x_2 = x + h *(a2_1*k1)
    k2, data2 = epsilon(model, x_2, sigmas[0] + h * c2, **extra_args)
    
    x_3 = x + h * (a3_1*k1 + a3_2*k2)
    k3, data3 = epsilon(model, x_3, sigmas[0] + h * c3, **extra_args)
    
    x_4 = x + h * (a4_1*k1 + a4_2*k2 + a4_3*k3)
    k4, data4 = epsilon(model, x_4, sigmas[0] + h * c4, **extra_args)

    xp1, xp2, xp3, xp4 = k1, k2, k3, k4
    
    xpp1 = (k2 - k1) / (h*c2)
    xpp2 = (k3 - k1) / (h*c3)
    xpp3 = (k4 - k1) / (h*c4)
    
    xp = k1
    xpp = (k4 - k1) / (h)
    
    x = x + h * ((1/4)*k1 + (1/4)*k2 + (1/4)*k3 + (1/4)*k4)
    
    if not extra_options_flag("disable_proper_derivs_init", extra_options):
        xp  =  xp + h*xpp + h**2 * (b1p*k1 + b2p*k2 + b3p*k3 + b4p*k4)
        xpp = xpp + h * (b1pp*k1 + b2pp*k2 + b3pp*k3 + b4pp*k4)
    
    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]

        sigma_up, sigma, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, eta, eta_var, noise_mode, sigma_next-sigma )
        h = sigma_down - sigma
        
        if sigma_down == 0:
            denoised = model(x, sigma * s_in, **extra_args)

        else:            
            k1, data1 = epsilon(model, x, sigma, **extra_args)
            
            #x_2 = x + h *(a2_1*k1)
            x_2 = x + h*c2*xp + ((h**2)/2)*(c2**2)*xpp + (h**3) * (a2_1*k1)
            k2, data2 = epsilon(model, x_2, sigma + h * c2, **extra_args)
            
            #x_3 = x + h * (a3_1*k1 + a3_2*k2)
            x_3 = x + h*c3*xp + ((h**2)/2)*(c3**2)*xpp + (h**3) * (a3_1*k1 + a3_2*k2)
            k3, data3 = epsilon(model, x_3, sigma + h * c3, **extra_args)
            
            #x_4 = x + h * (a4_1*k1 + a4_2*k2 + a4_3*k3)
            x_3 = x + h*c4*xp + ((h**2)/2)*(c4**2)*xpp + (h**3) * (a4_1*k1 + a4_2*k2 + a4_3*k3)
            k4, data4 = epsilon(model, x_4, sigma + h * c4, **extra_args)
            
            x = x + h*xp + ((h**2)/2)*xpp + (h**3) * (b1*k1 + b2*k2 + b3*k3 + b4*k4)

            
            if not extra_options_flag("enable_proper_derivs", extra_options):
                xp = k1.clone()
                #xp = h * (b1p * k1 + b2p * k2 + b3p * k3 + b4p * k4)
                xpp = (k4 - k1) / (h)
                xp1, xp2, xp3, xp4 = k1, k2, k3, k4
    
                xpp1 = (k2 - k1) / (h*c2)
                xpp2 = (k3 - k1) / (h*c3)
                xpp3 = (k4 - k1) / (h*c4)
            
            if not extra_options_flag("disable_proper_derivs", extra_options):
                xp  =  xp + h*xpp + h**2 * (b1p*k1 + b2p*k2 + b3p*k3 + b4p*k4)
                xpp = xpp + h * (b1pp*k1 + b2pp*k2 + b3pp*k3 + b4pp*k4)

            denoised = data4
            
        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised})

        noise = noise_sampler(sigma=sigma, sigma_next=sigma_next)
        noise = (noise - noise.mean()) / noise.std()
        x = alpha_ratio * x + noise * s_noise * sigma_up

    return denoised



def sample_rk_test3rd(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options=""):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    
    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
    
    h = sigmas[1] - sigmas[0]
    a2_1 = 1/48
    
    a3_1 = 1/12
    a3_2 = 1/12

    
    b1 = 1/12
    b2 = 1/12
    b3 = 0
    
    b1p = 1/6
    b2p = 1/3
    b3p = 0
    
    b1pp = 1/6
    b2pp = 2/3
    b3pp = 1/6
    
    c2 = 1/2
    c3 = 1
    
    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
    
    h = sigmas[1] - sigmas[0]
    
    k1, data1 = epsilon(model, x, sigmas[0], **extra_args)
    
    x_2 = x + h *(a2_1*k1)
    k2, data2 = epsilon(model, x_2, sigmas[0] + h * c2, **extra_args)
    
    x_3 = x + h * (a3_1*k1 + a3_2*k2)
    k3, data3 = epsilon(model, x_3, sigmas[0] + h * c3, **extra_args)

    xp1, xp2, xp3 = k1, k2, k3
    
    xpp1 = (k2 - k1) / (h*c2)
    xpp2 = (k3 - k1) / (h*c3)
    
    xp = k1
    xpp = (k3 - k1) / (h)
    
    x = x + h * ((1/3)*k1 + (1/3)*k2 + (1/3)*k3)
    
    if not extra_options_flag("disable_proper_derivs_init", extra_options):
        xp  =  xp + h*xpp + h**2 * (b1p*k1 + b2p*k2 + b3p*k3)
        xpp = xpp + h * (b1pp*k1 + b2pp*k2 + b3pp*k3)
    
    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]

        sigma_up, sigma, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, eta, eta_var, noise_mode, sigma_next-sigma )
        h = sigma_down - sigma
        
        if sigma_down == 0:
            denoised = model(x, sigma * s_in, **extra_args)

        else:            
            k1, data1 = epsilon(model, x, sigma, **extra_args)
            
            #x_2 = x + h *(a2_1*k1)
            x_2 = x + h*c2*xp + ((h**2)/2)*(c2**2)*xpp + (h**3) * (a2_1*k1)
            k2, data2 = epsilon(model, x_2, sigma + h * c2, **extra_args)
            
            #x_3 = x + h * (a3_1*k1 + a3_2*k2)
            x_3 = x + h*c3*xp + ((h**2)/2)*(c3**2)*xpp + (h**3) * (a3_1*k1 + a3_2*k2)
            k3, data3 = epsilon(model, x_3, sigma + h * c3, **extra_args)

            x = x + h*xp + ((h**2)/2)*xpp + (h**3) * (b1*k1 + b2*k2 + b3*k3)

            
            if not extra_options_flag("enable_proper_derivs", extra_options):
                xp = k1.clone()
                #xp = h * (b1p * k1 + b2p * k2 + b3p * k3 + b4p * k4)
                xpp = (k3 - k1) / (h)
                xp1, xp2, xp3 = k1, k2, k3
    
                xpp1 = (k2 - k1) / (h*c2)
                xpp2 = (k3 - k1) / (h*c3)
            
            if not extra_options_flag("disable_proper_derivs", extra_options):
                xp  =  xp + h*xpp + h**2 * (b1p*k1 + b2p*k2 + b3p*k3)
                xpp = xpp + h * (b1pp*k1 + b2pp*k2 + b3pp*k3)

            denoised = data3
            
        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised})

        noise = noise_sampler(sigma=sigma, sigma_next=sigma_next)
        noise = (noise - noise.mean()) / noise.std()
        x = alpha_ratio * x + noise * s_noise * sigma_up

    return denoised







def sample_rk_test(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options=""):
    
    if extra_options_flag("logit", extra_options) or extra_options_flag("logsnr", extra_options):
        return sample_rk_logit(model, x, sigmas, extra_args, callback, disable, noise_sampler, noise_sampler_type, noise_mode, rk_type, 
              sigma_fn_formula, t_fn_formula,
                  eta, eta_var, s_noise, alpha, k, scale, c2, c3, buffer, cfgpp, iter, sub_iter, reverse_weight, extra_options)
    
    if extra_options_flag("sixth_order", extra_options):
        return sample_rk_test6th(model, x, sigmas, extra_args, callback, disable, noise_sampler, noise_sampler_type, noise_mode, rk_type, 
              sigma_fn_formula, t_fn_formula,
                  eta, eta_var, s_noise, alpha, k, scale, c2, c3, buffer, cfgpp, iter, sub_iter, reverse_weight, extra_options)
    
    if extra_options_flag("third", extra_options):
        return sample_rk_test3rd(model, x, sigmas, extra_args, callback, disable, noise_sampler, noise_sampler_type, noise_mode, rk_type, 
              sigma_fn_formula, t_fn_formula,
                  eta, eta_var, s_noise, alpha, k, scale, c2, c3, buffer, cfgpp, iter, sub_iter, reverse_weight, extra_options)
    
    
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    
    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
    
    h = sigmas[1] - sigmas[0]
    k1, data1 = epsilon(model, x, sigmas[0], **extra_args)
    x_2 = x + h * (11/200) * k1
    k2, data2 = epsilon(model, x_2, sigmas[0] + h * (2/3), **extra_args)
    
    xp  =  ((1/4)*k1 + (1/4)*k2)
    xpp =  ((1/4)*k1 + (3/4)*k2)

    xp, xpp = (k1.clone() for _ in range(2))
    xpp = (k2 - k1) / (h*(2/3))
    
    x = x + h * ((1/2)*k1 + (1/2)*k2)
    #xp  =  xp + h*xpp + h**2 * ((1/4)*k1 + (1/4)*k2)
    #xpp = xpp + h * ((1/4)*k1 + (3/4)*k2)
    
    if not extra_options_flag("disable_proper_derivs_init", extra_options):
        xp  =  xp + h*xpp + h**2 * ((1/4)*k1 + (1/4)*k2)
        xpp = xpp + h * ((1/4)*k1 + (3/4)*k2)
    
    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]

        sigma_up, sigma, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, eta, eta_var, noise_mode, sigma_next-sigma )
        h = sigma_down - sigma
        
        if sigma_down == 0:
            denoised = model(x, sigma * s_in, **extra_args)

        else:
        
            k1, data1 = epsilon(model, x, sigma, **extra_args)
            
            x_2 = x   +   h*(2/3)*xp   +   ((h**2)/2)*((2/3)**2) * xpp   +   ((h**3)) * ((11/200)*k1)
            k2, data2 = epsilon(model, x_2, sigma + h*(2/3), **extra_args)
            x   =   x + h*xp + ((h**2)/2)*xpp + ((h**3)) * ((1/8)*k1 + (1/24)*k2)
            
            
            if not extra_options_flag("enable_proper_derivs", extra_options):
                xp = k1.clone()
                xpp = (k2 - k1) / (h*(2/3))
            
            if not extra_options_flag("disable_proper_derivs", extra_options):
                xp  =  xp + h*xpp + h**2 * ((1/4)*k1 + (1/4)*k2)
                
                #xpp = (k2 - k1) / h
                xpp = xpp + h * ((1/4)*k1 + (3/4)*k2)

            denoised = data2
            
        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised})

        noise = noise_sampler(sigma=sigma, sigma_next=sigma_next)
        noise = (noise - noise.mean()) / noise.std()
        x = alpha_ratio * x + noise * s_noise * sigma_up

    return denoised





def sample_rk_test2(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options=""):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    seed = torch.initial_seed() + 1
    
    sigma_min, sigma_max = model.inner_model.inner_model.model_sampling.sigma_min, model.inner_model.inner_model.model_sampling.sigma_max
    
    t_fn = lambda sigma: sigma
    sigma_fn = lambda t: T
    h_fn = lambda sigma, sigma_down: sigma_down - sigma
    
    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=seed, sigma_min=0.0, sigma_max=1.0)
    
    if noise_sampler_type == "fractal":
        noise_sampler.alpha = alpha
        noise_sampler.k = k
        noise_sampler.scale = scale
        
    a = [[0,0], [11/200, 0]]
    b = [
        [1/8, 1/24],
        [1/4, 1/4], 
        [1/4, 3/4],
    ]
    c = [0, 2/3]

    xp, xpp = (torch.zeros_like(x) for _ in range(2))
    k1, data1 = epsilon(model, x, sigmas[0], **extra_args)
    xp, xpp = k1.clone(), k1.clone()
    #xp, xpp = (x.clone() for _ in range(2))
    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
        
        #h_orig = t_fn(sigma_next)-t_fn(sigma)
        #sigma_up, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, eta, eta_var, noise_mode, h_orig)
        #sigma_down = sigma_next
        #t_down, t = t_fn(sigma_down), t_fn(sigma)
        #h = h_fn(sigma, sigma_down)
        h = sigma_next - sigma
                
        if sigma_next == 0:
            denoised = model(x, sigma * s_in, **extra_args)

        else:
            k1, data1 = epsilon_res(model, x, sigma, **extra_args)
            
            x_2 = x   +   h*(2/3)*xp   +   ((h**2)/2)*((2/3)**2) * xpp   +   (h**3) * ((11/200)*k1)
            k2, data2 = epsilon(model, x_2, sigma + h*(2/3), **extra_args)
            

            k1, data1 = epsilon(model, x, sigma, **extra_args)
            x   =   x + h*xp + ((h**2)/2)*xpp + (h**3) * ((1/8)*k1 + (1/24)*k2)
            
            xp  =  xp + h*xpp + h**2 * ((1/4)*k1 + (1/4)*k2)
            xpp = xpp + h * ((1/4)*k1 + (3/4)*k2)
            
            denoised = data2
            
         
        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised})
            
        """noise = noise_sampler(sigma=sigma, sigma_next=sigma_next)
        noise = (noise - noise.mean()) / noise.std()
        x = alpha_ratio * x + noise * s_noise * sigma_up"""

    return denoised












class SamplerRK_Test:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {#"momentum": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step":0.01, "round": False}),
                     "eta": ("FLOAT", {"default": 0.25, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Calculated noise amount to be added, then removed, after each step."}),
                     "eta_var": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Calculate variance-corrected noise amount (overrides eta/noise_mode settings). Cannot be used at very low sigma values; reverts to eta/noise_mode for final steps."}),
                     "s_noise": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Ratio of calculated noise amount actually added after each step. >1.0 will leave extra noise behind, <1.0 will remove more noise than it adds."}),
                     "alpha": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step":0.1, "round": False, "tooltip": "Fractal noise mode: <0 = extra high frequency noise, >0 = extra low frequency noise, 0 = white noise."}),
                     "k": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step":2.0, "round": False, "tooltip": "Fractal noise mode: all that matters is positive vs. negative. Effect unclear."}),
                     "cfgpp": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step":0.01, "round": False, "tooltip": "CFG++ scale. Replaces CFG."}),
                     "noise_sampler_type": (NOISE_GENERATOR_NAMES, {"default": "gaussian"}),
                     "noise_mode": (["hard", "hard_sq", "soft", "softer", "exp"], {"default": 'hard', "tooltip": "How noise scales with the sigma schedule. Hard is the most aggressive, the others start strong and drop rapidly."}),
                     "iter": ("INT", {"default": 0, "min": 0, "max": 100, "step":1, "tooltip": "Number of implicit refinement steps to run after each explicit step. Currently only working with CFG."}),
                     "sub_iter": ("INT", {"default": 0, "min": 0, "max": 100, "step":1, "tooltip": "Number of implicit refinement steps to run after each explicit step. Currently only working with CFG."}),
                     #"t_fn_formula": ("STRING", {"default": "1/((sigma).exp()+1)", "multiline": True}),
                     #"sigma_fn_formula": ("STRING", {"default": "((1-t)/t).log()", "multiline": True}),
                    "extra_options": ("STRING", {"default": "", "multiline": True}),   
                    },
                    "optional": 
                    {
                    }  
               }
    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "get_sampler"

    def get_sampler(self, eta=0.25, eta_var=0.0, s_noise=1.0, alpha=-1.0, k=1.0, cfgpp=0.0, buffer=0, extra_options="", noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", t_fn_formula=None, sigma_fn_formula=None, iter=0, sub_iter=0,
                    ):
        sampler_name = "rk_test"

        steps = 10000

        sampler = comfy.samplers.ksampler(sampler_name, {"eta": eta, "eta_var": eta_var, "s_noise": s_noise, "alpha": alpha, "k": k, "cfgpp": cfgpp, "buffer": buffer, "noise_sampler_type": noise_sampler_type, "noise_mode": noise_mode, "rk_type": rk_type, 
                                                         "t_fn_formula": t_fn_formula, "sigma_fn_formula": sigma_fn_formula, "iter": iter,"sub_iter": sub_iter, "extra_options": extra_options})
        return (sampler, )





class UltraSharkSamplerRBTest:  
    # for use with https://github.com/ClownsharkBatwing/UltraCascade
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "add_noise": ("BOOLEAN", {"default": True}),
                "noise_is_latent": ("BOOLEAN", {"default": False}),
                "noise_type": (NOISE_GENERATOR_NAMES, ),
                "alpha": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step":0.1, "round": 0.01}),
                "k": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step":2.0, "round": 0.01}),
                "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "cfg": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 100.0, "step":0.5, "round": 0.01}),
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "sampler": ("SAMPLER", ),
                "sigmas": ("SIGMAS", ),
                "latent_image": ("LATENT", ),               
                "guide_type": (['residual', 'weighted'], ),
                "guide_weight": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step":0.01, "round": 0.01}),
            },
            "optional": {
                "latent_noise": ("LATENT", ),
                "guide": ("LATENT",),
                "guide_weights": ("SIGMAS",),
                "style": ("CONDITIONING", ),
                "img_style": ("CONDITIONING", ),
            }
        }

    RETURN_TYPES = ("LATENT","LATENT","LATENT")
    RETURN_NAMES = ("output", "denoised_output", "latent_batch")

    FUNCTION = "main"

    CATEGORY = "RES4LYF/samplers/ultracascade"
    DESCRIPTION = "For use with Stable Cascade and UltraCascade."
    
    def main(self, model, add_noise, noise_is_latent, noise_type, noise_seed, cfg, alpha, k, positive, negative, sampler, 
               sigmas, guide_type, guide_weight, latent_image, latent_noise=None, guide=None, guide_weights=None, style=None, img_style=None): 

            if model.model.model_config.unet_config.get('stable_cascade_stage') == 'up':
                model = model.clone()
                x_lr = guide['samples'] if guide is not None else None
                guide_weights = initialize_or_scale(guide_weights, guide_weight, 10000)#("FLOAT", {"default": 1.0, "min": -10000, "max": 10000, "step":0.01}),
                #model.model.diffusion_model.set_guide_weights(guide_weights=guide_weights)
                #model.model.diffusion_model.set_guide_type(guide_type=guide_type)
                #model.model.diffusion_model.set_x_lr(x_lr=x_lr)
                patch = model.model_options.get("transformer_options", {}).get("patches_replace", {}).get("ultracascade", {}).get("main")
                if patch is not None:
                    patch.update(x_lr=x_lr, guide_weights=guide_weights, guide_type=guide_type)
                else:
                    model.model.diffusion_model.set_sigmas_schedule(sigmas_schedule=sigmas)
                    model.model.diffusion_model.set_sigmas_prev(sigmas_prev=sigmas[:1])
                    model.model.diffusion_model.set_guide_weights(guide_weights=guide_weights)
                    model.model.diffusion_model.set_guide_type(guide_type=guide_type)
                    model.model.diffusion_model.set_x_lr(x_lr=x_lr)
                
            elif model.model.model_config.unet_config['stable_cascade_stage'] == 'b':
                c_pos, c_neg = [], []
                for t in positive:
                    d_pos = t[1].copy()
                    d_neg = t[1].copy()
                    
                    d_pos['stable_cascade_prior'] = guide['samples']

                    pooled_output = d_neg.get("pooled_output", None)
                    if pooled_output is not None:
                        d_neg["pooled_output"] = torch.zeros_like(pooled_output)
                    
                    c_pos.append([t[0], d_pos])            
                    c_neg.append([torch.zeros_like(t[0]), d_neg])
                positive = c_pos
                negative = c_neg
                
            if style is not None:
                model.set_model_patch(style, 'style_cond')
            if img_style is not None:
                model.set_model_patch(img_style,'img_style_cond')
        
        
            # 1, 768      clip_style[0][0][1]['unclip_conditioning'][0]['clip_vision_output'].image_embeds.shape
            # 1, 1280     clip_style[0][0][1]['pooled_output'].shape 
            # 1, 77, 1280 clip_style[0][0][0].shape
        
        
            latent = latent_image
            latent_image = latent["samples"]
            torch.manual_seed(noise_seed)

            if not add_noise:
                noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
            elif latent_noise is None:
                batch_inds = latent["batch_index"] if "batch_index" in latent else None
                noise = prepare_noise(latent_image, noise_seed, noise_type, batch_inds, alpha, k)
            else:
                noise = latent_noise["samples"]#.to(torch.float64)

            if noise_is_latent:
                noise += latent_image.cpu()
                noise.sub_(noise.mean()).div_(noise.std())

            noise_mask = None
            if "noise_mask" in latent:
                noise_mask = latent["noise_mask"]

            x0_output = {}
            callback = latent_preview.prepare_callback(model, sigmas.shape[-1] - 1, x0_output)
            disable_pbar = False

            samples = comfy.sample.sample_custom(model, noise, cfg, sampler, sigmas, positive, negative, latent_image, 
                                                 noise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, 
                                                 seed=noise_seed)

            out = latent.copy()
            out["samples"] = samples
            if "x0" in x0_output:
                out_denoised = latent.copy()
                out_denoised["samples"] = model.model.process_latent_out(x0_output["x0"].cpu())
            else:
                out_denoised = out
                
            return (out, out_denoised)



def sample_rk_res_2s_orig_derp(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="res_2s", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    sigmas = sigmas.to(torch.float64)
    x = x.to(torch.float64)

    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
        x_0 = x.clone()
        
        h = -torch.log(sigma_next/sigma)
        
        c2 = 0.5

        a2_1 = c2 * phi(1, -h*c2)
        b1 =        phi(1, -h) - phi(2, -h)/c2
        b2 =        phi(2, -h)/c2

        denoised = model(x, sigma * s_in, **extra_args)
        if sigma_next == 0:
            return denoised
        
        eps = denoised - x
        
        x_2 = x + h * (a2_1 * eps)
        
        s2 = -torch.log(sigma) + h * c2
        sigma_2 = torch.exp(-s2)
        
        denoised_2 = model(x_2, sigma_2 * s_in, **extra_args)
        eps_2 = denoised_2 - x

        x = x + h * (b1 * eps + b2 * eps_2)

        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised})

    return denoised



