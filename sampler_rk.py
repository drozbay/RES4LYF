import torch
from torch import FloatTensor
from tqdm.auto import trange
from math import pi
import gc
import math

import torch.nn.functional as F
import torchvision.transforms as T

import functools

from .noise_classes import *
#from .extra_samplers_helpers import get_deis_coeff_list

import comfy.model_patcher

from .noise_sigmas_timesteps_scaling import get_res4lyf_step_with_model


def phi(j, neg_h):
  remainder = torch.zeros_like(neg_h)
  
  for k in range(j): 
    remainder += (neg_h)**k / math.factorial(k)
  phi_j_h = ((neg_h).exp() - remainder) / (neg_h)**j
  
  return phi_j_h
  
  
def calculate_gamma(c2, c3):
    return (3*(c3**3) - 2*c3) / (c2*(2 - 3*c2))


rk_coeff = {
    "dormand-prince_13s": (
        [
            [1/18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1/48, 1/16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1/32, 0, 3/32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [5/16, 0, -75/64, 75/64, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [3/80, 0, 0, 3/16, 3/20, 0, 0, 0, 0, 0, 0, 0, 0],
            [29443841/614563906, 0, 0, 77736538/692538347, -28693883/1125000000, 23124283/1800000000, 0, 0, 0, 0, 0, 0, 0],
            [16016141/946692911, 0, 0, 61564180/158732637, 22789713/633445777, 545815736/2771057229, -180193667/1043307555, 0, 0, 0, 0, 0, 0],
            [39632708/573591083, 0, 0, -433636366/683701615, -421739975/2616292301, 100302831/723423059, 790204164/839813087, 800635310/3783071287, 0, 0, 0, 0, 0],
            [246121993/1340847787, 0, 0, -37695042795/15268766246, -309121744/1061227803, -12992083/490766935, 6005943493/2108947869, 393006217/1396673457, 123872331/1001029789, 0, 0, 0, 0],
            [-1028468189/846180014, 0, 0, 8478235783/508512852, 1311729495/1432422823, -10304129995/1701304382, -48777925059/3047939560, 15336726248/1032824649, -45442868181/3398467696, 3065993473/597172653, 0, 0, 0],
            [185892177/718116043, 0, 0, -3185094517/667107341, -477755414/1098053517, -703635378/230739211, 5731566787/1027545527, 5232866602/850066563, -4093664535/808688257, 3962137247/1805957418, 65686358/487910083, 0, 0],
            [403863854/491063109, 0, 0, -5068492393/434740067, -411421997/543043805, 652783627/914296604, 11173962825/925320556, -13158990841/6184727034, 3936647629/1978049680, -160528059/685178525, 248638103/1413531060, 0, 0],
            [14005451/335480064, 0, 0, 0, 0, -59238493/1068277825, 181606767/758867731, 561292985/797845732, -1041891430/1371343529, 760417239/1151165299, 118820643/751138087, -528747749/2220607170, 1/4]
        ],
        [0, 1/18, 1/12, 1/8, 5/16, 3/8, 59/400, 93/200, 5490023248 / 9719169821, 13/20, 1201146811 / 1299019798, 1, 1],
    ),
    "dormand-prince_6s": (
        [
            [1/5, 0, 0, 0, 0, 0, 0],
            [3/40, 9/40, 0, 0, 0, 0, 0],
            [44/45, -56/15, 32/9, 0, 0, 0, 0],
            [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0, 0],
            [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0],
            [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0],
        ],
        [0, 1/5, 3/10, 4/5, 8/9, 1],
    ),
    "dormand-prince_7s": (
        [
            [1/5, 0, 0, 0, 0, 0, 0],
            [3/40, 9/40, 0, 0, 0, 0, 0],
            [44/45, -56/15, 32/9, 0, 0, 0, 0],
            [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0, 0],
            [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0],
            [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0],
        ],
        [0, 1/5, 3/10, 4/5, 8/9, 1],
    ),
    "rk4_4s": (
        [
            [1/2, 0, 0, 0],
            [0, 1/2, 0, 0],
            [0, 0, 1, 0],
            [1/6, 1/3, 1/3, 1/6]
        ],
        [0, 1/2, 1/2, 1],
    ),
    "rk38_4s": (
        [
            [1/3, 0, 0, 0],
            [-1/3, 1, 0, 0],
            [1, -1, 1, 0],
            [1/8, 3/8, 3/8, 1/8]
        ],
        [0, 1/3, 2/3, 1],
    ),
    "ralston_4s": (
        [
            [2/5, 0, 0, 0],
            [(-2889+1428 * 5**0.5)/1024,   (3785-1620 * 5**0.5)/1024,  0, 0],
            [(-3365+2094 * 5**0.5)/6040,   (-975-3046 * 5**0.5)/2552,  (467040+203968*5**0.5)/240845, 0],
            [(263+24*5**0.5)/1812, (125-1000*5**0.5)/3828, (3426304+1661952*5**0.5)/5924787, (30-4*5**0.5)/123]
        ],
        [0, 2/5, (14-3 * 5**0.5)/16, 1],
    ),
    "heun_3s": (
        [
            [1/3, 0, 0],
            [0, 2/3, 0],
            [1/4, 0, 3/4]
        ],
        [0, 1/3, 2/3],
    ),
    "kutta_3s": (
        [
            [1/2, 0, 0],
            [-1, 2, 0],
            [1/6, 2/3, 1/6]
        ],
        [0, 1/2, 1],
    ),
    "ralston_3s": (
        [
            [1/2, 0, 0],
            [0, 3/4, 0],
            [2/9, 1/3, 4/9]
        ],
        [0, 1/2, 3/4],
    ),
    "houwen-wray_3s": (
        [
            [8/15, 0, 0],
            [1/4, 5/12, 0],
            [1/4, 0, 3/4]
        ],
        [0, 8/15, 2/3],
    ),
    "ssprk3_3s": (
        [
            [1, 0, 0],
            [1/4, 1/4, 0],
            [1/6, 1/6, 2/3]
        ],
        [0, 1, 1/2],
    ),
    "midpoint_2s": (
        [
            [1/2, 0],
            [0, 1]
        ],
        [0, 1/2],
    ),
    "heun_2s": (
        [
            [1, 0],
            [1/2, 1/2]
        ],
        [0, 1],
    ),
    "ralston_2s": (
        [
            [2/3, 0],
            [1/4, 3/4]
        ],
        [0, 2/3],
    ),
    "euler": (
        [
            [1],
        ],
        [0],
    ),
}

def get_rk_methods(rk_type, h, c2=0.5, c3=0.75, h_prev=None):
    FSAL = False
    if rk_type in rk_coeff:
        ab, ci = rk_coeff[rk_type]
        ci = ci[:]
        ci.append(0)
        model_call = get_epsilon
        alpha_fn = lambda h: 1
        t_fn = lambda sigma: sigma
        sigma_fn = lambda t: t
        EPS_PRED = True
    else:
        model_call = get_denoised
        alpha_fn = lambda neg_h: torch.exp(neg_h)
        t_fn = lambda sigma: sigma.log().neg()
        sigma_fn = lambda t: t.neg().exp()
        EPS_PRED = False
    
    match rk_type:
        case "dormand-prince_6s":
            FSAL = True

        case "ddim":
            b1 = phi(1, -h)
            ab = [
                    [b1],
            ]
            ci = [0, 1]
            
        case "res_2s":
            #FSAL = True
            a2_1 = c2 * phi(1, -h*c2)
            b1 =        phi(1, -h) - phi(2, -h)/c2
            b2 =        phi(2, -h)/c2
            ab = [
                    [a2_1, 0],
                    [b1, b2],
            ]
            ci = [0, c2, 1]

        case "res_2m":
            FSAL = True
            if h_prev == None:
                return get_rk_methods("res_2s")
            
            c2 = -h_prev / h
            #a2_1 = c2 * phi(1, -h*c2)
            b1 =        phi(1, -h) - phi(2, -h)/c2
            b2 =        phi(2, -h)/c2
            ab = [
                    [b1, b2],
            ]
            ci = [0]
            
        case "res_3s":
            gamma = calculate_gamma(c2, c3)
            a2_1 = c2 * phi(1, -h*c2)
            a3_2 = gamma * c2 * phi(2, -h*c2) + (c3 ** 2 / c2) * phi(2, -h*c3) #phi_2_c3_h  # a32 from k2 to k3
            a3_1 = c3 * phi(1, -h*c3) - a3_2 # a31 from k1 to k3
            b3 = (1 / (gamma * c2 + c3)) * phi(2, -h)      
            b2 = gamma * b3  #simplified version of: b2 = (gamma / (gamma * c2 + c3)) * phi_2_h  
            b1 = phi(1, -h) - b2 - b3     
            ab = [
                    [a2_1, 0, 0],
                    [a3_1, a3_2, 0],
                    [b1, b2, b3],
            ]
            ci = [0, c2, c3, 1]

        case "dpmpp_2s":
            c2 = 0.5
            a2_1 =         c2   * phi(1, -h*c2)
            b1 = (1 - 1/(2*c2)) * phi(1, -h)
            b2 =     (1/(2*c2)) * phi(1, -h)
            ab = [
                    [a2_1, 0],
                    [b1, b2],
            ]
            ci = [0, c2, 1]

        case "dpmpp_sde_2s":
            c2 = 1.0
            a2_1 =         c2   * phi(1, -h*c2)
            b1 = (1 - 1/(2*c2)) * phi(1, -h)
            b2 =     (1/(2*c2)) * phi(1, -h)
            ab = [
                    [a2_1, 0],
                    [b1, b2],
            ]
            ci = [0, c2, 1]

        case "dpmpp_3s":
            a2_1 = c2 * phi(1, -h*c2)
            a3_2 = (c3**2 / c2) * phi(2, -h*c3)
            a3_1 = c3 * phi(1, -h*c3) - a3_2
            b2 = 0
            b3 = (1/c3) * phi(2, -h)
            b1 = phi(1, -h) - b2 - b3
            ab = [
                    [a2_1, 0, 0],
                    [a3_1, a3_2, 0],
                    [b1, b2, b3],
            ]
            ci = [0, c2, c3, 1]

    return ab, ci, model_call, alpha_fn, t_fn, sigma_fn, FSAL, EPS_PRED

def get_rk_methods_order(rk_type):
    ab, ci, model_call, alpha_fn, t_fn, sigma_fn, FSAL, EPS_PRED = get_rk_methods(rk_type, torch.tensor(1.0).to('cuda').to(torch.float64), c2=0.5, c3=0.75)
    return len(ci)-1

def get_rk_methods_order_and_fn(rk_type):
    ab, ci, model_call, alpha_fn, t_fn, sigma_fn, FSAL, EPS_PRED = get_rk_methods(rk_type, torch.tensor(1.0).to('cuda').to(torch.float64), c2=0.5, c3=0.75)
    MULTISTEP=False
    multistep_buffer_size = len(ab[0]) - len(ab) # x dim - y dim
    if len(ab) < len(ci):
        MULTISTEP=True
    return len(ci)-1, model_call, alpha_fn, t_fn, sigma_fn, FSAL, EPS_PRED #MULTISTEP

def get_rk_methods_coeff(rk_type, h, c2, c3, h_prev=None):
    ab, ci, model_call, alpha_fn, t_fn, sigma_fn, FSAL, EPS_PRED = get_rk_methods(rk_type, h, c2, c3, h_prev)
    return ab, ci

def get_epsilon(model, x, sigma, **extra_args):
    s_in = x.new_ones([x.shape[0]])
    x0 = model(x, sigma * s_in, **extra_args)
    eps = (x - x0) / (sigma * s_in) 
    return eps

def get_denoised(model, x, sigma, **extra_args):
    s_in = x.new_ones([x.shape[0]])
    x0 = model(x, sigma * s_in, **extra_args)
    return x0

 
def sample_rk(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="brownian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5):
    extra_args = {} if extra_args is None else extra_args
    
    
    #if cfgpp.sum().item() != 0.0:
    temp = [0]
    temp[0] = torch.full_like(x, 0.0)
    if cfgpp != 0.0:
        #temp = [0]
        def post_cfg_function(args):
            temp[0] = args["uncond_denoised"]
            return args["denoised"]
        model_options = extra_args.get("model_options", {}).copy()
        extra_args["model_options"] = comfy.model_patcher.set_model_options_post_cfg_function(model_options, post_cfg_function, disable_cfg1_optimization=True)

    BUF_ELEM=buffer
    seed = torch.initial_seed() + 1
    
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    if isinstance(model.inner_model.inner_model.model_sampling, comfy.model_sampling.CONST):
        sigma_max = torch.full_like(sigma_max, 1.0)
        sigma_min = torch.full_like(sigma_min, min(sigma_min.item(), 0.00001))
    if sigmas[-1] == 0.0:
        sigmas[-1] = min(sigmas[-2]**2, 0.00001)
    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=seed, sigma_min=0.0, sigma_max=1.0)
    
    if noise_sampler_type == "fractal":
        noise_sampler.alpha = alpha
        noise_sampler.k = k
        noise_sampler.scale = scale
        
    order, model_call, alpha_fn, t_fn, sigma_fn, FSAL, EPS_PRED = get_rk_methods_order_and_fn(rk_type)
    
    xi = [torch.zeros_like(x)] * (order+1)
    ki = [torch.zeros_like(x)] * order
    ki_tmp = [torch.zeros_like(x)] * order
    
    xi[0] = x
    h_prev = None
    denoised_buffer = []
    
    MULTISTEP = True
    FSAL = False
    
    for _ in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[_], sigmas[_+1]
        
        h_orig = t_fn(sigma_next)-t_fn(sigma)
        sigma_up, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, eta, eta_var, noise_mode, h_orig)
        t_down, t = t_fn(sigma_down), t_fn(sigma)
        h = t_down - t
        
        ab, ci = get_rk_methods_coeff(rk_type, h, c2, c3, h_prev)
        
        for i in range(order):
                
            #else:
            #    ki[i] = model_call(model, xi[i], sigma_fn(t + h*ci[i]), **extra_args)
                
            """if MULTISTEP == True and h_prev != None:
                buf_len = len(denoised_buffer)
                for n in range(buf_len, 0):"""
            #denoised = (1 - cfgpp[i]) * denoised + cfgpp[i] * temp[0]
            
            if FSAL == True and _ > 0 and i == 0:
                ki[0] = ki[order-1]
                ki[0] = denoised_last
            elif MULTISTEP == True and i < len(denoised_buffer):
                ki[i] = denoised_buffer[i]
                ki[i] = denoised_buffer.pop()
                #buf_len = len(denoised_buffer)
                #if i < buf_len:
                #    ki[i] = denoised_buffer[i]
            else:
                ki[i] = model_call(model, xi[i], sigma_fn(t + h*ci[i]), **extra_args)
                if cfgpp != 0.0:
                    ki[i] = temp[0] + cfgpp * (ki[i] - temp[0])
                ki_tmp[i] = temp[0]
            print(i, len(denoised_buffer))
                
            ks = torch.zeros_like(x)
            ks_tmp = torch.zeros_like(x)
            ab_sum=0
            for j in range(order):
                ks += ab[i][j] * ki[j]
                ks_tmp += ab[i][j] * ki_tmp[j]
                ab_sum += ab[i][j]
            if EPS_PRED == True:
                denoised = alpha_fn(-h*ci[i+1]) * xi[0] - sigma * ks
                dnl = ks / ab_sum
            else:
                denoised = ks / ab_sum
                dnl = denoised
            denoised_last = denoised
            denoised_buffer.append(dnl)
            if len(denoised_buffer) > BUF_ELEM:
                denoised_buffer = denoised_buffer[1:]
            #denoised_last = dnl
            #xi[(i+1)%order] = alpha_fn(-h*ci[i+1]) * xi[0] + h*ks

            xi[(i+1)%order] = alpha_fn(-h*ci[i+1]) * (xi[0] + cfgpp*h*(ks - ks_tmp)) + h*ks            

        denoised_last = denoised
        denoised_buffer.append(denoised)
        if len(denoised_buffer) > BUF_ELEM:
            denoised_buffer = denoised_buffer[1:]
            
        if callback is not None:
            callback({'x': xi[0], 'i': _, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised})
            
        noise = noise_sampler(sigma=sigma, sigma_next=sigma_next)
        noise = (noise - noise.mean()) / noise.std()
        xi[0] = alpha_ratio * xi[0] + noise * s_noise * sigma_up
            
        h_prev = h_orig
        
        gc.collect()
        torch.cuda.empty_cache()
        
    return xi[0]

