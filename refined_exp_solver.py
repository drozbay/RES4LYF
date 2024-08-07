import torch
from torch import no_grad, FloatTensor
from tqdm import tqdm
from itertools import pairwise
from typing import Protocol, Optional, Dict, Any, TypedDict, NamedTuple
import math

from .noise_classes import *

from comfy.k_diffusion.sampling import to_d
import comfy.model_patcher

import sys


class DenoiserModel(Protocol):
  def __call__(self, x: FloatTensor, t: FloatTensor, *args, **kwargs) -> FloatTensor: ...

class RefinedExpCallbackPayload(TypedDict):
  x: FloatTensor
  i: int
  sigma: FloatTensor
  sigma_hat: FloatTensor
  

class RefinedExpCallback(Protocol):
  def __call__(self, payload: RefinedExpCallbackPayload) -> None: ...

class NoiseSampler(Protocol):
  def __call__(self, x: FloatTensor) -> FloatTensor: ...

class StepOutput(NamedTuple):
  x_next: FloatTensor
  denoised: FloatTensor
  denoised2: FloatTensor
  vel: FloatTensor
  vel_2: FloatTensor

def _gamma(
  n: int,
) -> int:
  """
  https://en.wikipedia.org/wiki/Gamma_function
  for every positive integer n,
  Γ(n) = (n-1)!
  """
  return math.factorial(n-1)

def _incomplete_gamma(
  s: int,
  x: float,
  gamma_s: Optional[int] = None
) -> float:
  """
  https://en.wikipedia.org/wiki/Incomplete_gamma_function#Special_values
  if s is a positive integer,
  Γ(s, x) = (s-1)!*∑{k=0..s-1}(x^k/k!)
  """
  if gamma_s is None:
    gamma_s = _gamma(s)

  sum_: float = 0
  # {k=0..s-1} inclusive
  for k in range(s):
    numerator: float = x**k
    denom: int = math.factorial(k)
    quotient: float = numerator/denom
    sum_ += quotient
  incomplete_gamma_: float = sum_ * math.exp(-x) * gamma_s
  return incomplete_gamma_

# by Katherine Crowson
def _phi_1(neg_h: FloatTensor):
  return torch.nan_to_num(torch.expm1(neg_h) / neg_h, nan=1.0)

# by Katherine Crowson
def _phi_2(neg_h: FloatTensor):
  return torch.nan_to_num((torch.expm1(neg_h) - neg_h) / neg_h**2, nan=0.5)

# by Katherine Crowson
def _phi_3(neg_h: FloatTensor):
  return torch.nan_to_num((torch.expm1(neg_h) - neg_h - neg_h**2 / 2) / neg_h**3, nan=1 / 6)

def _phi(
  neg_h: float,
  j: int,
):
  """
  For j={1,2,3}: you could alternatively use Kat's phi_1, phi_2, phi_3 which perform fewer steps

  Lemma 1
  https://arxiv.org/abs/2308.02157
  ϕj(-h) = 1/h^j*∫{0..h}(e^(τ-h)*(τ^(j-1))/((j-1)!)dτ)

  https://www.wolframalpha.com/input?i=integrate+e%5E%28%CF%84-h%29*%28%CF%84%5E%28j-1%29%2F%28j-1%29%21%29d%CF%84
  = 1/h^j*[(e^(-h)*(-τ)^(-j)*τ(j))/((j-1)!)]{0..h}
  https://www.wolframalpha.com/input?i=integrate+e%5E%28%CF%84-h%29*%28%CF%84%5E%28j-1%29%2F%28j-1%29%21%29d%CF%84+between+0+and+h
  = 1/h^j*((e^(-h)*(-h)^(-j)*h^j*(Γ(j)-Γ(j,-h)))/(j-1)!)
  = (e^(-h)*(-h)^(-j)*h^j*(Γ(j)-Γ(j,-h))/((j-1)!*h^j)
  = (e^(-h)*(-h)^(-j)*(Γ(j)-Γ(j,-h))/(j-1)!
  = (e^(-h)*(-h)^(-j)*(Γ(j)-Γ(j,-h))/Γ(j)
  = (e^(-h)*(-h)^(-j)*(1-Γ(j,-h)/Γ(j))

  requires j>0
  """
  assert j > 0
  gamma_: float = _gamma(j)
  incomp_gamma_: float = _incomplete_gamma(j, neg_h, gamma_s=gamma_)

  phi_: float = math.exp(neg_h) * neg_h**-j * (1-incomp_gamma_/gamma_)

  return phi_

class RESDECoeffsSecondOrder(NamedTuple):
  a2_1: float
  b1: float
  b2: float

def _de_second_order(
  h: float,
  c2: float,
  simple_phi_calc = False,
) -> RESDECoeffsSecondOrder:
  """
  Table 3
  https://arxiv.org/abs/2308.02157
  ϕi,j := ϕi,j(-h) = ϕi(-cj*h)
  a2_1 = c2ϕ1,2
       = c2ϕ1(-c2*h)
  b1 = ϕ1 - ϕ2/c2
  """
  if simple_phi_calc:
    # Kat computed simpler expressions for phi for cases j={1,2,3}
    a2_1: float = c2 * _phi_1(-c2*h)
    phi1: float = _phi_1(-h)
    phi2: float = _phi_2(-h)
  else:
    # I computed general solution instead.
    # they're close, but there are slight differences. not sure which would be more prone to numerical error.
    a2_1: float = c2 * _phi(j=1, neg_h=-c2*h)
    phi1: float = _phi(j=1, neg_h=-h)
    phi2: float = _phi(j=2, neg_h=-h)
  phi2_c2: float = phi2/c2
  b1: float = phi1 - phi2_c2
  b2: float = phi2_c2
  return RESDECoeffsSecondOrder(
    a2_1=a2_1,
    b1=b1,
    b2=b2,
  )  

def _refined_exp_sosu_step(
  model,
  x,
  sigma,
  sigma_next,
  c2 = 0.5,
  extra_args: Dict[str, Any] = {},
  pbar: Optional[tqdm] = None,
  simple_phi_calc = False,
  momentum = 0.0,
  vel = None,
  vel_2 = None,
  time = None,
  eulers_mom = 0.0,
  cfgpp = 0.0,
) -> StepOutput:

  """Algorithm 1 "RES Second order Single Update Step with c2"
  https://arxiv.org/abs/2308.02157

  Parameters:
    model (`DenoiserModel`): a k-diffusion wrapped denoiser model (e.g. a subclass of DiscreteEpsDDPMDenoiser)
    x (`FloatTensor`): noised latents (or RGB I suppose), e.g. torch.randn((B, C, H, W)) * sigma[0]
    sigma (`FloatTensor`): timestep to denoise
    sigma_next (`FloatTensor`): timestep+1 to denoise
    c2 (`float`, *optional*, defaults to .5): partial step size for solving ODE. .5 = midpoint method
    extra_args (`Dict[str, Any]`, *optional*, defaults to `{}`): kwargs to pass to `model#__call__()`
    pbar (`tqdm`, *optional*, defaults to `None`): progress bar to update after each model call
    simple_phi_calc (`bool`, *optional*, defaults to `True`): True = calculate phi_i,j(-h) via simplified formulae specific to j={1,2}. False = Use general solution that works for any j. Mathematically equivalent, but could be numeric differences."""
  if cfgpp != 0.0:
    temp = [0]
    def post_cfg_function(args):
        temp[0] = args["uncond_denoised"]
        return args["denoised"]

    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = comfy.model_patcher.set_model_options_post_cfg_function(model_options, post_cfg_function, disable_cfg1_optimization=True)

  def momentum_func(diff, velocity, timescale=1.0, offset=-momentum / 2.0): # Diff is current diff, vel is previous diff
    if velocity is None:
        momentum_vel = diff
    else:
        momentum_vel = momentum * (timescale + offset) * velocity + (1 - momentum * (timescale + offset)) * diff
    return momentum_vel

  lam_next, lam = (s.log().neg() for s in (sigma_next, sigma))

  s_in = x.new_ones([x.shape[0]])
  h = lam_next - lam
  a2_1, b1, b2 = _de_second_order(h=h, c2=c2, simple_phi_calc=simple_phi_calc)
  
  denoised = model(x, sigma * s_in, **extra_args)
  
  if pbar is not None:
    pbar.update(0.5)

  c2_h = c2*h

  diff_2 = momentum_func(a2_1*h*denoised, vel_2, time)

  vel_2 = diff_2
  x_2 = math.exp(-c2_h)*x + diff_2
  if cfgpp == False:
    x_2 = math.exp(-c2_h)*x + diff_2
  else:
    x_2 = math.exp(-c2_h) * (x + cfgpp*denoised - cfgpp*temp[0]) + diff_2
  lam_2 = lam + c2_h
  sigma_2 = lam_2.neg().exp()

  denoised2 = model(x_2, sigma_2 * s_in, **extra_args)

  if pbar is not None:
    pbar.update(0.5)

  diff = momentum_func(h*(b1*denoised + b2*denoised2), vel, time)

  vel = diff

  if cfgpp == False:
    x_next = math.exp(-h)*x + diff
  else:
    x_next = math.exp(-h) * (x + cfgpp*denoised - cfgpp*temp[0]) + diff

  return StepOutput(
    x_next=x_next,
    denoised=denoised,
    denoised2=denoised2,
    vel=vel,
    vel_2=vel_2,
  )


#@cast_fp64
@no_grad()
def sample_refined_exp_s_advanced(
  model,
  x,
  sigmas,
  branch_mode,
  branch_depth,
  branch_width,
  guide_1=None,
  guide_2=None,
  guide_mode_1 = 0,
  guide_mode_2 = 0,
  guide_1_channels=None,
  denoise_to_zero: bool = True,
  extra_args: Dict[str, Any] = {},
  callback: Optional[RefinedExpCallback] = None,
  disable: Optional[bool] = None,
  eta=None,
  momentum=None,
  eulers_mom=None,
  c2=None,
  cfgpp=None,
  offset=None,
  alpha=None,
  latent_guide_1=None,
  latent_guide_2=None,
  noise_sampler: NoiseSampler = torch.randn_like,
  noise_sampler_type=None,
  simple_phi_calc = False,
  k=1.0,
  clownseed=0,
  latent_noise=None,
  latent_self_guide_1=False,
  latent_shift_guide_1=False,
): 
  """
  
  Refined Exponential Solver (S).
  Algorithm 2 "RES Single-Step Sampler" with Algorithm 1 second-order step
  https://arxiv.org/abs/2308.02157

  Parameters:
    model (`DenoiserModel`): a k-diffusion wrapped denoiser model (e.g. a subclass of DiscreteEpsDDPMDenoiser)
    x (`FloatTensor`): noised latents (or RGB I suppose), e.g. torch.randn((B, C, H, W)) * sigma[0]
    sigmas (`FloatTensor`): sigmas (ideally an exponential schedule!) e.g. get_sigmas_exponential(n=25, sigma_min=model.sigma_min, sigma_max=model.sigma_max)
    denoise_to_zero (`bool`, *optional*, defaults to `True`): whether to finish with a first-order step down to 0 (rather than stopping at sigma_min). True = fully denoise image. False = match Algorithm 2 in paper
    extra_args (`Dict[str, Any]`, *optional*, defaults to `{}`): kwargs to pass to `model#__call__()`
    callback (`RefinedExpCallback`, *optional*, defaults to `None`): you can supply this callback to see the intermediate denoising results, e.g. to preview each step of the denoising process
    disable (`bool`, *optional*, defaults to `False`): whether to hide `tqdm`'s progress bar animation from being printed
    eta (`FloatTensor`, *optional*, defaults to 0.): degree of stochasticity, η, for each timestep. tensor shape must be broadcastable to 1-dimensional tensor with length `len(sigmas) if denoise_to_zero else len(sigmas)-1`. each element should be from 0 to 1.
    c2 (`float`, *optional*, defaults to .5): partial step size for solving ODE. .5 = midpoint method
    noise_sampler (`NoiseSampler`, *optional*, defaults to `torch.randn_like`): method used for adding noise
    simple_phi_calc (`bool`, *optional*, defaults to `True`): True = calculate phi_i,j(-h) via simplified formulae specific to j={1,2}. False = Use general solution that works for any j. Mathematically equivalent, but could be numeric differences.
  """

  s_in = x.new_ones([x.shape[0]])

  #assert sigmas[-1] == 0
  sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()

  noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=clownseed, sigma_min=sigma_min, sigma_max=sigma_max)

  b, c, h, w = x.shape

  dt = None
  vel, vel_2 = None, None
  x_hat = None
  
  x_n   = [[None for _ in range(branch_width ** depth)] for depth in range(branch_depth + 1)]
  x_h   = [[None for _ in range(branch_width ** depth)] for depth in range(branch_depth + 1)]
  vel   = [[None for _ in range(branch_width ** depth)] for depth in range(branch_depth + 1)]
  vel_2 = [[None for _ in range(branch_width ** depth)] for depth in range(branch_depth + 1)]
  denoised   = [[None for _ in range(branch_width ** depth)] for depth in range(branch_depth + 1)]
  denoised2 = [[None for _ in range(branch_width ** depth)] for depth in range(branch_depth + 1)]
  denoised_ = None
  denoised2_ = None
  denoised2_prev = None
  
  i=0
  with tqdm(disable=disable, total=len(sigmas)-(1 if denoise_to_zero else 2)) as pbar:
    #for i, (sigma, sigma_next) in enumerate(pairwise(sigmas[:-1].split(1))):
    while i < len(sigmas) - 1 and sigmas[i+1] > 0.0:

      sigma = sigmas[i]
      sigma_next = sigmas[i+1]
      time = sigmas[i] / sigma_max

      if 'sigma' not in locals():
        sigma = sigmas[i]

      if latent_noise is not None:
        if latent_noise.size()[0] == 1:
          eps = latent_noise[0]
        else:
          eps = latent_noise[i]
      else:
        if noise_sampler_type == "fractal":
          noise_sampler.alpha = alpha[i]
          noise_sampler.k = k

      sigma_hat = sigma * (1 + eta[i])

      x_n[0][0] = x
      for depth in range(1, branch_depth+1):
        sigma = sigmas[i]
        sigma_next = sigmas[i+1]
        sigma_hat = sigma * (1 + eta[i])
        
        for m in range(branch_width**(depth-1)):
          for n in range(branch_width):
            idx = m * branch_width + n
            x_h[depth][idx] = x_n[depth-1][m] + (sigma_hat ** 2 - sigma ** 2).sqrt() * noise_sampler(sigma=sigma, sigma_next=sigma_next)   
            x_n[depth][idx], denoised[depth][idx], denoised2[depth][idx], vel[depth][idx], vel_2[depth][idx] = _refined_exp_sosu_step(model, x_h[depth][idx], sigma_hat, sigma_next, c2=c2[i],
                                                                          extra_args=extra_args, pbar=pbar, simple_phi_calc=simple_phi_calc,
                                                                          momentum = momentum[i], vel = vel[depth][idx], vel_2 = vel_2[depth][idx], time = time, eulers_mom = eulers_mom[i].item(), cfgpp = cfgpp[i].item()
                                                                          )
            denoised_ = denoised[depth][idx]
            denoised2_ = denoised2[depth][idx]
        i += 1
        
      if denoised2_prev is not None:
        x_n[0][0] = denoised2_prev
      x_next, x_hat, denoised2_prev = branch_mode_proc(x_n, x_h, denoised2, latent_guide_2, branch_mode, branch_depth, branch_width)
      
      d = to_d(x_hat, sigma_hat, x_next)
      dt = sigma_next - sigma_hat
      x_next = x_next + eulers_mom[i].item() * d * dt
      
      if callback is not None:
        payload = RefinedExpCallbackPayload(x=x, i=i, sigma=sigma, sigma_hat=sigma_hat, denoised=denoised_, denoised2=denoised2_prev,)         # added updated denoised2_prev that's selected from the same slot as x_next                      
        callback(payload)

      x = x_next - sigma_next*offset[i]
      
      x = guide_mode_proc(x, i, guide_mode_1, guide_mode_2, sigma_next, guide_1, guide_2,  latent_guide_1, latent_guide_2, guide_1_channels)
      
    if denoise_to_zero:
      final_eta = eta[-1]
      eps = noise_sampler(sigma=sigma, sigma_next=sigma_next).double()
      sigma_hat = sigma * (1 + final_eta)
      x_hat = x + (sigma_hat ** 2 - sigma ** 2) ** .5 * eps
      
      s_in = x.new_ones([x.shape[0]])
      x_next = model(x_hat, torch.zeros_like(sigma).to(x_hat.device) * s_in, **extra_args)
      pbar.update()
      x = x_next

  return x


from torch.nn.functional import cosine_similarity

@no_grad()
def branch_mode_proc(
  x_n, x_h,
  denoised2,
  latent,
  branch_mode,
  branch_depth,
  branch_width,

):
  if branch_mode == 'cos_reversal':
    x_next, x_hat, d_next = select_trajectory_with_reversal(x_n, x_h, branch_depth)
  if branch_mode == 'cos_similarity':
    x_next, x_hat, d_next = select_trajectory_based_on_cosine_similarity(x_n, x_h, branch_depth, branch_width)
  if branch_mode == 'cos_similarity_d':
    x_next, x_hat, d_next = select_trajectory_based_on_cosine_similarity_d(x_n, x_h, denoised2, branch_depth, branch_width)
  if branch_mode == 'cos_linearity':
    x_next, x_hat, d_next = select_most_linear_trajectory(x_n, x_h, branch_depth, branch_width) 
  if branch_mode == 'cos_linearity_d':
    x_next, x_hat, d_next = select_most_linear_trajectory_d(x_n, x_h, denoised2, branch_depth, branch_width) 
  if branch_mode == 'cos_perpendicular':
    x_next, x_hat, d_next = select_perpendicular_cosine_trajectory(x_n, x_h, branch_depth, branch_width) 
  if branch_mode == 'cos_perpendicular_d':
    x_next, x_hat, d_next = select_perpendicular_cosine_trajectory_d(x_n, x_h, denoised2, branch_depth, branch_width) 
    
  if branch_mode == 'latent_match':
    distances = [torch.norm(tensor - latent).item() for tensor in x_n[branch_depth]]
    closest_index = distances.index(min(distances))
    x_next = x_n[branch_depth][closest_index]
    x_hat = x_h[branch_depth][closest_index]
    d_next = denoised2[branch_depth][closest_index]
    
  if branch_mode == 'latent_match_d':
    distances = [torch.norm(tensor - latent).item() for tensor in denoised2[branch_depth]]
    closest_index = distances.index(min(distances))
    x_next = x_n[branch_depth][closest_index]
    x_hat = x_h[branch_depth][closest_index]
    d_next = denoised2[branch_depth][closest_index]
    
  if branch_mode == 'latent_match_sdxl_color_d':
      relevant_latent = latent[:, 1:3, :, :] 
      denoised2_relevant = [tensor[:, 1:3, :, :] for tensor in denoised2[branch_depth]]

      distances = [torch.norm(tensor - relevant_latent).item() for tensor in denoised2_relevant]
      closest_index = distances.index(min(distances))
      
      x_next = x_n[branch_depth][closest_index]
      x_hat = x_h[branch_depth][closest_index]
      d_next = denoised2[branch_depth][closest_index]
      
  if branch_mode == 'latent_match_sdxl_luminosity_d':
      relevant_latent = latent[:, 0:1, :, :] 
      denoised2_relevant = [tensor[:, 0:1, :, :] for tensor in denoised2[branch_depth]]

      distances = [torch.norm(tensor - relevant_latent).item() for tensor in denoised2_relevant]
      closest_index = distances.index(min(distances))
      
      x_next = x_n[branch_depth][closest_index]
      x_hat = x_h[branch_depth][closest_index]
      d_next = denoised2[branch_depth][closest_index]
      
  if branch_mode == 'latent_match_sdxl_pattern_d':
      relevant_latent = latent[:, 3:4, :, :] 
      denoised2_relevant = [tensor[:, 3:4, :, :] for tensor in denoised2[branch_depth]]

      distances = [torch.norm(tensor - relevant_latent).item() for tensor in denoised2_relevant]
      closest_index = distances.index(min(distances))
      
      x_next = x_n[branch_depth][closest_index]
      x_hat = x_h[branch_depth][closest_index]
      d_next = denoised2[branch_depth][closest_index]

    
  if branch_mode == 'mean':
    x_mean = torch.mean(torch.stack(x_n[branch_depth]), dim=0)
    distances = [torch.norm(tensor - x_mean).item() for tensor in x_n[branch_depth]]
    closest_index = distances.index(min(distances))
    x_next = x_n[branch_depth][closest_index]
    x_hat = x_h[branch_depth][closest_index]
    d_next = denoised2[branch_depth][closest_index]
    
  if branch_mode == 'mean_d':
    d_mean = torch.mean(torch.stack(denoised2[branch_depth]), dim=0)
    distances = [torch.norm(tensor - d_mean).item() for tensor in denoised2[branch_depth]]
    closest_index = distances.index(min(distances))
    x_next = x_n[branch_depth][closest_index]
    x_hat = x_h[branch_depth][closest_index]
    d_next = denoised2[branch_depth][closest_index]
    
  if branch_mode == 'median': #minimum median distance
    d_n_3 = [tensor for tensor in denoised2[branch_depth] if tensor is not None]
    x_n_3 = [tensor for tensor in x_n[branch_depth] if tensor is not None]
    x_h_3 = [tensor for tensor in x_h[branch_depth] if tensor is not None]
    num_tensors = len(x_n_3)
    distance_matrix = torch.zeros(num_tensors, num_tensors)

    for m in range(num_tensors):
        for n in range(num_tensors):
            if m != n:
                distance_matrix[m, n] = torch.norm(x_n_3[m] - x_n_3[n])
    median_distances = torch.median(distance_matrix, dim=1).values
    min_median_distance_index = torch.argmin(median_distances).item()
    x_next = x_n_3[min_median_distance_index]
    x_hat = x_h_3[min_median_distance_index]
    d_next = d_n_3[min_median_distance_index]
    
  if branch_mode == 'median_d': #minimum median distance
    d_n_3 = [tensor for tensor in denoised2[branch_depth] if tensor is not None]
    x_n_3 = [tensor for tensor in x_n[branch_depth] if tensor is not None]
    x_h_3 = [tensor for tensor in x_h[branch_depth] if tensor is not None]
    num_tensors = len(x_n_3)
    distance_matrix = torch.zeros(num_tensors, num_tensors)

    for m in range(num_tensors):
        for n in range(num_tensors):
            if m != n:
                distance_matrix[m, n] = torch.norm(d_n_3[m] - d_n_3[n])
    median_distances = torch.median(distance_matrix, dim=1).values
    min_median_distance_index = torch.argmin(median_distances).item()
    
    x_next = x_n_3[min_median_distance_index]
    x_hat = x_h_3[min_median_distance_index]
    d_next = d_n_3[min_median_distance_index]
    
  if branch_mode == 'zmean_d':
    d_mean = torch.mean(torch.stack(denoised2[branch_depth]), dim=0)
    distances = [torch.norm(tensor - d_mean).item() for tensor in denoised2[branch_depth]]
    closest_index = distances.index(max(distances))
    x_next = x_n[branch_depth][closest_index]
    x_hat = x_h[branch_depth][closest_index]
    d_next = denoised2[branch_depth][closest_index]
    
  if branch_mode == 'zmedian_d': #minimum median distance
    d_n_3 = [tensor for tensor in denoised2[branch_depth] if tensor is not None]
    x_n_3 = [tensor for tensor in x_n[branch_depth] if tensor is not None]
    x_h_3 = [tensor for tensor in x_h[branch_depth] if tensor is not None]
    num_tensors = len(x_n_3)
    distance_matrix = torch.zeros(num_tensors, num_tensors)

    for m in range(num_tensors):
        for n in range(num_tensors):
            if m != n:
                distance_matrix[m, n] = torch.norm(d_n_3[m] - d_n_3[n])
    median_distances = torch.median(distance_matrix, dim=1).values
    min_median_distance_index = torch.argmax(median_distances).item()
    
    x_next = x_n_3[min_median_distance_index]
    x_hat = x_h_3[min_median_distance_index]
    d_next = d_n_3[min_median_distance_index]    
    
  if branch_mode == 'gradient_max_full_d': # greatest gradient descent
    start_point = x_n[0][0]
    norms = [torch.norm(tensor - start_point).item() for tensor in denoised2[branch_depth] if tensor is not None]
    
    greatest_norm_index = norms.index(max(norms))
    
    x_next = x_n[branch_depth][greatest_norm_index]
    x_hat = x_h[branch_depth][greatest_norm_index]
    d_next = denoised2[branch_depth][greatest_norm_index]
    
  if branch_mode == 'gradient_min_full_d': # greatest gradient descent
    start_point = x_n[0][0]
    norms = [torch.norm(tensor - start_point).item() for tensor in denoised2[branch_depth] if tensor is not None]
    
    greatest_norm_index = norms.index(min(norms))
    
    x_next = x_n[branch_depth][greatest_norm_index]
    x_hat = x_h[branch_depth][greatest_norm_index]
    d_next = denoised2[branch_depth][greatest_norm_index]

  if branch_mode == 'gradient_max_full': # greatest gradient descent
    start_point = x_n[0][0]
    norms = [torch.norm(tensor - start_point).item() for tensor in x_n[branch_depth] if tensor is not None]
    
    greatest_norm_index = norms.index(max(norms))
    
    x_next = x_n[branch_depth][greatest_norm_index]
    x_hat = x_h[branch_depth][greatest_norm_index]
    d_next = denoised2[branch_depth][greatest_norm_index]
    
  if branch_mode == 'gradient_min_full': # greatest gradient descent
    start_point = x_n[0][0]
    norms = [torch.norm(tensor - start_point).item() for tensor in x_n[branch_depth] if tensor is not None]
    
    greatest_norm_index = norms.index(min(norms))
    
    x_next = x_n[branch_depth][greatest_norm_index]
    x_hat = x_h[branch_depth][greatest_norm_index]
    d_next = denoised2[branch_depth][greatest_norm_index]

  if branch_mode == 'gradient_max': #greatest gradient descent
    norms = [torch.norm(tensor).item() for tensor in x_n[branch_depth] if tensor is not None]
    greatest_norm_index = norms.index(max(norms))
    x_next = x_n[branch_depth][greatest_norm_index]
    x_hat  = x_h[branch_depth][greatest_norm_index]
    d_next = denoised2[branch_depth][greatest_norm_index]
    
  if branch_mode == 'gradient_min': #greatest gradient descent
    norms = [torch.norm(tensor).item() for tensor in x_n[branch_depth] if tensor is not None]
    min_norm_index = norms.index(min(norms))
    x_next = x_n[branch_depth][min_norm_index]
    x_hat  = x_h[branch_depth][min_norm_index]
    d_next = denoised2[branch_depth][min_norm_index]
    
  if branch_mode == 'gradient_max_d': #greatest gradient descent
    norms = [torch.norm(tensor).item() for tensor in denoised2[branch_depth] if tensor is not None]
    greatest_norm_index = norms.index(max(norms))
    x_next = x_n[branch_depth][greatest_norm_index]
    x_hat  = x_h[branch_depth][greatest_norm_index]
    d_next = denoised2[branch_depth][greatest_norm_index]
    
  if branch_mode == 'gradient_min_d': #greatest gradient descent
    norms = [torch.norm(tensor).item() for tensor in denoised2[branch_depth] if tensor is not None]
    min_norm_index = norms.index(min(norms))
    x_next = x_n[branch_depth][min_norm_index]
    x_hat  = x_h[branch_depth][min_norm_index]
    d_next = denoised2[branch_depth][min_norm_index]
    
  return x_next, x_hat, d_next
    
def select_trajectory_with_reversal(x_n, x_h, denoised2, branch_depth):
    x_n_depth = [tensor for tensor in x_n[branch_depth] if tensor is not None]
    x_h_depth = [tensor for tensor in x_h[branch_depth] if tensor is not None]
    d_n_depth = [tensor for tensor in denoised2[branch_depth] if tensor is not None]
    num_tensors = len(x_n_depth)

    negative_cos_sim_indices = []
    cos_sim_values = []

    for i in range(num_tensors):
        trajectory_cos_sims = []
        for j in range(1, len(x_n_depth[i]) - 1): 
            cos_sim = cosine_similarity(x_n_depth[i][j].unsqueeze(0), x_n_depth[i][j + 1].unsqueeze(0))
            trajectory_cos_sims.append(cos_sim.item())
        # check for reversal (negative cosine similarity)
        if any(cos_sim < 0 for cos_sim in trajectory_cos_sims):
            negative_cos_sim_indices.append(i)
            cos_sim_values.append(min(trajectory_cos_sims))

    if not negative_cos_sim_indices:
        # no reversal? fall back to the first available trajectory
        selected_index = 0
    else:
        # choose trajectory with most negative cosine similarity
        selected_index = negative_cos_sim_indices[torch.argmin(torch.tensor(cos_sim_values)).item()]

    x_next = x_n_depth[selected_index]
    x_hat = x_h_depth[selected_index]
    d_next = d_n_depth[selected_index]

    return x_next, x_hat, d_next

def select_trajectory_based_on_cosine_similarity(x_n, x_h, denoised2, branch_depth, branch_width):
    d_n_depth = [tensor for tensor in denoised2[branch_depth] if tensor is not None]
    x_n_depth = [tensor for tensor in x_n[branch_depth] if tensor is not None]
    x_h_depth = [tensor for tensor in x_h[branch_depth] if tensor is not None]

    max_cosine_similarity = float('-inf')
    best_idx = -1

    for n in range(len(x_n_depth)):
        direction_vector = x_n_depth[n] - x_n[0][0]
        total_cosine_similarity = 0.0

        for depth in range(1, branch_depth):
            for j in range(len(x_n[depth])):
                x1_direction = x_n[depth][j] - x_n[depth - 1][j // branch_width]
                x1_to_x3_direction = x_n_depth[n] - x_n[depth][j]
                cosine_similarity = torch.dot(x1_direction.flatten(), x1_to_x3_direction.flatten()) / (torch.norm(x1_direction) * torch.norm(x1_to_x3_direction))
                total_cosine_similarity += cosine_similarity

        if total_cosine_similarity > max_cosine_similarity:
            max_cosine_similarity = total_cosine_similarity
            best_idx = n

    x_next = x_n_depth[best_idx]
    x_hat = x_h_depth[best_idx]
    d_next = d_n_depth[best_idx]

    return x_next, x_hat, d_next

  
def select_trajectory_based_on_cosine_similarity_d(x_n, x_h, denoised2, branch_depth, branch_width):
    d_n_depth = [tensor for tensor in denoised2[branch_depth] if tensor is not None]
    x_n_depth = [tensor for tensor in x_n[branch_depth] if tensor is not None]
    x_h_depth = [tensor for tensor in x_h[branch_depth] if tensor is not None]
    
    denoised2[0][0] = x_n[0][0]

    max_cosine_similarity = float('-inf')
    best_idx = -1

    for n in range(len(d_n_depth)):
        direction_vector = d_n_depth[n] - x_n[0][0]
        total_cosine_similarity = 0.0

        for depth in range(1, branch_depth):
            for j in range(len(denoised2[depth])):
                x1_direction = denoised2[depth][j] - denoised2[depth - 1][j // branch_width]
                x1_to_x3_direction = d_n_depth[n] - denoised2[depth][j]
                cosine_similarity = torch.dot(x1_direction.flatten(), x1_to_x3_direction.flatten()) / (torch.norm(x1_direction) * torch.norm(x1_to_x3_direction))
                total_cosine_similarity += cosine_similarity

        if total_cosine_similarity > max_cosine_similarity:
            max_cosine_similarity = total_cosine_similarity
            best_idx = n

    x_next = x_n_depth[best_idx]
    x_hat = x_h_depth[best_idx]
    d_next = d_n_depth[best_idx]

    return x_next, x_hat, d_next

  

def select_most_linear_trajectory(x_n, x_h, denoised2, branch_depth, branch_width):
    d_n_depth = [tensor for tensor in denoised2[branch_depth] if tensor is not None]
    x_n_depth = [tensor for tensor in x_n[branch_depth] if tensor is not None]
    x_h_depth = [tensor for tensor in x_h[branch_depth] if tensor is not None]

    max_cosine_similarity_sum = float('-inf')
    best_idx = -1

    base_vector = x_n[0][0]

    # sum up  absolute cosine similarities for each trajectory
    for n in range(len(x_n_depth)):
        total_cosine_similarity = 0.0
        current_vector = x_n_depth[n]

        #cormpare trajectory's endpoint vs all intermediate steps
        for depth in range(1, branch_depth + 1):
            for j in range(len(x_n[depth])):
                if depth == 1:
                    previous_vector = base_vector
                else:
                    previous_vector = x_n[depth - 1][j // branch_width]

                direction_vector = x_n[depth][j] - previous_vector
                cosine_similarity = torch.dot(direction_vector.flatten(), (current_vector - previous_vector).flatten()) / (
                    torch.norm(direction_vector) * torch.norm(current_vector - previous_vector))

                total_cosine_similarity += torch.abs(cosine_similarity) #abs val is key here... allows reversals (180 degree swap in direction, i.e., convergence)

        if total_cosine_similarity > max_cosine_similarity_sum:
            max_cosine_similarity_sum = total_cosine_similarity
            best_idx = n

    x_next = x_n_depth[best_idx]
    x_hat = x_h_depth[best_idx]
    d_next = d_n_depth[best_idx]

    return x_next, x_hat, d_next

  
  
def select_most_linear_trajectory_d(x_n, x_h, denoised2, branch_depth, branch_width):
    d_n_depth = [tensor for tensor in denoised2[branch_depth] if tensor is not None]
    x_n_depth = [tensor for tensor in x_n[branch_depth] if tensor is not None]
    x_h_depth = [tensor for tensor in x_h[branch_depth] if tensor is not None]

    max_cosine_similarity_sum = float('-inf')
    best_idx = -1

    base_vector = x_n[0][0]

    # sum up  absolute cosine similarities for each trajectory
    for n in range(len(d_n_depth)):
        total_cosine_similarity = 0.0
        current_vector = d_n_depth[n]

        #cormpare trajectory's endpoint vs all intermediate steps
        for depth in range(1, branch_depth + 1):
            for j in range(len(denoised2[depth])):
                if depth == 1:
                    previous_vector = base_vector
                else:
                    previous_vector = denoised2[depth - 1][j // branch_width]

                direction_vector = denoised2[depth][j] - previous_vector
                cosine_similarity = torch.dot(direction_vector.flatten(), (current_vector - previous_vector).flatten()) / (
                    torch.norm(direction_vector) * torch.norm(current_vector - previous_vector))

                total_cosine_similarity += torch.abs(cosine_similarity) #abs val is key here... allows reversals (180 degree swap in direction, i.e., convergence)

        if total_cosine_similarity > max_cosine_similarity_sum:
            max_cosine_similarity_sum = total_cosine_similarity
            best_idx = n

    x_next = x_n_depth[best_idx]
    x_hat = x_h_depth[best_idx]
    d_next = d_n_depth[best_idx]

    return x_next, x_hat, d_next


def select_perpendicular_cosine_trajectory(x_n, x_h, denoised2, branch_depth, branch_width):
    d_n_depth = [tensor for tensor in denoised2[branch_depth] if tensor is not None]
    x_n_depth = [tensor for tensor in x_n[branch_depth] if tensor is not None]
    x_h_depth = [tensor for tensor in x_h[branch_depth] if tensor is not None]

    min_cosine_deviation_from_zero = float('inf')
    best_idx = -1

    # Calculate cosine similarities aiming for orthogonality at each step
    for n in range(len(x_n_depth)):
        total_cosine_deviation = 0.0

        # Iterate through the trajectory path
        for depth in range(1, branch_depth):
            for j in range(len(x_n[depth])):
                if depth == 1:
                    previous_vector = x_n[0][0]
                else:
                    previous_vector = x_n[depth - 1][j // branch_width]

                current_vector = x_n[depth][j] - previous_vector
                target_vector = x_n_depth[n] - x_n[depth][j]
                
                cosine_similarity = torch.dot(current_vector.flatten(), target_vector.flatten()) / (
                    torch.norm(current_vector) * torch.norm(target_vector))

                # Accumulate deviation from zero (ideal for perpendicular direction)
                total_cosine_deviation += (cosine_similarity ** 2)  # Squaring to emphasize smaller values

        # Update to select the trajectory with the minimum deviation from zero cosine similarity (most perpendicular)
        if total_cosine_deviation < min_cosine_deviation_from_zero:
            min_cosine_deviation_from_zero = total_cosine_deviation
            best_idx = n

    x_next = x_n_depth[best_idx]
    x_hat = x_h_depth[best_idx]
    d_next = d_n_depth[best_idx]

    return x_next, x_hat, d_next

  
  
def select_perpendicular_cosine_trajectory_d(x_n, x_h, denoised2, branch_depth, branch_width):
  d_n_depth = [tensor for tensor in denoised2[branch_depth] if tensor is not None]
  x_n_depth = [tensor for tensor in x_n[branch_depth] if tensor is not None]
  x_h_depth = [tensor for tensor in x_h[branch_depth] if tensor is not None]

  min_cosine_deviation_from_zero = float('inf')
  best_idx = -1

  # Calculate cosine similarities aiming for orthogonality at each step
  for n in range(len(d_n_depth)):
      total_cosine_deviation = 0.0

      # Iterate through the trajectory path
      for depth in range(1, branch_depth):
          for j in range(len(denoised2[depth])):
              if depth == 1:
                  previous_vector = x_n[0][0] #did i do this right???
              else:
                  previous_vector = denoised2[depth - 1][j // branch_width]

              current_vector = denoised2[depth][j] - previous_vector
              target_vector = d_n_depth[n] - denoised2[depth][j]
              
              cosine_similarity = torch.dot(current_vector.flatten(), target_vector.flatten()) / (
                  torch.norm(current_vector) * torch.norm(target_vector))

              # Accumulate deviation from zero (ideal for perpendicular direction)
              total_cosine_deviation += (cosine_similarity ** 2)  # Squaring to emphasize smaller values

      # Update to select the trajectory with the minimum deviation from zero cosine similarity (most perpendicular)
      if total_cosine_deviation < min_cosine_deviation_from_zero:
          min_cosine_deviation_from_zero = total_cosine_deviation
          best_idx = n

  x_next = x_n_depth[best_idx]
  x_hat = x_h_depth[best_idx]
  d_next = d_n_depth[best_idx]

  return x_next, x_hat, d_next





def guide_mode_proc(x, i, guide_mode_1, guide_mode_2, sigma_next, guide_1, guide_2,  latent_guide_1, latent_guide_2, guide_1_channels):
  if latent_guide_1 is not None:
    latent_guide_crushed_1 = (latent_guide_1 - latent_guide_1.min()) / (latent_guide_1 - latent_guide_1.min()).max()
  if latent_guide_2 is not None:
    latent_guide_crushed_2 = (latent_guide_2 - latent_guide_2.min()) / (latent_guide_2 - latent_guide_2.min()).max()

  b, c, h, w = x.shape
  
  if latent_guide_1 is not None:
    if(guide_mode_1 == 1):
      x = x - sigma_next * guide_1[i] * latent_guide_1 * guide_1_channels.view(1,c,1,1)

    if(guide_mode_1 == 2):
      x = x - sigma_next * guide_1[i] * latent_guide_crushed_1 * guide_1_channels.view(1,c,1,1)

    if(guide_mode_1 == 3):
      x = (1 - guide_1[i]) * x * guide_1_channels.view(1,c,1,1) + (guide_1[i] * latent_guide_1 * guide_1_channels.view(1,c,1,1))

    if(guide_mode_1 == 4):
      x = (1 - guide_1[i]) * x * guide_1_channels.view(1,c,1,1) + (guide_1[i] * latent_guide_crushed_1 * guide_1_channels.view(1,c,1,1))   

    if(guide_mode_1 == 5):
      x = (x - guide_1[i] * sigma_next * x * guide_1_channels.view(1,c,1,1)) + (guide_1[i] * sigma_next * latent_guide_1 * guide_1_channels.view(1,c,1,1))
    if(guide_mode_1 == 6):
      x = (x - guide_1[i] * sigma_next * x * guide_1_channels.view(1,c,1,1)) + (guide_1[i] * sigma_next * latent_guide_crushed_1 * guide_1_channels.view(1,c,1,1))
    if(guide_mode_1 == 7):
      hard_light_blend_1 = hard_light_blend(x, latent_guide_1)
      x = (x - guide_1[i] * sigma_next * x * guide_1_channels.view(1,c,1,1)) + (guide_1[i] * sigma_next * hard_light_blend_1 * guide_1_channels.view(1,c,1,1))
    if(guide_mode_1 == 8):
      hard_light_blend_1 = hard_light_blend(latent_guide_1, x)
      x = (x - guide_1[i] * sigma_next * x * guide_1_channels.view(1,c,1,1)) + (guide_1[i] * sigma_next * hard_light_blend_1 * guide_1_channels.view(1,c,1,1))
    if(guide_mode_1 == 9):
      soft_light_blend_1 = soft_light_blend(x, latent_guide_1)
      x = (x - guide_1[i] * sigma_next * x * guide_1_channels.view(1,c,1,1)) + (guide_1[i] * sigma_next * soft_light_blend_1 * guide_1_channels.view(1,c,1,1))
    if(guide_mode_1 == 10):
      soft_light_blend_1 = soft_light_blend(latent_guide_1, x)
      x = (x - guide_1[i] * sigma_next * x * guide_1_channels.view(1,c,1,1)) + (guide_1[i] * sigma_next * soft_light_blend_1 * guide_1_channels.view(1,c,1,1))
    if(guide_mode_1 == 11):
      linear_light_blend_1 = linear_light_blend(x, latent_guide_1)
      x = (x - guide_1[i] * sigma_next * x * guide_1_channels.view(1,c,1,1)) + (guide_1[i] * sigma_next * linear_light_blend_1 * guide_1_channels.view(1,c,1,1))
    if(guide_mode_1 == 12):
      linear_light_blend_1 = linear_light_blend(latent_guide_1, x)
      x = (x - guide_1[i] * sigma_next * x * guide_1_channels.view(1,c,1,1)) + (guide_1[i] * sigma_next * linear_light_blend_1 * guide_1_channels.view(1,c,1,1))
    if(guide_mode_1 == 13):
      vivid_light_blend_1 = vivid_light_blend(x, latent_guide_1)
      x = (x - guide_1[i] * sigma_next * x * guide_1_channels.view(1,c,1,1)) + (guide_1[i] * sigma_next * vivid_light_blend_1 * guide_1_channels.view(1,c,1,1))
    if(guide_mode_1 == 14):
      vivid_light_blend_1 = vivid_light_blend(latent_guide_1, x)
      x = (x - guide_1[i] * sigma_next * x * guide_1_channels.view(1,c,1,1)) + (guide_1[i] * sigma_next * vivid_light_blend_1 * guide_1_channels.view(1,c,1,1))
    if(guide_mode_1 == 801):
      hard_light_blend_1 = bold_hard_light_blend(x, latent_guide_1)
      x = (x - guide_1[i] * sigma_next * x * guide_1_channels.view(1,c,1,1)) + (guide_1[i] * sigma_next * hard_light_blend_1 * guide_1_channels.view(1,c,1,1))
    if(guide_mode_1 == 802):
      hard_light_blend_1 = bold_hard_light_blend(latent_guide_1, x)
      x = (x - guide_1[i] * sigma_next * x * guide_1_channels.view(1,c,1,1)) + (guide_1[i] * sigma_next * hard_light_blend_1 * guide_1_channels.view(1,c,1,1))
    if(guide_mode_1 == 803):
      hard_light_blend_1 = fix_hard_light_blend(latent_guide_1, x)
      x = (x - guide_1[i] * sigma_next * x * guide_1_channels.view(1,c,1,1)) + (guide_1[i] * sigma_next * hard_light_blend_1 * guide_1_channels.view(1,c,1,1))
    if(guide_mode_1 == 804):
      hard_light_blend_1 = fix2_hard_light_blend(latent_guide_1, x)
      x = (x - guide_1[i] * sigma_next * x * guide_1_channels.view(1,c,1,1)) + (guide_1[i] * sigma_next * hard_light_blend_1 * guide_1_channels.view(1,c,1,1))
    if(guide_mode_1 == 805):
      hard_light_blend_1 = fix3_hard_light_blend(latent_guide_1, x)
      x = (x - guide_1[i] * sigma_next * x * guide_1_channels.view(1,c,1,1)) + (guide_1[i] * sigma_next * hard_light_blend_1 * guide_1_channels.view(1,c,1,1))
    if(guide_mode_1 == 806):
      hard_light_blend_1 = fix4_hard_light_blend(latent_guide_1, x)
      x = (x - guide_1[i] * sigma_next * x * guide_1_channels.view(1,c,1,1)) + (guide_1[i] * sigma_next * hard_light_blend_1 * guide_1_channels.view(1,c,1,1))
    if(guide_mode_1 == 807):
      hard_light_blend_1 = fix4_hard_light_blend(x, latent_guide_1)
      x = (x - guide_1[i] * sigma_next * x * guide_1_channels.view(1,c,1,1)) + (guide_1[i] * sigma_next * hard_light_blend_1 * guide_1_channels.view(1,c,1,1))

  if latent_guide_2 is not None:
    if(guide_mode_2 == 1):
      x = x - sigma_next * guide_2[i] * latent_guide_2
    if(guide_mode_2 == 2):
      x = x - sigma_next * guide_2[i] * latent_guide_crushed_2
    if(guide_mode_2 == 3):
      x = (1 - guide_2[i]) * x + (guide_2[i] * latent_guide_2)
    if(guide_mode_2 == 4):
      x = (1 - guide_2[i]) * x + (guide_2[i] * latent_guide_crushed_2)   
    if(guide_mode_2 == 5):
      x = (x - guide_2[i] * sigma_next * x) + (guide_2[i] * sigma_next * latent_guide_2)
    if(guide_mode_2 == 6):
      x = (x - guide_2[i] * sigma_next * x) + (guide_2[i] * sigma_next * latent_guide_crushed_2)   
    if(guide_mode_2 == 7):
      hard_light_blend_2 = hard_light_blend(x, latent_guide_2)
      x = (x - guide_2[i] * sigma_next * x) + (guide_2[i] * sigma_next * hard_light_blend_2)
    if(guide_mode_2 == 8):
      hard_light_blend_2 = hard_light_blend(latent_guide_2, x)
      x = (x - guide_2[i] * sigma_next * x) + (guide_2[i] * sigma_next * hard_light_blend_2)
    if(guide_mode_2 == 9):
      soft_light_blend_2 = soft_light_blend(x, latent_guide_2)
      x = (x - guide_2[i] * sigma_next * x) + (guide_2[i] * sigma_next * soft_light_blend_2)
    if(guide_mode_2 == 10):
      soft_light_blend_2 = soft_light_blend(latent_guide_2, x)
      x = (x - guide_2[i] * sigma_next * x) + (guide_2[i] * sigma_next * soft_light_blend_2)
    if(guide_mode_2 == 11):
      linear_light_blend_2 = linear_light_blend(x, latent_guide_2)
      x = (x - guide_2[i] * sigma_next * x) + (guide_2[i] * sigma_next * linear_light_blend_2)
    if(guide_mode_2 == 12):
      linear_light_blend_2 = linear_light_blend(latent_guide_2, x)
      x = (x - guide_2[i] * sigma_next * x) + (guide_2[i] * sigma_next * linear_light_blend_2)
    if(guide_mode_2 == 13):
      vivid_light_blend_2 = vivid_light_blend(x, latent_guide_2)
      x = (x - guide_2[i] * sigma_next * x) + (guide_2[i] * sigma_next * vivid_light_blend_2)
    if(guide_mode_2 == 14):
      vivid_light_blend_2 = vivid_light_blend(latent_guide_2, x)
      x = (x - guide_2[i] * sigma_next * x) + (guide_2[i] * sigma_next * vivid_light_blend_2)
  return x




def fix4_hard_light_blend(base_latent, blend_latent):

    multiply_effect = 2 * base_latent * blend_latent
    screen_effect = base_latent + blend_latent - base_latent * blend_latent
    result = torch.where(blend_latent < 0, multiply_effect, screen_effect)
    return result

def fix3_hard_light_blend(base_latent, blend_latent):
    blend_mid = (blend_latent.max() + blend_latent.min()) / 2

    multiply_effect = 2 * base_latent * ((blend_latent - blend_latent.min()) / (blend_latent.max() - blend_latent.min()))
    screen_effect = (blend_latent.max() - blend_latent.min()) + base_latent - 2 * (base_latent * (blend_latent - blend_latent.min()) / (blend_latent.max() - blend_latent.min()))

    result = torch.where(blend_latent <= blend_mid, multiply_effect, screen_effect)
    return result

def fix2_hard_light_blend(base_latent, blend_latent):

    blend_range = blend_latent.max() - blend_latent.min()
    blend_mid = blend_latent.min() + blend_range / 2

    result = torch.where(blend_latent <= blend_mid,
                         2 * (blend_latent - blend_latent.min()) / blend_range * base_latent,
                         1 - 2 * (1 - (blend_latent - blend_latent.min()) / blend_range) * (1 - base_latent))
    return result

def fix_hard_light_blend(base_latent, blend_latent):

    blend_latent = blend_latent - blend_latent.min()
    base_latent = base_latent - base_latent.min()

    blend_max = blend_latent.max()
    blend_min = blend_latent.min()
    blend_half = blend_max/2

    result = torch.where(blend_latent < blend_half,
                                  2 * base_latent * blend_latent,
                                  blend_max - 2 * (blend_max - blend_latent) * (blend_max - blend_latent))

    result = result + base_latent.min()
    return result

def bold_hard_light_blend(base_latent, blend_latent):
    blend_latent = (blend_latent - blend_latent.min()) / (blend_latent.max() - blend_latent.min())
    blend_latent = blend_latent - blend_latent.min()

    blend_max = blend_latent.max()
    blend_min = blend_latent.min()
    blend_half = blend_max/2
    
    positive_mask = base_latent >= 0
    negative_mask = base_latent < 0
    
    positive_latent = base_latent * positive_mask.float()
    negative_latent = base_latent * negative_mask.float()
    
    positive_result = torch.where(blend_latent < blend_half,
                                  2 * positive_latent * blend_latent,
                                  1 - 2 * (1 - positive_latent) * (1 - blend_latent))

    negative_result = torch.where(blend_latent < blend_half,
                                  2 * negative_latent.abs() * blend_latent,
                                  1 - 2 * (1 - negative_latent.abs()) * (1 - blend_latent))
    negative_result = -negative_result 
    
    combined_result = positive_result * positive_mask.float() + negative_result * negative_mask.float()

    return combined_result

def bold_soft_light_blend(base_latent, blend_latent):
    blend_latent = (blend_latent - blend_latent.min()) / (blend_latent.max() - blend_latent.min())

    positive_mask = base_latent >= 0
    negative_mask = base_latent < 0

    positive_result = torch.where(blend_latent > 0.5,
                                  (1 - (1 - base_latent) * (1 - (blend_latent - 0.5) * 2)),
                                  base_latent * (blend_latent * 2))
    positive_result *= positive_mask.float()

    negative_base = base_latent.abs() * negative_mask.float()
    negative_result = torch.where(blend_latent > 0.5,
                                  (1 - (1 - negative_base) * (1 - (blend_latent - 0.5) * 2)),
                                  negative_base * (blend_latent * 2))
    negative_result *= negative_mask.float()
    negative_result = -negative_result

    return positive_result + negative_result

def bold_vivid_light_blend(base_latent, blend_latent):
    blend_latent = (blend_latent - blend_latent.min()) / (blend_latent.max() - blend_latent.min())

    positive_mask = base_latent >= 0
    negative_mask = base_latent < 0

    positive_result = torch.where(blend_latent > 0,
                                  1 - (1 - base_latent) / ((blend_latent - 0.5) * 2),
                                  base_latent / (1 - (blend_latent - 0.5) * 2))
    positive_result *= positive_mask.float()

    negative_base = base_latent.abs() * negative_mask.float()
    negative_result = torch.where(blend_latent > 0.5,
                                  1 - (1 - negative_base) / ((blend_latent - 0.5) * 2),
                                  negative_base / (1 - (blend_latent - 0.5) * 2))
    negative_result *= negative_mask.float()
    negative_result = -negative_result 

    return positive_result + negative_result

def hard_light_blend(base_latent, blend_latent):
    blend_latent = (blend_latent - blend_latent.min()) / (blend_latent.max() - blend_latent.min())

    positive_mask = base_latent >= 0
    negative_mask = base_latent < 0
    
    positive_latent = base_latent * positive_mask.float()
    negative_latent = base_latent * negative_mask.float()

    positive_result = torch.where(blend_latent < 0.5,
                                  2 * positive_latent * blend_latent,
                                  1 - 2 * (1 - positive_latent) * (1 - blend_latent))

    negative_result = torch.where(blend_latent < 0.5,
                                  2 * negative_latent.abs() * blend_latent,
                                  1 - 2 * (1 - negative_latent.abs()) * (1 - blend_latent))
    negative_result = -negative_result

    combined_result = positive_result * positive_mask.float() + negative_result * negative_mask.float()

    return combined_result

def soft_light_blend(base_latent, blend_latent):
    blend_latent = (blend_latent - blend_latent.min()) / (blend_latent.max() - blend_latent.min())

    positive_mask = base_latent >= 0
    negative_mask = base_latent < 0

    positive_result = torch.where(blend_latent > 0.5,
                                  (1 - (1 - base_latent) * (1 - (blend_latent - 0.5) * 2)),
                                  base_latent * (blend_latent * 2))
    positive_result *= positive_mask.float()

    negative_base = base_latent.abs() * negative_mask.float()
    negative_result = torch.where(blend_latent > 0.5,
                                  (1 - (1 - negative_base) * (1 - (blend_latent - 0.5) * 2)),
                                  negative_base * (blend_latent * 2))
    negative_result *= negative_mask.float()
    negative_result = -negative_result  

    return positive_result + negative_result

def vivid_light_blend(base_latent, blend_latent):
    blend_latent = (blend_latent - blend_latent.min()) / (blend_latent.max() - blend_latent.min())

    positive_mask = base_latent >= 0
    negative_mask = base_latent < 0

    positive_result = torch.where(blend_latent > 0.5,
                                  1 - (1 - base_latent) / ((blend_latent - 0.5) * 2),
                                  base_latent / (1 - (blend_latent - 0.5) * 2))
    positive_result *= positive_mask.float()

    negative_base = base_latent.abs() * negative_mask.float()
    negative_result = torch.where(blend_latent > 0.5,
                                  1 - (1 - negative_base) / ((blend_latent - 0.5) * 2),
                                  negative_base / (1 - (blend_latent - 0.5) * 2))
    negative_result *= negative_mask.float()
    negative_result = -negative_result 

    return positive_result + negative_result

def linear_light_blend(base_latent, blend_latent):
    blend_latent = (blend_latent - blend_latent.min()) / (blend_latent.max() - blend_latent.min())

    positive_mask = base_latent >= 0
    negative_mask = base_latent < 0

    positive_result = base_latent + 2 * blend_latent - 1
    positive_result *= positive_mask.float()

    negative_result = -base_latent.abs() + 2 * blend_latent - 1
    negative_result *= negative_mask.float()
    negative_result = -negative_result 

    return positive_result + negative_result

"""blend_modes = {
    'hard_light': hard_light_blend,
    'soft_light': soft_light_blend,
    'vivid_light': vivid_light_blend,
    'linear_light': linear_light_blend,
    'subtractive': subtractive_blend,
    'average': average_blend,
    'multiply': multiply_blend,
    'screen': screen_blend,
    'color_burn': color_burn_blend,
    'color_dodge': color_dodge_blend,
}
"""