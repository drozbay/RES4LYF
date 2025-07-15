import torch
from torch import Tensor
from typing import Optional, Callable, Tuple, List, Dict, Any, Union

import comfy.model_patcher
import comfy.supported_models


import itertools 
from dataclasses import dataclass


from .phi_functions        import Phi
from .rk_coefficients_beta import get_implicit_sampler_name_list, get_rk_methods_beta, is_exponential
from ..helper              import ExtraOptions, get_max_dtype
from ..latents             import get_orthogonal, get_collinear, get_cosine_similarity, tile_latent, untile_latent

from ..res4lyf             import RESplain

from ..models import PRED

MAX_STEPS = 10000


def get_data_from_step   (x:Tensor, x_next:Tensor, sigma:Tensor, sigma_next:Tensor) -> Tensor:
    h = sigma_next - sigma
    return (sigma_next * x - sigma * x_next) / h

def get_epsilon_from_step(x:Tensor, x_next:Tensor, sigma:Tensor, sigma_next:Tensor) -> Tensor:
    h = sigma_next - sigma
    return (x - x_next) / h


class NoiseOps:
    def __init__(self,
            model,
            noise_anchor  : float       = 1.0,
            model_device  : str         = 'cuda',
            work_device   : str         = 'cpu',
            dtype         : torch.dtype = torch.float64,
            extra_options : str         = ""
        ):
        
        self.model_device : torch.device = model_device
        self.work_device  : torch.device = work_device
        self.dtype        : torch.dtype  = dtype
        self.ANCHOR       : float        = noise_anchor
        self.EO           : ExtraOptions = ExtraOptions(extra_options)

        if hasattr(model, "model"):
            model_sampling = model.model.model_sampling
        elif hasattr(model, "inner_model"):
            model_sampling = model.inner_model.inner_model.model_sampling
            
        self.v_type                      = PRED.get_type(model_sampling)
        
        self.sigma_min    : Tensor       = model_sampling.sigma_min.to(dtype=dtype, device=work_device)
        self.sigma_max    : Tensor       = model_sampling.sigma_max.to(dtype=dtype, device=work_device)
        

    @staticmethod
    def create(
            model,
            rk_type       : str,
            noise_anchor  : float       = 1.0,
            model_device  : str         = 'cuda',
            work_device   : str         = 'cpu',
            dtype         : torch.dtype = torch.float64,
            extra_options : str         = ""
        ) : #-> "Union[VP_EXP, VE_EXP]":
        
        if hasattr(model, "model"):
            model_sampling = model.model.model_sampling
        elif hasattr(model, "inner_model"):
            model_sampling = model.inner_model.inner_model.model_sampling
        
        v_type = PRED.get_type(model_sampling)
        VP_SDE = v_type == CONST
        EXPONENTIAL = is_exponential(rk_type)
        
        if EXPONENTIAL:
            if   v_type in PRED.TYPE_VE:
                return VE_EXP(model, noise_anchor, model_device, work_device, dtype, extra_options)
            elif v_type in PRED.TYPE_VP:
                return VP_EXP(model, noise_anchor, model_device, work_device, dtype, extra_options)
        else:
            if   v_type in PRED.TYPE_VE:
                return VE_LIN(model, noise_anchor, model_device, work_device, dtype, extra_options)
            elif v_type in PRED.TYPE_VP:
                return VP_LIN(model, noise_anchor, model_device, work_device, dtype, extra_options)
    
    def __call__(self):
        raise NotImplementedError("This method got clownsharked!")



    def get_eps(self, *args):
        if   len(args) == 3:
            x, denoised, sigma = args
            return (x - denoised) / sigma
        elif len(args == 5):
            x_0, x, denoised, sigma, sub_sigma = args
            eps_anchor   = (x_0 - denoised) / sigma
            eps_unmoored =   (x - denoised) / sub_sigma
            return eps_unmoored + self.ANCHOR * (eps_anchor - eps_unmoored)
        else:
            raise ValueError(f"get_eps expected 3 or 5 arguments, got {len(args)}")

    def get_data_from_step   (self, x:Tensor, x_next:Tensor, sigma:Tensor, sigma_next:Tensor) -> Tensor:
        h = sigma_next - sigma
        return (sigma_next * x - sigma * x_next) / h

    def get_epsilon_from_step(self, x:Tensor, x_next:Tensor, sigma:Tensor, sigma_next:Tensor) -> Tensor:
        h = sigma_next - sigma
        return (x - x_next) / h
    
    def get_epsilon_remainder_from_step(self, x:Tensor, x_next:Tensor, sigma:Tensor, sigma_next:Tensor) -> Tensor:
        h = sigma_next - sigma
        return (x - x_next) / h

    def get_guide_epsilon(self, 
                            x_0           : Tensor, 
                            x             : Tensor, 
                            y             : Tensor, 
                            sigma         : Tensor, 
                            sigma_cur     : Tensor, 
                            sigma_down    : Optional[Tensor] = None, 
                            epsilon_scale : Optional[Tensor] = None, 
                            ) -> Tensor:

        if sigma_down > sigma:
            sigma_ratio = self.sigma_max - sigma_cur.clone()
        else:
            sigma_ratio = sigma_cur.clone()
        sigma_ratio = epsilon_scale if epsilon_scale is not None else sigma_ratio

        if sigma_down is None:
            return (x - y) / sigma_ratio
        else:
            if sigma_down > sigma:
                return (y - x) / sigma_ratio
            else:
                return (x - y) / sigma_ratio



    def swap_noise_step(self, x_0:Tensor, x_next:Tensor, mask:Optional[Tensor]=None) -> Tensor:
        eps_next      = (x_0 - x_next) / (self.sigma - self.sigma_next)
        denoised_next = x_0 - self.sigma * eps_next

        x_noised = self.alpha_ratio_eta * (denoised_next + self.sigma_down_eta * eps_next) + self.sigma_up_eta * noise * self.s_noise

        if mask is not None:
            x = mask * x_noised + (1-mask) * x_next
        else:
            x = x_noised
        
        return x

    def extract_latent_swap_noise(self, x:Tensor, x_noise_swapped:Tensor, sigma:Tensor, old_noise:Tensor) -> Tensor:
        return (x - x_noise_swapped) / sigma + old_noise

    def update_latent_swap_noise(self, x:Tensor, sigma:Tensor, old_noise:Tensor, new_noise:Tensor) -> Tensor:
        return x + sigma * (new_noise - old_noise)



class VP(NoiseOps):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def swap_data(self, x:Tensor, data_x:Tensor, data_y:Tensor, sigma:Tensor, mask:Optional[Tensor]=None,) -> Tensor:
        mask = 1.0 if mask is None else mask
        return x + mask * (self.sigma_max - sigma) * (data_y - data_x)
    
    def get_x(self, data:Tensor, noise:Tensor, sigma:Tensor) -> Tensor:
        return data + sigma * noise
    
    def get_guide_sync(self, eps_y:Tensor, sigma:Tensor, y0_bongflow:Tensor, noise_bongflow:Tensor,) -> Tensor:
        return y0_bongflow - noise_bongflow - eps_y
        #return self.get_guide_sync_factor(sigma) * (y0_bongflow - noise_bongflow) - eps_y
    

class VE(NoiseOps):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def swap_data(self, x:Tensor, data_x:Tensor, data_y:Tensor, sigma:Tensor, mask:Optional[Tensor]=None,) -> Tensor:
        mask = 1.0 if mask is None else mask
        return x + mask * (data_y - data_x)

    def get_x(self, data:Tensor, noise:Tensor, sigma:Tensor) -> Tensor:
        return (self.sigma_max - sigma) * data + sigma * noise

    def get_guide_sync(self, eps_y:Tensor, sigma:Tensor, y0_bongflow:Tensor, noise_bongflow:Tensor,) -> Tensor:
        return -noise_bongflow - eps_y
        #return self.get_guide_sync_factor(sigma) * -noise_bongflow - eps_y






class LIN(NoiseOps):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    @staticmethod
    def alpha_fn(neg_h:Tensor) -> Tensor:
        return torch.ones_like(neg_h)

    @staticmethod
    def sigma_fn(t:Tensor) -> Tensor:
        return t

    @staticmethod
    def t_fn(sigma:Tensor) -> Tensor:
        return sigma
    
    @staticmethod
    def h_fn(sigma_down:Tensor, sigma:Tensor) -> Tensor:
        return sigma_down - sigma

    def get_eps(self, *args):
        if   len(args) == 3:
            x, denoised, sigma = args
            return (x - denoised) / sigma
        elif len(args == 5):
            x_0, x, denoised, sigma, sub_sigma = args
            eps_anchor   = (x_0 - denoised) / sigma
            eps_unmoored =   (x - denoised) / sub_sigma
            return eps_unmoored + self.ANCHOR * (eps_anchor - eps_unmoored)
        else:
            raise ValueError(f"get_eps expected 3 or 5 arguments, got {len(args)}")

    #def get_guide_sync_factor(self, sigma:Tensor,) -> Tensor:
    #    return -torch.ones_like(sigma)

    def get_tableau_factors(self, h:Tensor, sigma:Tensor,) -> Tensor:
        return h/h, -sigma/sigma






class EXP(NoiseOps):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    @staticmethod
    def alpha_fn(neg_h:Tensor) -> Tensor:
        return torch.exp(neg_h)

    @staticmethod
    def sigma_fn(t:Tensor) -> Tensor:
        return t.neg().exp()

    @staticmethod
    def t_fn(sigma:Tensor) -> Tensor:
        return sigma.log().neg()
    
    @staticmethod
    def h_fn(sigma_down:Tensor, sigma:Tensor) -> Tensor:
        return -torch.log(sigma_down/sigma)
        
    def get_eps(self, *args):
        if   len(args) == 3:
            x, denoised, sigma = args
            return denoised - x
        elif len(args == 5):
            x_0, x, denoised, sigma, sub_sigma = args
            eps_anchored = (x_0 - denoised) / sigma
            eps_unmoored = (x   - denoised) / sub_sigma
            eps      = eps_unmoored + self.ANCHOR * (eps_anchored - eps_unmoored)
            denoised = x_0 - sigma * eps
            return denoised - x_0
        else:
            raise ValueError(f"get_eps expected 3 or 5 arguments, got {len(args)}")

    #def get_guide_sync_factor(self, sigma:Tensor,) -> Tensor:
    #    return sigma

    def get_tableau_factors(self, h:Tensor, sigma:Tensor,) -> Tensor:
        return h, -sigma








class VP_EXP(VP, EXP): pass
class VP_LIN(VP, LIN): pass
class VE_EXP(VE, EXP): pass
class VE_LIN(VE, LIN): pass

