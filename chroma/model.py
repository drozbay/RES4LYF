#Original code can be found on: https://github.com/black-forest-labs/flux

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from einops import rearrange, repeat
import comfy.ldm.common_dit

from ..helper import ExtraOptions

from ..latents import tile_latent, untile_latent, gaussian_blur_2d, median_blur_2d
from ..style_transfer import StyleMMDiT_Model, apply_scattersort_masked, apply_scattersort_tiled, adain_seq_inplace, adain_patchwise_row_batch_med, adain_patchwise_row_batch
from typing import Optional, Callable, Tuple, Dict, Any, Union, TYPE_CHECKING, TypeVar

from comfy.ldm.flux.layers import (
    EmbedND,
    timestep_embedding,
)

from .layers import (
    ReChromaDoubleStreamBlock,
    LastLayer,
    ReChromaSingleStreamBlock,
    Approximator,
    ChromaModulationOut,
)


@dataclass
class ChromaParams:
    in_channels        : int
    out_channels       : int
    context_in_dim     : int
    hidden_size        : int
    mlp_ratio          : float
    num_heads          : int
    depth              : int
    depth_single_blocks: int
    axes_dim           : list
    theta              : int
    patch_size         : int
    qkv_bias           : bool
    in_dim             : int
    out_dim            : int
    hidden_dim         : int
    n_layers           : int




class ReChroma(nn.Module):
    """
    Transformer model for flow matching on sequences.
    """

    def __init__(self, image_model=None, final_layer=True, dtype=None, device=None, operations=None, **kwargs):
        super().__init__()
        self.dtype        = dtype
        params            = ChromaParams(**kwargs)
        self.params       = params
        self.patch_size   = params.patch_size
        self.in_channels  = params.in_channels
        self.out_channels = params.out_channels
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(
                f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}"
            )
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(f"Got {params.axes_dim} but expected positional dim {pe_dim}")
        self.hidden_size = params.hidden_size
        self.num_heads   = params.num_heads
        self.in_dim      = params.in_dim
        self.out_dim     = params.out_dim
        self.hidden_dim  = params.hidden_dim
        self.n_layers    = params.n_layers
        self.pe_embedder = EmbedND(dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim)
        self.img_in      = operations.Linear(self.in_channels, self.hidden_size, bias=True, dtype=dtype, device=device)
        self.txt_in      = operations.Linear(params.context_in_dim, self.hidden_size, dtype=dtype, device=device)
        # set as nn identity for now, will overwrite it later.
        self.distilled_guidance_layer = Approximator(
                    in_dim=self.in_dim,
                    hidden_dim=self.hidden_dim,
                    out_dim=self.out_dim,
                    n_layers=self.n_layers,
                    dtype=dtype, device=device, operations=operations
                )


        self.double_blocks = nn.ModuleList(
            [
                ReChromaDoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    qkv_bias=params.qkv_bias,
                    dtype=dtype, device=device, operations=operations
                )
                for _ in range(params.depth)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                ReChromaSingleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=params.mlp_ratio, dtype=dtype, device=device, operations=operations)
                for _ in range(params.depth_single_blocks)
            ]
        )

        if final_layer:
            self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels, dtype=dtype, device=device, operations=operations)

        self.skip_mmdit = []
        self.skip_dit = []
        self.lite = False

    def get_modulations(self, tensor: torch.Tensor, block_type: str, *, idx: int = 0):
        # This function slices up the modulations tensor which has the following layout:
        #   single     : num_single_blocks * 3 elements
        #   double_img : num_double_blocks * 6 elements
        #   double_txt : num_double_blocks * 6 elements
        #   final      : 2 elements
        if block_type == "final":
            return (tensor[:, -2:-1, :], tensor[:, -1:, :])
        single_block_count = self.params.depth_single_blocks
        double_block_count = self.params.depth
        offset = 3 * idx
        if block_type == "single":
            return ChromaModulationOut.from_offset(tensor, offset)
        # Double block modulations are 6 elements so we double 3 * idx.
        offset *= 2
        if block_type in {"double_img", "double_txt"}:
            # Advance past the single block modulations.
            offset += 3 * single_block_count
            if block_type == "double_txt":
                # Advance past the double block img modulations.
                offset += 6 * double_block_count
            return (
                ChromaModulationOut.from_offset(tensor, offset),
                ChromaModulationOut.from_offset(tensor, offset + 3),
            )
        raise ValueError("Bad block_type")







    def process_img(self, x, index=0, h_offset=0, w_offset=0):
        bs, c, h, w = x.shape
        patch_size = self.patch_size
        x = comfy.ldm.common_dit.pad_to_patch_size(x, (patch_size, patch_size))

        img = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size)
        h_len = ((h + (patch_size // 2)) // patch_size)
        w_len = ((w + (patch_size // 2)) // patch_size)

        h_offset = ((h_offset + (patch_size // 2)) // patch_size)
        w_offset = ((w_offset + (patch_size // 2)) // patch_size)

        img_ids = torch.zeros((h_len, w_len, 3), device=x.device, dtype=x.dtype)
        img_ids[:, :, 0] = img_ids[:, :, 1] + index
        img_ids[:, :, 1] = img_ids[:, :, 1] + torch.linspace(h_offset, h_len - 1 + h_offset, steps=h_len, device=x.device, dtype=x.dtype).unsqueeze(1)
        img_ids[:, :, 2] = img_ids[:, :, 2] + torch.linspace(w_offset, w_len - 1 + w_offset, steps=w_len, device=x.device, dtype=x.dtype).unsqueeze(0)
        return img, repeat(img_ids, "h w c -> b (h w) c", b=bs)

    
    def _get_img_ids(self, x, bs, h_len, w_len, h_start, h_end, w_start, w_end):
        img_ids          = torch.zeros(  (h_len,   w_len, 3),              device=x.device, dtype=x.dtype)
        img_ids[..., 1] += torch.linspace(h_start, h_end - 1, steps=h_len, device=x.device, dtype=x.dtype)[:, None]
        img_ids[..., 2] += torch.linspace(w_start, w_end - 1, steps=w_len, device=x.device, dtype=x.dtype)[None, :]
        img_ids          = repeat(img_ids, "h w c -> b (h w) c", b=bs)
        return img_ids

    def forward(self,
                x,
                timestep,
                context,
                #y,             # no clip L for chroma
                guidance,
                ref_latents=None, 
                control             = None,
                transformer_options = {},
                mask                = None,
                **kwargs
                ):
        t = timestep
        self.max_seq = (128 * 128) // (2 * 2)
        x_orig      = x.clone()
        b, c, h, w  = x.shape
        h_len = ((h + (self.patch_size // 2)) // self.patch_size) # h_len 96
        w_len = ((w + (self.patch_size // 2)) // self.patch_size) # w_len 96
        img_len = h_len * w_len
        img_slice = slice(-img_len, None) #slice(None, img_len)
        txt_slice = slice(None, -img_len)
        SIGMA = t[0].clone() #/ 1000
        EO = transformer_options.get("ExtraOptions", ExtraOptions(""))
        if EO is not None:
            EO.mute = True

        if EO("zero_heads"):
            HEADS = 0
        else:
            HEADS = 24

        StyleMMDiT = transformer_options.get('StyleMMDiT', StyleMMDiT_Model())        
        StyleMMDiT.set_len(h_len, w_len, img_slice, txt_slice, HEADS=HEADS)
        StyleMMDiT.Retrojector = self.Retrojector if hasattr(self, "Retrojector") else None
        transformer_options['StyleMMDiT'] = None
        
        x_tmp = transformer_options.get("x_tmp")
        if x_tmp is not None:
            x_tmp = x_tmp.expand(x.shape[0], -1, -1, -1).clone()
            img = comfy.ldm.common_dit.pad_to_patch_size(x_tmp, (self.patch_size, self.patch_size))
        else:
            img = comfy.ldm.common_dit.pad_to_patch_size(x, (self.patch_size, self.patch_size))
        
        y0_style, img_y0_style = None, None

        img_orig, t_orig, context_orig = clone_inputs(img, t, context)
    
        weight    = -1 * transformer_options.get("regional_conditioning_weight", 0.0)
        floor     = -1 * transformer_options.get("regional_conditioning_floor",  0.0)
        update_cross_attn = transformer_options.get("update_cross_attn")
    
        z_ = transformer_options.get("z_")   # initial noise and/or image+noise from start of rk_sampler_beta() 
        rk_row = transformer_options.get("row") # for "smart noise"
        if z_ is not None:
            x_init = z_[rk_row].to(x)
        elif 'x_init' in transformer_options:
            x_init = transformer_options.get('x_init').to(x)

        # recon loop to extract exact noise pred for scattersort guide assembly
        RECON_MODE = StyleMMDiT.noise_mode == "recon"
        recon_iterations = 2 if StyleMMDiT.noise_mode == "recon" else 1
        for recon_iter in range(recon_iterations):
            y0_style = StyleMMDiT.guides
            y0_style_active = True if type(y0_style) == torch.Tensor else False
            
            RECON_MODE = True     if StyleMMDiT.noise_mode == "recon" and recon_iter == 0     else False
            
            if StyleMMDiT.noise_mode == "recon" and recon_iter == 1:
                x_recon = x_tmp if x_tmp is not None else x_orig
                noise_prediction = x_recon + (1-SIGMA.to(x_recon)) * eps.to(x_recon)
                denoised = x_recon - SIGMA.to(x_recon) * eps.to(x_recon)
                
                denoised = StyleMMDiT.apply_recon_lure(denoised, y0_style)

                new_x = (1-SIGMA.to(denoised)) * denoised + SIGMA.to(denoised) * noise_prediction
                img_orig = img = comfy.ldm.common_dit.pad_to_patch_size(new_x, (self.patch_size, self.patch_size))
                
                x_init = noise_prediction
            elif StyleMMDiT.noise_mode == "bonanza":
                x_init = torch.randn_like(x_init)

            if y0_style_active:
                if y0_style.sum() == 0.0 and y0_style.std() == 0.0:
                    y0_style_noised = img_orig.clone()
                    img_y0_style_orig = y0_style_noised.clone()             ### accomodate cfg-like attention debauchery?
                else:
                    SIGMA_ADAIN         = (SIGMA * EO("eps_adain_sigma_factor", 1.0)).to(y0_style)
                    y0_style_noised     = (1-SIGMA_ADAIN) * y0_style + SIGMA_ADAIN * x_init.expand_as(x).to(y0_style)   #always only use first batch of noise to avoid broadcasting
                    img_y0_style_orig   = comfy.ldm.common_dit.pad_to_patch_size(y0_style_noised, (self.patch_size, self.patch_size))

            mask_zero = None
            
            out_list = []
            for cond_iter in range(len(transformer_options['cond_or_uncond'])):
                UNCOND = transformer_options['cond_or_uncond'][cond_iter] == 1
                
                if update_cross_attn is not None:
                    update_cross_attn['UNCOND'] = UNCOND

                bsz_style = y0_style.shape[0] if y0_style_active else 0
                bsz       = 1 if RECON_MODE else bsz_style + 1

                img, t, context = clone_inputs(img_orig, t_orig, context_orig, index=cond_iter)
                
                mask = None
                if not UNCOND and 'AttnMask' in transformer_options: # and weight != 0:
                    mask = transformer_options['AttnMask'].attn_mask.mask.to('cuda')
                    if mask_zero is None:
                        mask_zero = torch.ones_like(mask)
                        mask_zero[txt_slice, txt_slice] = mask[txt_slice, txt_slice]

                    if weight == 0:
                        context = transformer_options['RegContext'].context.to(context.dtype).to(context.device)
                        mask = None
                    else:
                        context = transformer_options['RegContext'].context.to(context.dtype).to(context.device)

                if UNCOND and 'AttnMask_neg' in transformer_options: # and weight != 0:
                    mask = transformer_options['AttnMask_neg'].attn_mask.mask.to('cuda')
                    if mask_zero is None:
                        mask_zero = torch.ones_like(mask)
                        mask_zero[txt_slice, txt_slice] = mask[txt_slice, txt_slice]

                    if weight == 0:
                        context = transformer_options['RegContext_neg'].context.to(context.dtype).to(context.device)
                        mask = None
                    else:
                        context = transformer_options['RegContext_neg'].context.to(context.dtype).to(context.device)

                elif UNCOND and 'AttnMask' in transformer_options:
                    mask = transformer_options['AttnMask'].attn_mask.mask.to('cuda')
                    
                    if mask_zero is None:
                        mask_zero = torch.ones_like(mask)

                        mask_zero[txt_slice, txt_slice] = mask[txt_slice, txt_slice]
                    if weight == 0:                                                                             # ADDED 5/23/2025
                        context = transformer_options['RegContext'].context.to(context.dtype).to(context.device)  # ADDED 5/26/2025 14:53
                        mask = None
                    else:
                        A       = context
                        B       = transformer_options['RegContext'].context
                        context = A.repeat(1,    (B.shape[1] // A.shape[1]) + 1, 1)[:,   :B.shape[1], :]


                if y0_style_active and not RECON_MODE:
                    if mask is None:
                        context, _, _ = StyleMMDiT.apply_style_conditioning(
                            UNCOND = UNCOND,
                            base_context       = context,
                            base_y             = None,     # no clip L for chroma
                            base_llama3        = None,
                        )
                    else:
                        context = context.repeat(bsz_style + 1, 1, 1)
                    img_y0_style = img_y0_style_orig[cond_iter:cond_iter+1].clone()

                if mask is not None and not type(mask[0][0].item()) == bool:
                    mask = mask.to(x.dtype)
                if mask_zero is not None and not type(mask_zero[0][0].item()) == bool:
                    mask_zero = mask_zero.to(x.dtype)



                mod_index_length = 344
                distill_timestep = timestep_embedding(t.detach().clone(), 16).to(img.device, img.dtype)
                # guidance = guidance *
                distill_guidance = timestep_embedding(guidance.detach().clone(), 16).to(img.device, img.dtype)

                # get all modulation index
                modulation_index = timestep_embedding(torch.arange(mod_index_length), 32).to(img.device, img.dtype)
                # we need to broadcast the modulation index here so each batch has all of the index
                modulation_index = modulation_index.unsqueeze(0).repeat(img.shape[0], 1, 1).to(img.device, img.dtype)
                # and we need to broadcast timestep and guidance along too
                timestep_guidance = torch.cat([distill_timestep, distill_guidance], dim=1).unsqueeze(1).repeat(1, mod_index_length, 1).to(img.dtype).to(img.device, img.dtype)
                # then and only then we could concatenate it together
                input_vec = torch.cat([timestep_guidance, modulation_index], dim=-1).to(img.device, img.dtype)

                mod_vectors = self.distilled_guidance_layer(input_vec.to(torch.bfloat16))


        
                img_in_dtype = self.img_in.weight.data.dtype
                if img_in_dtype not in {torch.bfloat16, torch.float16, torch.float32, torch.float64}:
                    img_in_dtype = x.dtype
                
                img = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=self.patch_size, pw=self.patch_size)
                img = self.img_in(img.to(img_in_dtype))
                img_ids = self._get_img_ids(img, bsz, h_len, w_len, 0, h_len, 0, w_len)


                if y0_style_active and not RECON_MODE:
                    img_y0_style = rearrange(img_y0_style_orig, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=self.patch_size, pw=self.patch_size)
                    img_y0_style = self.img_in(img_y0_style.to(img_in_dtype))  # hidden_states 1,4032,2560         for 1024x1024: -> 1,4096,2560      ,64 -> ,2560 (x40)
                    img = torch.cat([img, img_y0_style], dim=0)

                # txt_ids -> 1,414,3
                txt_ids = torch.zeros((bsz, context.shape[-2], 3), device=img.device, dtype=x.dtype) 
                ids     = torch.cat((txt_ids, img_ids), dim=-2)   # ids -> 1,4446,3       # flipped from hidream
                rope    = self.pe_embedder(ids)                  # rope -> 1, 4446, 1, 64, 2, 2

                txt_init = self.txt_in(context)
                txt_init_len = txt_init.shape[-2]                                       # 271

                img = StyleMMDiT(img, "proj_in")
                
                img = img.to(x) if img is not None else None
                
                total_layers = len(self.double_blocks) + len(self.single_blocks)
                
                # DOUBLE STREAM
                ca_idx = 0
                for bid, (block, style_block) in enumerate(zip(self.double_blocks, StyleMMDiT.double_blocks)):
                    txt = txt_init
                    
                    clip = (
                        self.get_modulations(mod_vectors, "double_img", idx=bid),
                        self.get_modulations(mod_vectors, "double_txt", idx=bid),
                    )
                    
                    if   weight > 0 and mask is not None and     weight  <      bid/total_layers:
                        img, txt_init = block(img, txt, clip, rope, mask_zero, style_block=style_block)
                        
                    elif (weight < 0 and mask is not None and abs(weight) < (1 - bid/total_layers)):
                        img_tmpZ, txt_tmpZ = img.clone(), txt.clone()

                        # more efficient than the commented lines below being used instead in the loop?
                        img_tmpZ, txt_init = block(img_tmpZ, txt_tmpZ, clip, rope, mask, style_block=style_block)
                        img     , txt_tmpZ = block(img     , txt     , clip, rope, mask_zero, style_block=style_block)
                        
                    elif floor > 0 and mask is not None and     floor  >      bid/total_layers:
                        mask_tmp = mask.clone()
                        mask_tmp[img_slice,img_slice] = 1.0
                        img, txt_init = block(img, txt, clip, rope, mask_tmp, style_block=style_block)
                        
                    elif floor < 0 and mask is not None and abs(floor) > (1 - bid/total_layers):
                        mask_tmp = mask.clone()
                        mask_tmp[img_slice,img_slice] = 1.0
                        img, txt_init = block(img, txt, clip, rope, mask_tmp, style_block=style_block)
                        
                    elif update_cross_attn is not None and update_cross_attn['skip_cross_attn']:
                        img, txt_init = block(img, txt, clip, rope, mask, update_cross_attn=update_cross_attn)
                        
                    else:
                        img, txt_init = block(img, txt, clip, rope, mask, update_cross_attn=update_cross_attn, style_block=style_block)

                    if control is not None: 
                        control_i = control.get("input")
                        if bid < len(control_i):
                            add = control_i[bid]
                            if add is not None:
                                img[:1] += add
                    
                    if hasattr(self, "pulid_data"):
                        if self.pulid_data:
                            if bid % self.pulid_double_interval == 0:
                                for _, node_data in self.pulid_data.items():
                                    if torch.any((node_data['sigma_start'] >= timestep) & (timestep >= node_data['sigma_end'])):
                                        img = img + node_data['weight'] * self.pulid_ca[ca_idx](node_data['embedding'], img)
                                ca_idx += 1

                # END DOUBLE STREAM
                
                img       = torch.cat([txt_init, img], dim=-2)   # 4032 + 271 -> 4303     # txt embed from double stream block   # flipped from hidream

                double_layers = len(self.double_blocks)

                # SINGLE STREAM
                for bid, (block, style_block) in enumerate(zip(self.single_blocks, StyleMMDiT.single_blocks)):
                    clip = self.get_modulations(mod_vectors, "single", idx=bid)
                    if   weight > 0 and mask is not None and     weight  <      (bid+double_layers)/total_layers:
                        img = block(img, clip, rope, mask_zero, style_block=style_block)
                    
                    elif weight < 0 and mask is not None and abs(weight) < (1 - (bid+double_layers)/total_layers):
                        img = block(img, clip, rope, mask_zero, style_block=style_block)
                    
                    elif floor > 0 and mask is not None and     floor  >      (bid+double_layers)/total_layers:
                        mask_tmp = mask.clone()
                        mask_tmp[img_slice,img_slice] = 1.0
                        img = block(img, clip, rope, mask_tmp, style_block=style_block)
                    
                    elif floor < 0 and mask is not None and abs(floor) > (1 - (bid+double_layers)/total_layers):
                        mask_tmp = mask.clone()
                        mask_tmp[img_slice,img_slice] = 1.0
                        img = block(img, clip, rope, mask_tmp, style_block=style_block)
                    
                    else:
                        img = block(img, clip, rope, mask, style_block=style_block)
                    
                    if control is not None: # Controlnet
                        control_o = control.get("output")
                        if bid < len(control_o):
                            add = control_o[bid]
                            if add is not None:
                                img[:1, txt_slice, ...] += add
                                
                    if hasattr(self, "pulid_data"):
                        # PuLID attention
                        if self.pulid_data:
                            real_img, txt = img[:, img_slice, ...], img[:, txt_slice, ...]
                            if bid % self.pulid_single_interval == 0:
                                # Will calculate influence of all nodes at once
                                for _, node_data in self.pulid_data.items():
                                    if torch.any((node_data['sigma_start'] >= timestep) & (timestep >= node_data['sigma_end'])):
                                        real_img = real_img + node_data['weight'] * self.pulid_ca[ca_idx](node_data['embedding'], real_img)
                                ca_idx += 1
                            img = torch.cat((txt, real_img), 1)
                            
                # END SINGLE STREAM
                
                img = img[..., img_slice, :]
                clip = self.get_modulations(mod_vectors, "final")
                shift, scale = clip
                img = (1 + scale) * self.final_layer.norm_final(img) + shift
                
                img = StyleMMDiT(img, "proj_out")

                if y0_style_active and not RECON_MODE:
                    img = img[0:1]
                
                img = self.final_layer.linear(img)

                img = rearrange(img, "... b (h w) (c ph pw) -> ... b c (h ph) (w pw)", h=h_len, w=w_len, ph=self.patch_size, pw=self.patch_size)
                out_list.append(img)
                
            output = torch.cat(out_list, dim=0)
            eps = output[:, :, :h, :w]
            
            if recon_iter == 1:
                denoised = new_x - SIGMA.to(new_x) * eps.to(new_x)
                if x_tmp is not None:
                    eps = (x_tmp - denoised.to(x_tmp)) / SIGMA.to(x_tmp)
                else:
                    eps = (x_orig - denoised.to(x_orig)) / SIGMA.to(x_orig)
                    












        freqsep_lowpass_method = transformer_options.get("freqsep_lowpass_method")
        freqsep_sigma          = transformer_options.get("freqsep_sigma")
        freqsep_kernel_size    = transformer_options.get("freqsep_kernel_size")
        freqsep_inner_kernel_size    = transformer_options.get("freqsep_inner_kernel_size")
        freqsep_stride    = transformer_options.get("freqsep_stride")
        
        freqsep_lowpass_weight = transformer_options.get("freqsep_lowpass_weight")
        freqsep_highpass_weight= transformer_options.get("freqsep_highpass_weight")
        freqsep_mask           = transformer_options.get("freqsep_mask")

        y0_style_pos = transformer_options.get("y0_style_pos")
        y0_style_neg = transformer_options.get("y0_style_neg")
        
        # end recon loop
        self.style_dtype = torch.float32 if self.style_dtype is None else self.style_dtype
        dtype = eps.dtype if self.style_dtype is None else self.style_dtype
        
        if y0_style_pos is not None:
            y0_style_pos_weight    = transformer_options.get("y0_style_pos_weight")
            y0_style_pos_synweight = transformer_options.get("y0_style_pos_synweight")
            y0_style_pos_synweight *= y0_style_pos_weight
            y0_style_pos_mask = transformer_options.get("y0_style_pos_mask")
            y0_style_pos_mask_edge = transformer_options.get("y0_style_pos_mask_edge")

            y0_style_pos = y0_style_pos.to(dtype)
            x   = x_orig.to(dtype)
            eps = eps.to(dtype)
            eps_orig = eps.clone()
            
            sigma = SIGMA #t_orig[0].to(torch.float32) / 1000
            denoised = x - sigma * eps
            
            denoised_embed = self.Retrojector.embed(denoised)
            y0_adain_embed = self.Retrojector.embed(y0_style_pos)
            
            if transformer_options['y0_style_method'] == "scattersort":
                tile_h, tile_w = transformer_options.get('y0_style_tile_height'), transformer_options.get('y0_style_tile_width')
                pad = transformer_options.get('y0_style_tile_padding')
                if pad is not None and tile_h is not None and tile_w is not None:
                    
                    denoised_spatial = rearrange(denoised_embed, "b (h w) c -> b c h w", h=h_len, w=w_len)
                    y0_adain_spatial = rearrange(y0_adain_embed, "b (h w) c -> b c h w", h=h_len, w=w_len)
                    
                    if EO("scattersort_median_LP"):
                        denoised_spatial_LP = median_blur_2d(denoised_spatial, kernel_size=EO("scattersort_median_LP",7))
                        y0_adain_spatial_LP = median_blur_2d(y0_adain_spatial, kernel_size=EO("scattersort_median_LP",7))
                        
                        denoised_spatial_HP = denoised_spatial - denoised_spatial_LP
                        y0_adain_spatial_HP = y0_adain_spatial - y0_adain_spatial_LP
                        
                        denoised_spatial_LP = apply_scattersort_tiled(denoised_spatial_LP, y0_adain_spatial_LP, tile_h, tile_w, pad)
                        
                        denoised_spatial = denoised_spatial_LP + denoised_spatial_HP
                        denoised_embed = rearrange(denoised_spatial, "b c h w -> b (h w) c")
                    else:
                        denoised_spatial = apply_scattersort_tiled(denoised_spatial, y0_adain_spatial, tile_h, tile_w, pad)
                    
                    denoised_embed = rearrange(denoised_spatial, "b c h w -> b (h w) c")
                    
                else:
                    denoised_embed = apply_scattersort_masked(denoised_embed, y0_adain_embed, y0_style_pos_mask, y0_style_pos_mask_edge, h_len, w_len)



            elif transformer_options['y0_style_method'] == "AdaIN":
                if freqsep_mask is not None:
                    freqsep_mask = freqsep_mask.view(1, 1, *freqsep_mask.shape[-2:]).float()
                    freqsep_mask = F.interpolate(freqsep_mask.float(), size=(h_len, w_len), mode='nearest-exact')
                
                if hasattr(self, "adain_tile"):
                    tile_h, tile_w = self.adain_tile
                    
                    denoised_pretile = rearrange(denoised_embed, "b (h w) c -> b c h w", h=h_len, w=w_len)
                    y0_adain_pretile = rearrange(y0_adain_embed, "b (h w) c -> b c h w", h=h_len, w=w_len)
                    
                    if self.adain_flag:
                        h_off = tile_h // 2
                        w_off = tile_w // 2
                        denoised_pretile = denoised_pretile[:,:,h_off:-h_off, w_off:-w_off]
                        self.adain_flag = False
                    else:
                        h_off = 0
                        w_off = 0
                        self.adain_flag = True
                    
                    tiles,    orig_shape, grid, strides = tile_latent(denoised_pretile, tile_size=(tile_h,tile_w))
                    y0_tiles, orig_shape, grid, strides = tile_latent(y0_adain_pretile, tile_size=(tile_h,tile_w))
                    
                    tiles_out = []
                    for i in range(tiles.shape[0]):
                        tile = tiles[i].unsqueeze(0)
                        y0_tile = y0_tiles[i].unsqueeze(0)
                        
                        tile    = rearrange(tile,    "b c h w -> b (h w) c", h=tile_h, w=tile_w)
                        y0_tile = rearrange(y0_tile, "b c h w -> b (h w) c", h=tile_h, w=tile_w)
                        
                        tile = adain_seq_inplace(tile, y0_tile)
                        tiles_out.append(rearrange(tile, "b (h w) c -> b c h w", h=tile_h, w=tile_w))
                    
                    tiles_out_tensor = torch.cat(tiles_out, dim=0)
                    tiles_out_tensor = untile_latent(tiles_out_tensor, orig_shape, grid, strides)

                    if h_off == 0:
                        denoised_pretile = tiles_out_tensor
                    else:
                        denoised_pretile[:,:,h_off:-h_off, w_off:-w_off] = tiles_out_tensor
                    denoised_embed = rearrange(denoised_pretile, "b c h w -> b (h w) c", h=h_len, w=w_len)

                elif freqsep_lowpass_method is not None and freqsep_lowpass_method.endswith("pw"): #EO("adain_pw"):

                    denoised_spatial = rearrange(denoised_embed, "b (h w) c -> b c h w", h=h_len, w=w_len)
                    y0_adain_spatial = rearrange(y0_adain_embed, "b (h w) c -> b c h w", h=h_len, w=w_len)

                    if   freqsep_lowpass_method == "median_pw":
                        denoised_spatial_new = adain_patchwise_row_batch_med(denoised_spatial.clone(), y0_adain_spatial.clone().repeat(denoised_spatial.shape[0],1,1,1), sigma=freqsep_sigma, kernel_size=freqsep_kernel_size, use_median_blur=True, lowpass_weight=freqsep_lowpass_weight, highpass_weight=freqsep_highpass_weight)
                    elif freqsep_lowpass_method == "gaussian_pw": 
                        denoised_spatial_new = adain_patchwise_row_batch(denoised_spatial.clone(), y0_adain_spatial.clone().repeat(denoised_spatial.shape[0],1,1,1), sigma=freqsep_sigma, kernel_size=freqsep_kernel_size)
                    
                    denoised_embed = rearrange(denoised_spatial_new, "b c h w -> b (h w) c", h=h_len, w=w_len)

                elif freqsep_lowpass_method is not None: 
                    denoised_spatial = rearrange(denoised_embed, "b (h w) c -> b c h w", h=h_len, w=w_len)
                    y0_adain_spatial = rearrange(y0_adain_embed, "b (h w) c -> b c h w", h=h_len, w=w_len)
                    
                    if   freqsep_lowpass_method == "median":
                        denoised_spatial_LP = median_blur_2d(denoised_spatial, kernel_size=freqsep_kernel_size)
                        y0_adain_spatial_LP = median_blur_2d(y0_adain_spatial, kernel_size=freqsep_kernel_size)
                    elif freqsep_lowpass_method == "gaussian":
                        denoised_spatial_LP = gaussian_blur_2d(denoised_spatial, sigma=freqsep_sigma, kernel_size=freqsep_kernel_size)
                        y0_adain_spatial_LP = gaussian_blur_2d(y0_adain_spatial, sigma=freqsep_sigma, kernel_size=freqsep_kernel_size)
                    
                    denoised_spatial_HP = denoised_spatial - denoised_spatial_LP
                    
                    if EO("adain_fs_uhp"):
                        y0_adain_spatial_HP = y0_adain_spatial - y0_adain_spatial_LP
                        
                        denoised_spatial_ULP = gaussian_blur_2d(denoised_spatial, sigma=EO("adain_fs_uhp_sigma", 1.0), kernel_size=EO("adain_fs_uhp_kernel_size", 3))
                        y0_adain_spatial_ULP = gaussian_blur_2d(y0_adain_spatial, sigma=EO("adain_fs_uhp_sigma", 1.0), kernel_size=EO("adain_fs_uhp_kernel_size", 3))
                        
                        denoised_spatial_UHP = denoised_spatial_HP  - denoised_spatial_ULP
                        y0_adain_spatial_UHP = y0_adain_spatial_HP  - y0_adain_spatial_ULP
                        
                        #denoised_spatial_HP  = y0_adain_spatial_ULP + denoised_spatial_UHP
                        denoised_spatial_HP  = denoised_spatial_ULP + y0_adain_spatial_UHP
                    
                    denoised_spatial_new = freqsep_lowpass_weight * y0_adain_spatial_LP + freqsep_highpass_weight * denoised_spatial_HP
                    denoised_embed = rearrange(denoised_spatial_new, "b c h w -> b (h w) c", h=h_len, w=w_len)

                else:
                    denoised_embed = adain_seq_inplace(denoised_embed, y0_adain_embed)
                    
                for adain_iter in range(EO("style_iter", 0)):
                    denoised_embed = adain_seq_inplace(denoised_embed, y0_adain_embed)
                    denoised_embed = self.Retrojector.embed(self.Retrojector.unembed(denoised_embed))
                    denoised_embed = adain_seq_inplace(denoised_embed, y0_adain_embed)
                    
            elif transformer_options['y0_style_method'] == "WCT":
                self.StyleWCT.set(y0_adain_embed)
                denoised_embed = self.StyleWCT.get(denoised_embed)
                
                if transformer_options.get('y0_standard_guide') is not None:
                    y0_standard_guide = transformer_options.get('y0_standard_guide')
                    
                    y0_standard_guide_embed = self.Retrojector.embed(y0_standard_guide)
                    f_cs = self.StyleWCT.get(y0_standard_guide_embed)
                    self.y0_standard_guide = self.Retrojector.unembed(f_cs)

                if transformer_options.get('y0_inv_standard_guide') is not None:
                    y0_inv_standard_guide = transformer_options.get('y0_inv_standard_guide')

                    y0_inv_standard_guide_embed = self.Retrojector.embed(y0_inv_standard_guide)
                    f_cs = self.StyleWCT.get(y0_inv_standard_guide_embed)
                    self.y0_inv_standard_guide = self.Retrojector.unembed(f_cs)

            elif transformer_options['y0_style_method'] == "WCT2":
                self.WaveletStyleWCT.set(y0_adain_embed, h_len, w_len)
                denoised_embed = self.WaveletStyleWCT.get(denoised_embed, h_len, w_len)
                
                if transformer_options.get('y0_standard_guide') is not None:
                    y0_standard_guide = transformer_options.get('y0_standard_guide')
                    
                    y0_standard_guide_embed = self.Retrojector.embed(y0_standard_guide)
                    f_cs = self.WaveletStyleWCT.get(y0_standard_guide_embed, h_len, w_len)
                    self.y0_standard_guide = self.Retrojector.unembed(f_cs)

                if transformer_options.get('y0_inv_standard_guide') is not None:
                    y0_inv_standard_guide = transformer_options.get('y0_inv_standard_guide')

                    y0_inv_standard_guide_embed = self.Retrojector.embed(y0_inv_standard_guide)
                    f_cs = self.WaveletStyleWCT.get(y0_inv_standard_guide_embed, h_len, w_len)
                    self.y0_inv_standard_guide = self.Retrojector.unembed(f_cs)

            denoised_approx = self.Retrojector.unembed(denoised_embed)
            
            eps = (x - denoised_approx) / sigma

            if not UNCOND:
                if eps.shape[0] == 2:
                    eps[1] = eps_orig[1] + y0_style_pos_weight * (eps[1] - eps_orig[1])
                    eps[0] = eps_orig[0] + y0_style_pos_synweight * (eps[0] - eps_orig[0])
                else:
                    eps[0] = eps_orig[0] + y0_style_pos_weight * (eps[0] - eps_orig[0])
            elif eps.shape[0] == 1 and UNCOND:
                eps[0] = eps_orig[0] + y0_style_pos_synweight * (eps[0] - eps_orig[0])
            
            #eps = eps.float()
        
        if y0_style_neg is not None:
            y0_style_neg_weight    = transformer_options.get("y0_style_neg_weight")
            y0_style_neg_synweight = transformer_options.get("y0_style_neg_synweight")
            y0_style_neg_synweight *= y0_style_neg_weight
            y0_style_neg_mask = transformer_options.get("y0_style_neg_mask")
            y0_style_neg_mask_edge = transformer_options.get("y0_style_neg_mask_edge")
            
            y0_style_neg = y0_style_neg.to(dtype)
            x   = x_orig.to(dtype)
            eps = eps.to(dtype)
            eps_orig = eps.clone()
            
            sigma = SIGMA #t_orig[0].to(torch.float32) / 1000
            denoised = x - sigma * eps

            denoised_embed = self.Retrojector.embed(denoised)
            y0_adain_embed = self.Retrojector.embed(y0_style_neg)
            
            if transformer_options['y0_style_method'] == "scattersort":
                tile_h, tile_w = transformer_options.get('y0_style_tile_height'), transformer_options.get('y0_style_tile_width')
                pad = transformer_options.get('y0_style_tile_padding')
                if pad is not None and tile_h is not None and tile_w is not None:
                    
                    denoised_spatial = rearrange(denoised_embed, "b (h w) c -> b c h w", h=h_len, w=w_len)
                    y0_adain_spatial = rearrange(y0_adain_embed, "b (h w) c -> b c h w", h=h_len, w=w_len)
                    
                    denoised_spatial = apply_scattersort_tiled(denoised_spatial, y0_adain_spatial, tile_h, tile_w, pad)
                    
                    denoised_embed = rearrange(denoised_spatial, "b c h w -> b (h w) c")

                else:
                    denoised_embed = apply_scattersort_masked(denoised_embed, y0_adain_embed, y0_style_neg_mask, y0_style_neg_mask_edge, h_len, w_len)
            
            
            elif transformer_options['y0_style_method'] == "AdaIN":
                denoised_embed = adain_seq_inplace(denoised_embed, y0_adain_embed)
                for adain_iter in range(EO("style_iter", 0)):
                    denoised_embed = adain_seq_inplace(denoised_embed, y0_adain_embed)
                    denoised_embed = self.Retrojector.embed(self.Retrojector.unembed(denoised_embed))
                    denoised_embed = adain_seq_inplace(denoised_embed, y0_adain_embed)
                    
            elif transformer_options['y0_style_method'] == "WCT":
                self.StyleWCT.set(y0_adain_embed)
                denoised_embed = self.StyleWCT.get(denoised_embed)

            elif transformer_options['y0_style_method'] == "WCT2":
                self.WaveletStyleWCT.set(y0_adain_embed, h_len, w_len)
                denoised_embed = self.WaveletStyleWCT.get(denoised_embed, h_len, w_len)

            denoised_approx = self.Retrojector.unembed(denoised_embed)

            if UNCOND:
                eps = (x - denoised_approx) / sigma
                eps[0] = eps_orig[0] + y0_style_neg_weight * (eps[0] - eps_orig[0])
                if eps.shape[0] == 2:
                    eps[1] = eps_orig[1] + y0_style_neg_synweight * (eps[1] - eps_orig[1])
            elif eps.shape[0] == 1 and not UNCOND:
                eps[0] = eps_orig[0] + y0_style_neg_synweight * (eps[0] - eps_orig[0])
            
            #eps = eps.float()
        
        if EO("model_eps_out"):
            self.eps_out = eps.clone()
        return eps
    

    def expand_timesteps(self, t, batch_size, device):
        if not torch.is_tensor(t):
            is_mps = device.type == "mps"
            if isinstance(t, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32   if is_mps else torch.int64
            t = Tensor([t], dtype=dtype, device=device)
        elif len(t.shape) == 0:
            t = t[None].to(device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        t = t.expand(batch_size)
        return t

def clone_inputs(*args, index: int=None):

    if index is None:
        return tuple(x.clone() for x in args)
    else:
        return tuple(x[index].unsqueeze(0).clone() for x in args)



