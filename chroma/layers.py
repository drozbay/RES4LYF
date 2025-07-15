import torch
from torch import Tensor, nn

#from comfy.ldm.flux.math import attention
from comfy.ldm.flux.layers import (
    MLPEmbedder,
    RMSNorm,
    QKNorm,
    SelfAttention,
    ModulationOut,
)
from typing import Optional, Callable, Tuple, Dict, Any, Union, TYPE_CHECKING, TypeVar

from .math import attention, rope, apply_rope

class ChromaModulationOut(ModulationOut):
    @classmethod
    def from_offset(cls, tensor: torch.Tensor, offset: int = 0) -> ModulationOut:
        return cls(
            shift=tensor[:, offset : offset + 1, :],
            scale=tensor[:, offset + 1 : offset + 2, :],
            gate=tensor[:, offset + 2 : offset + 3, :],
        )




class Approximator(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, n_layers = 5, dtype=None, device=None, operations=None):
        super().__init__()
        self.in_proj = operations.Linear(in_dim, hidden_dim, bias=True, dtype=dtype, device=device)
        self.layers = nn.ModuleList([MLPEmbedder(hidden_dim, hidden_dim, dtype=dtype, device=device, operations=operations) for x in range( n_layers)])
        self.norms = nn.ModuleList([RMSNorm(hidden_dim, dtype=dtype, device=device, operations=operations) for x in range( n_layers)])
        self.out_proj = operations.Linear(hidden_dim, out_dim, dtype=dtype, device=device)

    @property
    def device(self):
        # Get the device of the module (assumes all parameters are on the same device)
        return next(self.parameters()).device

    def forward(self, x: Tensor) -> Tensor:
        x = self.in_proj(x)

        for layer, norms in zip(self.layers, self.norms):
            x = x + layer(norms(x))

        x = self.out_proj(x)

        return x


class ReChromaDoubleStreamBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, qkv_bias: bool = False, flipped_img_txt=False, dtype=None, device=None, operations=None):
        super().__init__()

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.img_norm1 = operations.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        self.img_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias, dtype=dtype, device=device, operations=operations)

        self.img_norm2 = operations.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        self.img_mlp = nn.Sequential(
            operations.Linear(hidden_size, mlp_hidden_dim, bias=True, dtype=dtype, device=device),
            nn.GELU(approximate="tanh"),
            operations.Linear(mlp_hidden_dim, hidden_size, bias=True, dtype=dtype, device=device),
        )

        self.txt_norm1 = operations.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        self.txt_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias, dtype=dtype, device=device, operations=operations)

        self.txt_norm2 = operations.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        self.txt_mlp = nn.Sequential(
            operations.Linear(hidden_size, mlp_hidden_dim, bias=True, dtype=dtype, device=device),
            nn.GELU(approximate="tanh"),
            operations.Linear(mlp_hidden_dim, hidden_size, bias=True, dtype=dtype, device=device),
        )
        self.flipped_img_txt = flipped_img_txt


    def forward(self, img: Tensor, txt: Tensor, vec: Tensor, pe: Tensor, mask=None, idx=0, update_cross_attn=None, style_block=None) -> Tuple[Tensor, Tensor]: # vec 1,3072  # vec 1,3072    #mask.shape 4608,4608  #img_attn.shape 1,4096,3072    txt_attn.shape 1,512,3072
        (img_mod1, img_mod2), (txt_mod1, txt_mod2) = vec


        img_len = img.shape[-2]
        txt_len = txt.shape[-2]

        #img_mod1, img_mod2  = self.img_mod(vec) # -> 3072, 3072
        #txt_mod1, txt_mod2  = self.txt_mod(vec)
        
        img_norm = self.img_norm1(img)
        txt_norm = self.txt_norm1(txt)
        
        img_norm = style_block.img(img_norm, "attn_norm")
        txt_norm = style_block.txt(txt_norm, "attn_norm")

        img_norm = img_norm * (1+img_mod1.scale) + img_mod1.shift
        txt_norm = txt_norm * (1+txt_mod1.scale) + txt_mod1.shift

        img_norm = style_block.img(img_norm, "attn_norm_mod")
        txt_norm = style_block.txt(txt_norm, "attn_norm_mod")
        
        
        
        ### ATTN ###
        img_qkv             = self.img_attn.qkv(img_norm)
        img_q, img_k, img_v = img_qkv.view(img_qkv.shape[0], img_qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        
        img_q = style_block.img.ATTN(img_q, "q_proj")
        img_k = style_block.img.ATTN(img_k, "k_proj")
        img_v = style_block.img.ATTN(img_v, "v_proj")
        
        img_q, img_k        = self.img_attn.norm(img_q, img_k, img_v)
        
        img_q = style_block.img.ATTN(img_q, "q_norm")
        img_k = style_block.img.ATTN(img_k, "k_norm")
        
        txt_qkv             = self.txt_attn.qkv(txt_norm)
        txt_q, txt_k, txt_v = txt_qkv.view(txt_qkv.shape[0], txt_qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        
        txt_q = style_block.txt.ATTN(txt_q, "q_proj")
        txt_k = style_block.txt.ATTN(txt_k, "k_proj")
        txt_v = style_block.txt.ATTN(txt_v, "v_proj")
        
        txt_q, txt_k        = self.txt_attn.norm(txt_q, txt_k, txt_v)
        
        txt_q = style_block.txt.ATTN(txt_q, "q_norm")
        txt_k = style_block.txt.ATTN(txt_k, "k_norm")

        q, k, v = torch.cat((txt_q, img_q), dim=2), torch.cat((txt_k, img_k), dim=2), torch.cat((txt_v, img_v), dim=2)
        attn = attention(q, k, v, pe=pe, mask=mask)
        
        txt_attn = attn[:,:txt_len]                         # 1, 768,3072
        img_attn = attn[:,txt_len:]  
        
        img_attn = style_block.img.ATTN(img_attn, "out")
        txt_attn = style_block.txt.ATTN(txt_attn, "out")
        
        img_attn = self.img_attn.proj(img_attn)    #to_out
        txt_attn = self.txt_attn.proj(txt_attn)
        ### ATTN ###



        img_attn = style_block.img(img_attn, "attn")
        txt_attn = style_block.txt(txt_attn, "attn")
        
        img_attn *= img_mod1.gate
        txt_attn *= txt_mod1.gate
        
        img_attn = style_block.img(img_attn, "attn_gated")
        txt_attn = style_block.txt(txt_attn, "attn_gated")
        
        img += img_attn
        txt += txt_attn
        
        img = style_block.img(img, "attn_res")
        txt = style_block.txt(txt, "attn_res")
        
        
        
        img_norm = self.img_norm2(img)
        txt_norm = self.txt_norm2(txt)
        
        img_norm = style_block.img(img_norm, "ff_norm")
        txt_norm = style_block.txt(txt_norm, "ff_norm")
        
        img_norm = img_norm * (1+img_mod2.scale) + img_mod2.shift
        txt_norm = txt_norm * (1+txt_mod2.scale) + txt_mod2.shift
        
        img_norm = style_block.img(img_norm, "ff_norm_mod")
        txt_norm = style_block.txt(txt_norm, "ff_norm_mod")
        
        img_mlp = self.img_mlp(img_norm)
        txt_mlp = self.txt_mlp(txt_norm)
        
        img_mlp = style_block.img(img_mlp, "ff")
        txt_mlp = style_block.txt(txt_mlp, "ff")
        
        img_mlp *= img_mod2.gate
        txt_mlp *= txt_mod2.gate
        
        img_mlp = style_block.img(img_mlp, "ff_gated")
        txt_mlp = style_block.txt(txt_mlp, "ff_gated")
        
        img += img_mlp
        txt += txt_mlp
        
        img = style_block.img(img, "ff_res")
        txt = style_block.txt(txt, "ff_res")

        if update_cross_attn is not None:
            if not update_cross_attn['skip_cross_attn']:
                UNCOND      = update_cross_attn['UNCOND']
                
                txt_update = self.txt_norm1(txt.cpu()).float()
                txt_update = (1 + txt_mod1.scale.to(txt_update)) * txt_update + txt_mod1.shift.to(txt_update)
                
                if UNCOND:
                    t5_start    = update_cross_attn['src_t5_start']
                    t5_end      = update_cross_attn['src_t5_end']
                
                    txt_src    = txt_update[:,t5_start:t5_end,:].cpu() #.float()
                    self.c_src = txt_src.transpose(-2,-1).squeeze(0)    # shape [C,1]
                else:
                    t5_start    = update_cross_attn['tgt_t5_start']
                    t5_end      = update_cross_attn['tgt_t5_end']
                    
                    lamb  = update_cross_attn['lamb']
                    erase = update_cross_attn['erase']

                    c_guide   = txt_update[:,t5_start:t5_end,:].transpose(-2,-1).squeeze(0)  # [C,1]
                    
                    Wv_old       = self.txt_attn.qkv.weight.data.to(c_guide)              # [C,C]

                    v_star       = Wv_old @ c_guide                             # [C,1]

                    c_src        = self.c_src  #.cpu()                                   # [C,1]

                    lamb         = lamb
                    erase_scale  = erase
                    d            = c_src.shape[0]

                    C            = c_src @ c_src.T                              # [C,C]
                    I            = torch.eye(d, device=C.device, dtype=C.dtype)

                    mat1_v       = lamb*Wv_old + erase_scale*(v_star @ c_src.T)     # [C,C]
                    mat2_v       = lamb*I      + erase_scale*(C)                    # [C,C]
                    
                    I       = I.to("cpu")
                    C       = C.to("cpu")
                    c_src   = c_src.to("cpu")
                    self.c_src   = self.c_src.to("cpu")
                    v_star  = v_star.to("cpu")
                    Wv_old  = Wv_old.to("cpu")
                    c_guide = c_guide.to("cpu")
                    del I, C, c_src, self.c_src, v_star, Wv_old, c_guide

                    #Wv_new       = mat1_v @ torch.inverse(mat2_v.float()).to(mat1_v)                   # [C,C]
                    Wv_new = torch.linalg.solve(mat2_v.T, mat1_v.T).T

                    mat1_v = mat1_v.to("cpu")
                    mat2_v = mat2_v.to("cpu")
                    del mat1_v, mat2_v

                    update_q = update_cross_attn['update_q']
                    update_k = update_cross_attn['update_k']
                    update_v = update_cross_attn['update_v']
                    
                    if not update_q:
                        Wv_new[:3072,    :] = self.txt_attn.qkv.weight.data[:3072,    :].to(Wv_new)
                    if not update_k:
                        Wv_new[3072:6144,:] = self.txt_attn.qkv.weight.data[3072:6144,:].to(Wv_new)
                    if not update_v:
                        Wv_new[6144:    ,:] = self.txt_attn.qkv.weight.data[6144:    ,:].to(Wv_new)
                    
                    self.txt_attn.qkv.weight.data.copy_(Wv_new.to(self.txt_attn.qkv.weight.data.dtype))
                    
                    Wv_new = Wv_new.to("cpu")
                    del Wv_new
                    #torch.cuda.empty_cache()
                    
        return img, txt
        



    def forward_blah(self, img: Tensor, txt: Tensor, pe: Tensor, vec: Tensor, attn_mask=None):
        (img_mod1, img_mod2), (txt_mod1, txt_mod2) = vec

        # prepare image for attention
        img_modulated = self.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = self.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = img_qkv.view(img_qkv.shape[0], img_qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        # prepare txt for attention
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = txt_qkv.view(txt_qkv.shape[0], txt_qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        # run actual attention
        attn = attention(torch.cat((txt_q, img_q), dim=2),
                         torch.cat((txt_k, img_k), dim=2),
                         torch.cat((txt_v, img_v), dim=2),
                         pe=pe, mask=attn_mask)

        txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]

        # calculate the img bloks
        img = img + img_mod1.gate * self.img_attn.proj(img_attn)
        img = img + img_mod2.gate * self.img_mlp((1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift)

        # calculate the txt bloks
        txt += txt_mod1.gate * self.txt_attn.proj(txt_attn)
        txt += txt_mod2.gate * self.txt_mlp((1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift)

        if txt.dtype == torch.float16:
            txt = torch.nan_to_num(txt, nan=0.0, posinf=65504, neginf=-65504)

        return img, txt


class ReChromaSingleStreamBlock(nn.Module):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_scale: float = None,
        dtype=None,
        device=None,
        operations=None
    ):
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # qkv and mlp_in
        self.linear1 = operations.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim, dtype=dtype, device=device)
        # proj and mlp_out
        self.linear2 = operations.Linear(hidden_size + self.mlp_hidden_dim, hidden_size, dtype=dtype, device=device)

        self.norm = QKNorm(head_dim, dtype=dtype, device=device, operations=operations)

        self.hidden_size = hidden_size
        self.pre_norm = operations.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)

        self.mlp_act = nn.GELU(approximate="tanh")


    # vec 1,3072    x 1,9984,3072
    def forward(self, img: Tensor, vec: Tensor, pe: Tensor, mask=None, idx=0, style_block=None) -> Tensor:   # x 1,9984,3072 if 2 reg embeds, 1,9472,3072 if none    # 9216x4096 = 16x1536x1536
        #mod, _    = self.modulation(vec)
        mod = vec
        
        img_norm = self.pre_norm(img)
        img_norm = style_block.img(img_norm, "attn_norm")
        
        img_norm  = (1 + mod.scale) * img_norm + mod.shift   # mod => vec
        img_norm = style_block.img(img_norm, "attn_norm_mod")
        
        
        
        ### ATTN ###
        qkv, mlp = torch.split(self.linear1(img_norm), [3*self.hidden_size, self.mlp_hidden_dim], dim=-1)

        q, k, v  = qkv.view(qkv.shape[0], qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)     #q, k, v  = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        
        q = style_block.img.ATTN(q, "q_proj")
        k = style_block.img.ATTN(k, "k_proj")
        v = style_block.img.ATTN(v, "v_proj")
        
        q, k     = self.norm(q, k, v)
        
        q = style_block.img.ATTN(q, "q_norm")
        k = style_block.img.ATTN(k, "k_norm")
        
        attn = attention(q, k, v, pe=pe, mask=mask)
        attn = style_block.img.ATTN(attn, "out")
        ### ATTN ###



        mlp = style_block.img(mlp, "ff_norm")

        mlp_act = self.mlp_act(mlp)
        mlp_act = style_block.img(mlp_act, "ff_norm_mod")

        img_ff_i  = self.linear2(torch.cat((attn, mlp_act), 2))   # effectively FF smooshed into one line
        img_ff_i = style_block.img(img_ff_i, "ff")

        img_ff_i *= mod.gate
        img_ff_i = style_block.img(img_ff_i, "ff_gated")

        img      += img_ff_i
        img = style_block.img(img, "ff_res")
        
        return img


    def forward_blah(self, x: Tensor, pe: Tensor, vec: Tensor, attn_mask=None) -> Tensor:
        mod = vec
        x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
        qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)

        q, k, v = qkv.view(qkv.shape[0], qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k = self.norm(q, k, v)

        # compute attention
        attn = attention(q, k, v, pe=pe, mask=attn_mask)
        # compute activation in mlp stream, cat again and run second linear layer
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        x += mod.gate * output
        if x.dtype == torch.float16:
            x = torch.nan_to_num(x, nan=0.0, posinf=65504, neginf=-65504)
        return x


class LastLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int, dtype=None, device=None, operations=None):
        super().__init__()
        self.norm_final = operations.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        self.linear = operations.Linear(hidden_size, out_channels, bias=True, dtype=dtype, device=device)

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        shift, scale = vec
        shift = shift.squeeze(1)
        scale = scale.squeeze(1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.linear(x)
        return x
