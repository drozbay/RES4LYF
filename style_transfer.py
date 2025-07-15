
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor, FloatTensor
from typing import Optional, Callable, Tuple, Dict, List, Any, Union

import einops 
from einops import rearrange
import copy
import comfy

from torch_dct import dct, idct
from torch_dct import dct_2d, idct_2d  # requires torch_dct >= 0.1.5
from .latents import gaussian_blur_2d, median_blur_2d

# WIP... not yet in use...
class StyleTransfer:  
    def __init__(self,
        style_method  = "WCT",
        embedder_method = None,
        patch_size    = 1,
        pinv_dtype    = torch.float64,
        dtype         = torch.float64,
    ):
        self.style_method  = style_method
        
        self.embedder_method   = None
        self.unembedder_method = None

        if embedder_method is not None:
            self.set_embedder_method(embedder_method)
        
        self.patch_size    = patch_size
        
        #if embedder_type == "conv2d":
        #    self.unembedder = self.invert_conv2d
        self.pinv_dtype = pinv_dtype
        self.dtype      = dtype
        
        self.patchify   = None
        self.unpatchify = None
        
        self.orig_shape = None
        self.grid_sizes = None
        
        #self.x_embed_ndim = 0
        
        

    def set_patchify_method(self, patchify_method=None):
        self.patchify_method = patchify_method

    def set_unpatchify_method(self, unpatchify_method=None):
        self.unpatchify_method = unpatchify_method
        
    def set_embedder_method(self, embedder_method):
        self.embedder_method = copy.deepcopy(embedder_method).to(self.pinv_dtype)
        self.W = self.embedder_method.weight
        self.B = self.embedder_method.bias    
        
        if   isinstance(embedder_method, nn.Linear):
            self.unembedder_method = self.invert_linear
        
        elif isinstance(embedder_method, nn.Conv2d):
            self.unembedder_method = self.invert_conv2d
            
        elif isinstance(embedder_method, nn.Conv3d):
            self.unembedder_method = self.invert_conv3d
            
    def set_patch_size(self, patch_size):
        self.patch_size = patch_size

    def unpatchify(self, x: Tensor) -> List[Tensor]:
        x_arr = []
        for i, img_size in enumerate(self.img_sizes):   #  [[64,64]]   , img_sizes: List[Tuple[int, int]]
            pH, pW = img_size
            x_arr.append(
                einops.rearrange(x[i, :pH*pW].reshape(1, pH, pW, -1), 'B H W (p1 p2 C) -> B C (H p1) (W p2)',
                    p1=self.patch_size, p2=self.patch_size)
            )
        x = torch.cat(x_arr, dim=0)
        return x

    def patchify(self, x: Tensor):
        x = comfy.ldm.common_dit.pad_to_patch_size(x, (self.patch_size, self.patch_size))
        
        pH, pW         = x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size
        self.img_sizes = [[pH, pW]] * x.shape[0]
        x              = einops.rearrange(x, 'B C (H p1) (W p2) -> B (H W) (p1 p2 C)', p1=self.patch_size, p2=self.patch_size)
        return x
        
        
    def embedder(self, x):
        if isinstance(self.embedder_method, nn.Linear):
            x = self.patchify(x)
        
        self.orig_shape = x.shape
        x = self.embedder_method(x)
        self.grid_sizes = x.shape[2:]
        
        #self.x_embed_ndim = x.ndim
        #if x.ndim > 3:
        #    x = einops.rearrange(x, "B C H W -> B (H W) C")
        
        return x
        
    def unembedder(self, x):
        #if self.x_embed_ndim > 3:
        #    x = einops.rearrange(x, "B (H W) C -> B C H W", W=self.orig_shape[-1])
        
        x = self.unembedder_method(x)
        return x
        
        
    def invert_linear(self, x : torch.Tensor,) -> torch.Tensor:
        x = x.to(self.pinv_dtype)
        #x = (x - self.B.to(self.dtype)) @ torch.linalg.pinv(self.W.to(self.pinv_dtype)).T.to(self.dtype)
        x = (x - self.B) @ torch.linalg.pinv(self.W).T
        
        return x.to(self.dtype)

        
        
    def invert_conv2d(self, z: torch.Tensor,) -> torch.Tensor:
        z = z.to(self.pinv_dtype)
        conv = self.embedder_method
        
        B, C_in, H, W      = self.orig_shape
        C_out, _, kH, kW   = conv.weight.shape
        stride_h, stride_w = conv.stride
        pad_h,    pad_w    = conv.padding

        b = conv.bias.view(1, C_out, 1, 1).to(z)
        z_nobias = z - b

        W_flat = conv.weight.view(C_out, -1).to(z)  
        W_pinv = torch.linalg.pinv(W_flat)    

        Bz, Co, Hp, Wp = z_nobias.shape
        z_flat = z_nobias.reshape(Bz, Co, -1)  

        x_patches = W_pinv @ z_flat   

        x_sum = F.fold(
            x_patches,
            output_size=(H + 2*pad_h, W + 2*pad_w),
            kernel_size=(kH, kW),
            stride=(stride_h, stride_w),
        )
        ones = torch.ones_like(x_patches)
        count = F.fold(
            ones,
            output_size=(H + 2*pad_h, W + 2*pad_w),
            kernel_size=(kH, kW),
            stride=(stride_h, stride_w),
        )  

        x_recon = x_sum / count.clamp(min=1e-6)
        if pad_h > 0 or pad_w > 0:
            x_recon = x_recon[..., pad_h:pad_h+H, pad_w:pad_w+W]

        return x_recon.to(self.dtype)



    def invert_conv3d(self, z: torch.Tensor, ) -> torch.Tensor:
        z = z.to(self.pinv_dtype)
        conv = self.embedder_method
        grid_sizes = self.grid_sizes

        B, C_in, D, H, W = self.orig_shape
        pD, pH, pW = self.patch_size
        sD, sH, sW = pD, pH, pW

        if z.ndim == 3:
            # [B, S, C_out] -> reshape to [B, C_out, D', H', W']   
            S = z.shape[1]
            if grid_sizes is None:
                Dp = D // pD
                Hp = H // pH   # getting actual patchified dims
                Wp = W // pW
            else:
                Dp, Hp, Wp = grid_sizes
            C_out = z.shape[2]
            z = z.transpose(1, 2).reshape(B, C_out, Dp, Hp, Wp)
        else:
            B2, C_out, Dp, Hp, Wp = z.shape
            assert B2 == B, "Batch size mismatch... ya sharked it."

        b = conv.bias.view(1, C_out, 1, 1, 1)         # need to kncokout bias to invert via weight
        z_nobias = z - b

        # 2D filter -> pinv
        w3 = conv.weight         # [C_out, C_in, 1, pH, pW]
        w2 = w3.squeeze(2)                       # [C_out, C_in, pH, pW]
        out_ch, in_ch, kH, kW = w2.shape
        W_flat = w2.view(out_ch, -1)            # [C_out, in_ch*pH*pW]
        W_pinv = torch.linalg.pinv(W_flat)      # [in_ch*pH*pW, C_out]

        # merge depth for 2D unfold wackiness
        z2 = z_nobias.permute(0,2,1,3,4).reshape(B*Dp, C_out, Hp, Wp)

        # apply pinv ... get patch vectors
        z_flat    = z2.reshape(B*Dp, C_out, -1)  # [B*Dp, C_out, L]
        x_patches = W_pinv @ z_flat              # [B*Dp, in_ch*pH*pW, L]

        # fold -> restore spatial frames
        x2 = F.fold(
            x_patches,
            output_size=(H, W),
            kernel_size=(pH, pW),
            stride=(sH, sW)
        )  # → [B*Dp, C_in, H, W]

        # unmerge depth (de-depth charge)
        x2 = x2.reshape(B, Dp, in_ch, H, W)           # [B, Dp,  C_in, H, W]
        x_recon = x2.permute(0,2,1,3,4).contiguous()  # [B, C_in,   D, H, W]
        return x_recon.to(self.dtype)



    def adain_seq_inplace(self, content: torch.Tensor, style: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
        mean_c = content.mean(1, keepdim=True)
        std_c  = content.std (1, keepdim=True).add_(eps) 
        mean_s = style.mean  (1, keepdim=True)
        std_s  = style.std   (1, keepdim=True).add_(eps)

        content.sub_(mean_c).div_(std_c).mul_(std_s).add_(mean_s)
        return content






class StyleWCT:  
    def __init__(self, dtype=torch.float64, use_svd=False,):
        self.dtype          = dtype
        self.use_svd        = use_svd
        self.y0_adain_embed = None
        self.mu_s           = None
        self.y0_color       = None
        self.spatial_shape  = None
        
    def whiten(self, f_s_centered: torch.Tensor, set=False):
        cov = (f_s_centered.T.double() @ f_s_centered.double()) / (f_s_centered.size(0) - 1)

        if self.use_svd:
            cov = cov.to(torch.float32)
            U_svd, S_svd, Vh_svd = torch.linalg.svd(cov)
            #U_svd, S_svd, Vh_svd = torch.linalg.svd(cov + 1e-5 * torch.eye(cov.size(0), dtype=cov.dtype, device=cov.device))
            S_eig = S_svd.to(torch.float64)
            U_eig = U_svd.to(torch.float64)
        else:
            cov = cov + 1e-5 * torch.eye(cov.size(0), dtype=cov.dtype, device=cov.device)
            #cov = cov.to(torch.float32)
            #S_eig, U_eig = torch.linalg.eigh(cov + 1e-4 * torch.eye(cov.size(0), dtype=cov.dtype, device=cov.device))
            S_eig, U_eig = torch.linalg.eigh(cov)
            S_eig = S_eig.to(torch.float64)
            U_eig = U_eig.to(torch.float64)
        
        if set:
            S_eig_root = S_eig.clamp(min=0).sqrt() # eigenvalues -> singular values
        else:
            S_eig_root = S_eig.clamp(min=0).rsqrt() # inverse square root
        
        whiten = U_eig @ torch.diag(S_eig_root) @ U_eig.T    # eigenvector @ diagonal eigenvalues @ eigenvectors.T
        return whiten.to(f_s_centered)

    def set(self, y0_adain_embed: torch.Tensor, spatial_shape=None):
        if self.y0_color is None or (self.y0_adain_embed is None or self.y0_adain_embed.shape != y0_adain_embed.shape or torch.norm(self.y0_adain_embed - y0_adain_embed) > 0):
            self.y0_adain_embed = y0_adain_embed.clone()
            if spatial_shape is not None:
                self.spatial_shape = spatial_shape
            
            f_s          = y0_adain_embed[0] # if y0_adain_embed.ndim > 4 else y0_adain_embed
            self.mu_s    = f_s.mean(dim=0, keepdim=True)
            f_s_centered = f_s - self.mu_s
            
            self.y0_color = self.whiten(f_s_centered, set=True)
            
    def get(self, denoised_embed: torch.Tensor):
        for wct_i in range(denoised_embed.shape[0]):
            f_c          = denoised_embed[wct_i]
            mu_c         = f_c.mean(dim=0, keepdim=True)
            f_c_centered = f_c - mu_c

            whiten = self.whiten(f_c_centered)

            f_c_whitened = f_c_centered @ whiten.T
            f_cs         = f_c_whitened @ self.y0_color.T + self.mu_s
            
            denoised_embed[wct_i] = f_cs
            
        return denoised_embed


class StyleWCT_barfalotz:
    def __init__(self, dtype=torch.float32):
        self.dtype = dtype
        self.mu_s = None
        self.color_transform = None

    def _compute_whitening(self, f_s_centered: torch.Tensor, set=False):
        eps = 1e-5
        cov = (f_s_centered.T @ f_s_centered) / (f_s_centered.size(0) - 1)

        # Diagonal regularization
        cov += eps * torch.eye(cov.size(0), dtype=cov.dtype, device=cov.device)

        # Cholesky decomposition
        try:
            L = torch.linalg.cholesky(cov)
        except RuntimeError as e:
            raise RuntimeError("Cholesky failed, consider increasing eps or using SVD") from e

        if set:
            # Coloring transform (square root of cov):  A @ L
            transform = L
        else:
            # Whitening transform: A @ L⁻¹.T
            L_inv = torch.cholesky_inverse(L)
            transform = L_inv.T  # (L⁻¹).T

        return transform.to(f_s_centered)

    def _compute_coloring(self, f: torch.Tensor, eps=1e-5):
        """
        f: [B, N, C] — features already mean-centered
        returns: [B, C, C] coloring matrices
        """
        B, N, C = f.shape
        cov = torch.matmul(f.transpose(1, 2), f) / (N - 1)      # [B, C, C]
        cov += eps * torch.eye(C, device=f.device, dtype=f.dtype).unsqueeze(0)

        L = torch.linalg.cholesky(cov)                          # [B, C, C]
        return L  # acts as "coloring" since white → color = L @ white

    def set(self, y0_adain_embed: torch.Tensor):
        """
        y0_adain_embed: [B=1, N, C] — reference style features
        """
        y0 = y0_adain_embed.to(self.dtype)
        self.mu_s = y0.mean(dim=1, keepdim=True)               # [1, 1, C]
        f_s_centered = y0 - self.mu_s                          # [1, N, C]
        self.color_transform = self._compute_coloring(f_s_centered)

    def get(self, denoised_embed: torch.Tensor):
        """
        denoised_embed: [B, N, C] — content features to transform
        returns: styled features
        """
        x = denoised_embed.to(self.dtype)
        mu_c = x.mean(dim=1, keepdim=True)                     # [B, 1, C]
        x_centered = x - mu_c                                  # [B, N, C]

        whiten = self._compute_whitening(x_centered)           # [B, C, C]
        x_white = torch.matmul(x_centered, whiten.transpose(1, 2))  # [B, N, C]

        B = x.shape[0]
        color = self.color_transform.expand(B, -1, -1)         # [B, C, C]
        x_colored = torch.matmul(x_white, color.transpose(1, 2))  # [B, N, C]
        x_final = x_colored + self.mu_s                        # [B, N, C]

        return x_final



def matrix_inv_sqrt_newton(A, n_iter=6, eps=1e-5):
    """
    Approximate A^{-1/2} via Newton-Schulz on GPU in FP32.
    A: [B,C,C] or [C,C], must be SPD.
    returns: A_inv_sqrt of same shape & dtype.
    """
    # make sure it's float32
    orig_dtype = A.dtype
    A = A.to(torch.float32)
    if A.dim() == 2:
        A = A.unsqueeze(0)   # batch of 1

    B, C, _ = A.shape
    # normalize so ||A|| < 1
    trace = A.diagonal(0, -2, -1).sum(-1).view(B,1,1)
    Y = A / trace
    I = torch.eye(C, device=A.device).unsqueeze(0).expand_as(A)
    Z = torch.eye(C, device=A.device).unsqueeze(0).expand_as(A)

    for _ in range(n_iter):
        T  = 0.5 * (3.*I - Z @ Y)
        Y  = Y @ T
        Z  = T @ Z

    # Z ≈ A^{-1/2} * sqrt(trace)
    A_inv_sqrt = Z / torch.sqrt(trace)
    if orig_dtype != torch.float32:
        A_inv_sqrt = A_inv_sqrt.to(orig_dtype)
    return A_inv_sqrt.squeeze(0) if orig_dtype!=torch.float32 and A.dim()==2 else A_inv_sqrt

def matrix_sqrt_newton(A, n_iter=6, eps=1e-5):
    """Similarly approximate A^{+1/2}."""
    # same normalization trick, but return Y * sqrt(trace)
    orig_dtype = A.dtype
    A = A.to(torch.float32)
    if A.dim() == 2:
        A = A.unsqueeze(0)
    B,C,_ = A.shape
    trace = A.diagonal(0,-2,-1).sum(-1).view(B,1,1)
    Y = A / trace
    I = torch.eye(C, device=A.device).unsqueeze(0).expand_as(A)
    Z = torch.eye(C, device=A.device).unsqueeze(0).expand_as(A)

    for _ in range(n_iter):
        T  = 0.5*(3.*I - Z @ Y)
        Y  = T @ Y
        Z  = Z @ T

    A_sqrt = Y * torch.sqrt(trace)
    if orig_dtype!=torch.float32:
        A_sqrt = A_sqrt.to(orig_dtype)
    return A_sqrt.squeeze(0) if orig_dtype!=torch.float32 and A.dim()==2 else A_sqrt








class StyleWCT_Fast:
    def __init__(self, dtype=torch.float32, use_svd=False, eps=1e-5):
        """
        dtype: do all the linear algebra in this torch dtype (FP64 by default)
        use_svd: if True, use torch.linalg.svd instead of eigh
        """
        self.dtype    = dtype
        self.use_svd  = use_svd
        self.eps      = eps

        # will be set in .set():
        self.mu_s         = None   # [1,1,C]
        self.color_tf     = None   # [C, C]

    def set(self, y0_adain_embed: torch.Tensor, spatial_shape=None):
        """
        y0_adain_embed: [1, N, C]  (reference style features)
        """
        # --- (1) cast & mean-center style features ---
        y0 = y0_adain_embed.to(self.dtype)            # [1,N,C]
        mu_s = y0.mean(dim=1, keepdim=True)           # [1,1,C]
        Xs = y0 - mu_s                                # [1,N,C]
        N, C = Xs.shape[1], Xs.shape[2]

        # --- (2) compute style covariance [C,C] ---
        cov_s = (Xs.transpose(1,2) @ Xs)[0] / (N - 1) # [C,C]
        cov_s = cov_s + self.eps * torch.eye(C, device=cov_s.device, dtype=cov_s.dtype)

        # --- (3) eigendecompose or SVD ---
        #if self.use_svd:
        #    U, S, _ = torch.linalg.svd(cov_s)
        #else:
        #    S, U = torch.linalg.eigh(cov_s)

        ## --- (4) build coloring transform = U · diag(√S) · Uᵀ ---
        #S_sqrt = S.clamp(min=0).sqrt()
        #self.color_tf = (U     @ torch.diag(S_sqrt) @ U.T)  # [C,C]
        self.color_tf = matrix_sqrt_newton(cov_s.to(torch.float32), n_iter=6).to(cov_s.dtype)
        self.mu_s     = mu_s                               # [1,1,C]


    def get(self, denoised_embed: torch.Tensor):
        """
        denoised_embed: [B, N, C] (content features)
        returns:         [B, N, C] style-transferred.
        """
        orig_dtype = denoised_embed.dtype
        x = denoised_embed.to(self.dtype)       # [B,N,C]
        B, N, C = x.shape

        # --- (1) mean-center content ---
        mu_c = x.mean(dim=1, keepdim=True)      # [B,1,C]
        Xc   = x - mu_c                         # [B,N,C]

        # --- (2) content covariance per batch [B,C,C] ---
        cov_c = (Xc.transpose(1,2) @ Xc) / (N - 1)  # [B,C,C]
        cov_c = cov_c + self.eps * torch.eye(C, device=cov_c.device, dtype=cov_c.dtype).unsqueeze(0)

        # --- (3) batch-eigendecompose or SVD ---
        """if self.use_svd:
            # torch.linalg.svd currently doesn’t batch on GPU for all releases,
            # but you could loop if needed. Here we fallback to eigh.
            S_c, U_c = torch.linalg.eigh(cov_c)
        else:
            S_c, U_c = torch.linalg.eigh(cov_c)      # S_c:[B,C], U_c:[B,C,C]

        # --- (4) build whitening transforms: U_c · diag(1/√S_c) · U_cᵀ ---
        inv_sqrt = S_c.clamp(min=1e-12).rsqrt()      # [B,C]
        # diag_embed will broadcast: result [B,C,C]
        W_c = U_c @ torch.diag_embed(inv_sqrt) @ U_c.transpose(-2,-1)  # [B,C,C]"""
        
        W_c = matrix_inv_sqrt_newton(cov_c.to(torch.float32), n_iter=6)
        W_c = W_c.to(cov_c.dtype)

        # --- (5) whiten content ---
        X_white = Xc @ W_c.transpose(-2,-1)         # [B,N,C]

        # --- (6) color with style transform (broadcast style->batch) ---
        C_tf = self.color_tf.to(X_white.dtype).expand(B,-1,-1)  # [B,C,C]
        X_cs = X_white @ C_tf.transpose(-2,-1)       # [B,N,C]

        # --- (7) add style mean back & cast to original dtype ---
        out = X_cs + self.mu_s.to(X_cs.dtype)        # [B,N,C]
        return out.to(orig_dtype)

class StyleWCT_highnoise:
    def __init__(self, dtype=torch.float32, base_eps=1e-5, max_tries=5):
        self.dtype     = dtype
        self.base_eps  = base_eps
        self.max_tries = max_tries
        self.mu_s            = None   # [1,1,C]
        self.color_transform = None   # [C,C]

    def set(self, y0_adain_embed: torch.Tensor, spatial_shape=None):
        """
        y0_adain_embed: [1, N, C]
        """
        style = y0_adain_embed.to(self.dtype)
        # 1) style mean + center
        self.mu_s = style.mean(dim=1, keepdim=True)           # [1,1,C]
        f_s_centered = style - self.mu_s                      # [1,N,C]

        # 2) style covariance [1,C,C]
        N = f_s_centered.size(1)
        cov_s = f_s_centered.transpose(1,2) @ f_s_centered / (N-1)
        cov_s = cov_s + self.base_eps * torch.eye(cov_s.size(-1),
                                                  device=cov_s.device,
                                                  dtype=cov_s.dtype).unsqueeze(0)

        # 3) Cholesky to get L so that cov_s = L @ Lᵀ
        L = self._batch_cholesky(cov_s)                       # [1,C,C]
        self.color_transform = L[0]                           # store [C,C]

    def get(self, denoised_embed: torch.Tensor):
        """
        denoised_embed: [B, N, C]
        returns: [B,N,C]
        """
        x = denoised_embed.to(self.dtype)
        B,N,C = x.shape

        # 1) content mean + center
        mu_c = x.mean(dim=1, keepdim=True)                    # [B,1,C]
        x_centered = x - mu_c                                 # [B,N,C]

        # 2) content covariance [B,C,C]
        cov_c = x_centered.transpose(1,2) @ x_centered / (N-1)
        cov_c = cov_c + self.base_eps * torch.eye(C,
                                                  device=cov_c.device,
                                                  dtype=cov_c.dtype).unsqueeze(0)

        # 3) get batched inverse‐Cholesky → whitening transform
        L_c_inv = self._batch_cholesky_inverse(cov_c)         # [B,C,C]
        whiten_tf = L_c_inv.transpose(1,2)                    # [B,C,C]

        # 4) whiten
        x_white = x_centered @ whiten_tf                      # [B,N,C]

        # 5) color  (broadcasted)
        color_tf = self.color_transform.unsqueeze(0).expand(B,-1,-1)  # [B,C,C]
        x_colored = x_white @ color_tf.transpose(1,2)                # [B,N,C]

        # 6) add style mean back
        return x_colored + self.mu_s.to(x_colored.dtype)

    def _batch_cholesky(self, A):
        """
        A: [B, C, C], assumed symmetric.
        returns L: [B, C, C] so that A_b ≈ L_b @ L_bᵀ
        """
        B,C,_ = A.shape
        L = torch.zeros_like(A)
        I   = torch.eye(C, device=A.device, dtype=A.dtype)
        for b in range(B):
            eps = self.base_eps
            for _ in range(self.max_tries):
                try:
                    L[b] = torch.linalg.cholesky(A[b] + eps*I)
                    break
                except RuntimeError:
                    eps *= 10
            else:
                # fallback to eigh+sqrt if chol never succeeded
                vals, vecs = torch.linalg.eigh(A[b] + eps*I)
                vals = vals.clamp(min=0).sqrt()
                L[b] = vecs @ torch.diag(vals) @ vecs.T
        return L

    def _batch_cholesky_inverse(self, A):
        """
        A: [B, C, C]
        returns invL: [B, C, C]  where invL[b] = (L_b)⁻¹ from cholesky on A[b].
        """
        B,C,_ = A.shape
        invL = torch.zeros_like(A)
        I    = torch.eye(C, device=A.device, dtype=A.dtype)
        for b in range(B):
            eps = self.base_eps
            for _ in range(self.max_tries):
                try:
                    Lb = torch.linalg.cholesky(A[b] + eps*I)
                    invL[b] = torch.cholesky_inverse(Lb)
                    break
                except RuntimeError:
                    eps *= 10
            else:
                # fallback via eigh if necessary
                vals, vecs = torch.linalg.eigh(A[b] + eps*I)
                inv_vals = vals.clamp(min=1e-12).rsqrt()
                invL[b] = vecs @ torch.diag(inv_vals) @ vecs.T
        return invL


class StyleWCT_purenoise:
    def __init__(self, dtype=torch.float32):
        self.dtype = dtype
        self.mu_s = None
        self.color_transform = None

    def _compute_whitening(self, x_centered: torch.Tensor):
        """
        Compute whitening matrix for centered content features.
        x_centered: [B, N, C]
        returns: [B, C, C] whitening matrix
        """
        B, N, C = x_centered.shape
        eps = 1e-5
        cov = torch.matmul(x_centered.transpose(1, 2), x_centered) / (N - 1)  # [B, C, C]
        cov = cov + eps * torch.eye(C, dtype=self.dtype, device=x_centered.device).unsqueeze(0)

        try:
            L = torch.linalg.cholesky(cov)                     # [B, C, C]
            L_inv = torch.cholesky_inverse(L)                  # [B, C, C]
            whitening = L_inv.transpose(1, 2)                  # (L⁻¹).T
        except RuntimeError:
            # fallback to SVD
            U, S, _ = torch.linalg.svd(cov)
            whitening = U @ torch.diag_embed(S.rsqrt()) @ U.transpose(1, 2)

        return whitening

    def _compute_coloring(self, f_centered: torch.Tensor):
        """
        Compute coloring matrix from centered style features.
        f_centered: [1, N, C]
        returns: [1, C, C] coloring matrix
        """
        N, C = f_centered.shape[1:]
        eps = 1e-5
        cov = torch.matmul(f_centered.transpose(1, 2), f_centered) / (N - 1)  # [1, C, C]
        cov = cov + eps * torch.eye(C, dtype=self.dtype, device=f_centered.device).unsqueeze(0)

        try:
            L = torch.linalg.cholesky(cov)  # [1, C, C]
            return L
        except RuntimeError:
            # fallback to SVD
            U, S, _ = torch.linalg.svd(cov)
            return U @ torch.diag_embed(S.sqrt()) @ U.transpose(1, 2)

    def set(self, y0_adain_embed: torch.Tensor):
        """
        y0_adain_embed: [1, N, C] — reference style features
        """
        y0 = y0_adain_embed.to(self.dtype)
        self.mu_s = y0.mean(dim=1, keepdim=True)                    # [1, 1, C]
        f_s_centered = y0 - self.mu_s                               # [1, N, C]
        self.color_transform = self._compute_coloring(f_s_centered)  # [1, C, C]

    def get(self, denoised_embed: torch.Tensor):
        """
        denoised_embed: [B, N, C] — content features to transform
        returns: styled features matching style covariance and mean
        """
        x = denoised_embed.to(self.dtype)
        mu_c = x.mean(dim=1, keepdim=True)                          # [B, 1, C]
        x_centered = x - mu_c                                       # [B, N, C]

        whitening = self._compute_whitening(x_centered)             # [B, C, C]
        x_white = torch.matmul(x_centered, whitening.transpose(1, 2))  # [B, N, C]

        B = x_white.shape[0]
        color = self.color_transform.expand(B, -1, -1)              # [B, C, C]
        x_colored = torch.matmul(x_white, color.transpose(1, 2))    # [B, N, C]

        x_final = x_colored + self.mu_s                             # [B, N, C]
        return x_final



class WaveletStyleWCT(StyleWCT):
    def set(self, y0_adain_embed: torch.Tensor, h_len, w_len):
        if self.y0_adain_embed is None or self.y0_adain_embed.shape != y0_adain_embed.shape or torch.norm(self.y0_adain_embed - y0_adain_embed) > 0:
            self.y0_adain_embed = y0_adain_embed.clone()
            
            B, HW, C = y0_adain_embed.shape
            LL, _, _, _ = haar_wavelet_decompose(y0_adain_embed.contiguous().view(B, C, h_len, w_len))

            B_LL, C_LL, H_LL, W_LL = LL.shape
            #flat = rearrange(LL, 'b c h w -> b (h w) c')
            flat = LL.contiguous().view(B_LL, H_LL * W_LL, C_LL)

            f_s = flat[0]  # assuming batch size 1 or using only the first
            self.mu_s = f_s.mean(dim=0, keepdim=True)
            f_s_centered = f_s - self.mu_s
            self.y0_color = self.whiten(f_s_centered, set=True)
            #self.y0_adain_embed = flat  # cache if needed
    
    def get(self, denoised_embed: torch.Tensor, h_len, w_len, stylize_highfreq=False):

        B, HW, C = denoised_embed.shape
        
        denoised_embed = denoised_embed.contiguous().view(B, C, h_len, w_len)
        
        for i in range(B):
            x = denoised_embed[i:i+1]  # [1, C, H, W]
            LL, LH, HL, HH = haar_wavelet_decompose(x)

            def process_band(band):
                Bc, Cc, Hc, Wc = band.shape
                flat = band.contiguous().view(Bc, Hc * Wc, Cc)
                
                styled = super(WaveletStyleWCT, self).get(flat)
                return styled.contiguous().view(Bc, Cc, Hc, Wc)

            #LL_styled = process_band(LL)
            LL_styled = LL
            #LH_styled = LH
            #HL_styled = HL
            #HH_styled = HH

            if stylize_highfreq:
                LH_styled = process_band(LH)
                HL_styled = process_band(HL)
                HH_styled = process_band(HH)
            else:
                LH_styled, HL_styled, HH_styled = LH, HL, HH

            recon = haar_wavelet_reconstruct(LL_styled, LH_styled, HL_styled, HH_styled)
            denoised_embed[i] = recon.squeeze(0)

        return denoised_embed.view(B, HW, C)



def haar_wavelet_decompose(x):
    """
    Orthonormal Haar decomposition.
    Input:  [B, C, H, W]
    Output: LL, LH, HL, HH with shape [B, C, H//2, W//2]
    """
    if x.dtype != torch.float32:
        x = x.float()
    
    B, C, H, W = x.shape
    assert H % 2 == 0 and W % 2 == 0, "Input must have even H, W"

    # Precompute
    norm = 1 / 2**0.5

    x00 = x[:, :, 0::2, 0::2]
    x01 = x[:, :, 0::2, 1::2]
    x10 = x[:, :, 1::2, 0::2]
    x11 = x[:, :, 1::2, 1::2]

    LL = (x00 + x01 + x10 + x11) * norm * 0.5
    LH = (x00 - x01 + x10 - x11) * norm * 0.5
    HL = (x00 + x01 - x10 - x11) * norm * 0.5
    HH = (x00 - x01 - x10 + x11) * norm * 0.5

    return LL, LH, HL, HH

def haar_wavelet_reconstruct(LL, LH, HL, HH):
    """
    Orthonormal inverse Haar reconstruction.
    Input:  LL, LH, HL, HH [B, C, H, W]
    Output: Reconstructed [B, C, H*2, W*2]
    """
    norm = 1 / 2**0.5
    B, C, H, W = LL.shape

    x00 = (LL + LH + HL + HH) * norm
    x01 = (LL - LH + HL - HH) * norm
    x10 = (LL + LH - HL - HH) * norm
    x11 = (LL - LH - HL + HH) * norm

    out = torch.zeros(B, C, H * 2, W * 2, device=LL.device, dtype=LL.dtype)
    out[:, :, 0::2, 0::2] = x00
    out[:, :, 0::2, 1::2] = x01
    out[:, :, 1::2, 0::2] = x10
    out[:, :, 1::2, 1::2] = x11

    return out








"""

class StyleFeatures:  
    def __init__(self, dtype=torch.float64,):
        self.dtype = dtype

    def set(self, y0_adain_embed: torch.Tensor):
            
    def get(self, denoised_embed: torch.Tensor):

        return "Norpity McNerp"

"""




class Retrojector:  
    def __init__(self, proj=None, W_inv=None, patch_size=2, pinv_dtype=torch.float64, dtype=torch.float64, ENDO=False):
        self.proj       = proj
        self.patch_size = patch_size
        self.pinv_dtype = pinv_dtype
        self.dtype      = dtype
        
        self.LINEAR     = isinstance(proj, nn.Linear)
        self.CONV2D     = isinstance(proj, nn.Conv2d)
        self.CONV3D     = isinstance(proj, nn.Conv3d)
        self.ENDO       = ENDO
        self.W          = proj.weight.data.to(dtype=dtype).cuda()
        
        if W_inv is not None:
            self.W_inv = W_inv.to(dtype=pinv_dtype).cuda()
        else:
            if self.LINEAR:
                self.W_inv = torch.linalg.pinv(proj.weight.data.to(dtype=pinv_dtype).cuda()).to(dtype=dtype)
            elif self.CONV2D:
                C_out, _, kH, kW = proj.weight.shape
                W_flat = proj.weight.data.view(C_out, -1).cuda().to(dtype=pinv_dtype)
                self.W_inv = torch.linalg.pinv(W_flat)
        
        self.W_inv = self.W_inv.cuda().to(dtype)
        
        if proj.bias is None:
            if self.LINEAR:
                bias_size = proj.out_features
            else:
                bias_size = proj.out_channels
            self.b = torch.zeros(bias_size, dtype=dtype, device=self.W_inv.device)
        else:
            self.b = proj.bias.data.to(dtype=dtype).to(self.W_inv.device)
        
    def embed(self, img: torch.Tensor):
        self.h = img.shape[-2] // self.patch_size
        self.w = img.shape[-1] // self.patch_size
        if img.ndim == 3:
            self.h, self.w = -1,-1
        img = comfy.ldm.common_dit.pad_to_patch_size(img, (self.patch_size, self.patch_size))
        
        if   self.CONV2D:
            self.orig_shape = img.shape  # for unembed
            img_embed = F.conv2d(
                img.to(self.W), 
                weight=self.W, 
                bias=self.b, 
                stride=self.proj.stride, 
                padding=self.proj.padding
            )
            #img_embed = rearrange(img_embed, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=self.patch_size, pw=self.patch_size) 
            img_embed = rearrange(img_embed, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=1, pw=1) 
        
        elif self.LINEAR:
            if img.ndim == 4:
                img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=self.patch_size, pw=self.patch_size) 
            if self.ENDO:
                img_embed = F.linear(img.to(self.b) - self.b, self.W_inv)
            else:
                img_embed = F.linear(img.to(self.W), self.W, self.b)
        
        return img_embed.to(img)
    
    def unembed(self, img_embed: torch.Tensor):
        if   self.CONV2D:
            #img_embed = rearrange(img_embed, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=self.h, w=self.w, ph=self.patch_size, pw=self.patch_size)
            img_embed = rearrange(img_embed, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=self.h, w=self.w, ph=1, pw=1)
            img = self.invert_conv2d(img_embed)
        
        elif self.LINEAR:
            if self.ENDO:
                img = F.linear(img_embed.to(self.W), self.W, self.b)
            else:
                img = F.linear(img_embed.to(self.b) - self.b, self.W_inv)
            if img.ndim == 3 and self.h > 0 and self.w > 0:
                img = rearrange(img, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=self.h, w=self.w, ph=self.patch_size, pw=self.patch_size)
        
        return img.to(img_embed)
    
    def invert_conv2d(self, z: torch.Tensor,) -> torch.Tensor:
        z_dtype = z.dtype
        z = z.to(self.pinv_dtype)
        conv = self.proj
        
        B, C_in, H, W      = self.orig_shape
        C_out, _, kH, kW   = conv.weight.shape
        stride_h, stride_w = conv.stride
        pad_h,    pad_w    = conv.padding

        b = conv.bias.view(1, C_out, 1, 1).to(z)
        z_nobias = z - b

        #W_flat = conv.weight.view(C_out, -1).to(z)  
        #W_pinv = torch.linalg.pinv(W_flat)    

        Bz, Co, Hp, Wp = z_nobias.shape
        z_flat = z_nobias.reshape(Bz, Co, -1)  

        x_patches = self.W_inv @ z_flat   

        x_sum = F.fold(
            x_patches,
            output_size=(H + 2*pad_h, W+ 2*pad_w),
            kernel_size=(kH, kW),
            stride=(stride_h, stride_w),
        )
        ones = torch.ones_like(x_patches)
        count = F.fold(
            ones,
            output_size=(H + 2*pad_h, W + 2*pad_w),
            kernel_size=(kH, kW),
            stride=(stride_h, stride_w),
        )  

        x_recon = x_sum / count.clamp(min=1e-6)
        if pad_h > 0 or pad_w > 0:
            x_recon = x_recon[..., pad_h:pad_h+H, pad_w:pad_w+W]

        return x_recon.to(z_dtype)
    
    def invert_patch_embedding(self, z: torch.Tensor, original_shape: torch.Size, grid_sizes: Optional[Tuple[int,int,int]] = None) -> torch.Tensor:

        B, C_in, D, H, W = original_shape
        pD, pH, pW = self.patch_size
        sD, sH, sW = pD, pH, pW

        if z.ndim == 3:
            # [B, S, C_out] -> reshape to [B, C_out, D', H', W']
            S = z.shape[1]
            if grid_sizes is None:
                Dp = D // pD
                Hp = H // pH
                Wp = W // pW
            else:
                Dp, Hp, Wp = grid_sizes
            C_out = z.shape[2]
            z = z.transpose(1, 2).reshape(B, C_out, Dp, Hp, Wp)
        else:
            B2, C_out, Dp, Hp, Wp = z.shape
            assert B2 == B, "Batch size mismatch... ya sharked it."

        # kncokout bias
        b = self.patch_embedding.bias.view(1, C_out, 1, 1, 1)
        z_nobias = z - b

        # 2D filter -> pinv
        w3 = self.patch_embedding.weight         # [C_out, C_in, 1, pH, pW]
        w2 = w3.squeeze(2)                       # [C_out, C_in, pH, pW]
        out_ch, in_ch, kH, kW = w2.shape
        W_flat = w2.view(out_ch, -1)            # [C_out, in_ch*pH*pW]
        W_pinv = torch.linalg.pinv(W_flat)      # [in_ch*pH*pW, C_out]

        # merge depth for 2D unfold wackiness
        z2 = z_nobias.permute(0,2,1,3,4).reshape(B*Dp, C_out, Hp, Wp)

        # apply pinv ... get patch vectors
        z_flat    = z2.reshape(B*Dp, C_out, -1)  # [B*Dp, C_out, L]
        x_patches = W_pinv @ z_flat              # [B*Dp, in_ch*pH*pW, L]

        # fold -> spatial frames
        x2 = F.fold(
            x_patches,
            output_size=(H, W),
            kernel_size=(pH, pW),
            stride=(sH, sW)
        )  # → [B*Dp, C_in, H, W]

        # un-merge depth
        x2 = x2.reshape(B, Dp, in_ch, H, W)           # [B, Dp,  C_in, H, W]
        x_recon = x2.permute(0,2,1,3,4).contiguous()  # [B, C_in,   D, H, W]
        return x_recon






def invert_conv2d(
    conv: torch.nn.Conv2d,
    z:    torch.Tensor,
    original_shape: torch.Size,
) -> torch.Tensor:
    import torch.nn.functional as F

    B, C_in, H, W = original_shape
    C_out, _, kH, kW = conv.weight.shape
    stride_h, stride_w = conv.stride
    pad_h,    pad_w    = conv.padding

    if conv.bias is not None:
        b = conv.bias.view(1, C_out, 1, 1).to(z)
        z_nobias = z - b
    else:
        z_nobias = z

    W_flat = conv.weight.view(C_out, -1).to(z)  
    W_pinv = torch.linalg.pinv(W_flat)    

    Bz, Co, Hp, Wp = z_nobias.shape
    z_flat = z_nobias.reshape(Bz, Co, -1)  

    x_patches = W_pinv @ z_flat   

    x_sum = F.fold(
        x_patches,
        output_size=(H + 2*pad_h, W + 2*pad_w),
        kernel_size=(kH, kW),
        stride=(stride_h, stride_w),
    )
    ones = torch.ones_like(x_patches)
    count = F.fold(
        ones,
        output_size=(H + 2*pad_h, W + 2*pad_w),
        kernel_size=(kH, kW),
        stride=(stride_h, stride_w),
    )  

    x_recon = x_sum / count.clamp(min=1e-6)
    if pad_h > 0 or pad_w > 0:
        x_recon = x_recon[..., pad_h:pad_h+H, pad_w:pad_w+W]

    return x_recon



def adain_seq_inplace(content: torch.Tensor, style: torch.Tensor, dim=1, eps: float = 1e-7) -> torch.Tensor:
    mean_c = content.mean(dim, keepdim=True)
    std_c  = content.std (dim, keepdim=True).add_(eps)  # in-place add
    mean_s = style.mean  (dim, keepdim=True)
    std_s  = style.std   (dim, keepdim=True).add_(eps)

    content.sub_(mean_c).div_(std_c).mul_(std_s).add_(mean_s)  # in-place chain
    return content

def adain_seq(content: torch.Tensor, style: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    return ((content - content.mean(1, keepdim=True)) / (content.std(1, keepdim=True) + eps)) * (style.std(1, keepdim=True) + eps) + style.mean(1, keepdim=True)









def apply_scattersort_tiled(
    denoised_spatial : torch.Tensor, 
    y0_adain_spatial : torch.Tensor, 
    tile_h           : int, 
    tile_w           : int, 
    pad              : int,
):
    """
    Apply spatial scattersort between denoised_spatial and y0_adain_spatial
    using local tile-wise sorted value matching.

    Args:
        denoised_spatial (Tensor): (B, C, H, W) tensor.
        y0_adain_spatial  (Tensor): (B, C, H, W) reference tensor.
        tile_h (int): tile height.
        tile_w (int): tile width.
        pad    (int): padding size to apply around tiles.

    Returns:
        denoised_embed (Tensor): (B, H*W, C) tensor after sortmatch.
    """
    denoised_padded = F.pad(denoised_spatial, (pad, pad, pad, pad), mode='reflect')
    y0_padded       = F.pad(y0_adain_spatial, (pad, pad, pad, pad), mode='reflect')

    denoised_padded_out = denoised_padded.clone()
    _, _, h_len, w_len = denoised_spatial.shape

    for ix in range(pad, h_len, tile_h):
        for jx in range(pad, w_len, tile_w):
            tile    = denoised_padded[:, :, ix - pad:ix + tile_h + pad, jx - pad:jx + tile_w + pad]
            y0_tile = y0_padded[:, :, ix - pad:ix + tile_h + pad, jx - pad:jx + tile_w + pad]

            tile    = rearrange(tile,    "b c h w -> b c (h w)", h=tile_h + pad * 2, w=tile_w + pad * 2)
            y0_tile = rearrange(y0_tile, "b c h w -> b c (h w)", h=tile_h + pad * 2, w=tile_w + pad * 2)

            src_sorted, src_idx =    tile.sort(dim=-1)
            ref_sorted, ref_idx = y0_tile.sort(dim=-1)

            new_tile = tile.scatter(dim=-1, index=src_idx, src=ref_sorted.expand(src_sorted.shape))
            new_tile = rearrange(new_tile, "b c (h w) -> b c h w", h=tile_h + pad * 2, w=tile_w + pad * 2)

            denoised_padded_out[:, :, ix:ix + tile_h, jx:jx + tile_w] = (
                new_tile if pad == 0 else new_tile[:, :, pad:-pad, pad:-pad]
            )

    denoised_padded_out = denoised_padded_out if pad == 0 else denoised_padded_out[:, :, pad:-pad, pad:-pad]
    return denoised_padded_out



def apply_scattersort_masked(
    denoised_embed         : torch.Tensor,
    y0_adain_embed         : torch.Tensor,
    y0_style_pos_mask      : torch.Tensor | None,
    y0_style_pos_mask_edge : torch.Tensor | None,
    h_len                  : int,
    w_len                  : int
):
    if y0_style_pos_mask is None:
        flatmask = torch.ones((1,1,h_len,w_len)).bool().flatten().bool()
    else:
        flatmask   = F.interpolate(y0_style_pos_mask, size=(h_len, w_len)).bool().flatten().cpu()
    flatunmask = ~flatmask

    if y0_style_pos_mask_edge is not None:
        edgemask = F.interpolate(
            y0_style_pos_mask_edge.unsqueeze(0), size=(h_len, w_len)
        ).bool().flatten()
        flatmask   = flatmask   & (~edgemask)
        flatunmask = flatunmask & (~edgemask)

    denoised_masked = denoised_embed[:, flatmask, :].clone()
    y0_adain_masked = y0_adain_embed[:, flatmask, :].clone()

    src_sorted, src_idx = denoised_masked.sort(dim=-2)
    ref_sorted, ref_idx = y0_adain_masked.sort(dim=-2)

    denoised_embed[:, flatmask, :] = src_sorted.scatter(dim=-2, index=src_idx, src=ref_sorted.expand(src_sorted.shape))

    if (flatunmask == True).any():
        denoised_unmasked = denoised_embed[:, flatunmask, :].clone()
        y0_adain_unmasked = y0_adain_embed[:, flatunmask, :].clone()

        src_sorted, src_idx = denoised_unmasked.sort(dim=-2)
        ref_sorted, ref_idx = y0_adain_unmasked.sort(dim=-2)

        denoised_embed[:, flatunmask, :] = src_sorted.scatter(dim=-2, index=src_idx, src=ref_sorted.expand(src_sorted.shape))

    if y0_style_pos_mask_edge is not None:
        denoised_edgemasked = denoised_embed[:, edgemask, :].clone()
        y0_adain_edgemasked = y0_adain_embed[:, edgemask, :].clone()

        src_sorted, src_idx = denoised_edgemasked.sort(dim=-2)
        ref_sorted, ref_idx = y0_adain_edgemasked.sort(dim=-2)

        denoised_embed[:, edgemask, :] = src_sorted.scatter(dim=-2, index=src_idx, src=ref_sorted.expand(src_sorted.shape))

    return denoised_embed




def apply_scattersort(
    denoised_embed         : torch.Tensor,
    y0_adain_embed         : torch.Tensor,
):
    #src_sorted, src_idx = denoised_embed.cpu().sort(dim=-2)
    src_idx    = denoised_embed.argsort(dim=-2)
    ref_sorted = y0_adain_embed.sort(dim=-2)[0]

    denoised_embed.scatter_(dim=-2, index=src_idx, src=ref_sorted.expand(ref_sorted.shape))

    return denoised_embed

def apply_scattersort_spatial(
    denoised_spatial         : torch.Tensor,
    y0_adain_spatial         : torch.Tensor,
):
    denoised_embed = rearrange(denoised_spatial, "b c h w -> b (h w) c")
    y0_adain_embed = rearrange(y0_adain_spatial, "b c h w -> b (h w) c")
    src_sorted, src_idx = denoised_embed.sort(dim=-2)
    ref_sorted, ref_idx = y0_adain_embed.sort(dim=-2)

    denoised_embed = src_sorted.scatter(dim=-2, index=src_idx, src=ref_sorted.expand(src_sorted.shape))
    
    return rearrange(denoised_embed, "b (h w) c -> b c h w", h=denoised_spatial.shape[-2], w=denoised_spatial.shape[-1])





def apply_scattersort_spatial(
    x_spatial : torch.Tensor,
    y_spatial : torch.Tensor,
):
    x_emb = rearrange(x_spatial, "b c h w -> b (h w) c")
    y_emb = rearrange(y_spatial, "b c h w -> b (h w) c")
    
    x_sorted, x_idx = x_emb.sort(dim=-2)
    y_sorted, y_idx = y_emb.sort(dim=-2)

    x_emb = x_sorted.scatter(dim=-2, index=x_idx, src=y_sorted.expand(x_sorted.shape))
    
    return rearrange(x_emb, "b (h w) c -> b c h w", h=x_spatial.shape[-2], w=x_spatial.shape[-1])




def apply_adain_spatial(
    x_spatial : torch.Tensor,
    y_spatial : torch.Tensor,
):
    x_emb = rearrange(x_spatial, "b c h w -> b (h w) c")
    y_emb = rearrange(y_spatial, "b c h w -> b (h w) c")
    
    x_mean = x_emb.mean(-2, keepdim=True)
    x_std  = x_emb.std (-2, keepdim=True)
    y_mean = y_emb.mean(-2, keepdim=True)
    y_std  = y_emb.std (-2, keepdim=True)

    assert (x_std == 0).any() == 0, "Target tensor has no variance!"
    assert (y_std == 0).any() == 0, "Reference tensor has no variance!"
    
    x_emb_adain = (x_emb - x_mean) / x_std
    x_emb_adain = (x_emb_adain * y_std) + y_mean
    
    return x_emb_adain.reshape_as(x_spatial)





















def adain_patchwise(content: torch.Tensor, style: torch.Tensor, sigma: float = 1.0, kernel_size: int = None, eps: float = 1e-5) -> torch.Tensor:
    # this one is really slow
    B, C, H, W = content.shape
    device     = content.device
    dtype      = content.dtype

    if kernel_size is None:
        kernel_size = int(2 * math.ceil(3 * sigma) + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1

    pad    = kernel_size // 2
    coords = torch.arange(kernel_size, dtype=torch.float64, device=device) - pad
    gauss  = torch.exp(-0.5 * (coords / sigma) ** 2)
    gauss /= gauss.sum()
    kernel_2d = (gauss[:, None] * gauss[None, :]).to(dtype=dtype)

    weight = kernel_2d.view(1, 1, kernel_size, kernel_size)

    content_padded = F.pad(content, (pad, pad, pad, pad), mode='reflect')
    style_padded   = F.pad(style,   (pad, pad, pad, pad), mode='reflect')
    result = torch.zeros_like(content)

    for i in range(H):
        for j in range(W):
            c_patch = content_padded[:, :, i:i + kernel_size, j:j + kernel_size]
            s_patch =   style_padded[:, :, i:i + kernel_size, j:j + kernel_size]
            w = weight.expand_as(c_patch)

            c_mean =  (c_patch              * w).sum(dim=(-1, -2), keepdim=True)
            c_std  = ((c_patch - c_mean)**2 * w).sum(dim=(-1, -2), keepdim=True).sqrt() + eps
            s_mean =  (s_patch              * w).sum(dim=(-1, -2), keepdim=True)
            s_std  = ((s_patch - s_mean)**2 * w).sum(dim=(-1, -2), keepdim=True).sqrt() + eps

            normed =  (c_patch[:, :, pad:pad+1, pad:pad+1] - c_mean) / c_std
            stylized = normed * s_std + s_mean
            result[:, :, i, j] = stylized.squeeze(-1).squeeze(-1)

    return result


def adain_patchwise_row_batch(content: torch.Tensor, style: torch.Tensor, sigma: float = 1.0, kernel_size: int = None, eps: float = 1e-5) -> torch.Tensor:

    B, C, H, W = content.shape
    device, dtype = content.device, content.dtype

    if kernel_size is None:
        kernel_size = int(2 * math.ceil(3 * sigma) + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1

    pad = kernel_size // 2
    coords = torch.arange(kernel_size, dtype=torch.float64, device=device) - pad
    gauss = torch.exp(-0.5 * (coords / sigma) ** 2)
    gauss = (gauss / gauss.sum()).to(dtype)
    kernel_2d = (gauss[:, None] * gauss[None, :])

    weight = kernel_2d.view(1, 1, kernel_size, kernel_size)

    content_padded = F.pad(content, (pad, pad, pad, pad), mode='reflect')
    style_padded = F.pad(style, (pad, pad, pad, pad), mode='reflect')
    result = torch.zeros_like(content)

    for i in range(H):
        c_row_patches = torch.stack([
            content_padded[:, :, i:i+kernel_size, j:j+kernel_size]
            for j in range(W)
        ], dim=0)  # [W, B, C, k, k]

        s_row_patches = torch.stack([
            style_padded[:, :, i:i+kernel_size, j:j+kernel_size]
            for j in range(W)
        ], dim=0)

        w = weight.expand_as(c_row_patches[0])

        c_mean = (c_row_patches * w).sum(dim=(-1, -2), keepdim=True)
        c_std  = ((c_row_patches - c_mean) ** 2 * w).sum(dim=(-1, -2), keepdim=True).sqrt() + eps
        s_mean = (s_row_patches * w).sum(dim=(-1, -2), keepdim=True)
        s_std  = ((s_row_patches - s_mean) ** 2 * w).sum(dim=(-1, -2), keepdim=True).sqrt() + eps

        center = kernel_size // 2
        central = c_row_patches[:, :, :, center:center+1, center:center+1]
        normed = (central - c_mean) / c_std
        stylized = normed * s_std + s_mean

        result[:, :, i, :] = stylized.squeeze(-1).squeeze(-1).permute(1, 2, 0)  # [B,C,W]

    return result



def adain_patchwise_row_batch_med(content: torch.Tensor, style: torch.Tensor, sigma: float = 1.0, kernel_size: int = None, eps: float = 1e-5, mask: torch.Tensor = None, use_median_blur: bool = False, lowpass_weight=1.0, highpass_weight=1.0) -> torch.Tensor:
    B, C, H, W = content.shape
    device, dtype = content.device, content.dtype

    if kernel_size is None:
        kernel_size = int(2 * math.ceil(3 * abs(sigma)) + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1

    pad = kernel_size // 2

    content_padded = F.pad(content, (pad, pad, pad, pad), mode='reflect')
    style_padded = F.pad(style, (pad, pad, pad, pad), mode='reflect')
    result = torch.zeros_like(content)

    scaling = torch.ones((B, 1, H, W), device=device, dtype=dtype)
    sigma_scale = torch.ones((H, W), device=device, dtype=torch.float32)
    if mask is not None:
        with torch.no_grad():
            padded_mask = F.pad(mask.float(), (pad, pad, pad, pad), mode="reflect")
            blurred_mask = F.avg_pool2d(padded_mask, kernel_size=kernel_size, stride=1, padding=pad)
            blurred_mask = blurred_mask[..., pad:-pad, pad:-pad]
            edge_proximity = blurred_mask * (1.0 - blurred_mask)
            scaling = 1.0 - (edge_proximity / 0.25).clamp(0.0, 1.0)
            sigma_scale = scaling[0, 0]  # assuming single-channel mask broadcasted across B, C

    if not use_median_blur:
        coords = torch.arange(kernel_size, dtype=torch.float64, device=device) - pad
        base_gauss = torch.exp(-0.5 * (coords / sigma) ** 2)
        base_gauss = (base_gauss / base_gauss.sum()).to(dtype)
        gaussian_table = {}
        for s in sigma_scale.unique():
            sig = float((sigma * s + eps).clamp(min=1e-3))
            gauss_local = torch.exp(-0.5 * (coords / sig) ** 2)
            gauss_local = (gauss_local / gauss_local.sum()).to(dtype)
            kernel_2d = gauss_local[:, None] * gauss_local[None, :]
            gaussian_table[s.item()] = kernel_2d

    for i in range(H):
        row_result = torch.zeros(B, C, W, dtype=dtype, device=device)
        for j in range(W):
            c_patch = content_padded[:, :, i:i+kernel_size, j:j+kernel_size]
            s_patch = style_padded[:, :, i:i+kernel_size, j:j+kernel_size]

            if use_median_blur:
                # Median blur with residual restoration
                unfolded_c = c_patch.reshape(B, C, -1)
                unfolded_s = s_patch.reshape(B, C, -1)

                c_median = unfolded_c.median(dim=-1, keepdim=True).values
                s_median = unfolded_s.median(dim=-1, keepdim=True).values

                center = kernel_size // 2
                central = c_patch[:, :, center, center].view(B, C, 1)
                residual = central - c_median
                stylized = lowpass_weight * s_median + residual * highpass_weight
            else:
                k = gaussian_table[float(sigma_scale[i, j].item())]
                local_weight = k.view(1, 1, kernel_size, kernel_size).expand(B, C, kernel_size, kernel_size)

                c_mean = (c_patch * local_weight).sum(dim=(-1, -2), keepdim=True)
                c_std = ((c_patch - c_mean) ** 2 * local_weight).sum(dim=(-1, -2), keepdim=True).sqrt() + eps
                s_mean = (s_patch * local_weight).sum(dim=(-1, -2), keepdim=True)
                s_std = ((s_patch - s_mean) ** 2 * local_weight).sum(dim=(-1, -2), keepdim=True).sqrt() + eps

                center = kernel_size // 2
                central = c_patch[:, :, center:center+1, center:center+1]
                normed = (central - c_mean) / c_std
                stylized = normed * s_std + s_mean

            local_scaling = scaling[:, :, i, j].view(B, 1, 1)
            stylized = central * (1 - local_scaling) + stylized * local_scaling

            row_result[:, :, j] = stylized.squeeze(-1)
        result[:, :, i, :] = row_result

    return result







def weighted_mix_n(tensor_list, weight_list, dim=-1, offset=0):
    assert all(t.shape == tensor_list[0].shape for t in tensor_list)
    assert len(tensor_list) == len(weight_list)

    total_weight = sum(weight_list)
    ratios = [w / total_weight for w in weight_list]

    length = tensor_list[0].shape[dim]
    idx = torch.arange(length)

    # Create a bin index tensor based on weighted slots
    float_bins = (idx + offset) * len(ratios) / length
    bin_idx = torch.floor(float_bins).long() % len(ratios)

    # Allocate slots based on ratio using a cyclic pattern
    counters = [0.0 for _ in ratios]
    slots = torch.empty_like(idx)

    for i in range(length):
        # Assign to the group that's most under-allocated
        expected = [r * (i + 1) for r in ratios]
        errors = [expected[j] - counters[j] for j in range(len(ratios))]
        k = max(range(len(errors)), key=lambda j: errors[j])
        slots[i] = k
        counters[k] += 1

    # Create mask for each tensor
    out = tensor_list[0].clone()
    for i, tensor in enumerate(tensor_list):
        mask = slots == i
        while mask.dim() < tensor.dim():
            mask = mask.unsqueeze(0)
        mask = mask.expand_as(tensor)
        out = torch.where(mask, tensor, out)
    
    return out






from torch import vmap

BLOCK_NAMES = {"double_blocks", "single_blocks", "up_blocks", "middle_blocks", "down_blocks", "input_blocks", "output_blocks"}

DEFAULT_BLOCK_WEIGHTS_MMDIT = {
    "attn_norm"    : 0.0,
    "attn_norm_mod": 0.0,
    "attn"         : 1.0,
    "attn_gated"   : 0.0,
    "attn_res"     : 1.0,
    "ff_norm"      : 0.0,
    "ff_norm_mod"  : 0.0,
    "ff"           : 1.0,
    "ff_gated"     : 0.0,
    "ff_res"       : 1.0,
    
    "h_tile"       : 8,
    "w_tile"       : 8,
}

DEFAULT_ATTN_WEIGHTS_MMDIT = {
    "qkv": 0.0,
    "q_proj": 0.0,
    "k_proj": 0.0,
    "v_proj": 1.0,
    "q_norm": 0.0,
    "k_norm": 0.0,
    "out"   : 1.0,
    
    "h_tile": 8,
    "w_tile": 8,
}

DEFAULT_BASE_WEIGHTS_MMDIT = {
    "proj_in" : 1.0,
    "proj_out": 1.0,
    
    "h_tile"  : 8,
    "w_tile"  : 8,
}

class Stylizer:
    buffer = {}
    
    CLS_WCT = StyleWCT()
    CLS_WCT2 = WaveletStyleWCT()
    CLS_WCT_fast = StyleWCT_Fast()

    
    def __init__(self, dtype=torch.float64, device=torch.device("cuda")):
        self.dtype = dtype
        self.device = device
        self.mask  = [None]
        self.apply_to = [""]
        self.method = ["passthrough"]
        self.h_tile = [-1]
        self.w_tile = [-1]
        
        self.w_len   = 0
        self.h_len   = 0
        self.img_len = 0
        
        self.energy_band0 = 0.5
        self.energy_band1 = 0.5
        
        self.IMG_1ST = True
        self.HEADS = 0
        self.KONTEXT = 0
    def set_mode(self, mode):
        self.method = [mode] #[getattr(self, mode)]
    
    def set_weights(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, [v])
    
    def set_weights_recursive(self, **kwargs):
        for name, val in kwargs.items():
            if hasattr(self, name):
                setattr(self, name, [val])

        for attr_name, attr_val in vars(self).items():
            if isinstance(attr_val, Stylizer):
                attr_val.set_weights_recursive(**kwargs)

        for list_name in BLOCK_NAMES:
            lst = getattr(self, list_name, None)
            if isinstance(lst, list):
                for element in lst:
                    if isinstance(element, Stylizer):
                        element.set_weights_recursive(**kwargs)
    
    def merge_weights(self, other):
        def recursive_merge(a, b, path):
            if isinstance(a, list) and isinstance(b, list):
                if path in BLOCK_NAMES:
                    out = []
                    for i in range(max(len(a), len(b))):
                        if i < len(a) and i < len(b):
                            out.append(recursive_merge(a[i], b[i], path=None))
                        elif i < len(a):
                            out.append(a[i])
                        else:
                            out.append(b[i])
                    return out
                return a + b

            if isinstance(a, dict) and isinstance(b, dict):
                merged = dict(a)
                for k, v_b in b.items():
                    if k in merged:
                        merged[k] = recursive_merge(merged[k], v_b, path=None)
                    else:
                        merged[k] = v_b
                return merged

            if hasattr(a, "__dict__") and hasattr(b, "__dict__"):
                for attr, val_b in vars(b).items():
                    val_a = getattr(a, attr, None)
                    if val_a is not None:
                        setattr(a, attr, recursive_merge(val_a, val_b, path=attr))
                    else:
                        setattr(a, attr, val_b)
                return a
            return b

        for attr in vars(self):
            if attr in BLOCK_NAMES:
                merged = recursive_merge(getattr(self, attr), getattr(other, attr, []), path=attr)
            elif hasattr(other, attr):
                merged = recursive_merge(getattr(self, attr), getattr(other, attr), path=attr)
            else:
                continue
            setattr(self, attr, merged)
    
    def set_len(self, h_len, w_len, img_slice, txt_slice, HEADS):
        self.h_len  = h_len
        self.w_len  = w_len
        self.img_slice = img_slice
        self.txt_slice = txt_slice
        self.img_len = h_len * w_len
        self.HEADS = HEADS

    @staticmethod
    def middle_slice(length, weight):
        """
        Returns a slice object that selects the middle `weight` fraction of a dimension.
        Example: weight=1.0 → full slice; weight=0.5 → middle 50%
        """
        if weight >= 1.0:
            return slice(None)
        wr = int((length * (1 - weight)) // 2)
        return slice(wr, -wr if wr > 0 else None)

    @staticmethod
    def get_outer_slice(x, weight):
        if weight >= 0.0:
            return x
        length = x.shape[-2]
        wr = int((length * (1 - (-weight))) // 2) 
        
        return torch.cat([x[...,:wr,:], x[...,-wr:,:]], dim=-2)

    @staticmethod
    def restore_outer_slice(x, x_outer, weight):
        if weight >= 0.0:
            return x
        length = x.shape[-2]
        wr = int((length * (1 - (-weight))) // 2) 
        
        x[...,:wr,:]  = x_outer[...,:wr,:]
        x[...,-wr:,:] = x_outer[...,-wr:,:]
        return x

    def __call__(self, x, attr):
        if x.shape[0] == 1 and not self.KONTEXT:
            return x
        
        weight_list = getattr(self, attr)
        weights_all_zero = all(weight == 0.0 for weight in weight_list)
        if weights_all_zero:
            return x
        
        #if attr in {"q_norm", "k_norm", "q_proj", "k_proj"}:
        #    x = x[[1, 0]]
        
        #self.HEADS=24
        #x_ndim = x.ndim
        #if x_ndim == 3:
        #    B, HW, C = x.shape
        #    if x.shape[-2] != self.HEADS and self.HEADS != 0:
        #        x = x.reshape(B,self.HEADS,HW,-1)
        
        HEAD_DIM = x.shape[1]
        if HEAD_DIM == self.HEADS:
            B, HEAD_DIM, HW, C = x.shape
            x = x.reshape(B, HW, C*HEAD_DIM)
        
        if hasattr(self, "KONTEXT") and self.KONTEXT == 1:
            x = x.reshape(2, x.shape[1] // 2, x.shape[2])
        
        txt_slice, img_slice, ktx_slice = self.txt_slice, self.img_slice, None
        if hasattr(self, "KONTEXT") and self.KONTEXT == 2:
            ktx_slice = self.img_slice # slice(2 * self.img_slice.start, None)
            img_slice = slice(2 * self.img_slice.start, self.img_slice.start)
            txt_slice = slice(None, 2 * self.txt_slice.stop)
        
        weights_all_one         = all(weight == 1.0           for weight in weight_list)
        weights_all_same = all(weight == weight_list[0] for weight in weight_list)
        methods_all_scattersort = all(name   == "scattersort" for name   in self.method)
        masks_all_none = all(mask is None for mask in self.mask)
        
        if weights_all_one and methods_all_scattersort and len(weight_list) > 1 and masks_all_none:
            buf = Stylizer.buffer
            buf['src_idx']   = x[0:1].argsort(dim=-2)
            buf['ref_sorted'], buf['ref_idx'] = x[1:].reshape(1, -1, x.shape[-1]).sort(dim=-2)
            buf['src'] = buf['ref_sorted'][:,::len(weight_list)].expand_as(buf['src_idx'])    #            interleave_stride = len(weight_list)
            
            #x[0:1] = x[0:1].scatter_(dim=-2, index=buf['src_idx'], src=buf['src'],)
            slc = Stylizer.middle_slice(buf['src'].shape[-2], weight_list[0]) 
            
            x[0:1] = x[0:1].scatter_(dim=-2, index=buf['src_idx'][...,slc,:], src=buf['src'][...,slc,:],)
        else:
            for i, (weight, mask) in enumerate(zip(weight_list, self.mask)):
                if mask is not None:
                    x01 = x[0:1].clone()
                slc = Stylizer.middle_slice(x.shape[-2], abs(weight))
                if weight < 0:
                    #x_base = x.clone()
                    x = x[[1, 0]]
                
                txt_method_name = self.method[i].removeprefix("tiled_")
                txt_method = getattr(self, txt_method_name)
                
                method_name = self.method[i].removeprefix("tiled_") if self.img_len > x.shape[-2] or self.h_len < 0 else self.method[i]
                method = getattr(self, method_name)
                apply_to = self.apply_to[i]
                if   weight == 0.0:
                    continue
                else: # if weight == 1.0:
                    if weight > 0 and weight < 1:
                        x_clone = x.clone()
                    if   self.img_len == x.shape[-2]  or apply_to == "img+txt" or self.h_len < 0:
                        x = method(x, idx=i+1, slc=slc)
                    elif   self.img_len < x.shape[-2]:
                        if "img" in apply_to:
                            x[...,img_slice,:] = method(x[...,img_slice,:], idx=i+1, slc=slc)
                            #if ktx_slice is not None:
                            #    x[...,ktx_slice,:] = method(x[...,ktx_slice,:], idx=i+1)
                            #x[:,:self.img_len,:] = method(x[:,:self.img_len,:], idx=i+1)
                        if "txt" in apply_to:
                            x[...,txt_slice,:] = txt_method(x[...,txt_slice,:], idx=i+1, slc=slc)
                            #x[:,self.img_len:,:] = method(x[:,self.img_len:,:], idx=i+1)
                        if not "img" in apply_to and not "txt" in apply_to:
                            pass
                    elif not "img" in apply_to:
                        x = method(x, idx=i+1, slc=slc)
                    if weight > 0 and weight < 1 and txt_method_name != "scattersort":
                        x = torch.lerp(x_clone, x, weight)
                #else:
                #    x = torch.lerp(x, method(x.clone(), idx=i+1), weight)
                
                if mask is not None:
                    x[0:1,...,img_slice,:] = torch.lerp(x01[...,img_slice,:], x[0:1,...,img_slice,:], mask.view(1, -1, 1))  
                    if ktx_slice is not None:
                        x[0:1,...,ktx_slice,:] = torch.lerp(x01[...,ktx_slice,:], x[0:1,...,ktx_slice,:], mask.view(1, -1, 1))  
                    #x[0:1,:self.img_len] = torch.lerp(x01[:,:self.img_len], x[0:1,:self.img_len], mask.view(1, -1, 1))
                if weight < 0:
                    x = x[[1, 0]]
                #    if self.method[i] == "scattersort":
                #        x = 2 * x_base - x
                #    else:
                #        x = x_base + abs(weight) * (x_base - x)

        #if attr in {"q_norm", "k_norm", "q_proj", "k_proj"}:
        #    x = x[[1, 0]]
        
        #if x_ndim == 3:
        #    return x.view(B,HW,C)
        if hasattr(self, "KONTEXT") and  self.KONTEXT == 1:
            x = x.reshape(1, x.shape[1] * 2, x.shape[2])
        
        if HEAD_DIM == self.HEADS:
            return x.reshape(B, HEAD_DIM, HW, C)
        else:
            return x

    def WCT_fast(self, x, idx=1, *args, **kwargs):
        Stylizer.CLS_WCT_fast.set(x[idx:idx+1])
        x[0:1] = Stylizer.CLS_WCT_fast.get(x[0:1])
        return x

    def WCT_batch(self, x, idx=1, *args, **kwargs):
        x_dtype = x.dtype
        
        x = x.to(torch.float64)
        x_norm = x - x.mean(dim=-2, keepdim=True)
        
        cov = (x_norm.transpose(-2,-1) @ x_norm) / (x_norm.size(-2) - 1)
        
        cov = cov.to(torch.float32)
        U, S, Vh = torch.linalg.svd(cov)
        U, S, Vh = U.to(torch.float64), S.to(torch.float64), Vh
        
        S_root_content = torch.diag(S[0]  .clamp(min=0).rsqrt()).unsqueeze(0)
        S_root_style   = torch.diag(S[idx].clamp(min=0). sqrt()).unsqueeze(0)
        
        S_root = torch.cat([S_root_content, S_root_style], dim=0)
        
        whiten = U @ S_root @ U.transpose(-2,-1)
        
        f_c_whitened = x_norm[0:1] @ whiten[0:1].transpose(-2,-1)
        f_cs         = f_c_whitened @ whiten[idx:idx+1].transpose(-2,-1) + x.mean(dim=-2, keepdim=True)[idx:idx+1]
        
        x[0:1] = f_cs.to(x_dtype)
        
        return x.to(x_dtype)
    
    def WCT_batch_eigh(self, x, idx=1, *args, **kwargs):
        x_dtype = x.dtype
        
        x = x.to(torch.float64)
        x_norm = x  - x.mean(dim=-2, keepdim=True)
        
        cov = (x_norm.transpose(-2,-1) @ x_norm) / (x_norm.size(-2) - 1)
        cov = cov + 1e-4 * torch.eye(cov.size(-1), dtype=cov.dtype, device=cov.device)
        
        cov = cov.to(torch.float32)
        S, U = torch.linalg.eigh(cov)
        S, U = S.to(torch.float64), U.to(torch.float64)
        
        S_root_content = torch.diag(S[0]  .clamp(min=0).rsqrt()).unsqueeze(0)
        S_root_style   = torch.diag(S[idx].clamp(min=0). sqrt()).unsqueeze(0)
        
        S_root = torch.cat([S_root_content, S_root_style], dim=0)
        
        whiten = U @ S_root @ U.transpose(-2,-1)
        
        f_c_whitened = x_norm[0:1] @ whiten[0:1].transpose(-2,-1)
        f_cs         = f_c_whitened @ whiten[idx:idx+1].transpose(-2,-1) + x.mean(dim=-2, keepdim=True)[idx:idx+1]
        
        x[0:1] = f_cs.to(x_dtype)
        
        return x.to(x_dtype)

    def WCT_batch_cholesky(self, x, idx=1, eps=1e-4, *args, **kwargs):
        x_dtype = x.dtype
        x = x.to(torch.float64)
        B, N, C = x.shape

        x_centered = x - x.mean(dim=-2, keepdim=True)
        cov = (x_centered.transpose(-2, -1) @ x_centered) / (N - 1)
        cov = cov + eps * torch.eye(C, device=x.device, dtype=x.dtype)

        # Compute Cholesky factor
        try:
            L, info = torch.linalg.cholesky_ex(cov.float()) # [B, C, C]
            L = L.to(cov)
        except RuntimeError:
            print("Cholesky failed — covariance not positive definite")
            return x

        # Whitening for content
        L_content = L[0]
        #with torch.autocast(device_type='cuda', dtype=torch.float32):
        #    x0_white = torch.linalg.solve_triangular(L_content, x_centered[0].transpose(-2,-1), upper=False).transpose(-2,-1)

        x0_white = torch.linalg.solve_triangular(L_content.float(), x_centered[0].transpose(-2,-1).float(), upper=False).transpose(-2,-1).to(L_content)
        # Coloring for style
        L_style = L[idx]
        x_colored = (x0_white @ L_style.transpose(-2,-1))

        # Add mean back
        x_mean = x.mean(dim=-2, keepdim=True)[idx]
        x_result = x_colored + x_mean

        x[0] = x_result.to(x_dtype)
        return x.to(x_dtype)

    def WCT_batch_svd_direct_aoeu(self, x: torch.Tensor, idx=1, *args, **kwargs):
        """
        WCT using SVD applied directly to raw feature matrices (no covariance matrix).
        x: Tensor of shape [B, N, C] where N = spatial tokens, C = feature dim.
        """
        x_dtype = x.dtype
        x = x.to(torch.float32)

        # Center features per sample
        x_mean = x.mean(dim=-2, keepdim=True)  # [B, 1, C]
        x_centered = x - x_mean                # [B, N, C]

        # Get content and style samples
        f_c = x_centered[0]      # [N, C]
        f_s = x_centered[idx]    # [N, C]

        # SVD of content and style
        Uc, Sc, Vhc = torch.linalg.svd(f_c, full_matrices=False)
        Us, Ss, Vhs = torch.linalg.svd(f_s, full_matrices=False)

        # Clamp singular values to avoid explosion/division by zero
        eps = 1e-5
        Sc_inv = Sc.clamp(min=eps).reciprocal()
        Ss_sqrt = Ss.clamp(min=eps).sqrt()

        # Whitening
        f_c_white = (f_c @ Vhc.T) * Sc_inv.unsqueeze(0)  # [N, C]

        # Coloring
        f_c_recolored = f_c_white * Ss_sqrt.unsqueeze(0) @ Vhs  # [N, C]

        # Recenter to style mean
        f_final = f_c_recolored + x_mean[idx]

        # Write result back into x[0]
        x[0] = f_final.to(x_dtype)

        return x.to(x_dtype)

    def WCT_batch_svd_direct(self, x: torch.Tensor, idx=1, *args, **kwargs):
        x_dtype = x.dtype
        x = x.to(torch.float32)
        eps = 1e-5
        rank = 64  # try smaller if still unstable

        # Center
        x_mean = x.mean(dim=-2, keepdim=True)
        x_centered = x - x_mean

        f_c = x_centered[0]
        f_s = x_centered[idx]

        # SVD
        Uc, Sc, Vhc = torch.linalg.svd(f_c, full_matrices=False)
        Us, Ss, Vhs = torch.linalg.svd(f_s, full_matrices=False)

        # Truncate
        Vhc = Vhc[:rank]
        Sc = Sc[:rank]
        Vhs = Vhs[:rank]
        Ss = Ss[:rank]

        # Whitening: safe division
        f_c_proj = f_c @ Vhc.T
        f_c_white = f_c_proj / Sc.clamp(min=eps).unsqueeze(0)

        # Recoloring
        f_c_recolored = (f_c_white * Ss.sqrt().clamp(min=eps).unsqueeze(0)) @ Vhs

        # Add style mean
        f_final = f_c_recolored + x_mean[idx]

        x[0] = f_final.to(x_dtype)
        return x.to(x_dtype)


    def WCT_batch_lowrank(self, x, idx=1, rank=32, niter=1, *args, **kwargs):
        x_dtype = x.dtype
        x = x.to(torch.float64)
        
        # Center the batch
        x_mean = x.mean(dim=-2, keepdim=True)
        x_norm = x - x_mean

        # Compute covariance matrix: [B, C, C]
        cov = x_norm.transpose(-2, -1) @ x_norm / (x_norm.size(-2) - 1)
        cov = cov.to(torch.float32)

        # Apply low-rank SVD
        U, S, Vh = torch.svd_lowrank(cov, q=rank, niter=niter)  # q = rank
        U = U.to(torch.float64)
        S = S.to(torch.float64)

        # Whitening/coloring diagonal matrices
        S_root_content = torch.diag(S[0].clamp(min=0).rsqrt()).unsqueeze(0)
        S_root_style   = torch.diag(S[idx].clamp(min=0).sqrt()).unsqueeze(0)
        S_root = torch.cat([S_root_content, S_root_style], dim=0)

        # Whitening matrix (symmetric approx): [B, C, C]
        whiten = U @ S_root @ U.transpose(-2, -1)

        # Apply whitening and recoloring
        f_c_whitened = x_norm[0:1] @ whiten[0:1].transpose(-2, -1)
        f_cs = f_c_whitened @ whiten[idx:idx+1].transpose(-2, -1) + x_mean[idx:idx+1]

        x[0:1] = f_cs.to(x_dtype)
        return x.to(x_dtype)

    def WCT_batch_ldl(self, x: torch.Tensor, idx=1, *args, **kwargs):
        """
        Approximate WCT using an LDLᵀ decomposition.
        Expects x.shape == [2, N, C]: 0 = content, idx = style.
        Returns x with x[0] replaced by whitened-&-recolored features.
        """
        # keep original dtype
        orig_dtype = x.dtype

        # do the linear algebra in higher precision
        x64 = x.to(torch.float64)           # [B=2, N, C]
        B, N, C = x64.shape
        assert B >= 2, "need at least content + style"

        # 1) center
        mu    = x64.mean(dim=1, keepdim=True)     # [B,1,C]
        Xc    = x64 - mu                          # [B,N,C]

        # 2) covariance per sample
        cov   = (Xc.transpose(1,2) @ Xc) / (N - 1)  # [B,C,C]
        eps   = 1e-4
        eye   = torch.eye(C, dtype=cov.dtype, device=cov.device)
        cov  += eps * eye.unsqueeze(0)              # regularize

        # 3) LDLᵀ factorization
        #    LD: packed lower + D diagonal, pivots ignored for SPD
        LD, pivots = torch.linalg.ldl_factor(cov)   # LD: [B,C,C]

        # 4) unpack L and D
        #    L = unit‐lower‐triangular part of LD
        L = torch.tril(LD, diagonal=-1)             # strictly lower
        L = L + eye.unsqueeze(0)                    # add ones on diag

        #    D = diag(LD) → shape [B,C]
        D = torch.diagonal(LD, dim1=-2, dim2=-1)    # [B,C]

        # 5) build D^{-1/2} and D^{+1/2} as [B,C,C]
        D_inv_sqrt = torch.diag_embed(D.clamp(min=1e-8).rsqrt())  # [B,C,C]
        D_sqrt     = torch.diag_embed(D.clamp(min=0).sqrt())     # [B,C,C]

        # 6) whitening and coloring transforms
        #    W_c = L · D^{-1/2} · Lᵀ     for content
        W_c = L @ D_inv_sqrt @ L.transpose(-2, -1)               # [B,C,C]
        #    C_s = L · D^{+1/2} · Lᵀ     for style
        C_s = L @ D_sqrt     @ L.transpose(-2, -1)               # [B,C,C]

        # 7) apply to content slice (batch[0]) and style slice (batch[idx])
        #    whiten content:
        X_white = Xc[0:1] @ W_c[0:1].transpose(-2, -1)           # [1,N,C]
        #    then recolor with style transform
        X_cs    = X_white @    C_s[idx:idx+1].transpose(-2, -1)  # [1,N,C]

        # 8) readd style mean and cast back
        out = X_cs + mu[idx:idx+1]                              # [1,N,C]
        x[0:1] = out.to(orig_dtype)

        return x

    def WCT(self, x, idx=1, *args, **kwargs):
        Stylizer.CLS_WCT.use_svd = False
        Stylizer.CLS_WCT.set(x[idx:idx+1])
        x[0:1] = Stylizer.CLS_WCT.get(x[0:1])
        return x
    
    def WCT_SVD(self, x, idx=1, *args, **kwargs):
        Stylizer.CLS_WCT.use_svd = True
        Stylizer.CLS_WCT.set(x[idx:idx+1])
        x[0:1] = Stylizer.CLS_WCT.get(x[0:1])
        return x
    
    def WCT2(self, x, idx=1, *args, **kwargs):
        Stylizer.CLS_WCT2.use_svd = False
        Stylizer.CLS_WCT2.set(x[idx:idx+1], self.h_len, self.w_len)
        x[0:1] = Stylizer.CLS_WCT2.get(x[0:1].clone(), self.h_len, self.w_len)
        return x
    
    def WCT2_SVD(self, x, idx=1, *args, **kwargs):
        Stylizer.CLS_WCT2.use_svd = True
        Stylizer.CLS_WCT2.set(x[idx:idx+1], self.h_len, self.w_len)
        x[0:1] = Stylizer.CLS_WCT2.get(x[0:1].clone(), self.h_len, self.w_len)
        return x

    @staticmethod
    def AdaIN_(x, y, eps: float = 1e-7) -> torch.Tensor:
        mean_c = x.mean(-2, keepdim=True)
        std_c  = x.std (-2, keepdim=True).add_(eps)  # in-place add
        mean_s = y.mean  (-2, keepdim=True)
        std_s  = y.std   (-2, keepdim=True).add_(eps)
        x.sub_(mean_c).div_(std_c).mul_(std_s).add_(mean_s)  # in-place chain
        return x

    def AdaIN(self, x, idx=1, eps: float = 1e-7, *args, **kwargs) -> torch.Tensor:
        mean_c = x[0:1].mean(-2, keepdim=True)
        std_c  = x[0:1].std (-2, keepdim=True).add_(eps)  # in-place add
        mean_s = x[idx:idx+1].mean  (-2, keepdim=True)
        std_s  = x[idx:idx+1].std   (-2, keepdim=True).add_(eps)
        x[0:1].sub_(mean_c).div_(std_c).mul_(std_s).add_(mean_s)  # in-place chain
        return x
    
    #@staticmethod
    def adain_bandwise_dct_all(self, x: torch.Tensor, idx=1, band='all', eps=1e-7, *args, **kwargs):
        x = self.adain_bandwise_dct(x, idx, band=band, eps=eps)
        #x = self.adain_bandwise_dct(x.transpose(-2,-1), idx, 'high', eps).transpose(-2,-1)
        #x = self.adain_bandwise_dct(x.transpose(-2,-1), idx, band, eps).transpose(-2,-1)
        return x
        
    #@staticmethod
    def adain_bandwise_dct_low(self, x: torch.Tensor, idx=1, band='low', eps=1e-7, *args, **kwargs):

        x = self.adain_bandwise_dct(x, idx, band=band, eps=eps)
        #x = self.adain_bandwise_dct(x.transpose(-2,-1), idx, 'high', eps).transpose(-2,-1)

        #x = self.adain_bandwise_dct(x.transpose(-2,-1), idx, band, eps).transpose(-2,-1)

        return x
            
    #@staticmethod
    def adain_bandwise_dct_mid(self, x: torch.Tensor, idx=1, band='mid', eps=1e-7, *args, **kwargs):

        x = self.adain_bandwise_dct(x, idx, band=band, eps=eps)
        #x = self.adain_bandwise_dct(x.transpose(-2,-1), idx, 'high', eps).transpose(-2,-1)
        #x = self.adain_bandwise_dct(x.transpose(-2,-1), idx, band, eps).transpose(-2,-1)

        return x
        
    #@staticmethod
    def adain_bandwise_dct_high(self, x: torch.Tensor, idx=1, band='high', eps=1e-7, *args, **kwargs):
        x = self.adain_bandwise_dct(x, idx, band=band, eps=eps)
        #x = self.adain_bandwise_dct(x.transpose(-2,-1), idx, 'high', eps).transpose(-2,-1)

        return x
        
        x_clone = x.clone()
        x = self.adain_bandwise_dct(x, idx, band, eps)
        
        x[idx:idx+1] = x_clone[0:1]
        x = self.adain_bandwise_dct2d(x, idx, band, eps)
        
        return x
    
    #@staticmethod  # decomp spatial then features
    def adain_bandwise_dct(self, x: torch.Tensor, idx=1, band='low', eps=1e-7, *args, **kwargs):
        """
        x: [B, HW, C]  → we do 1D DCT over C (last dim) without specifying dim
        Applies AdaIN only on the selected 'low'/'mid'/'high' frequency band.
        """
        # keep original dtype
        orig_dtype = x.dtype
        #x = self.adain_bandwise_dct2d(x, idx, 'all', eps)

        # work in float
        x = x.float().clone()
        x_c = x[0:1]       # [1, HW, C]
        x_s = x[idx:idx+1] # [1, HW, C]

        C = x.shape[-1]
        third = C // 3
        #if   band == 'low':  slice_range = slice(0,      third) 
        #elif band == 'mid':  slice_range = slice(third,  2*third)
        #elif band == 'high': slice_range = slice(2*third, C)
        if   band == 'low':  slice_range = slice(0,      int(C*self.energy_band0))
        elif band == 'mid':  slice_range = slice(int(C*self.energy_band0),  int(C*self.energy_band1))
        elif band == 'high': slice_range = slice(int(C*self.energy_band1), C)
        elif band == 'all': slice_range = slice(None)
        else: raise ValueError("band must be 'low', 'mid' or 'high'")

        x_c = x_c.transpose(1, 2)   # B,HW,C -> B,C,HW
        x_s = x_s.transpose(1, 2)
        x_c_dct = dct(x_c, norm='ortho') 
        x_s_dct = dct(x_s, norm='ortho')
        
        
        
        C = x_c_dct.shape[-1]
        if   band == 'low':  slice_range = slice(0,      int(C*self.energy_band0))
        elif band == 'mid':  slice_range = slice(int(C*self.energy_band0),  int(C*self.energy_band1))
        elif band == 'high': slice_range = slice(int(C*self.energy_band1), C)
        elif band == 'all': slice_range = slice(None)
        else: raise ValueError("band must be 'low', 'mid' or 'high'")
        
        xc_band = x_c_dct[:, :, slice_range]  # [1, HW, band_width]
        xs_band = x_s_dct[:, :, slice_range]
        
        #mean_c = xc_band .mean(-1, keepdim=True)
        #std_c  = xc_band .std (-1, keepdim=True, unbiased=False).add_(eps)
        #mean_s = xs_band .mean(-1, keepdim=True)
        #std_s  = xs_band .std (-1, keepdim=True, unbiased=False).add_(eps)
        #xc_band = (xc_band - mean_c) / std_c * std_s + mean_s   # B,C,HW   spatial adain
        
        
        xc_band = Stylizer.scattersort_(xc_band.transpose(-2,-1), xs_band.transpose(-2,-1)).transpose(-2,-1)
        
        x_c_dct[:, :, slice_range] = xc_band
        
        x_c_dct = x_c_dct.transpose(1, 2)   # B,C,HW -> B,HW,C
        x_s_dct = x_s_dct.transpose(1, 2)
        x_c_dct = dct(x_c_dct, norm='ortho')
        x_s_dct = dct(x_s_dct, norm='ortho')

        # --- slice out the frequency band ---
        
        
        C = x_c_dct.shape[-1]
        if   band == 'low':  slice_range = slice(0,      int(C*self.energy_band0))
        elif band == 'mid':  slice_range = slice(int(C*self.energy_band0),  int(C*self.energy_band1))
        elif band == 'high': slice_range = slice(int(C*self.energy_band1), C)
        elif band == 'all': slice_range = slice(None)
        else: raise ValueError("band must be 'low', 'mid' or 'high'")
        
        xc_band = x_c_dct[:, :, slice_range]  # [1, HW, band_width]
        xs_band = x_s_dct[:, :, slice_range]



        #mean_c = xc_band .mean(-1, keepdim=True)
        #std_c  = xc_band .std (-1, keepdim=True, unbiased=False).add_(eps)
        #mean_s = xs_band .mean(-1, keepdim=True)
        #std_s  = xs_band .std (-1, keepdim=True, unbiased=False).add_(eps)
        #xc_band = (xc_band - mean_c) / std_c * std_s + mean_s



        #mean_c = xc_band .mean(-2, keepdim=True)
        #std_c  = xc_band .std (-2, keepdim=True, unbiased=False).add_(eps)
        #mean_s = xs_band .mean(-2, keepdim=True)
        #std_s  = xs_band .std (-2, keepdim=True, unbiased=False).add_(eps)
        #xc_band = (xc_band - mean_c) / std_c * std_s + mean_s    # B,HW,C   channelwise adain
        
        xc_band = Stylizer.scattersort_(xc_band, xs_band)
        

        #mean_c = xc_band .mean(dim=(-2,-1), keepdim=True)
        #std_c  = xc_band .std (dim=(-2,-1), keepdim=True, unbiased=False).add_(eps)
        #mean_s = xs_band .mean(dim=(-2,-1), keepdim=True)
        #std_s  = xs_band .std (dim=(-2,-1), keepdim=True, unbiased=False).add_(eps)

        # --- normalize & apply style ---
        #xc_band = (xc_band - mean_c) / std_c * std_s + mean_s

        # --- write back and inverse DCT ---
        x_c_dct[:, :, slice_range] = xc_band
        
        #x_c_dct = x_c_dct.transpose(1, 2)
        x_out = idct(x_c_dct, norm='ortho')  # again, default on last dim
        x_out = x_out.transpose(1, 2)

        x_out = idct(x_out, norm='ortho')  # again, default on last dim
        x_out = x_out.transpose(1, 2)

        x[0:1] = x_out
        x = x.to(orig_dtype)
        return x
    
    
    
    
    
    
    #@staticmethod  # decomp spatial then features
    def adain_bandwise_dct_wct(self, x: torch.Tensor, idx=1, band='low', eps=1e-7, *args, **kwargs):
        """
        x: [B, HW, C]  → we do 1D DCT over C (last dim) without specifying dim
        Applies AdaIN only on the selected 'low'/'mid'/'high' frequency band.
        """
        # keep original dtype
        orig_dtype = x.dtype
        #x = self.adain_bandwise_dct2d(x, idx, 'all', eps)

        # work in float
        x = x.float().clone()
        x_c = x[0:1]       # [1, HW, C]
        x_s = x[idx:idx+1] # [1, HW, C]

        C = x.shape[-1]
        third = C // 3
        #if   band == 'low':  slice_range = slice(0,      third) 
        #elif band == 'mid':  slice_range = slice(third,  2*third)
        #elif band == 'high': slice_range = slice(2*third, C)
        if   band == 'low':  slice_range = slice(0,      int(C*self.energy_band0))
        elif band == 'mid':  slice_range = slice(int(C*self.energy_band0),  int(C*self.energy_band1))
        elif band == 'high': slice_range = slice(int(C*self.energy_band1), C)
        elif band == 'all': slice_range = slice(None)
        else: raise ValueError("band must be 'low', 'mid' or 'high'")

        x_c = x_c.transpose(1, 2)   # B,HW,C -> B,C,HW
        x_s = x_s.transpose(1, 2)
        x_c_dct = dct(x_c, norm='ortho') 
        x_s_dct = dct(x_s, norm='ortho')
        
        #mean_c = x_c_dct .mean(-1, keepdim=True)
        #std_c  = x_c_dct .std (-1, keepdim=True, unbiased=False).add_(eps)
        #mean_s = x_s_dct .mean(-1, keepdim=True)
        #std_s  = x_s_dct .std (-1, keepdim=True, unbiased=False).add_(eps)
        #x_c_dct = (x_c_dct - mean_c) / std_c * std_s + mean_s   # B,C,HW   spatial adain
        
        
        #x_c_dct = Stylizer.scattersort_(x_c_dct.transpose(-2,-1), x_s_dct.transpose(-2,-1)).transpose(-2,-1)
        
        
        
        x_c_dct = x_c_dct.transpose(1, 2)   # B,C,HW -> B,HW,C
        x_s_dct = x_s_dct.transpose(1, 2)
        x_c_dct = dct(x_c_dct, norm='ortho')
        x_s_dct = dct(x_s_dct, norm='ortho')

        # --- slice out the frequency band ---
        xc_band = x_c_dct[:, :, slice_range]  # [1, HW, band_width]
        xs_band = x_s_dct[:, :, slice_range]



        #mean_c = xc_band .mean(-1, keepdim=True)
        #std_c  = xc_band .std (-1, keepdim=True, unbiased=False).add_(eps)
        #mean_s = xs_band .mean(-1, keepdim=True)
        #std_s  = xs_band .std (-1, keepdim=True, unbiased=False).add_(eps)
        #xc_band = (xc_band - mean_c) / std_c * std_s + mean_s



        #mean_c = xc_band .mean(-2, keepdim=True)
        #std_c  = xc_band .std (-2, keepdim=True, unbiased=False).add_(eps)
        #mean_s = xs_band .mean(-2, keepdim=True)
        #std_s  = xs_band .std (-2, keepdim=True, unbiased=False).add_(eps)
        #xc_band = (xc_band - mean_c) / std_c * std_s + mean_s    # B,HW,C   channelwise adain
        
        #xc_band = Stylizer.scattersort_(xc_band, xs_band)
        Stylizer.CLS_WCT.set(xc_band)
        xc_band = Stylizer.CLS_WCT.get(xc_band)
        
        

        #mean_c = xc_band .mean(dim=(-2,-1), keepdim=True)
        #std_c  = xc_band .std (dim=(-2,-1), keepdim=True, unbiased=False).add_(eps)
        #mean_s = xs_band .mean(dim=(-2,-1), keepdim=True)
        #std_s  = xs_band .std (dim=(-2,-1), keepdim=True, unbiased=False).add_(eps)

        # --- normalize & apply style ---
        #xc_band = (xc_band - mean_c) / std_c * std_s + mean_s
        
        
        



        # --- write back and inverse DCT ---
        x_c_dct[:, :, slice_range] = xc_band
        
        #x_c_dct = x_c_dct.transpose(1, 2)
        x_out = idct(x_c_dct, norm='ortho')  # again, default on last dim
        x_out = x_out.transpose(1, 2)

        x_out = idct(x_out, norm='ortho')  # again, default on last dim
        x_out = x_out.transpose(1, 2)

        x[0:1] = x_out
        x = x.to(orig_dtype)
        return x
    
    
    
    
    
    
    
    
    #@staticmethod
    def adain_bandwise_dct_regular(self, x: torch.Tensor, idx=1, band='low', eps=1e-7, *args, **kwargs):
        """
        x: [B, HW, C]  → we do 1D DCT over C (last dim) without specifying dim
        Applies AdaIN only on the selected 'low'/'mid'/'high' frequency band.
        """
        # keep original dtype
        orig_dtype = x.dtype
        #x = self.adain_bandwise_dct2d(x, idx, 'all', eps)

        # work in float
        x = x.float().clone()
        x_c = x[0:1]       # [1, HW, C]
        x_s = x[idx:idx+1] # [1, HW, C]

        C = x.shape[-1]
        third = C // 3
        #if   band == 'low':  slice_range = slice(0,      third) 
        #elif band == 'mid':  slice_range = slice(third,  2*third)
        #elif band == 'high': slice_range = slice(2*third, C)
        if   band == 'low':  slice_range = slice(0,      int(C*self.energy_band0))
        elif band == 'mid':  slice_range = slice(int(C*self.energy_band0),  int(C*self.energy_band1))
        elif band == 'high': slice_range = slice(int(C*self.energy_band1), C)
        elif band == 'all': slice_range = slice(None)
        else: raise ValueError("band must be 'low', 'mid' or 'high'")

        # --- DCT over channels (last dim) ---
        #x_c = x_c.transpose(1, 2)
        #x_s = x_s.transpose(1, 2)
        x_c_dct = dct(x_c, norm='ortho')  # default runs on last dim
        x_s_dct = dct(x_s, norm='ortho')
        #x_c_dct = x_c_dct.transpose(1, 2)
        #x_s_dct = x_s_dct.transpose(1, 2)

        # --- slice out the frequency band ---
        xc_band = x_c_dct[:, :, slice_range]  # [1, HW, band_width]
        xs_band = x_s_dct[:, :, slice_range]

        # --- AdaIN stats over spatial dim (HW) ---
        mean_c = xc_band .mean(-2, keepdim=True)
        std_c  = xc_band .std (-2, keepdim=True, unbiased=False).add_(eps)
        mean_s = xs_band .mean(-2, keepdim=True)
        std_s  = xs_band .std (-2, keepdim=True, unbiased=False).add_(eps)

        # --- normalize & apply style ---
        xc_band = (xc_band - mean_c) / std_c * std_s + mean_s

        # --- write back and inverse DCT ---
        x_c_dct[:, :, slice_range] = xc_band
        
        #x_c_dct = x_c_dct.transpose(1, 2)
        x_out = idct(x_c_dct, norm='ortho')  # again, default on last dim
        #x_out = x_out.transpose(1, 2)

        x[0:1] = x_out
        x = x.to(orig_dtype)
        return x

    def adain_bandwise_dct2d_all(self, x: torch.Tensor, idx=1, band='all', eps=1e-7, *args, **kwargs):
        return self.adain_bandwise_dct2d(x, idx, band, eps)

    def adain_bandwise_dct2d_low(self, x: torch.Tensor, idx=1, band='low', eps=1e-7, *args, **kwargs):
        return self.adain_bandwise_dct2d(x, idx, band, eps)
        
    def adain_bandwise_dct2d_mid(self, x: torch.Tensor, idx=1, band='mid', eps=1e-7, *args, **kwargs):
        return self.adain_bandwise_dct2d(x, idx, band, eps)
    
    def adain_bandwise_dct2d_high(self, x: torch.Tensor, idx=1, band='high', eps=1e-7, *args, **kwargs):
        return self.adain_bandwise_dct2d(x, idx, band, eps)

    def adain_bandwise_dct2d(self, x: torch.Tensor, idx=1, band='low', eps=1e-7, *args, **kwargs):
        """
        Applies AdaIN in DCT space, using 2D DCT over spatial dimensions [H, W].
        x: [B, H*W, C]
        Returns: [B, H*W, C]
        """
        orig_dtype = x.dtype
        x = x.float()

        B, HW, C = x.shape
        #H = W = int(HW ** 0.5)
        H = self.h_len
        W = self.w_len
        assert H * W == HW, "Input must be square for now"

        # Reshape to [B, C, H, W]
        x = x.transpose(1, 2).reshape(B, C, H, W)

        x_c = x[0:1]       # [1, C, H, W]
        x_s = x[idx:idx+1] # [1, C, H, W]

        # 2D DCT over H and W
        #x_c_dct = dct(dct(x_c, norm='ortho', dim=-1), norm='ortho', dim=-2)  # W then H
        #x_s_dct = dct(dct(x_s, norm='ortho', dim=-1), norm='ortho', dim=-2)
        
        x_c_dct = dct_2d_torch_dct(x_c)
        x_s_dct = dct_2d_torch_dct(x_s)

        # Frequency bands are per-spatial-frequency, so we'll do a circular mask over (u,v)
        H_freq = x_c_dct.shape[-2]
        W_freq = x_c_dct.shape[-1]
        yy, xx = torch.meshgrid(torch.arange(H_freq), torch.arange(W_freq), indexing="ij")
        radius = (yy**2 + xx**2).sqrt().to(x.device)

        max_radius = radius.max()
        third = max_radius / 3

        if band == 'low':
            mask = (radius < third).float()
        elif band == 'mid':
            mask = ((radius >= third) & (radius < 2 * third)).float()
        elif band == 'high':
            mask = (radius >= 2 * third).float()
        elif band == 'all':
            mask = torch.ones((H, W), dtype=torch.float32, device=x.device)
        else:
            raise ValueError("band must be 'low', 'mid' or 'high'")
        #mask = torch.ones((H, W), dtype=torch.float32, device=x.device)
        # Expand mask to match shape: [1, C, H, W]
        xc_low_mask, xc_mid_mask, xc_high_mask = channelwise_energy_masks_fast(x_c_dct, self.energy_band0, self.energy_band1)
        xs_low_mask, xs_mid_mask, xs_high_mask = channelwise_energy_masks_fast(x_s_dct, self.energy_band0, self.energy_band1)
        
        if band == 'low':
            xc_mask = xc_low_mask.float()
            xs_mask = xs_low_mask.float()
        elif band == 'mid':
            xc_mask = xc_mid_mask.float()
            xs_mask = xs_mid_mask.float()
        elif band == 'high':
            xc_mask = xc_high_mask.float()
            xs_mask = xs_high_mask.float()
        
        mask = mask[None, None, :, :]
        if band == 'all':
            xc_mask = mask
            xs_mask = mask

        # Compute AdaIN on masked area only
        #xc_band = x_c_dct * mask
        #xs_band = x_s_dct * mask
        
        xc_band = x_c_dct * xc_mask
        xs_band = x_s_dct * xs_mask

        mean_c = xc_band.mean(dim=(2, 3), keepdim=True)
        std_c  = xc_band.std (dim=(2, 3), keepdim=True, unbiased=False).add_(eps)
        mean_s = xs_band.mean(dim=(2, 3), keepdim=True)
        std_s  = xs_band.std (dim=(2, 3), keepdim=True, unbiased=False).add_(eps)

        xc_band = (xc_band - mean_c) / std_c * std_s + mean_s

        # Combine with unmodified frequencies
        #x_c_dct = x_c_dct * (1 - mask) + xc_band * mask
        x_c_dct = x_c_dct * (1 - xc_mask) + xc_band * xc_mask

        # Inverse DCT
        #x_out = idct(idct(x_c_dct, norm='ortho', dim=-1), norm='ortho', dim=-2)  # inverse over W then H
        
        x_out = idct_2d_torch_dct(x_c_dct)

        # Put back
        x[0:1] = x_out

        # Return as [B, HW, C] again
        return x.reshape(B, C, H*W).transpose(1, 2).to(orig_dtype)




    #@staticmethod
    def fft_adain_bandwise_low(self, x: torch.Tensor, idx=1, band='low', eps=1e-7, *args, **kwargs):
        return self.fft_adain_bandwise(x, idx, band, eps)
        
    #@staticmethod
    def fft_adain_bandwise_mid(self, x: torch.Tensor, idx=1, band='mid', eps=1e-7, *args, **kwargs):
        return self.fft_adain_bandwise(x, idx, band, eps)
    
    #@staticmethod
    def fft_adain_bandwise_high(self, x: torch.Tensor, idx=1, band='high', eps=1e-7, *args, **kwargs):
        return self.fft_adain_bandwise(x, idx, band, eps)

    def fft_adain_bandwise(self, x: torch.Tensor, idx=1, band='mid', eps=1e-7):
        """
        Applies AdaIN to x[0:1] using x[idx:idx+1] as style,
        only over a spatial frequency band, using FFT.

        x: [B, HW, C]  ← flattened input
        Returns: same shape as input
        """
        dtype = x.dtype
        B, HW, C = x.shape
        H, W = self.h_len, self.w_len

        assert HW == H * W, f"Expected HW = {H}×{W}, got {HW}"

        # Reshape to image grid
        x = x.float().reshape(B, C, H, W)

        x_c = x[0:1]       # [1, C, H, W]
        x_s = x[idx:idx+1] # [1, C, H, W]

        # FFT2 over spatial dimensions
        x_c_fft = torch.fft.fft2(x_c, norm='ortho')
        x_s_fft = torch.fft.fft2(x_s, norm='ortho')

        # Shift FFT so DC is centered
        x_c_fft = torch.fft.fftshift(x_c_fft, dim=(-2, -1))
        x_s_fft = torch.fft.fftshift(x_s_fft, dim=(-2, -1))

        # Frequency radius mask
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, H, device=x.device),
            torch.linspace(-1, 1, W, device=x.device),
            indexing='ij'
        )
        radius = torch.sqrt(xx ** 2 + yy ** 2)

        if band == 'low':
            mask = radius <= 0.3
        elif band == 'mid':
            mask = (radius > 0.3) & (radius <= 0.6)
        elif band == 'high':
            mask = radius > 0.6
        else:
            raise ValueError("band must be 'low', 'mid', or 'high'")

        mask = mask[None, None, :, :]  # [1, 1, H, W]

        # Separate magnitude and phase
        c_mag, c_phase = x_c_fft.abs(), x_c_fft.angle()
        s_mag = x_s_fft.abs()

        # AdaIN only on magnitude in band
        c_band = c_mag * mask
        s_band = s_mag * mask

        mean_c = c_band.mean(dim=(2, 3), keepdim=True)
        std_c  = c_band.std (dim=(2, 3), keepdim=True).add_(eps)
        mean_s = s_band.mean(dim=(2, 3), keepdim=True)
        std_s  = s_band.std (dim=(2, 3), keepdim=True).add_(eps)

        c_mag = (c_mag - mean_c) / std_c * std_s + mean_s

        # Reconstruct complex FFT with modified magnitude
        x_c_fft_mod = c_mag * torch.exp(1j * c_phase)

        # Shift back and iFFT
        x_c_fft_mod = torch.fft.ifftshift(x_c_fft_mod, dim=(-2, -1))
        x_out = torch.fft.ifft2(x_c_fft_mod, norm='ortho').real  # discard imaginary part

        # Insert back into batch
        x[0:1] = x_out

        return x.to(dtype).reshape(B, HW, C)


    def injection(self, x:torch.Tensor, idx=1, *args, **kwargs) -> torch.Tensor:
        x[0:1] = x[idx:idx+1]
        return x
    
    @staticmethod
    def injection_(x:torch.Tensor, y:torch.Tensor) -> torch.Tensor:
        return y
    
    @staticmethod
    def passthrough(x:torch.Tensor, idx=1, *args, **kwargs) -> torch.Tensor:
        return x
    
    @staticmethod
    def decompose_magnitude_direction(x, dim=-1, eps=1e-8):
        magnitude = x.norm(p=2, dim=dim, keepdim=True)
        direction = x / (magnitude + eps)
        return magnitude, direction

    @staticmethod
    def scattersort_dir_(x, y, dim=-2):
        #buf = Stylizer.buffer
        #buf['src_sorted'], buf['src_idx'] = x.sort(dim=-2)
        #buf['ref_sorted'], buf['ref_idx'] = y.sort(dim=-2)
        #mag, _ = Stylizer.decompose_magnitude_direction(buf['src_sorted'], dim)
        #_, dir = Stylizer.decompose_magnitude_direction(buf['ref_sorted'], dim)
        mag, _ = Stylizer.decompose_magnitude_direction(x.to(torch.float64), dim)
        
        buf = Stylizer.buffer
        buf['src_idx']                    = x.argsort(dim=-2)
        buf['ref_sorted'], buf['ref_idx'] = y   .sort(dim=-2)
        x.scatter_(dim=-2, index=buf['src_idx'], src=buf['ref_sorted'].expand_as(buf['src_idx']))
        
        
        _, dir = Stylizer.decompose_magnitude_direction(x.to(torch.float64), dim)
        
        return (mag * dir).to(x)


    @staticmethod
    def scattersort_dir2_(x, y, dim=-2):
        #buf = Stylizer.buffer
        #buf['src_sorted'], buf['src_idx'] = x.sort(dim=-2)
        #buf['ref_sorted'], buf['ref_idx'] = y.sort(dim=-2)
        #mag, _ = Stylizer.decompose_magnitude_direction(buf['src_sorted'], dim)
        #_, dir = Stylizer.decompose_magnitude_direction(buf['ref_sorted'], dim)
        
        
        buf = Stylizer.buffer
        buf['src_sorted'], buf['src_idx'] = x.sort(dim=dim)
        buf['ref_sorted'], buf['ref_idx'] = y.sort(dim=dim)
        



        buf['x_sub'], buf['x_sub_idx'] = buf['src_sorted'].sort(dim=-1)
        buf['y_sub'], buf['y_sub_idx'] = buf['ref_sorted'].sort(dim=-1)
        
        mag, _ = Stylizer.decompose_magnitude_direction(buf['x_sub'].to(torch.float64), -1)
        _, dir = Stylizer.decompose_magnitude_direction(buf['y_sub'].to(torch.float64), -1)
        
        buf['y_sub'] = (mag * dir).to(x)
        
        buf['ref_sorted'].scatter_(dim=-1, index=buf['y_sub_idx'], src=buf['y_sub'].expand_as(buf['y_sub_idx']))



        mag, _ = Stylizer.decompose_magnitude_direction(buf['src_sorted'].to(torch.float64), dim)
        _, dir = Stylizer.decompose_magnitude_direction(buf['ref_sorted'].to(torch.float64), dim)
        
        buf['ref_sorted'] = (mag * dir).to(x)
        
        x.scatter_(dim=dim, index=buf['src_idx'], src=buf['ref_sorted'].expand_as(buf['src_idx']))

        return x


    @staticmethod
    def scattersort_dir(x, idx=1, slc=slice(None), *args, **kwargs):
        x[0:1] = Stylizer.scattersort_dir_(x[0:1], x[idx:idx+1])
        return x
    

    @staticmethod
    def scattersort_dir2(x, idx=1, slc=slice(None), *args, **kwargs):
        x[0:1] = Stylizer.scattersort_dir2_(x[0:1], x[idx:idx+1])
        return x



    @staticmethod
    def scattersort2(x, idx=1, slc=slice(None), *args, **kwargs):
        x[0:1] = Stylizer.scattersort2_(x[0:1], x[idx:idx+1])
        return x
    
    @staticmethod
    def scattersort2_(x, y, dim=-2):
        #buf = Stylizer.buffer
        #buf['src_sorted'], buf['src_idx'] = x.sort(dim=-2)
        #buf['ref_sorted'], buf['ref_idx'] = y.sort(dim=-2)
        #mag, _ = Stylizer.decompose_magnitude_direction(buf['src_sorted'], dim)
        #_, dir = Stylizer.decompose_magnitude_direction(buf['ref_sorted'], dim)
        
        
        buf = Stylizer.buffer
        buf['src_sorted'], buf['src_idx'] = x.sort(dim=dim)
        buf['ref_sorted'], buf['ref_idx'] = y.sort(dim=dim)
        



        buf['x_sub'], buf['x_sub_idx'] = buf['src_sorted'].sort(dim=-1)
        buf['y_sub'], buf['y_sub_idx'] = buf['ref_sorted'].sort(dim=-1)
        
        #mag, _ = Stylizer.decompose_magnitude_direction(buf['x_sub'].to(torch.float64), -1)
        #_, dir = Stylizer.decompose_magnitude_direction(buf['y_sub'].to(torch.float64), -1)
        #
        #buf['y_sub'] = (mag * dir).to(x)
        
        buf['ref_sorted'].scatter_(dim=-1, index=buf['y_sub_idx'], src=buf['y_sub'].expand_as(buf['y_sub_idx']))



        #mag, _ = Stylizer.decompose_magnitude_direction(buf['src_sorted'].to(torch.float64), dim)
        #_, dir = Stylizer.decompose_magnitude_direction(buf['ref_sorted'].to(torch.float64), dim)
        #
        #buf['ref_sorted'] = (mag * dir).to(x)
        
        x.scatter_(dim=dim, index=buf['src_idx'], src=buf['ref_sorted'].expand_as(buf['src_idx']))

        return x


    @staticmethod
    def scattersort_(x, y, slc=slice(None)):
        buf = Stylizer.buffer
        buf['src_idx']                    = x.argsort(dim=-2)
        buf['ref_sorted'], buf['ref_idx'] = y   .sort(dim=-2)

        return x.scatter_(dim=-2, index=buf['src_idx'][...,slc,:], src=buf['ref_sorted'][...,slc,:].expand_as(buf['src_idx'][...,slc,:]))
    

    @staticmethod
    def scattersort_double(x, y, *args, **kwargs):
        buf = Stylizer.buffer
        buf['src_sorted'], buf['src_idx'] = x.sort(dim=-2)
        buf['ref_sorted'], buf['ref_idx'] = y.sort(dim=-2)
        
        buf['x_sub_idx']               = buf['src_sorted'].argsort(dim=-1)
        buf['y_sub'], buf['y_sub_idx'] = buf['ref_sorted'].sort(dim=-1)
        
        x.scatter_(dim=-1, index=buf['x_sub_idx'], src=buf['y_sub'].expand_as(buf['x_sub_idx']))

        return x.scatter_(dim=-2, index=buf['src_idx'], src=buf['ref_sorted'].expand_as(buf['src_idx']))
    
    
    def scattersort_aoeu(self, x, idx=1, slc=slice(None)):
        x[0:1] = Stylizer.scattersort_(x[0:1], x[idx:idx+1], slc)
        return x
    
    def gram222_scattersort(self, x, idx=1, slc=slice(None), *args, **kwargs):
        if x.shape[0] != 2:
            x[0:1] = Stylizer.scattersort_(x[0:1], x[idx:idx+1], slc)
            return x
        
        buf = Stylizer.buffer
        buf['sorted'], buf['idx'] = x.sort(dim=-2)


        srt0 = buf['sorted'][0:1]
        srt1 = buf['sorted'][1:2]

        # 1. Center
        mean0 = srt0.mean(dim=-1, keepdim=True)
        mean1 = srt1.mean(dim=-1, keepdim=True)
        Xc = srt0 - mean0
        Yc = srt1 - mean1

        # 2. Compute covariances
        C0 = Xc @ Xc.transpose(-2, -1) / (srt0.shape[-1] - 1)
        C1 = Yc @ Yc.transpose(-2, -1) / (srt1.shape[-1] - 1)

        # 3. Eigen-decompose
        eigvals0, eigvecs0 = torch.linalg.eigh(C0)
        eigvals1, eigvecs1 = torch.linalg.eigh(C1)

        # 4. Whitening (remove style)
        eps = 1e-5
        diag0_inv_sqrt = torch.diag_embed((eigvals0 + eps).rsqrt())
        whiten = eigvecs0 @ diag0_inv_sqrt @ eigvecs0.transpose(-2, -1)
        X_white = whiten @ Xc

        # 5. Coloring (apply new style)
        diag1_sqrt = torch.diag_embed((eigvals1 + eps).sqrt())
        color = eigvecs1 @ diag1_sqrt @ eigvecs1.transpose(-2, -1)
        X_colored = color @ X_white + mean1

        buf['sorted'][1:2], _ = X_colored.sort(dim=-2)


        return x.scatter_(dim=-2, index=buf['idx'][0:1][...,slc,:], src=buf['sorted'][1:2][...,slc,:].expand_as(buf['idx'][0:1][...,slc,:]))
    
    def gram_scattersort(self, x, idx=1, slc=slice(None), *args, **kwargs):
        if x.shape[0] != 2:
            x[0:1] = Stylizer.scattersort_(x[0:1], x[idx:idx+1], slc)
            return x

        buf = Stylizer.buffer
        buf['sorted'], buf['idx'] = x.sort(dim=-2)

        srt0 = buf['sorted'][0:1]  # [1, H*W, C]
        srt1 = buf['sorted'][1:2]

        # Transpose to [1, C, H*W] for channel-wise whitening
        srt0_t = srt0.transpose(1, 2)
        srt1_t = srt1.transpose(1, 2)

        # 1. Center
        mean0 = srt0_t.mean(dim=-1, keepdim=True)
        mean1 = srt1_t.mean(dim=-1, keepdim=True)
        Xc = srt0_t - mean0  # [1, C, H*W]
        Yc = srt1_t - mean1

        # 2. Covariance: [C, C]
        C0 = Xc @ Xc.transpose(-2, -1) / (Xc.shape[-1] - 1)
        C1 = Yc @ Yc.transpose(-2, -1) / (Yc.shape[-1] - 1)

        # 3. Cholesky whitening/coloring
        eps = 1e-4
        eye = torch.eye(C0.shape[-1], device=x.device, dtype=x.dtype).expand(C0.shape[0], -1, -1)
        L0 = torch.linalg.cholesky(C0 + eps * eye)
        L1 = torch.linalg.cholesky(C1 + eps * eye)

        X_white = torch.cholesky_solve(Xc, L0)  # [1, C, H*W]
        X_colored = L1 @ X_white + mean1       # [1, C, H*W]

        # Transpose back to [1, H*W, C]
        X_colored = X_colored.transpose(1, 2)

        buf['sorted'][1:2], _ = X_colored.sort(dim=-2)
        
        #buf['sorted'][1:2] = X_colored

        return x.scatter_(
            dim=-2,
            index=buf['idx'][0:1][..., slc, :],
            src=buf['sorted'][1:2][..., slc, :].expand_as(buf['idx'][0:1][..., slc, :])
        )

    @staticmethod
    def orthogonal_procrustes_polar(Fc, Fr):
        """
        Fc, Fr: [B, N, C] content & reference feature batches
        returns: Fc warped by the optimal orthonormal T so that Fc·T ≈ Fr
        """
        # 1) center
        Fc0 = Fc - Fc.mean(dim=1, keepdim=True)
        Fr0 = Fr - Fr.mean(dim=1, keepdim=True)

        # 2) form M = Frᵀ @ Fc
        #    (we’ll do transpose on the last two dims)
        M = Fr0.transpose(-2, -1) @ Fc0  # → [B, C, C]

        # 3) polar decomposition M = Q·H  ⇒  Q is the orthonormal factor we want
        Q, _ = torch.linalg.polar(M)     # [B, C, C]

        # 4) apply the rotation back to the *right* of Fc
        return Fc @ Q.transpose(-2, -1)   # [B, N, C]

    @staticmethod
    def orthogonal_procrustes_eigh(Fc, Fr, eps=1e-6):
        # 1) center
        Fc0 = Fc - Fc.mean(dim=1, keepdim=True)
        Fr0 = Fr - Fr.mean(dim=1, keepdim=True)

        # 2) M = Frᵀ @ Fc
        M = Fr0.transpose(-2, -1) @ Fc0  # [B, C, C]

        # 3) form symmetric P = Mᵀ M
        P = M.transpose(-2, -1) @ M      # [B, C, C]

        # 4) eigen-decompose P = V·D·Vᵀ
        D, V = torch.linalg.eigh(P)      # D:[B,C], V:[B,C,C]

        # 5) compute P^{-1/2} = V · diag(1/√D) · Vᵀ
        inv_sqrt = torch.diag_embed(D.clamp(min=eps).rsqrt())  # [B,C,C]
        P_inv_sqrt = V @ inv_sqrt @ V.transpose(-2, -1)

        # 6) build the rotation T = M · P^{-1/2}
        T = M @ P_inv_sqrt               # [B, C, C]

        # 7) apply it
        return Fc @ T.transpose(-2, -1)  # [B, N, C]

    @staticmethod
    def orthogonal_procrustes(Fc, Fr):
        Fc = Fc - Fc.mean(dim=-2, keepdim=True)
        Fr = Fr - Fr.mean(dim=-2, keepdim=True)

        # [B, N, C] → transpose last two dims to match
        Fc_t = Fc.transpose(-2, -1)  # [B, C, N]
        Fr_t = Fr.transpose(-2, -1)  # [B, C, N]

        # Solve per batch element
        U, _, Vt = torch.linalg.svd(Fr_t @ Fc_t.transpose(-2, -1), full_matrices=False)
        T = U @ Vt

        # Apply transform
        Fc_styled = Fc @ T.transpose(-2, -1)
        Fc_styled = Fc_styled + Fr.mean(dim=-2, keepdim=True)

        return Fc_styled


    
    def scattercrust(self, x, idx=1, slc=slice(None), *args, **kwargs):
        if x.shape[0] != 2:
            x[0:1] = Stylizer.scattersort_(x[0:1], x[idx:idx+1], slc)
            return x
        
        x[0:1] = Stylizer.orthogonal_procrustes(x[0:1].to(torch.float32), x[1:2].to(torch.float32)).to(x)

        return x
    
    def scattercrust_polar(self, x, idx=1, slc=slice(None), *args, **kwargs):
        if x.shape[0] != 2:
            x[0:1] = Stylizer.scattersort_(x[0:1], x[idx:idx+1], slc)
            return x
        
        x[0:1] = Stylizer.orthogonal_procrustes_polar(x[0:1].to(torch.float32), x[1:2].to(torch.float32)).to(x)

        return x
    
    def scattercrust_eigh(self, x, idx=1, slc=slice(None), *args, **kwargs):
        if x.shape[0] != 2:
            x[0:1] = Stylizer.scattersort_(x[0:1], x[idx:idx+1], slc)
            return x
        
        x[0:1] = Stylizer.orthogonal_procrustes_eigh(x[0:1].to(torch.float32), x[1:2].to(torch.float32)).to(x)

        return x
    
    def scattersort(self, x, idx=1, slc=slice(None), *args, **kwargs):
        if x.shape[0] != 2:
            x[0:1] = Stylizer.scattersort_(x[0:1], x[idx:idx+1], slc)
            return x
        
        buf = Stylizer.buffer
        buf['sorted'], buf['idx'] = x.sort(dim=-2)

        return x.scatter_(dim=-2, index=buf['idx'][0:1][...,slc,:], src=buf['sorted'][1:2][...,slc,:].expand_as(buf['idx'][0:1][...,slc,:]))
    
    def swappersort(self, x, idx=1, slc=slice(None), *args, **kwargs):
        if x.shape[0] != 2:
            x[0:1] = Stylizer.scattersort_(x[0:1], x[idx:idx+1], slc)
            return x
        
        buf = Stylizer.buffer
        buf['sorted'], buf['idx'] = x.sort(dim=-2)

        x[0:1] = x[0:1].scatter_(dim=-2, index=buf['idx'][0:1][...,slc,:], src=buf['sorted'][1:2][...,slc,:].expand_as(buf['idx'][0:1][...,slc,:]))
        #x[1:2] = x[1:2].scatter_(dim=-2, index=buf['idx'][1:2][...,slc,:], src=buf['sorted'][0:1][...,slc,:].expand_as(buf['idx'][1:2][...,slc,:]))
        return x
    

        #def haar_scattersort(self, denoised_embed: torch.Tensor, h_len, w_len, stylize_highfreq=False):
    def haar_scattersort(self, x, idx=1, slc=slice(None), stylize_highfreq=False, *args, **kwargs):

        B, HW, C = x.shape
        
        x_spatial = x.contiguous().view(B, C, self.h_len, self.w_len)
    
        LL, LH, HL, HH = haar_wavelet_decompose(x_spatial)

        def process_band(band, idx=1):
            Bc, Cc, Hc, Wc = band.shape
            flat = band.contiguous().view(Bc, Hc * Wc, Cc)
            
            styled = self.scattersort(band, idx=idx)
            return styled.contiguous().view(Bc, Cc, Hc, Wc)

        LL_styled = process_band(LL)

        if stylize_highfreq:
            LH_styled = process_band(LH)
            HL_styled = process_band(HL)
            HH_styled = process_band(HH)
        else:
            LH_styled, HL_styled, HH_styled = LH, HL, HH

        x_spatial = haar_wavelet_reconstruct(LL_styled, LH_styled, HL_styled, HH_styled)
        #x_spatial[0] = recon.squeeze(0)

        return x_spatial.view(B, HW, C).to(x)

    
    #def channelwise_nn_lookup(self, content: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    @staticmethod
    def channelwise_nn_lookup(x: torch.Tensor, idx=1, slc=slice(None), *args, **kwargs) -> torch.Tensor:

        """
        For each element in `content` (shape [B, N, C]), find the closest value
        in `reference` (shape [B, M, C]) along dim=1, *allowing reuse*, channel-wise.

        Returns: Tensor of shape [B, N, C] of nearest-neighbor values.
        """
        content = x[0:1]
        reference = x[1:]      #reference = x[1:2]
        B, N, C = content.shape
        M = reference.shape[1]

        # 1) Bring channels to leading dim: [B, C, *] → flatten to [B*C, *]
        #    we want content_flat[i] to correspond to channel c of batch b.
        content_flat = content.permute(0, 2, 1).reshape(-1, N)    # [B*C, N]
        ref_flat     = reference.permute(0, 2, 1).reshape(-1, M)  # [B*C, M]

        # 2) Sort each reference row:
        sorted_ref, _ = torch.sort(ref_flat, dim=1)  # [B*C, M]

        # 3) For each content value, find where it would be inserted in sorted_ref:
        #    `idx` has shape [B*C, N], values in [0..M]
        idx = torch.searchsorted(sorted_ref, content_flat)

        # 4) For each insertion idx, the two nearest candidates are at idx-1 and idx:
        idx_lo = torch.clamp(idx - 1, min=0)      # no lower than 0
        idx_hi = torch.clamp(idx, max=M - 1)      # no higher than M-1

        # 5) Gather those two candidates:
        val_lo = sorted_ref.gather(1, idx_lo)     # [B*C, N]
        val_hi = sorted_ref.gather(1, idx_hi)     # [B*C, N]

        # 6) Pick the closer one:
        dist_lo = (content_flat - val_lo).abs()
        dist_hi = (val_hi - content_flat).abs()
        use_hi  = dist_hi < dist_lo

        matched_flat = torch.where(use_hi, val_hi, val_lo)  # [B*C, N]

        # 7) Reshape back to [B, N, C]
        x[0:1] = matched_flat.reshape(B, C, N).permute(0, 2, 1)
        return x

    def lookup_flipped_adain(self, x, idx=1, slc=slice(None), *args, **kwargs):
        #if x.shape[0] != 2:
        #    x[0:1] = Stylizer.scattersort_(x[0:1], x[idx:idx+1], slc)
        #    return x
        x_orig = x.clone()
        x = self.AdaIN(x, idx)
        
        x[1:2] = x_orig[0:1]
        
        x = Stylizer.channelwise_nn_lookup(x, idx)
        
        x[1:2] = x_orig[1:2]
        
        return x
        
    def lookup(self, x, idx=1, slc=slice(None), *args, **kwargs):
        #if x.shape[0] != 2:
        #    x[0:1] = Stylizer.scattersort_(x[0:1], x[idx:idx+1], slc)
        #    return x        
        x = Stylizer.channelwise_nn_lookup(x, idx)
        
        return x
        
    
    def tiled_scattersort(self, x, idx=1, *args, **kwargs): #, h_tile=None, w_tile=None):
        #if HDModel.RECON_MODE:
        #    return denoised_embed
        #den   = x[0:1]      [:,:self.img_len,:].view(-1, 2560, self.h_len, self.w_len)
        #style = x[idx:idx+1][:,:self.img_len,:].view(-1, 2560, self.h_len, self.w_len)
        #h_tile = self.h_tile[idx-1] if h_tile is None else h_tile
        #w_tile = self.w_tile[idx-1] if w_tile is None else w_tile
        
        C = x.shape[-1]
        den   = x[0:1]      [:,self.img_slice,:].reshape(-1, C, self.h_len, self.w_len)
        style = x[idx:idx+1][:,self.img_slice,:].reshape(-1, C, self.h_len, self.w_len)
        
        tiles     = Stylizer.get_tiles_as_strided(den,   self.h_tile[idx-1], self.w_tile[idx-1])
        ref_tile  = Stylizer.get_tiles_as_strided(style, self.h_tile[idx-1], self.w_tile[idx-1])

        # rearrange for vmap to run on (nH, nW) ( as outer axes)
        tiles_v    = tiles   .permute(2, 3, 0, 1, 4, 5) # (nH, nW, B, C, tile_h, tile_w)
        ref_tile_v = ref_tile.permute(2, 3, 0, 1, 4, 5) # (nH, nW, B, C, tile_h, tile_w)

        # vmap over spatial dimms (nH, nW)... num of tiles high, num tiles wide
        vmap2   = torch.vmap(torch.vmap(Stylizer.apply_scattersort_per_tile, in_dims=0), in_dims=0)
        result  = vmap2(tiles_v, ref_tile_v)  # (nH, nW, B, C, tile_h, tile_w)

        # --> (B, C, nH, nW, tile_h, tile_w)
        result = result.permute(2, 3, 0, 1, 4, 5)  #( B, C, nH, nW, tile_h, tile_w)

        # in-place copy, werx if result has same shape/strides as tiles... overwrites same mem location "content" is using
        tiles.copy_(result)

        return x
    
    
    def tiled_AdaIN(self, x, idx=1, *args, **kwargs):
        #if HDModel.RECON_MODE:
        #    return denoised_embed
        #den   = x[0:1]      [:,:self.img_len,:].view(-1, 2560, self.h_len, self.w_len)
        #style = x[idx:idx+1][:,:self.img_len,:].view(-1, 2560, self.h_len, self.w_len)
        C = x.shape[-1]
        den   = x[0:1]      [:,self.img_slice,:].reshape(-1, C, self.h_len, self.w_len)
        style = x[idx:idx+1][:,self.img_slice,:].reshape(-1, C, self.h_len, self.w_len)
        
        tiles     = Stylizer.get_tiles_as_strided(den,   self.h_tile[idx-1], self.w_tile[idx-1])
        ref_tile  = Stylizer.get_tiles_as_strided(style, self.h_tile[idx-1], self.w_tile[idx-1])

        # rearrange for vmap to run on (nH, nW) ( as outer axes)
        tiles_v    = tiles   .permute(2, 3, 0, 1, 4, 5) # (nH, nW, B, C, tile_h, tile_w)
        ref_tile_v = ref_tile.permute(2, 3, 0, 1, 4, 5) # (nH, nW, B, C, tile_h, tile_w)

        # vmap over spatial dimms (nH, nW)... num of tiles high, num tiles wide
        vmap2   = torch.vmap(torch.vmap(Stylizer.apply_AdaIN_per_tile, in_dims=0), in_dims=0)
        result  = vmap2(tiles_v, ref_tile_v)  # (nH, nW, B, C, tile_h, tile_w)

        # --> (B, C, nH, nW, tile_h, tile_w)
        result = result.permute(2, 3, 0, 1, 4, 5)  #( B, C, nH, nW, tile_h, tile_w)

        # in-place copy, werx if result has same shape/strides as tiles... overwrites same mem location "content" is using
        tiles.copy_(result)

        return x
    
    
    @staticmethod
    def get_tiles_as_strided(x, tile_h, tile_w):
        B, C, H, W = x.shape
        stride = x.stride()
        nH = H // tile_h
        nW = W // tile_w

        tiles = x.as_strided(
            size=(B, C, nH, nW, tile_h, tile_w),
            stride=(stride[0], stride[1], stride[2] * tile_h, stride[3] * tile_w, stride[2], stride[3])
        )
        return tiles  # shape: (B, C, nH, nW, tile_h, tile_w)

    @staticmethod
    def apply_scattersort_per_tile(tile, ref_tile):
        flat     = tile    .flatten(-2, -1)
        ref_flat = ref_tile.flatten(-2, -1)

        sorted_ref, _ = ref_flat  .sort(dim=-1)
        src_sorted, src_idx = flat.sort(dim=-1)
        
        out = flat.scatter(dim=-1, index=src_idx, src=sorted_ref)
        return out.view_as(tile)

    @staticmethod
    def apply_AdaIN_per_tile(tile, ref_tile, eps: float = 1e-7):
        mean_c = tile.mean(-2, keepdim=True)
        std_c  = tile.std (-2, keepdim=True).add_(eps)  # in-place add
        mean_s = ref_tile.mean  (-2, keepdim=True)
        std_s  = ref_tile.std   (-2, keepdim=True).add_(eps)
        tile.sub_(mean_c).div_(std_c).mul_(std_s).add_(mean_s)  # in-place chain
        return tile

class StyleMMDiT_Attn(Stylizer):
    def __init__(self, mode):
        super().__init__()
        
        self.qkv    = [0.0]
        
        self.q_proj = [0.0]
        self.k_proj = [0.0]
        self.v_proj = [0.0]

        self.q_norm = [0.0]
        self.k_norm = [0.0]
        
        self.out    = [0.0]

class StyleMMDiT_FF(Stylizer): # these hit img or joint only, never txt
    def __init__(self, mode):
        super().__init__()
    
        self.ff_1      = [0.0]
        self.ff_1_silu = [0.0]
        self.ff_3      = [0.0]
        self.ff_13     = [0.0]
        self.ff_2      = [0.0]
        
class StyleMMDiT_MoE(Stylizer): # these hit img or joint only, never txt
    def __init__(self, mode):
        super().__init__()
        
        self.FF_SHARED   = StyleMMDiT_FF(mode)
        self.FF_SEPARATE = StyleMMDiT_FF(mode)
        
        self.shared      = [0.0]
        self.gate        = [False]
        self.topk_weight = [0.0]

        self.separate    = [0.0]
        self.sum         = [0.0]
        self.out         = [0.0]





class StyleMMDiT_SubBlock(Stylizer):
    def __init__(self, mode):
        super().__init__()
        
        self.ATTN = StyleMMDiT_Attn(mode)  # options for attn itself: qkv proj, qk norm, attn out

        self.attn_norm     = [0.0]
        self.attn_norm_mod = [0.0]
        self.attn          = [0.0]
        self.attn_gated    = [0.0]
        self.attn_res      = [0.0]
        
        self.ff_norm       = [0.0]
        self.ff_norm_mod   = [0.0]
        self.ff            = [0.0]
        self.ff_gated      = [0.0]
        self.ff_res        = [0.0]
        
        self.mask = [None]
        
    def set_len(self, h_len, w_len, img_slice, txt_slice, HEADS):
        super().set_len(h_len, w_len, img_slice, txt_slice, HEADS)
        self.ATTN.set_len(h_len, w_len, img_slice, txt_slice, HEADS)

class StyleMMDiT_IMG_Block(StyleMMDiT_SubBlock):  # img or joint
    def __init__(self, mode):
        super().__init__(mode)
        self.FF = StyleMMDiT_MoE(mode)  # options for MoE if img or joint
    
    def set_len(self, h_len, w_len, img_slice, txt_slice, HEADS):
        super().set_len(h_len, w_len, img_slice, txt_slice, HEADS)
        self.FF.set_len(h_len, w_len, img_slice, txt_slice, HEADS)
        
class StyleMMDiT_TXT_Block(StyleMMDiT_SubBlock):   # txt only
    def __init__(self, mode):
        super().__init__(mode)
        self.FF  = StyleMMDiT_FF(mode)   # options for FF within MoE for img or joint; or for txt alone
    
    def set_len(self, h_len, w_len, img_slice, txt_slice, HEADS):
        super().set_len(h_len, w_len, img_slice, txt_slice, HEADS)
        self.FF.set_len(h_len, w_len, img_slice, txt_slice, HEADS)





class StyleMMDiT_BaseBlock:
    def __init__(self, mode="passthrough"):

        self.img = StyleMMDiT_IMG_Block(mode)
        self.txt = StyleMMDiT_TXT_Block(mode)
        
        self.mask      = [None]
        self.attn_mask = [None]
    
    def set_len(self, h_len, w_len, img_slice, txt_slice, HEADS):
        self.h_len  = h_len
        self.w_len  = w_len
        self.img_len = h_len * w_len
        
        self.img_slice = img_slice
        self.txt_slice = txt_slice
        self.HEADS = HEADS
        
        self.img.set_len(h_len, w_len, img_slice, txt_slice, HEADS)
        self.txt.set_len(-1, -1, img_slice, txt_slice, HEADS)
        
        for i, mask in enumerate(self.mask):
            if mask is not None and mask.ndim > 1:
                self.mask[i] = F.interpolate(mask.unsqueeze(0), size=(h_len, w_len)).flatten().to(torch.bfloat16).cuda()
            self.img.mask = self.mask
        for i, mask in enumerate(self.attn_mask):
            if mask is not None and mask.ndim > 1:
                self.attn_mask[i] = F.interpolate(mask.unsqueeze(0), size=(h_len, w_len)).flatten().to(torch.bfloat16).cuda()
            self.img.ATTN.mask = self.attn_mask      

class StyleMMDiT_DoubleBlock(StyleMMDiT_BaseBlock):
    def __init__(self, mode="passthrough"):
        super().__init__(mode)
        self.txt = StyleMMDiT_TXT_Block(mode)
    
    def set_len(self, h_len, w_len, img_slice, txt_slice, HEADS):
        super().set_len(h_len, w_len, img_slice, txt_slice, HEADS)
        self.txt.set_len(-1, -1, img_slice, txt_slice, HEADS)

class StyleMMDiT_SingleBlock(StyleMMDiT_BaseBlock):
    def __init__(self, mode="passthrough"):
        super().__init__(mode)




























class StyleUNet_Resample(Stylizer):
    def __init__(self, mode):
        super().__init__()
        self.conv = [0.0]

class StyleUNet_Attn(Stylizer):
    def __init__(self, mode):
        super().__init__()
        self.q_proj = [0.0]
        self.k_proj = [0.0]
        self.v_proj = [0.0]
        self.out    = [0.0]

class StyleUNet_FF(Stylizer):
    def __init__(self, mode):
        super().__init__()
        self.proj   = [0.0]
        self.geglu  = [0.0]
        self.linear = [0.0]
        
class StyleUNet_TransformerBlock(Stylizer): 
    def __init__(self, mode):
        super().__init__()
        
        self.ATTN1 = StyleUNet_Attn(mode)  # self-attn
        self.FF    = StyleUNet_FF  (mode)  
        self.ATTN2 = StyleUNet_Attn(mode)  # cross-attn

        self.self_attn  = [0.0]
        self.ff         = [0.0]
        self.cross_attn = [0.0]
        
        self.self_attn_res  = [0.0]
        self.cross_attn_res = [0.0]
        self.ff_res = [0.0]
        
        self.norm1 = [0.0]
        self.norm2 = [0.0]
        self.norm3 = [0.0]
        
    def set_len(self, h_len, w_len, img_slice, txt_slice, HEADS):
        super().set_len(h_len, w_len, img_slice, txt_slice, HEADS)
        self.ATTN1.set_len(h_len, w_len, img_slice, txt_slice, HEADS)
        self.ATTN2.set_len(h_len, w_len, img_slice, txt_slice, HEADS)

class StyleUNet_SpatialTransformer(Stylizer): 
    def __init__(self, mode):
        super().__init__()
        
        self.TFMR = StyleUNet_TransformerBlock(mode)

        self.spatial_norm_in     = [0.0]
        self.spatial_proj_in     = [0.0]
        self.spatial_transformer_block = [0.0]
        self.spatial_transformer = [0.0]
        self.spatial_proj_out    = [0.0]
        self.spatial_res         = [0.0]
        
    def set_len(self, h_len, w_len, img_slice, txt_slice, HEADS):
        super().set_len(h_len, w_len, img_slice, txt_slice, HEADS)
        self.TFMR.set_len(h_len, w_len, img_slice, txt_slice, HEADS)

class StyleUNet_ResBlock(Stylizer):
    def __init__(self, mode):
        super().__init__()

        self.in_norm    = [0.0]
        self.in_silu    = [0.0]
        self.in_conv    = [0.0]

        self.emb_silu   = [0.0]
        self.emb_linear = [0.0]
        self.emb_res    = [0.0]

        self.out_norm   = [0.0]
        self.out_silu   = [0.0]
        self.out_conv   = [0.0]
        
        self.residual   = [0.0]


class StyleUNet_BaseBlock(Stylizer):
    def __init__(self, mode="passthrough"):

        self.resample_block = StyleUNet_Resample(mode)
        self.res_block      = StyleUNet_ResBlock(mode)
        self.spatial_block  = StyleUNet_SpatialTransformer(mode)
        
        self.resample = [0.0]
        self.res      = [0.0]
        self.spatial  = [0.0]
        
        self.mask      = [None]
        self.attn_mask = [None]
        
        self.KONTEXT = 0

    
    def set_len(self, h_len, w_len, img_slice, txt_slice, HEADS):
        self.h_len  = h_len
        self.w_len  = w_len
        self.img_len = h_len * w_len
        
        self.img_slice = img_slice
        self.txt_slice = txt_slice
        self.HEADS = HEADS
        
        self.resample_block.set_len(h_len, w_len, img_slice, txt_slice, HEADS)
        self.res_block     .set_len(h_len, w_len, img_slice, txt_slice, HEADS)
        self.spatial_block .set_len(h_len, w_len, img_slice, txt_slice, HEADS)
        
        for i, mask in enumerate(self.mask):
            if mask is not None and mask.ndim > 1:
                self.mask[i] = F.interpolate(mask.unsqueeze(0), size=(h_len, w_len)).flatten().to(torch.bfloat16).cuda()
            self.resample_block.mask = self.mask
            self.res_block.mask      = self.mask
            self.spatial_block.mask  = self.mask
            self.spatial_block.TFMR.mask  = self.mask
            
        for i, mask in enumerate(self.attn_mask):
            if mask is not None and mask.ndim > 1:
                self.attn_mask[i] = F.interpolate(mask.unsqueeze(0), size=(h_len, w_len)).flatten().to(torch.bfloat16).cuda()
            self.spatial_block.TFMR.ATTN1.mask = self.attn_mask     
            
    def __call__(self, x, attr):
        B, C, H, W = x.shape
        x = super().__call__(x.reshape(B, H*W, C), attr)
        return x.reshape(B,C,H,W)
        

class StyleUNet_InputBlock(StyleUNet_BaseBlock):
    def __init__(self, mode="passthrough"):
        super().__init__(mode)    

class StyleUNet_MiddleBlock(StyleUNet_BaseBlock):
    def __init__(self, mode="passthrough"):
        super().__init__(mode)

class StyleUNet_OutputBlock(StyleUNet_BaseBlock):
    def __init__(self, mode="passthrough"):
        super().__init__(mode)
















class Style_Model(Stylizer):

    def __init__(self, dtype=torch.float64, device=torch.device("cuda")):
        super().__init__(dtype, device)
        self.guides = []
        self.GUIDES_INITIALIZED = False
        
        #self.double_blocks = [StyleMMDiT_DoubleBlock() for _ in range(100)]
        #self.single_blocks = [StyleMMDiT_SingleBlock() for _ in range(100)]
        
        self.h_len   = -1
        self.w_len   = -1
        self.img_len = -1
        self.h_tile  = [-1]
        self.w_tile  = [-1]
        
        self.proj_in  = [0.0]  # these are for img only! not sliced
        self.proj_out = [0.0]
        
        self.cond_pos = [None]
        self.cond_neg = [None]
        
        self.noise_mode = "update"
        self.recon_lure = "none"
        self.data_shock = "none"
        
        self.recon_lure_weight = 0.0
        self.data_shock_weight = 0.0
        
        self.data_shock_start_step = 0
        self.data_shock_end_step   = 0
        
        self.Retrojector = None
        self.Endojector  = None
        
        self.IMG_1ST = True
        self.HEADS = 0
        self.KONTEXT = 0
    def __call__(self, x, attr):
        if x.shape[0] == 1 and not self.KONTEXT:
            return x
        
        weight_list = getattr(self, attr)
        weights_all_zero = all(weight == 0.0 for weight in weight_list)
        if weights_all_zero:
            return x
        
        """x_ndim = x.ndim
        if x_ndim == 4:
            B, HEAD, HW, C = x.shape
            
        if x_ndim == 3:
            B, HW, C = x.shape
            if x.shape[-2] != self.HEADS and self.HEADS != 0:
                x = x.reshape(B,self.HEADS,HW,-1)"""
        
        HEAD_DIM = x.shape[1]
        if HEAD_DIM == self.HEADS:
            B, HEAD_DIM, HW, C = x.shape
            x = x.reshape(B, HW, C*HEAD_DIM)  # TODO: FIX WITH PERMUTE SHIT
            
        if self.KONTEXT == 1:
            x = x.reshape(2, x.shape[1] // 2, x.shape[2])
            
        weights_all_one         = all(weight == 1.0           for weight in weight_list)
        weights_all_same = all(weight == weight_list[0] for weight in weight_list)
        methods_all_scattersort = all(name   == "scattersort" for name   in self.method)
        masks_all_none = all(mask is None for mask in self.mask)
        
        if weights_all_same and methods_all_scattersort and len(weight_list) > 1 and masks_all_none:
            buf = Stylizer.buffer
            buf['src_idx']   = x[0:1].argsort(dim=-2)
            buf['ref_sorted'], buf['ref_idx'] = x[1:].reshape(1, -1, x.shape[-1]).sort(dim=-2)
            buf['src'] = buf['ref_sorted'][:,::len(weight_list)].expand_as(buf['src_idx'])    #            interleave_stride = len(weight_list)
            
            #x[0:1] = x[0:1].scatter_(dim=-2, index=buf['src_idx'], src=buf['src'],)
            slc = Stylizer.middle_slice(buf['src'].shape[-2], weight_list[0]) 
            
            x[0:1] = x[0:1].scatter_(dim=-2, index=buf['src_idx'][...,slc,:], src=buf['src'][...,slc,:],)
        else:
            for i, (weight, mask) in enumerate(zip(weight_list, self.mask)):
                if weight > 0 and weight < 1:
                    x_clone = x.clone()
                if mask is not None:
                    x01 = x[0:1].clone()
                slc = Stylizer.middle_slice(x.shape[-2], abs(weight))
                if weight < 0:
                    x_base = x.clone()   
                
                method = getattr(self, self.method[i])
                if   weight == 0.0:
                    continue
                elif weight == 1.0:
                    x = method(x, idx=i+1)
                else:
                    x = method(x, idx=i+1, slc=slc)
                if weight > 0 and weight < 1 and self.method[i] != "scattersort":
                    x = torch.lerp(x_clone, x, weight)
                    
                #else:
                #    x = torch.lerp(x, method(x.clone(), idx=i), weight)
                
                if mask is not None:
                    x[0:1] = torch.lerp(x01, x[0:1], mask.view(1, -1, 1))
        
                if weight < 0:
                    if self.method[i] == "scattersort":
                        x = 2 * x_base - x
                    else:
                        x = x_base + abs(weight) * (x_base - x)
                                
        #if x_ndim == 3:
        #    return x.view(B,HW,C)
        if self.KONTEXT == 1:
            x = x.reshape(1, x.shape[1] * 2, x.shape[2])
            
        if HEAD_DIM == self.HEADS:
            return x.reshape(B, HEAD_DIM, HW, C)
        else:
            return x

    def set_len(self, h_len, w_len, img_slice, txt_slice, HEADS):
        self.h_len  = h_len
        self.w_len  = w_len
        self.img_len = h_len * w_len
        
        self.img_slice = img_slice
        self.txt_slice = txt_slice
        self.HEADS = HEADS
        
        #for block in self.double_blocks:
        #    block.set_len(h_len, w_len, img_slice, txt_slice, HEADS)
        #for block in self.single_blocks:
        #    block.set_len(h_len, w_len, img_slice, txt_slice, HEADS)
        
        for i, mask in enumerate(self.mask):
            if mask is not None and mask.ndim > 1:
                self.mask[i] = F.interpolate(mask.unsqueeze(0), size=(h_len, w_len)).flatten().to(torch.bfloat16).cuda()

    def init_guides(self, model):
        if not self.GUIDES_INITIALIZED:
            if self.guides == []:
                self.guides = None
            elif self.guides is not None:
                for i, latent in enumerate(self.guides):
                    if type(latent) is dict:
                        latent = model.inner_model.inner_model.process_latent_in(latent['samples']).to(dtype=self.dtype, device=self.device)
                    elif type(latent) is torch.Tensor:
                        latent = latent.to(dtype=self.dtype, device=self.device)
                    else:
                        latent = None
                        #raise ValueError(f"Invalid latent type: {type(latent)}")

                    #if self.VIDEO and latent.shape[2] == 1:
                    #    latent = latent.repeat(1, 1, x.shape[2], 1, 1)

                    self.guides[i] = latent
                if any(g is None for g in self.guides):
                    self.guides = None
                    print("Style guide nonetype set for Kontext.")
                else:
                    self.guides = torch.cat(self.guides, dim=0)
            self.GUIDES_INITIALIZED = True
    
    def set_conditioning(self, positive, negative):
        self.cond_pos = [positive]
        self.cond_neg = [negative] 

    def apply_style_conditioning(self, UNCOND, base_context, base_y=None, base_llama3=None):

        def get_max_token_lengths(style_conditioning, base_context, base_y=None, base_llama3=None):
            context_max_len = base_context.shape[-2]
            llama3_max_len  = base_llama3.shape[-2]  if base_llama3 is not None else -1
            y_max_len       = base_y.shape[-1]       if base_y      is not None else -1

            for style_cond in style_conditioning:
                if style_cond is None:
                    continue
                context_max_len = max(context_max_len, style_cond[0][0].shape[-2])
                if base_llama3 is not None:
                    llama3_max_len  = max(llama3_max_len,  style_cond[0][1]['conditioning_llama3'].shape[-2])
                if base_y is not None:
                    y_max_len       = max(y_max_len,       style_cond[0][1]['pooled_output'].shape[-1])

            return context_max_len, llama3_max_len, y_max_len

        def pad_to_len(x, target_len, pad_value=0.0, dim=1):
            if target_len < 0:
                return x
            cur_len = x.shape[dim]
            if cur_len == target_len:
                return x
            return F.pad(x, (0, 0, 0, target_len - cur_len), value=pad_value)

        style_conditioning = self.cond_pos if not UNCOND else self.cond_neg
        
        context_max_len, llama3_max_len, y_max_len = get_max_token_lengths(
            style_conditioning = style_conditioning,
            base_context       = base_context,
            base_y             = base_y,
            base_llama3        = base_llama3,
        )
        
        bsz_style = len(style_conditioning)
        
        context = base_context.repeat(bsz_style + 1, 1, 1)
        y = base_y.repeat(bsz_style + 1, 1)                   if base_y      is not None else None
        llama3  =  base_llama3.repeat(bsz_style + 1, 1, 1, 1) if base_llama3 is not None else None

        context = pad_to_len(context, context_max_len, dim=-2)
        llama3  = pad_to_len(llama3, llama3_max_len, dim=-2)   if base_llama3 is not None else None
        y       = pad_to_len(y,      y_max_len, dim=-1)        if base_y      is not None else None
        
        for ci, style_cond in enumerate(style_conditioning):
            if style_cond is None:
                continue
            context[ci+1:ci+2] = pad_to_len(style_cond[0][0], context_max_len, dim=-2).to(context)
            if llama3 is not None:
                llama3 [ci+1:ci+2] = pad_to_len(style_cond[0][1]['conditioning_llama3'], llama3_max_len, dim=-2).to(llama3)
            if y is not None:
                y      [ci+1:ci+2] = pad_to_len(style_cond[0][1]['pooled_output'],       y_max_len, dim=-1).to(y)
        
        return context, y, llama3
    
    def WCT_data(self, denoised_embed, y0_style_embed):
        Stylizer.CLS_WCT.set(y0_style_embed.to(denoised_embed))
        return Stylizer.CLS_WCT.get(denoised_embed)

    def WCT2_data(self, denoised_embed, y0_style_embed):
        Stylizer.CLS_WCT2.set(y0_style_embed.to(denoised_embed))
        return Stylizer.CLS_WCT2.get(denoised_embed)

    def apply_to_data(self, denoised, y0_style=None, mode="none"):
        if mode == "none":
            return denoised
        y0_style = self.guides if y0_style is None else y0_style
        
        y0_style_embed = self.Retrojector.embed(y0_style)
        denoised_embed = self.Retrojector.embed(denoised)
        B,HW,C = y0_style_embed.shape
        embed  = torch.cat([denoised_embed, y0_style_embed.view(1,B*HW,C)[:,::B,:]], dim=0)
        method = getattr(self, mode)
        if mode == "scattersort":
            slc = Stylizer.middle_slice(embed.shape[-2], self.data_shock_weight)
            embed = method(embed, slc=slc)
        else:
            embed  = method(embed)
        return self.Retrojector.unembed(embed[0:1])

    def apply_to_data2_basic(self, denoised, y0_style=None, mode="none"):
        if mode == "none":
            return denoised
        y0_style = self.guides if y0_style is None else y0_style
        
        y0_style_embed = self.Retrojector.embed(y0_style)
        denoised_embed = self.Retrojector.embed(denoised)
        y0_style_embed = self.Retrojector2.embed(y0_style_embed)
        denoised_embed = self.Retrojector2.embed(denoised_embed)
        B,HW,C = y0_style_embed.shape
        embed  = torch.cat([denoised_embed, y0_style_embed.view(1,B*HW,C)[:,::B,:]], dim=0)
        method = getattr(self, mode)
        if mode == "scattersort":
            slc = Stylizer.middle_slice(embed.shape[-2], self.data_shock_weight)
            embed = method(embed, slc=slc)
        else:
            embed  = method(embed)
        unembed = self.Retrojector2.unembed(embed[0:1])
        return self.Retrojector.unembed(unembed)


    def apply_to_data2(self, denoised, y0_style=None, mode="none"):
        if mode == "none":
            return denoised
        y0_style = self.guides if y0_style is None else y0_style
        
        y0_style_embed = self.Retrojector.embed(y0_style)
        denoised_embed = self.Retrojector.embed(denoised)
        
        #y0_style_embed = self.FV.norm(y0_style_embed)
        #y0_style_embed_norm_cache = y0_style_embed.clone()
        #denoised_embed = self.FV.norm(denoised_embed)
        
        #y0_style_embed = self.FV.mod(y0_style_embed)
        #denoised_embed = self.FV.mod(denoised_embed)
        
        y0_style_embed = self.Retrojector2.embed(y0_style_embed)
        denoised_embed = self.Retrojector2.embed(denoised_embed)
        B,HW,C = y0_style_embed.shape
        embed  = torch.cat([denoised_embed, y0_style_embed.view(1,B*HW,C)[:,::B,:]], dim=0)
        method = getattr(self, mode)
        if mode == "scattersort":
            slc = Stylizer.middle_slice(embed.shape[-2], self.data_shock_weight)
            embed = method(embed, slc=slc)
        else:
            embed  = method(embed)
        unembed = self.Retrojector2.unembed(embed[0:1])
        
        #unembed = self.FV.unmod(unembed)
        #unembed = self.FV.unnorm(unembed, y0_style_embed_norm_cache)
        
        return self.Retrojector.unembed(unembed)





    def apply_recon_lure(self, denoised, y0_style):
        if self.recon_lure == "none":
            return denoised
        for i in range(denoised.shape[0]):
            denoised[i:i+1] = self.apply_to_data(denoised[i:i+1], y0_style, self.recon_lure)
        return denoised

    def apply_data_shock(self, denoised):
        if self.data_shock == "none":
            return denoised
        datashock_ref = getattr(self, "datashock_ref", None)
        if self.data_shock == "scattersort":
            return self.apply_to_data(denoised, datashock_ref, self.data_shock)
        else:
            return torch.lerp(denoised, self.apply_to_data(denoised, datashock_ref, self.data_shock), torch.Tensor([self.data_shock_weight]).double().cuda())




class StyleMMDiT_Model(Style_Model):

    def __init__(self, dtype=torch.float64, device=torch.device("cuda")):
        super().__init__(dtype, device)
        self.double_blocks = [StyleMMDiT_DoubleBlock() for _ in range(100)]
        self.single_blocks = [StyleMMDiT_SingleBlock() for _ in range(100)]

    def set_len(self, h_len, w_len, img_slice, txt_slice, HEADS):
        super().set_len(h_len, w_len, img_slice, txt_slice, HEADS)
        for block in self.double_blocks:
            block.set_len(h_len, w_len, img_slice, txt_slice, HEADS)
        for block in self.single_blocks:
            block.set_len(h_len, w_len, img_slice, txt_slice, HEADS)


class StyleUNet_Model(Style_Model):

    def __init__(self, dtype=torch.float64, device=torch.device("cuda")):
        super().__init__(dtype, device)
        self.input_blocks  = [StyleUNet_InputBlock()  for _ in range(100)]
        self.middle_blocks = [StyleUNet_MiddleBlock() for _ in range(100)]
        self.output_blocks = [StyleUNet_OutputBlock() for _ in range(100)]

    def set_len(self, h_len, w_len, img_slice, txt_slice, HEADS):
        super().set_len(h_len, w_len, img_slice, txt_slice, HEADS)
        for block in self.input_blocks:
            block.set_len(h_len, w_len, img_slice, txt_slice, HEADS)
        for block in self.middle_blocks:
            block.set_len(h_len, w_len, img_slice, txt_slice, HEADS)
        for block in self.output_blocks:
            block.set_len(h_len, w_len, img_slice, txt_slice, HEADS)

    def __call__(self, x, attr):
        B, C, H, W = x.shape
        x = super().__call__(x.reshape(B, H*W, C), attr)
        return x.reshape(B,C,H,W)


import torch_dct


def dct_2d_torch_dct(x):
    # x: [B, C, H, W]
    # Apply DCT along W (last dim)
    x = torch_dct.dct(x, norm='ortho')

    # Apply DCT along H (second to last dim) — need to permute
    x = x.transpose(-1, -2)
    x = torch_dct.dct(x, norm='ortho')
    x = x.transpose(-1, -2)

    return x

def idct_2d_torch_dct(x):
    # Inverse of the above
    x = x.transpose(-1, -2)
    x = torch_dct.idct(x, norm='ortho')
    x = x.transpose(-1, -2)
    x = torch_dct.idct(x, norm='ortho')
    return x


def channelwise_energy_masks_fast(x_dct, p1=0.3, p2=0.7, eps=1e-8):
    """
    x_dct: [1, C, H, W]  — batched DCT coefficients
    p1, p2: low/mid and mid/high energy percentiles
    returns low, mid, high masks of shape [1, C, H, W] (float32)
    """
    _, C, H, W = x_dct.shape
    device = x_dct.device

    # 1) radius map (H*W)
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )
    radius = torch.sqrt(xx.float()**2 + yy.float()**2)        # [H, W]
    radius_flat = radius.flatten()                            # [N]
    sort_idx = torch.argsort(radius_flat)                     # [N]
    r_sorted = radius_flat[sort_idx]                          # [N]

    # 2) flatten DCT‐energy per channel
    E = (x_dct[0]**2).reshape(C, -1)       # [C, N]
    E_sorted = E[:, sort_idx]              # [C, N]
    cumE     = torch.cumsum(E_sorted, dim=1)  # [C, N]
    totE     = cumE[:, -1]                    # [C]

    # 3) compute threshold indices vectorized
    #    find first idx where cumE >= p1*totE and p2*totE
    thresh1 = totE.unsqueeze(1) * p1       # [C, 1]
    thresh2 = totE.unsqueeze(1) * p2       # [C, 1]

    mask1 = cumE >= thresh1                # [C, N]
    mask2 = cumE >= thresh2                # [C, N]

    # argmax gives first True (if none, returns 0)
    r1_idx = mask1.float().argmax(dim=1)   # [C]
    r2_idx = mask2.float().argmax(dim=1)

    # 4) detect collapsed or zero‐energy channels
    N = radius_flat.numel()
    fb1, fb2 = int(p1*N), int(p2*N)
    collapsed = (totE < eps) | (r2_idx <= r1_idx)
    r1_idx[collapsed] = fb1
    r2_idx[collapsed] = fb2

    # 5) map back to actual radii
    r1 = r_sorted[r1_idx]  # [C]
    r2 = r_sorted[r2_idx]

    # 6) build per‐channel masks
    rm   = radius.view(1,1,H,W)          # [1,1,H,W]
    r1b  = r1.view(1,C,1,1)              # [1,C,1,1]
    r2b  = r2.view(1,C,1,1)

    low  = (rm <= r1b).float()
    mid  = ((rm > r1b) & (rm <= r2b)).float()
    high = (rm > r2b).float()

    # 7) normalize so low+mid+high == 1.0
    S = low + mid + high
    low  /= (S + eps)
    mid  /= (S + eps)
    high/= (S + eps)

    return low, mid, high


def channelwise_energy_masks_lastdim(x_dct, p1=0.3, p2=0.7, eps=1e-8):
    """
    x_dct: [B, HW, C] — batch of DCT-transformed data across channels
    p1, p2: energy percentiles (low/mid and mid/high)
    Returns: low, mid, high masks of shape [B, HW, C]
    """
    B, N, C = x_dct.shape
    device = x_dct.device

    # 1) Create "channel radius" (0..C-1)
    radius = torch.arange(C, device=device).float()  # [C]
    sort_idx = torch.argsort(radius)                 # [C]
    r_sorted = radius[sort_idx]                      # [C]

    # 2) Compute energy: square the values
    E = (x_dct ** 2)                                 # [B, N, C]
    E_sorted = E[:, :, sort_idx]                     # [B, N, C]
    cumE = torch.cumsum(E_sorted, dim=-1)            # [B, N, C]
    totE = cumE[:, :, -1]                            # [B, N]

    # 3) thresholds
    thresh1 = totE.unsqueeze(-1) * p1                # [B, N, 1]
    thresh2 = totE.unsqueeze(-1) * p2

    mask1 = cumE >= thresh1                          # [B, N, C]
    mask2 = cumE >= thresh2

    r1_idx = mask1.float().argmax(dim=-1)            # [B, N]
    r2_idx = mask2.float().argmax(dim=-1)

    # fallback indices
    fb1 = int(p1 * C)
    fb2 = int(p2 * C)

    collapsed = (totE < eps) | (r2_idx <= r1_idx)
    r1_idx[collapsed] = fb1
    r2_idx[collapsed] = fb2

    # 4) Map back to real "frequency positions"
    r1 = r_sorted[r1_idx]                            # [B, N]
    r2 = r_sorted[r2_idx]                            # [B, N]

    # 5) Build broadcast masks
    freq = radius.view(1, 1, C)                      # [1, 1, C]
    r1b  = r1.unsqueeze(-1)                          # [B, N, 1]
    r2b  = r2.unsqueeze(-1)

    low  = (freq <= r1b).float()
    mid  = ((freq > r1b) & (freq <= r2b)).float()
    high = (freq > r2b).float()

    # 6) Normalize
    S = low + mid + high
    low  /= (S + eps)
    mid  /= (S + eps)
    high /= (S + eps)

    return low, mid, high


