import torch
import torch.nn as nn
import numpy as np
import os
import sys
import cv2

import basicblock as B

# ==============================================================================
# SECTION 1: NETWORK ARCHITECTURE (UNetRes)
# ==============================================================================
# This architecture matches the 'drunet_gray.pth' weights from KAIR.
# Credit: Kai Zhang (https://github.com/cszn/KAIR)

class UNetRes(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nc=[64, 128, 256, 512], nb=4,
                 act_mode='R', downsample_mode='strideconv',
                 upsample_mode='convtranspose', bias=True):
        super(UNetRes, self).__init__()

        self.m_head = B.conv(in_nc, nc[0], bias=bias, mode='C')

        # downsample
        if downsample_mode == 'avgpool':
            downsample_block = B.downsample_avgpool
        elif downsample_mode == 'maxpool':
            downsample_block = B.downsample_maxpool
        elif downsample_mode == 'strideconv':
            downsample_block = B.downsample_strideconv
        else:
            raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))

        self.m_down1 = B.sequential(*[B.ResBlock(nc[0], nc[0], bias=bias, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[0], nc[1], bias=bias, mode='2'))
        self.m_down2 = B.sequential(*[B.ResBlock(nc[1], nc[1], bias=bias, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[1], nc[2], bias=bias, mode='2'))
        self.m_down3 = B.sequential(*[B.ResBlock(nc[2], nc[2], bias=bias, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[2], nc[3], bias=bias, mode='2'))

        self.m_body  = B.sequential(*[B.ResBlock(nc[3], nc[3], bias=bias, mode='C'+act_mode+'C') for _ in range(nb)])

        # upsample
        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        self.m_up3 = B.sequential(upsample_block(nc[3], nc[2], bias=bias, mode='2'), *[B.ResBlock(nc[2], nc[2], bias=bias, mode='C'+act_mode+'C') for _ in range(nb)])
        self.m_up2 = B.sequential(upsample_block(nc[2], nc[1], bias=bias, mode='2'), *[B.ResBlock(nc[1], nc[1], bias=bias, mode='C'+act_mode+'C') for _ in range(nb)])
        self.m_up1 = B.sequential(upsample_block(nc[1], nc[0], bias=bias, mode='2'), *[B.ResBlock(nc[0], nc[0], bias=bias, mode='C'+act_mode+'C') for _ in range(nb)])

        self.m_tail = B.conv(nc[0], out_nc, bias=bias, mode='C')

    def forward(self, x0):
        # the image height and width must be multiples of 8 (three 2x downsamplings)
        x1 = self.m_head(x0)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x+x4)
        x = self.m_up2(x+x3)
        x = self.m_up1(x+x2)
        x = self.m_tail(x+x1)

        return x

# ==============================================================================
# SECTION 2: PRIOR WRAPPER (Grayscale DPIR)
# ==============================================================================

class GrayscaleDPIR(nn.Module):
    def __init__(self, model_path, device='cuda'):
        super().__init__()
        self.device = device
        
        # Instantiate model with in_nc=2 (1 Img + 1 Noise Map)
        self.model = UNetRes(in_nc=2, out_nc=1, nc=[64, 128, 256, 512], nb=4, bias=False)
        
        # Load Weights safely
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}. Please download 'drunet_gray.pth'.")
                    
        try:
            state_dict = torch.load(model_path, map_location=device)
            # Remove 'module.' prefix if present (from DataParallel)
            new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            self.model.load_state_dict(new_state_dict, strict=True)
        except Exception as e:
            print(f"Error loading weights: {e}")
            sys.exit(1)
            
        self.model.eval().to(device)

    def forward(self, x, sigma):
        # Create noise map and concatenate
        
        if sigma.numel() == 1:
            sigma_map = torch.full((x.shape[0], 1, x.shape[2], x.shape[3]),
                                    sigma, device=x.device)
        else:
            sigma_map = sigma
            
        x_in = torch.cat([x, sigma_map], dim=1)
        return self.model(x_in)

# ==============================================================================
# SECTION 3: SOLVER UTILITIES (Auto-Adjoint & CG)
# ==============================================================================

def compute_grad_data_term(forward_op, x, y):
    """Computes gradient of 0.5 * ||Ax - y||^2 using Autograd."""
    x_in = x.detach().requires_grad_(True)
    Ax = forward_op(x_in)
    diff = Ax - y
    loss = 0.5 * torch.sum(diff ** 2)
    grad = torch.autograd.grad(loss, x_in, create_graph=False)[0]
    return grad, Ax.detach()

def conjugate_gradient(A_func, b, x_init, n_iter=10):
    """Solves Ax = b using Conjugate Gradient."""
    x = x_init.clone()
    r = b - A_func(x)
    p = r.clone()
    rsold = torch.sum(r * r)

    for i in range(n_iter):
        Ap = A_func(p)
        alpha = rsold / (torch.sum(p * Ap) + 1e-10)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = torch.sum(r * r)
        if torch.sqrt(rsnew) < 1e-6:
            break
        p = r + (rsnew / (rsold + 1e-10)) * p
        rsold = rsnew
    return x

# ==============================================================================
# SECTION 4: MAIN SOLVER (PnP-HQS Blackbox)
# ==============================================================================

def pnp_hqs_blackbox(y, forward_op, denoiser_fn, x_init=None, 
                     n_iters=8, sigma_0=49/255, sigma_end=15/255, 
                     cg_iters=5, lamb=1.0, device='cuda', sigma_map=None):
    """
    Solves y = A(x) + n using Plug-and-Play HQS.
    No explicit Matrix A or A^T required.
    """
    if x_init is None:
        # Simple initialization: Assume y is same shape as x (or close enough)
        # For valid results in complex inverse problems, pass a better x_init.
        x = torch.zeros_like(y)
    else:
        x = x_init.clone()

    sigmas = torch.logspace(np.log10(sigma_0), np.log10(sigma_end), n_iters).to(device)

    if sigma_map is None:
        sigma_map = 1.0

    for i in range(n_iters):
        sigma = sigmas[i]
        rho = lamb / (sigma ** 2)

        # --- Step 1: Data Consistency (CG Solver) ---
        # Solve: (A^T A + rho I) x = A^T y + rho z
        z = x.clone()
        
        # Calculate RHS: A^T y + rho * z
        # A^T y is -grad(0.5||A(0)-y||^2)
        zeros = torch.zeros_like(x)
        grad_at_zero, _ = compute_grad_data_term(forward_op, zeros, y)
        At_y = -grad_at_zero
        
        rhs = At_y + rho * z

        # Define LHS Operator: u -> A^T A u + rho u
        def LHS_op(u):
            # Gradient of 0.5||Av||^2 at v=u is A^T(Au)
            # We compute grad of ||Au - 0||^2
            grad_AtAu, _ = compute_grad_data_term(forward_op, u, torch.zeros_like(y))
            return grad_AtAu + rho * u

        x = conjugate_gradient(LHS_op, rhs, x_init=x, n_iter=cg_iters)

        # --- Step 2: Denoising Prior ---
        with torch.no_grad():
            x = denoiser_fn(x, sigma*sigma_map)
            
        # Optional: Print progress
        print(f"Iter {i+1}/{n_iters} | Sigma: {sigma:.4f} | Range: [{x.min():.2f}, {x.max():.2f}]")

        im_rec = x.squeeze().cpu().numpy().astype(np.float32)

        maxval = im_rec[10:-10, 10:-10].max()
        cv2.imshow('Reconstruction', im_rec/maxval)
        cv2.waitKey(1)
    return x