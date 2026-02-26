import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
import pdb
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
#        h, w = x.size()[-2:]
#        paddingBottom = int(np.ceil(h/8)*8-h)
#        paddingRight = int(np.ceil(w/8)*8-w)
#        x = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x)

        x1 = self.m_head(x0)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x+x4)
        x = self.m_up2(x+x3)
        x = self.m_up1(x+x2)
        x = self.m_tail(x+x1)
#        x = x[..., :h, :w]

        return x

class UNetRes_Gemini(nn.Module):
    def __init__(self, in_nc=2, out_nc=1, nc=[64, 128, 256, 512], nb=4, 
                 act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose'):
        super(UNetRes, self).__init__()
        
        self.m_head = nn.Conv2d(in_nc, nc[0], kernel_size=3, padding=1, bias=True)

        # helper for ResBlock
        def ResBlock(channels):
            return nn.Sequential(
                nn.Conv2d(channels, channels, 3, 1, 1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, 3, 1, 1, bias=True)
            )

        # Downsample
        self.m_down1 = nn.ModuleList([ResBlock(nc[0]) for _ in range(nb)])
        self.m_down2 = nn.ModuleList([ResBlock(nc[1]) for _ in range(nb)])
        self.m_down3 = nn.ModuleList([ResBlock(nc[2]) for _ in range(nb)])
        self.m_down4 = nn.ModuleList([ResBlock(nc[3]) for _ in range(nb)])

        self.d_l1 = nn.Conv2d(nc[0], nc[1], 2, 2)
        self.d_l2 = nn.Conv2d(nc[1], nc[2], 2, 2)
        self.d_l3 = nn.Conv2d(nc[2], nc[3], 2, 2)

        # Upsample
        self.u_l3 = nn.ConvTranspose2d(nc[3], nc[2], 2, 2)
        self.u_l2 = nn.ConvTranspose2d(nc[2], nc[1], 2, 2)
        self.u_l1 = nn.ConvTranspose2d(nc[1], nc[0], 2, 2)

        self.m_up3 = nn.ModuleList([ResBlock(nc[2]) for _ in range(nb)])
        self.m_up2 = nn.ModuleList([ResBlock(nc[1]) for _ in range(nb)])
        self.m_up1 = nn.ModuleList([ResBlock(nc[0]) for _ in range(nb)])

        self.m_tail = nn.Conv2d(nc[0], out_nc, 3, 1, 1, bias=True)

    def forward(self, x):
        # Padding to ensure dimensions are multiples of 8
        h, w = x.size()[-2:]
        paddingBottom = int(np.ceil(h/8)*8) - h
        paddingRight = int(np.ceil(w/8)*8) - w
        x = nn.functional.pad(x, (0, paddingRight, 0, paddingBottom), mode='reflect')

        x1 = self.m_head(x)
        x2 = self.d_l1(self._seq_forward(self.m_down1, x1))
        x3 = self.d_l2(self._seq_forward(self.m_down2, x2))
        x4 = self.d_l3(self._seq_forward(self.m_down3, x3))
        x = self._seq_forward(self.m_down4, x4)

        x = self._seq_forward(self.m_up3, x + self.u_l3(x))
        x = self._seq_forward(self.m_up2, x + self.u_l2(x3))
        x = self._seq_forward(self.m_up1, x + self.u_l1(x2))

        x = self.m_tail(x + x1)
        return x[..., :h, :w]

    def _seq_forward(self, module_list, x):
        for m in module_list:
            x = m(x) + x
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

def pnp_pgm_robust(
    y,                      # Observations
    forward_op,             # Approximate Forward Operator
    denoiser_fn,            # The Prior (GrayscaleDPIR.forward)
    x_init=None,
    n_iters=50,             # PGM needs more iterations than HQS (50-100)
    step_size=0.9,          # IMPORTANT: Must be < 2 / Lipschitz constant of A
    sigma_0=49/255,
    sigma_end=15/255,
    lamb=1.0,               # Regularization strength
    device='cuda'
):
    """
    Robust PnP Solver for Approximate Operators.
    Uses Proximal Gradient Method (Forward-Backward Splitting).
    
    Rule of Thumb: Treat Model Error as extra Noise.
    """
    if x_init is None:
        x = torch.zeros_like(y) # Or use y if dimensions match
    else:
        x = x_init.clone()

    # Decay strategy: We might not want to decay sigma fully if the model is bad.
    # Keeping sigma slightly higher masks model mismatch artifacts.
    sigmas = torch.logspace(np.log10(sigma_0), np.log10(sigma_end), n_iters).to(device)

    for i in range(n_iters):
        sigma = sigmas[i]
        
        # --- Step 1: Gradient Descent on Data Term (Forward Step) ---
        # x_new = x - step_size * grad( ||Ax - y||^2 )
        
        # We reuse the auto-differentiation helper from before
        grad_data, _ = compute_grad_data_term(forward_op, x, y)
        #pdb.set_trace()
        
        # Gradient Update
        z = x - step_size * grad_data

        # --- Step 2: Denoising (Backward / Proximal Step) ---
        # x = Denoiser(z, sigma * sqrt(lambda))
        # Note: In PGM, the effective noise level passed to the denoiser 
        # scales with the step size. 
        
        # Refined noise level for the denoiser:
        denoise_sigma = sigma * np.sqrt(lamb / step_size)
        
        with torch.no_grad():
            x = denoiser_fn(z, denoise_sigma)

        # Optional: Monitoring
        if i % 10 == 0:
            norm_grad = torch.mean(grad_data**2).item()
            print(f"Iter {i} | Sigma: {denoise_sigma:.4f} | Grad Mag: {norm_grad:.6f}")

            im_rec = x.squeeze().cpu().numpy().astype(np.float32)

            cv2.imshow('Reconstruction', im_rec)
            cv2.waitKey(1)

    return x

# ==============================================================================
# SECTION 5: DEMO / TEST
# ==============================================================================

if __name__ == "__main__":
    # Settings
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    WEIGHT_PATH = 'drunet_gray.pth' # <--- MAKE SURE THIS FILE EXISTS
    
    print(f"Running on {device}")
    
    if not os.path.exists(WEIGHT_PATH):
        print(f"\n[!] Missing Weights: {WEIGHT_PATH}")
        print("Please download it from: https://github.com/cszn/KAIR/releases/download/v1.0/drunet_gray.pth")
        sys.exit()

    # 1. Load Prior
    prior = GrayscaleDPIR(WEIGHT_PATH, device=device)

    # 2. Create Synthetic Data
    # 256x256 Image (Circle)
    gt = torch.zeros(1, 1, 256, 256).to(device)
    Y, X = torch.meshgrid(torch.arange(256), torch.arange(256), indexing='ij')
    mask_circle = ((X - 128)**2 + (Y - 128)**2) < 60**2
    gt[:, :, mask_circle] = 0.8
    gt = gt.to(device)

    # 3. Define Forward Operator: Random Inpainting
    # We remove 70% of pixels randomly
    mask = torch.rand_like(gt) > 0.7
    mask = mask.float().to(device)

    def forward_inpainting(x):
        return x * mask

    # Create Observation
    y_obs = forward_inpainting(gt)
    
    # 4. Solve
    print("Starting PnP Reconstruction...")
    x_rec = pnp_hqs_blackbox(
        y=y_obs,
        forward_op=forward_inpainting,
        denoiser_fn=prior.forward,
        x_init=y_obs.clone(), # Start with the masked image
        n_iters=12,
        cg_iters=10,
        lamb=1.0,
        device=device
    )
    
    print("Done.")
    
    # 5. Save/Show results (Simple ASCII art check)
    mse = torch.mean((x_rec - gt)**2)
    print(f"Reconstruction MSE: {mse:.6f}")
    
    # Save if you want
    try:
        from torchvision.utils import save_image
        save_image(torch.cat([gt, y_obs, x_rec], dim=3), "pnp_result.png")
        print("Saved comparison to 'pnp_result.png'")
    except ImportError:
        print("torchvision not installed, skipping image save.")

def estimate_step_size(forward_op, shape, device='cuda', num_iters=10):
    """
    Estimates the Lipschitz constant L of (A^T A) using Power Iteration.
    Returns safe step_size = 1.0 / L
    """
    print("Estimating optimal step size...")
    # Random noise vector
    b_k = torch.randn(shape, device=device)
    b_k = b_k / torch.norm(b_k)

    for _ in range(num_iters):
        # 1. Apply Operator A (Forward)
        # We need A * b_k. 
        # Since forward_op might expect specific ranges, we just pass b_k.
        b_in = b_k.requires_grad_(True)
        Ab = forward_op(b_in)
        
        # 2. Apply Adjoint A^T (Backward)
        # Vector-Jacobian Product of Ab with itself (dummy way to get A^T(Ab))
        # grad( 0.5 * ||Ax||^2 ) = A^T A x
        loss = 0.5 * torch.sum(Ab**2)
        At_Ab = torch.autograd.grad(loss, b_in, create_graph=False)[0]
        
        # 3. Normalize
        b_k = At_Ab.detach()
        l_norm = torch.norm(b_k)
        b_k = b_k / (l_norm + 1e-8)
        
    # L is approximately the largest eigenvalue magnitude
    L = l_norm.item()
    print(f"Estimated Lipschitz Constant L: {L:.4f}")
    
    # Theoretical Limit is 2/L. We use 1.0/L to be safe.
    safe_step = 0.9 * (1.0 / (L + 1e-6))
    print(f"Recommended Step Size: {safe_step:.6f}")
    return safe_step