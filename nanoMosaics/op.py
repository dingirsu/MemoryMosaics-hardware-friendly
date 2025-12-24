import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _leaky_avg_fwd_kernel(
    k_ptr,          # Input: [B, H, T, D]
    alpha_ptr,      # Alpha: [1, H, 1, 1]
    out_ptr,        # Output: [B, H, T, D]
    stride_kb, stride_kh, stride_kt, stride_kd,
    stride_ab, stride_ah, stride_at, stride_ad,
    stride_ob, stride_oh, stride_ot, stride_od,
    B, H, T, D,
    BLOCK_D: tl.constexpr
):
    # 1. Map PID to (b, h, d_block)
    pid = tl.program_id(0)
    
    # Grid is (B * H * ceil(D/BLOCK_D))
    num_d_blocks = tl.cdiv(D, BLOCK_D)
    
    idx_d_block = pid % num_d_blocks
    idx_h = (pid // num_d_blocks) % H
    idx_b = (pid // num_d_blocks) // H
    
    offs_d = idx_d_block * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    alpha_val = tl.load(alpha_ptr + idx_h * stride_ah)
    k_base = k_ptr + idx_b * stride_kb + idx_h * stride_kh + offs_d * stride_kd
    out_base = out_ptr + idx_b * stride_ob + idx_h * stride_oh + offs_d * stride_od
    curr_k = tl.load(k_base, mask=mask_d, other=0.0)
    running_out = curr_k
    tl.store(out_base, running_out, mask=mask_d)
    for t in range(1, T):
        k_curr_ptr = k_base + t * stride_kt
        out_curr_ptr = out_base + t * stride_ot
        
        curr_k = tl.load(k_curr_ptr, mask=mask_d, other=0.0)
        running_out = curr_k + alpha_val * running_out
        
        tl.store(out_curr_ptr, running_out, mask=mask_d)

@triton.jit
def _leaky_avg_bwd_kernel(
    grad_out_ptr,   # [B, H, T, D]
    out_ptr,        # [B, H, T, D]
    alpha_ptr,      # [1, H, 1, 1]
    grad_k_ptr,     # [B, H, T, D]
    grad_alpha_ptr, # [B, H, D] (Partial sums)
    stride_gob, stride_goh, stride_got, stride_god,
    stride_ob, stride_oh, stride_ot, stride_od,
    stride_ah,
    stride_gkb, stride_gkh, stride_gkt, stride_gkd,
    stride_gab, stride_gah, stride_gad,
    B, H, T, D,
    BLOCK_D: tl.constexpr
):
    pid = tl.program_id(0)
    num_d_blocks = tl.cdiv(D, BLOCK_D)
    
    idx_d_block = pid % num_d_blocks
    idx_h = (pid // num_d_blocks) % H
    idx_b = (pid // num_d_blocks) // H
    
    offs_d = idx_d_block * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    alpha_val = tl.load(alpha_ptr + idx_h * stride_ah)

    go_base = grad_out_ptr + idx_b * stride_gob + idx_h * stride_goh + offs_d * stride_god
    out_base = out_ptr + idx_b * stride_ob + idx_h * stride_oh + offs_d * stride_od
    gk_base = grad_k_ptr + idx_b * stride_gkb + idx_h * stride_gkh + offs_d * stride_gkd

    acc_grad_alpha = tl.zeros([BLOCK_D], dtype=tl.float32)
    running_delta = tl.zeros([BLOCK_D], dtype=tl.float32)
    for t in range(T - 1, 0, -1):
        go_val = tl.load(go_base + t * stride_got, mask=mask_d, other=0.0)
        
        curr_delta = go_val + alpha_val * running_delta
        
        tl.store(gk_base + t * stride_gkt, curr_delta, mask=mask_d)
        
        prev_out = tl.load(out_base + (t - 1) * stride_ot, mask=mask_d, other=0.0)
        acc_grad_alpha += curr_delta * prev_out
        
        running_delta = curr_delta

    go_val_0 = tl.load(go_base, mask=mask_d, other=0.0)
    curr_delta_0 = go_val_0 + alpha_val * running_delta
    tl.store(gk_base, curr_delta_0, mask=mask_d)

    ga_ptr = grad_alpha_ptr + idx_b * stride_gab + idx_h * stride_gah + offs_d * stride_gad
    tl.store(ga_ptr, acc_grad_alpha, mask=mask_d)

class LeakyAvgTritonFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, k, alpha):
        # k: [B, H, T, D]
        # alpha: [1, H, 1, 1]
        k = k.contiguous()
        alpha = alpha.contiguous()
        
        B, H, T, D = k.shape
        out = torch.empty_like(k)
        
        # Heuristics for block size
        BLOCK_D = 64
        if D >= 128: BLOCK_D = 128
        
        grid = (B * H * triton.cdiv(D, BLOCK_D), )
        
        _leaky_avg_fwd_kernel[grid](
            k, alpha, out,
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            alpha.stride(0), alpha.stride(1), alpha.stride(2), alpha.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            B, H, T, D,
            BLOCK_D=BLOCK_D
        )
        
        ctx.save_for_backward(k, alpha, out)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        k, alpha, out = ctx.saved_tensors
        B, H, T, D = k.shape
        
        grad_out = grad_out.contiguous()
        grad_k = torch.empty_like(k)
        
        grad_alpha_part = torch.empty((B, H, D), device=k.device, dtype=torch.float32)
        
        BLOCK_D = 64
        if D >= 128: BLOCK_D = 128
        
        grid = (B * H * triton.cdiv(D, BLOCK_D), )
        
        _leaky_avg_bwd_kernel[grid](
            grad_out, out, alpha, grad_k, grad_alpha_part,
            grad_out.stride(0), grad_out.stride(1), grad_out.stride(2), grad_out.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            alpha.stride(1), # stride_ah
            grad_k.stride(0), grad_k.stride(1), grad_k.stride(2), grad_k.stride(3),
            grad_alpha_part.stride(0), grad_alpha_part.stride(1), grad_alpha_part.stride(2),
            B, H, T, D,
            BLOCK_D=BLOCK_D
        )
        
        grad_alpha = grad_alpha_part.sum(dim=(0, 2)).view(1, H, 1, 1)
        
        return grad_k, grad_alpha

class LeakyAvg(nn.Module):
    def __init__(self, n_head: int):
        super().__init__()
        self.exp_scaling = 10
        self.leaky_key_beta = nn.Parameter(
            torch.linspace(0.5, 5, n_head).view(1, n_head, 1, 1)
            / self.exp_scaling
        )

    def forward(self, k: torch.Tensor):
        beta = self.leaky_key_beta.abs() * self.exp_scaling
        alpha = torch.exp(-beta)

        return LeakyAvgTritonFunction.apply(k, alpha)
