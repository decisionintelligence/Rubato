# Copyright (c) 2023, Tri Dao.

import math
try:
    from scipy.linalg import hadamard
except ImportError:
    hadamard = None

import torch
import torch.nn.functional as F


import fast_hadamard_transform_cuda


class HadamardTransformFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, scale=1.0):
        ctx._hadamard_transform_scale = scale
        return fast_hadamard_transform_cuda.fast_hadamard_transform(x, scale)

    @staticmethod
    def backward(ctx, dout):
        # The Hadamard transform matrix is symmetric, so in the backward pass we multiply by its
        # transpose, which is itself.
        return fast_hadamard_transform_cuda.fast_hadamard_transform(dout, ctx._hadamard_transform_scale), None


def hadamard_transform(x, scale=1.0):
    """
    Arguments:
        x: (..., dim)
        scale: float. Multiply the output by this number.
    Returns:
        out: (..., dim)

    Multiply each row of x by the Hadamard transform matrix.
    Equivalent to F.linear(x, torch.tensor(scipy.linalg.hadamard(dim))) * scale.
    If dim is not a power of 2, we implicitly pad x with zero so that dim is the next power of 2.
    """
    return HadamardTransformFn.apply(x, scale)


class HadamardTransform12NFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, scale=1.0):
        ctx._hadamard_transform_scale = scale
        return fast_hadamard_transform_cuda.fast_hadamard_transform_12N(x, scale)

    @staticmethod
    def backward(ctx, dout):
        # The Hadamard transform matrix is symmetric, so in the backward pass we multiply by its
        # transpose, which is itself.
        return fast_hadamard_transform_cuda.fast_hadamard_transform_12N(dout, ctx._hadamard_transform_scale), None
    

def hadamard_transform_12N(x, scale=1.0):
    """
    Arguments:
        x: (..., dim)
        scale: float. Multiply the output by this number.
    Returns:
        out: (..., dim)

    Multiply each row of x by the Hadamard transform matrix, where dim = 12 * power of 2.
    If dim is not 12 * a power of 2, we implicitly pad x with zero so that dim is 12 * the next power of 2.
    """
    return HadamardTransform12NFn.apply(x, scale)



class HadamardTransform20NFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, scale=1.0):
        ctx._hadamard_transform_scale = scale
        return fast_hadamard_transform_cuda.fast_hadamard_transform_20N(x, scale)

    @staticmethod
    def backward(ctx, dout):
        # The Hadamard transform matrix is symmetric, so in the backward pass we multiply by its
        # transpose, which is itself.
        return fast_hadamard_transform_cuda.fast_hadamard_transform_20N(dout, ctx._hadamard_transform_scale), None


def hadamard_transform_20N(x, scale=1.0):
    """
    Arguments:
        x: (..., dim)
        scale: float. Multiply the output by this number.
    Returns:
        out: (..., dim)

    Multiply each row of x by the Hadamard transform matrix, where dim = 20 * power of 2.
    If dim is not 20 * a power of 2, we implicitly pad x with zero so that dim is 20 * the next power of 2.
    """
    return HadamardTransform20NFn.apply(x, scale)


class HadamardTransform28NFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, scale=1.0):
        ctx._hadamard_transform_scale = scale
        return fast_hadamard_transform_cuda.fast_hadamard_transform_28N(x, scale)

    @staticmethod
    def backward(ctx, dout):
        # The Hadamard transform matrix is symmetric, so in the backward pass we multiply by its
        # transpose, which is itself.
        return fast_hadamard_transform_cuda.fast_hadamard_transform_28N(dout, ctx._hadamard_transform_scale), None


def hadamard_transform_28N(x, scale=1.0):
    """
    Arguments:
        x: (..., dim)
        scale: float. Multiply the output by this number.
    Returns:
        out: (..., dim)

    Multiply each row of x by the Hadamard transform matrix, where dim = 28 * power of 2.
    If dim is not 28 * a power of 2, we implicitly pad x with zero so that dim is 28 * the next power of 2.
    """
    return HadamardTransform28NFn.apply(x, scale)


class HadamardTransform40NFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, scale=1.0):
        ctx._hadamard_transform_scale = scale
        return fast_hadamard_transform_cuda.fast_hadamard_transform_40N(x, scale)

    @staticmethod
    def backward(ctx, dout):
        # The Hadamard transform matrix is symmetric, so in the backward pass we multiply by its
        # transpose, which is itself.
        return fast_hadamard_transform_cuda.fast_hadamard_transform_40N(dout, ctx._hadamard_transform_scale), None

def hadamard_transform_40N(x, scale=1.0):
    """
    Arguments:
        x: (..., dim)
        scale: float. Multiply the output by this number.
    Returns:
        out: (..., dim)

    Multiply each row of x by the Hadamard transform matrix, where dim = 40 * power of 2.
    If dim is not 40 * a power of 2, we implicitly pad x with zero so that dim is 40 * the next power of 2.
    """
    return HadamardTransform40NFn.apply(x, scale)

class SORFTransformFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, diag, scale=1.0):
        ctx.save_for_backward(diag)
        ctx._sorf_scale = scale
        return fast_hadamard_transform_cuda.fast_sorf_transform(x, diag, scale)

    @staticmethod
    def backward(ctx, dout):
        diag, = ctx.saved_tensors
        scale = ctx._sorf_scale
        # Forward: z = scale * H*D_{n-1}*H*D_{n-2}*...*H*D_0 * x
        # Backward: grad_x = scale * D_0*H*D_1*H*...*D_{n-1}*H * dout
        # The backward kernel does (FWHT then diag) per round,
        # with diag order reversed via flip.
        diag_rev = diag.flip(0)
        dx = fast_hadamard_transform_cuda.fast_sorf_backward_transform(dout, diag_rev, scale)
        return dx, None, None


def sorf_transform(x, diag, scale=1.0):
    """
    Fused SORF (Structured Orthogonal Random Features) transform.

    Computes scale * H * D_{n} * H * D_{n-1} * ... * H * D_1 * x in a single
    CUDA kernel launch, where H is the (unnormalized) Hadamard matrix and
    D_i are diagonal matrices.

    Arguments:
        x: (..., dim) input tensor
        diag: (n_rounds, dim_padded) diagonal matrices (typically ±1)
        scale: float. Multiply the output by this number.
            For normalized SORF, set scale = (1/sqrt(dim))^n_rounds.
    Returns:
        out: (..., dim)
    """
    return SORFTransformFn.apply(x, diag, scale)


def sorf_backward_transform(x, diag, scale=1.0):
    """
    SORF backward transform (FWHT then diagonal, per round).

    This is the transpose of sorf_transform. Caller must pass diag
    with rounds already reversed (diag.flip(0)).

    Arguments:
        x: (..., dim) input tensor (typically grad_output)
        diag: (n_rounds, dim_padded) diagonal matrices with reversed round order
        scale: float
    Returns:
        out: (..., dim)
    """
    return fast_hadamard_transform_cuda.fast_sorf_backward_transform(x, diag, scale)


class SORFRFFTransformFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, diag, input_scale, sorf_scale, rff_scale, rff_dim):
        # Save x and diag for recomputing z in backward
        ctx.save_for_backward(x, diag)
        ctx._input_scale = input_scale
        ctx._sorf_scale = sorf_scale
        ctx._rff_scale = rff_scale
        ctx._rff_dim = rff_dim

        # Compute [cos(z), sin(z)] * rff_scale using the fused kernel
        return fast_hadamard_transform_cuda.fast_sorf_rff_transform(
            x, diag, input_scale, sorf_scale, rff_scale, rff_dim)

    @staticmethod
    def backward(ctx, dout):
        x, diag = ctx.saved_tensors
        rff_dim = ctx._rff_dim
        sorf_scale = ctx._sorf_scale
        input_scale = ctx._input_scale
        rff_scale = ctx._rff_scale

        # Recompute z = sorf_scale * SORF(input_scale * x)
        # Pre-pad x to power-of-2 so sorf_transform returns full padded output
        # (otherwise it slices to dim_og which may be < rff_dim)
        dim_og = x.shape[-1]
        d_pad = max(8, 1 << (dim_og - 1).bit_length())
        x_scaled = x * input_scale if input_scale != 1.0 else x
        if dim_og != d_pad:
            x_scaled = F.pad(x_scaled, (0, d_pad - dim_og))
        z_full = fast_hadamard_transform_cuda.fast_sorf_transform(x_scaled, diag, sorf_scale)
        # z_full: (..., d_pad) — no slicing since input was already power-of-2

        # dout: (..., 2*rff_dim) = [d_cos_part, d_sin_part]
        d_cos = dout[..., :rff_dim]
        d_sin = dout[..., rff_dim:]
        z_slice = z_full[..., :rff_dim]

        # Gradient through cos/sin:
        # out = [cos(z), sin(z)] * rff_scale
        # d_loss/d_z = (-sin(z) * d_cos + cos(z) * d_sin) * rff_scale
        grad_z = (-torch.sin(z_slice) * d_cos + torch.cos(z_slice) * d_sin) * rff_scale

        # Pad grad_z back to d_pad for backward kernel
        if rff_dim < d_pad:
            grad_z = F.pad(grad_z, (0, d_pad - rff_dim))

        # Gradient through SORF (backward kernel)
        diag_rev = diag.flip(0)
        grad_x_padded = fast_hadamard_transform_cuda.fast_sorf_backward_transform(
            grad_z, diag_rev, sorf_scale)

        # Slice back to original dim and apply input_scale chain rule
        grad_x = grad_x_padded[..., :dim_og]
        if input_scale != 1.0:
            grad_x = grad_x * input_scale

        return grad_x, None, None, None, None, None


def sorf_rff_transform(x, diag, input_scale=1.0, sorf_scale=1.0, rff_scale=1.0, rff_dim=None):
    """
    Fully fused SORF random feature computation in a single CUDA kernel.

    Computes: [cos(z), sin(z)] * rff_scale
    where z = sorf_scale * H*Dn*...*H*D1 * (input_scale * x)

    Arguments:
        x: (..., dim) input tensor
        diag: (n_rounds, dim_padded) diagonal matrices (typically ±1)
        input_scale: float, multiply input (e.g. 1/sigma)
        sorf_scale: float, applied after SORF (e.g. (1/sqrt(d))^n_rounds)
        rff_scale: float, multiply cos/sin output (e.g. 1/sqrt(D))
        rff_dim: int, number of features D (default: dim)
    Returns:
        out: (..., 2*rff_dim)
    """
    if rff_dim is None:
        rff_dim = x.shape[-1]
    return SORFRFFTransformFn.apply(x, diag, input_scale, sorf_scale, rff_scale, rff_dim)


class SORFRFFTransformBatchedFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, diag, input_scale, sorf_scale, rff_scale, rff_dim):
        # diag: (G, n_rounds, dim_padded)
        ctx.save_for_backward(x, diag)
        ctx._input_scale = input_scale
        ctx._sorf_scale = sorf_scale
        ctx._rff_scale = rff_scale
        ctx._rff_dim = rff_dim
        return fast_hadamard_transform_cuda.fast_sorf_rff_transform_batched(
            x, diag, input_scale, sorf_scale, rff_scale, rff_dim)

    @staticmethod
    def backward(ctx, dout):
        # dout: (..., G * 2 * rff_dim), layout per batch [cos_g0|sin_g0|...|cos_g{G-1}|sin_g{G-1}]
        x, diag = ctx.saved_tensors
        rff_dim = ctx._rff_dim
        sorf_scale = ctx._sorf_scale
        input_scale = ctx._input_scale
        rff_scale = ctx._rff_scale
        num_groups = diag.shape[0]
        n_rounds = diag.shape[1]

        dim_og = x.shape[-1]
        d_pad = max(8, 1 << (dim_og - 1).bit_length())

        # Recompute z for all groups: (..., G, d_pad)
        x_scaled = x * input_scale if input_scale != 1.0 else x
        if dim_og != d_pad:
            x_scaled = F.pad(x_scaled, (0, d_pad - dim_og))
        z_full = fast_hadamard_transform_cuda.fast_sorf_transform_batched(
            x_scaled, diag, sorf_scale)
        # z_full: (..., G * d_pad)
        leading = list(x.shape[:-1])
        z_per_group = z_full.view(*leading, num_groups, d_pad)
        z_slice = z_per_group[..., :rff_dim]  # (..., G, rff_dim)

        # Chain rule through cos/sin per group
        dout_per_group = dout.view(*leading, num_groups, 2 * rff_dim)
        d_cos = dout_per_group[..., :rff_dim]
        d_sin = dout_per_group[..., rff_dim:]
        grad_z = (-torch.sin(z_slice) * d_cos + torch.cos(z_slice) * d_sin) * rff_scale  # (..., G, rff_dim)
        if rff_dim < d_pad:
            grad_z = F.pad(grad_z, (0, d_pad - rff_dim))  # (..., G, d_pad)

        # Backward SORF per group: reshape (..., G, d_pad) → (..., * G, d_pad), call batched backward
        # via the existing single-group kernel by flattening G into the batch dim.
        grad_z_flat = grad_z.reshape(-1, d_pad).contiguous()  # (B*L*G, d_pad)
        # diag for backward needs rounds reversed per group: (G, n_rounds, d_pad)
        diag_rev = diag.flip(1).contiguous()
        # Run backward per-group via single-group kernel by re-using the batch dim.
        # We need diag_rev[g] applied to the g-th slice of grad_z_flat groups.
        # Layout: grad_z_flat[i * G + g] uses diag_rev[g]. So we replicate diag_rev
        # across the batch dim and call the single-group backward in a loop... that
        # defeats the point. Instead use a per-group call but fewer launches: G calls
        # on (B*L, d_pad) chunks via stride trick.
        #
        # Simpler & still launch-efficient: G separate single-group backward calls
        # (each at full B*L batch). G is small (≤16 for our config) and backward is
        # called once per training step, not per layer; the eager loop here is OK.
        n_total = grad_z.shape[0] if grad_z.dim() == 2 else None
        B_eff = grad_z[..., 0].numel() // num_groups  # elements per group

        # grad_z reshape: (..., G, d_pad) → for each g, slice grad_z[..., g, :]
        # Get a contiguous (B_eff, d_pad) view per group.
        grad_z_per_g = grad_z.reshape(-1, num_groups, d_pad)  # (B_eff, G, d_pad)
        grad_x_padded = torch.zeros(grad_z_per_g.shape[0], d_pad,
                                     device=x.device, dtype=x.dtype)
        for g in range(num_groups):
            gz_g = grad_z_per_g[:, g, :].contiguous()
            gx_g = fast_hadamard_transform_cuda.fast_sorf_backward_transform(
                gz_g, diag_rev[g].contiguous(), sorf_scale)
            grad_x_padded += gx_g

        grad_x = grad_x_padded[..., :dim_og]
        if input_scale != 1.0:
            grad_x = grad_x * input_scale
        grad_x = grad_x.reshape(*leading, dim_og)

        return grad_x, None, None, None, None, None


def sorf_rff_transform_batched(x, diag, input_scale=1.0, sorf_scale=1.0, rff_scale=1.0, rff_dim=None):
    """
    Batched-diag variant: applies G independent SORF blocks in one CUDA launch.

    Arguments:
        x: (..., dim) input tensor (shared across all G groups)
        diag: (G, n_rounds, dim_padded) — independent Rademacher diagonals per group
        input_scale, sorf_scale, rff_scale, rff_dim: as in sorf_rff_transform
    Returns:
        out: (..., G * 2 * rff_dim) laid out per batch element as
             [cos_g0|sin_g0 | cos_g1|sin_g1 | ... | cos_g{G-1}|sin_g{G-1}]
             (each cos_g* / sin_g* is rff_dim long).
    """
    if rff_dim is None:
        rff_dim = x.shape[-1]
    return SORFRFFTransformBatchedFn.apply(x, diag, input_scale, sorf_scale, rff_scale, rff_dim)


def hadamard_transform_ref(x, scale=1.0):
    """
    x: (..., dim)
    out: (..., dim)
    """
    if hadamard is None:
        raise ImportError("Please install scipy")
    x_shape = x.shape
    dim = x.shape[-1]
    x = x.reshape(-1, dim)
    log_dim = math.ceil(math.log2(dim))
    dim_padded = 2 ** log_dim
    if dim != dim_padded:
        x = F.pad(x, (0, dim_padded - dim))
    out = F.linear(x, torch.tensor(hadamard(dim_padded, dtype=float), dtype=x.dtype, device=x.device))
    out = out * scale
    return out[..., :dim].reshape(*x_shape)
