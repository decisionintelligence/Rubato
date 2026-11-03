/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#include <torch/extension.h>
#if TORCH_VERSION_MAJOR > 2 || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 6)
#include <c10/core/DeviceGuard.h>
#else
#include <c10/cuda/CUDAGuard.h>
#endif
#include <c10/cuda/CUDAStream.h>
#include <vector>

#include "fast_hadamard_transform.h"

#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")

#define DISPATCH_ITYPE_FLOAT_AND_HALF_AND_BF16(ITYPE, NAME, ...)                    \
    if (ITYPE == at::ScalarType::Half) {                                            \
        using input_t = at::Half;                                                   \
        __VA_ARGS__();                                                              \
    } else if (ITYPE == at::ScalarType::BFloat16) {                                 \
        using input_t = at::BFloat16;                                               \
        __VA_ARGS__();                                                              \
    } else if (ITYPE == at::ScalarType::Float) {                                    \
        using input_t = float;                                                      \
        __VA_ARGS__();                                                              \
    } else {                                                                        \
        AT_ERROR(#NAME, " not implemented for input type '", toString(ITYPE), "'"); \
    }

template<typename input_t>
void fast_hadamard_transform_cuda(HadamardParamsBase &params, cudaStream_t stream);

template<typename input_t>
void fast_sorf_transform_cuda(SORFParamsBase &params, cudaStream_t stream);

template<typename input_t>
void fast_sorf_rff_transform_cuda(SORFParamsBase &params, cudaStream_t stream);

template<typename input_t>
void fast_sorf_backward_transform_cuda(SORFParamsBase &params, cudaStream_t stream);

template<typename input_t>
void fast_hadamard_transform_12N_cuda(HadamardParamsBase &params, cudaStream_t stream);

template<typename input_t>
void fast_hadamard_transform_20N_cuda(HadamardParamsBase &params, cudaStream_t stream);

template<typename input_t>
void fast_hadamard_transform_28N_cuda(HadamardParamsBase &params, cudaStream_t stream);

template<typename input_t>
void fast_hadamard_transform_40N_cuda(HadamardParamsBase &params, cudaStream_t stream);

void set_hadamard_params(HadamardParamsBase &params,
                         // sizes
                         const size_t batch,
                         const size_t dim,
                         const size_t multiple,
                         // device pointers
                         const at::Tensor x,
                         const at::Tensor out,
                         float scale
                         ) {

    // Reset the parameters
    memset(&params, 0, sizeof(params));

    params.batch = batch;
    params.dim = dim;
    params.log_N = int(ceil(std::log2(dim / multiple)));

    // Set the pointers and strides.
    params.x_ptr = x.data_ptr();
    params.out_ptr = out.data_ptr();
    // All stride are in elements, not bytes.
    params.x_batch_stride = x.stride(0);
    params.out_batch_stride = out.stride(0);

    params.scale = scale;
}


at::Tensor
fast_hadamard_transform(at::Tensor &x, float scale) {
    auto input_type = x.scalar_type();
    TORCH_CHECK(input_type == at::ScalarType::Float || input_type == at::ScalarType::Half || input_type == at::ScalarType::BFloat16);

    TORCH_CHECK(x.is_cuda());

    const auto shapes_og = x.sizes();
    const int dim_og = x.size(-1);
    x = x.reshape({-1, dim_og});
    if (x.stride(-1) != 1) { x = x.contiguous(); }
    const auto sizes = x.sizes();
    const int batch_size = sizes[0];

    CHECK_SHAPE(x, batch_size, dim_og);
    TORCH_CHECK(x.stride(1) == 1);

    if (dim_og % 8 != 0) {
        x = torch::nn::functional::pad(x, torch::nn::functional::PadFuncOptions({0, 8 - dim_og % 8}));
    }
    const int dim = x.size(1);

    TORCH_CHECK(dim % 8 == 0, "fast_hadamard_transform only supports hidden dimension divisible by 8 for now");
    TORCH_CHECK(dim <= 32768, "fast_hadamard_transform only supports hidden dimension at most 32768 for now");

    at::Tensor out = torch::empty_like(x);

    HadamardParamsBase params;
    set_hadamard_params(params, batch_size, dim, 1, x, out, scale);

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
#if TORCH_VERSION_MAJOR > 2 || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 6)
    c10::DeviceGuard device_guard(x.device());
#else
    at::cuda::CUDAGuard device_guard{x.device()};
#endif
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    DISPATCH_ITYPE_FLOAT_AND_HALF_AND_BF16(x.scalar_type(), "fast_hadamard_transform", [&] {
        fast_hadamard_transform_cuda<input_t>(params, stream);
    });
    if (dim_og % 8 != 0) {
        out = out.index({torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, dim_og)});
    }
    return out.reshape(shapes_og);
}

at::Tensor
fast_hadamard_transform_12N(at::Tensor &x, float scale) {
    auto input_type = x.scalar_type();
    TORCH_CHECK(input_type == at::ScalarType::Float || input_type == at::ScalarType::Half || input_type == at::ScalarType::BFloat16);

    TORCH_CHECK(x.is_cuda());

    const auto shapes_og = x.sizes();
    const int dim_og = x.size(-1);
    x = x.reshape({-1, dim_og});
    if (x.stride(-1) != 1) { x = x.contiguous(); }
    const auto sizes = x.sizes();
    const int batch_size = sizes[0];

    CHECK_SHAPE(x, batch_size, dim_og);
    TORCH_CHECK(x.stride(1) == 1);

    if (dim_og % (4 * 12) != 0) {
        x = torch::nn::functional::pad(x, torch::nn::functional::PadFuncOptions({0, (4 * 12) - dim_og % (4 * 12)}));
    }
    const int dim = x.size(1);

    TORCH_CHECK(dim % (4 * 12) == 0, "fast_hadamard_transform_12N only supports hidden dimension divisible by 48 for now");
    TORCH_CHECK(dim <= 12 * 1024, "fast_hadamard_transform_12N only supports hidden dimension at most 12288 for now");

    at::Tensor out = torch::empty_like(x);

    HadamardParamsBase params;
    set_hadamard_params(params, batch_size, dim, 12, x, out, scale);

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
#if TORCH_VERSION_MAJOR > 2 || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 6)
    c10::DeviceGuard device_guard(x.device());
#else
    at::cuda::CUDAGuard device_guard{x.device()};
#endif
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    DISPATCH_ITYPE_FLOAT_AND_HALF_AND_BF16(x.scalar_type(), "fast_hadamard_transform", [&] {
        fast_hadamard_transform_12N_cuda<input_t>(params, stream);
    });
    if (dim_og % (4 * 12) != 0) {
        out = out.index({torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, dim_og)});
    }
    return out.reshape(shapes_og);
}

at::Tensor
fast_hadamard_transform_20N(at::Tensor &x, float scale) {
    auto input_type = x.scalar_type();
    TORCH_CHECK(input_type == at::ScalarType::Float || input_type == at::ScalarType::Half || input_type == at::ScalarType::BFloat16);

    TORCH_CHECK(x.is_cuda());

    const auto shapes_og = x.sizes();
    const int dim_og = x.size(-1);
    x = x.reshape({-1, dim_og});
    if (x.stride(-1) != 1) { x = x.contiguous(); }
    const auto sizes = x.sizes();
    const int batch_size = sizes[0];

    CHECK_SHAPE(x, batch_size, dim_og);
    TORCH_CHECK(x.stride(1) == 1);

    if (dim_og % (4 * 20) != 0) {
        x = torch::nn::functional::pad(x, torch::nn::functional::PadFuncOptions({0, (4 * 20) - dim_og % (4 * 20)}));
    }
    const int dim = x.size(1);

    TORCH_CHECK(dim % (4 * 20) == 0, "fast_hadamard_transform_20N only supports hidden dimension divisible by 80 for now");
    TORCH_CHECK(dim <= 20 * 1024, "fast_hadamard_transform_20N only supports hidden dimension at most 20480 for now");

    at::Tensor out = torch::empty_like(x);

    HadamardParamsBase params;
    set_hadamard_params(params, batch_size, dim, 20, x, out, scale);

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
#if TORCH_VERSION_MAJOR > 2 || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 6)
    c10::DeviceGuard device_guard(x.device());
#else
    at::cuda::CUDAGuard device_guard{x.device()};
#endif
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    DISPATCH_ITYPE_FLOAT_AND_HALF_AND_BF16(x.scalar_type(), "fast_hadamard_transform", [&] {
        fast_hadamard_transform_20N_cuda<input_t>(params, stream);
    });
    if (dim_og % (4 * 20) != 0) {
        out = out.index({torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, dim_og)});
    }
    return out.reshape(shapes_og);
}

at::Tensor
fast_hadamard_transform_28N(at::Tensor &x, float scale) {
    auto input_type = x.scalar_type();
    TORCH_CHECK(input_type == at::ScalarType::Float || input_type == at::ScalarType::Half || input_type == at::ScalarType::BFloat16);

    TORCH_CHECK(x.is_cuda());

    const auto shapes_og = x.sizes();
    const int dim_og = x.size(-1);
    x = x.reshape({-1, dim_og});
    if (x.stride(-1) != 1) { x = x.contiguous(); }
    const auto sizes = x.sizes();
    const int batch_size = sizes[0];

    CHECK_SHAPE(x, batch_size, dim_og);
    TORCH_CHECK(x.stride(1) == 1);

    if (dim_og % (4 * 28) != 0) {
        x = torch::nn::functional::pad(x, torch::nn::functional::PadFuncOptions({0, (4 * 28) - dim_og % (4 * 28)}));
    }
    const int dim = x.size(1);

    TORCH_CHECK(dim % (4 * 28) == 0, "fast_hadamard_transform_28N only supports hidden dimension divisible by 112 for now");
    TORCH_CHECK(dim <= 28 * 1024, "fast_hadamard_transform_28N only supports hidden dimension at most 28672 for now");

    at::Tensor out = torch::empty_like(x);

    HadamardParamsBase params;
    set_hadamard_params(params, batch_size, dim, 28, x, out, scale);

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
#if TORCH_VERSION_MAJOR > 2 || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 6)
    c10::DeviceGuard device_guard(x.device());
#else
    at::cuda::CUDAGuard device_guard{x.device()};
#endif
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    DISPATCH_ITYPE_FLOAT_AND_HALF_AND_BF16(x.scalar_type(), "fast_hadamard_transform", [&] {
        fast_hadamard_transform_28N_cuda<input_t>(params, stream);
    });
    if (dim_og % (8 * 28) != 0) {
        out = out.index({torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, dim_og)});
    }
    return out.reshape(shapes_og);
}

at::Tensor
fast_hadamard_transform_40N(at::Tensor &x, float scale) {
    auto input_type = x.scalar_type();
    TORCH_CHECK(input_type == at::ScalarType::Float || input_type == at::ScalarType::Half || input_type == at::ScalarType::BFloat16);

    TORCH_CHECK(x.is_cuda());

    const auto shapes_og = x.sizes();
    const int dim_og = x.size(-1);
    x = x.reshape({-1, dim_og});
    if (x.stride(-1) != 1) { x = x.contiguous(); }
    const auto sizes = x.sizes();
    const int batch_size = sizes[0];

    CHECK_SHAPE(x, batch_size, dim_og);
    TORCH_CHECK(x.stride(1) == 1);

    if (dim_og % (4 * 40) != 0) {
        x = torch::nn::functional::pad(x, torch::nn::functional::PadFuncOptions({0, (4 * 40) - dim_og % (4 * 40)}));
    }
    const int dim = x.size(1);

    TORCH_CHECK(dim % (4 * 40) == 0, "fast_hadamard_transform_40N only supports hidden dimension divisible by 160 for now");
    TORCH_CHECK(dim <= 40 * 1024, "fast_hadamard_transform_40N only supports hidden dimension at most 40960 for now");

    at::Tensor out = torch::empty_like(x);

    HadamardParamsBase params;
    set_hadamard_params(params, batch_size, dim, 40, x, out, scale);

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
#if TORCH_VERSION_MAJOR > 2 || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 6)
    c10::DeviceGuard device_guard(x.device());
#else
    at::cuda::CUDAGuard device_guard{x.device()};
#endif
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    DISPATCH_ITYPE_FLOAT_AND_HALF_AND_BF16(x.scalar_type(), "fast_hadamard_transform", [&] {
        fast_hadamard_transform_40N_cuda<input_t>(params, stream);
    });
    if (dim_og % (8 * 40) != 0) {
        out = out.index({torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, dim_og)});
    }
    return out.reshape(shapes_og);
}

// Pad dimension to next power of 2 (and at least 8) for SORF correctness.
// SORF multi-round diag*FWHT requires all positions up to N=2^k to have
// proper diagonal values; padding only to multiple-of-8 leaves positions
// dim..N-1 with default diag=1.0, causing systematic bias.
static int sorf_pad_dim(int dim_og) {
    if (dim_og <= 8) return 8;
    int p = 1;
    while (p < dim_og) p <<= 1;
    return p;
}

at::Tensor
fast_sorf_transform(at::Tensor &x, at::Tensor &diag, float scale) {
    auto input_type = x.scalar_type();
    TORCH_CHECK(input_type == at::ScalarType::Float || input_type == at::ScalarType::Half || input_type == at::ScalarType::BFloat16);
    TORCH_CHECK(x.is_cuda());
    TORCH_CHECK(diag.is_cuda());
    TORCH_CHECK(diag.scalar_type() == input_type, "diag must have same dtype as x");
    TORCH_CHECK(diag.dim() >= 2, "diag must be at least 2D: (n_rounds, dim)");

    const int n_rounds = diag.size(0);
    TORCH_CHECK(n_rounds >= 1 && n_rounds <= 8, "n_rounds must be between 1 and 8");

    const auto shapes_og = x.sizes();
    const int dim_og = x.size(-1);
    x = x.reshape({-1, dim_og});
    if (x.stride(-1) != 1) { x = x.contiguous(); }
    const auto sizes = x.sizes();
    const int batch_size = sizes[0];

    CHECK_SHAPE(x, batch_size, dim_og);
    TORCH_CHECK(x.stride(1) == 1);

    // Pad to next power of 2 (>= 8) for SORF correctness
    const int dim_padded = sorf_pad_dim(dim_og);
    if (dim_og != dim_padded) {
        x = torch::nn::functional::pad(x, torch::nn::functional::PadFuncOptions({0, dim_padded - dim_og}));
    }
    const int dim = dim_padded;

    TORCH_CHECK(dim % 8 == 0, "dimension must be divisible by 8");
    TORCH_CHECK(dim <= 32768, "dimension at most 32768");

    // Ensure diag covers the padded dimension
    TORCH_CHECK(diag.size(1) >= dim, "diag dim must be >= padded input dim");
    if (diag.stride(-1) != 1) { diag = diag.contiguous(); }

    at::Tensor out = torch::empty_like(x);

    SORFParamsBase params;
    memset(&params, 0, sizeof(params));
    params.batch = batch_size;
    params.dim = dim;
    params.log_N = int(ceil(std::log2(dim)));
    params.x_ptr = x.data_ptr();
    params.out_ptr = out.data_ptr();
    params.x_batch_stride = x.stride(0);
    params.out_batch_stride = out.stride(0);
    params.scale = scale;
    params.diag_ptr = diag.data_ptr();
    params.diag_stride = diag.stride(0);
    params.n_rounds = n_rounds;
    params.input_scale = 1.0f;
    params.rff_scale = 0.0f;
    params.rff_dim = 0;
    params.num_groups = 1;
    params.group_diag_stride = 0;
    params.group_out_stride = 0;

#if TORCH_VERSION_MAJOR > 2 || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 6)
    c10::DeviceGuard device_guard(x.device());
#else
    at::cuda::CUDAGuard device_guard{x.device()};
#endif
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    DISPATCH_ITYPE_FLOAT_AND_HALF_AND_BF16(x.scalar_type(), "fast_sorf_transform", [&] {
        fast_sorf_transform_cuda<input_t>(params, stream);
    });
    if (dim_og != dim_padded) {
        out = out.index({torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, dim_og)});
    }
    return out.reshape(shapes_og);
}

at::Tensor
fast_sorf_rff_transform(at::Tensor &x, at::Tensor &diag, float input_scale, float sorf_scale, float rff_scale, int rff_dim) {
    auto input_type = x.scalar_type();
    TORCH_CHECK(input_type == at::ScalarType::Float || input_type == at::ScalarType::Half || input_type == at::ScalarType::BFloat16);
    TORCH_CHECK(x.is_cuda());
    TORCH_CHECK(diag.is_cuda());
    TORCH_CHECK(diag.scalar_type() == input_type, "diag must have same dtype as x");
    TORCH_CHECK(diag.dim() >= 2, "diag must be at least 2D: (n_rounds, dim)");

    const int n_rounds = diag.size(0);
    TORCH_CHECK(n_rounds >= 1 && n_rounds <= 8, "n_rounds must be between 1 and 8");

    const auto shapes_og = x.sizes();
    const int dim_og = x.size(-1);
    x = x.reshape({-1, dim_og});
    if (x.stride(-1) != 1) { x = x.contiguous(); }
    const int batch_size = x.size(0);

    // Pad to next power of 2 (>= 8) for SORF correctness
    const int dim_padded = sorf_pad_dim(dim_og);
    if (dim_og != dim_padded) {
        x = torch::nn::functional::pad(x, torch::nn::functional::PadFuncOptions({0, dim_padded - dim_og}));
    }
    const int dim = dim_padded;

    TORCH_CHECK(dim % 8 == 0);
    TORCH_CHECK(dim <= 32768);
    TORCH_CHECK(diag.size(1) >= dim);
    TORCH_CHECK(rff_dim > 0 && rff_dim <= dim, "rff_dim must be in (0, dim]");
    if (diag.stride(-1) != 1) { diag = diag.contiguous(); }

    // Output: (batch, 2 * rff_dim) — [cos, sin]
    auto out_shape = std::vector<int64_t>(shapes_og.begin(), shapes_og.end() - 1);
    out_shape.push_back(2 * rff_dim);
    at::Tensor out = torch::empty({batch_size, 2 * rff_dim}, x.options());

    SORFParamsBase params;
    memset(&params, 0, sizeof(params));
    params.batch = batch_size;
    params.dim = dim;
    params.log_N = int(ceil(std::log2(dim)));
    params.x_ptr = x.data_ptr();
    params.out_ptr = out.data_ptr();
    params.x_batch_stride = x.stride(0);
    params.out_batch_stride = out.stride(0);
    params.scale = sorf_scale;
    params.diag_ptr = diag.data_ptr();
    params.diag_stride = diag.stride(0);
    params.n_rounds = n_rounds;
    params.input_scale = input_scale;
    params.rff_scale = rff_scale;
    params.rff_dim = rff_dim;
    params.num_groups = 1;
    params.group_diag_stride = 0;
    params.group_out_stride = 0;

#if TORCH_VERSION_MAJOR > 2 || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 6)
    c10::DeviceGuard device_guard(x.device());
#else
    at::cuda::CUDAGuard device_guard{x.device()};
#endif
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    DISPATCH_ITYPE_FLOAT_AND_HALF_AND_BF16(x.scalar_type(), "fast_sorf_rff_transform", [&] {
        fast_sorf_rff_transform_cuda<input_t>(params, stream);
    });
    return out.reshape(out_shape);
}

// Batched-diag variant: applies G independent SORF blocks in a single launch.
// Inputs:
//   x:    (..., dim_og)
//   diag: (G, n_rounds, dim_padded) — one set of n_rounds Rademacher diagonals per group
// Output:
//   (..., G * 2 * rff_dim) laid out per batch element as
//   [cos_g0..cos_g{G-1} | sin_g0..sin_g{G-1}], each cos_g* / sin_g* of length rff_dim.
//   This packs all-cos before all-sin so the caller can slice [..., :G*rff_dim]
//   and [..., G*rff_dim:] for cos/sin halves without further re-arranging.
at::Tensor
fast_sorf_rff_transform_batched(at::Tensor &x, at::Tensor &diag,
                                float input_scale, float sorf_scale, float rff_scale, int rff_dim) {
    auto input_type = x.scalar_type();
    TORCH_CHECK(input_type == at::ScalarType::Float || input_type == at::ScalarType::Half || input_type == at::ScalarType::BFloat16);
    TORCH_CHECK(x.is_cuda());
    TORCH_CHECK(diag.is_cuda());
    TORCH_CHECK(diag.scalar_type() == input_type, "diag must have same dtype as x");
    TORCH_CHECK(diag.dim() == 3, "diag must be 3D: (num_groups, n_rounds, dim)");

    const int num_groups = diag.size(0);
    const int n_rounds = diag.size(1);
    TORCH_CHECK(num_groups >= 1, "num_groups must be >= 1");
    TORCH_CHECK(n_rounds >= 1 && n_rounds <= 8, "n_rounds must be between 1 and 8");

    const auto shapes_og = x.sizes();
    const int dim_og = x.size(-1);
    x = x.reshape({-1, dim_og});
    if (x.stride(-1) != 1) { x = x.contiguous(); }
    const int batch_size = x.size(0);

    const int dim_padded = sorf_pad_dim(dim_og);
    if (dim_og != dim_padded) {
        x = torch::nn::functional::pad(x, torch::nn::functional::PadFuncOptions({0, dim_padded - dim_og}));
    }
    const int dim = dim_padded;

    TORCH_CHECK(dim % 8 == 0);
    TORCH_CHECK(dim <= 32768);
    TORCH_CHECK(diag.size(2) >= dim, "diag last dim must be >= padded input dim");
    TORCH_CHECK(rff_dim > 0 && rff_dim <= dim, "rff_dim must be in (0, dim]");
    if (!diag.is_contiguous()) { diag = diag.contiguous(); }

    // Output layout per batch element: [cos_g0..cos_g{G-1} | sin_g0..sin_g{G-1}]
    // Stride between groups for cos: rff_dim. Stride between cos-half and sin-half: G*rff_dim.
    // So group g writes cos at offset g*rff_dim, sin at offset G*rff_dim + g*rff_dim.
    // This matches "out_batch_stride = 2*G*rff_dim" with "group_out_stride = rff_dim",
    // BUT requires the kernel's internal store_rff_output to know that sin-half offset
    // is group_out_stride * num_groups (not just rff_dim like single-group case).
    //
    // To avoid changing store_rff_output semantics, we use a simpler layout:
    //   [cos_g0|sin_g0 | cos_g1|sin_g1 | ... | cos_g{G-1}|sin_g{G-1}]
    // i.e., each group's output is laid out exactly like the single-group output
    // (cos then sin, each rff_dim long), and groups are concatenated.
    // group_out_stride = 2*rff_dim, out_batch_stride = num_groups * 2 * rff_dim.
    // Caller can re-layout to [cos_all|sin_all] cheaply via a view + cat.
    const int per_group_out = 2 * rff_dim;
    const int total_out = num_groups * per_group_out;

    auto out_shape = std::vector<int64_t>(shapes_og.begin(), shapes_og.end() - 1);
    out_shape.push_back(total_out);
    at::Tensor out = torch::empty({batch_size, total_out}, x.options());

    SORFParamsBase params;
    memset(&params, 0, sizeof(params));
    params.batch = batch_size;
    params.dim = dim;
    params.log_N = int(ceil(std::log2(dim)));
    params.x_ptr = x.data_ptr();
    params.out_ptr = out.data_ptr();
    params.x_batch_stride = x.stride(0);
    params.out_batch_stride = out.stride(0);  // = total_out
    params.scale = sorf_scale;
    params.diag_ptr = diag.data_ptr();
    params.diag_stride = diag.stride(1);  // stride between rounds
    params.n_rounds = n_rounds;
    params.input_scale = input_scale;
    params.rff_scale = rff_scale;
    params.rff_dim = rff_dim;
    params.num_groups = num_groups;
    params.group_diag_stride = diag.stride(0);  // stride between groups in diag
    params.group_out_stride = per_group_out;    // stride between groups in output

#if TORCH_VERSION_MAJOR > 2 || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 6)
    c10::DeviceGuard device_guard(x.device());
#else
    at::cuda::CUDAGuard device_guard{x.device()};
#endif
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    DISPATCH_ITYPE_FLOAT_AND_HALF_AND_BF16(x.scalar_type(), "fast_sorf_rff_transform_batched", [&] {
        fast_sorf_rff_transform_cuda<input_t>(params, stream);
    });
    return out.reshape(out_shape);
}

at::Tensor
fast_sorf_backward_transform(at::Tensor &x, at::Tensor &diag, float scale) {
    // Backward of SORF: FWHT then diagonal (reversed order from forward).
    // Forward: z = scale * H*D_{n-1}*...*H*D_0 * x
    // Backward: grad_x = scale * D_0*H*...*D_{n-1}*H * grad_z
    // Caller must pass diag with rounds already reversed (diag.flip(0)).
    auto input_type = x.scalar_type();
    TORCH_CHECK(input_type == at::ScalarType::Float || input_type == at::ScalarType::Half || input_type == at::ScalarType::BFloat16);
    TORCH_CHECK(x.is_cuda());
    TORCH_CHECK(diag.is_cuda());
    TORCH_CHECK(diag.scalar_type() == input_type, "diag must have same dtype as x");
    TORCH_CHECK(diag.dim() >= 2, "diag must be at least 2D: (n_rounds, dim)");

    const int n_rounds = diag.size(0);
    TORCH_CHECK(n_rounds >= 1 && n_rounds <= 8, "n_rounds must be between 1 and 8");

    const auto shapes_og = x.sizes();
    const int dim_og = x.size(-1);
    x = x.reshape({-1, dim_og});
    if (x.stride(-1) != 1) { x = x.contiguous(); }
    const int batch_size = x.size(0);

    const int dim_padded = sorf_pad_dim(dim_og);
    if (dim_og != dim_padded) {
        x = torch::nn::functional::pad(x, torch::nn::functional::PadFuncOptions({0, dim_padded - dim_og}));
    }
    const int dim = dim_padded;

    TORCH_CHECK(dim % 8 == 0);
    TORCH_CHECK(dim <= 32768);
    TORCH_CHECK(diag.size(1) >= dim);
    if (diag.stride(-1) != 1) { diag = diag.contiguous(); }

    at::Tensor out = torch::empty_like(x);

    SORFParamsBase params;
    memset(&params, 0, sizeof(params));
    params.batch = batch_size;
    params.dim = dim;
    params.log_N = int(ceil(std::log2(dim)));
    params.x_ptr = x.data_ptr();
    params.out_ptr = out.data_ptr();
    params.x_batch_stride = x.stride(0);
    params.out_batch_stride = out.stride(0);
    params.scale = scale;
    params.diag_ptr = diag.data_ptr();
    params.diag_stride = diag.stride(0);
    params.n_rounds = n_rounds;
    params.input_scale = 1.0f;
    params.rff_scale = 0.0f;
    params.rff_dim = 0;
    params.num_groups = 1;
    params.group_diag_stride = 0;
    params.group_out_stride = 0;

#if TORCH_VERSION_MAJOR > 2 || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 6)
    c10::DeviceGuard device_guard(x.device());
#else
    at::cuda::CUDAGuard device_guard{x.device()};
#endif
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    DISPATCH_ITYPE_FLOAT_AND_HALF_AND_BF16(x.scalar_type(), "fast_sorf_backward_transform", [&] {
        fast_sorf_backward_transform_cuda<input_t>(params, stream);
    });
    if (dim_og != dim_padded) {
        out = out.index({torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, dim_og)});
    }
    return out.reshape(shapes_og);
}

// Batched-diag forward of plain SORF (no cos/sin). Used by autograd backward
// to recompute z. Output shape: (..., G * dim_padded), layout per batch:
//   [z_g0 | z_g1 | ... | z_g{G-1}]
at::Tensor
fast_sorf_transform_batched(at::Tensor &x, at::Tensor &diag, float scale) {
    auto input_type = x.scalar_type();
    TORCH_CHECK(input_type == at::ScalarType::Float || input_type == at::ScalarType::Half || input_type == at::ScalarType::BFloat16);
    TORCH_CHECK(x.is_cuda());
    TORCH_CHECK(diag.is_cuda());
    TORCH_CHECK(diag.scalar_type() == input_type, "diag must have same dtype as x");
    TORCH_CHECK(diag.dim() == 3, "diag must be 3D: (num_groups, n_rounds, dim)");

    const int num_groups = diag.size(0);
    const int n_rounds = diag.size(1);
    TORCH_CHECK(num_groups >= 1);
    TORCH_CHECK(n_rounds >= 1 && n_rounds <= 8);

    const auto shapes_og = x.sizes();
    const int dim_og = x.size(-1);
    x = x.reshape({-1, dim_og});
    if (x.stride(-1) != 1) { x = x.contiguous(); }
    const int batch_size = x.size(0);

    const int dim_padded = sorf_pad_dim(dim_og);
    if (dim_og != dim_padded) {
        x = torch::nn::functional::pad(x, torch::nn::functional::PadFuncOptions({0, dim_padded - dim_og}));
    }
    const int dim = dim_padded;

    TORCH_CHECK(dim % 8 == 0);
    TORCH_CHECK(dim <= 32768);
    TORCH_CHECK(diag.size(2) >= dim);
    if (!diag.is_contiguous()) { diag = diag.contiguous(); }

    auto out_shape = std::vector<int64_t>(shapes_og.begin(), shapes_og.end() - 1);
    out_shape.push_back(num_groups * dim);
    at::Tensor out = torch::empty({batch_size, num_groups * dim}, x.options());

    SORFParamsBase params;
    memset(&params, 0, sizeof(params));
    params.batch = batch_size;
    params.dim = dim;
    params.log_N = int(ceil(std::log2(dim)));
    params.x_ptr = x.data_ptr();
    params.out_ptr = out.data_ptr();
    params.x_batch_stride = x.stride(0);
    params.out_batch_stride = out.stride(0);
    params.scale = scale;
    params.diag_ptr = diag.data_ptr();
    params.diag_stride = diag.stride(1);
    params.n_rounds = n_rounds;
    params.input_scale = 1.0f;
    params.rff_scale = 0.0f;
    params.rff_dim = 0;
    params.num_groups = num_groups;
    params.group_diag_stride = diag.stride(0);
    params.group_out_stride = dim;

#if TORCH_VERSION_MAJOR > 2 || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 6)
    c10::DeviceGuard device_guard(x.device());
#else
    at::cuda::CUDAGuard device_guard{x.device()};
#endif
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    DISPATCH_ITYPE_FLOAT_AND_HALF_AND_BF16(x.scalar_type(), "fast_sorf_transform_batched", [&] {
        fast_sorf_transform_cuda<input_t>(params, stream);
    });
    return out.reshape(out_shape);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fast_hadamard_transform", &fast_hadamard_transform, "Fast Hadamard transform");
    m.def("fast_sorf_transform", &fast_sorf_transform, "Fused SORF transform: n rounds of diagonal * Hadamard");
    m.def("fast_sorf_backward_transform", &fast_sorf_backward_transform, "Fused SORF backward: n rounds of Hadamard * diagonal (reversed order)");
    m.def("fast_sorf_rff_transform", &fast_sorf_rff_transform, "Fused SORF RFF: input_scale * x -> n rounds diag*H -> [cos,sin]*rff_scale");
    m.def("fast_sorf_rff_transform_batched", &fast_sorf_rff_transform_batched, "Fused SORF RFF over G groups in one launch; diag is (G, n_rounds, dim)");
    m.def("fast_sorf_transform_batched", &fast_sorf_transform_batched, "Fused SORF (no cos/sin) over G groups in one launch; diag is (G, n_rounds, dim)");
    m.def("fast_hadamard_transform_12N", &fast_hadamard_transform_12N, "Fast Hadamard transform with dimension = 12 * power of 2");
    m.def("fast_hadamard_transform_20N", &fast_hadamard_transform_20N, "Fast Hadamard transform with dimension = 20 * power of 2");
    m.def("fast_hadamard_transform_28N", &fast_hadamard_transform_28N, "Fast Hadamard transform with dimension = 28 * power of 2");
    m.def("fast_hadamard_transform_40N", &fast_hadamard_transform_40N, "Fast Hadamard transform with dimension = 40 * power of 2");
}
