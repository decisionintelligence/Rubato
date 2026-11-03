/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

// #pragma once

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/cuda/CUDAException.h>  // For C10_CUDA_CHECK and C10_CUDA_KERNEL_LAUNCH_CHECK

#include "fast_hadamard_transform.h"
#include "fast_hadamard_transform_common.h"
#include "fast_hadamard_transform_special.h"
#include "static_switch.h"

template<int kNThreads_, int kLogN_, typename input_t_>
struct fast_hadamard_transform_kernel_traits {
    using input_t = input_t_;
    static constexpr int kNThreads = kNThreads_;
    static constexpr int kLogN = kLogN_;
    static constexpr int N = 1 << kLogN;
    static constexpr int kNBytes = sizeof(input_t);
    static_assert(kNBytes == 2 || kNBytes == 4);
    static constexpr int kNElts = kNBytes == 4 ? 4 : 8;
    // It's possible that we need to do 2 rounds of exchange if input_t is 16 bits
    // (since then we'd have 8 values of float, and each round we can exchange 4 floats).
    static constexpr int kNExchangePerVec = sizeof(float) / sizeof(input_t);
    using vec_t = typename BytesToType<kNBytes * kNElts>::Type;
    static constexpr int kNChunks = N / (kNElts * kNThreads);
    // We don't want to use more than 32 KB of shared memory.
    static constexpr int kSmemExchangeSize = std::min(N * 4, 32 * 1024);
    static constexpr int kNExchangeRounds = N * 4 / kSmemExchangeSize;
    static_assert(kNExchangeRounds * kSmemExchangeSize == N * 4);
    static constexpr int kSmemSize = kSmemExchangeSize;
};

template<int kNThreads_, int kLogN_, typename input_t_>
struct fast_hadamard_transform_12N_kernel_traits {
    using input_t = input_t_;
    static constexpr int kNThreads = kNThreads_;
    static constexpr int kLogN = kLogN_;
    static constexpr int N = (1 << kLogN) * 12;
    static_assert(N <= 12 * 1024, "fast_hadamard_transform_12 only supports dim <= 12288");
    static constexpr int kNBytes = sizeof(input_t);
    static_assert(kNBytes == 2 || kNBytes == 4);
    static constexpr int kNElts = 4;
    // It's possible that we need to do 2 rounds of exchange if input_t is 16 bits
    // (since then we'd have 8 values of float, and each round we can exchange 4 floats).
    static constexpr int kNExchangePerVec = sizeof(float) / sizeof(input_t);
    using vec_t = typename BytesToType<kNBytes * kNElts>::Type;
    static constexpr int kNChunks = N / (kNElts * kNThreads);
    static_assert(kNChunks == 12);
    // We don't want to use more than 24 KB of shared memory.
    static constexpr int kSmemExchangeSize = std::min(N * 4, 24 * 1024);
    static constexpr int kNExchangeRounds = N * 4 / kSmemExchangeSize;
    static_assert(kNExchangeRounds * kSmemExchangeSize == N * 4);
    static constexpr int kSmemSize = kSmemExchangeSize;
};

template<int kNThreads_, int kLogN_, typename input_t_>
struct fast_hadamard_transform_20N_kernel_traits {
    using input_t = input_t_;
    static constexpr int kNThreads = kNThreads_;
    static constexpr int kLogN = kLogN_;
    static constexpr int N = (1 << kLogN) * 20;
    static_assert(N <= 20 * 1024, "fast_hadamard_transform_20 only supports dim <= 20480");
    static constexpr int kNBytes = sizeof(input_t);
    static_assert(kNBytes == 2 || kNBytes == 4);
    static constexpr int kNElts = 4;
    // It's possible that we need to do 2 rounds of exchange if input_t is 16 bits
    // (since then we'd have 8 values of float, and each round we can exchange 4 floats).
    static constexpr int kNExchangePerVec = sizeof(float) / sizeof(input_t);
    using vec_t = typename BytesToType<kNBytes * kNElts>::Type;
    static constexpr int kNChunks = N / (kNElts * kNThreads);
    static_assert(kNChunks == 20);
    // We don't want to use more than 40 KB of shared memory.
    static constexpr int kSmemExchangeSize = std::min(N * 4, 40 * 1024);
    static constexpr int kNExchangeRounds = N * 4 / kSmemExchangeSize;
    static_assert(kNExchangeRounds * kSmemExchangeSize == N * 4);
    static constexpr int kSmemSize = kSmemExchangeSize;
};

template<int kNThreads_, int kLogN_, typename input_t_>
struct fast_hadamard_transform_28N_kernel_traits {
    using input_t = input_t_;
    static constexpr int kNThreads = kNThreads_;
    static constexpr int kLogN = kLogN_;
    static constexpr int N = (1 << kLogN) * 28;
    static_assert(N <= 28 * 1024, "fast_hadamard_transform_28 only supports dim <= 28672");
    static constexpr int kNBytes = sizeof(input_t);
    static_assert(kNBytes == 2 || kNBytes == 4);
    static constexpr int kNElts = 4;
    // It's possible that we need to do 2 rounds of exchange if input_t is 16 bits
    // (since then we'd have 8 values of float, and each round we can exchange 4 floats).
    static constexpr int kNExchangePerVec = sizeof(float) / sizeof(input_t);
    using vec_t = typename BytesToType<kNBytes * kNElts>::Type;
    static constexpr int kNChunks = N / (kNElts * kNThreads);
    static_assert(kNChunks == 28);
    // We don't want to use more than 28 KB of shared memory.
    static constexpr int kSmemExchangeSize = std::min(N * 4, 28 * 1024);
    static constexpr int kNExchangeRounds = N * 4 / kSmemExchangeSize;
    static_assert(kNExchangeRounds * kSmemExchangeSize == N * 4);
    static constexpr int kSmemSize = kSmemExchangeSize;
};

template<int kNThreads_, int kLogN_, typename input_t_>
struct fast_hadamard_transform_40N_kernel_traits {
    using input_t = input_t_;
    static constexpr int kNThreads = kNThreads_;
    static constexpr int kLogN = kLogN_;
    static constexpr int N = (1 << kLogN) * 40;
    static_assert(N <= 40 * 1024, "fast_hadamard_transform_40 only supports dim <= 40960");
    static constexpr int kNBytes = sizeof(input_t);
    static_assert(kNBytes == 2 || kNBytes == 4);
    static constexpr int kNElts = 4;
    // It's possible that we need to do 2 rounds of exchange if input_t is 16 bits
    // (since then we'd have 8 values of float, and each round we can exchange 4 floats).
    static constexpr int kNExchangePerVec = sizeof(float) / sizeof(input_t);
    using vec_t = typename BytesToType<kNBytes * kNElts>::Type;
    static constexpr int kNChunks = N / (kNElts * kNThreads);
    static_assert(kNChunks == 40);
    // We don't want to use more than 40 KB of shared memory.
    static constexpr int kSmemExchangeSize = std::min(N * 4, 40 * 1024);
    static constexpr int kNExchangeRounds = N * 4 / kSmemExchangeSize;
    static_assert(kNExchangeRounds * kSmemExchangeSize == N * 4);
    static constexpr int kSmemSize = kSmemExchangeSize;
};

template <int kNChunks>
__device__ __forceinline__ void hadamard_mult_thread_chunk_12(float x[kNChunks][12]) {
    #pragma unroll
    for (int c = 0; c < kNChunks; ++c) { hadamard_mult_thread_12(x[c]); }
}

template <int kNChunks>
__device__ __forceinline__ void hadamard_mult_thread_chunk_20(float x[kNChunks][20]) {
    #pragma unroll
    for (int c = 0; c < kNChunks; ++c) { hadamard_mult_thread_20(x[c]); }
}

template <int kNChunks>
__device__ __forceinline__ void hadamard_mult_thread_chunk_28(float x[kNChunks][28]) {
    #pragma unroll
    for (int c = 0; c < kNChunks; ++c) { hadamard_mult_thread_28(x[c]); }
}

template <int kNChunks>
__device__ __forceinline__ void hadamard_mult_thread_chunk_40(float x[kNChunks][40]) {
    #pragma unroll
    for (int c = 0; c < kNChunks; ++c) { hadamard_mult_thread_40(x[c]); }
}

template<typename Ktraits>
__global__ __launch_bounds__(Ktraits::kNThreads)
void fast_hadamard_transform_kernel(HadamardParamsBase params) {
    constexpr int kNThreads = Ktraits::kNThreads;
    constexpr int kNElts = Ktraits::kNElts;
    constexpr int kNExchangePerVec = Ktraits::kNExchangePerVec;
    constexpr int kNExchangeRounds = Ktraits::kNExchangeRounds;
    constexpr int kNChunks = Ktraits::kNChunks;
    using input_t = typename Ktraits::input_t;
    using vec_t = typename Ktraits::vec_t;

    constexpr int kLogNElts = cilog2(Ktraits::kNElts);
    static_assert(1 << kLogNElts == kNElts, "kNElts must be a power of 2");
    constexpr int kWarpSize = std::min(kNThreads, 32);
    constexpr int kLogWarpSize = cilog2(kWarpSize);
    static_assert(1 << kLogWarpSize == kWarpSize, "Warp size must be a power of 2");
    constexpr int kNWarps = kNThreads / kWarpSize;
    constexpr int kLogNWarps = cilog2(kNWarps);
    static_assert(1 << kLogNWarps == kNWarps, "kNWarps must be a power of 2");
    constexpr int kLoadsPerExchange = Ktraits::kSmemExchangeSize / (sizeof(vec_t) * kNThreads);
    static_assert(kLoadsPerExchange * sizeof(vec_t) * kNThreads == Ktraits::kSmemExchangeSize, "kSmemExchangeSize should be a power of 2");
    static_assert(kNExchangeRounds * kLoadsPerExchange * sizeof(vec_t) == kNChunks * kNElts * sizeof(float));

    constexpr int kChunksPerExchange = Ktraits::kSmemExchangeSize / (sizeof(vec_t) * kNExchangePerVec * kNThreads);
    static_assert(kChunksPerExchange * sizeof(vec_t) * kNExchangePerVec * kNThreads == Ktraits::kSmemExchangeSize);
    constexpr int kNExchanges = kNChunks / kChunksPerExchange;
    static_assert(kNExchanges * kChunksPerExchange == kNChunks);

    // Shared memory.
    extern __shared__ char smem_[];
    vec_t *smem_exchange = reinterpret_cast<vec_t *>(smem_);

    const int batch_id = blockIdx.x;
    input_t *x = reinterpret_cast<input_t *>(params.x_ptr) + batch_id * params.x_batch_stride;
    input_t *out = reinterpret_cast<input_t *>(params.out_ptr) + batch_id * params.out_batch_stride;

    float x_vals[kNChunks][kNElts];
    load_input<kNChunks, kNElts, input_t>(x, x_vals, params.dim);

    hadamard_mult_thread<kLogNElts, kNChunks>(x_vals);
    hadamard_mult_warp<kLogWarpSize, 0, kNChunks, kNElts>(x_vals);

    if constexpr (kNWarps > 1) {
        exchange_smem_pre<kNChunks, kChunksPerExchange, kNElts, kWarpSize, kNWarps, true, vec_t>(x_vals, smem_exchange);
        hadamard_mult_warp<kLogNWarps, 0, kNChunks, kNElts>(x_vals);
        exchange_smem_pre<kNChunks, kChunksPerExchange, kNElts, kWarpSize, kNWarps, false, vec_t>(x_vals, smem_exchange);
    }

    if constexpr (kNChunks > 1) {
        float x_vals_transposed[kNElts][kNChunks];
        #pragma unroll
        for (int c = 0; c < kNChunks; ++c) {
            #pragma unroll
            for (int i = 0; i < kNElts; ++i) { x_vals_transposed[i][c] = x_vals[c][i]; }
        }
        if constexpr (kNChunks == 12) {
            hadamard_mult_thread_chunk_12<kNElts>(x_vals_transposed);
        } else if constexpr (kNChunks == 20) {
            hadamard_mult_thread_chunk_20<kNElts>(x_vals_transposed);
        } else if constexpr (kNChunks == 28) {
            hadamard_mult_thread_chunk_28<kNElts>(x_vals_transposed);
        } else if constexpr (kNChunks == 40) {
            hadamard_mult_thread_chunk_40<kNElts>(x_vals_transposed);
        } else {
            constexpr int kLogNChunks = cilog2(kNChunks);
            static_assert(1 << kLogNChunks == kNChunks, "kNChunks must be a power of 2");
            hadamard_mult_thread<kLogNChunks, kNElts>(x_vals_transposed);
        }
        #pragma unroll
        for (int c = 0; c < kNChunks; ++c) {
            #pragma unroll
            for (int i = 0; i < kNElts; ++i) { x_vals[c][i] = x_vals_transposed[i][c]; }
        }
    }

    store_output<kNChunks, kNElts, input_t>(out, x_vals, params.dim, params.scale);
}

template<int kNThreads, int kLogN, typename input_t>
void fast_hadamard_transform_launch(HadamardParamsBase &params, cudaStream_t stream) {
    using Ktraits = fast_hadamard_transform_kernel_traits<kNThreads, kLogN, input_t>;
    constexpr int kSmemSize = Ktraits::kSmemSize;
    dim3 grid(params.batch);
    auto kernel = &fast_hadamard_transform_kernel<Ktraits>;
    if (kSmemSize >= 48 * 1024) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
        }
    kernel<<<grid, Ktraits::kNThreads, kSmemSize, stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<typename input_t>
void fast_hadamard_transform_cuda(HadamardParamsBase &params, cudaStream_t stream) {
    if (params.log_N == 3) {
        fast_hadamard_transform_launch<1, 3, input_t>(params, stream);
    } else if (params.log_N == 4) {
        fast_hadamard_transform_launch<2, 4, input_t>(params, stream);
    } else if (params.log_N == 5) {
        fast_hadamard_transform_launch<4, 5, input_t>(params, stream);
    } else if (params.log_N == 6) {
        fast_hadamard_transform_launch<8, 6, input_t>(params, stream);
    } else if (params.log_N == 7) {
        fast_hadamard_transform_launch<16, 7, input_t>(params, stream);
    } else if (params.log_N == 8) {
        fast_hadamard_transform_launch<32, 8, input_t>(params, stream);
    } else if (params.log_N == 9) {
        fast_hadamard_transform_launch<32, 9, input_t>(params, stream);
    } else if (params.log_N == 10) {
        fast_hadamard_transform_launch<128, 10, input_t>(params, stream);
    } else if (params.log_N == 11) {
        fast_hadamard_transform_launch<256, 11, input_t>(params, stream);
    } else if (params.log_N == 12) {
        fast_hadamard_transform_launch<256, 12, input_t>(params, stream);
    } else if (params.log_N == 13) {
        fast_hadamard_transform_launch<256, 13, input_t>(params, stream);
    } else if (params.log_N == 14) {
        fast_hadamard_transform_launch<256, 14, input_t>(params, stream);
    } else if (params.log_N == 15) {
        fast_hadamard_transform_launch<256, 15, input_t>(params, stream);
    }
}

template<int kNThreads, int kLogN, typename input_t>
void fast_hadamard_transform_12N_launch(HadamardParamsBase &params, cudaStream_t stream) {
    using Ktraits = fast_hadamard_transform_12N_kernel_traits<kNThreads, kLogN, input_t>;
    constexpr int kSmemSize = Ktraits::kSmemSize;
    dim3 grid(params.batch);
    auto kernel = &fast_hadamard_transform_kernel<Ktraits>;
    if (kSmemSize >= 48 * 1024) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
        }
    kernel<<<grid, Ktraits::kNThreads, kSmemSize, stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<typename input_t>
void fast_hadamard_transform_12N_cuda(HadamardParamsBase &params, cudaStream_t stream) {
    if (params.log_N == 2) {
        fast_hadamard_transform_12N_launch<1, 2, input_t>(params, stream);
    } else if (params.log_N == 3) {
        fast_hadamard_transform_12N_launch<2, 3, input_t>(params, stream);
    } else if (params.log_N == 4) {
        fast_hadamard_transform_12N_launch<4, 4, input_t>(params, stream);
    } else if (params.log_N == 5) {
        fast_hadamard_transform_12N_launch<8, 5, input_t>(params, stream);
    } else if (params.log_N == 6) {
        fast_hadamard_transform_12N_launch<16, 6, input_t>(params, stream);
    } else if (params.log_N == 7) {
        fast_hadamard_transform_12N_launch<32, 7, input_t>(params, stream);
    } else if (params.log_N == 8) {
        fast_hadamard_transform_12N_launch<64, 8, input_t>(params, stream);
    } else if (params.log_N == 9) {
        fast_hadamard_transform_12N_launch<128, 9, input_t>(params, stream);
    } else if (params.log_N == 10) {
        fast_hadamard_transform_12N_launch<256, 10, input_t>(params, stream);
    } 
}

template<int kNThreads, int kLogN, typename input_t>
void fast_hadamard_transform_20N_launch(HadamardParamsBase &params, cudaStream_t stream) {
    using Ktraits = fast_hadamard_transform_20N_kernel_traits<kNThreads, kLogN, input_t>;
    constexpr int kSmemSize = Ktraits::kSmemSize;
    dim3 grid(params.batch);
    auto kernel = &fast_hadamard_transform_kernel<Ktraits>;
    if (kSmemSize >= 48 * 1024) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
        }
    kernel<<<grid, Ktraits::kNThreads, kSmemSize, stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<typename input_t>
void fast_hadamard_transform_20N_cuda(HadamardParamsBase &params, cudaStream_t stream) {
    if (params.log_N == 2) {
        fast_hadamard_transform_20N_launch<1, 2, input_t>(params, stream);
    } else if (params.log_N == 3) {
        fast_hadamard_transform_20N_launch<2, 3, input_t>(params, stream);
    } else if (params.log_N == 4) {
        fast_hadamard_transform_20N_launch<4, 4, input_t>(params, stream);
    } else if (params.log_N == 5) {
        fast_hadamard_transform_20N_launch<8, 5, input_t>(params, stream);
    } else if (params.log_N == 6) {
        fast_hadamard_transform_20N_launch<16, 6, input_t>(params, stream);
    } else if (params.log_N == 7) {
        fast_hadamard_transform_20N_launch<32, 7, input_t>(params, stream);
    } else if (params.log_N == 8) {
        fast_hadamard_transform_20N_launch<64, 8, input_t>(params, stream);
    } else if (params.log_N == 9) {
        fast_hadamard_transform_20N_launch<128, 9, input_t>(params, stream);
    } else if (params.log_N == 10) {
        fast_hadamard_transform_20N_launch<256, 10, input_t>(params, stream);
    }
}

template<int kNThreads, int kLogN, typename input_t>
void fast_hadamard_transform_28N_launch(HadamardParamsBase &params, cudaStream_t stream) {
    using Ktraits = fast_hadamard_transform_28N_kernel_traits<kNThreads, kLogN, input_t>;
    constexpr int kSmemSize = Ktraits::kSmemSize;
    dim3 grid(params.batch);
    auto kernel = &fast_hadamard_transform_kernel<Ktraits>;
    if (kSmemSize >= 48 * 1024) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
        }
    kernel<<<grid, Ktraits::kNThreads, kSmemSize, stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<typename input_t>
void fast_hadamard_transform_28N_cuda(HadamardParamsBase &params, cudaStream_t stream) {
    if (params.log_N == 2) {
        fast_hadamard_transform_28N_launch<1, 2, input_t>(params, stream);
    } else if (params.log_N == 3) {
        fast_hadamard_transform_28N_launch<2, 3, input_t>(params, stream);
    } else if (params.log_N == 4) {
        fast_hadamard_transform_28N_launch<4, 4, input_t>(params, stream);
    } else if (params.log_N == 5) {
        fast_hadamard_transform_28N_launch<8, 5, input_t>(params, stream);
    } else if (params.log_N == 6) {
        fast_hadamard_transform_28N_launch<16, 6, input_t>(params, stream);
    } else if (params.log_N == 7) {
        fast_hadamard_transform_28N_launch<32, 7, input_t>(params, stream);
    } else if (params.log_N == 8) {
        fast_hadamard_transform_28N_launch<64, 8, input_t>(params, stream);
    } else if (params.log_N == 9) {
        fast_hadamard_transform_28N_launch<128, 9, input_t>(params, stream);
    } else if (params.log_N == 10) {
        fast_hadamard_transform_28N_launch<256, 10, input_t>(params, stream);
    }
}

template<int kNThreads, int kLogN, typename input_t>
void fast_hadamard_transform_40N_launch(HadamardParamsBase &params, cudaStream_t stream) {
    using Ktraits = fast_hadamard_transform_40N_kernel_traits<kNThreads, kLogN, input_t>;
    constexpr int kSmemSize = Ktraits::kSmemSize;
    dim3 grid(params.batch);
    auto kernel = &fast_hadamard_transform_kernel<Ktraits>;
    if (kSmemSize >= 48 * 1024) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
        }
    kernel<<<grid, Ktraits::kNThreads, kSmemSize, stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<typename input_t>
void fast_hadamard_transform_40N_cuda(HadamardParamsBase &params, cudaStream_t stream) {
    if (params.log_N == 2) {
        fast_hadamard_transform_40N_launch<1, 2, input_t>(params, stream);
    } else if (params.log_N == 3) {
        fast_hadamard_transform_40N_launch<2, 3, input_t>(params, stream);
    } else if (params.log_N == 4) {
        fast_hadamard_transform_40N_launch<4, 4, input_t>(params, stream);
    } else if (params.log_N == 5) {
        fast_hadamard_transform_40N_launch<8, 5, input_t>(params, stream);
    } else if (params.log_N == 6) {
        fast_hadamard_transform_40N_launch<16, 6, input_t>(params, stream);
    } else if (params.log_N == 7) {
        fast_hadamard_transform_40N_launch<32, 7, input_t>(params, stream);
    } else if (params.log_N == 8) {
        fast_hadamard_transform_40N_launch<64, 8, input_t>(params, stream);
    } else if (params.log_N == 9) {
        fast_hadamard_transform_40N_launch<128, 9, input_t>(params, stream);
    } else if (params.log_N == 10) {
        fast_hadamard_transform_40N_launch<256, 10, input_t>(params, stream);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Fused SORF kernel: n_rounds of (diagonal * FWHT) in a single kernel launch.
// Data stays in registers/smem between rounds, eliminating global memory round-trips.
////////////////////////////////////////////////////////////////////////////////////////////////////

template <int kNChunks, int kNElts, typename input_t>
inline __device__ void apply_diagonal(float x_vals[kNChunks][kNElts], const input_t *diag, int dim) {
    using vec_t = typename BytesToType<sizeof(input_t) * kNElts>::Type;
    input_t diag_load[kNChunks][kNElts];
    #pragma unroll
    for (int c = 0; c < kNChunks; ++c) {
        if ((c * blockDim.x + threadIdx.x) * kNElts < dim) {
            reinterpret_cast<vec_t*>(diag_load)[c] = reinterpret_cast<const vec_t*>(diag)[c * blockDim.x + threadIdx.x];
        } else {
            #pragma unroll
            for (int i = 0; i < kNElts; ++i) { diag_load[c][i] = input_t(1.0f); }
        }
    }
    #pragma unroll
    for (int c = 0; c < kNChunks; ++c) {
        #pragma unroll
        for (int i = 0; i < kNElts; ++i) { x_vals[c][i] *= float(diag_load[c][i]); }
    }
}

template <int kNChunks, int kNElts, typename output_t>
inline __device__ void store_rff_output(output_t *out, float x_vals[kNChunks][kNElts],
                                         int dim, int rff_dim, float sorf_scale, float rff_scale,
                                         int out_batch_stride) {
    // Write cos(z*sorf_scale)*rff_scale to out[0..rff_dim) and
    //       sin(z*sorf_scale)*rff_scale to out[rff_dim..2*rff_dim)
    #pragma unroll
    for (int c = 0; c < kNChunks; ++c) {
        #pragma unroll
        for (int i = 0; i < kNElts; ++i) {
            int idx = (c * blockDim.x + threadIdx.x) * kNElts + i;
            if (idx < rff_dim) {
                float val = x_vals[c][i] * sorf_scale;
                out[idx] = output_t(cosf(val) * rff_scale);
                out[idx + rff_dim] = output_t(sinf(val) * rff_scale);
            }
        }
    }
}

// Helper: perform one round of in-register FWHT
template<typename Ktraits>
__device__ __forceinline__ void do_fwht_round(
    float x_vals[Ktraits::kNChunks][Ktraits::kNElts],
    typename Ktraits::vec_t *smem_exchange) {
    constexpr int kNThreads = Ktraits::kNThreads;
    constexpr int kNElts = Ktraits::kNElts;
    constexpr int kNExchangePerVec = Ktraits::kNExchangePerVec;
    constexpr int kNChunks = Ktraits::kNChunks;
    using vec_t = typename Ktraits::vec_t;

    constexpr int kLogNElts = cilog2(Ktraits::kNElts);
    constexpr int kWarpSize = std::min(kNThreads, 32);
    constexpr int kLogWarpSize = cilog2(kWarpSize);
    constexpr int kNWarps = kNThreads / kWarpSize;
    constexpr int kLogNWarps = cilog2(kNWarps);
    constexpr int kChunksPerExchange = Ktraits::kSmemExchangeSize / (sizeof(vec_t) * kNExchangePerVec * kNThreads);

    hadamard_mult_thread<kLogNElts, kNChunks>(x_vals);
    hadamard_mult_warp<kLogWarpSize, 0, kNChunks, kNElts>(x_vals);

    if constexpr (kNWarps > 1) {
        exchange_smem_pre<kNChunks, kChunksPerExchange, kNElts, kWarpSize, kNWarps, true, vec_t>(x_vals, smem_exchange);
        hadamard_mult_warp<kLogNWarps, 0, kNChunks, kNElts>(x_vals);
        exchange_smem_pre<kNChunks, kChunksPerExchange, kNElts, kWarpSize, kNWarps, false, vec_t>(x_vals, smem_exchange);
    }

    if constexpr (kNChunks > 1) {
        float x_vals_transposed[kNElts][kNChunks];
        #pragma unroll
        for (int c = 0; c < kNChunks; ++c) {
            #pragma unroll
            for (int i = 0; i < kNElts; ++i) { x_vals_transposed[i][c] = x_vals[c][i]; }
        }
        constexpr int kLogNChunks = cilog2(kNChunks);
        static_assert(1 << kLogNChunks == kNChunks, "kNChunks must be a power of 2");
        hadamard_mult_thread<kLogNChunks, kNElts>(x_vals_transposed);
        #pragma unroll
        for (int c = 0; c < kNChunks; ++c) {
            #pragma unroll
            for (int i = 0; i < kNElts; ++i) { x_vals[c][i] = x_vals_transposed[i][c]; }
        }
    }
}

// kReverse=false (forward): each round does Diagonal then FWHT
// kReverse=true  (backward): each round does FWHT then Diagonal
template<typename Ktraits, bool kComputeRFF, bool kReverse>
__global__ __launch_bounds__(Ktraits::kNThreads)
void fast_sorf_transform_kernel(SORFParamsBase params) {
    constexpr int kNChunks = Ktraits::kNChunks;
    constexpr int kNElts = Ktraits::kNElts;
    using input_t = typename Ktraits::input_t;
    using vec_t = typename Ktraits::vec_t;

    extern __shared__ char smem_[];
    vec_t *smem_exchange = reinterpret_cast<vec_t *>(smem_);

    const int batch_id = blockIdx.x;
    const int group_id = blockIdx.y;  // 0 if num_groups == 1
    input_t *x = reinterpret_cast<input_t *>(params.x_ptr) + batch_id * params.x_batch_stride;
    input_t *out = reinterpret_cast<input_t *>(params.out_ptr)
                   + batch_id * params.out_batch_stride
                   + group_id * params.group_out_stride;
    input_t *diag_base = reinterpret_cast<input_t *>(params.diag_ptr)
                         + group_id * params.group_diag_stride;

    float x_vals[kNChunks][kNElts];
    load_input<kNChunks, kNElts, input_t>(x, x_vals, params.dim);

    // Apply input scale (e.g. 1/sigma)
    if (params.input_scale != 1.0f) {
        #pragma unroll
        for (int c = 0; c < kNChunks; ++c) {
            #pragma unroll
            for (int i = 0; i < kNElts; ++i) { x_vals[c][i] *= params.input_scale; }
        }
    }

    // n_rounds of (diag, FWHT) or (FWHT, diag) depending on kReverse
    for (int round = 0; round < params.n_rounds; ++round) {
        if constexpr (kReverse) {
            // Backward order: FWHT first, then diagonal
            do_fwht_round<Ktraits>(x_vals, smem_exchange);
            apply_diagonal<kNChunks, kNElts, input_t>(
                x_vals, diag_base + round * params.diag_stride, params.dim);
        } else {
            // Forward order: diagonal first, then FWHT
            apply_diagonal<kNChunks, kNElts, input_t>(
                x_vals, diag_base + round * params.diag_stride, params.dim);
            do_fwht_round<Ktraits>(x_vals, smem_exchange);
        }
    }

    if constexpr (kComputeRFF) {
        store_rff_output<kNChunks, kNElts, input_t>(
            out, x_vals, params.dim, params.rff_dim,
            params.scale, params.rff_scale, params.out_batch_stride);
    } else {
        store_output<kNChunks, kNElts, input_t>(out, x_vals, params.dim, params.scale);
    }
}

template<int kNThreads, int kLogN, typename input_t, bool kComputeRFF, bool kReverse>
void fast_sorf_transform_launch(SORFParamsBase &params, cudaStream_t stream) {
    using Ktraits = fast_hadamard_transform_kernel_traits<kNThreads, kLogN, input_t>;
    constexpr int kSmemSize = Ktraits::kSmemSize;
    // Grid Y dim is num_groups (default 1 → backward compatible single-group launch).
    dim3 grid(params.batch, params.num_groups > 0 ? params.num_groups : 1);
    auto kernel = &fast_sorf_transform_kernel<Ktraits, kComputeRFF, kReverse>;
    if (kSmemSize >= 48 * 1024) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
    }
    kernel<<<grid, Ktraits::kNThreads, kSmemSize, stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<typename input_t, bool kComputeRFF, bool kReverse>
void fast_sorf_transform_cuda_impl(SORFParamsBase &params, cudaStream_t stream) {
    if (params.log_N == 3) {
        fast_sorf_transform_launch<1, 3, input_t, kComputeRFF, kReverse>(params, stream);
    } else if (params.log_N == 4) {
        fast_sorf_transform_launch<2, 4, input_t, kComputeRFF, kReverse>(params, stream);
    } else if (params.log_N == 5) {
        fast_sorf_transform_launch<4, 5, input_t, kComputeRFF, kReverse>(params, stream);
    } else if (params.log_N == 6) {
        fast_sorf_transform_launch<8, 6, input_t, kComputeRFF, kReverse>(params, stream);
    } else if (params.log_N == 7) {
        fast_sorf_transform_launch<16, 7, input_t, kComputeRFF, kReverse>(params, stream);
    } else if (params.log_N == 8) {
        fast_sorf_transform_launch<32, 8, input_t, kComputeRFF, kReverse>(params, stream);
    } else if (params.log_N == 9) {
        fast_sorf_transform_launch<32, 9, input_t, kComputeRFF, kReverse>(params, stream);
    } else if (params.log_N == 10) {
        fast_sorf_transform_launch<128, 10, input_t, kComputeRFF, kReverse>(params, stream);
    } else if (params.log_N == 11) {
        fast_sorf_transform_launch<256, 11, input_t, kComputeRFF, kReverse>(params, stream);
    } else if (params.log_N == 12) {
        fast_sorf_transform_launch<256, 12, input_t, kComputeRFF, kReverse>(params, stream);
    } else if (params.log_N == 13) {
        fast_sorf_transform_launch<256, 13, input_t, kComputeRFF, kReverse>(params, stream);
    } else if (params.log_N == 14) {
        fast_sorf_transform_launch<256, 14, input_t, kComputeRFF, kReverse>(params, stream);
    } else if (params.log_N == 15) {
        fast_sorf_transform_launch<256, 15, input_t, kComputeRFF, kReverse>(params, stream);
    }
}

// Forward: diag then FWHT
template<typename input_t>
void fast_sorf_transform_cuda(SORFParamsBase &params, cudaStream_t stream) {
    fast_sorf_transform_cuda_impl<input_t, false, false>(params, stream);
}

template<typename input_t>
void fast_sorf_rff_transform_cuda(SORFParamsBase &params, cudaStream_t stream) {
    fast_sorf_transform_cuda_impl<input_t, true, false>(params, stream);
}

// Backward: FWHT then diag (caller must reverse diag order)
template<typename input_t>
void fast_sorf_backward_transform_cuda(SORFParamsBase &params, cudaStream_t stream) {
    fast_sorf_transform_cuda_impl<input_t, false, true>(params, stream);
}

template void fast_sorf_transform_cuda<float>(SORFParamsBase &params, cudaStream_t stream);
template void fast_sorf_transform_cuda<at::Half>(SORFParamsBase &params, cudaStream_t stream);
template void fast_sorf_transform_cuda<at::BFloat16>(SORFParamsBase &params, cudaStream_t stream);

template void fast_sorf_rff_transform_cuda<float>(SORFParamsBase &params, cudaStream_t stream);
template void fast_sorf_rff_transform_cuda<at::Half>(SORFParamsBase &params, cudaStream_t stream);
template void fast_sorf_rff_transform_cuda<at::BFloat16>(SORFParamsBase &params, cudaStream_t stream);

template void fast_sorf_backward_transform_cuda<float>(SORFParamsBase &params, cudaStream_t stream);
template void fast_sorf_backward_transform_cuda<at::Half>(SORFParamsBase &params, cudaStream_t stream);
template void fast_sorf_backward_transform_cuda<at::BFloat16>(SORFParamsBase &params, cudaStream_t stream);

template void fast_hadamard_transform_cuda<float>(HadamardParamsBase &params, cudaStream_t stream);
template void fast_hadamard_transform_cuda<at::Half>(HadamardParamsBase &params, cudaStream_t stream);
template void fast_hadamard_transform_cuda<at::BFloat16>(HadamardParamsBase &params, cudaStream_t stream);

template void fast_hadamard_transform_12N_cuda<float>(HadamardParamsBase &params, cudaStream_t stream);
template void fast_hadamard_transform_12N_cuda<at::Half>(HadamardParamsBase &params, cudaStream_t stream);
template void fast_hadamard_transform_12N_cuda<at::BFloat16>(HadamardParamsBase &params, cudaStream_t stream);

template void fast_hadamard_transform_20N_cuda<float>(HadamardParamsBase &params, cudaStream_t stream);
template void fast_hadamard_transform_20N_cuda<at::Half>(HadamardParamsBase &params, cudaStream_t stream);
template void fast_hadamard_transform_20N_cuda<at::BFloat16>(HadamardParamsBase &params, cudaStream_t stream);

template void fast_hadamard_transform_28N_cuda<float>(HadamardParamsBase &params, cudaStream_t stream);
template void fast_hadamard_transform_28N_cuda<at::Half>(HadamardParamsBase &params, cudaStream_t stream);
template void fast_hadamard_transform_28N_cuda<at::BFloat16>(HadamardParamsBase &params, cudaStream_t stream);

template void fast_hadamard_transform_40N_cuda<float>(HadamardParamsBase &params, cudaStream_t stream);
template void fast_hadamard_transform_40N_cuda<at::Half>(HadamardParamsBase &params, cudaStream_t stream);
template void fast_hadamard_transform_40N_cuda<at::BFloat16>(HadamardParamsBase &params, cudaStream_t stream);