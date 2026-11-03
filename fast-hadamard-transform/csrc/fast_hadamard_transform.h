/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

////////////////////////////////////////////////////////////////////////////////////////////////////

struct HadamardParamsBase {
    using index_t = int64_t;

    int batch, dim, log_N;

    index_t x_batch_stride;
    index_t out_batch_stride;

    float scale;

    // Common data pointers.
    void *__restrict__ x_ptr;
    void *__restrict__ out_ptr;
};

struct SORFParamsBase : HadamardParamsBase {
    void *__restrict__ diag_ptr;   // (num_groups, n_rounds, dim) diagonal matrices
    index_t diag_stride;           // stride between rounds in elements
    int n_rounds;                  // number of diag*FWHT rounds (typically 3)
    float input_scale;             // multiply input by this (e.g. 1/sigma)
    float rff_scale;               // output [cos,sin]*rff_scale; 0 = disabled
    int rff_dim;                   // D: number of random features (before cos/sin doubling)
    // Multi-group fields. For backward compatibility, num_groups=1 disables grouping.
    int num_groups;                // G: number of independent SORF blocks (default 1)
    index_t group_diag_stride;     // stride between groups in diag_ptr (elements)
    index_t group_out_stride;      // stride between groups in out_ptr (elements)
};
