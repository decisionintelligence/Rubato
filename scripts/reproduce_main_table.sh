#!/usr/bin/env bash
# Reproduce all rows of the main forecasting table.
# Loads each pinned checkpoint and runs the deterministic test phase.

set -e
cd "$(dirname "$(readlink -f "$0")")/.."

run() {
    local ds="$1"; shift
    python main.py \
        --is_training 0 --collate_fn collate_fn --loss MSE \
        --use_multi_gpu 0 \
        --dataset_root_path "storage/datasets/$ds" \
        --dataset_name "$ds" --dataset_id "$ds" --features M \
        --model_name Rubato \
        --seed 2022 --itr 5 --normalize_method zscore --rubato_M 2 \
        "$@"
}

run MIMIC \
    --model_id MI_v3_s1_v2 --seq_len 24 --pred_len 6 \
    --enc_in 96 --dec_in 96 --c_out 96 \
    --d_model 128 --d_ff 512 --n_heads 8 --e_layers 2 \
    --batch_size 16 --dropout 0.1 --rubato_num_patches 200 \
    --rubato_sigma_init 1.0 --split_strategy sample --single_chunk_on_sample 1

run P12 \
    --model_id Rubato --seq_len 24 --pred_len 6 \
    --enc_in 41 --dec_in 41 --c_out 41 \
    --d_model 256 --d_ff 1024 --n_heads 8 --e_layers 2 \
    --batch_size 16 --dropout 0.1 --rubato_num_patches 100 \
    --rubato_sigma_init 0.5 --split_strategy sample --single_chunk_on_sample 1

run USHCN \
    --model_id USHCN_s1 --seq_len 24 --pred_len 6 \
    --enc_in 5 --dec_in 5 --c_out 5 \
    --d_model 128 --d_ff 512 --n_heads 8 --e_layers 2 \
    --batch_size 16 --dropout 0.1 --rubato_num_patches 50 \
    --rubato_sigma_init 1.0 --split_strategy sample --single_chunk_on_sample 0

run FNSPID \
    --model_id FNSPID_s2 --seq_len 730 --pred_len 30 \
    --enc_in 6 --dec_in 6 --c_out 6 \
    --d_model 128 --d_ff 512 --n_heads 4 --e_layers 2 \
    --batch_size 32 --dropout 0.1 --rubato_num_patches 100 \
    --rubato_sigma_init 2.0 --split_strategy time --single_chunk_on_sample 0

run EPA_AIR \
    --model_id Rubato --seq_len 336 --pred_len 168 \
    --enc_in 4 --dec_in 4 --c_out 4 \
    --d_model 128 --d_ff 512 --n_heads 1 --e_layers 2 \
    --batch_size 32 --dropout 0.1 --rubato_num_patches 112 \
    --rubato_sigma_init 0.5 --split_strategy time --single_chunk_on_sample 0

run ClusterTrace \
    --model_id Rubato --seq_len 168 --pred_len 24 \
    --enc_in 3 --dec_in 3 --c_out 3 \
    --d_model 256 --d_ff 512 --n_heads 4 --e_layers 2 \
    --batch_size 32 --dropout 0.05 --rubato_num_patches 100 \
    --rubato_sigma_init 0.5 --split_strategy time --single_chunk_on_sample 0
