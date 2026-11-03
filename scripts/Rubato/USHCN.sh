#!/usr/bin/env bash
# Rubato on USHCN (seq_len=24, pred_len=6, 5 variables)
# Best config: sigma_init=1.0, grad=1.0, np=50
# seed=2022 5-iter: MAE=0.3062±0.0033 MSE=0.5133±0.0044
set -e

use_multi_gpu=0
if [ $use_multi_gpu -eq 0 ]; then
    launch_command="python"
else
    launch_command="accelerate launch"
fi

. "$(dirname "$(readlink -f "$0")")/../globals.sh"

dataset_name=$(basename "$0" .sh)
dataset_subset_name=""
dataset_id=$dataset_name
get_dataset_info "$dataset_name" "$dataset_subset_name"

model_name="$(basename "$(dirname "$(readlink -f "$0")")")"
model_id=$model_name

$launch_command main.py \
    --is_training 1 \
    --collate_fn "collate_fn" \
    --loss "MSE" \
    --allow_tf32 0 \
    --use_multi_gpu $use_multi_gpu \
    --dataset_root_path $dataset_root_path \
    --model_id $model_id \
    --model_name $model_name \
    --dataset_name $dataset_name \
    --dataset_id $dataset_id \
    --features M \
    --seq_len 24 \
    --pred_len 6 \
    --enc_in $n_variables \
    --dec_in $n_variables \
    --c_out $n_variables \
        --split_strategy sample \
        --single_chunk_on_sample 0 \
        --normalize_method zscore \
    --train_epochs 20 \
    --patience 3 \
    --val_interval 1 \
    --itr 5 \
    --seed 2022 \
    --learning_rate 1e-3 \
    --optimizer AdamW \
    --max_grad_norm 1.0 \
    --d_model 128 \
    --d_ff 512 \
    --n_heads 8 \
    --e_layers 2 \
    --batch_size 16 \
    --dropout 0.1 \
    --weight_decay 5e-5 \
    --rubato_M 2 \
    --rubato_num_patches 50 \
    --rubato_sigma_init 1.0
