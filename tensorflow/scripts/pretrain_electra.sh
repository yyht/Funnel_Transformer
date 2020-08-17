#!/bin/bash

TPU_NAME=albert1

# TPU v2-8 (base models) or v3-8 (large models)
NUM_HOSTS=1
NUM_CORE_PER_HOST=8

GS_ROOT=gs://yyht_source/pretrain
GS_INIT_CKPT_DIR=${GS_ROOT}/model/B4-4-4H768-ELEC-FULL-TF

GS_PROC_DATA_DIR=${GS_ROOT}/proc_data/squad
GS_MODEL_DIR=${GS_ROOT}/proc_data/squad/my

uncased=True
tokenizer_type=word_piece
tokenizer_path=${GS_INIT_CKPT_DIR}/vocab.uncased.txt
init_checkpoint=${GS_INIT_CKPT_DIR}/model.ckpt
model_config=${GS_INIT_CKPT_DIR}/net_config.json

data_dir=/home/htxu91/squad

nohup python pretrain.py \
    --use_tpu=True \
    --tpu=${TPU_NAME} \
    --use_bfloat16=True \
    --num_hosts=${NUM_HOSTS} \
    --num_core_per_host=${NUM_CORE_PER_HOST} \
    --output_dir=${GS_PROC_DATA_DIR} \
    --model_dir=${GS_MODEL_DIR} \
    --predict_dir=${GS_MODEL_DIR}/prediction \
    --predict_file=${data_dir}/dev-v2.0.json \
    --init_checkpoint=${init_checkpoint} \
    --model_config=${model_config} \
    --uncased=${uncased} \
    --tokenizer_type=${tokenizer_type} \
    --tokenizer_path=${tokenizer_path} \
    --learning_rate=1e-4 \
    --train_steps=1000000 \
    --warmup_steps=10000 \
    --iterations=1000 \
    --save_steps=1000 \
    --train_batch_size=384 \
    --eval_batch_size=8 \
    --do_train=True
    $@
