#!/bin/bash


TPU_NAME=albert0

# TPU v2-8 (base models) or v3-8 (large models)
NUM_HOSTS=1
NUM_CORE_PER_HOST=8

GS_ROOT=gs://yyht_source/pretrain
GS_INIT_CKPT_DIR=${GS_ROOT}/model/english/tiny/bert_small_span_mask_real_attn_net_config_tiny_enc_dec_no_skip_normal_ce

task=sts-b
root_data_dir=${GS_ROOT}/glue
root_model_dir=${GS_ROOT}/proc_data/glue/enc_dec_normal_ce

# task=sts-b
# data_dir=${GS_ROOT}/data/glue/STS-B

# task=mrpc
# data_dir=${GS_ROOT}/data/glue/MRPC

# task=mnli_matched
# data_dir=${GS_ROOT}/data/glue/MNLI

# GS_PROC_DATA_DIR=${GS_ROOT}/proc_data/glue/mlm/${task}
# GS_MODEL_DIR=${GS_ROOT}/proc_data/glue/mlm/${task}

uncased=True
tokenizer_type=word_piece
tokenizer_path=${GS_INIT_CKPT_DIR}/vocab.uncased.txt
init_checkpoint=${GS_INIT_CKPT_DIR}/model.ckpt-1100800
model_config=${GS_INIT_CKPT_DIR}/net_config_small_chinese_enc_dec_no_skip_uncased_english.json

bsz=16
NUM_HOSTS=1
NUM_CORE_PER_HOST=1


nohup python classifier_all_datasets.py \
    --use_tpu=True \
    --tpu=${TPU_NAME} \
    --root_data_dir=${root_data_dir} \
    --uncased=${uncased} \
    --num_hosts=${NUM_HOSTS} \
    --num_core_per_host=${NUM_CORE_PER_HOST} \
    --tokenizer_type=${tokenizer_type} \
    --tokenizer_path=${tokenizer_path} \
    --root_model_dir=${root_model_dir} \
    --init_checkpoint=${init_checkpoint} \
    --model_config=${model_config} \
    --learning_rate=0.0001 \
    --warmup_steps=155 \
    --train_steps=1550 \
    --train_batch_size=${bsz} \
    --lr_layer_decay_rate=0.9 \
    --iterations=155 \
    --save_steps=155 \
    --do_train=True \
    --do_eval=True \
    --do_submit=True \
    --task_name=${task} \
    --use_wd_exclusion=True \
    --adam_correction=False \
    --weight_decay=0.01 \
    --adam_correction=False \
    --verbose=True
    $@

