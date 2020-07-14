#!/bin/bash


TPU_NAME=albert1

# TPU v2-8 (base models) or v3-8 (large models)
NUM_HOSTS=1
NUM_CORE_PER_HOST=8


GS_ROOT=gs://yyht_source/pretrain
GS_INIT_CKPT_DIR=${GS_ROOT}/model/B4-4-4H768-ELEC-FULL-TF

task=cola
data_dir=${GS_ROOT}/glue/CoLA

# task=sts-b
# data_dir=${GS_ROOT}/data/glue/STS-B

# task=mrpc
# data_dir=${GS_ROOT}/data/glue/MRPC

# task=mnli_matched
# data_dir=${GS_ROOT}/data/glue/MNLI

GS_PROC_DATA_DIR=${GS_ROOT}/proc_data/glue/${task}
GS_MODEL_DIR=${GS_ROOT}/proc_data/glue/${task}

uncased=True
tokenizer_type=word_piece
tokenizer_path=${GS_INIT_CKPT_DIR}/vocab.uncased.txt
init_checkpoint=${GS_INIT_CKPT_DIR}/model.ckpt
model_config=${GS_INIT_CKPT_DIR}/net_config.json

bsz=64
NUM_HOSTS=1
NUM_CORE_PER_HOST=1


nohup python classifier.py \
    --use_tpu=True \
    --tpu=${TPU_NAME} \
    --data_dir=${data_dir} \
    --output_dir=${GS_PROC_DATA_DIR} \
    --uncased=${uncased} \
    --num_hosts=${NUM_HOSTS} \
    --num_core_per_host=${NUM_CORE_PER_HOST} \
    --tokenizer_type=${tokenizer_type} \
    --tokenizer_path=${tokenizer_path} \
    --model_dir=${GS_MODEL_DIR} \
    --init_checkpoint=${init_checkpoint} \
    --model_config=${model_config} \
    --learning_rate=0.00002 \
    --warmup_steps=534 \
    --train_steps=5344 \
    --train_batch_size=${bsz} \
    --iterations=534 \
    --save_steps=534 \
    --do_train=True \
    --do_eval=True \
    --do_submit=True \
    --task_name=${task} \
    --use_wd_exclusion=True \
    --adam_correction=False \
    --weight_decay=0.01 \
    --verbose=True \
    --max_seq_length=64
    $@

