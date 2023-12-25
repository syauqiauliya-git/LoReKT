export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

MODEL_NAME=lorekt
DATASET_NAME=pretrain 
MODEL_SIZE=221M



pretrain(){
    python -m torch.distributed.launch --nproc_per_node=${N_TASKS} --nnodes=${NODE_NUM} --master_port=${PORT} wandb_lorekt_train.py \
        --num_epochs=${EPOCHS} \
        --learning_rate=${LR} \
        --dataset_name=${DATASET_NAME} \
        --model_name=${MODEL_NAME} \
        --save_dir=${SAVE_DIR} \
        --global_bs=${GLB_BATCH} \
        --batch_size=${PER_DEVICE_BATCH} \
        --final_fc_dim=${FC1} \
        --final_fc_dim2=${FC2} \
        --d_model=${D_MODEL} \
        --d_ff=${D_FF} \
        --num_attn_heads=${HEADS} \
        --n_blocks=${N_BLOCKS} \
        --seq_len=${SEQ_LEN} \
        --num_gpus=${NUM_GPUS} \
        --use_qc_emb=${USE_QC_EMBED} \
        --add_dataset_embed=${ADD_DATASET_EMBED} \
        --concat_dataset_embed=${CONCAT_DATASET_EMBED} \
        --use_qc_placeholder_embed=${USE_QC_PLACEHOLDER_EMBED} \
        --use_wandb=0 \
        --compute_soft_mask=${COMPUTE_SOFT_MASK} \
        --ckpt_path=${CKPT_PATH} \
        --exclude_dataset=${EXCLUDE_DATASET} \
        2>&1 | tee ${SAVE_DIR}/train_compute_mask_${COMPUTE_SOFT_MASK}.log
}

eval(){
    python -u wandb_predict.py \
        --bz ${TEST_BATCH} \
        --save_dir ${SAVE_DIR} \
        --dataset_name ${TEST_DATASET} \
        --pretrain_suffix ${DATASET_NAME} \
        --use_wandb 0
}

########################################### pretrain ##########################################
GLB_BATCH=512 
PER_DEVICE_BATCH=64 
FC1=512
FC2=1024
D_MODEL=512
D_FF=1024
HEADS=16
N_BLOCKS=24
SEQ_LEN=200



USE_QC_EMBED=1
ADD_DATASET_EMBED=0
CONCAT_DATASET_EMBED=1
USE_QC_PLACEHOLDER_EMBED=1


COMPUTE_SOFT_MASK=0
CKPT_PATH=None 

EXCLUDE_DATASET=assist2009,algebra2005,nips_task34 
IMPORTANT_PARAM=${GLB_BATCH}-d_spec_${D_SPEC_TOK_NUM}-q_spec_${Q_SPEC_TOK_NUM}-c_spec_${C_SPEC_TOK_NUM}-use_qc_${USE_QC_EMBED}-add_data_embed_${ADD_DATASET_EMBED}-concat_data_embed${CONCAT_DATASET_EMBED}-use_qc_place_${USE_QC_PLACEHOLDER_EMBED}

SAVE_DIR=saved_model/${MODEL_NAME}-${MODEL_SIZE}-${DATASET_NAME}-exclude_${EXCLUDE_DATASET}-${IMPORTANT_PARAM}

mkdir -p ${SAVE_DIR}

LR=1e-4
EPOCHS=200
N_TASKS=8
NODE_NUM=1

pretrain

########################################### pretrain ##########################################





########################################### compute softmask importance vector ##########################################
COMPUTE_SOFT_MASK=1
EXCLUDE_DATASET=bridge2algebra2006,peiyou,ednet5w
CKPT_PATH=${SAVE_DIR}
pretrain

########################################### compute softmask importance vector ##########################################




########################################### eval ##########################################
TEST_BATCH=1024

declare -a TEST_DATASETS
TEST_DATASETS=('assist2009' 'algebra2005' 'bridge2algebra2006' 'ednet5w' 'peiyou' 'nips_task34')
for TEST_DATASET in "${TEST_DATASETS[@]}";do
    eval | tee ${SAVE_DIR}/pretrain_test_${TEST_DATASET}.log 
done

########################################### eval ##########################################

