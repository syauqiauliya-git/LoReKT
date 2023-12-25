import argparse
import os
#TODO: 改成 v3
from wandb_train import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--dataset_name", type=str, default="pretrain")
    parser.add_argument("--model_name", type=str, default="gpt4kt")
    parser.add_argument("--emb_type", type=str, default="qid")
    parser.add_argument("--save_dir", type=str, default="saved_model")
    # parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--final_fc_dim", type=int, default=256)
    parser.add_argument("--final_fc_dim2", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--nheads", type=int, default=8)
    parser.add_argument("--loss1", type=float, default=0.5)
    parser.add_argument("--loss2", type=float, default=0.5)
    parser.add_argument("--loss3", type=float, default=0.5)
    parser.add_argument("--start", type=int, default=50)
    
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--d_ff", type=int, default=256)
    parser.add_argument("--num_attn_heads", type=int, default=8)
    parser.add_argument("--n_blocks", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--optimizer", type=str, default=None)
    parser.add_argument("--save_opt", type=int, default=0, help='.')

    # multi-task
    parser.add_argument("--cf_weight", type=float, default=0.1)
    parser.add_argument("--t_weight", type=float, default=0.1)

    parser.add_argument("--seq_len", type=int, default=200)

    parser.add_argument("--use_wandb", type=int, default=1)
    parser.add_argument("--add_uuid", type=int, default=1)
    
    parser.add_argument("--local_rank", type=int, default=os.getenv('LOCAL_RANK', -1))
    parser.add_argument("--num_gpus", type=int, default=8)
    parser.add_argument("--global_bs", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=64, help='per device batch')

    # pretrain
    parser.add_argument("--dataset_special_token_num", type=int, default=1, help='special token num for dataset')
    parser.add_argument("--q_special_token_num", type=int, default=1, help='specitial token num for question')
    parser.add_argument("--c_special_token_num", type=int, default=1, help='specitial token num for concept')
    parser.add_argument("--use_qc_emb", type=int, default=1, help='use question concept embedding or not when encoding')
    parser.add_argument("--add_dataset_embed", type=int, default=1, help='adding dataset embedding to qa_embed')
    parser.add_argument("--concat_dataset_embed", type=int, default=1, help='concat dataset embedding when prediction')
    parser.add_argument("--use_qc_placeholder_embed", type=int, default=1, help='use question concept placeholder embed')
    parser.add_argument("--exclude_dataset", type=str, default='', help='the dataset you want to exclude')

    # compute soft_mask 
    parser.add_argument("--compute_soft_mask", type=int, default=0, help='compute soft_mask or not')
    parser.add_argument("--ckpt_path", type=str, default='', help='the ckpt for computing soft mask')

    # train with soft_mask 
    parser.add_argument("--train_with_softmask_forward", type=int, default=0, help='.')
    parser.add_argument("--apply_softmask", type=str, default='input_projection,output_projection,attention', help='.')

    # finetune
    parser.add_argument("--pretrain_ckpt_path", type=str, default='None', help='.')
    parser.add_argument("--finetune_dataset_name", type=str, default='None', help='.')
    

    


    args = parser.parse_args()

    params = vars(args)
    main(params,args)
