import os
import argparse
import json
import copy
import torch
import numpy as np
import sys
sys.path.append('..')
from pykt.models import evaluate,evaluate_question,load_model, evaluate_testset
from pykt.datasets import init_test_datasets

device = "cpu" if not torch.cuda.is_available() else "cuda"
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:2'

with open("../configs/wandb.json") as fin:
    wandb_config = json.load(fin)

def main(params):
    if params['use_wandb'] ==1:
        import wandb
        os.environ['WANDB_API_KEY'] = wandb_config["api_key"]
        wandb.init(project="wandb_predict")

    save_dir, batch_size, fusion_type = params["save_dir"], params["bz"], params["fusion_type"].split(",")


    for dir_path in os.listdir(save_dir):
        if 'log' in dir_path or 'softmasks' in dir_path:
            continue
        elif params['pretrain_model'] == 0 and 'train_with_softmask_forward' in dir_path and f'{args.finetune_dataset_name}' in dir_path and f'apply_mask-{args.apply_mask}-' in dir_path and f'lr-{args.learning_rate}' in dir_path:
            ckpt_path = dir_path
            break
        elif params['pretrain_model'] and 'train_with_softmask_forward' not in dir_path:
            ckpt_path = dir_path
            break

    save_dir = os.path.join(save_dir, ckpt_path)

    win200 = params["win200"]

    with open(os.path.join(save_dir, "config.json")) as fin:
        config = json.load(fin)
        model_config = copy.deepcopy(config["model_config"])
        for remove_item in ['use_wandb','learning_rate','add_uuid','l2']:
            if remove_item in model_config:
                del model_config[remove_item]    
        trained_params = config["params"]
        model_name, dataset_name, emb_type, fold = trained_params["model_name"], trained_params["dataset_name"], trained_params["emb_type"], trained_params["fold"]
        if model_name in ["saint", "sakt", "cdkt"]:
            train_config = config["train_config"]
            seq_len = train_config["seq_len"]
            model_config["seq_len"] = seq_len   

        if model_name in ["stosakt"]:
            train_args = argparse.ArgumentParser()
            print(f"train_args;{train_args}")


    with open("../configs/data_config.json") as fin:
        curconfig = copy.deepcopy(json.load(fin))
        if model_name in ["lorekt"]:
            dataset_name = params["dataset_name"]
        data_config = curconfig[dataset_name]
        data_config["dataset_name"] = dataset_name
        if model_name in ["dkt_forget", "bakt_time"] or emb_type.find("time") != -1:
            data_config["num_rgap"] = config["data_config"]["num_rgap"]
            data_config["num_sgap"] = config["data_config"]["num_sgap"]
            data_config["num_pcount"] = config["data_config"]["num_pcount"]
        elif model_name in ["lpkt"]:
            print("running  prediction")
            data_config["num_at"] = config["data_config"]["num_at"]
            data_config["num_it"] = config["data_config"]["num_it"] 
        elif model_name in ["lorekt"]:
            data_config["num_q"] = config["data_config"]["num_q"]
            data_config["num_c"] = config["data_config"]["num_c"] 
    
    test_loader, test_window_loader, test_question_loader, test_question_window_loader = init_test_datasets(data_config, model_name, batch_size,fold,win200,params['pretrain_suffix'])

    print(f"Start predicting model: {model_name}, embtype: {emb_type}, save_dir: {save_dir}, dataset_name: {dataset_name}")
    print(f"model_config: {model_config}")
    print(f"data_config: {data_config}")

    if model_name in ["stosakt"]:
        model = load_model(model_name, model_config, data_config, emb_type, save_dir, train_args)
    else:

        for remove_item in ['use_wandb','learning_rate','add_uuid','l2','global_bs','num_gpus','pretrain_path', 'num_epochs', 'batch_size']:
            if remove_item in model_config:
                del model_config[remove_item]
        if params["load_finetune"] == "1":
            model = load_model(model_name, model_config, data_config, emb_type, save_dir, args=args, mode="train", finetune=True)
        else:
            print(f"loading model from {save_dir} to evaluate ...")
            model, opt = load_model(model_name, model_config, data_config, emb_type, save_dir, args=args, mode="test")
    model.to(device)
    testauc, testacc = -1, -1

    

    window_testauc, window_testacc = -1, -1
    if dataset_name in ["statics2011", "assist2015", "poj"] or model_name in ["lpkt", "gnn4kt"]:
        save_test_window_path = os.path.join(save_dir, model.emb_type+"_test_window_predictions.txt")
        window_testauc, window_testacc = evaluate(model, test_window_loader, model_name, save_test_window_path, dataset_name, fold)
        print(f"window_testauc: {window_testauc}, window_testacc: {window_testacc}")
    
    if model_name in ["lorekt"]:
        save_test_window_path = ""

        window_testauc, window_testacc = evaluate_testset(model, test_window_loader, model_name, save_test_window_path, dataset_name, fold, soft_mask_path=None)
        print(f"window_testauc: {window_testauc}, window_testacc: {window_testacc}")

    question_testauc, question_testacc = -1, -1
    question_window_testauc, question_window_testacc = -1, -1
  
    dres = {
        "testauc": testauc, "testacc": testacc, "window_testauc": window_testauc, "window_testacc": window_testacc,
    }  

    q_testaucs, q_testaccs = -1,-1
    qw_testaucs, qw_testaccs = -1,-1


    if dataset_name in ["nips_task34", "assist2009", "algebra2005","bridge2algebra2006", "peiyou", "ednet", "ednet5w"]:     
        if "test_question_window_file" in data_config and not test_question_window_loader is None:
            save_test_question_window_path = os.path.join(save_dir, model.emb_type+"_test_question_window_predictions.txt")
            qw_testaucs, qw_testaccs = evaluate_question(model, test_question_window_loader, model_name, fusion_type, save_test_question_window_path, dataset_name, fold)
            for key in qw_testaucs:
                dres["windowauc"+key] = qw_testaucs[key]
            for key in qw_testaccs:
                dres["windowacc"+key] = qw_testaccs[key]

        
    print(f"testauc: {testauc}, testacc: {testacc}, window_testauc: {window_testauc}, window_testacc: {window_testacc}")
    print(f"question_testauc: {question_testauc}, question_testacc: {question_testacc}, question_window_testauc: {question_window_testauc}, question_window_testacc: {question_window_testacc}")
    
    print(dres)

    json.dump(dres, open(os.path.join(save_dir, f'{dataset_name}_pretrain_model_{params["pretrain_model"]}_test_result.json'), 'w+'), indent=4)

    raw_config = json.load(open(os.path.join(save_dir,"config.json")))
    dres.update(raw_config['params'])
    dres["dataset_name"] = dataset_name

    if params['use_wandb'] ==1:
        wandb.log(dres)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bz", type=int, default=256)
    parser.add_argument("--save_dir", type=str, default="saved_model")
    parser.add_argument("--fusion_type", type=str, default="late_fusion")
    parser.add_argument("--use_wandb", type=int, default=1)
    parser.add_argument("--dataset_name", type=str, default="test dataset name")
    parser.add_argument("--pretrain_suffix", type=str, default="pretrain")
    parser.add_argument("--win200", type=bool, default=True)
    parser.add_argument("--load_finetune", type=str, default="0")

    parser.add_argument("--local_rank", type=int, default=0) 


    parser.add_argument("--pretrain_model", type=int, default=0)

    parser.add_argument("--finetune_dataset_name", type=str, default='None')

    parser.add_argument("--apply_mask", type=str, default='None')
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    
    parser.add_argument("--fold", type=str, default='0')



    args = parser.parse_args()
    print(args)
    params = vars(args)
    
    main(params)
