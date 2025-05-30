import os
import argparse
import json
from pickletools import optimize
from tqdm import tqdm
import torch
torch.set_num_threads(4) 
from torch.optim import SGD, Adam
import copy
import sys
sys.path.append('..')

from pykt.models import train_model,evaluate,init_model,load_model
from pykt.utils import debug_print,set_seed
from pykt.datasets import init_dataset4train
import datetime
import subprocess
from pykt.models import compute_soft_mask
from pykt.models import load_soft_mask, get_pretrain_overall_mask


os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# device = "cpu" if not torch.cuda.is_available() else "cuda"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:2'
# local_rank = int(os.environ['LOCAL_RANK'])
# torch.cuda.set_device(local_rank)

def save_config(train_config, model_config, data_config, params, save_dir, args=None):
    # print(f"type_args:{type(args)}")
    if args:
        d = {"train_config": train_config, 'model_config': model_config, "data_config": data_config, "params": params, "train_args":vars(args)}
    else:
        d = {"train_config": train_config, 'model_config': model_config, "data_config": data_config, "params": params}
    save_path = os.path.join(save_dir, "config.json")
    with open(save_path, "w") as fout:
        json.dump(d, fout)

def get_pretrain_info(args):
    flag = 0
    for ckpt_dir in os.listdir(args.ckpt_path):
        if 'log' not in ckpt_dir and 'softmasks' not in ckpt_dir and 'train_with_softmask_forward' not in ckpt_dir:
            flag = 1
            break

    if not flag:
        import sys
        print('cannot find the right ckpt path ...')
        sys.exit()
    # import pdb; pdb.set_trace()
    ckpt_dir = os.path.join(args.ckpt_path, ckpt_dir)
    print(f'loading model state dict from {ckpt_dir} ...')
    with open(os.path.join(ckpt_dir, "config.json")) as fin:
        ckpt_config = json.load(fin)
        model_config = ckpt_config['model_config']
    return ckpt_dir, model_config


def save_all_config(save_dir, params_str, args, params, train_config, model_config, data_config):
    if not args.compute_soft_mask:

        
        if params['train_with_softmask_forward']:
            print('creating train with softmaks forward ckpt dir ...')
            params_str = 'train_with_softmask_forward-' + f'finetune-dataset_{args.finetune_dataset_name}-' + f'epoch-{args.num_epochs}-'
            params_str += f'apply_mask-{args.apply_softmask}-lr-{args.learning_rate}'





       

        ckpt_path = os.path.join(save_dir, params_str)

        if not os.path.isdir(ckpt_path):
            if args.local_rank == 0:
                os.makedirs(ckpt_path)

        if params['model_name'] in ["stosakt"]:
            if args.local_rank == 0:
                if args.finetune_dataset_name != "None":
                    pretrain_ckpt_path, pretrain_model_config = get_pretrain_info(args)
                    model_config = pretrain_model_config

                save_config(train_config, model_config, data_config[params['dataset_name']], params, ckpt_path, args)
        else:
            if args.local_rank == 0:
                if args.finetune_dataset_name != "None":
                    pretrain_ckpt_path, pretrain_model_config = get_pretrain_info(args)
                    model_config = pretrain_model_config
                save_config(train_config, model_config, data_config[params['dataset_name']], params, ckpt_path)
        
        
        
        print(f"Start training model: {params['model_name']}, embtype: {params['emb_type']}, save_dir: {ckpt_path}, dataset_name: {params['dataset_name']}")
        print(f"model_config: {model_config}")
        print(f"train_config: {train_config}")
    else:
        ckpt_path, model_config = None, model_config

    return ckpt_path, model_config
    

def main(params, args=None):
    args.local_rank = int(os.environ["LOCAL_RANK"])

    if "use_wandb" not in params:
        params['use_wandb'] = 1

    if params['use_wandb']==1:
        import wandb
        wandb.init()
    if args.local_rank != -1:
        torch.distributed.init_process_group(backend='nccl')
        torch.cuda.set_device(args.local_rank)

    set_seed(params["seed"])

    model_name, dataset_name, fold, emb_type, save_dir = params["model_name"], params["dataset_name"], \
        params["fold"], params["emb_type"], params["save_dir"]
        
    debug_print(text = "load config files.",fuc_name="main")
    
    with open("../configs/kt_config.json") as f:
        config = json.load(f)
        train_config = config["train_config"]
        if model_name in ["lorekt"]:
            seqlen = params['seq_len']
            train_config["seq_len"] = seqlen
            if seqlen == 1024:
                if params["d_model"] <= 1024:
                    train_config["batch_size"] = 16 ## because of OOM
                else:
                    train_config["batch_size"] = 16 ## because of OOM
            elif seqlen == 200:
                train_config["batch_size"] = 32 ## because of OOM
            else: # seqlen = 512
                train_config["batch_size"] = 32 ## because of OOM
        if model_name in ["dkvmn","deep_irt", "sakt", "saint","saint++", "akt", "atkt", "lpkt", "skvmn", "gnn4kt"]:
            train_config["batch_size"] = 64 ## because of OOM
        if model_name in ["bakt", "bakt_time", "bakt_qikt","simplekt_sr", "stosakt", "parkt", "mikt"]:
            train_config["batch_size"] = 64 ## because of OOM
        if model_name in ["gkt"]:
            train_config["batch_size"] = 16 
        if model_name in ["qdkt","qikt"] and dataset_name in ['algebra2005','bridge2algebra2006']:
            train_config["batch_size"] = 32 
        model_config = copy.deepcopy(params)
        for key in ["model_name", "dataset_name", "emb_type", "save_dir", "fold", "seed"]:
            del model_config[key]
        if 'batch_size' in params:
            train_config["batch_size"] = params['batch_size']
        if 'num_epochs' in params:
            train_config["num_epochs"] = params['num_epochs']
        # model_config = {"d_model": params["d_model"], "n_blocks": params["n_blocks"], "dropout": params["dropout"], "d_ff": params["d_ff"]}
    batch_size, num_epochs = train_config["batch_size"], train_config["num_epochs"]
    if params['optimizer']:
        train_config['optimizer'] = params['optimizer']
  

    optimizer = train_config["optimizer"]
    opt = None

    


    with open("../configs/data_config.json") as fin:
        data_config = json.load(fin)
    # if 'maxlen' in data_config[dataset_name]:#prefer to use the maxlen in data config
    #     train_config["seq_len"] = data_config[dataset_name]['maxlen']
    seq_len = train_config["seq_len"]

    print("Start init data")
    print(dataset_name, model_name, data_config, fold, batch_size)
    
    debug_print(text="init_dataset",fuc_name="main")
    if model_name not in ["simplekt_sr", "parkt", "lorekt"]:
        train_loader, valid_loader, curtrain = init_dataset4train(dataset_name, model_name, emb_type, data_config, fold, batch_size,args)
        # print(f"curtrain:{len(curtrain)}") 
    elif model_name in ["simplekt_sr", "lorekt"]:
        if args.compute_soft_mask:
            print('getting dataloaders for computing soft-mask ...')
            all_train_subset_loaders = init_dataset4train(dataset_name, model_name, emb_type, data_config, fold, batch_size, args)
        else:

            train_loader, valid_loader, curtrain = init_dataset4train(dataset_name, model_name, emb_type, data_config, fold, batch_size, args)

    elif model_name in ["parkt"]:
        if emb_type.find("cl") != -1 or emb_type.find("uid") != -1:
            train_loader, valid_loader, curtrain = init_dataset4train(dataset_name, model_name, emb_type, data_config, fold, batch_size, args)
        else:
            train_loader, valid_loader, curtrain = init_dataset4train(dataset_name, model_name, emb_type, data_config, fold, batch_size)


    params_str = "_".join([str(v) for k,v in params.items() if not k in ['save_dir', 'ckpt_path', 'other_config', 'pretrain_ckpt_path']])

    print(f"params: {params}, params_str: {params_str}")
    
    if params['add_uuid'] == 1 and params["use_wandb"] == 1:
        import uuid
        # if not model_name in ['saint','saint++']:
        params_str = params_str+f"_{ str(uuid.uuid4())}"
   
  
    
    ckpt_path, model_config = save_all_config(save_dir, params_str, args, params, train_config, model_config, data_config)

   
        
    learning_rate = params["learning_rate"]
    for remove_item in ['use_wandb','learning_rate','add_uuid','l2','global_bs','num_gpus', 'num_epochs', 'batch_size']:
        if remove_item in model_config:
            del model_config[remove_item]
    if model_name in ["saint","saint++", "sakt", "cdkt", "bakt", "bakt_time"]:
        model_config["seq_len"] = seq_len
        
    
    debug_print(text = "init_model",fuc_name="main")
    print(f"model_name:{model_name}")
    if model_name in ["parkt"]:
        dpath = os.path.join(data_config[dataset_name]["dpath"], "keyid2idx.json")
        with open(dpath, "r") as f:
            map_json = json.load(f)
            num_stu = len(map_json["uid"])
        model = init_model(model_name, model_config, data_config[dataset_name], emb_type, args, num_stu)
        print(f"model_parameter:{sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())}")
    else:
       
        if args.compute_soft_mask or args.finetune_dataset_name != "None":

            pretrain_ckpt_dir, pretrain_model_config = get_pretrain_info(args)
            model, opt = load_model(model_name=model_name, model_config=pretrain_model_config, data_config=data_config[dataset_name], emb_type=emb_type, ckpt_path=pretrain_ckpt_dir, args=args, mode="train", finetune=True, optimizer=optimizer)
        else:
            model = init_model(model_name, model_config, data_config[dataset_name], emb_type, args)
        print(f"model_parameter:{sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())}")
    print(f"model is {model}")

    
    if not opt:
        if model_name == "hawkes":
            weight_p, bias_p = [], []
            for name, p in filter(lambda x: x[1].requires_grad, model.named_parameters()):
                if 'bias' in name:
                    bias_p.append(p)
                else:
                    weight_p.append(p)
            optdict = [{'params': weight_p}, {'params': bias_p, 'weight_decay': 0}]
            opt = torch.optim.Adam(optdict, lr=learning_rate, weight_decay=params['l2'])
        elif model_name == "iekt":
            opt = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)
        else:
            if optimizer == "sgd":
                opt = SGD(model.parameters(), learning_rate, momentum=0.9)
            elif optimizer == "adam":
                opt = Adam(model.parameters(), learning_rate)
    # import pdb; pdb.set_trace()
    print(f"using {optimizer} optmizer ..")
   
    testauc, testacc = -1, -1
    window_testauc, window_testacc = -1, -1
    validauc, validacc = -1, -1
    best_epoch = -1
    save_model = True
    
    debug_print(text = "train model",fuc_name="main")
    
    if emb_type.find("cl") != -1:
        # print(f"curtrain:{len(curtrain)}")
        testauc, testacc, window_testauc, window_testacc, validauc, validacc, best_epoch = train_model(model, train_loader, valid_loader, num_epochs, opt, ckpt_path, None, None, save_model, dataset_name, fold, curtrain=curtrain, batch_size=batch_size)
    elif model_name in ["lorekt"]:
        global_bs = params['global_bs']
        num_gpus = params['num_gpus']
        gradient_accumulation_steps = max(global_bs/num_gpus/train_config["batch_size"],1.0)
        print(f"gradient_accumulation_steps:{gradient_accumulation_steps}")

        
        if args.compute_soft_mask:
            for dataset_name, cur_subset_data_loader in all_train_subset_loaders.items():

                compute_soft_mask(model, cur_subset_data_loader, gradient_accumulation_steps, pretrain_model_config, dataset_name, args)

            return 
        

        elif args.finetune_dataset_name != "None":
           
            if args.train_with_softmask_forward:
                soft_mask_path = os.path.join(params["save_dir"], f'{args.finetune_dataset_name}_softmasks')
                print(f'loading current dataset softmask from {soft_mask_path} ....')
                softmask_for_forward = load_soft_mask(soft_mask_path=soft_mask_path, device=device, args=args)

            else:
                print('not loading current dataset softmask !!')
                softmask_for_forward = None

            

            testauc, testacc, window_testauc, window_testacc, validauc, validacc, best_epoch = train_model(model, train_loader, valid_loader, num_epochs, opt, ckpt_path, None, None, save_model, dataset_name, fold, gradient_accumulation_steps=gradient_accumulation_steps, softmask_for_forward=softmask_for_forward, model_config=pretrain_model_config, args=args)
        
        else:
            testauc, testacc, window_testauc, window_testacc, validauc, validacc, best_epoch = train_model(model, train_loader, valid_loader, num_epochs, opt, ckpt_path, None, None, save_model, dataset_name, fold, gradient_accumulation_steps=gradient_accumulation_steps, args=args)
    else:
        testauc, testacc, window_testauc, window_testacc, validauc, validacc, best_epoch = train_model(model, train_loader, valid_loader, num_epochs, opt, ckpt_path, None, None, save_model, dataset_name, fold)
    
   

    model_save_path = os.path.join(ckpt_path, emb_type+"_model.module.ckpt")

    print("fold\tmodelname\tembtype\ttestauc\ttestacc\twindow_testauc\twindow_testacc\tvalidauc\tvalidacc\tbest_epoch")
    print(str(fold) + "\t" + model_name + "\t" + emb_type + "\t" + str(round(testauc, 4)) + "\t" + str(round(testacc, 4)) + "\t" + str(round(window_testauc, 4)) + "\t" + str(round(window_testacc, 4)) + "\t" + str(validauc) + "\t" + str(validacc) + "\t" + str(best_epoch))
    model_save_path = os.path.join(ckpt_path, emb_type+"_model.module.ckpt")
    print(f"end:{datetime.datetime.now()}")
    
    if params['use_wandb']==1:
        wandb.log({ 
                    "validauc": validauc, "validacc": validacc, "best_epoch": best_epoch,"model_save_path":model_save_path})
