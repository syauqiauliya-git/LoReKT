import torch.distributed as dist
import torch
from pykt.config import que_type_models
from .train_model import model_forward
import numpy as np
import os
from tqdm import tqdm






def initial_impt(config):

    # encoder_ffn_dim, embed_dim
    # decoder_ffn_dim
    n_encoder_layer, n_encoder_heads = config['n_blocks'], config['num_attn_heads']


    intermediate_impt = torch.zeros(n_encoder_layer, config['d_ff']).cuda()
    intermediate_mask = torch.ones(n_encoder_layer, config['d_ff']).cuda()

    intermediate_mask.requires_grad_(requires_grad=True)

    output_impt = torch.zeros(n_encoder_layer, config['d_model']).cuda()
    output_mask = torch.ones(n_encoder_layer, config['d_model']).cuda()
    output_mask.requires_grad_(requires_grad=True)

    head_impt = torch.zeros(n_encoder_layer, n_encoder_heads).cuda()
    head_mask = torch.ones(n_encoder_layer, n_encoder_heads).cuda()
    head_mask.requires_grad_(requires_grad=True)

    tot_tokens = 0.0

    return  head_impt, intermediate_impt, output_impt,head_mask, intermediate_mask, output_mask, tot_tokens

    

def get_pretrain_overall_mask(args, soft_mask=None, device=None, finetune_dataset_name=None):

    head_impt_list = []
    intermediate_impt_list = []
    output_impt_list = []

    if not device:
        device = torch.device('cuda')
    all_pretrain_softmask_dirs = []

    for ckpt_dir in os.listdir(args.pretrain_ckpt_path):
        if '_softmasks' in ckpt_dir:
            if finetune_dataset_name in ckpt_dir:
                print(f'excluding finetune dataset:{finetune_dataset_name} softmask into overall softmask !')
                continue
            all_pretrain_softmask_dirs.append(ckpt_dir)
    
    for cur_temp_dataset_softmask_dir in all_pretrain_softmask_dirs:
        cur_dataset_softmask_dir = os.path.join(args.pretrain_ckpt_path, cur_temp_dataset_softmask_dir)
        print(f'aggregaing soft-mask from {cur_dataset_softmask_dir} ...')

        # head
        cur_head_imp_path = os.path.join(cur_dataset_softmask_dir, "head_impt.npy")
        assert cur_head_imp_path, 'head imp is not exist ! please check !'

        cur_head_impt = torch.Tensor(np.load(cur_head_imp_path)).to(device)

        cur_head_impt = impt_norm(cur_head_impt)
        head_impt_list.append(cur_head_impt)
        
        # intermediate
        cur_intermediate_imp_path = os.path.join(cur_dataset_softmask_dir, "intermediate_impt.npy")
        assert cur_intermediate_imp_path, 'intermeidate imp is not exist ! please check !'
        cur_intermediate_impt = torch.Tensor(np.load(cur_intermediate_imp_path)).to(device)
        cur_intermediate_impt = impt_norm(cur_intermediate_impt)
        intermediate_impt_list.append(cur_intermediate_impt)

        cur_output_imp_path = os.path.join(cur_dataset_softmask_dir, "output_impt.npy")
        assert cur_output_imp_path, 'output imp is not exist ! please check !'
        cur_output_impt = torch.Tensor(np.load(cur_output_imp_path)).to(device)
        cur_output_impt = impt_norm(cur_output_impt)
        output_impt_list.append(cur_output_impt)

    if soft_mask:

        print('appending current finetune dataset softmask into overall softmask ...')
        intermediate_impt_list.append(soft_mask['input_projection'])
        output_impt_list.append(soft_mask['output_projection'])
        head_impt_list.append(soft_mask['attention'])
    else:
        print('not appending current finetune dataset softmask into overall softmask !')
    

    if len(head_impt_list) > 0:
        head_impts = torch.stack(head_impt_list)
        head_impt, _ = head_impts.max(0)

        intermediate_impts = torch.stack(intermediate_impt_list)
        intermediate_impt, _ = intermediate_impts.max(0)  # take a max, so that block all impt nurons for all previous tasks;

        output_impts = torch.stack(output_impt_list)
        output_impt, _ = output_impts.max(0)  # take a max, so that block all impt nurons for all previous tasks;

        overall_soft_mask = {
            'input_projection':intermediate_impt, 
            'output_projection':output_impt,
            'attention':head_impt
        }


        all_parts = ['input_projection', 'output_projection', 'attention']
        if args.apply_softmask != 'None':
            apply_parts = args.apply_softmask.split(',')
            for a in all_parts:

                if a not in apply_parts:
                    print(f'not using {a} softmask !!')
                    overall_soft_mask[a] = None
                else:
                    print(f'using {a} softmask !!')

        

    else:
        overall_soft_mask = None

    return overall_soft_mask



def load_soft_mask(soft_mask_path=None, device=None, args=None):


    if soft_mask_path:
        print(f'loading soft-mask from {soft_mask_path} ...')
        inp_proj_soft_mask = torch.Tensor(np.load(os.path.join(soft_mask_path, "intermediate_impt.npy"))).to(device)
        out_proj_soft_mask = torch.Tensor(np.load(os.path.join(soft_mask_path, "output_impt.npy"))).to(device)
        attention_soft_mask = torch.Tensor(np.load(os.path.join(soft_mask_path, "head_impt.npy"))).to(device)

        inp_proj_soft_mask = impt_norm(inp_proj_soft_mask)
        out_proj_soft_mask = impt_norm(out_proj_soft_mask)
        attention_soft_mask = impt_norm(attention_soft_mask)


        soft_mask = {
            'input_projection':inp_proj_soft_mask, 
            'output_projection':out_proj_soft_mask,
            'attention':attention_soft_mask
        }

        all_parts = ['input_projection', 'output_projection', 'attention']
        if args.apply_softmask != 'None':
            apply_parts = args.apply_softmask.split(',')
            for a in all_parts:
                if a not in apply_parts:
                    print(f'not using {a} softmask !!')
                    soft_mask[a] = None
                else:
                    print(f'using {a} softmask !!')
    else:
        soft_mask = None
    return soft_mask


def impt_norm(impt):
    tanh = torch.nn.Tanh()

    for layer in range(impt.size(0)):
        impt[layer] = (impt[layer] - impt[layer].mean()) / impt[
            layer].std()  # 2D, we need to deal with this for each layer

    impt = tanh(impt).abs()

    return impt


def gather_by_mean(head_impt):
    head_impt_list = [torch.zeros_like(head_impt) for _ in range(dist.get_world_size())]
    dist.all_gather(tensor_list=head_impt_list,
                    tensor=head_impt.contiguous())  # everyone need to do this
    head_impt_list = torch.stack(head_impt_list)
    head_impt = torch.mean(head_impt_list, dim=0)
    return head_impt


def compute_soft_mask(model, train_subset_loader, gradient_accumulation_steps, model_config, dataset_name, args):
    head_impt, intermediate_impt, output_impt, \
    head_mask, intermediate_mask, output_mask, tot_tokens = initial_impt(model_config)


    soft_mask = {
        'input_projection':intermediate_mask, 
        'output_projection':output_mask,
        'attention':head_mask
    }

    print(f'computing {dataset_name} soft-mask ...')
    num_epochs = 1
    for i in range(1, num_epochs + 1):
        train_subset_loader.sampler.set_epoch(i)
        for j, data in enumerate(tqdm(train_subset_loader, desc=f'Computing {dataset_name} soft_mask ...', total=len(train_subset_loader), disable=args.local_rank)):
            if model.module.model_name in que_type_models and model.module.model_name not in ["gnn4kt"]:
                model.module.train()
            else:
                model.module.train()
            if model.module.model_name.find("bakt") != -1:
                if j == 0 or model.module.emb_type.find("grad") == -1 and model.module.emb_type != "qid":attn_grads=None
                
                loss = model_forward(model, data, attn_grads)
            else:

                loss = model_forward(model, data, attn_grads=i, soft_mask=soft_mask)
            
            loss = loss / gradient_accumulation_steps
            
            loss.backward() #compute gradients 

            head_impt += head_mask.grad.detach()
            intermediate_impt += intermediate_mask.grad.detach()
            output_impt += output_mask.grad.detach()

            tot_tokens += data["smasks"].float().detach().sum().data
    
    # Normalize
    head_impt /= tot_tokens

    intermediate_impt /= tot_tokens
    output_impt /= tot_tokens

    head_impt = gather_by_mean(head_impt)
    intermediate_impt = gather_by_mean(intermediate_impt)
    output_impt = gather_by_mean(output_impt)


    if args.local_rank <= 0:

        print(f'saving soft_mask for {dataset_name} ...')
        soft_mask_save_dir = os.path.join(args.save_dir, f"{dataset_name}_softmasks")

        if not os.path.isdir(soft_mask_save_dir):
            os.makedirs(soft_mask_save_dir)

        np.save(os.path.join(soft_mask_save_dir, "head_impt.npy"), head_impt.detach().cpu().numpy())
        np.save(os.path.join(soft_mask_save_dir, "intermediate_impt.npy"),intermediate_impt.detach().cpu().numpy())
        np.save(os.path.join(soft_mask_save_dir, "output_impt.npy"), output_impt.detach().cpu().numpy())

    return head_impt, intermediate_impt, output_impt
            
