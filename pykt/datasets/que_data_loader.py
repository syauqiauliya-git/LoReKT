#!/usr/bin/env python
# coding=utf-8

import os, sys
import pandas as pd
import torch
from torch.utils.data import Dataset
# from torch.cuda import FloatTensor, LongTensor
from torch import FloatTensor, LongTensor
import numpy as np
import joblib


datasets_dic = {"assist2009": 0, "algebra2005": 1, "bridge2algebra2006": 2, "nips_task34": 3, "ednet": 4, "peiyou": 5, "ednet5w": 6}

class KTQueDataset(Dataset):
    """Dataset for KT
        can use to init dataset for: (for models except dkt_forget)
            train data, valid data
            common test data(concept level evaluation), real educational scenario test data(question level evaluation).

    Args:
        file_path (str): train_valid/test file path
        input_type (list[str]): the input type of the dataset, values are in ["questions", "concepts"]
        folds (set(int)): the folds used to generate dataset, -1 for test data
        qtest (bool, optional): is question evaluation or not. Defaults to False.
    """

    def __init__(self, file_path, input_type, folds,concept_num,max_concepts, qtest=False, dataset_name=None, exclude_dataset=None):
        super(KTQueDataset, self).__init__()
        sequence_path = file_path
        self.input_type = input_type
        self.concept_num = concept_num
        self.max_concepts = max_concepts
        if "questions" not in input_type or "concepts" not in input_type:
            raise("The input types must contain both questions and concepts")

        folds = sorted(list(folds))
        folds_str = "_" + "_".join([str(_) for _ in folds])
        if dataset_name:
            processed_data = file_path + folds_str + f'{dataset_name}' + "_qlevel.pkl"
        elif exclude_dataset:
            processed_data = file_path + folds_str + f'exclude_{exclude_dataset}' + "_qlevel.pkl"
        else:
            processed_data = file_path + folds_str + "_qlevel.pkl"
        

        if not os.path.exists(processed_data):
            print(f"Start preprocessing {file_path} fold: {folds_str}...")

            self.dori = self.__load_data__(sequence_path=sequence_path, folds=folds, dataset_name=dataset_name, exclude_dataset=exclude_dataset)
            save_data = self.dori
            pd.to_pickle(save_data, processed_data)
        else:
            print(f"Read data from processed file: {processed_data}")
            try:
                self.dori = pd.read_pickle(processed_data)
            except MemoryError:
                with open(processed_data, 'rb') as f:
                    self.dori = joblib.load(f)
        print(f"file path: {file_path}, qlen: {len(self.dori['qseqs'])}, clen: {len(self.dori['cseqs'])}, rlen: {len(self.dori['rseqs'])}")

    def __len__(self):
        """return the dataset length

        Returns:
            int: the length of the dataset
        """
        return len(self.dori["rseqs"])

    def __getitem__(self, index):
        """
        Args:
            index (int): the index of the data want to get

        Returns:
            (tuple): tuple containing:
            
            - **q_seqs (torch.tensor)**: question id sequence of the 0~seqlen-2 interactions
            - **c_seqs (torch.tensor)**: knowledge concept id sequence of the 0~seqlen-2 interactions
            - **r_seqs (torch.tensor)**: response id sequence of the 0~seqlen-2 interactions
            - **qshft_seqs (torch.tensor)**: question id sequence of the 1~seqlen-1 interactions
            - **cshft_seqs (torch.tensor)**: knowledge concept id sequence of the 1~seqlen-1 interactions
            - **rshft_seqs (torch.tensor)**: response id sequence of the 1~seqlen-1 interactions
            - **mask_seqs (torch.tensor)**: masked value sequence, shape is seqlen-1
            - **select_masks (torch.tensor)**: is select to calculate the performance or not, 0 is not selected, 1 is selected, only available for 1~seqlen-1, shape is seqlen-1
            - **dcur (dict)**: used only self.qtest is True, for question level evaluation
        """
        dcur = dict()
        mseqs = self.dori["masks"][index]
        for key in self.dori:
            if key in ["masks", "smasks","dataset"]:
                continue
            if len(self.dori[key]) == 0:
                dcur[key] = self.dori[key]
                dcur["shft_"+key] = self.dori[key]
                continue
            # print(f"key: {key}, len: {len(self.dori[key])}")
            if key=='cseqs':
                seqs = self.dori[key][index][:-1,:]
                shft_seqs = self.dori[key][index][1:,:]
            else:
                seqs = self.dori[key][index][:-1] * mseqs
                shft_seqs = self.dori[key][index][1:] * mseqs
            dcur[key] = seqs
            dcur["shft_"+key] = shft_seqs
        dcur["masks"] = mseqs
        dcur["smasks"] = self.dori["smasks"][index]
        dcur["dataset_id"] = self.dori["dataset"][index]
        # print("tseqs", dcur["tseqs"])
        return dcur

    def get_skill_multi_hot(self, this_skills):
        skill_emb = [0] * self.concept_num
        for s in this_skills:
            skill_emb[s] = 1
        return skill_emb


    def __load_data__(self, sequence_path, folds, pad_val=-1, dataset_name=None, exclude_dataset=None):
        """
        Args:
            sequence_path (str): file path of the sequences
            folds (list[int]): 
            pad_val (int, optional): pad value. Defaults to -1.

        Returns: 
            (tuple): tuple containing

            - **q_seqs (torch.tensor)**: question id sequence of the 0~seqlen-1 interactions
            - **c_seqs (torch.tensor)**: knowledge concept id sequence of the 0~seqlen-1 interactions
            - **r_seqs (torch.tensor)**: response id sequence of the 0~seqlen-1 interactions
            - **mask_seqs (torch.tensor)**: masked value sequence, shape is seqlen-1
            - **select_masks (torch.tensor)**: is select to calculate the performance or not, 0 is not selected, 1 is selected, only available for 1~seqlen-1, shape is seqlen-1
            - **dqtest (dict)**: not null only self.qtest is True, for question level evaluation
        """
        #！ 先要 time 相关的了
        # dori = {"qseqs": [], "cseqs": [], "rseqs": [], "tseqs": [], "utseqs": [], "smasks": [], "dataset": []}
        dori = {"qseqs": [], "cseqs": [], "rseqs": [], "smasks": [], "dataset": []}

        df = pd.read_csv(sequence_path)
        df = df[df["fold"].isin(folds)].copy()#[0:1000]


        if dataset_name:
            print(f'loading {dataset_name} ...')
            dataset_id = datasets_dic[dataset_name]

            df = df[df['dataset'] == dataset_id]

           
        elif exclude_dataset:
            exclude_datasets = exclude_dataset.split(',')
            for cur_exclude_dataset in exclude_datasets:
                print(f'excluding dataset {cur_exclude_dataset} ...')
                cur_exclude_dataset_id = datasets_dic[cur_exclude_dataset]

                df = df[df['dataset'] != cur_exclude_dataset_id]
        else:
            print('loading all pretrain data ..')


    

        interaction_num = 0
        for i, row in df.iterrows():
            # import pdb; pdb.set_trace()
            #use kc_id or question_id as input
            if "concepts" in self.input_type:
                row_skills = []
                raw_skills = row["concepts"].split(",")
                for concept in raw_skills:
                    if concept == "-1":
                        skills = [-1] * self.max_concepts
                    else:
                        skills = [int(_) for _ in concept.split("_")]
                        skills = skills +[-1]*(self.max_concepts-len(skills))
                    row_skills.append(skills)
                dori["cseqs"].append(row_skills)
            if "questions" in self.input_type:
                try:
                    dori["qseqs"].append([int(_) for _ in row["questions"].split(",")])
                except:
                    que_seq = row["questions"]
                    print(f"i:{i}, questions:{que_seq}")

                
            dori["rseqs"].append([int(_) for _ in row["responses"].split(",")])
            dori["smasks"].append([int(_) for _ in row["selectmasks"].split(",")])

            interaction_num += dori["smasks"][-1].count(1)
            dori["dataset"].append(int(row["dataset"]))


        for key in dori:
            if key not in ["rseqs"]:#in ["smasks", "tseqs"]:
                try:
                    dori[key] = LongTensor(dori[key])
                except ValueError:
                    import pdb; pdb.set_trace()
            else:
                dori[key] = FloatTensor(dori[key])

        mask_seqs = (dori["rseqs"][:,:-1] != pad_val) * (dori["rseqs"][:,1:] != pad_val)
        dori["masks"] = mask_seqs

        dori["smasks"] = (dori["smasks"][:, 1:] != pad_val)
        print(f"interaction_num: {interaction_num}")
        # print("load data tseqs: ", dori["tseqs"])
        return dori
