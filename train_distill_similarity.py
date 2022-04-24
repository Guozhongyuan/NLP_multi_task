import torch
import torch.nn as nn
import numpy as np
from GPT2 import GPT2Model, GPT2Tokenizer
import json
from tqdm import tqdm
from torch.utils.data import Dataset
import transformers
import pandas as pd
import random
from torch.nn import functional as F

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
device = 'cuda' # cuda or cpu



class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    """
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = torch.abs(output1 - output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) + (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))     
 
        return loss_contrastive


class LayerNorm(nn.Module):
    r"""
    Layer normalization.
    """

    def __init__(self, hidden_size, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class MLP(nn.Module):
    def __init__(self, n_in, n_out):
        super(MLP, self).__init__()
        self.linear = nn.Linear(n_in, n_out)
        self.layer_norm = LayerNorm(n_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.linear(x)
        x = self.layer_norm(x)
        x = self.relu(x)
        return x


class GPT2_SIMILARITY(nn.Module):
    def __init__(self):
        super(GPT2_SIMILARITY, self).__init__()
        
        self.GPT2model = GPT2Model(
            vocab_size=30000,
            layer_size=12,
            block_size=1024,
            embedding_dropout=0.0,
            embedding_size=768,
            num_attention_heads=12,
            attention_dropout=0.0,
            residual_dropout=0.0
        )

        self.mlp =  MLP(30000, 256)

    def forward(self, x, length):
        x = self.GPT2model(x)
        classify = []
        for i in range(len(length)):
            classify.append(x[i, length[i]].view(-1))
        classify = torch.stack(classify)
        classify = self.mlp(classify)
        return classify


def load_data(data_path, tokenizer, seq_length=1024):

    data = pd.read_csv(data_path)

    all = []
    for i in range(len(data)):
        item = data.iloc[i]
        all.append(
            {
                'stockName': item['stockName'],
                'stockCode': str(item['stockCode']).zfill(6),
                'indvInduName': item['indvInduName'],
                'indvInduCode': int(item['indvInduCode']),
                'info': '。'.join(item['info'].split('\n')[1:3]),  # 要点二和要点三
            }
        )

    list_T_pairs = []
    list_F_pairs = []

    for i in tqdm(range(0, len(all)-1)):
        for j in range(i, len(all)):

                if all[i]['indvInduName']==all[j]['indvInduName']:
                    list_T_pairs.append([i,j,0])

                if all[i]['indvInduName']!=all[j]['indvInduName']:
                    list_F_pairs.append([i,j,1])
    
    print('list_T_pairs', len(list_T_pairs))
    print('list_F_pairs', len(list_F_pairs))


    pad_id = tokenizer.encoder['<pad>']

    all_tokens = []
    all_last_idx = []
    
    for item in tqdm(all):

        sentence = item['info']
        tokenized_sentence = tokenizer.encode(sentence)[:seq_length-20]
       
        tokens = tokenized_sentence
        token_length = len(tokens)

        pads = [pad_id] * (seq_length - token_length)
        tokens.extend(pads)

        all_last_idx.append(token_length)
        all_tokens.append(tokens)
    
    all_tokens = torch.tensor(all_tokens, dtype=torch.long).cuda()
    all_last_idx = torch.tensor(all_last_idx, dtype=torch.long).cuda()

    data_dict = {
        'tokens': all_tokens,
        'last_idxs': all_last_idx,
        'T_pairs': list_T_pairs,
        'F_pairs': list_F_pairs,
    }

    return data_dict
  

class PreprocessDataset(Dataset):
    
    def __init__(self, data_dict):
        self.tokens = data_dict['tokens']
        self.last_idxs = data_dict['last_idxs']
        self.T_pairs = data_dict['T_pairs']
        self.F_pairs = data_dict['F_pairs']

        pair_num = min(len(self.T_pairs), len(self.F_pairs))
        pairs = self.T_pairs[:pair_num] + self.F_pairs[:pair_num]
        self.pairs = pairs
        random.shuffle(self.pairs)

    def shuffle(self):
        random.shuffle(self.T_pairs)
        random.shuffle(self.F_pairs)
        pair_num = min(len(self.T_pairs), len(self.F_pairs))
        pairs = self.T_pairs[:pair_num] + self.F_pairs[:pair_num]
        self.pairs = pairs
        random.shuffle(self.pairs)
        
    def __getitem__(self, idx):
        pair_idx = self.pairs[idx]
        token_0 = self.tokens[pair_idx[0]]
        token_1 = self.tokens[pair_idx[1]]
        last_idx_0 = self.last_idxs[pair_idx[0]]
        last_idx_1 = self.last_idxs[pair_idx[1]]
        label = pair_idx[2]
        data = {
            'token_0': token_0,
            'token_1': token_1,
            'last_idx_0': last_idx_0,
            'last_idx_1': last_idx_1,
            'label': torch.tensor(label).cuda()
        }
        return data
    
    def __len__(self):
        return len(self.pairs)


def train(epoch, model, dataset):
    # model.mlp.train()
    losses = []
    num = 0
    for item in tqdm(dataset):

        try:
            output_0 = model(item['token_0'].unsqueeze(0), [item['last_idx_0']])
            output_1 = model(item['token_1'].unsqueeze(0), [item['last_idx_1']])

            loss = loss_fcn(output_0[0], output_1[0], item['label'])
            losses.append(loss.item())

            loss.backward()
            num = num + 1

        except Exception as err:
            print(err)

        if num == 32:
            optimizer.step()
            optimizer.zero_grad()
            num = 0

    print(f'epoch {epoch}, loss {np.mean(losses[-10000:])}')
    



if __name__ == '__main__':

    state_dict = torch.load('../models/model_pretrain_distill.pth', map_location='cpu')

    tokenizer = GPT2Tokenizer(
        'GPT2/bpe/vocab.json',
        'GPT2/bpe/chinese_vocab.model',
        max_len=512)

    data_dict = load_data('../data/stock_list_info.csv', tokenizer)
    dataset = PreprocessDataset(data_dict)


    model = GPT2_SIMILARITY()
    model.GPT2model.load_state_dict(state_dict)
    # model.GPT2model.eval()
    model.to(device)
    model.train()

    # optimizer = transformers.AdamW(model.mlp.parameters(), lr=1e-4, eps=1.0e-6)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6, eps=1.0e-8)

    loss_fcn = ContrastiveLoss()
    loss_fcn.to(device)

    for epoch in range(10):
        dataset.shuffle()
        train(epoch, model, dataset)
        torch.save(model, f"../models/similarity_{str(epoch)}.pth")

