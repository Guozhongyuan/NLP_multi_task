import torch
import numpy as np
from GPT2 import GPT2Model, GPT2Tokenizer
from data.samplers import RandomSampler
from tqdm import tqdm
import torch.nn as nn
import transformers
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = 'cuda' #'cuda'

class GenDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, split, tokenizer: GPT2Tokenizer, seq_length=1024, ratio=1):
        self.split = split
        self.tokenizer = tokenizer
        self.ratio = ratio

        self.pad_id = tokenizer.encoder['<pad>']
        self.eod_token = tokenizer.encoder['<eod>']
        self.seq_length = seq_length
        
        with open(data_path, "r", encoding='utf-8') as f:
            data = f.readlines()
        self.samples = self.process(data)
        

    def process(self, data):
        samples = []
        for doc in tqdm(data[:int(self.ratio * len(data))]):
            token_ids = self.tokenizer.encode(doc)
            token_ids.append(self.eod_token)
            start = 0
            while start + self.seq_length + 1 < len(token_ids):
                samples.append(token_ids[start: start + self.seq_length + 1])
                start = start + self.seq_length + 1
            samples.append(token_ids[start:] + [self.pad_id] * (self.seq_length + 1 - (len(token_ids) - start)))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def collate(self, samps):
        bs = len(samps)

        # the data that need to go through the model
        batch_sample = {
            "input_ids": torch.ones(bs, self.seq_length).long() * self.pad_id,
        }

        # the data that do not need to go through the model
        no_model_sample = {
            "labels": torch.ones(bs, self.seq_length).long() * self.pad_id,
            "loss_mask": torch.zeros(bs, self.seq_length).float()
        }

        for i, samp in enumerate(samps):
            assert len(samp) == self.seq_length + 1, (len(samp), self.seq_length)
            batch_sample["input_ids"][i] = torch.tensor(samp[:-1], dtype=torch.long)
            no_model_sample["labels"][i] = torch.tensor(samp[1:], dtype=torch.long)
            no_model_sample["loss_mask"][i] = (no_model_sample["labels"][i] != self.pad_id).float()

        return batch_sample, no_model_sample


def load_data(data_path, data_type, tokenizer, ratio=1):

    # Dataset
    # filename = os.path.join(data_path, data_type + '.txt')
    filename = data_path
    dataset = GenDataset(filename, data_type, tokenizer, ratio=ratio)
    
    # Use a random sampler with distributed batch sampler.
    if data_type == 'train':
        sampler = RandomSampler(dataset)
    else:
        sampler = torch.utils.data.SequentialSampler(dataset)
    
    # Torch dataloader.
    return torch.utils.data.DataLoader(dataset,
                                       sampler=sampler,
                                       batch_size=4,
                                       num_workers=0,
                                       pin_memory=True,
                                       collate_fn=dataset.collate), dataset


def train(model, dataloader, optimizer, device, mode="train"):
    model.train()
    bs=0
    for batch, no_model_batch in tqdm(dataloader):

        for k in batch.keys():
            batch[k] = batch[k].to(device)
        for k in no_model_batch.keys():
            no_model_batch[k] = no_model_batch[k].to(device)
        output = model(batch['input_ids'])
        predict = output.contiguous().float().reshape(-1,30000)
        target = no_model_batch["labels"].reshape(-1)
        losses = torch.tensor([loss_fcn(predict[i].view(-1,30000), target[i].view(-1)) for i in range(len(target))], requires_grad=True).to(device)

        loss_mask = no_model_batch["loss_mask"].view(-1)
        loss = torch.sum(losses * loss_mask, dim=-1) / loss_mask.sum(dim=-1)
        loss.backward()

        bs = bs + 4
        if bs==16:
            optimizer.step()
            optimizer.zero_grad()
            bs = 0


if __name__ == '__main__':

    model = GPT2Model(
        vocab_size=30000,
        layer_size=12,
        block_size=1024,
        embedding_dropout=0.0,
        embedding_size=768,
        num_attention_heads=12,
        attention_dropout=0.0,
        residual_dropout=0.0)

    state_dict = torch.load('./data/model_pretrain_distill.pth', map_location='cpu')

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    tokenizer = GPT2Tokenizer(
        'GPT2/bpe/vocab.json',
        'GPT2/bpe/chinese_vocab.model',
        max_len=512)

    train_dataloader, dataset = load_data('./data/HKFinancialStatements_zh_cn.txt', 'train', tokenizer, ratio=0.5)

    loss_fcn = nn.CrossEntropyLoss()
    loss_fcn.to(device)

    optimizer = transformers.AdamW(model.parameters(), lr=1.5e-4, eps=1.0e-9)  # lr=1.5e-4, eps=1.0e-9

    train(model, train_dataloader, optimizer, device)
    torch.save(model.state_dict(), "./data/financial.pth")