import torch
import numpy as np
from GPT2 import GPT2Model, GPT2Tokenizer

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = 'cuda' #'cuda'

import os
import json
import random
from torch.utils.data import TensorDataset
import torch.nn.functional as F
from data.samplers import RandomSampler
import transformers
import torch.nn as nn


def get_masks_and_position_ids(data,
                               eod_token,
                               reset_position_ids,
                               reset_attention_mask):
    # Extract batch size and sequence length.
    batch_size, seq_length = data.size()

    # Attention mask (lower triangular).
    if reset_attention_mask:
        att_mask_batch = batch_size
    else:
        att_mask_batch = 1
    attention_mask = torch.tril(torch.ones(
        (att_mask_batch, seq_length, seq_length), device=data.device)).view(att_mask_batch, 1, seq_length, seq_length)

    # Loss mask.
    loss_mask = torch.ones(data.size(), dtype=torch.float, device=data.device)
    loss_mask[data == eod_token] = 0.0

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long,
                                device=data.device)
    position_ids = position_ids.unsqueeze(0).expand_as(data)
    # We need to clone as the ids will be modifed based on batch index.
    if reset_position_ids:
        position_ids = position_ids.clone()

    if reset_position_ids or reset_attention_mask:
        # Loop through the batches:
        for b in range(batch_size):

            # Find indecies where EOD token is.
            eod_index = position_ids[b, data[b] == eod_token]
            # Detach indecies from positions if going to modify positions.
            if reset_position_ids:
                eod_index = eod_index.clone()

            # Loop through EOD indecies:
            prev_index = 0
            for j in range(eod_index.size()[0]):
                i = eod_index[j]
                # Mask attention loss.
                if reset_attention_mask:
                    attention_mask[b, 0, (i+1):, :(i+1)] = 0
                # Reset positions.
                if reset_position_ids:
                    position_ids[b, (i+1):] -= (i + 1 - prev_index)
                    prev_index = i + 1

    return attention_mask, loss_mask, position_ids


def get_batch(context_tokens, tokenizer, batchsize):
    tokens = context_tokens
    tokens = tokens.view(batchsize, -1).contiguous()

    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_masks_and_position_ids(
        tokens,
        tokenizer.encoder['<eod>'],
        reset_position_ids=True,
        reset_attention_mask=True)

    return tokens, attention_mask, position_ids

def top_k_logits(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
 
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
        
    if top_p > 0.0:
        #convert to 1D
        logits=logits.view(logits.size()[1]).contiguous()
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
        #going back to 2D
        logits=logits.view(1, -1).contiguous()
	
    return logits


def load_tnews_data(data_path, data_type, tokenizer, few_shot=False, seq_length=1024):
    # args = get_args()

    filename = os.path.join(data_path, data_type+'.json')
    objs = []
    with open(filename) as fin:
        for line in fin:
            objs.append(json.loads(line.strip()))

    pad_id = tokenizer.encoder['<pad>']
    # args.eod_token = tokenizer.encoder['<eod>']

    labels = []
    label_map = {}
    label_reverse = {}
    with open(os.path.join(data_path, 'labels.json')) as fin:
        for i, line in enumerate(fin):
            obj = json.loads(line.strip())
            labels.append(obj['label_desc'])
            label_map[obj['label_desc']] = i
            label_reverse[obj['label']] = obj['label_desc']

    all_tokens = []
    all_masks = []
    all_labels = []
    for _, obj in enumerate(objs):
        sentence = obj['sentence']
        tokenized_sentence = tokenizer.encode(sentence)[:seq_length-20]
        obj['label_desc'] = label_reverse[obj['label']]

        if few_shot:
            cur_labels = random.sample(labels, 3)
            while obj['label_desc'] in cur_labels:
                cur_labels = random.sample(labels, 3)
            cur_labels.append(obj['label_desc'])
            cur_label = cur_labels.index(obj['label_desc'])
            assert cur_label != -1
        else:
            cur_labels = labels
            cur_label = label_map[obj['label_desc']]

        all_labels.append(cur_label)

        for _, label in enumerate(cur_labels):
            prompt = "这是关于{}的文章：".format(label)
            prompt_tokens = tokenizer.encode(prompt)
            prompt_len = len(prompt_tokens)
            tokens = prompt_tokens + tokenized_sentence
            second_mask = [0] * (seq_length-1)
            for idx in range(prompt_len-1, len(tokens)-1):
                second_mask[idx] = 1
            all_masks.append(second_mask)
            token_length = len(tokens)
            assert token_length < seq_length
            tokens.extend([pad_id] * (seq_length - token_length))
            all_tokens.append(tokens)
    
    all_tokens = torch.tensor(all_tokens, dtype=torch.long)
    all_masks = torch.tensor(all_masks, dtype=torch.float)
    dataset = TensorDataset(all_tokens, all_masks)

    if data_type == 'train':
        sampler = RandomSampler(dataset)
    else:
        sampler = torch.utils.data.SequentialSampler(dataset)
    
    # Torch dataloader.
    return torch.utils.data.DataLoader(dataset,
                                       sampler=sampler,
                                       num_workers=0,
                                       pin_memory=True), all_labels


from tqdm import tqdm
import torch.nn as nn


# 每一条新闻标题，通过交叉熵损失，在15个label下有各自的相关性损失。argin看看哪个损失最小
def eval(model, train_dataloader, all_labels,device, tokenizer):
    model.eval()

    with torch.no_grad():
        res = []
        for batch in tqdm(train_dataloader):
            tokens, masks = [x.to(device) for x in batch]
            tokens, _, _ = get_batch(tokens, tokenizer, batchsize=1)
            output = model(tokens)
            predict = output[:, :-1, :].contiguous().float().reshape(-1,30000)
            target = tokens[:, 1:].reshape(-1)
            losses = torch.tensor([loss_fcn(predict[i].view(-1,30000), target[i].view(-1)) for i in range(len(target))]).to(device)
            loss = torch.sum(losses * masks, 1) / torch.sum(masks, -1)
            res.append(loss)
        
    cnt = 0
    label_size = 15
    num_inst = len(res) // label_size
    for x in range(num_inst):
        label = all_labels[x]
        cur_res = res[x*label_size:(x+1)*label_size]
        pos = np.argmin(torch.tensor(cur_res).numpy())
        if pos == label:
            cnt += 1
            print(cnt, '个分类正确')
    print('准确率', float(cnt)/num_inst)


if __name__ == '__main__':

    model = GPT2Model(
            vocab_size=30000,
            layer_size=32,
            block_size=1024,
            embedding_dropout=0.0,
            embedding_size=2560,
            num_attention_heads=32,
            attention_dropout=0.0,
            residual_dropout=0.0)

    state_dict = torch.load('./data/model_pretrain_large.pth', map_location='cpu')

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    tokenizer = GPT2Tokenizer(
        'GPT2/bpe/vocab.json',
        'GPT2/bpe/chinese_vocab.model',
        max_len=512)

    eval_dataloader, eval_labels = load_tnews_data('../nlpdata/tnews_public', 'dev', tokenizer)

    loss_fcn = nn.CrossEntropyLoss()
    loss_fcn.to(device)

    eval(model, eval_dataloader, eval_labels, device, tokenizer)