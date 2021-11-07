import torch
import numpy as np
from GPT2 import GPT2Model, GPT2Tokenizer
import re

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
device = 'cpu' #'cuda'


def sample(text, max_len=10):
    ids = tokenizer.encode(text)
    input_id = torch.tensor((np.array(ids).reshape(1, -1).astype('int64'))).to(device)
    output, cached_kvs = model(input_id, use_cache=True)
    nid = int(np.argmax(output[0, -1].detach().cpu().numpy()))
    ids += [nid]
    out = [nid]
    for i in range(max_len):
        input_id = torch.tensor(np.array([nid]).reshape(1, -1).astype('int64')).to(device)
        output, cached_kvs = model(input_id, cached_kvs, use_cache=True)
        nid = int(np.argmax(output[0, -1].detach().cpu().numpy()))
        ids += [nid]
        if nid==3:
            break
        out.append(nid)
    return (tokenizer.decode(out))

def similarity(nextstr, reslist):
    '''
        如果下一个字符串与之前的60%以上重复则不选取
    '''
    res = 0
    for stri in reslist:
        num = 0
        for i in nextstr:
            if i in stri:
                num = num + 1
        if num/len(nextstr) > 0.6:
            res = 1
    return res

def ask_question(question, max_len=10):
    '''
        正则去标点符号，去重后拼接句子
    '''
    res = sample('''问题：%s 答案：''' % question, max_len)
    res = re.findall(r"[\w']+", res)
    reslist = [res[0]]
    for i in range(0,len(res)-1):
        if len(res[i+1]) < 2:
            continue
        if similarity(res[i+1], reslist) == 0:
            reslist.append(res[i+1])
    res = ','.join(reslist)+'。'
    return res if len(res) > 10 else '对不起，我不知道。'


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

print(ask_question('什么是更好的期限或永久人寿保险？', max_len=40))