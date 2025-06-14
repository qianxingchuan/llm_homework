#!/usr/bin/env python
# coding: utf-8

# In[1]:


#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('BAAI/bge-reranker-large', cache_dir='/root/autodl-tmp/models')


# In[2]:


import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('/root/autodl-tmp/models/BAAI/bge-reranker-large')
model = AutoModelForSequenceClassification.from_pretrained('/root/autodl-tmp/models/BAAI/bge-reranker-large')
model.eval()

pairs = [['what is panda?', 'The giant panda is a bear species endemic to China.']]
inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt')
scores = model(**inputs).logits.view(-1).float()
print(scores)  # 输出相关性分数


# In[3]:


pairs = [
    ['what is panda?', 'The giant panda is a bear species endemic to China.'],  # 高相关
    ['what is panda?', 'Pandas are cute.'],                                     # 中等相关
    ['what is panda?', 'The Eiffel Tower is in Paris.']                        # 不相关
]
inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt')
scores = model(**inputs).logits.view(-1).float()
print(scores)  # 输出相关性分数

