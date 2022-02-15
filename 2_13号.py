import torch
import torch.nn as nn
import torch.utils.data as tud
import torch.nn.functional as F
from collections import Counter #记录单词出现次数
import numpy as np
import scipy
import math
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
import random
from torch.utils.data import DataLoader
USE_CUDA=torch.cuda.is_available()
random.seed(123)
np.random.seed(123)
torch.manual_seed(123)
if USE_CUDA:
    torch.cuda.manual_seed()
c=3 #window size
k=100 #number of negative sample
num_epochs=10
max_vocab_size=30000
batch_size=128
learning_rate=1e-4
embedding_size=100

def word_tokenize(text):
    return text.split()

with open('text8/text8.train.txt','r') as f:
    text=f.read()
text=text.split()
# print(text[:10]) ['anarchism', 'originated', 'as', 'a', 'term', 'of', 'abuse', 'first', 'used', 'against']
vocab=dict(Counter(text).most_common(max_vocab_size-1))
# print(vocab)
vocab['<unk>']=len(text)-np.sum(list(vocab.values()))
# print(vocab) 得到字典数据 {'the': 959616, 'of': 537144, 'and': 376233, 'one': 363077, 'in': 335477, 'a': 293387, 'to': 285950,
idx_to_word=[ word for word in vocab.keys() ]
# print(idx_to_word[:100]) #得到列表单词
word_to_idx={word:i for i,word in enumerate(idx_to_word)}
# print(list(word_to_idx.items())[:20]) word_to_idx是字典
# print(list(word_to_idx)[:20])['the', 'of', 'and', 'one', 'in', 'a', 'to', 'zero', 'nine', 'two', 'is', 'as', 'eight', 'for', 's', 'five', 'three', 'was', 'by', 'that']
"""得到每一个单词的频率"""
word_counts=np.array([count for count  in vocab.values()],dtype=np.float32)
word_frequent=word_counts/np.sum(word_counts)
word_frequent=word_frequent**(3./4.)
word_frequent=word_counts/np.sum(word_counts)
vocab_size=len(idx_to_word)

"""实现dataloader 这个库可以轻松返回batch"""
class WordEmbedding(tud.Dataset):
    def __init__(self,text,word_to_idx,idx_to_word,word_frequent,word_counts):
        super(WordEmbedding,self).__init__()
        self.text_encoded=[word_to_idx.get(word,word_to_idx['<unk>']) for word in text]
        self.text_encoded=torch.LongTensor(self.text_encoded)
        self.word_to_idx=word_to_idx
        self.idx_to_word=idx_to_word
        self.word_frequent=torch.LongTensor(word_frequent)
        self.word_counts=torch.LongTensor(word_counts)
    def __len__(self):
        return len(self.text_encoded)
    def __getitem__(self, idx):
        """只有这三个函数才能组合成dataset"""
        center_word=self.text_encoded(idx)
        pos_indices=list(range(idx-C))+list(range(idx+1,idx+C+1))
        pos_indices=[i%len(self.text_encoded) for i in pos_indices]#取余，防止超出text长度
        pos_words=self.text_encoded(pos_indices)
        neg_words=torch.multinomial(self.word_frequent,K*pos_words.shape[0],True)
        return  center_word,pos_words,neg_words

#创建dataset,dataloader
dataset=WordEmbedding(text,word_to_idx,idx_to_word,word_frequent,word_counts)
dataloader=DataLoader(dataset,batch_size=batch_size)
"""定义模型
"""
class EmbeddingModel(nn.Module):
    def __init__(self):
        super(EmbeddingModel,self).__init__()
        self.vocab_size=vocab_size
        self.embedding_size=embedding_size
        self.in_embed=nn.Embedding(self.vocab_size,self.embedding_size)
        self.out_embed=nn.Embedding(self.vocab_size,self.embedding_size)


    def forward(self,input_label,pos_label,neg_label):


