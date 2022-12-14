import torch
import torch.nn as nn
import torch.nn.functional as F
from  .utils import max_word_length

class FF_Text(nn.Module):
    #def __init__(self, n_classes, vocab_len, embed_size, weight_matrix, hidden_size, dropout_rate = 0.1):
    def __init__(self, n_classes, vocab_len, embed_size, weight_matrix, hidden_size):
        super(FF_Text, self).__init__()
        self.embedding = nn.Embedding(vocab_len, embed_size)
        self.embedding.weight = nn.Parameter(weight_matrix)
        self.embedding.weight.requires_grad = False
        self.l1 = nn.Linear(max_word_length*embed_size, hidden_size) 
        self.relu = nn.ReLU()
        #self.sigmoid = nn.Sigmoid()
        self.l2 = nn.Linear(hidden_size, n_classes)
        #self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        #x = self.embedding(x).unsqueeze(1)
        x = self.embedding(x)
        #import pdb
        out = self.l1(torch.flatten(x,1)) # flatten all dimensions except the batch dimension
        #pdb.set_trace()
        out = self.relu(out)
        #out = self.sigmoid(out)
        out = self.l2(out)
        #pdb.set_trace()
        return out
    
class FF_Text_withWindows(nn.Module):
    #def __init__(self, n_classes, vocab_len, embed_size, weight_matrix, hidden_size, dropout_rate = 0.1):
    def __init__(self, n_classes, vocab_len, embed_size, weight_matrix, hidden_size, window_size):
        super(FF_Text_withWindows, self).__init__()
        self.embedding = nn.Embedding(vocab_len, embed_size)
        self.embedding.weight = nn.Parameter(weight_matrix)
        self.embedding.weight.requires_grad = False
        self.l1 = nn.Linear((max_word_length+window_size-1)*embed_size, hidden_size) #ovaj max_word_length+window_size-1 se dobije nakon maxpoolinga od vektora 
        self.relu = nn.ReLU()
        #self.sigmoid = nn.Sigmoid()
        self.l2 = nn.Linear(hidden_size, n_classes)
        self.window_size = window_size
        #self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        #x = self.embedding(x)
        #import pdb
        x_padded = F.pad(input=x, pad=(self.window_size-1, self.window_size-1, 0, 0), mode='constant', value=0)
        x_padded = self.embedding(x_padded)
        #pdb.set_trace()
        out = F.max_pool1d(torch.transpose(x_padded, 2, 1), kernel_size=self.window_size, stride=1) #moramo okrenuti x_padded da mapravi max_pooling po retcima
        out = torch.transpose(out,1, 2) #vrati natrag da budu batch, vocab_len-window_size+1, embed_size
        #pdb.set_trace()
        #import pdb
        out = self.l1(torch.flatten(out,1)) # flatten all dimensions except the batch dimension
        #pdb.set_trace()
        out = self.relu(out)
        #out = self.sigmoid(out)
        out = self.l2(out)
        #pdb.set_trace()
        return out