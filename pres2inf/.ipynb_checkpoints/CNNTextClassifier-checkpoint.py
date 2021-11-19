import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_Text(nn.Module):
    def __init__(self, n_classes, vocab_len, embed_size, weight_matrix, filter_sizes=[1,2,3,5], num_filters = 36, dropout_rate = 0.1):
        super(CNN_Text, self).__init__()
        filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.embedding = nn.Embedding(vocab_len, embed_size)
        self.embedding.weight = nn.Parameter(weight_matrix)
        self.embedding.weight.requires_grad = False
        self.convs1 = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=self.num_filters, kernel_size=(K, embed_size)) for K in filter_sizes])
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(len(filter_sizes)*self.num_filters, n_classes)

    def forward(self, x):
        pad = []
        for c in self.convs1:
            pad.append(c.kernel_size[0])
        x_padded= []
        for p in pad:
            x_padded.append(F.pad(input=x, pad=(p-1, p-1, 0, 0), mode='constant', value=0))
        x_padded = [self.embedding(i).unsqueeze(1) for i in x_padded]
        x = [F.relu(conv(x)).squeeze(3) for (x,conv) in zip(x_padded,self.convs1)]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)  
        logit = self.fc1(x)
        
        return logit