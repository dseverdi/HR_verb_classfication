import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import mean_squared_error, f1_score
import os
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import math

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict

from .utils import get_pad_chars, words2charindices, max_word_length, char_pad_token, suffixes, class_present, char_list, vocab_len
from .FFTextClassifier import FF_Text, FF_Text_withWindows

from typing import Any

def evaluateOnTestSet(model : Any,test_set : str,all_categories : list):
    correct  = 0
    total    = 0
    sum_loss = 0.0
    sum_rmse = 0.0
    
            
    confusion = torch.zeros(len(all_categories), len(all_categories))    
    
    # evaluate model   
    model.eval()
    test_data_x = [get_pad_chars(words2charindices(d[0]),max_word_length, char_pad_token) for d in test_set]
    test_data_y = [int(d[1]) for d in test_set]
    test_data_x = torch.from_numpy(np.array(test_data_x)).long()
    test_data_y = torch.from_numpy(np.array(test_data_y)).long()
    y_hat = model.forward(test_data_x)
    loss = F.cross_entropy(y_hat, test_data_y)
    pred = torch.max(y_hat, 1)[1]    
    correct += (pred == test_data_y).float().sum()
      
    for i in range(len(pred)):
        confusion[pred[i]][test_data_y[i]] += 1.0
    total += test_data_y.shape[0]
    sum_loss += loss.item()*test_data_y.shape[0]
    sum_rmse += np.sqrt(mean_squared_error(pred, test_data_y.unsqueeze(-1)))*test_data_y.shape[0]
    microf1 = f1_score(test_data_y, pred, average='macro')
    macrof1 = f1_score(test_data_y, pred, average='micro')
    weightedf1 = f1_score(test_data_y, pred, average='weighted')
    print(" --------------Evaluation metrics: ---------------------- \
          \n\r * test loss: %.3f\n\r * test accuracy: %.3f,\n\r * test rmse: %.3f,\n\r * test microF1: %.3f,\n\r * test macroF1: %.3f,\n\r * test weightedF1: %.3f" % (sum_loss/total, correct/total, sum_rmse/total, microf1, macrof1, weightedf1))
    
    
    # Normalizacija dijeljenjem svakog retka sumom tog retka.
    for i in range(len(all_categories)):
        confusion[i] = confusion[i] / confusion[i].sum()

    
    # Definiranje prikaza matrice zbunjenosti   
    size = len(suffixes)    
    df_cm = pd.DataFrame(confusion.numpy(), range(size), range(size))
    fig = plt.figure(figsize=(10,7))
    sn.set(font_scale=1.4) # for label size
    ax = sn.heatmap(df_cm,cmap="viridis", annot=True, annot_kws={"size": 18}) # font size
    
    # Postavljanje osi    
    ax.set_xticklabels( suffixes, rotation=90)
    ax.set_yticklabels(suffixes)
    
    #ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    #ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    
    plt.savefig('inf2pres_confusion.png')

    return plt.show()

def class_heatmap(glagol,v2,klasa, model):
    pad = []
    for conv in model.convs1:
        pad.append(conv.kernel_size[0])
    x = []
    for p in pad:
        x.append(F.pad(input=v2, pad=(p-1, p-1), mode='constant', value=0))
    v = []
    for i in x:
        v.append(model.embedding(i.unsqueeze(dim=0)).unsqueeze(1))
    brojac = 0
    hm = []
    tezine = model.fc1.weight[klasa]
    tezine = tezine.reshape(len(model.convs1),-1)
    for (vekt, conv) in zip(v,model.convs1):
        rez = conv(vekt)
        rez = F.relu(rez).squeeze(3)
        Fl = tezine[brojac]
        Fl=Fl.unsqueeze(0)
        brojac += 1
        matr = Fl @ rez
        mp = F.max_pool1d(matr, kernel_size=conv.kernel_size[0], stride=1)
        hm.append(mp)

    hm = sum(hm)
    odrezaniHeatmap = hm.detach().squeeze(0)[:,:len(glagol)].numpy()
   

    fig = plt.figure()
    fig.tight_layout()
    #plt.figure(figsize=(15,10))
    ax = fig.add_subplot(111)    
    cax = ax.matshow(odrezaniHeatmap,cmap='viridis')

    # Postavljanje osi
    suffixes=list(glagol)
    ax.grid(False)
    ax.set_xticklabels([''] + suffixes)
    ax.set_yticklabels([])

    # Prikaz oznake na svakoj vrijednosti
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    return plt.show()

def heatmap(model, glagol):
    model.eval()
    vekt = torch.from_numpy(np.array(get_pad_chars(words2charindices(glagol),max_word_length, char_pad_token))).long()
    pred = model(vekt.unsqueeze(dim=0))
    #print(pred)
    pred_klase = torch.max(pred, 1)[1]
    print('INF2PRES class:',class_present[pred_klase.item()])
    return class_heatmap(glagol,vekt,pred_klase,model)
    
    
def predict(model, glagol):
    model.eval()
    vekt = torch.from_numpy(np.array(get_pad_chars(words2charindices(glagol),max_word_length, char_pad_token))).long()
    pred = model(vekt.unsqueeze(dim=0))    
    pred_klase = torch.topk(pred, 2)[1].squeeze(dim=0)        
    return class_present[pred_klase[0].item()]
    
    
   
    
params = {
    "embed_size"     : 300,        
    "n_classes"      : 4,
    "vocab_len"      : vocab_len,
    "hidden_size"     : 300
}

params["weight_matrix"] = torch.from_numpy(np.zeros((len(char_list)+1, params["embed_size"]))).float()
    
    
def loadModel(model_path = 'results/20211208_162956/model.weights'):
    # define model
    global params
    model = FF_Text(**params)    
    model.load_state_dict(torch.load(model_path))
    
    return model


   
def loadModel(model_path = 'results/20211208_162956/model.weights'):
    # define model
    global params
    model = FF_Text(**params)    
    #import pdb
    #pdb.set_trace()
    model.load_state_dict(torch.load(model_path))
    
    return model

params_withWindow = {
    "embed_size"     : 300,        
    "n_classes"      : 4,
    "vocab_len"      : vocab_len,
    "hidden_size"     : 300,
    "window_size"     : 5
}

params_withWindow["weight_matrix"] = torch.from_numpy(np.zeros((len(char_list)+1, params["embed_size"]))).float()

def loadModel_withWindow(model_path = 'results/model.weights'):
    # define model
    global params_withWindow
    model = FF_Text_withWindows(**params_withWindow)    
    model.load_state_dict(torch.load(model_path))
    
    return model



def probabilities(model : Any, glagol : str) -> None:
    probs = {}
    model.eval()
    vekt = torch.from_numpy(np.array(get_pad_chars(words2charindices(glagol),max_word_length, char_pad_token))).long()
    pred = model(vekt.unsqueeze(dim=0))
    poz = torch.add(pred,-pred.min().item())
    vjer = torch.div(poz,poz.sum().item())
    
    probs = {class_present[k] : round(v,3) for (k,v) in enumerate(vjer[0].detach().tolist())}
                                             
    return dict(OrderedDict(sorted(probs.items(), key=lambda kv: kv[1],reverse=True)))
    
    