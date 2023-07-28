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

from .utils import get_pad_chars, words2charindices, max_word_length, char_pad_token, suffixes, class_infinitive, char_list,vocab_len
from .CNNTextClassifier import CNN_Text

#from utils import get_pad_chars, words2charindices, max_word_length, char_pad_token, suffixes, class_infinitive, char_list,vocab_len
#from CNNTextClassifier import CNN_Text

from collections import OrderedDict

from typing import Any

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt




# model specifications
    
cnn_4_params = {
    "embed_size"     : 300,        
    "n_classes"      : 8,
    "vocab_len"      : vocab_len,
    "filter_sizes" : [1,2,3,5],
    "num_filters"  : 36,
    "dropout_rate" : 0.1    
}

cnn_1_params = {
    "embed_size"     : 300,        
    "n_classes"      : 8,
    "vocab_len"      : vocab_len,
    "filter_sizes" : [5],
    "num_filters"  : 36,
    "dropout_rate" : 0.1    
}






# evaluations
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
        confusion[pred[i]][test_data_y[i]] += 1
    total += test_data_y.shape[0]
    sum_loss += loss.item()*test_data_y.shape[0]
    sum_rmse += np.sqrt(mean_squared_error(pred, test_data_y.unsqueeze(-1)))*test_data_y.shape[0]
    microf1 = f1_score(test_data_y, pred, average='micro')
    macrof1 = f1_score(test_data_y, pred, average='macro')
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
    ax = sn.heatmap(df_cm,cmap="viridis", annot=True, annot_kws={"size": 16}) # font size
    
    # Postavljanje osi    
    ax.set_xticklabels( suffixes, rotation=90)
    ax.set_yticklabels(suffixes)

    plt.savefig('pres2inf_confusion.png')
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
    ax = fig.add_subplot(111)    
    cax = ax.matshow(odrezaniHeatmap,cmap='viridis')
    

    # Postavljanje osi
    suffixes=list(glagol)
    ax.grid(False)
    ax.set_xticklabels([''] + suffixes)
    
    # turn off ticks
    ax.tick_params(left=False, bottom=True) ## other options are right and top
    ax.set_yticklabels([])

    # Prikaz oznake na svakoj vrijednosti
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))    
   
    return plt.show()

def heatmap(glagol, model):
    model.eval()
    vekt = torch.from_numpy(np.array(get_pad_chars(words2charindices(glagol),max_word_length, char_pad_token))).long()
    pred = model(vekt.unsqueeze(dim=0))
    #print(pred)
    pred_klase = torch.max(pred, 1)[1]
    print('PRES2INF class:',class_infinitive[pred_klase.item()])
    return class_heatmap(glagol,vekt,pred_klase,model)
    



    
def predict(model, glagol):
    model.eval()
    vekt = torch.from_numpy(np.array(get_pad_chars(words2charindices(glagol),max_word_length, char_pad_token))).long()
    pred = model(vekt.unsqueeze(dim=0))    
    pred_klase = torch.topk(pred, 2)[1].squeeze(dim=0)        
    return class_infinitive[pred_klase[0].item()]    




### model loading

def loadModel(model_path = 'results/model.weights',params = cnn_4_params):
    params["weight_matrix"] = torch.from_numpy(np.zeros((len(char_list)+1, params["embed_size"]))).float()
    model = CNN_Text(**params)    
    model.load_state_dict(torch.load(model_path))
    
    return model

### probs
def probabilities(model : Any, glagol : str) -> None:
    model.eval()
    vekt = torch.from_numpy(np.array(get_pad_chars(words2charindices(glagol),max_word_length, char_pad_token))).long()
    pred = model(vekt.unsqueeze(dim=0))
    poz = torch.add(pred,-pred.min().item())
    vjer = torch.div(poz,poz.sum().item())
    
    probs = {class_infinitive[k] : round(v,3)  for (k,v) in enumerate(vjer[0].detach().tolist())}
                                             
    return dict(OrderedDict(sorted(probs.items(), key=lambda kv: kv[1],reverse=True)))
    
    