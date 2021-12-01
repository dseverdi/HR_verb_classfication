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
from utils import get_pad_chars, words2charindices, max_word_length, char_pad_token, nazivi, klase_infinitiv

def evaluateOnTestSet(model, params_outputh_path, test_set,sve_kategorije):
    correct = 0
    total = 0
    sum_loss = 0.0
    sum_rmse = 0.0
    confusion = torch.zeros(len(sve_kategorije), len(sve_kategorije))
    output_path = params_outputh_path
    model.load_state_dict(torch.load(output_path))
    print("Final evaluation on test set",)
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
    microf1 = f1_score(test_data_y, pred, average='macro')
    macrof1 = f1_score(test_data_y, pred, average='micro')
    weightedf1 = f1_score(test_data_y, pred, average='weighted')
    print("test loss %.3f, test accuracy %.3f, test rmse %.3f, test microF1 %.3f, test macroF1 %.3f, test weightedF1 %.3f" % (sum_loss/total, correct/total, sum_rmse/total, microf1, macrof1, weightedf1))

    # Normalizacija dijeljenjem svakog retka sumom tog retka.
    for i in range(len(sve_kategorije)):
        confusion[i] = confusion[i] / confusion[i].sum()

    print(confusion)

    # Definiranje grafa
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion.numpy())
    fig.colorbar(cax)

    # Postavljanje osi
    ax.set_xticklabels([''] + nazivi, rotation=90)
    ax.set_yticklabels([''] + nazivi)

    # Prikaz oznake na svakoj vrijednosti
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # sphinx_gallery_thumbnail_number = 2
    plt.show()

def heatmapZaKlasu(glagol,v2,klasa, model):
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
    odrezaniHeatmap = hm.detach().squeeze(0)[:,:len(glagol)]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(odrezaniHeatmap)

    # Postavljanje osi
    nazivi=list(glagol)
    ax.set_xticklabels([''] + nazivi)
    ax.set_yticklabels([])

    # Prikaz oznake na svakoj vrijednosti
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.show()

def heatmap(glagol, model):
    model.eval()
    vekt = torch.from_numpy(np.array(get_pad_chars(words2charindices(glagol),max_word_length, char_pad_token))).long()
    pred = model(vekt.unsqueeze(dim=0))
    #print(pred)
    pred_klase = torch.max(pred, 1)[1]
    print('infinitiv završava na:',klase_infinitiv[pred_klase.item()])
    heatmapZaKlasu(glagol,vekt,pred_klase,model)
    
    
def predict(model, glagol):
    model.eval()
    vekt = torch.from_numpy(np.array(get_pad_chars(words2charindices(glagol),max_word_length, char_pad_token))).long()
    pred = model(vekt.unsqueeze(dim=0))    
    pred_klase = torch.topk(pred, 2)[1].squeeze(dim=0)        
    return klase_prezent[pred_klase[0].item()],klase_prezent[pred_klase[1].item()]
    
    
   
    
params = {
    "embed_size"     : 300,        
    "n_classes"      : 5,
    "vocab_len"      : vocab_len,
    "filter_sizes" : [1,2,3,5],
    "num_filters"  : 36,
    "dropout_rate" : 0.1    
}

params["weight_matrix"] = torch.from_numpy(np.zeros((len(char_list)+1, params["embed_size"]))).float()
    
    
def loadModel(model_path = 'results/20211117_193445/model.weights'):
    # define model
    global params
    model = CNN_Text(**params)    
    model.load_state_dict(torch.load(model_path))
    
    return model