import torch
import torch.nn as nn
import random
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, f1_score
import os
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import fasttext
import fasttext.util
import math

fast_text_path = '/home/rcoric/cc.hr.300.bin'

#učitaj fasttext vektore
ft = fasttext.load_model(fast_text_path)

#napravi preslikavanje znakova u indekse
char_pad_token = 0
max_word_length = 20

klase_infinitiv = {0: 'ati', 1: 'iti', 2:'ijeti', 3:'jeti', 4:'eti', 5:'uti', 6:'ći'}
nazivi = ['ati', 'iti', 'ijeti', 'jeti', 'eti', 'uti', 'ći']

char_list = list("""abcčćdđefghijklmnoprsštuvzž""")
char2id = dict() # Pretvara znakove u cijele brojeve
char2id['<pad>'] = 0
for i, c in enumerate(char_list):
    char2id[c] = len(char2id)
    
vocab_len = len(char2id)

id2char = {v: k for k, v in char2id.items()} # Pretvara cijele brojeve u znakove

#pretvara rijec u listu indeksa iz char_list
def words2charindices(w):
    return [char2id[ch] for ch in w]

#ovoj funkciji treba proslijediti što vrati funkcija words2charindices
def get_pad_chars(chars, max_word_length, char_pad_token):
    if len(chars) > max_word_length:
        return chars[:max_word_length]
        
    return chars + [char_pad_token] * (max_word_length - len(chars))

#napravi pretrained embeddings matricu
matrix_len = len(char_list)+1
weights_matrix = np.zeros((matrix_len, 300))

for i, word in enumerate(char_list):
    weights_matrix[i+1] = ft.get_word_vector(word)
    
weights_matrix = torch.from_numpy(weights_matrix).float()

def ucitajPodatke(path=''):
    """Učitava glagole u skupove za treniranje, validaciju i testiranje.
    path treba biti u obliku putanja_do_fileova i obvezno na kraju /."""
    glagoli_train_file = open(path+'glagoli_train.txt')
    train_set = []
    sve_kategorije = []

    for line in glagoli_train_file:
        [gl, kategorija] = line.strip().split(',')
        kategorija = int(kategorija)
        train_set.append([gl,kategorija])
        if kategorija not in sve_kategorije:
            sve_kategorije.append(kategorija)

    glagoli_train_file.close()

    glagoli_val_file = open(path+'glagoli_validation.txt')
    val_set = []

    for line in glagoli_val_file:
        [gl, kategorija] = line.strip().split(',')
        val_set.append([gl,int(kategorija)])

    glagoli_val_file.close()

    glagoli_test_file = open(path+'glagoli_test.txt')
    test_set = []

    for line in glagoli_test_file:
        [gl, kategorija] = line.strip().split(',')
        test_set.append([gl,int(kategorija)])

    glagoli_test_file.close()
    return train_set, val_set, test_set, sve_kategorije

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
    
def getBatch(batch_size, train_data):
    random.shuffle(train_data)
    train_data_x = [get_pad_chars(words2charindices(t[0]),max_word_length, char_pad_token) for t in train_data]
    train_data_y = [t[1] for t in train_data]
    sindex = 0
    eindex = batch_size
    while eindex < len(train_data): 
        batch = [train_data_x[sindex: eindex], train_data_y[sindex: eindex]]
        temp = eindex
        eindex = eindex + batch_size
        sindex = temp
        yield batch
    
    if eindex >= len(train_data):
        batch = [train_data_x[sindex:], train_data_y[sindex:]]
        yield batch
        
def train_model(model, train, val, output_path, batch_size = 100, epochs=10, lr=0.001):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=lr)
    best_val_loss = 1e10
    bestModelSumLoss = 1e10
    bestModelValAcc = 0
    bestModelValRmse = 10
    for i in range(epochs):
        model.train()
        sum_loss = 0.0
        total = 0
        for j, [train_x, train_y] in enumerate(getBatch(batch_size, train)):
            x = torch.from_numpy(np.array(train_x)).long()
            y = torch.from_numpy(np.array(train_y)).long()
            y_pred = model(x)
            optimizer.zero_grad()
            loss = F.cross_entropy(y_pred, y)
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()*y.shape[0]
            total += y.shape[0]
        val_loss, val_acc, val_rmse = validation_metrics(model, val)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            bestModelSumLoss = sum_loss/total
            bestModelValAcc = val_acc
            bestModelValRmse = val_rmse
            torch.save(model.state_dict(), output_path)
        if i % 50 == 1:
            print("train loss %.3f, val loss %.3f, val accuracy %.3f, and val rmse %.3f" % (sum_loss/total, val_loss, val_acc, val_rmse))
    return bestModelSumLoss, best_val_loss, bestModelValAcc, bestModelValRmse

def validation_metrics (model, val_set):
    model.eval()
    correct = 0
    total = 0
    sum_loss = 0.0
    sum_rmse = 0.0
    for i, [val_x, val_y] in enumerate(getBatch(batch_size, val_set)):
        x = torch.from_numpy(np.array(val_x)).long()
        y = torch.from_numpy(np.array(val_y)).long()
        y_hat = model(x)
        loss = F.cross_entropy(y_hat, y)
        pred = torch.max(y_hat, 1)[1]
        correct += (pred == y).float().sum()
        total += y.shape[0]
        sum_loss += loss.item()*y.shape[0]
        sum_rmse += np.sqrt(mean_squared_error(pred, y.unsqueeze(-1)))*y.shape[0]
    return sum_loss/total, correct/total, sum_rmse/total


def train(model, train_set, val_set, batch_size, output_dir, epochs, lr):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = output_dir + "model.weights"
    train_model(model, train_set, val_set, output_path, batch_size = batch_size, epochs=epochs, lr=lr)
    return output_path
    
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
    
if __name__ == "__main__":
    #ucitaj podatke
    train_set, val_set, test_set, sve_kategorije = ucitajPodatke()
    
    #treniraj model
    emb_size=300
    out_size=len(sve_kategorije)
    batch_size=25
    output_dir="results/{:%Y%m%d_%H%M%S}/".format(datetime.now())
    epochs=500
    lr=0.005
    filter_sizes=[1,2,3,5]
    num_filters = 36
    dropout_rate=0.1
    model = CNN_Text(out_size, vocab_len, emb_size, weights_matrix, filter_sizes, num_filters, dropout_rate)
    model_weights_path = train(model, train_set, val_set, batch_size, output_dir, epochs, lr)
    print('naučene težine modela u spremljene u:', model_weights_path)
    
    
    #evaluiraj na skupu za testiranje
    #model_weights_path='results/20211116_130204/model.weights'
    evaluateOnTestSet(model, model_weights_path, test_set)
    
    glagoli = ['kroatiziram','peckam','promiješam','potpisujem','uneređujem','sperem']
    for gl in glagoli:
        #napravi heatmap
        heatmap(gl,model)
    