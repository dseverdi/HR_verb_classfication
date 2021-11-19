import random
from utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from sklearn.metrics import mean_squared_error, f1_score

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
        val_loss, val_acc, val_rmse = validation_metrics(model, val, batch_size)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            bestModelSumLoss = sum_loss/total
            bestModelValAcc = val_acc
            bestModelValRmse = val_rmse
            torch.save(model.state_dict(), output_path)
        if i % 50 == 1:
            print("train loss %.3f, val loss %.3f, val accuracy %.3f, and val rmse %.3f" % (sum_loss/total, val_loss, val_acc, val_rmse))
    return bestModelSumLoss, best_val_loss, bestModelValAcc, bestModelValRmse

def validation_metrics (model, val_set, batch_size):
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