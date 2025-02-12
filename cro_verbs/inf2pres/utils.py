import fasttext
import fasttext.util
import torch
import numpy as np

#učitaj fasttext vektore
def loadFastText(path):
    ft = fasttext.load_model(path+'cc.hr.300.bin')
    return ft

#napravi preslikavanje znakova u indekse
char_pad_token = 0
max_word_length = 20

class_present = {0: 'am', 1: 'im', 2:'jem', 3:'em'}
suffixes = ['am', 'im', 'jem', 'em']

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
def getWeightsMatrix(char_list, ft):
    matrix_len = len(char_list)+1
    weights_matrix = np.zeros((matrix_len, 300))

    for i, word in enumerate(char_list):
        weights_matrix[i+1] = ft.get_word_vector(word)

    weights_matrix = torch.from_numpy(weights_matrix).float()
    return weights_matrix

def loadData(path='',prefix='glagoli'):
    """Učitava glagole u skupove za treniranje, validaciju i testiranje.
    path treba biti u obliku putanja_do_fileova i obvezno na kraju /."""
    glagoli_train_file = open(path+f'{prefix}_train.txt')
    train_set = []
    all_categories = []

    for line in glagoli_train_file:
        [verb, category] = line.strip().split(',')
        category = int(category)
        train_set.append([verb,category])
        if category not in all_categories:
            all_categories.append(category)

    glagoli_train_file.close()

    glagoli_val_file = open(path+f'{prefix}_val.txt')
    val_set = []

    for line in glagoli_val_file:
        [verb, category] = line.strip().split(',')
        val_set.append([verb,int(category)])

    glagoli_val_file.close()

    glagoli_test_file = open(path+f'{prefix}_test.txt')
    test_set = []

    for line in glagoli_test_file:
        [verb, category] = line.strip().split(',')
        test_set.append([verb,int(category)])

    glagoli_test_file.close()
    return train_set, val_set, test_set, all_categories