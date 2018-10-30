# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division

import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import warnings
from torch.nn.parameter import Parameter

warnings.simplefilter('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(0)

from utils import plot_stroke

strokes = np.load('data/strokes.npy', encoding='latin1')
stroke = strokes[0]

with open('data/sentences.txt') as f:
    texts = f.readlines()
    
texts = [a.split('\n')[0] for a in texts]

idx = 0
stroke = strokes[idx]
text = texts[idx]
plot_stroke(stroke)
print ('TEXT:', text)


l=[]
for i in strokes:
  l.append(i.shape[0])
  
max_seq = max(l)

import time
import math

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def get_data(ind=0, batch_size=1, max_seq=400):
  
  big_x,big_y = [],[]
  #print ('here')
  
  for k in range(batch_size):
    X = strokes[ind+k]
    if len(X)<max_seq:
        continue
    #print 'here'
    halt = int(len(X)/max_seq) + 1
    #print 'here',len(X),'halt',halt
    count  = 0
    for j in range(0,len(X),max_seq):
        y = []
        x = [[0,0,0]]
        if count == halt-1:
            for i in range(len(X)-max_seq, len(X)):
              y.append(X[i])
            y.append(X[i])
            x.extend(X[len(X)-max_seq:])
            big_x.append(x)
            big_y.append(y)
            continue
        else:
            for i in range(j,min(j+max_seq,len(X))):
                y.append(X[i])
            y.append(X[i])
            x.extend(X[j:min(j+max_seq,len(X))])
            y= np.array(y)
            big_x.append(x)
            big_y.append(y)
        count+=1
  X = np.array(big_x)
  y = np.array(big_y)
  return X,y

def get_strokes_text(ind=0, batch_size=1, min_seq=400, max_seq=800, max_text_len = 40):
  
  big_x,big_y,big_text = [],[],[]
  stroke_mask, text_mask, len_text = [],[],[]
  
  k = 0
  count = 0
  
  #for k in range(batch_size):
  while (count<batch_size):
    #print (k)
      
    X = strokes[ind+k]
    mask = np.ones(max_seq)

    if len(X)<min_seq:
        k+=1
        #k = k-2
        continue 

    x = []
    for i in range(min(len(X),max_seq)):
      #print ('here')
      #print i
      x.append(X[i].tolist())
    #print('here')
    if len(X)<max_seq:
        for i in range(max_seq-len(X)):
          x.append([0,0,0])
          
        mask[len(X):] = 0
    stroke_mask.append(mask)
    X = np.array(x)

    y = []
    for i in range(1,len(X)):
      y.append(X[i])
    y.append(X[len(X)-1])
    y= np.array(y)
    
    char_list = ' ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz,."\'?-!'
    char_to_code = {}
    code_to_char = {}
    c = 0
    for ch in char_list:
        char_to_code[ch] = c
        code_to_char[c] = ch
        c += 1
    text = texts[ind+k]
    text = text[0:min(max_text_len,len(text))]
    #print (text)
    #bre
    vectors = np.zeros((max_text_len, len(char_to_code)+1))
    #print (vectors.shape)
    #bre
    mask = np.ones(max_text_len)
    for p,q in enumerate(text):
        try:
            vectors[p][char_to_code[q]] = 1
        except:
            #print (q)
            vectors[p][-1] = 1
            continue
        
    if len(text) < max_text_len:
        mask[len(text):] = 0
    #big_y.append(vectors)
    text_mask.append(mask)
    len_text.append(len(text))

    big_x.append(X)
    big_y.append(y)
    big_text.append(vectors)
    
    k+=1
    count+=1
  #print('here')
  X = np.array(big_x)
  y = np.array(big_y)
  text = np.array(big_text)
  return [X, y, text], [stroke_mask,text_mask], len_text, char_to_code, code_to_char
  #print X.shape

def get_data_seq(ind=0, batch_size=1, max_seq=400):
  
  big_x,big_y = [],[]
  
  for k in range(batch_size):
    X = strokes[ind+k]
    for j in range(0,len(X),max_seq):
        y = []
        for i in range(j+1,min(j+max_seq,len(X))):
          y.append(X[i])
        y.append(X[i])

        y= np.array(y)
        big_x.append(X[j:min(j+max_seq,len(X))])
        big_y.append(y)
  X = np.array(big_x)
  y = np.array(big_y) 
  return X,y

import os

def save_checkpoint(epoch, model, optimizer, directory, \
                    filename='best.pt'):
    checkpoint=({'epoch': epoch+1,
    'model': model.state_dict(),
    'optimizer' : optimizer.state_dict()
    })
    try:
        torch.save(checkpoint, os.path.join(directory, filename))
        
    except:
        os.mkdir(directory)
        torch.save(checkpoint, os.path.join(directory, filename))


#enc = LSTM(150, input_shape=(n_timesteps_in, n_features), return_sequences=False)(inpx)
#data = RepeatVector(n_timesteps_in)(enc)
#dec = LSTM(150, input_shape=(n_timesteps_in, n_features), return_sequences=True)(data)
##dec = LSTM(150, input_shape=(n_timesteps_in, n_features), return_sequences=False)(dec)
#dec = TimeDistributed(Dense(n_features, activation='linear'))(dec)
##dec = Dense(n_features, activation='linear')(dec)
#out = dec
#model = Model(inpx, out)