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
torch.cuda.set_device(1)

from utils import plot_stroke

strokes = np.load('data/strokes.npy', encoding='latin1')
stroke = strokes[0]

with open('data/sentences.txt') as f:
    texts = f.readlines()

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

def get_data(ind=0, batch_size=5):
  
  big_x,big_y = [],[]
  
  for k in range(batch_size):
    X = strokes[ind+k]

    x = []
    for i in X:
      #print i
      x.append(i.tolist())
    for i in range(max_seq-len(X)):
      x.append([0,0,0])
    X = np.array(x)

    y = []
    for i in range(1,len(X)):
      y.append(X[i])
    y.append(X[len(X)-1])

    y= np.array(y)

    #X = X.reshape((1, X.shape[0], X.shape[1]))
    #y = y.reshape((1, y.shape[0], y.shape[1]))

    big_x.append(X)
    big_y.append(y)
  X = np.array(big_x)
  y = np.array(big_y)
  #print X.shape
  #print y.shape
  
  #X = X.reshape((batch_size, X.shape[0], X.shape[1]))
  #y = y.reshape((batch_size, y.shape[0], y.shape[1]))
  
  return X,y

def get_data_window(ind=0, steps=20):
  
  X = strokes[ind].tolist()
  X.append([-2,-1,-1])
  next_chars_f = []
  sentences_f = []
  
  if True:
      temp = [[0.0,0.0,0.0] for i in range(steps)]
      flag = False
      for word in X:
          temp.remove(temp[0])
          temp.append(word)
          if flag == True:
              next_chars_f.append(word)
          if word!=[-2,-1,-1]:
              temp1 = []
              for i in temp:
                  temp1.append(i)
              sentences_f.append(temp1)
          flag = True
  next_chars_f[-1] = [0.0,0.0,0.0]
  return np.array(sentences_f), np.array(next_chars_f)
          
steps = max_seq
n_features = 3
n_timesteps_in = steps
n_timesteps_out = max_seq
batch_size = 10


#enc = LSTM(150, input_shape=(n_timesteps_in, n_features), return_sequences=False)(inpx)
#data = RepeatVector(n_timesteps_in)(enc)
#dec = LSTM(150, input_shape=(n_timesteps_in, n_features), return_sequences=True)(data)
##dec = LSTM(150, input_shape=(n_timesteps_in, n_features), return_sequences=False)(dec)
#dec = TimeDistributed(Dense(n_features, activation='linear'))(dec)
##dec = Dense(n_features, activation='linear')(dec)
#out = dec
#model = Model(inpx, out)