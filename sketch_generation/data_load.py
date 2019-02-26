"""
** deeplean-ai.com **
created by :: GauravBh1010tt
contact :: gauravbhatt.deeplearn@gmail.com
"""

import time
import math
import pandas as pd
import numpy as np
import torch
import os

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

def to_big_strokes(stroke, max_len=100):

    result = np.zeros((max_len, 5), dtype=float)
    l = len(stroke)
    assert l <= max_len
    result[0:l, 0:2] = stroke[:, 0:2]
    result[0:l, 3] = stroke[:, 2]
    result[0:l, 2] = 1 - result[0:l, 3]
    result[l:, 4] = 1
    
    return result
    
def to_normal_strokes(big_stroke):
  l = 0
  for i in range(len(big_stroke)):
    if big_stroke[i, 4] > 0:
      l = i
      break
  if l == 0:
    l = len(big_stroke)
  result = np.zeros((l, 3))
  result[:, 0:2] = big_stroke[0:l, 0:2]
  result[:, 2] = big_stroke[0:l, 3]
  return result


def purify(strokes, max_seq=200):
    data = []
    for seq in strokes:
        if seq.shape[0] <= max_seq and seq.shape[0] > 10:
            seq = np.minimum(seq, 1000)
            seq = np.maximum(seq, -1000)
            seq = np.array(seq, dtype=np.float32)
            data.append(seq)
    return data

def calculate_normalizing_scale_factor(strokes):
    data = []
    for i in range(len(strokes)):
        for j in range(len(strokes[i])):
            data.append(strokes[i][j, 0])
            data.append(strokes[i][j, 1])
    data = np.array(data)
    return np.std(data)

def normalize(strokes):
    data = []
    scale_factor = calculate_normalizing_scale_factor(strokes)
    for seq in strokes:
        seq[:, 0:2] /= scale_factor
        data.append(seq)
    return data

def get_batch_validation(data_enc, data_dec, batch_size):
    
    batch_idx = np.random.choice(len(data_enc),batch_size)    
    data_e, data_d =[], []
    for i in batch_idx:
        data_e.append(data_enc[i])
        data_d.append(data_dec[i])
    
    return data_e, data_d


def get_data(data_type='kanji', max_len=200):
    if data_type == 'kanji':
        raw_data = pd.read_pickle('sketch-rnn-datasets/kanji/kanji.cpkl')
    elif data_type == 'cat':
        raw_data = np.load('sketch-rnn-datasets/cat/cat.npz', encoding='latin1')['train']        
        
    all_len = [len(i)for i in raw_data]
    max_len = max(all_len)
    
    raw_data = purify(raw_data)
    
    data_enc = np.zeros((len(raw_data), max_len, 5))
    data_dec = np.zeros((len(raw_data), max_len+1, 5))
    data_enc[:,:,-1] = 1
    data_dec[:,:,-1] = 1
    data_dec[:,0,:] = [0,0,1,0,0]
    
    for i,j in enumerate(raw_data):
        big_strokes = to_big_strokes(j, max_len)
        data_enc[i] = big_strokes
        data_dec[i,1:,] = big_strokes
        
    data_enc = normalize(data_enc)
    data_dec = normalize(data_dec)
        
    return data_enc, data_dec, max_len

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