"""
** deeplean-ai.com **
created by :: GauravBh1010tt
contact :: gauravbhatt.deeplearn@gmail.com
"""

from __future__ import unicode_literals, print_function, division

import time
import torch
import numpy as np
import warnings
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from data_load import timeSince, get_data, save_checkpoint
from model import encoder_skrnn, decoder_skrnn, skrnn_loss, skrnn_sample
from eval_skrnn import draw_image

warnings.simplefilter('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''if using multiple GPUs use this option'''
#torch.cuda.set_device(1)

hidden_enc_dim = 256
hidden_dec_dim = 256
n_layers = 1
num_gaussian = 20 
dropout_p = 0.2
batch_size = 50
latent_dim = 64 
weight_kl = 0.5
kl_tolerance = 0.2
eta_min = 0.01
R_step =  0.99995
learning_rate = 0.0008
clip = 1.
epochs = 60

print_every = batch_size*200 # print loss after this much iteration, change the multiplier aacording to dataset
plot_every = 1 # plot the strokes using current trained model

rnn_dir = 2 # 1 for unidirection,  2 for bi-direction
bi_mode = 2 # bidirectional mode:- 1 for addition 2 for concatenation
cond_gen = False # use either unconditional or conditional generation
data_type = 'cat' # 'cat' and 'kanji'

if not cond_gen:
    weight_kl = 0.0
    rnn_dir = 1
    bi_mode = 1

encoder = encoder_skrnn(input_size = 5, hidden_size = hidden_enc_dim, hidden_dec_size=hidden_dec_dim,\
                    dropout_p = dropout_p,n_layers = n_layers, batch_size = batch_size, latent_dim = latent_dim,\
                    device = device, cond_gen= cond_gen, bi_mode= bi_mode, rnn_dir = rnn_dir).to(device)

decoder = decoder_skrnn(input_size = 5, hidden_size = hidden_dec_dim, num_gaussian = num_gaussian,\
                        dropout_p = dropout_p, n_layers = n_layers, batch_size = batch_size,\
                        latent_dim = latent_dim, device = device, cond_gen= cond_gen).to(device)

encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)

try:
    aaaaa = data_enc[0]
except:
    data_enc, data_dec, max_seq_len = get_data(data_type=data_type)

num_mini_batch = len(data_dec) - (len(data_dec) % batch_size)

for epoch in range(epochs):
    start = time.time()
    
    print_loss_total, print_LRloss_total, print_KLloss_total = [], [], []
    print ('ep  t<taken  left>    data seen    tr_err  tr_LR   tr_LKL')
    
    for batch_id in range(0, num_mini_batch, batch_size):
        
        encoder.train()
        decoder.train()

        hidden_enc = hidden_dec = encoder.initHidden()
        inp_enc = torch.tensor(data_enc[batch_id:batch_id+batch_size], dtype=torch.float, device=device)
        inp_dec = torch.tensor(data_dec[batch_id:batch_id+batch_size], dtype=torch.float, device=device)
        
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        
        if cond_gen:   
            z, hidden_dec, mu, sigma = encoder(inp_enc, hidden_enc)  
        else:
            z = mu = sigma = torch.zeros(batch_size, latent_dim, device=device)

        gmm_params, _ = decoder(inp_dec, z, hidden_dec)
           
        loss_lr, loss_kl = skrnn_loss(gmm_params, [mu,sigma], inp_dec[:,1:,], device=device)
        loss_kl = torch.max(loss_kl, torch.tensor(kl_tolerance, dtype=torch.float, device=device))
        
        eta_step = 1 - (1-eta_min)*R_step
        
        loss = loss_lr + weight_kl*eta_step*loss_kl

        loss.backward()         
        torch.nn.utils.clip_grad_norm(encoder.parameters(), clip)
        torch.nn.utils.clip_grad_norm(decoder.parameters(), clip)            
        encoder_optimizer.step()
        decoder_optimizer.step()
        
        print_loss_total.append(loss.item())
        print_LRloss_total.append(loss_lr.item())
        print_KLloss_total.append(weight_kl*eta_step*loss_kl.item())
        
        if batch_id % print_every == 0 and batch_id>0:

            print('%d   %s  (%d %d%%)  %.4f  %.4f  %.4f' % (epoch,timeSince(start, batch_id / num_mini_batch),
                                           batch_id, batch_id / num_mini_batch * 100, np.mean(print_loss_total),
                                           np.mean(print_LRloss_total), np.mean(print_KLloss_total)))

    if epoch % plot_every == 0 and epoch>0:
        if not cond_gen:
            strokes, mix_params = skrnn_sample(encoder, decoder, hidden_enc_dim, latent_dim, time_step=max_seq_len, 
                                               cond_gen=cond_gen, device=device, bi_mode= bi_mode)
        else:
            enc_rnd = inp_enc[np.random.randint(0,batch_size)].unsqueeze(0)
            strokes, mix_params = skrnn_sample(encoder, decoder, hidden_dec_dim, latent_dim, inp_enc = enc_rnd,
                                               time_step=max_seq_len, cond_gen=cond_gen,bi_mode= bi_mode, device=device)
        draw_image(strokes)
    
if cond_gen:
    fname_enc = 'CondEnc_'+data_type+'.pt'
    fname_dec = 'CondDec_'+data_type+'.pt'
    save_checkpoint(epoch, encoder, encoder_optimizer, 'saved_model', \
                        filename = fname_enc)
    save_checkpoint(epoch, decoder, decoder_optimizer, 'saved_model', \
                        filename = fname_dec)
else:
    fname_enc = 'UncondEnc_'+data_type+'.pt'
    fname_dec = 'UncondDec_'+data_type+'.pt'
    save_checkpoint(epoch, encoder, encoder_optimizer, 'saved_model', \
                        filename = fname_enc)
    save_checkpoint(epoch, decoder, decoder_optimizer, 'saved_model', \
                        filename = fname_dec)