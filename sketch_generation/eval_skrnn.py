"""
** deeplean-ai.com **
created by :: GauravBh1010tt
contact :: gauravbhatt.deeplearn@gmail.com
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

from data_load import get_data
from model import encoder_skrnn, decoder_skrnn

def draw_image(a, color=None, save=False, save_dir=None):
    x = np.cumsum(a[:,0], 0)
    y = np.cumsum(a[:,1], 0)
    
    stroke = np.stack([x,y],1)
    cuts = np.where(a[:,2]>0)[0] + 1
    
    strokes = np.split(stroke[:-1],cuts)
    
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    
    for s in strokes:
        plt.plot(s[:,0],-s[:,1],color=color)
    if save:
        plt.savefig(save_dir+str(np.random.randint(0,5000)))
    plt.show()
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_pretrained_uncond(data_type):
    ####################################################
    # default parameters, do not change
    hidden_enc_dim = 256
    hidden_dec_dim = 256
    n_layers = 1
    num_gaussian = 20 
    dropout_p = 0.2
    batch_size = 50
    latent_dim = 64 

    rnn_dir = 2 # 1 for unidirection,  2 for bi-direction
    bi_mode = 2 # bidirectional mode:- 1 for addition 2 for concatenation
    cond_gen = False

    if not cond_gen:
        rnn_dir = 1
        bi_mode = 1

    ####################################################
    
    encoder = encoder_skrnn(input_size = 5, hidden_size = hidden_enc_dim, hidden_dec_size=hidden_dec_dim,\
                        dropout_p = dropout_p,n_layers = n_layers, batch_size = batch_size, latent_dim = latent_dim,\
                        device = device, cond_gen= cond_gen, bi_mode= bi_mode, rnn_dir = rnn_dir).to(device)
    
    decoder = decoder_skrnn(input_size = 5, hidden_size = hidden_dec_dim, num_gaussian = num_gaussian,\
                            dropout_p = dropout_p, n_layers = n_layers, batch_size = batch_size,\
                            latent_dim = latent_dim, device = device, cond_gen= cond_gen).to(device)

    if data_type == 'cat':
        encoder.load_state_dict(torch.load('saved_model/UncondEnc_cat.pt')['model'])
        decoder.load_state_dict(torch.load('saved_model/UncondDec_cat.pt')['model'])
        data_enc, data_dec, max_seq_len = get_data(data_type='cat')
    else:
        encoder.load_state_dict(torch.load('saved_model/UncondEnc_kanji.pt')['model'])
        decoder.load_state_dict(torch.load('saved_model/UncondDec_kanji.pt')['model'])
        data_enc, data_dec, max_seq_len = get_data(data_type='kanji')
    return encoder, decoder, hidden_enc_dim, latent_dim, max_seq_len, cond_gen, bi_mode, device

def load_pretrained_congen(data_type):
    ####################################################
    # default parameters, do not change
    hidden_enc_dim = 256
    hidden_dec_dim = 256
    n_layers = 1
    num_gaussian = 20 
    dropout_p = 0.2
    batch_size = 50
    latent_dim = 64 

    rnn_dir = 2 # 1 for unidirection,  2 for bi-direction
    bi_mode = 2 # bidirectional mode:- 1 for addition 2 for concatenation
    cond_gen = True

    if not cond_gen:
        rnn_dir = 1
        bi_mode = 1

    ####################################################
    
    encoder = encoder_skrnn(input_size = 5, hidden_size = hidden_enc_dim, hidden_dec_size=hidden_dec_dim,\
                        dropout_p = dropout_p,n_layers = n_layers, batch_size = batch_size, latent_dim = latent_dim,\
                        device = device, cond_gen= cond_gen, bi_mode= bi_mode, rnn_dir = rnn_dir).to(device)
    
    decoder = decoder_skrnn(input_size = 5, hidden_size = hidden_dec_dim, num_gaussian = num_gaussian,\
                            dropout_p = dropout_p, n_layers = n_layers, batch_size = batch_size,\
                            latent_dim = latent_dim, device = device, cond_gen= cond_gen).to(device)
    if data_type == 'cat':
        encoder.load_state_dict(torch.load('saved_model/condEnc_cat.pt')['model'])
        decoder.load_state_dict(torch.load('saved_model/condDec_cat.pt')['model'])
        data_enc, data_dec, max_seq_len = get_data(data_type='cat')
    else:
        encoder.load_state_dict(torch.load('saved_model/condEnc_kanji.pt')['model'])
        decoder.load_state_dict(torch.load('saved_model/condDec_kanji.pt')['model'])
        data_enc, data_dec, max_seq_len = get_data(data_type='kanji')
    return encoder, decoder, hidden_dec_dim, latent_dim, max_seq_len, cond_gen, bi_mode, device
