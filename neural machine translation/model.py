"""
** deeplean-ai.com **
created by :: GauravBh1010tt
contact :: gauravbhatt.deeplearn@gmail.com
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, bi_dir=True):
        super(EncoderRNN, self).__init__()
        if bi_dir == True:
            self.bi = 2
        else:
            self.bi = 1
        self.hidden_size = hidden_size
        self.n_layers=n_layers

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, batch_first=False, bidirectional = bi_dir)

    def forward(self, input, hidden):
        seq_len = len(input)
        embedded = self.embedding(input).view(seq_len, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(self.n_layers*self.bi, 1, self.hidden_size, device=device)
    

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, attn_type = 'gen', n_layers=1, dropout_p=0.1, bi_dir=True):
        super(AttnDecoderRNN, self).__init__()
        if bi_dir == True:
            self.bi = 2
        else:
            self.bi = 1
        self.hidden_size = hidden_size
        self.maxlen = 0
        self.attn_type = attn_type

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,  dropout=dropout_p, 
                          batch_first=False, bidirectional = bi_dir)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax()        
        self.wd = nn.Linear(hidden_size*2*self.bi, hidden_size)
        self.ws = nn.Linear(hidden_size*self.bi, hidden_size*self.bi)
        self.out2 = nn.Linear(hidden_size*2, output_size)
        
    def score(self, dec_st, enc_st):
        if self.attn_type == 'dot':
            return torch.mm(dec_st, enc_st.t())
        elif self.attn_type == 'gen':
            return torch.mm(self.ws(enc_st), dec_st.t())
        
    def align(self, dec_state, enc_states):  
        alpha = self.score(dec_state, enc_states)
        alpha = torch.softmax(alpha.t(), dim=1)
        return torch.mm(alpha, enc_states), alpha
        
    def forward(self, input, hidden, encoder_input):
        
        output = self.embedding(input).view(1,1,-1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        dec_out = output[0]
        c_t, attn_wt = self.align(dec_out, encoder_input)
        h_t = F.tanh(self.wd(torch.cat((c_t,dec_out),dim=1)))
        out_word = self.softmax(self.out(h_t))
        return out_word, hidden, attn_wt
