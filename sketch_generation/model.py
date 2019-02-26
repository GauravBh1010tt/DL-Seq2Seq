"""
** deeplean-ai.com **
created by :: GauravBh1010tt
contact :: gauravbhatt.deeplearn@gmail.com
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class encoder_skrnn(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_dec_size, dropout_p = 0.05, n_layers=1,\
                 bi_mode= 1, rnn_dir = 1, batch_size=1, latent_dim=64, device= None, cond_gen = False):
        super(encoder_skrnn, self).__init__()
        
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.n_layers=n_layers
        self.Nz = latent_dim
        self.device = device
        self.cond_gen = cond_gen
        self.rnn_dir = rnn_dir
        self.bi_mode = bi_mode
        
        self.rnn = nn.LSTM(input_size, hidden_size, n_layers, dropout=dropout_p,\
                           batch_first=True, bidirectional=rnn_dir==2)
        self.initial = nn.Linear(self.Nz, hidden_dec_size*2)
        
        self.mu = nn.Linear(hidden_size*bi_mode, self.Nz)
        self.sigma = nn.Linear(hidden_size*bi_mode, self.Nz)

    def forward(self, inp_enc, hidden):
        
        output, (hidden, cell_state) = self.rnn(inp_enc, hidden)

        if self.rnn_dir == 2:
            hidden_forward, hidden_backward = torch.split(hidden,1,0)
            if self.bi_mode == 1:
                hidden_cat = hidden_forward + hidden_backward
            else:
                hidden_cat = torch.cat([hidden_forward.squeeze(0), hidden_backward.squeeze(0)],1)
        else:
            if self.training:
                hidden = hidden_cat.squeeze(0)
            else:
                hidden = hidden_cat.view(1,self.hidden_size)

        mu = self.mu(hidden_cat)
        sigma_hat = self.sigma(hidden_cat)
        sigma = torch.exp(sigma_hat/2.)
        
        z = mu + sigma*torch.normal(torch.zeros(self.Nz,device=self.device),torch.ones(self.Nz, device=self.device))

        initial_params = torch.tanh(self.initial(z))
        
        (dec_hidden, dec_cell_state) = initial_params[:,:self.hidden_size].contiguous(), initial_params[:,self.hidden_size:].contiguous()

        return z, (dec_hidden.unsqueeze(0), dec_cell_state.unsqueeze(0)), mu, sigma_hat

    def initHidden(self):
        return (torch.zeros(self.n_layers*self.rnn_dir, self.batch_size, self.hidden_size, device=self.device), 
                torch.zeros(self.n_layers*self.rnn_dir, self.batch_size, self.hidden_size, device=self.device))

    
class decoder_skrnn(nn.Module):
    def __init__(self, input_size, hidden_size, num_gaussian, dropout_p = 0.05, n_layers=1,\
                 batch_size=1, latent_dim=64, device= None, cond_gen=False):
        super(decoder_skrnn, self).__init__()

        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.n_layers=n_layers
        self.num_gaussian = num_gaussian
        self.Nz = latent_dim
        self.cond_gen = cond_gen
        if cond_gen:
            self.rnn = nn.LSTM(self.Nz+input_size, hidden_size, n_layers, dropout=dropout_p, batch_first=True)
            self.initial = nn.Linear(self.Nz, 2*hidden_size)
        else:
            self.rnn = nn.LSTM(input_size, hidden_size, n_layers, dropout=dropout_p, batch_first=True)

        self.gmm = nn.Linear(hidden_size, num_gaussian*6+3)

    def forward(self, inp_dec, z, hidden):

        if self.cond_gen:
            
            if self.training:
                z_split = torch.stack([z.view(-1)]*(inp_dec.shape[1])).split(self.Nz,1)
                z_stack = torch.stack(z_split)                
            else:
                z_stack = z.unsqueeze(0)
            inp_dec = torch.cat([inp_dec, z_stack],dim=2)
            
        output, hidden = self.rnn(inp_dec, hidden)    

        if self.training:
            y_t = self.gmm(output.contiguous().view(-1, self.hidden_size))
        else:
            y_t = self.gmm(hidden[0].view(-1,self.hidden_size))

        y_t = torch.split(y_t, 6, dim=1)
        gmm_params = torch.stack(y_t[:-1])
        q_t = y_t[-1]
        
        pi_t, mu1_t, mu2_t, s1_t, s2_t, rho_t = torch.split(gmm_params,1,2)

        q_t = F.softmax(q_t, dim=1).view(-1,3)
        pi_t = F.softmax(pi_t.squeeze(2).transpose(0,1)).view(-1,self.num_gaussian)
        mu1_t = mu1_t.squeeze(2).transpose(0,1).view(-1,self.num_gaussian)
        mu2_t = mu2_t.squeeze(2).transpose(0,1).view(-1,self.num_gaussian)  
        s1_t = torch.exp(s1_t.squeeze(2).transpose(0,1)).view(-1,self.num_gaussian)
        s2_t = torch.exp(s2_t.squeeze(2).transpose(0,1)).view(-1,self.num_gaussian)       
        rho_t = torch.tanh(rho_t.squeeze(2).transpose(0,1)).view(-1,self.num_gaussian)
        
        params = [q_t, pi_t, mu1_t, mu2_t, s1_t, s2_t, rho_t]
        return params, hidden
    

def skrnn_loss(gmm_params, kl_params, data, mask=[], device =None):
       
    def get_2d_normal(x1,x2,mu1,mu2,s1,s2,rho):
      ##### implementing Eqn. 24 and 25 of the paper ###########
        norm1 = torch.sub(x1,mu1)
        norm2 = torch.sub(x2,mu2)
        s1s2 = torch.mul(s1,s2)
        z = torch.div(norm1**2,s1**2) + torch.div(norm2**2,s2**2) - 2*torch.div(torch.mul(rho, torch.mul(norm1,norm2)),s1s2)
        deno = 2*np.pi*s1s2*torch.sqrt(1-rho**2)
        numer = torch.exp(torch.div(-z,2*(1-rho**2)))
      ##########################################################
        return numer / deno
    
    eos = torch.stack([torch.Tensor([0,0,0,0,1])]*data.size()[0], device = device).unsqueeze(1)
    data = torch.cat([data, eos], 1) 
    
    target = data.view(-1, 5)
    x1, x2, eos = target[:,0].unsqueeze(1), target[:,1].unsqueeze(1), target[:,2:]

    q_t, pi_t = gmm_params[0], gmm_params[1]
    res = get_2d_normal(x1,x2,gmm_params[2],gmm_params[3],gmm_params[4],gmm_params[5],gmm_params[6])
    epsilon = torch.tensor(1e-5, dtype=torch.float)  # to prevent overflow

    Ls = torch.sum(torch.mul(pi_t,res),dim=1).unsqueeze(1)
    Ls = -torch.log(Ls + epsilon)
    mask_zero_out = 1-eos[:,2]
    
    Ls = torch.mul(Ls, mask_zero_out.view(-1,1))
    Lp = -torch.sum(eos*torch.log(q_t), -1).view(-1,1)
    Lr = Ls + Lp
    mu, sigma = kl_params[0], kl_params[1]

    L_kl = -(0.5)*torch.mean(1 + sigma - mu**2 - torch.exp(sigma))

    return Lr.mean(), L_kl


def skrnn_sample(encoder, decoder, hidden_size, latent_dim, start=[0,0,1,0,0], temperature=1.0, \
                  time_step=100, scale = 20, bi_mode= 1, random_state= 98, cond_gen=False, inp_enc=False, device=None):
    
    np.random.seed(random_state)
    encoder.train(False)
    decoder.train(False)

    def adjust_temp(pi_pdf, temp):
        pi_pdf = np.log(pi_pdf) / temp
        pi_pdf -= pi_pdf.max()
        pi_pdf = np.exp(pi_pdf)
        pi_pdf /= pi_pdf.sum()
        return pi_pdf
    
    def get_pi_id(x, dist, temp=1.0):            
        # implementing the cumulative index retrieval
        dist = adjust_temp(np.copy(dist.detach().cpu().numpy()), temp)

        N = dist.shape[0]
        accumulate = 0
        for i in range(0, N):
            accumulate += dist[i]
            if (accumulate >= x):
                return i
        return -1
    
    def sample_gaussian_2d(mu1, mu2, s1, s2, rho, temp=1.0):
        s1 *= temp * temp
        s2 *= temp * temp
        mean = [mu1, mu2]
        cov = [[s1 * s1, rho * s1 * s2], [rho * s1 * s2, s2 * s2]]
        x = np.random.multivariate_normal(mean, cov, 1)
        return x[0][0], x[0][1]
        
    
    prev_x = torch.tensor(start,dtype=torch.float, device=device)
    strokes = np.zeros((time_step, 5), dtype=np.float32)
    mixture_params = []
    
    hidden_enc = (torch.zeros(bi_mode, 1, hidden_size, device=device), torch.zeros(bi_mode, 1, hidden_size, device=device))
    hidden_dec = (torch.zeros(1, 1, hidden_size, device=device), torch.zeros(1, 1, hidden_size, device=device))
    
    if cond_gen:
        z, hidden_dec, mu, sigma = encoder(inp_enc, hidden_enc)
    else:
        z = torch.zeros(1, latent_dim, device=device)
    
    end_stroke = time_step
    
    for i in range(time_step):
        gmm_params, hidden_dec = decoder(prev_x.unsqueeze(0).unsqueeze(0), z, hidden_dec)
        q, pi, mu1, mu2, s1, s2, rho = gmm_params[0][0],gmm_params[1][0],gmm_params[2][0],gmm_params[3][0],gmm_params[4][0],gmm_params[5][0],gmm_params[6][0]
        
        idx = get_pi_id(np.random.random(), pi, temperature)        
        eos_id = get_pi_id(np.random.random(), q, temperature)
        
        eos = [0, 0, 0]
        eos[eos_id] = 1
    
        next_x1, next_x2 = sample_gaussian_2d(mu1[idx].detach().cpu().numpy(), mu2[idx].detach().cpu().numpy(), 
                            s1[idx].detach().cpu().numpy(), s2[idx].detach().cpu().numpy(), 
                            rho[idx].detach().cpu().numpy())
        
        mixture_params.append([float(mu1[idx].detach().cpu()),float(mu2[idx].detach().cpu()), float(s1[idx].detach().cpu()), 
                            float(s2[idx].detach().cpu()), float(rho[idx].detach().cpu()), q])
        strokes[i, :] = [next_x1, next_x2, eos[0], eos[1], eos[2]]
        if eos[-1] == 1:
            end_stroke = i+1
            break
        prev_x[0], prev_x[1], prev_x[2], prev_x[3], prev_x[4] = next_x1, next_x2, eos[0], eos[1], eos[2]
        
    mix_params = np.array(mixture_params)
    return strokes[:end_stroke,[0,1,3]], mix_params