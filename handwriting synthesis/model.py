"""
** deeplean-ai.com **
created by :: GauravBh1010tt
contact :: gauravbhatt.deeplearn@gmail.com
"""

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class model_uncond(nn.Module):
    def __init__(self, input_size, hidden_size, num_gaussian, dropout_p = 0.05, n_layers=1,\
                 rnn_type = 2, batch_size=1, bi_dir=True, bi_mode = 2):
        super(model_uncond, self).__init__()
        
        if bi_dir == True:
            self.bi = 2
        else:
            self.bi = 1
            
        self.bi_mode = bi_mode
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.n_layers=n_layers
        self.num_gaussian = num_gaussian
        if rnn_type == 1:
            self.rnn1 = nn.GRU(input_size, hidden_size, n_layers, dropout=dropout_p, batch_first=True, bidirectional = True)
            self.rnn2 = nn.GRU(hidden_size*self.bi_mode+input_size, hidden_size, n_layers, dropout=dropout_p, batch_first=True, bidirectional = True)
        else:
            self.rnn1 = nn.LSTM(input_size, hidden_size, n_layers, dropout=dropout_p, batch_first=True, bidirectional = True)
            self.rnn2 = nn.LSTM(hidden_size*self.bi_mode+input_size, hidden_size, n_layers, dropout=dropout_p, batch_first=True, bidirectional = True)
        self.mdn = nn.Linear(hidden_size*2*self.bi_mode, num_gaussian*6+1)

    def forward(self, inp, hidden1, hidden2):

        if len(inp.size()) == 2:
            embed=inp.unsqueeze(1)
        else:
            embed = inp
        
        output1, hidden1 = self.rnn1(embed, hidden1)
        if self.bi_mode == 1:
            output1 = output1[:,:,0:self.hidden_size] + output1[:,:,self.hidden_size:]            

        inp_skip = torch.cat([output1, embed], dim=-1)  # implementing skip connection 
        output2, hidden2 = self.rnn2(inp_skip, hidden2)        
        if self.bi_mode == 1:
            output2 = output2[:,:,0:self.hidden_size] + output2[:,:,self.hidden_size:]  
        
        output = torch.cat([output1,output2], dim=-1)
        
        ##### implementing Eqn. 17 to 22 of the paper ###########
        y_t = self.mdn(output.squeeze(1))
        e_t = y_t[:,0:1]
        
        pi_t, mu1_t, mu2_t, s1_t, s2_t, rho_t = torch.split(y_t[:,1:], self.num_gaussian, dim=1)
        e_t = F.sigmoid(e_t)
        pi_t = F.softmax(pi_t)
        s1_t, s2_t = torch.exp(s1_t), torch.exp(s2_t)
        rho_t = torch.tanh(rho_t)
        #######################################################
        
        mdn_params = [e_t, pi_t, mu1_t, mu2_t, s1_t, s2_t, rho_t]
        return mdn_params, hidden1, hidden2

    def initHidden(self):
        return torch.zeros(self.n_layers*self.bi, self.batch_size, self.hidden_size, device=device)
    
    def initLHidden(self):
        return (torch.zeros(self.n_layers*self.bi, self.batch_size, self.hidden_size, device=device), 
                torch.zeros(self.n_layers*self.bi, self.batch_size, self.hidden_size, device=device))
    

class model_congen(nn.Module):
    def __init__(self, input_size, hidden_size, num_gaussian=20, char_vec_len=61, dropout_p=0.05,\
                 rnn_type=2, n_layers=1, batch_size=1, bi_dir=True, bi_mode=2, num_attn_gaussian=10):
        super(model_congen, self).__init__()
        
        if bi_dir == True:
            self.bi = 2
        else:
            self.bi = 1
            
        self.bi_mode = bi_mode
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.n_layers=n_layers
        self.num_gaussian = num_gaussian
        self.num_attn_gaussian = num_attn_gaussian
        self.char_vec_len = char_vec_len
        
        if rnn_type == 1:
            self.rnn1 = nn.GRU(input_size, hidden_size, n_layers, dropout=dropout_p, batch_first=True, bidirectional = True)
            self.rnn2 = nn.GRU(hidden_size*self.bi_mode+input_size, hidden_size, n_layers, dropout=dropout_p, batch_first=True, bidirectional = True)
        else:
            self.rnn1 = nn.LSTM(input_size+char_vec_len, hidden_size, n_layers, dropout=dropout_p, batch_first=True, bidirectional = True)
            self.rnn2 = nn.LSTM(hidden_size*self.bi_mode+input_size+char_vec_len, hidden_size, n_layers, dropout=dropout_p, batch_first=True, bidirectional = True)
        self.mdn = nn.Linear(hidden_size*2*self.bi_mode, num_gaussian*6+1)
        self.window = nn.Linear(hidden_size*self.bi_mode, num_attn_gaussian*3)

    def forward(self, inp, char_vec, old_k, old_w, text_len, hidden1, hidden2, bias = 0):
        
        if len(inp.size()) == 2:
            inp=inp.unsqueeze(1)
            
        embed = torch.cat([inp, old_w], dim=-1)   # adding attention window to the input of rnn
        
        output1, hidden1 = self.rnn1(embed, hidden1)
        if self.bi_mode == 1:
            output1 = output1[:,:,0:self.hidden_size] + output1[:,:,self.hidden_size:]            
        
        ##### implementing Eqn. 48 - 51 of the paper ###########
        abk_t = self.window(output1.squeeze(1)).exp()
        a_t, b_t, k_t = abk_t.split(self.num_attn_gaussian, dim=1)
        k_t = old_k + k_t        
        #######################################################
        
        
        ##### implementing Eqn. 46 and 47 of the paper ###########
        u = torch.linspace(1, char_vec.shape[1], char_vec.shape[1], device=device)
        phi_bku = torch.exp(torch.mul(torch.sub(k_t.unsqueeze(2).repeat((1,1,len(u))),u)**2,
                                      -b_t.unsqueeze(2)))
        phi = torch.sum(torch.mul(a_t.unsqueeze(2),phi_bku),dim=1)* (char_vec.shape[1]/text_len)
        win_t = torch.sum(torch.mul(phi.unsqueeze(2), char_vec),dim=1)
        ##########################################################
        
        
        inp_skip = torch.cat([output1, inp, win_t.unsqueeze(1)], dim=-1)  # implementing skip connection
        output2, hidden2 = self.rnn2(inp_skip, hidden2)        
        if self.bi_mode == 1:
            output2 = output2[:,:,0:self.hidden_size] + output2[:,:,self.hidden_size:]          
        output = torch.cat([output1,output2], dim=-1)

        ##### implementing Eqn. 17 to 22 of the paper ###########
        y_t = self.mdn(output.squeeze(1))  

        e_t = y_t[:,0:1]
        pi_t, mu1_t, mu2_t, s1_t, s2_t, rho_t = torch.split(y_t[:,1:], self.num_gaussian, dim=1)
        e_t = F.sigmoid(e_t)
        pi_t = F.softmax(pi_t*(1+bias))  # bias would be used during inference
        s1_t, s2_t = torch.exp(s1_t), torch.exp(s2_t)
        rho_t = torch.tanh(rho_t)
        ##########################################################
        
        mdn_params = [e_t, pi_t, mu1_t, mu2_t, s1_t, s2_t, rho_t, phi, win_t, k_t]       
        return mdn_params, hidden1, hidden2

    def initHidden(self):
        return torch.zeros(self.n_layers*self.bi, self.batch_size, self.hidden_size, device=device)
    
    def initLHidden(self):
        return (torch.zeros(self.n_layers*self.bi, self.batch_size, self.hidden_size, device=device), 
                torch.zeros(self.n_layers*self.bi, self.batch_size, self.hidden_size, device=device))
        
def mdn_loss(mdn_params, data, mask=[]):

    def get_2d_normal(x1,x2,mu1,mu2,s1,s2,rho):
      ##### implementing Eqn. 24 and 25 of the paper ###########
      norm1 = torch.sub(x1.view(-1,1),mu1)
      norm2 = torch.sub(x2.view(-1,1),mu2)
      s1s2 = torch.mul(s1,s2)
      z = torch.div(norm1**2,s1**2) + torch.div(norm2**2,s2**2) - 2*torch.div(torch.mul(rho, torch.mul(norm1,norm2)),s1s2)
      deno = 2*np.pi*s1s2*torch.sqrt(1-rho**2)
      numer = torch.exp(torch.div(-z,2*(1-rho**2)))
      ##########################################################
      return numer / deno

    eos, x1, x2 = data[:,0], data[:,1], data[:,2]
    e_t, pi_t = mdn_params[0], mdn_params[1]
    res = get_2d_normal(x1,x2,mdn_params[2],mdn_params[3],mdn_params[4],mdn_params[5],mdn_params[6])
    
    epsilon = torch.tensor(1e-20, dtype=torch.float, device=device)  # to prevent overflow

    res1 = torch.sum(torch.mul(pi_t,res),dim=1)
    res1 = -torch.log(torch.max(res1,epsilon))
    res2 = torch.mul(eos, e_t.t()) + torch.mul(1-eos,1-e_t.t())
    res2 = -torch.log(res2)
    
    if len(mask)!=0:        # using masking in case of padding
        res1 = torch.mul(res1,mask)
        res2 = torch.mul(res2,mask)
    return torch.sum(res1+res2)

def get_pi_id(x, dist):    
    # implementing the cumulative index retrieval
    N = dist.shape[0]
    accumulate = 0
    for i in range(0, N):
        accumulate += dist[i]
        if (accumulate >= x):
            return i
    return -1

def sample_gaussian_2d(mu1, mu2, s1, s2, rho):
    mean = [mu1, mu2]
    cov = [[s1 * s1, rho * s1 * s2], [rho * s1 * s2, s2 * s2]]
    x = np.random.multivariate_normal(mean, cov, 1)
    return x[0][0], x[0][1]

def sample_congen(lr_model, text, char_to_vec, hidden_size, start=[0,0,0], time_step=1000, scale = 50,\
                rnn_type = 2, bias1 = 1, bias2 = 1, num_attn_gaussian = 10, bi_dir=True, random_state= 98):
    np.random.seed(random_state)
    
    if bi_dir == True:
        bi = 2
    else:
        bi = 1
    
    prev_x = torch.tensor(start,dtype=torch.float, device=device)
    prev_x[0] = 1
    strokes = np.zeros((time_step, 3), dtype=np.float32)
    old_k = torch.zeros((1,num_attn_gaussian), dtype=torch.float, device=device)
    text_len = torch.tensor([[len(text)]], dtype=torch.float, device=device)
    
    vectors = np.zeros((len(text), len(char_to_vec)+1))

    for p,q in enumerate(text):
        try:
            vectors[p][char_to_vec[q]] = 1
        except:
            vectors[p][-1] = 1
            continue    
        
    text_tensor = torch.tensor(vectors, dtype=torch.float, device=device)
    old_w = text_tensor.narrow(0,0,1).unsqueeze(0)

    phis, win = [],[]
    count = 0
    stop = False
    
    mixture_params = []
    if rnn_type == 1:
        hidden1 =  torch.zeros(bi, 1, hidden_size, device=device)
        hidden2 =  torch.zeros(bi, 1, hidden_size, device=device)
    else:
        hidden1 = (torch.zeros(bi, 1, hidden_size, device=device), torch.zeros(bi, 1, hidden_size, device=device))
        hidden2 = (torch.zeros(bi, 1, hidden_size, device=device), torch.zeros(bi, 1, hidden_size, device=device))
    
    for i in range(time_step):
        mdn_params, hidden1,hidden2 = lr_model(prev_x.unsqueeze(0),text_tensor.unsqueeze(0),
                                               old_k, old_w, text_len, hidden1, hidden2, bias1)
        old_k = mdn_params[-1]
        old_w = mdn_params[-2].unsqueeze(1)
        idx = get_pi_id(np.random.random(), mdn_params[1][0])
        eos = 1 if np.random.random() < mdn_params[0][0] else 0
        
        log_s1 = mdn_params[4][0][idx].log() - bias2
        log_s2 = mdn_params[5][0][idx].log() - bias2
        log_s1 = log_s1.exp()
        log_s2 = log_s2.exp()
    
        next_x1, next_x2 = sample_gaussian_2d(mdn_params[2][0][idx].detach().cpu().numpy(), mdn_params[3][0][idx].detach().cpu().numpy(), 
                            log_s1.detach().cpu().numpy(), log_s2.detach().cpu().numpy(), 
                            mdn_params[6][0][idx].detach().cpu().numpy())
        
        strokes[i, :] = [eos, next_x1, next_x2]
        mixture_params.append([float(mdn_params[2][0][idx].detach().cpu()), 
                               float(mdn_params[3][0][idx].detach().cpu()), 
                            float(log_s1.detach().cpu()), float(log_s2.detach().cpu()), 
                            float(mdn_params[6][0][idx].detach().cpu()), eos])
        #prev_x = np.zeros((1, 1, 3), dtype=np.float32)
        prev_x[0],prev_x[1],prev_x[2] = eos, next_x1, next_x2
        
        phis.append(mdn_params[-3].squeeze(0))
        win.append(mdn_params[-2])
        old_phi = mdn_params[-3].squeeze(0)
        old_phi = old_phi.data.cpu().numpy()     
        
        if count >=40 and np.max(old_phi) == old_phi[-1]:
            stop = True
        else:
            count += 1
        
    phis = torch.stack(phis).data.cpu().numpy().T
    win = torch.stack(win).data.cpu().numpy().T
    #attention_plot(phis)    
    mix_params = np.array(mixture_params)
    mix_params[:,:2] = np.cumsum(mix_params[:,:2], axis=0)
    
    #phi_window_plots(phis,win.squeeze(1))
    #gauss_params_plot()
    strokes[:, 1:3] *= scale  # scaling the output strokes
    return strokes[:count+scale,:], mix_params[:count+scale,:],\
             phis[:,:count+scale], win.squeeze(1)[:,:count+scale]

def sample_uncond(lr_model, hidden_size, start=[0,0,0], rnn_type =2, \
                  time_step=1000, scale = 20, bi_dir =True, random_state= np.random.randint(0,10000)):
    np.random.seed(random_state)
        
    if bi_dir == True:
        bi = 2
    else:
        bi = 1
    
    prev_x = torch.tensor(start,dtype=torch.float, device=device)
    prev_x[0] = 1
    strokes = np.zeros((time_step, 3), dtype=np.float32)
    mixture_params = []
    if rnn_type == 1:
        hidden1 =  torch.zeros(bi, 1, hidden_size, device=device)
        hidden2 =  torch.zeros(bi, 1, hidden_size, device=device)
    else:
        hidden1 = (torch.zeros(bi, 1, hidden_size, device=device), torch.zeros(bi, 1, hidden_size, device=device))
        hidden2 = (torch.zeros(bi, 1, hidden_size, device=device), torch.zeros(bi, 1, hidden_size, device=device))
    
    for i in range(time_step):
        mdn_params, hidden1,hidden2 = lr_model(prev_x.unsqueeze(0), hidden1,hidden2)
        idx = get_pi_id(np.random.random(), mdn_params[1][0])
        eos = 1 if np.random.random() < mdn_params[0][0] else 0
    
        next_x1, next_x2 = sample_gaussian_2d(mdn_params[2][0][idx].detach().cpu().numpy(), mdn_params[3][0][idx].detach().cpu().numpy(), 
                            mdn_params[4][0][idx].detach().cpu().numpy(), mdn_params[5][0][idx].detach().cpu().numpy(), 
                            mdn_params[6][0][idx].detach().cpu().numpy())
        
        mixture_params.append([float(mdn_params[2][0][idx].detach().cpu()), 
                               float(mdn_params[3][0][idx].detach().cpu()), 
                            float(mdn_params[4][0][idx].detach().cpu()), 
                            float(mdn_params[5][0][idx].detach().cpu()), 
                            float(mdn_params[6][0][idx].detach().cpu()), eos])
        
        strokes[i, :] = [eos, next_x1, next_x2]
        #prev_x = np.zeros((1, 1, 3), dtype=np.float32)
        prev_x[0],prev_x[1],prev_x[2] = eos, next_x1, next_x2
        
    strokes[:, 1:3] *= scale
    mix_params = np.array(mixture_params)
    mix_params[:,:2] = np.cumsum(mix_params[:,:2], axis=0)
    return strokes, mix_params
        
def scheduled_sample(lr_model, hidden_size, prev_x, batch_size = 50,  bi_dir=True, rnn_type=2, random_state=123):
    #np.random.seed(random_state)
    
    if bi_dir == True:
        bi = 2
    else:
        bi = 1
        
    def get_pi_batch(x, dist):
        dist = dist.cumsum(1)
        c = x.unsqueeze(0).t() - dist
        c[c<0] = 0
        return c.argmin(1)

    def sample_gaussian_2d_batch(mu1, mu2, s1, s2, rho):
        mean = torch.tensor([mu1, mu2], device=device)
        cov = torch.tensor([[s1 * s1, rho * s1 * s2], [rho * s1 * s2, s2 * s2]], device=device)
        #x = np.random.multivariate_normal(mean, cov, 1)
        mod = torch.distributions.multivariate_normal.MultivariateNormal(mean,cov)
        return mod.sample()

    next_x = torch.zeros((batch_size, 3),dtype=torch.float, device=device)
    #prev_x[:,0] = 1
    if rnn_type == 1:
        hidden1 =  torch.zeros(bi, batch_size, hidden_size, device=device)
        hidden2 =  torch.zeros(bi, batch_size, hidden_size, device=device)
    else:
        hidden1 = (torch.zeros(bi, batch_size, hidden_size, device=device), torch.zeros(bi, batch_size, hidden_size, device=device))
        hidden2 = (torch.zeros(bi, batch_size, hidden_size, device=device), torch.zeros(bi, batch_size, hidden_size, device=device))

    mdn_params, hidden1,hidden2 = lr_model(prev_x, hidden1,hidden2)
    idx = get_pi_batch(torch.rand(batch_size, device=device), mdn_params[1])
    eos = mdn_params[0].t()[0] > torch.rand(batch_size, device=device)

    for i,j in enumerate(idx):
        next_x[i,1], next_x[i,2] = sample_gaussian_2d_batch(mdn_params[2][i][j],
                                      mdn_params[3][i][j], mdn_params[4][i][j], mdn_params[5][i][j], 
                                      mdn_params[6][i][j])

    #prev_x = np.zeros((1, 1, 3), dtype=np.float32)
    next_x[:,0] = eos

    return next_x
