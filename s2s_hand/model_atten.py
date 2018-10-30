# -*- coding: utf-8 -*-

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

'''
data = torch.tensor([[0.0000, 0.7000, 0.7500],[0.0000, 1.0000, 0.1500]])
mdn_p = [torch.tensor([[0.5317],[0.5309]]),
 torch.tensor([[0.3464, 0.2960, 0.3576],
         [0.3494, 0.2955, 0.3551]]),
 torch.tensor([[ 0.0840, -0.1579, -0.1997],
         [ 0.0917, -0.1582, -0.1874]]),
 torch.tensor([[-0.1352, -0.1710, -0.2975],
         [-0.1304, -0.1704, -0.3077]]),
 torch.tensor([[1.0845, 1.0115, 0.9486],
         [1.0979, 1.0160, 0.9539]]),
 torch.tensor([[1.0803, 0.9188, 0.8369],
         [1.0840, 0.9162, 0.8346]]),
 torch.tensor([[0.0546, 0.1053, 0.0143],
         [0.0485, 0.1062, 0.0176]])]
'''

bi = 2
bi_add = 2 # 1 for adding bi-directional o/p and 2 for concat
rnn_mode = 2 # 1 for gru, 2 for lstm
num_attn_gaussian = 10

class model(nn.Module):
    def __init__(self, input_size, hidden_size, num_gaussian, char_vec_len=61, dropout_p = 0.05, n_layers=1, batch_size=1):
        super(model, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.n_layers=n_layers
        self.num_gaussian = num_gaussian
        self.num_attn_gaussian = num_attn_gaussian
        self.char_vec_len = char_vec_len
        #print (char_vec_len)
        #print (hidden_size*bi_add+input_size+char_vec_len)
        if rnn_mode == 1:
            self.gru1 = nn.GRU(input_size, hidden_size, n_layers, dropout=dropout_p, batch_first=True, bidirectional = True)
            self.gru2 = nn.GRU(hidden_size*bi_add+input_size, hidden_size, n_layers, dropout=dropout_p, batch_first=True, bidirectional = True)
        else:
            self.gru1 = nn.LSTM(input_size+char_vec_len, hidden_size, n_layers, dropout=dropout_p, batch_first=True, bidirectional = True)
            self.gru2 = nn.LSTM(hidden_size*bi_add+input_size+char_vec_len, hidden_size, n_layers, dropout=dropout_p, batch_first=True, bidirectional = True)
        self.mdn = nn.Linear(hidden_size*2*bi_add, num_gaussian*6+1)
        self.window = nn.Linear(hidden_size*bi_add, num_attn_gaussian*3)
        #self.gru = nn.GRU(input_size, hidden_size, batch_first=False)

    def forward(self, inp, char_vec, old_k, old_w, hidden1, hidden2):
        
        #print (output.shape)
        #print (hidden.shape)
        #print ('inp',len(inp.size()))
        #embed = inp.view(self.batch_size, 1, -1)
        #self.char_vec_len = char_vec.shape[1]
        if len(inp.size()) == 2:
            inp=inp.unsqueeze(1)
            
        embed = torch.cat([inp, old_w], dim=-1)
        
        output1, hidden1 = self.gru1(embed, hidden1)
        if bi_add == 1:
            output1 = output1[:,:,0:self.hidden_size] + output1[:,:,self.hidden_size:]            
        #print (output.shape)
        #print (embed.shape)
        #print ('out1', output1.shape)
        abk_t = self.window(output1.squeeze(1)).exp()
        #print (abk_t.shape)
        
        a_t, b_t, k_t = abk_t.split(self.num_attn_gaussian, dim=1)
        #a_t,b_t = torch.exp(a_t), torch.exp(b_t)
        k_t = old_k + k_t
        
        u = torch.linspace(1, char_vec.shape[1], char_vec.shape[1], device=device)
        # print (u)
        #print (k_t.unsqueeze(2).shape)
        #print (torch.sub(k_t.unsqueeze(2).repeat((1,1,len(u))),u)**2)
        
        phi_bku = torch.exp(torch.mul(torch.sub(k_t.unsqueeze(2).repeat((1,1,len(u))),u)**2,
                                      -b_t.unsqueeze(2)))
        phi = torch.sum(torch.mul(a_t.unsqueeze(2),phi_bku),dim=1)
        
        #print (phi.unsqueeze(2).shape)
        #print (char_vec.shape)
        win_t = torch.sum(torch.mul(phi.unsqueeze(2), char_vec),dim=1)
        #print (win_t.unsqueeze(1).shape)
        #print (embed.shape)
        inp_skip = torch.cat([output1, inp, win_t.unsqueeze(1)], dim=-1)
        #print (inp_skip.shape)
        #print (inp_skip.shape)
        #print (inp_skip)
        output2, hidden2 = self.gru2(inp_skip, hidden2)
        
        if bi_add == 1:
            output2 = output2[:,:,0:self.hidden_size] + output2[:,:,self.hidden_size:]  
        
        output = torch.cat([output1,output2], dim=-1)
        
        #print ('out',output.shape)
        #print (output)
        #bre
        y_t = self.mdn(output.squeeze(1))
        #print ('y_t',y_t.shape)
        #print (y_t)
        
        e_t = y_t[:,0:1]
        pi_t, mu1_t, mu2_t, s1_t, s2_t, rho_t = torch.split(y_t[:,1:], self.num_gaussian, dim=1)
        #print (e_t,pi_t,mu1_t,mu2_t,s1_t, s2_t, rho_t)
        #bre
        e_t = F.sigmoid(e_t)
        pi_t = F.softmax(pi_t)
        s1_t, s2_t = torch.exp(s1_t), torch.exp(s2_t)
        rho_t = torch.tanh(rho_t)
        
        #print('final ',e_t,pi_t,mu1_t,mu2_t,s1_t, s2_t, rho_t)
        
        mdn_params = [e_t, pi_t, mu1_t, mu2_t, s1_t, s2_t, rho_t, win_t, k_t]
        #for i in mdn_params:
        #    print (i)
        
        return mdn_params, hidden1, hidden2

    def initHidden(self):
        #print (self.batch_size)
        return torch.zeros(self.n_layers*bi, self.batch_size, self.hidden_size, device=device)
    
    def initLHidden(self):
        return (torch.zeros(self.n_layers*bi, self.batch_size, self.hidden_size, device=device), 
                torch.zeros(self.n_layers*bi, self.batch_size, self.hidden_size, device=device))
    

def mdn_loss(mdn_params, data, mask):

    def get_2d_normal(x1,x2,mu1,mu2,s1,s2,rho):
      #print (x1,mu1)
      norm1 = torch.sub(x1.view(-1,1),mu1)
      norm2 = torch.sub(x2.view(-1,1),mu2)
      s1s2 = torch.mul(s1,s2)
      #print (x1,mu1)
      #print (s1s2)
      z = torch.div(norm1**2,s1**2) + torch.div(norm2**2,s2**2) - 2*torch.div(torch.mul(rho, torch.mul(norm1,norm2)),s1s2)
      #print (z)
      deno = 2*np.pi*s1s2*torch.sqrt(1-rho**2)
      numer = torch.exp(torch.div(-z,2*(1-rho**2)))
      return numer / deno

    #print (data)
    eos, x1, x2 = data[:,0], data[:,1], data[:,2]
    e_t, pi_t = mdn_params[0], mdn_params[1]
    res = get_2d_normal(x1,x2,mdn_params[2],mdn_params[3],mdn_params[4],mdn_params[5],mdn_params[6])
    #print (res)
    #bre

    epsilon = torch.tensor(1e-20, dtype=torch.float, device=device)
    #v = torch.ones(batch_size,device=device)
    #print(torch.mul(pi_t,res).shape)
    res1 = torch.sum(torch.mul(pi_t,res),dim=1)
    res1 = -torch.log(torch.max(res1,epsilon))
    #print(res1)
    res2 = torch.mul(eos, e_t.t()) + torch.mul(1-eos,1-e_t.t())
    res2 = -torch.log(res2)
    #res_final = v*torch.sum(res1+res2,dim=1)
    #print (res1.shape)
    #print (res2.shape)
    #print (mask.shape)
    res1 = torch.mul(res1,mask)
    res2 = torch.mul(res2,mask)
    return torch.sum(res1+res2)


def sample(lr_model, text, char_to_vec, start=[0,0,0], num=1000, scale = 20, random_state= np.random.randint(0,6000)):
    np.random.seed(random_state)
    
    def get_pi_idx(x, pdf):
        N = pdf.shape[0]
        accumulate = 0
        for i in range(0, N):
            accumulate += pdf[i]
            if (accumulate >= x):
                return i
        #print('error with sampling ensemble')
        return -1

    def sample_gaussian_2d(mu1, mu2, s1, s2, rho):
        mean = [mu1, mu2]
        cov = [[s1 * s1, rho * s1 * s2], [rho * s1 * s2, s2 * s2]]
        x = np.random.multivariate_normal(mean, cov, 1)
        return x[0][0], x[0][1]
    
    prev_x = torch.tensor(start,dtype=torch.float, device=device)
    prev_x[0] = 1
    strokes = np.zeros((num, 3), dtype=np.float32)
    old_k = torch.zeros((1,num_attn_gaussian), dtype=torch.float, device=device)
    
    
    vectors = np.zeros((max_text_seq, len(char_to_vec)+1))

    for p,q in enumerate(text):
        try:
            vectors[p][char_to_vec[q]] = 1
        except:
            #print (q)
            vectors[p][-1] = 1
            continue    
        
    text_tensor = torch.tensor(vectors, dtype=torch.float, device=device)
    old_w = text_tensor.narrow(0,0,1).unsqueeze(0)
    #print (prev_x.unsqueeze(0).shape)
    #print (text_tensor.unsqueeze(0).shape)
    mixture_params = []
    if rnn_mode == 1:
        hidden1 =  torch.zeros(bi, 1, hidden_size, device=device)
        hidden2 =  torch.zeros(bi, 1, hidden_size, device=device)
    else:
        hidden1 = (torch.zeros(bi, 1, hidden_size, device=device), torch.zeros(bi, 1, hidden_size, device=device))
        hidden2 = (torch.zeros(bi, 1, hidden_size, device=device), torch.zeros(bi, 1, hidden_size, device=device))
    
    for i in range(num):
        mdn_params, hidden1,hidden2 = lr_model(prev_x.unsqueeze(0),text_tensor.unsqueeze(0),
                                               old_k, old_w, hidden1,hidden2)
        old_k = mdn_params[-1]
        old_w = mdn_params[-2].unsqueeze(1)
        idx = get_pi_idx(np.random.random(), mdn_params[1][0])
        eos = 1 if np.random.random() < mdn_params[0][0] else 0
    
        next_x1, next_x2 = sample_gaussian_2d(mdn_params[2][0][idx].detach().cpu().numpy(), mdn_params[3][0][idx].detach().cpu().numpy(), 
                            mdn_params[4][0][idx].detach().cpu().numpy(), mdn_params[5][0][idx].detach().cpu().numpy(), 
                            mdn_params[6][0][idx].detach().cpu().numpy())
        
        strokes[i, :] = [eos, next_x1, next_x2]
        mixture_params.append(mdn_params)
        #prev_x = np.zeros((1, 1, 3), dtype=np.float32)
        prev_x[0],prev_x[1],prev_x[2] = eos, next_x1, next_x2
        
    strokes[:, 1:3] *= scale
    return strokes, mixture_params
        
hidden_size = 400
n_layers = 1
num_gaussian = 20
dropout_p = 0.2
batch_size = 50

max_seq = 600
min_seq = 400
max_text_seq = 40
print_every = batch_size*1
plot_every = batch_size*80

lr_model = model(3, hidden_size, num_gaussian, dropout_p = dropout_p, 
                 n_layers= n_layers, batch_size=batch_size).to(device)

#encoder_optimizer = optim.SGD(encoder.parameters(), lr=0.01)
#decoder_optimizer = optim.SGD(decoder.parameters(), lr=0.01)
#final_optimizer = optim.SGD(final_decode.parameters(), lr=0.01)

learning_rate = 0.0005
model_optimizer = optim.Adam(lr_model.parameters(), lr=learning_rate)

print_loss = 0
total_loss = torch.Tensor([0]).cuda()
print_loss_total = 0  # Reset every print_every
start = time.time()
teacher_forcing_ratio = 1
clip = 10.0

#data_x, data_y = get_data(batch_size=6000, max_seq = max_seq)

#epochs = len(data_x)
np.random.seed(9987)

epochs = 6000 - batch_size
#epochs = batch_size

for big_epoch in range(20):
    start = time.time()
    print_loss_total = 0
    for i in range(0,epochs,batch_size):        
      if rnn_mode==1:
          hidden1 = lr_model.initHidden()
          hidden2 = lr_model.initHidden()
      else:
          hidden1 = lr_model.initLHidden()
          hidden2 = lr_model.initLHidden()          
      #print (i)
      data, mask, text_len, char_to_vec, vec_to_char = get_strokes_text(i, batch_size, min_seq, max_seq, max_text_seq)
      
      stroke_tensor = torch.tensor(data[0], dtype=torch.float, device=device)
      target_tensor = torch.tensor(data[1], dtype=torch.float, device=device)
      text_tensor = torch.tensor(data[2], dtype=torch.float, device=device)
      stroke_mask = torch.tensor(mask[0], dtype=torch.float, device=device)
      text_mask = torch.tensor(mask[1], dtype=torch.float, device=device)
      text_len = torch.tensor(text_len, dtype=torch.float, device=device)
      old_k = torch.zeros((batch_size,num_attn_gaussian), dtype=torch.float, device=device)
      old_w = text_tensor.narrow(1,0,1)
      
      #bre
      model_optimizer.zero_grad()
    
      loss = 0
      md = []
      for stroke in range(0,max_seq):
          #output, hidden = encoder(x_data[:,word].unsqueeze(1), hidden)
          #encoder_output, encoder_hidden = encoder( input_tensor[:,word].unsqueeze(1), encoder_hidden)
          mdn_params, hidden1, hidden2 = lr_model( stroke_tensor[:,stroke,:], text_tensor,
                                                  old_k, old_w, hidden1, hidden2)
          old_k = mdn_params[-1]
          old_w = mdn_params[-2].unsqueeze(1)
          #bre
          #md.append(mdn_loss(mdn_params, target_tensor[:,word,:]))
          md.append(mdn_loss(mdn_params, target_tensor[:,stroke,:], stroke_mask[:,stroke]))
          #print (md)
          if torch.isnan(md[-1]):
              bre
          #loss += md[-1]
          #loss += mdn_loss(mdn_params, target_tensor[:,word,:])
          loss += mdn_loss(mdn_params, target_tensor[:,stroke,:], stroke_mask[:,stroke])
          #print (loss/100)
      #loss = loss/max_seq
      loss = loss/torch.sum(stroke_mask)
      #print (loss)
        
      loss.backward()
      
      torch.nn.utils.clip_grad_norm(lr_model.parameters(), clip)
    
      model_optimizer.step()
      #print ('here')
      print_loss_total += loss.item()/torch.sum(stroke_mask)
      #print ('here')
      #bre
      if i % print_every == 0 and i>0:
        print_loss_avg = print_loss_total / print_every
        print_loss_total = 0
        #print(print_loss_avg)
        print('%d  %s (%d %d%%) %.4f' % (big_epoch,timeSince(start, i / epochs),
                                             i, i / epochs * 100, print_loss_avg))
        #print ('actual ', out_tensor,'predicted ', decode)
        
      if i % plot_every == 0 and i>0:
        a,b = sample(lr_model,'welcome to lyrebird',char_to_vec,num=800)
        plot_stroke(a)
        
      print_loss+=1