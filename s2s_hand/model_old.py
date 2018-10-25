# -*- coding: utf-8 -*-

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

bi = 2
bi_add = 2 # 1 for adding bi-directional o/p and 2 for concat

class model(nn.Module):
    def __init__(self, input_size, hidden_size, num_gaussian, dropout_p = 0.05, n_layers=1):
        super(model, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers=n_layers
        self.num_gaussian = num_gaussian
        self.gru1 = nn.GRU(input_size, hidden_size, n_layers, dropout=dropout_p, batch_first=False, bidirectional = True)
        self.gru2 = nn.GRU(hidden_size+input_size, hidden_size, n_layers, dropout=dropout_p, batch_first=False, bidirectional = True)
        self.mdn = nn.Linear(hidden_size*bi_add, num_gaussian*6+1)
        #self.gru = nn.GRU(input_size, hidden_size, batch_first=False)

    def forward(self, inp, hidden):
        
        #print (output.shape)
        #print (hidden.shape)
        #print (inp)
        embed = inp.view(1,1,-1)
        
        output, hidden = self.gru1(embed, hidden)
        if bi_add == 1:
            output = output[:,:,0:self.hidden_size] + output[:,:,self.hidden_size:]            
        #print (output.shape)
        #print (embed.shape)
        #inp_skip = torch.cat([embed,output], dim=-1)
        #print (inp_skip)
        #output, hidden = self.gru2(inp_skip, hidden)
        #bre
        
        #print (output[0].shape)
        y_t = self.mdn(output[0][0])
        #print (y_t)
        
        e_t = y_t[0:1]
        pi_t, mu1_t, mu2_t, s1_t, s2_t, rho_t = torch.split(y_t[1:], self.num_gaussian)
        
        e_t = F.sigmoid(e_t)
        pi_t = F.softmax(pi_t)
        s1_t, s2_t = torch.exp(s1_t), torch.exp(s2_t)
        rho_t = torch.tanh(rho_t)
        
        #print('final ',output.shape,hidden.shape)
        mdn_params = [e_t, pi_t, mu1_t, mu2_t, s1_t, s2_t, rho_t]
        #for i in mdn_params:
        #    print (i)
        
        return mdn_params, hidden

    def initHidden(self):
        return torch.zeros(self.n_layers*bi, 1, self.hidden_size, device=device)
    

def mdn_loss(mdn_params, data):
    
    def get_2d_normal(x1,x2,mu1,mu2,s1,s2,rho):
      norm1 = torch.sub(x1,mu1)
      norm2 = torch.sub(x2,mu2)
      s1s2 = torch.mul(s1,s2)
      z = torch.div(norm1**2,s1**2) + torch.div(norm2**2,s2**2) - 2*torch.div(torch.mul(rho, torch.mul(norm1,norm2)),s1s2)
      #print (z)
      deno = 2*np.pi*s1s2*torch.sqrt(1-rho**2)
      numer = torch.exp(torch.div(-z,2*(1-rho**2)))
      return numer / deno
    
    eos, x1, x2 = data[0], data[1], data[2]
    e_t, pi_t = mdn_params[0], mdn_params[1]
    res = get_2d_normal(x1,x2,mdn_params[2],mdn_params[3],mdn_params[4],mdn_params[5],mdn_params[6])
    #print (res)
    
    epsilon = torch.tensor(1e-20, dtype=torch.float, device=device)
    
    res1 = torch.sum(torch.mul(pi_t,res))
    res1 = -torch.log(torch.max(res1,epsilon))
    res2 = torch.mul(eos, e_t) + torch.mul(1-eos,1-e_t)
    res2 = -torch.log(res2)
    
    return res1+res2

def sample(lr_model, num=1000, scale = 20):
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
    
    prev_x = torch.zeros((3),dtype=torch.float, device=device)
    prev_x[0] = 1
    strokes = np.zeros((num, 3), dtype=np.float32)
    mixture_params = []
    hidden = lr_model.initHidden()
    
    for i in range(num):
        mdn_params, hidden = lr_model( prev_x, hidden)
        idx = get_pi_idx(np.random.random(), mdn_params[1])
        eos = 1 if np.random.random() < mdn_params[0] else 0
    
        next_x1, next_x2 = sample_gaussian_2d(mdn_params[2][idx].detach().cpu().numpy(), mdn_params[3][idx].detach().cpu().numpy(), mdn_params[4][idx].detach().cpu().numpy(), 
                                          mdn_params[5][idx].detach().cpu().numpy(), mdn_params[6][idx].detach().cpu().numpy())
        
        strokes[i, :] = [eos, next_x1, next_x2]
        mixture_params.append(mdn_params)
        #prev_x = np.zeros((1, 1, 3), dtype=np.float32)
        prev_x[0],prev_x[1],prev_x[2] = eos, next_x1, next_x2
        
    strokes[:, 1:3] *= scale
    return strokes, mixture_params
 
hidden_size = 400
print_every = 50
plot_every = 200
n_layers = 2
num_gaussian = 20
dropout_p = 0.2

lr_model = model(3, hidden_size, num_gaussian, dropout_p, n_layers).to(device)

#encoder_optimizer = optim.SGD(encoder.parameters(), lr=0.01)
#decoder_optimizer = optim.SGD(decoder.parameters(), lr=0.01)
#final_optimizer = optim.SGD(final_decode.parameters(), lr=0.01)

learning_rate = 0.0003
model_optimizer = optim.Adam(lr_model.parameters(), lr=learning_rate)

print_loss = 0
total_loss = torch.Tensor([0]).cuda()
print_loss_total = 0  # Reset every print_every
start = time.time()
clip = 10.0

data_x, data_y = get_data_seq(max_seq=400, batch_size=3000)
ids = np.arange(len(data_x))
np.random.shuffle(ids)
epochs = len(data_x)

for i in range(1,epochs+1):
  
  hidden = lr_model.initHidden()
  
  #data_x, data_y = get_data(i-1,1)
  input_tensor = torch.tensor(data_x[ids[i-1]], dtype=torch.float, device=device)
  target_tensor = torch.tensor(data_y[ids[i-1]], dtype=torch.float, device=device)
   
  model_optimizer.zero_grad()

  loss = 0
  md = []
  for word in range(0,len(input_tensor)):
    #output, hidden = encoder(x_data[:,word].unsqueeze(1), hidden)
    #encoder_output, encoder_hidden = encoder( input_tensor[:,word].unsqueeze(1), encoder_hidden)
    mdn_params, hidden = lr_model( input_tensor[word], hidden)
    #bre
    #md.append(mdn_loss(mdn_params, target_tensor[word]))
    #print (md)
    #if torch.isnan(md[-1]):
    #    bre
    #loss += md[-1]
    loss += mdn_loss(mdn_params, target_tensor[word])
    #print (loss/100)
    
  loss = loss/len(input_tensor)
  #print (loss)
    
  loss.backward()
  
  torch.nn.utils.clip_grad_norm(lr_model.parameters(), clip)

  model_optimizer.step()
  print_loss_total += loss.item()/target_tensor.size()[0]
  #bre
  if i % print_every == 0:
    print_loss_avg = print_loss_total / print_every
    print_loss_total = 0
    #print(print_loss_avg)
    print('%s (%d %d%%) %.4f' % (timeSince(start, i / epochs),
                                         i, i / epochs * 100, print_loss_avg))
    
  #bre
  if i % plot_every == 0:
    a,b = sample(lr_model,800)
    plot_stroke(a)
    #print ('actual ', out_tensor,'predicted ', decode)
    
  print_loss+=1