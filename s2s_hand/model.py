# -*- coding: utf-8 -*-

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

bi = 2
bi_add = 1 # 1 for adding bi-directional o/p and 2 for concat

class model(nn.Module):
    def __init__(self, input_size, hidden_size, num_gaussian, n_layers=1):
        super(model, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers=n_layers
        self.gru = nn.GRU(input_size, hidden_size, n_layers, batch_first=False, bidirectional = True)
        self.mdn = nn.Linear(hidden_size*bi_add, num_gaussian*6+1)
        #self.gru = nn.GRU(input_size, hidden_size, batch_first=False)

    def forward(self, inp, hidden):
        
        #print (output.shape)
        #print (hidden.shape)
        
        output, hidden = self.gru(inp, hidden)
        if bi_add == 1:
            output = output[:,:,0:self.hidden_size] + output[:,:,self.hidden_size:]            
        
        y_t = self.mdn(output)
        
        e_t = y_t[0:1]
        pi_t, mu1_t, mu2_t, s1_t, s2_t, rho_t = torch.split(y_t[1:], 6)
        
        e_t = F.sigmoid(e_t)
        pi_t = F.softmax(pi_t)
        s1_t, s2_t = torch.exp(s1_t), torch.exp(s2_t)
        rho_t = torch.tanh(rho_t)
        
        #print('final ',output.shape,hidden.shape)
        mdn_params = [e_t, pi_t, mu1_t, mu2_t, s1_t, s2_t, rho_t]
        return mdn_params, hidden

    def initHidden(self):
        return torch.zeros(self.n_layers*bi, 1, self.hidden_size, device=device)
    

def mdn_loss(mdn_params, data):
    
    def get_2d_normal(x1,x2,mu1,mu2,s1,s2,rho):
      norm1 = torch.sub(x1,mu1)
      norm2 = torch.sub(x2,mu2)
      s1s2 = torch.mul(s1,s2)
      z = torch.div(norm1**2,s1**2) + torch.div(norm2**2,s2**2) + 2*torch.div(torch.mul(norm1,norm2),s1s2)
      deno = 1/(np.pi*s1s2*torch.sqrt(1-rho**2))
      numer = torch.exp(torch.div(-z,2*(1-rho**2)))
      return numer / deno
    
    x1, x2, eos = data[0],data[1],data[2]
    e_t, pi_t = mdn_params[0], mdn_params[1]
    res = get_2d_normal(x1,x2,mdn_params[2],mdn_params[3],mdn_params[4],mdn_params[5],mdn_params[6])
    
    epsilon = torch.tensor(1e-20)
    
    res1 = torch.sum(torch.mul(pi_t,res))
    res1 = -torch.log(torch.max(res1,epsilon))
    res2 = torch.mul(eos, e_t) + torch.mu(1-eos,1-e_t)
    res2 = -torch.log(res2)
    
    return res1+res2
    
        
hidden_size = 24
epochs = 2
print_every = 1
n_layers = 1
num_gaussian = 3
dropout_p = 0.05

lr_model = model(3, hidden_size, num_gaussian, n_layers).to(device)
bre
#encoder_optimizer = optim.SGD(encoder.parameters(), lr=0.01)
#decoder_optimizer = optim.SGD(decoder.parameters(), lr=0.01)
#final_optimizer = optim.SGD(final_decode.parameters(), lr=0.01)

learning_rate = 0.0005
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
criterion = nn.NLLLoss()

print_loss = 0
total_loss = torch.Tensor([0]).cuda()
plot_losses = []
print_loss_total = 0  # Reset every print_every
plot_loss_total = 0  # Reset every plot_every
start = time.time()
teacher_forcing_ratio = 0.5
clip = 5.0

for i in range(1,epochs):
  
  encoder_hidden = encoder.initHidden()
  
  training_pair = training_pairs[i - 1]
  input_tensor = training_pair[0]
  target_tensor = training_pair[1]
   
  encoder_optimizer.zero_grad()
  decoder_optimizer.zero_grad()
  #final_optimizer.zero_grad()
  
  #decoder_input = torch.zeros(batch_size, 1, n_features_in, device=device)
  encoder_outputs = torch.zeros(max_length, encoder.hidden_size*bi, device=device)
  #encoder_outputs = []
  loss = 0
  
  for word in range(input_tensor.size()[0]):
    #output, hidden = encoder(x_data[:,word].unsqueeze(1), hidden)
    #encoder_output, encoder_hidden = encoder( input_tensor[:,word].unsqueeze(1), encoder_hidden)
    encoder_output, hidden = encoder( input_tensor[word], encoder_hidden)
    encoder_outputs[word] = encoder_output[0, 0]
    #encoder_outputs.append(encoder_output[0, 0])
    
  encoder_outputs, hidden = encoder( input_tensor, encoder_hidden)
  encoder_outputs = encoder_outputs.squeeze(1)
  decoder_input = torch.tensor([[SOS_token]], device=device)
  decoder_attentions = torch.zeros(max_length, max_length)
  decoder_attentions = []
  use_teacher_forcing = random.random() < teacher_forcing_ratio

  if use_teacher_forcing:
    for word in range(target_tensor.size()[0]):
      decoder_output, hidden, decoder_attention = decoder(decoder_input, hidden, encoder_outputs)
      loss += criterion(decoder_output, target_tensor[word])
      decoder_input = target_tensor[word]
      #decode_out[word] = decoder_output
      #decoder_attentions[word] = decoder_attention
      decoder_attentions.append(decoder_attention)
      #decode.append(decoder_output.view(-1).argmax().cpu().detach().numpy().tolist())
  else:
    # Without teacher forcing: use its own predictions as the next input
    for di in range(target_tensor.size()[0]):
      decoder_output, hidden, decoder_attention = decoder(decoder_input, hidden, encoder_outputs)
      #topv, topi = decoder_output.topk(1)
      #decoder_input = topi.squeeze().detach()  # detach from history as input
      #decoder_attentions[word] = decoder_attention
      
      topv, topi = decoder_output.data.topk(1)
      ni = topi[0][0]
      if ni == EOS_token:
          break
      decoder_input = Variable(torch.cuda.LongTensor([[ni]]))
      
      decoder_attentions.append(decoder_attention)
      loss += criterion(decoder_output, target_tensor[di])
      #decode_out[word] = decoder_output
      
      #if decoder_input.item() == EOS_token:
      #  break
  
  loss.backward()
  
  torch.nn.utils.clip_grad_norm(encoder.parameters(), clip)
  torch.nn.utils.clip_grad_norm(decoder.parameters(), clip)

  encoder_optimizer.step()
  decoder_optimizer.step()
  
  print_loss_total += loss.item()/target_tensor.size()[0]
  plot_loss_total += loss
  
  if i % print_every == 0:
    print_loss_avg = print_loss_total / print_every
    print_loss_total = 0
    #print(print_loss_avg)
    print('%s (%d %d%%) %.4f' % (timeSince(start, i / epochs),
                                         i, i / epochs * 100, print_loss_avg))
    #print ('actual ', out_tensor,'predicted ', decode)
    print (decoder_attention)
    
  print_loss+=1














