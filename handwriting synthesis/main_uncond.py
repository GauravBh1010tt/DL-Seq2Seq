# -*- coding: utf-8 -*-

from data_load import *
from model import model_uncond, mdn_loss, sample_uncond, scheduled_sample
from eval_hand import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


hidden_size = 400
n_layers = 1
num_gaussian = 20
dropout_p = 0.2
batch_size = 50
max_seq = 400
print_every = batch_size*40
plot_every = 4

learning_rate = 0.0005    
print_loss = 0
total_loss = torch.Tensor([0]).cuda()
print_loss_total = 0 
teacher_forcing_ratio = 1
clip = 10.0 
epochs = 50
rnn_type = 2 # 1 for gru, 2 for lstm

data_x, data_y = get_data_uncond(batch_size=6000, max_seq = max_seq)

lr_model = model_uncond(input_size = 3, hidden_size = hidden_size, num_gaussian = num_gaussian,\
                        dropout_p = dropout_p, n_layers = n_layers, batch_size = batch_size,\
                        rnn_type = rnn_type).to(device)
model_optimizer = optim.Adam(lr_model.parameters(), lr=learning_rate)

num_mini_batch = len(data_x) - (len(data_x) % batch_size)

for big_epoch in range(epochs):
    start = time.time()
    print_loss_total = 0
    for i in range(0, num_mini_batch, batch_size):
        
      if rnn_type==1:
          hidden1 = lr_model.initHidden()
          hidden2 = lr_model.initHidden()
      else:
          hidden1 = lr_model.initLHidden()
          hidden2 = lr_model.initLHidden()          

      input_tensor = torch.tensor(data_x[i:i+batch_size], dtype=torch.float, device=device)
      target_tensor = torch.tensor(data_y[i:i+batch_size], dtype=torch.float, device=device)
      model_optimizer.zero_grad()
    
      loss = 0
      for stroke in range(0,input_tensor.size()[1]):
        mdn_params, hidden1, hidden2 = lr_model( input_tensor[:,stroke,:], hidden1, hidden2)
        
        if np.random.random() > teacher_forcing_ratio:
            out_sample = scheduled_sample(lr_model, hidden_size, input_tensor[:,stroke,:], batch_size)
        else:
            out_sample = target_tensor[:,stroke,:]
        
        loss += mdn_loss(mdn_params, out_sample)
      loss = loss/input_tensor.size()[1]
      
      loss.backward()         
      torch.nn.utils.clip_grad_norm(lr_model.parameters(), clip)    
      model_optimizer.step()
      
      print_loss_total += loss.item()/target_tensor.size()[1]
      
      if i % print_every == 0 and i>0:
        print_loss_avg = print_loss_total / print_every
        print_loss_total = 0
        print('%d  %s (%d %d%%) %.4f' % (big_epoch,timeSince(start, i / num_mini_batch),
                                             i, i / num_mini_batch * 100, print_loss_avg))
      print_loss+=1
        
    if big_epoch % plot_every == 0 and big_epoch>0:
        a,b = sample_uncond(lr_model,hidden_size,time_step=800)
        plot_stroke(a)
        
save_checkpoint(big_epoch, lr_model, model_optimizer, 'saved_model', \
                    filename='model_uncond.pt')
