"""
** deeplean-ai.com **
created by :: GauravBh1010tt
contact :: gauravbhatt.deeplearn@gmail.com
"""

from data_load import *  
from model import model_congen, mdn_loss, sample_congen
from eval_hand import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hidden_size = 400
n_layers = 1
num_gaussian = 20   # number of gaussian for Mixture Density Network
num_attn_gaussian = 10  # number of gaussians for attention window
dropout_p = 0.2
batch_size = 100
max_seq = 700
min_seq = 400       # omit samples below min sample length
max_text_seq = 40       # max length of text sequence 
print_every = batch_size*20
plot_every = 3
    
learning_rate = 0.0005
print_loss = 0
total_loss = torch.Tensor([0]).cuda()
print_loss_total = 0
teacher_forcing_ratio = 1      # do not change this right now
clip = 10.0
np.random.seed(9987)
epochs = 60
rnn_type = 2 # 1 for gru, 2 for lstm

lr_model = model_congen(input_size = 3, hidden_size=hidden_size, num_gaussian=num_gaussian,\
                 dropout_p = dropout_p, n_layers= n_layers, batch_size=batch_size,\
                 num_attn_gaussian = num_attn_gaussian, rnn_type = rnn_type).to(device)

model_optimizer = optim.Adam(lr_model.parameters(), lr=learning_rate)

num_mini_batch = 6000 - batch_size   # to ensure last batch is of batch length

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
          
      data, mask, text_len, char_to_vec, vec_to_char = get_strokes_text(i, batch_size, min_seq, max_seq, max_text_seq)
      
      stroke_tensor = torch.tensor(data[0], dtype=torch.float, device=device)
      target_tensor = torch.tensor(data[1], dtype=torch.float, device=device)
      text_tensor = torch.tensor(data[2], dtype=torch.float, device=device)
      stroke_mask = torch.tensor(mask[0], dtype=torch.float, device=device)
      text_mask = torch.tensor(mask[1], dtype=torch.float, device=device)
      text_len = torch.tensor(text_len, dtype=torch.float, device=device)
      old_k = torch.zeros((batch_size,num_attn_gaussian), dtype=torch.float, device=device)
      old_w = text_tensor.narrow(1,0,1)
      
      model_optimizer.zero_grad()    
      loss = 0
      for stroke in range(0,max_seq):
          mdn_params, hidden1, hidden2 = lr_model( stroke_tensor[:,stroke,:], text_tensor,
                                                  old_k, old_w, text_len.unsqueeze(1), hidden1, hidden2)
          old_k = mdn_params[-1]
          old_w = mdn_params[-2].unsqueeze(1)
          loss += mdn_loss(mdn_params, target_tensor[:,stroke,:], stroke_mask[:,stroke])
      loss = loss/max_seq
      
      loss.backward()      
      torch.nn.utils.clip_grad_norm(lr_model.parameters(), clip)    
      model_optimizer.step()
      
      print_loss_total += loss.item()/max_seq     
      if i % print_every == 0 and i>0:
        print_loss_avg = print_loss_total / print_every
        print_loss_total = 0
        #print(print_loss_avg)
        print('%d  %s (%d %d%%) %.4f' % (big_epoch,timeSince(start, i / num_mini_batch),
                                             i, i / num_mini_batch * 100, print_loss_avg))  
      print_loss+=1
      
    if big_epoch % plot_every == 0 and big_epoch>0:
       a,b,c,d = sample_congen(lr_model,'welcome to lyrebird',char_to_vec, hidden_size, time_step=800)
       plot_stroke(a)
       
save_checkpoint(big_epoch, lr_model, model_optimizer, 'saved_model', \
                    filename='model_congen.pt')
