from torch.nn.parameter import Parameter

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size=1, bi_add=1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.bi_add = bi_add

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=False, bidirectional = True)
        #self.gru = nn.GRU(input_size, hidden_size, batch_first=False)

    def forward(self, input, hidden):
        len_inp = 1
        if len(input.shape) == 2:
          len_inp = input.shape[0]
        elif len(input.shape) > 2:
          len_inp = input.shape[1]
          
        
        #print(input.size(),input.shape)
        if len(input.shape) == 2:
          len_inp = input.shape[0]
        elif len(input.shape) > 2:
          len_inp = input.shape[1]
        #print (input.shape)
        #print (len_inp)
        embedded = self.embedding(input).view(1,1,-1)
        output = embedded
        #output = input.view(1,1,-1)
        #print (output.shape)
        #print (hidden.shape)
        output, hidden = self.gru(output, hidden)
        if self.bi_add == 1:
            output = output[:,:,0:self.hidden_size] + output[:,:,self.hidden_size:]
        #print('final ',output.shape,hidden.shape)
        return output, hidden

    def initHidden(self):
        return torch.zeros(2, 1, self.hidden_size, device=device)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, batch_size=1):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.maxlen = 0

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=False, bidirectional = True)
        #self.gru = nn.GRU(output_size, hidden_size, batch_first=False)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax()
        self.W_d_m = Parameter(torch.zeros(self.hidden_size, self.hidden_size, device=device))
        self.W_e_m = Parameter(torch.zeros(self.hidden_size, self.hidden_size, device=device))
        self.W_m_s = Parameter(torch.zeros(1, self.hidden_size, device=device))
        #self.M_t = torch.zeros(self.maxlen, self.hidden_size, device=device, require_grads=False)
        
        nn.init.kaiming_uniform_(self.W_d_m, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_e_m, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_m_s, a=math.sqrt(5))
        
        self.wd = nn.Linear(hidden_size, hidden_size)
        self.we = nn.Linear(hidden_size, hidden_size)
        self.wm = nn.Linear(1, hidden_size)
        
    def forward(self, input, hidden, encoder_input):
        len_inp = 1
        if len(input.shape) == 2:
          len_inp = input.shape[0]
        elif len(input.shape) > 2:
          len_inp = input.shape[1]
          
        if len(input.shape) == 1:
          hid_size = 1
        else:
          hid_size = self.batch_size
        #print (input.shape)
        output = self.embedding(input).view(1,1,-1)
        #output = output.view(1,1,-1)
        output = F.relu(output)
        #print (output.shape)
        output, hidden = self.gru(output, hidden)
        output = output[:,:,0:self.hidden_size] + output[:,:,self.hidden_size:]
        dec_out = output[0]
        
        #print ('dec_out',dec_out)
        
        mt = F.tanh(self.wd(dec_out) + self.we(encoder_input))
        
        #print ('mt',mt)
        #print (mt.shape)
        #st = F.softmax(self.wm(mt))
        #M_t = torch.mm(dec_out, self.W_d_m) + torch.mm(encoder_input, self.W_e_m)
        #print (self.wm.weight.shape)
        
        #mid = torch.mm(M_t,self.W_m_s.t())
        mid = torch.mm(mt,self.wm.weight)
        #print ('mid',mid.t())
        s_t = F.softmax(mid.t())
        #print ('s_t',s_t)
        
        #print('final ',output.shape,hidden.shape)
        #output = self.softmax(self.out(output[0]))
        #print ('soft ',output.shape)
        #print (s_t.shape)
        return output, hidden, s_t
      

class dummy_trans(nn.Module):
    def __init__(self, hidden_size, output_size, batch_size=1):
        super(dummy_trans, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, input):

        output = self.softmax(F.tanh(self.out(input)))
        #print ('soft ',output.shape)
        return output

     
import time
import math

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))
      
#training_pair = training_pairs[0]
#input_tensor = training_pair[0]
#target_tensor = training_pair[1]

max_length = MAX_LENGTH
      
hidden_size = 256
epochs = 60000
batch_size = 1
bi = 1

training_pairs = [tensorsFromPair(random.choice(pairs))for i in range(epochs)]

#encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
#attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)

encoder = EncoderRNN(input_lang.n_words, hidden_size, batch_size = batch_size).to(device)
decoder = DecoderRNN(hidden_size, output_lang.n_words, batch_size = batch_size).to(device)
final_decode = dummy_trans(hidden_size, output_lang.n_words, batch_size = batch_size).to(device)

encoder_optimizer = optim.SGD(encoder.parameters(), lr=0.01)
decoder_optimizer = optim.SGD(decoder.parameters(), lr=0.01)
final_optimizer = optim.SGD(final_decode.parameters(), lr=0.01)

criterion = nn.NLLLoss()

print_loss = 0
total_loss = torch.Tensor([0]).cuda()
plot_losses = []
print_loss_total = 0  # Reset every print_every
plot_loss_total = 0  # Reset every plot_every
print_every = 5000
start = time.time()
teacher_forcing_ratio = 0.5

for i in range(1,epochs):
  
  encoder_hidden = encoder.initHidden()
  
  training_pair = training_pairs[i - 1]
  input_tensor = training_pair[0]
  target_tensor = training_pair[1]
  
  #X,y = X1,y1
  #batch_size = 1
  
  #input_tensor = torch.tensor(X, dtype=torch.float, device=device)
  #target_tensor = torch.tensor(y, dtype=torch.float, device=device)
  
  #out_tensor = [one_hot_decode(m) for m in y]
  #out_tensor = torch.tensor(out_tensor, dtype=torch.long, device=device)
  
  encoder_optimizer.zero_grad()
  decoder_optimizer.zero_grad()
  final_optimizer.zero_grad()
  
  #decoder_input = torch.zeros(batch_size, 1, n_features_in, device=device)
  encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
  
  loss = 0
  
  for word in range(input_tensor.size()[0]):
    #output, hidden = encoder(x_data[:,word].unsqueeze(1), hidden)
    #encoder_output, encoder_hidden = encoder( input_tensor[:,word].unsqueeze(1), encoder_hidden)
    encoder_output, hidden = encoder( input_tensor[word], encoder_hidden)
    encoder_outputs[word] = encoder_output[0, 0]
    
  decoder_input = torch.tensor([[SOS_token]], device=device)
  decoder_attentions = torch.zeros(max_length, max_length)
    
  #decode_out = []
  decode_out = torch.zeros(max_length, decoder.hidden_size, device=device)
  #decode_out = torch.zeros(max_length, output_lang.n_words, device=device)
  
  #decode_out1 = []
  
  use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
  
  #use_teacher_forcing = True

  if use_teacher_forcing:
    for word in range(target_tensor.size()[0]):
      decoder_output, hidden, decoder_attention = decoder(decoder_input, hidden, encoder_outputs)
      #loss += criterion(decoder_output, target_tensor[word])
      decoder_input = target_tensor[word]
      decode_out[word] = decoder_output
      #decode_out1.append(decoder_output)
      decoder_attentions[word] = decoder_attention
      #decode.append(decoder_output.view(-1).argmax().cpu().detach().numpy().tolist())
  else:
    # Without teacher forcing: use its own predictions as the next input
    for di in range(target_tensor.size()[0]):
      decoder_output, hidden, decoder_attention = decoder(decoder_input, hidden, encoder_outputs)
      topv, topi = decoder_output.topk(1)
      decoder_input = topi.squeeze().detach()  # detach from history as input
      decoder_attentions[word] = decoder_attention
      #loss += criterion(decoder_output, target_tensor[di])
      decode_out[word] = decoder_output
      if decoder_input.item() == EOS_token:
        break
  #print (decoder_attentions)
  norm_decode = torch.mm(decoder_attentions, decode_out.detach().cpu())
  norm_decode = torch.tensor(norm_decode, dtype=torch.float, device=device)
  decode_out = norm_decode
  
  for word_ind in range(target_tensor.size()[0]):
    out_word = final_decode(decode_out[word_ind])
    #out_word = decode_out[word]
    loss += criterion(out_word.view(1,-1), target_tensor[word_ind])
    #loss += criterion(decode_out[word], target_tensor[word_ind])
  loss.backward()

  encoder_optimizer.step()
  decoder_optimizer.step()
  final_optimizer.step()
  
  print_loss_total += loss.item()/target_tensor.size()[0]
  plot_loss_total += loss
  
  if i % print_every == 0:
    print_loss_avg = print_loss_total / print_every
    print_loss_total = 0
    #print(print_loss_avg)
    print('%s (%d %d%%) %.4f' % (timeSince(start, i / epochs),
                                         i, i / epochs * 100, print_loss_avg))
    #print ('actual ', out_tensor,'predicted ', decode)
    
  print_loss+=1
  
  #out = one_hot_decode(dec_out[0].detach().numpy())
  #print('expected ',out_tensor,'predicted ',out)

#encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)