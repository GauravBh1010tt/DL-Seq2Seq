from torch.nn.parameter import Parameter

bi=2

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers=n_layers

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, batch_first=False, bidirectional = True)
        #self.gru = nn.GRU(input_size, hidden_size, batch_first=False)

    def forward(self, input, hidden):

        #embedded = self.embedding(input).view(1,1,-1)
        seq_len = len(input)
        #print (seq_len)
        #bre
        embedded = self.embedding(input).view(seq_len, 1, -1)
        output = embedded
        #output = input.view(1,1,-1)
        #print (output.shape)
        #print (hidden.shape)
        

        output, hidden = self.gru(output, hidden)
        #output = output[:,:,0:self.hidden_size] + output[:,:,self.hidden_size:]
        #print('final ',output.shape,hidden.shape)
        return output, hidden

    def initHidden(self):
        return torch.zeros(self.n_layers*bi, 1, self.hidden_size, device=device)
    

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, attn_type = 'gen', n_layers=1, dropout_p=0.1):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.maxlen = 0
        self.attn_type = attn_type

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,  dropout=dropout_p, batch_first=False, bidirectional = True)
        #self.gru = nn.GRU(output_size, hidden_size, batch_first=False)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax()
        #self.W_d_m = Parameter(torch.zeros(self.hidden_size, self.hidden_size, device=device))
        #self.M_t = torch.zeros(self.maxlen, self.hidden_size, device=device, require_grads=False)
        
        #nn.init.kaiming_uniform_(self.W_d_m, a=math.sqrt(5))
        #nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        
        self.wd = nn.Linear(hidden_size*2*bi, hidden_size)
        self.ws = nn.Linear(hidden_size*bi, hidden_size*bi)
        self.out2 = nn.Linear(hidden_size*2, output_size)
        
    def score(self, dec_st, enc_st):
        if self.attn_type == 'dot':
            return torch.mm(dec_st, enc_st.t())
        elif self.attn_type == 'gen':
            #return torch.mm(torch.mm(dec_st, self.W), enc_st)
            return torch.mm(self.ws(enc_st), dec_st.t())
        
    def align(self, dec_state, enc_states):  
        #print (dec_state.shape)
        #print (enc_states.shape)
        alpha = self.score(dec_state, enc_states)
        #print ('initial',alpha.t())
        #print (enc_states.shape)
        alpha = torch.softmax(alpha.t(), dim=1)
        #print ('alpha', alpha.t().shape)
        #bre
        #return torch.sum(alpha.t() * enc_states, dim=0).unsqueeze(0), alpha
        return torch.mm(alpha, enc_states), alpha
        
    def forward(self, input, hidden, encoder_input):
        
        #print (input.shape)
        output = self.embedding(input).view(1,1,-1)
        #output = output.view(1,1,-1)
        output = F.relu(output)
        #print (output.shape)
        output, hidden = self.gru(output, hidden)
        #output = output[:,:,0:self.hidden_size] + output[:,:,self.hidden_size:]
        dec_out = output[0]
        
        #print ('dec_out', dec_out.shape)
        #print ('enc_out', encoder_input.shape)
        c_t, attn_wt = self.align(dec_out, encoder_input)
        #print ('c_t',torch.cat((c_t,dec_out),dim=1).shape)
        
        h_t = F.tanh(self.wd(torch.cat((c_t,dec_out),dim=1)))
        
        #h_t = torch.cat((c_t,dec_out),dim=1)

        out_word = self.softmax(self.out(h_t))
        #print ('s_t',s_t)
        
        #print('final ',output.shape,hidden.shape)
        #output = self.softmax(self.out(output[0]))
        #print ('soft ',output.shape)
        #print (s_t.shape)
        return out_word, hidden, attn_wt

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
      
hidden_size = 500
epochs = 50000
print_every = 1000
n_layers = 2
dropout_p = 0.05

training_pairs = [tensorsFromPair(random.choice(pairs))for i in range(epochs)]

#encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
#attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)

encoder = EncoderRNN(input_lang.n_words, hidden_size, n_layers).to(device)

#decoder = LuongAttnDecoderRNN('general', hidden_size, output_lang.n_words, 1).to(device)
decoder = DecoderRNN(hidden_size, output_lang.n_words,'gen', n_layers, dropout_p).to(device)
#final_decode = dummy_trans(hidden_size, output_lang.n_words, batch_size = batch_size).to(device)

#encoder_optimizer = optim.SGD(encoder.parameters(), lr=0.01)
#decoder_optimizer = optim.SGD(decoder.parameters(), lr=0.01)
#final_optimizer = optim.SGD(final_decode.parameters(), lr=0.01)

learning_rate = 0.0001
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
  
  #out = one_hot_decode(dec_out[0].detach().numpy())
  #print('expected ',out_tensor,'predicted ',out)

#encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)