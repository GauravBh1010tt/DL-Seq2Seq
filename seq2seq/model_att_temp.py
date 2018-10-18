# -*- coding: utf-8 -*-
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size=1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=False, bidirectional=True)
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
        #print('final ',output.shape,hidden.shape)
        output = output[:,:,0:self.hidden_size] + output[:,:,self.hidden_size:]
        return output, hidden

    def initHidden(self):
        return torch.zeros(2, 1, self.hidden_size, device=device)
      
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, bidirectional=True)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        
        output = output[:,:,0:self.hidden_size] + output[:,:,self.hidden_size:]

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(2, 1, self.hidden_size, device=device)
      
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
epochs = 5000
print_every = 80
batch_size = 1

training_pairs = [tensorsFromPair(random.choice(pairs))for i in range(epochs)]

#encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
#attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)

encoder = EncoderRNN(input_lang.n_words, hidden_size, batch_size = batch_size).to(device)
#decoder = DecoderRNN(hidden_size, output_lang.n_words, batch_size = batch_size).to(device)
decoder = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)

encoder_optimizer = optim.SGD(encoder.parameters(), lr=0.01)
decoder_optimizer = optim.SGD(decoder.parameters(), lr=0.01)

criterion = nn.NLLLoss()

print_loss = 0
total_loss = torch.Tensor([0]).cuda()
plot_losses = []
print_loss_total = 0  # Reset every print_every
plot_loss_total = 0  # Reset every plot_every

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
  
  #decoder_input = torch.zeros(batch_size, 1, n_features_in, device=device)
  encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
  
  loss = 0
  
  for word in range(input_tensor.size()[0]):
    #output, hidden = encoder(x_data[:,word].unsqueeze(1), hidden)
    #encoder_output, encoder_hidden = encoder( input_tensor[:,word].unsqueeze(1), encoder_hidden)
    encoder_output, hidden = encoder( input_tensor[word], encoder_hidden)
    encoder_outputs[word] = encoder_output[0, 0]
    
  decoder_input = torch.tensor([[SOS_token]], device=device)
    
  decode = []
  
  use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
  
  #use_teacher_forcing = False

  if use_teacher_forcing:
    for word in range(target_tensor.size()[0]):
      decoder_output, hidden, decoder_attention = decoder(
                decoder_input, hidden, encoder_outputs)
      loss += criterion(decoder_output, target_tensor[word])
      decoder_input = target_tensor[word]
      #decode.append(decoder_output.view(-1).argmax().cpu().detach().numpy().tolist())
  else:
    # Without teacher forcing: use its own predictions as the next input
    for di in range(target_tensor.size()[0]):
      decoder_output, hidden, decoder_attention = decoder(
                decoder_input, hidden, encoder_outputs)
      topv, topi = decoder_output.topk(1)
      decoder_input = topi.squeeze().detach()  # detach from history as input
      loss += criterion(decoder_output, target_tensor[di])
      if decoder_input.item() == EOS_token:
        break
  
  loss.backward()

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
