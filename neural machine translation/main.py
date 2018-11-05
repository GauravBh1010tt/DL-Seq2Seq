"""
** deeplean-ai.com **
created by :: GauravBh1010tt
contact :: gauravbhatt.deeplearn@gmail.com
"""

###################################################
# For training from scratch
###################################################

from data_load import *
from model import *

hidden_size = 500
epochs = 50000
print_every = 1000
n_layers = 2
dropout_p = 0.05
learning_rate = 0.0001

print_loss = 0
total_loss = torch.Tensor([0]).cuda()
print_loss_total = 0
start = time.time()
teacher_forcing_ratio = 0.5
clip = 5.0
max_length = 10
fra_to_eng = False
attn_score = 'gen' # can be - 'gen', 'concat', 'dot'

input_lang, output_lang, pairs = prepareData('eng', 'fra', fra_to_eng)
print(random.choice(pairs))

training_pairs = [tensorsFromPair(random.choice(pairs), input_lang, output_lang) for i in range(epochs)]

encoder = EncoderRNN(input_lang.n_words, hidden_size, n_layers).to(device)
decoder = AttnDecoderRNN(hidden_size, output_lang.n_words, attn_score, n_layers, dropout_p).to(device)
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
criterion = nn.NLLLoss()

for i in range(1,epochs):
  
  encoder_hidden = encoder.initHidden()
  
  training_pair = training_pairs[i - 1]
  input_tensor = training_pair[0]
  target_tensor = training_pair[1] 
  encoder_optimizer.zero_grad()
  decoder_optimizer.zero_grad()
  loss = 0
  
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
      decoder_attentions.append(decoder_attention)
  else:
    for di in range(target_tensor.size()[0]):
      decoder_output, hidden, decoder_attention = decoder(decoder_input, hidden, encoder_outputs)      
      topv, topi = decoder_output.data.topk(1)
      ni = topi[0][0]
      if ni == EOS_token:
          break
      decoder_input = Variable(torch.cuda.LongTensor([[ni]]))      
      decoder_attentions.append(decoder_attention)
      loss += criterion(decoder_output, target_tensor[di])
  
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
    print('%s (%d %d%%) %.4f' % (timeSince(start, i / epochs),
                                         i, i / epochs * 100, print_loss_avg))
    print (decoder_attention)
    
  print_loss+=1
  
if fra_to_eng:
    save_checkpoint(epochs, encoder, encoder_optimizer, 'saved_model\fra-eng', \
                        filename='encoder.pt')
    save_checkpoint(epochs, decoder, decoder_optimizer, 'saved_model\fra-eng', \
                        filename='decoder.pt')
else:
    save_checkpoint(epochs, encoder, encoder_optimizer, 'saved_model\eng-fra', \
                        filename='encoder.pt')
    save_checkpoint(epochs, decoder, decoder_optimizer, 'saved_model\eng-fra', \
                        filename='decoder.pt')