from torch.nn.parameter import Parameter

USE_CUDA = 'True'

bi = 2

class Attn(nn.Module):
    def __init__(self, method, hidden_size, max_length=MAX_LENGTH):
        super(Attn, self).__init__()
        
        self.method = method
        self.hidden_size = hidden_size
        
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size*bi, hidden_size*bi)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.other = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, hidden, encoder_outputs):
        seq_len = len(encoder_outputs)

        # Create variable to store attention energies
        attn_energies = Variable(torch.zeros(seq_len)) # B x 1 x S
        if USE_CUDA: attn_energies = attn_energies.cuda()

        # Calculate energies for each encoder output
        #for i in range(seq_len):
        #    attn_energies[i] = self.score(hidden, encoder_outputs[i])
            
        #print ('old ',attn_energies)
        #print ('old ',encoder_outputs.shape)
        #print (self.attn(encoder_outputs).shape)
        attn_energies = torch.mm(self.attn(encoder_outputs.squeeze(1)), hidden.t())
        
        #print ('new ',attn_energies.t()[0])
        # Normalize energies to weights in range 0 to 1, resize to 1 x 1 x seq_len
        return F.softmax(attn_energies.t()[0]).unsqueeze(0).unsqueeze(0)
    
    def score(self, hidden, encoder_output):
        
        if self.method == 'dot':
            #print (hidden.shape)
            #print (encoder_output.shape)
            energy = hidden.view(-1).dot(encoder_output)
            #print (energy)
            return energy
        
        elif self.method == 'general':
            #print (hidden.shape)
            #print (encoder_output.shape)
            energy = self.attn(encoder_output)
            #print (energy.shape)
            energy = hidden.view(-1).dot(energy.view(-1))
            return energy
        
        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_output), 1))
            energy = self.other.dot(energy)
            return energy
      
class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, hidden_size, output_size, n_layers=1, dropout_p=0.1):
        super(LuongAttnDecoderRNN, self).__init__()
        
        # Keep parameters for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        bi = 2
        
        # Define layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout_p, bidirectional = True)
        self.out = nn.Linear(hidden_size, output_size)
        self.W = nn.Linear(hidden_size * 2 * bi, hidden_size)
        
        # Choose attention model
        if attn_model != 'none':
            self.attn = Attn(attn_model, hidden_size)
    
    def forward(self, word_input, last_hidden, encoder_outputs):
        # Note: we run this one step at a time
        
        # Get the embedding of the current input word (last output word)
        word_embedded = self.embedding(word_input).view(1, 1, -1) # S=1 x B x N
        
        # Combine embedded input word and last context, run through RNN
        #rnn_input = torch.cat((word_embedded, last_context.unsqueeze(0)), 2)
        rnn_output, hidden = self.gru(word_embedded, last_hidden)
        
        #rnn_output = output[:,:,0:self.hidden_size] + output[:,:,self.hidden_size:]

        # Calculate attention from current RNN state and all encoder outputs; apply to encoder outputs
        attn_weights = self.attn(rnn_output.squeeze(0), encoder_outputs)
        
        #encoder_outputs = encoder_outputs.unsqueeze(1)
        
        #print (encoder_outputs.shape)
        #print (attn_weights.shape)
        
        
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1)) # B x 1 x N

        #context = torch.sum(attn_weights.squeeze(0).t() * encoder_outputs.squeeze(1), dim=0).unsqueeze(0)
        
        #print (context.shape)
        
        # Final output layer (next word prediction) using the RNN hidden state and context vector
        rnn_output = rnn_output.squeeze(0) # S=1 x B x N -> B x N
        context = context.squeeze(1)         # B x S=1 x N -> B x N
        #print (rnn_output.shape)
        #print (context.shape)
        #print (torch.cat((rnn_output, context), 1).shape)
        conc = F.tanh(self.W(torch.cat((rnn_output, context), 1)))
        output = F.log_softmax(self.out(conc))
        
        # Return final output, hidden state, and attention weights (for visualization)
        return output, hidden, attn_weights



class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers=n_layers

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, batch_first=False, bidirectional = True)
        #self.gru = nn.GRU(input_size, hidden_size, batch_first=False)

    def forward(self, input, hidden):

        embedded = self.embedding(input).view(1,1,-1)
        seq_len = len(input)
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
    def __init__(self, attn_type, hidden_size, output_size, n_layers=1, dropout_p=0.1):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.maxlen = 0
        self.attn_type = attn_type

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,  dropout=dropout_p, batch_first=False, bidirectional = True)
        #self.gru = nn.GRU(output_size, hidden_size, batch_first=False)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax()
        
        self.wd = nn.Linear(hidden_size*2*bi, hidden_size)
        self.ws = nn.Linear(hidden_size*bi, hidden_size*bi)
        self.out2 = nn.Linear(hidden_size*2, output_size)
        
    def score(self, dec_st, enc_st):
        if self.attn_type == 'dot':
            return torch.mm(dec_st, enc_st.t())
        elif self.attn_type == 'general':
            #return torch.mm(torch.mm(dec_st, self.W), enc_st)
            return torch.mm(self.ws(enc_st), dec_st.t())
        
    def align(self, dec_state, enc_states):  
        #print (dec_state.shape)
        #print (enc_states.shape)
        enc_states=enc_states.squeeze(1)
        alpha = self.score(dec_state, enc_states)

        alpha = torch.softmax(alpha.t(), dim=1)

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
      

def train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):

    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0 # Added onto for each word

    # Get size of input and target sentences
    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]

    # Run words through encoder
    encoder_hidden = encoder.initHidden()
    
    #print(input_variable.shape)
    #print(encoder.shape)
    
    encoder_outputs, encoder_hidden = encoder(input_variable, encoder_hidden)
    
    # Prepare input and output variables
    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_context = Variable(torch.zeros(1, decoder.hidden_size))
    decoder_hidden = encoder_hidden # Use last hidden state from encoder to start decoder
    if USE_CUDA:
        decoder_input = decoder_input.cuda()
        decoder_context = decoder_context.cuda()

    # Choose whether to use teacher forcing
    use_teacher_forcing = random.random() < teacher_forcing_ratio
    if use_teacher_forcing:
        
        # Teacher forcing: Use the ground-truth target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_variable[di])
            decoder_input = target_variable[di] # Next target is next input

    else:
        # Without teacher forcing: use network's own prediction as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_variable[di])
            
            # Get most likely word index (highest value) from output
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]
            
            decoder_input = Variable(torch.LongTensor([[ni]])) # Chosen word is next input
            if USE_CUDA: decoder_input = decoder_input.cuda()

            # Stop at end of sentence (not necessary when using known targets)
            if ni == EOS_token: break

    # Backpropagation
    loss.backward()
    torch.nn.utils.clip_grad_norm(encoder.parameters(), clip)
    torch.nn.utils.clip_grad_norm(decoder.parameters(), clip)
    encoder_optimizer.step()
    decoder_optimizer.step()
    
    return loss.data[0] / target_length, decoder_attention


# Configuring training
n_epochs = 50000
#plot_every = 200
print_every = 1000



attn_model = 'general'
hidden_size = 500
n_layers = 2
dropout_p = 0.05
teacher_forcing_ratio = 0.5
clip = 5.0

# Initialize models
encoder = EncoderRNN(input_lang.n_words, hidden_size, n_layers).to(device)
#decoder = LuongAttnDecoderRNN(attn_model, hidden_size, output_lang.n_words, n_layers, dropout_p=dropout_p).to(device)
decoder = DecoderRNN(attn_model, hidden_size, output_lang.n_words, n_layers, dropout_p=dropout_p).to(device)

# Move models to GPU

# Initialize optimizers and criterion
learning_rate = 0.0001
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
criterion = nn.NLLLoss()

# Keep track of time elapsed and running averages
start = time.time()
plot_losses = []
print_loss_total = 0 # Reset every print_every
plot_loss_total = 0 # Reset every plot_every

for epoch in range(1, n_epochs + 1):
    
    # Get training data for this cycle
    training_pair = variables_from_pair(random.choice(pairs))
    input_variable = training_pair[0]
    target_variable = training_pair[1]

    # Run the train function
    loss, dec_attn = train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)

    # Keep track of loss
    print_loss_total += loss
    plot_loss_total += loss

    if epoch == 0: continue

    if epoch % print_every == 0:
        print_loss_avg = print_loss_total / print_every
        print_loss_total = 0
        print_summary = '%s (%d %d%%) %.4f' % (time_since(start, epoch / n_epochs), epoch, epoch / n_epochs * 100, print_loss_avg)
        print (print_summary)
        print (dec_attn)