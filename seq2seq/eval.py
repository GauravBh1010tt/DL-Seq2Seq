import matplotlib.pyplot as plt

def evaluate(sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

#        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
#
#        for ei in range(input_length):
#            encoder_output, encoder_hidden = encoder(input_tensor[ei],
#                                                     encoder_hidden)
#            encoder_outputs[ei] += encoder_output[0, 0]
#            
            
        encoder_outputs, encoder_hidden = encoder( input_tensor, encoder_hidden)
        encoder_outputs = encoder_outputs.squeeze(1)

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)
        
        #decoder_attentions = []
        
        #print (decoder_input.shape)
        #rint (decoder_hidden.shape)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            #decoder_attentions[di] = decoder_attention.data
            #print (decoder_attention.shape[1])
            decoder_attentions[di,:decoder_attention.shape[1]] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions
      
def evaluateRandomly(n=1):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attn = evaluate(pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')
        
        plt.matshow(attn.numpy())
        #print (attn)
        