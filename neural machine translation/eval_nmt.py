"""
** deeplean-ai.com **
created by :: GauravBh1010tt
contact :: gauravbhatt.deeplearn@gmail.com
"""

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as ticker
from data_load import *
from model import *
import torch

def load_pre_trained(lang):
    ####################################################
    # default parameters, do not change
    fra_to_eng = False
    hidden_size = 500
    n_layers = 2
    dropout_p = 0.05
    attn_score = 'gen'
    ####################################################
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_lang, output_lang, pairs = prepareData('eng', 'fra', fra_to_eng)
    encoder = EncoderRNN(input_lang.n_words, hidden_size, n_layers).to(device)
    decoder = AttnDecoderRNN(hidden_size, output_lang.n_words, attn_score, n_layers, dropout_p).to(device)
    if lang == 'eng-fra':
        encoder.load_state_dict(torch.load('saved_model/eng-fra/encoder.pt')['model'])
        decoder.load_state_dict(torch.load('saved_model/eng-fra/decoder.pt')['model'])
    else:
        encoder.load_state_dict(torch.load('saved_model/fra-eng/encoder.pt')['model'])
        decoder.load_state_dict(torch.load('saved_model/fra-eng/decoder.pt')['model'])

    return encoder, decoder

def evaluate(sentence, encoder, decoder, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs, encoder_hidden = encoder( input_tensor, encoder_hidden)
        encoder_outputs = encoder_outputs.squeeze(1)

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

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
    
def viz_attn(input_sentence, output_words, attentions):
    maxi = max(len(input_sentence.split()),len(output_words))
    attentions = attentions[:maxi,:maxi]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap=cm.bone)
    fig.colorbar(cax)

    ax.set_xticklabels([''] + input_sentence.split(' ') +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
    
def evalRand(n=1):
    for i in range(n):
        pair = random.choice(pairs)
        print('French Text -->', pair[0])
        print('Ground Truth ->', pair[1])
        output_words, attn = evaluate(pair[0])
        output_sentence = ' '.join(output_words)
        print('English o/p -->', output_sentence)
        print('')
        maxi = max(len(pair[0].split()),len(output_words))
        
def evalText(inp, encoder, decoder, inp_lang = 'English', out_lang = 'French'):
    print('%s Text -->'%inp_lang, inp)
    output_words, attn = evaluate(inp, encoder, decoder)
    output_sentence = ' '.join(output_words)
    print('%s o/p -->'%out_lang, output_sentence)
    print('')
    return inp, output_words, attn