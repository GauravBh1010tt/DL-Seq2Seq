"""
** deeplean-ai.com **
created by :: GauravBh1010tt
contact :: gauravbhatt.deeplearn@gmail.com
"""

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import warnings
import numpy as np
import matplotlib.pyplot as plt
import zipfile

zip_ref = zipfile.ZipFile('mnist_data.zip', 'r')
zip_ref.extractall()
zip_ref.close()

warnings.simplefilter('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(0)

batch_size = 100
original_dim = 784 # Number of pixels in MNIST images.
latent_dim = 5 # d, dimensionality of the latent code t.
intermediate_dim = 256 # Size of the hidden layer.
epochs = 30
learning_rate = 0.0008

def vlb_binomial(x, x_decoded_mean, t_mean, t_log_var):   
    generation_loss = -torch.sum(x * torch.log(1e-8 + x_decoded_mean) +\
                                 (1-x) * torch.log(1e-8 + 1 - x_decoded_mean),1)    
    latent_loss = 0.5 * torch.sum(t_mean**2 +\
                                  torch.exp(t_log_var)**2 - t_log_var - 1,1)
    vlb = torch.mean(generation_loss + latent_loss)
    
    return vlb

class Encoder(nn.Module):
    def __init__(self, original_dim, intermediate_dim):
        super(Encoder, self).__init__()
        self.out1 = nn.Linear(original_dim, intermediate_dim)
        self.out2 = nn.Linear(intermediate_dim, 2 * latent_dim)
        #self.gru = nn.GRU(input_size, hidden_size, batch_first=False)

    def forward(self, inp):
        output = F.relu(self.out1(inp))
        output = self.out2(output)
        
        return output
    
class Decoder(nn.Module):
    def __init__(self, original_dim, intermediate_dim):
        super(Decoder, self).__init__()
        self.out1 = nn.Linear(latent_dim, intermediate_dim)
        self.out2 = nn.Linear(intermediate_dim, original_dim)
        #self.gru = nn.GRU(input_size, hidden_size, batch_first=False)

    def forward(self, inp):
        output = F.relu(self.out1(inp))
        output = F.sigmoid(self.out2(output))
        return output

def sample_fig(encoder, decoder, x_train, x_test):
    fig = plt.figure(figsize=(10, 10))
    for fid_idx, (data, title) in enumerate(
                zip([x_train, x_test], ['Train', 'Validation'])):
        n = 10  # figure with 10 x 2 digits
        digit_size = 28
        figure = np.zeros((digit_size * n, digit_size * 2))
        data_enc = torch.tensor(data[:batch_size,:], dtype=torch.float, device=device)
        
        enc_output = encoder(data_enc)       
        t_mean, t_log_var = enc_output[:,0:latent_dim], enc_output[:,latent_dim:]       
        z = sampling([t_mean, t_log_var])
        decoded = decoder(z).detach().cpu().numpy()
        
        for i in range(10):
            figure[i * digit_size: (i + 1) * digit_size,
                   :digit_size] = data[i, :].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   digit_size:] = decoded[i, :].reshape(digit_size, digit_size)
        ax = fig.add_subplot(1, 2, fid_idx + 1)
        ax.imshow(figure, cmap='Greys_r')
        ax.set_title(title)
        ax.axis('off')
    plt.show()


def hallucinate(decoder, n_samples=10):

    sample = np.random.normal(0,1, size=(n_samples,latent_dim))
    inp_sample = torch.tensor(sample, dtype=torch.float, device=device)
    sampled_im_mean = decoder(inp_sample).detach().cpu().numpy()

    # Show the sampled images.
    plt.figure()
    for i in range(n_samples):
        ax = plt.subplot(n_samples // 5 + 1, 5, i + 1)
        plt.imshow(sampled_im_mean[i, :].reshape(28, 28), cmap='gray')
        ax.axis('off')
        #ax.set_title('images generated from latent distribution')
    plt.show()
    

def sampling(args):
    t_mean, t_log_var = args
    ep = torch.normal(torch.zeros(latent_dim,device=device), torch.ones(latent_dim, device=device))
    t = (ep * torch.exp(t_log_var/2)) + t_mean    
    
    return t

encoder = Encoder(original_dim, intermediate_dim).to(device)
decoder = Decoder(original_dim, intermediate_dim).to(device)
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)

x_train, x_test, y_train, y_test = np.load('x_train.npy'), np.load('x_test.npy'), np.load('y_train.npy'), np.load('y_test.npy')

num_mini_batch = len(x_train) - (len(x_train) % batch_size)

for epoch in range(epochs):
    for batch_id in range(0,num_mini_batch,batch_size):
        
        data_enc = torch.tensor(x_train[batch_id:batch_id+batch_size], dtype=torch.float, device=device)
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        
        enc_output = encoder(data_enc)       
        t_mean, t_log_var = enc_output[:,0:latent_dim], enc_output[:,latent_dim:]       
        z = sampling([t_mean, t_log_var])
        x_decoded_mean = decoder(z)
        
        loss = vlb_binomial(data_enc, x_decoded_mean, t_mean, t_log_var)
        
        loss.backward()                   
        encoder_optimizer.step()
        decoder_optimizer.step()
        
    if epoch % 1 == 0:
        print(epoch, loss.item())

sample_fig(encoder, decoder, x_train, x_test)
hallucinate(decoder, n_samples=50)
