"""
** deeplean-ai.com **
created by :: GauravBh1010tt
contact :: gauravbhatt.deeplearn@gmail.com
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

plt.rcParams['figure.figsize'] = (8, 8)
np.random.seed(42)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

def get_data(n_samples=2000, inv=True):
    n = n_samples
    d = 1
    x_train = np.random.uniform(0, 1, (n, d)).astype(np.float32)
    noise = np.random.uniform(-0.1, 0.1, (n, d)).astype(np.float32)
    y_train = x_train + 0.3*np.sin(2*np.pi*x_train) + noise
    x_test = np.linspace(0, 1, n).reshape(-1, 1).astype(np.float32)
    
    if inv:
        x_train_inv = y_train
        y_train_inv = x_train
    else:
        x_train_inv = x_train
        y_train_inv = y_train
    x_test = np.linspace(-0.1, 1.1, n).reshape(-1, 1).astype(np.float32) 
    
    fig = plt.figure(figsize=(8, 8))
    plt.plot(x_train_inv, y_train_inv, 'go', alpha=0.5)
    plt.show()
    return x_train_inv, y_train_inv, x_test

def generate_data(n_samples):
    epsilon = np.random.normal(size=(n_samples))
    x_data = np.random.uniform(-10.5, 10.5, n_samples)
    y_data = 7*np.sin(0.75*x_data) + 0.5*x_data + epsilon
    return x_data, y_data
    
n_samples = 1000
n_input = 1
x_data, y_data, x_test = generate_data(n_samples)

mdn_x_data = y_data
mdn_y_data = x_data

x_tensor = torch.from_numpy(np.float32(x_data).reshape(n_samples, n_input))
y_tensor = torch.from_numpy(np.float32(y_data).reshape(n_samples, n_input))

mdn_x_tensor = y_tensor
mdn_y_tensor = x_tensor

x_variable = Variable(mdn_x_tensor)
y_variable = Variable(mdn_y_tensor, requires_grad=False)

x_test_data = np.linspace(-10, 10, n_samples)

x_test_tensor = torch.from_numpy(np.float32(x_test_data).reshape(n_samples, n_input))
x_test_variable = Variable(x_test_tensor)

x_test_data = np.linspace(-15, 15, n_samples)
x_test_tensor = torch.from_numpy(np.float32(x_test_data).reshape(n_samples, n_input))
x_test_variable.data = x_test_tensor

import numpy as np

class MDN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, batch_size):
        super(MDN, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.z_h = nn.Linear(input_size, hidden_size)
        self.z_pi = nn.Linear(hidden_size, output_size)
        self.z_mu = nn.Linear(hidden_size, output_size)
        self.z_sig = nn.Linear(hidden_size, output_size)

    def forward(self, inp):
        z_h = F.tanh(self.z_h(inp))
        pi = F.softmax(self.z_pi(z_h))
        mu = self.z_mu(z_h)
        sig = torch.exp(self.z_sig(z_h))
                
        return pi, sig, mu
 
oneDivSqrtTwoPI = 1.0 / np.sqrt(2.0*np.pi) # normalization factor for Gaussians

def gaussian_distribution(y, mu, sigma):
    # make |mu|=K copies of y, subtract mu, divide by sigma
    result = (y.expand_as(mu) - mu) * torch.reciprocal(sigma)
    result = -0.5 * (result * result)
    return (torch.exp(result) * torch.reciprocal(sigma)) * oneDivSqrtTwoPI

def mdn_loss_fn(pi, sigma, mu, y):
    result = gaussian_distribution(y, mu, sigma) * pi
    result = torch.sum(result, dim=1)
    result = -torch.log(result)
    return torch.mean(result)

x_train_inv, y_train_inv = get_data()

x = Variable(torch.from_numpy(x_train_inv))
y = Variable(torch.from_numpy(y_train_inv))

model = MDN(1,20,5,n_samples)
opt = optim.Adam(model.parameters())

hidden = None

for epoch in range(10000):
  pi_variable, sigma_variable, mu_variable = model(x)
  loss = mdn_loss_fn(pi_variable, sigma_variable, mu_variable, y)
  opt.zero_grad()
  loss.backward()
  opt.step()
        
  if epoch % 500 == 0:
    print(epoch, loss.item())

x_test = np.linspace(-0.1, 1.1, n_samples).reshape(-1, 1).astype(np.float32) 

x_test_variable = Variable(torch.from_numpy(x_test))

pi_variable, sigma_variable, mu_variable = model(x_test_variable)

pi_data = pi_variable.data.numpy()
sigma_data = sigma_variable.data.numpy()
mu_data = mu_variable.data.numpy()


def gumbel_sample(x, axis=1):
    z = np.random.gumbel(loc=0, scale=1, size=x.shape)
    return (np.log(x) + z).argmax(axis=axis)

k = gumbel_sample(pi_data)

indices = (np.arange(n_samples), k)
rn = np.random.randn(n_samples)
sampled = rn * sigma_data[indices] + mu_data[indices]

plt.figure(figsize=(8, 8))
plt.scatter(x_train_inv, y_train_inv, alpha=0.2)
plt.scatter(x_test, sampled, alpha=0.2, color='red')
plt.show()
