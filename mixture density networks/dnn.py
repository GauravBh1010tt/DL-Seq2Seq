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

x_train, y_train, x_test = get_data(inv=True)

class DNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DNN, self).__init__()
        self.out1 = nn.Linear(input_size, hidden_size,)
        self.out2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, inp):
        output = F.tanh(self.out1(inp))
        output = self.out2(output)
        return output
    
model = DNN(1,15,1)
opt = optim.SGD(model.parameters(), lr=0.09, momentum=0.9, nesterov=True)

x = Variable(torch.from_numpy(x_train))
y = Variable(torch.from_numpy(y_train))

for e in range(4000):
  opt.zero_grad()
  out = model(x)
  loss = F.mse_loss(out, y)  # negative log likelihood assuming a Gaussian distribution
  if e % 100 == 0:
    print(e, loss.item())
  loss.backward()
  opt.step()
  
out = model(Variable(torch.from_numpy(x_test)))
#out2 = forward(Variable(torch.from_numpy(x_test)))

fig = plt.figure(figsize=(8, 8))
plt.plot(x_train, y_train, 'go', alpha=0.5)
plt.plot(x_test, out.data.numpy(), 'r', linewidth=3.0)
plt.show()