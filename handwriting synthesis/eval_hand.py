"""
** deeplean-ai.com **
created by :: GauravBh1010tt
contact :: gauravbhatt.deeplearn@gmail.com
"""

import numpy
from matplotlib import pyplot
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch
from model import model_uncond, mdn_loss, sample_uncond, scheduled_sample, model_congen, sample_congen

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_pretrained_uncond():
    ####################################################
    # default parameters, do not change

    hidden_size = 400
    n_layers = 1
    num_gaussian = 20
    dropout_p = 0.2
    batch_size = 50
    max_seq = 400
    print_every = batch_size*40
    plot_every = 4
    
    learning_rate = 0.0005    
    print_loss = 0
    total_loss = torch.Tensor([0]).cuda()
    print_loss_total = 0 
    teacher_forcing_ratio = 1
    clip = 10.0 
    epochs = 40
    rnn_type = 2 # 1 for gru, 2 for lstm

    ####################################################
    
    lr_model = model_uncond(input_size = 3, hidden_size = hidden_size, num_gaussian = num_gaussian,\
                        dropout_p = dropout_p, n_layers = n_layers, batch_size = batch_size,\
                        rnn_type = rnn_type).to(device)
    
    lr_model.load_state_dict(torch.load('saved_model/model_uncond.pt')['model'])
    return lr_model

def load_pretrained_congen():
    ####################################################
    # default parameters, do not change

    hidden_size = 400
    n_layers = 1
    num_gaussian = 20
    num_attn_gaussian = 10
    dropout_p = 0.2
    batch_size = 100
    max_seq = 700
    min_seq = 400       # omit samples below min sample length
    max_text_seq = 40       # max length of text sequence 
    print_every = batch_size*20
    plot_every = 3
        
    learning_rate = 0.0005
    print_loss = 0
    total_loss = torch.Tensor([0]).cuda()
    print_loss_total = 0
    teacher_forcing_ratio = 1      # do not chnage this right now
    clip = 10.0
    np.random.seed(9987)
    epochs = 50
    rnn_type = 2 # 1 for gru, 2 for lstm

    ####################################################
    
    lr_model = model_congen(input_size = 3, hidden_size=hidden_size, num_gaussian=num_gaussian,\
                     dropout_p = dropout_p, n_layers= n_layers, batch_size=batch_size,\
                     num_attn_gaussian = num_attn_gaussian, rnn_type = rnn_type).to(device)
    
    lr_model.load_state_dict(torch.load('saved_model/model_congen.pt')['model'])
    return lr_model

def phi_window_plots(phis, windows):
    #print (phis.shape)
    #print (windows.shape)    
    plt.figure(figsize=(16,4))
    plt.subplot(121)
    plt.title('Phis', fontsize=20)
    plt.ylabel('char code', fontsize=15)
    plt.xlabel("time steps", fontsize=15)
    plt.imshow(phis, interpolation='nearest', aspect='auto', cmap=cm.gist_stern)
    plt.subplot(122)
    plt.title('Soft attention window', fontsize=20)
    plt.ylabel("one-hot char vector", fontsize=15)
    plt.xlabel("time steps", fontsize=15)
    plt.imshow(windows, interpolation='nearest', aspect='auto', cmap=cm.gist_stern)
    

def gauss_params_plot(strokes, title ='Distribution of Gaussian Mixture parameters', figsize = (20,2)):
    plt.figure(figsize=figsize)
    import matplotlib.mlab as mlab
    buff = 1 ; epsilon = 1e-4
    minx, maxx = np.min(strokes[:,0])-buff, np.max(strokes[:,0])+buff
    miny, maxy = np.min(strokes[:,1])-buff, np.max(strokes[:,1])+buff
    delta = abs(maxx-minx)/400. ;
    
    x = np.arange(minx, maxx, delta)
    y = np.arange(miny, maxy, delta)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for i in range(strokes.shape[0]):
        gauss = mlab.bivariate_normal(X, Y, mux=strokes[i,0], muy=strokes[i,1], \
            sigmax=strokes[i,2], sigmay=strokes[i,3], sigmaxy=0) # sigmaxy=strokes[i,4] gives error
        Z += gauss * np.power(strokes[i,3] + strokes[i,2], .4) / (np.max(gauss) + epsilon)
    
    plt.title(title, fontsize=20)
    plt.imshow(np.flipud(Z), cmap=cm.gist_stern)
    
def plot_stroke(stroke, save_name=None):
    # Plot a single example.
    f, ax = pyplot.subplots()

    x = numpy.cumsum(stroke[:, 1])
    y = numpy.cumsum(stroke[:, 2])

    size_x = x.max() - x.min() + 1.
    size_y = y.max() - y.min() + 1.

    f.set_size_inches(5. * size_x / size_y, 5.)

    cuts = numpy.where(stroke[:, 0] == 1)[0]
    start = 0

    for cut_value in cuts:
        ax.plot(x[start:cut_value], y[start:cut_value],
                'k-', linewidth=3)
        start = cut_value + 1
    ax.axis('equal')
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

    if save_name is None:
        pyplot.show()
    else:
        try:
            pyplot.savefig(
                save_name,
                bbox_inches='tight',
                pad_inches=0.5)
        except Exception:
            print ("Error building image!: ", save_name)

    pyplot.close()
