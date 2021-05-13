import torch
import torch.nn as nn
import numpy as np 

class AttetionLayer(nn.Module):
    def __init__(self, encoder_dim: int, decoder_dim: int, additive=True):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.v = nn.Parameter(torch.FloatTensor(self.decoder_dim).uniform_(-0.1, 0.1))
        self.W_1 = nn.Linear(self.decoder_dim, self.decoder_dim)
        self.W_2 = nn.Linear(self.encoder_dim, self.decoder_dim)
        self.W = torch.nn.Parameter(torch.FloatTensor(
                                    self.decoder_dim, 
                                    self.encoder_dim).uniform_(-0.1, 0.1))
        self.additive = additive
        self.Soft = nn.Softmax(dim=0)

    def forward(self, 
        query: torch.Tensor,   # [decoder_dim]
        values: torch.Tensor): # [seq_length, encoder_dim]

        if (self.additive):
            # query = query.repeat(values.size(0), 1)  # [seq_length, decoder_dim]
            weights = self.W_1(query) + self.W_2(values)  # [seq_length, decoder_dim]
            weights = torch.tanh(weights) * self.v  # [seq_length]
        else: #Multiplicative
            weights = query * self.W * values.T  # [seq_length]
            weights = weights/np.sqrt(self.decoder_dim)  # [seq_length]

        weights = self.Soft(weights)
        return weights * values

def hyperparameters(w=320, 
                    h=239, 
                    latent_dim=128, 
                    seq_len=50,
                    history_length=8, 
                    future_length=12,
                    cnn_filters=["32", "64", "128", "256", "512"],
                    lin_neurons=["1024", "1024"],
                    enc_layers=2,
                    lstm_dim=128,
                    output_dim=2,
                    up_criterion=1.0,
                    down_criterion=0.0,
                    alpha=0.15,
                    beta=0.15,
                    attention='add'):

    w1, h1 = (w+1)/3, (h+1)/3
    w2, h2 = (w1+1)/3, (h1+1)/3
    w3, h3 = w2/3, h2/3
    w4, h4 = w3/3, h3/3
    w5, h5 = w4-1, h4-1

    cnn_filters_int = []
    for filter_len in cnn_filters:
        cnn_filters_int.append(int(filter_len))

    lin_neurons_int = []
    for filter_len in lin_neurons:
        lin_neurons_int.append(int(filter_len))

    onadict = {
        'latent_dim'     : latent_dim,
        'seq_len'        : seq_len,
        'history'        : history_length,
        'predict_seq'    : future_length,
        'target_dim'     : 2,
        'afterconv'      : int(cnn_filters_int[-1]*w5*h5),
        'cnn_filters'    : cnn_filters_int,
        'lin_neurons'    : lin_neurons_int,
        'enc_layers'     : enc_layers,
        'lstm_dim'       : lstm_dim,
        'output_dim'     : output_dim,
        'up_criterion'   : up_criterion,
        'down_criterion' : down_criterion, 
        'alpha'          : alpha,
        'beta'           : beta, 
        'attention'      : attention

    }

    return onadict

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class AddUniformNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.rand(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def count_parameters(model):
    totalparams = 0
    for params in list(model.parameters()):
        curr = 1

        for s in list(params.size()):
            curr *= s
        
        totalparams += curr

    return totalparams
        