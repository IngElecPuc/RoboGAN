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
                    enc_layers=2,
                    lstm_dim=128,
                    output_dim=2,
                    attention='add'):

    w1, h1 = (w+1)/3, (h+1)/3
    w2, h2 = (w1+1)/3, (h1+1)/3
    w3, h3 = w2/3, h2/3
    w4, h4 = w3/3, h3/3
    w5, h5 = w4-1, h4-1

    onadict = {
        'latent_dim' : latent_dim,
        'seq_len'    : seq_len,
        'history'    : history_length,
        'predict_seq': future_length,
        'afterconv'  : int(512*w5*h5),
        'enc_layers' : enc_layers,
        'lstm_dim'   : lstm_dim,
        'output_dim' : output_dim,
        'attention'  : attention
    }

    return onadict