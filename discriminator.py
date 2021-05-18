import torch
import torch.nn as nn
from utils import AttetionLayer

class Discriminator(nn.Module):
    def __init__(self, params, device):
        super().__init__()
        self.p = params
        self.device = device
        #Arquitecture
        #Conv layers
        self.Conv2D1 = nn.Sequential(
                nn.Conv2d(in_channels=4, 
                          out_channels=params['cnn_filters'][0], 
                          kernel_size=3, 
                          stride=3, 
                          padding=1),
                #nn.BatchNorm2d(params['cnn_filters'][0], momentum=0.15),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Dropout(0.25))
        self.Conv2D2 = nn.Sequential(
                nn.Conv2d(in_channels=params['cnn_filters'][0], 
                          out_channels=params['cnn_filters'][1], 
                          kernel_size=3, 
                          stride=3, 
                          padding=1),
                nn.BatchNorm2d(params['cnn_filters'][1], momentum=0.15),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Dropout(0.25))
        self.Conv2D3 = nn.Sequential(
                nn.Conv2d(in_channels=params['cnn_filters'][1], 
                          out_channels=params['cnn_filters'][2], 
                          kernel_size=3, 
                          stride=3, 
                          padding=0),
                nn.BatchNorm2d(params['cnn_filters'][2], momentum=0.15),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Dropout(0.25))
        self.Conv2D4 = nn.Sequential(
                nn.Conv2d(in_channels=params['cnn_filters'][2], 
                          out_channels=params['cnn_filters'][3], 
                          kernel_size=3, 
                          stride=3, 
                          padding=0),
                nn.BatchNorm2d(params['cnn_filters'][3], momentum=0.15),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Dropout(0.25))
        self.Conv2D5 = nn.Sequential(
                nn.Conv2d(in_channels=params['cnn_filters'][3], 
                          out_channels=params['cnn_filters'][4], 
                          kernel_size=2, 
                          stride=1, 
                          padding=0),
                nn.BatchNorm2d(params['cnn_filters'][4], momentum=0.15),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Dropout(0.25))

        #Linear layers    
        self.LinearI  =   nn.Sequential(nn.Linear(params['afterconv'], 2*params['latent_dim']), 
                                  #nn.BatchNorm1d(2*params['latent_dim']), 
                                  nn.LeakyReLU(negative_slope=0.2),
                                  nn.Dropout(0.25))
        self.LinearT  =   nn.Sequential(nn.Linear(params['output_dim'], params['lin_neurons'][0]),
                                  #nn.BatchNorm1d(params['lin_neurons'][0]), 
                                  nn.LeakyReLU(negative_slope=0.2),
                                  nn.Dropout(0.25),
                                  nn.Linear(params['lin_neurons'][0],  params['lin_neurons'][1]), 
                                  nn.BatchNorm1d(params['lin_neurons'][1]), 
                                  nn.LeakyReLU(negative_slope=0.2),
                                  nn.Dropout(0.25),                                                  
                                  nn.Linear(params['lin_neurons'][1],  params['latent_dim']), 
                                  nn.BatchNorm1d(params['latent_dim']), 
                                  nn.LeakyReLU(negative_slope=0.2))
        self.LinearP  =   nn.Sequential(nn.Linear(params['output_dim'], params['lin_neurons'][0]),
                                  #nn.BatchNorm1d(params['lin_neurons'][0]), 
                                  nn.LeakyReLU(negative_slope=0.2),
                                  nn.Dropout(0.25),
                                  nn.Linear(params['lin_neurons'][0],  params['lin_neurons'][1]), 
                                  nn.BatchNorm1d(params['lin_neurons'][1]), 
                                  nn.LeakyReLU(negative_slope=0.2),
                                  nn.Dropout(0.25),                                                  
                                  nn.Linear(params['lin_neurons'][1],  params['latent_dim']), 
                                  nn.BatchNorm1d(params['latent_dim']), 
                                  nn.LeakyReLU(negative_slope=0.2),
                                  nn.Dropout(0.25))
        self.LinearO  =   nn.Sequential(nn.Linear(2, params['latent_dim']), 
                                  nn.BatchNorm1d(params['latent_dim']), 
                                  nn.LeakyReLU(negative_slope=0.2),
                                  nn.Dropout(0.25))

        self.DownTime =   nn.Sequential(nn.Conv1d(params['latent_dim'], params['latent_dim'], 5), 
                                  #nn.BatchNorm1d(params['latent_dim']), 
                                  nn.LeakyReLU(negative_slope=0.2))
        
        #Endocer/Decoder + final Linear
        self.Encoder  =   nn.LSTM(4*params['latent_dim'], 
                          params['lstm_dim'], 
                          params['enc_layers'], 
                          batch_first=True, 
                          bidirectional=False)
        typeattention = True if params['attention'] == 'add' else False
        self.Attention =  AttetionLayer(params['lstm_dim'], params['lstm_dim'], additive=typeattention)
        self.Decoder   =  nn.LSTMCell(params['lstm_dim'], params['lstm_dim'], bias=True)
        self.LinearOut =  nn.Sequential(nn.Linear(params['lstm_dim'] *
                                                  params['predict_seq'], 1), 
                                  nn.Sigmoid())

    def forward(self, imgs, prediction, past_traj):
        b, s, c, w, h = imgs.shape #[batch, seq_len, num_channels, width, height]
        imgs = imgs.reshape(b*s, c, w, h) #4 dim tensor to Conv2D

        x = self.Conv2D1(imgs)
        x = self.Conv2D2(x)
        x = self.Conv2D3(x)
        x = self.Conv2D4(x)
        x = self.Conv2D5(x)

        _, c, w, h = x.shape
        x = x.reshape(-1, c*w*h)
        
        x = self.LinearI(x)        
        t = self.LinearT(past_traj.reshape(-1, self.p['output_dim']))
        x = torch.cat((x.reshape(b, s, -1), t.reshape(b, s, -1)), 2)
        p = self.LinearP(prediction.reshape(-1, self.p['output_dim']))
        p = p.reshape(b, -1, self.p['latent_dim'])
        p = self.DownTime(p.permute(0, 2, 1))
        x = torch.cat((x, p.permute(0, 2, 1)), 2)
        
        h0_enc = torch.zeros((self.p['enc_layers'], b, self.p['lstm_dim'])).to(self.device)
        c0_enc = torch.zeros((self.p['enc_layers'], b, self.p['lstm_dim'])).to(self.device) 
        unpacked, (ht_enc, ct_enc) = self.Encoder(x, (h0_enc, c0_enc))

        out_seq = list()
        ht_dec = torch.zeros((b, self.p['lstm_dim'])).to(self.device)
        ct_dec = torch.zeros((b, self.p['lstm_dim'])).to(self.device)

        for step in range(self.p['predict_seq']):
             context_vector = self.Attention(ht_dec, ht_enc[self.p['enc_layers']-1])
             ht_dec, ct_dec = self.Decoder(context_vector, (ht_dec, ct_dec))
             out_seq.append(ht_dec)

        out = torch.stack(out_seq, dim=1)
        out = self.LinearOut(out.reshape(b, -1))

        return out





