import torch
import torch.nn as nn
from utils import AttetionLayer

class Generator(nn.Module):
    def __init__(self, params, device):
        super().__init__()
        self.p = params
        self.device = device
        #Arquitecture
        #Conv layers
        self.Conv2D1 = nn.Sequential(
                nn.Conv2d(in_channels=4, 
                          out_channels=32, 
                          kernel_size=3, 
                          stride=3, 
                          padding=1),
                nn.BatchNorm2d(32, momentum=0.15),
                nn.ReLU())
        self.Conv2D2 = nn.Sequential(
                nn.Conv2d(in_channels=32, 
                          out_channels=64, 
                          kernel_size=3, 
                          stride=3, 
                          padding=1),
                nn.BatchNorm2d(64, momentum=0.15),
                nn.ReLU())
        self.Conv2D3 = nn.Sequential(
                nn.Conv2d(in_channels=64, 
                          out_channels=128, 
                          kernel_size=3, 
                          stride=3, 
                          padding=0),
                nn.BatchNorm2d(128, momentum=0.15),
                nn.ReLU())
        self.Conv2D4 = nn.Sequential(
                nn.Conv2d(in_channels=128, 
                          out_channels=256, 
                          kernel_size=3, 
                          stride=3, 
                          padding=0),
                nn.BatchNorm2d(256, momentum=0.15),
                nn.ReLU())
        self.Conv2D5 = nn.Sequential(
                nn.Conv2d(in_channels=256, 
                          out_channels=512, 
                          kernel_size=2, 
                          stride=1, 
                          padding=0),
                nn.BatchNorm2d(512, momentum=0.15),
                nn.ReLU())

        #Linear layers    
        self.LinearI  =   nn.Sequential(nn.Linear(params['afterconv'], 2*params['latent_dim']), 
                                  nn.BatchNorm1d(2*params['latent_dim']), 
                                  nn.ReLU())
        self.LinearT  =   nn.Sequential(nn.Linear(2, params['latent_dim']), 
                                  nn.BatchNorm1d(params['latent_dim']), 
                                  nn.ReLU())
        self.LinearZ  =   nn.Sequential(nn.Linear(params['latent_dim'], params['latent_dim']), 
                                  nn.BatchNorm1d(params['latent_dim']), 
                                  nn.ReLU())
        self.LinearO  =   nn.Sequential(nn.Linear(2, params['latent_dim']), 
                                  nn.BatchNorm1d(params['latent_dim']), 
                                  nn.ReLU())

        #Endocer/Decoder + final Linear
        self.Encoder  =   nn.LSTM(4*params['latent_dim'], 
                          params['lstm_dim'], 
                          params['enc_layers'], 
                          batch_first=True, 
                          bidirectional=False)
        self.Attention =  AttetionLayer(params['lstm_dim'], params['lstm_dim'])
        self.Decoder   =  nn.LSTMCell(params['lstm_dim'], params['lstm_dim'], bias=True)
        self.LinearOut =  nn.Sequential(nn.Linear(params['lstm_dim'] + params['output_dim'], 2), 
                                  nn.Tanh())

    def forward(self, imgs, z, past_traj, target):
        b, s, c, w, h = imgs.shape #[batch, seq_len, num_channels, width, height]
        imgs = imgs.view(b*s, c, w, h) #4 dim tensor to Conv2D

        x = self.Conv2D1(imgs)
        x = self.Conv2D2(x)
        x = self.Conv2D3(x)
        x = self.Conv2D4(x)
        x = self.Conv2D5(x)

        _, c, w, h = x.shape
        x = x.view(-1, c*w*h)
        
        x = self.LinearI(x)        
        t = self.LinearT(past_traj.view(-1, 2))
        x = torch.cat((x.view(b, s, -1), t.view(b, s, -1)), 2)
        z = self.LinearZ(z.view(-1, self.p['latent_dim']))
        x = torch.cat((x, z.view(b, s, -1)), 2)
        
        h0_enc = torch.zeros((self.p['enc_layers'], b, self.p['lstm_dim'])).to(self.device)
        c0_enc = torch.zeros((self.p['enc_layers'], b, self.p['lstm_dim'])).to(self.device) 
        unpacked, (ht_enc, ct_enc) = self.Encoder(x, (h0_enc, c0_enc))
        
        h0_dec = torch.zeros((b, self.p['lstm_dim'])).to(self.device) 
        c0_dec = torch.zeros((b, self.p['lstm_dim'])).to(self.device)
        
        target = target.permute(1,0,2)[s-1] #current robot target position (the last in the array)

        for step in range(self.p['predict_seq']):
            if step == 0:
                context_vector = self.Attention(h0_dec, ht_enc[self.p['enc_layers']-1])
                ht_dec, ct_dec = self.Decoder(context_vector, (h0_dec, c0_dec))
                x = torch.cat((ht_dec, target), 1)
                out = self.LinearOut(x).view(b, 1, -1)
            else:
                context_vector = self.Attention(ht_dec, ht_enc[self.p['enc_layers']-1])
                ht_dec, ct_dec = self.Decoder(context_vector, (ht_dec, ct_dec))
                x = torch.cat((ht_dec, target), 1)
                x = self.LinearOut(x)
                out = torch.cat((out, x.view(b, 1, -1)), 1)    
        
        return out