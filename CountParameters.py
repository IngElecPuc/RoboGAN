import argparse
import torch
import torch.nn as nn
from utils import *
from generator import Generator
from discriminator import Discriminator
import numpy as np 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for training')
    parser.add_argument('--name', help='Sesion name', default='simple', type=str)
    #parser.add_argument('--latent_dim', help='Z latent dimension', default=512, type=int)
    parser.add_argument('--latent_dim', help='Z latent dimension', default=128, type=int)
    parser.add_argument('--history_length', help='history window', default=8, type=int)
    parser.add_argument('--future_length', help='prediction steps', default=12, type=int)
    parser.add_argument('--width', help='image width', default=320, type=int)
    parser.add_argument('--height', help='image height', default=239, type=int)
    #parser.add_argument('--cnn_filters', help='cnn filters in a list (without square parentesis)', nargs="+", default=["32", "64", "128", "256", "512"], type=int)
    parser.add_argument('--cnn_filters', help='cnn filters in a list (without square parentesis)', nargs="+", default=["16", "32", "64", "128", "256"], type=int)
    #parser.add_argument('--lin_neurons', help='neurons of each linae input layer in a list (without square parentesis)', nargs="+", default=["1024", "1024"], type=int)
    parser.add_argument('--lin_neurons', help='neurons of each linae input layer in a list (without square parentesis)', nargs="+", default=["256", "256"], type=int)
    parser.add_argument('--enc_layers', help='encoder layers', default=2, type=int)
    #parser.add_argument('--lstm_dim', help='lstm latent dimension', default=512, type=int)
    parser.add_argument('--lstm_dim', help='lstm latent dimension', default=128, type=int)
    parser.add_argument('--output_dim', help='generator ouptut dimension', default=2, type=int)
    parser.add_argument('--epochs', help='training epochs', default=20, type=int)
    parser.add_argument('--backbone', help='CNN backbone [CNN_own, resnet18]',default='CNN_own', type=str)
    parser.add_argument('--attention', help='type of attention [add, mult]', default='add', type=str)
    parser.add_argument('--gopti', help='type of optimizator for generator [adam, sgd]', default='adam', type=str)
    parser.add_argument('--dopti', help='type of optimizator for discriminator [adam, sgd]', default='sgd', type=str)
    parser.add_argument('--genlr', help='generator learning rate', default=1e-3, type=float)
    parser.add_argument('--dislr', help='discriminator rate', default=1e-3, type=float)
    parser.add_argument('--up', help='up adversarial criterion', default=0.9, type=float)
    parser.add_argument('--down', help='down adversarial criterion', default=0.0, type=float)
    parser.add_argument('--alpha', help='generator loss final point weight', default=0.15, type=float)
    parser.add_argument('--beta', help='generator loss velocity weight', default=0.15, type=float)
    parser.add_argument('--batch_size', help='batch_size', default=32, type=int)
    parser.add_argument('--num_workers', help='workers number', default=1, type=int)
    args = parser.parse_args()

    seq_len = 170

    netparams = hyperparameters(w=args.width, 
                            h=args.height, 
                            latent_dim=args.latent_dim, 
                            history_length=args.history_length, 
                            future_length=args.future_length,
                            cnn_filters=args.cnn_filters,
                            lin_neurons=args.lin_neurons,
                            enc_layers=args.enc_layers,
                            lstm_dim=args.lstm_dim,
                            output_dim=args.output_dim,
                            up_criterion=args.up,
                            down_criterion=args.down,
                            alpha=args.alpha,
                            beta=args.beta,
                            attention=args.attention)                             

    #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    gen = Generator(netparams, device).to(device)
    dis = Discriminator(netparams, device).to(device)

    print(F'The generator has {count_parameters(gen)} parameters')
    print(F'The discriminator has {count_parameters(dis)} parameters')