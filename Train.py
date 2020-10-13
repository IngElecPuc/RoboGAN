import argparse
import torch
import torch.nn as nn
from torchvision import transforms as tfs
from torch.utils.data import DataLoader
from utils import hyperparameters
from dataset import RobotDataset, dataset_explore
from generator import Generator
from discriminator import Discriminator
from training import train_gan, test_gan
import numpy as np 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for training')
    parser.add_argument('--latent_dim', help='Z latent dimension', default=128, type=int)
    parser.add_argument('--history_length', help='history window', default=12, type=int)
    parser.add_argument('--future_length', help='prediction steps', default=8, type=int)
    parser.add_argument('--width', help='image width', default=320, type=int)
    parser.add_argument('--height', help='image height', default=239, type=int)
    parser.add_argument('--enc_layers', help='encoder layers', default=2, type=int)
    parser.add_argument('--lstm_dim', help='lstm latent dimension', default=128, type=int)
    parser.add_argument('--output_dim', help='generator ouptut dimension', default=2, type=int)
    parser.add_argument('--epochs', help='training epochs', default=20, type=int)
    parser.add_argument('--backbone', help='CNN backbone [CNN_own, resnet18]',default='CNN_own', type=str)
    parser.add_argument('--attention', help='type of attention [add, mult]', default='add', type=str)
    parser.add_argument('--opti', help='type of optimizator [adam, sgd]', default='adam', type=str)
    parser.add_argument('--genlr',help='generator learning rate', default=1e-3, type=float)
    parser.add_argument('--dislr',help='discriminator rate', default=1e-3, type=float)
    parser.add_argument('--batch_size', help='batch_size', default=16, type=int)
    parser.add_argument('--num_workers', help='workers number', default=1, type=int)
    args = parser.parse_args()

    nethparams = hyperparameters(w=args.width, 
                            h=args.height, 
                            latent_dim=args.latent_dim, 
                            history_length=args.history_length, 
                            future_length=args.future_length,
                            enc_layers=args.enc_layers,
                            lstm_dim=args.lstm_dim,
                            output_dim=args.output_dim,
                            attention=args.attention)                            

    data_transforms = tfs.Compose([tfs.Resize((320, 239)),
                               tfs.ToTensor(),
                               tfs.Normalize([0.5], [0.5])])

    train_set = RobotDataset('train', 
                            args.latent_dim, 
                            dataset_explore('train'), 
                            data_transforms, 
                            train=True)
    valid_set = RobotDataset('train', 
                            args.latent_dim, 
                            dataset_explore('train'),
                            data_transforms, 
                            train=False)
    test__set = RobotDataset('train', 
                            args.latent_dim, 
                            dataset_explore('train'),
                            data_transforms, 
                            train=False)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False)
    test__loader = DataLoader(test__set, batch_size=args.batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    gen = Generator(nethparams, device).to(device)
    dis = Discriminator(nethparams, device).to(device)

    if args.opti == 'adam':
        gen_opti = torch.optim.Adam(gen.parameters(), lr=args.genlr)
        dis_opti = torch.optim.Adam(dis.parameters(), lr=args.dislr)
    else:
        gen_opti = torch.optim.SGD(gen.parameters(), lr=args.genlr)
        dis_opti = torch.optim.SGD(dis.parameters(), lr=args.dislr)

    train_gan(args.epochs, gen, dis, train_loader, valid_loader, gen_opti, dis_opti, nethparams, device)
    test_gan(gen, dis, test__loader, gen_opti, dis_opti, nethparams, device)