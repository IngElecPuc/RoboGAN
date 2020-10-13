import argparse
import torch
import torch.nn as nn
from torchvision import transforms as tfs
from torch.utils.data import DataLoader
from dataset import RobotDataset
from generator import Generator
from discriminator import Discriminator
from utils import basic_parameters
import numpy as np 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for training')
    parser.add_argument('-ld', '--latent_dim', help='Z latent dimension', default=128, type=int)
    parser.add_argument('-hl', '--history_length', help='history window', default=12, type=int)
    parser.add_argument('-fl', '--future_length', help='prediction steps', default=8, type=int)
    parser.add_argument('-w', '--width', help='image width', default=320, type=int)
    parser.add_argument('-h', '--height', help='image height', default=239, type=int)
    parser.add_argument('-enc', '--enc_layers', help='encoder layers', default=2, type=int)
    parser.add_argument('-lstm', '--lstm_dim', help='lstm latent dimension', default=128, type=int)
    parser.add_argument('-out', '--out_layers', help='decoder layers', default=2, type=int)
    parser.add_argument('-e', '--epochs', help='training epochs', default=20, type=int)
    parser.add_argument('-bk', '--backbone', help='CNN backbone [CNN_own, resnet18]',default='CNN_own', type=str)
    parser.add_argument('-at', '--attention', help='type of attention [add, mult]', default='add', type=str)
    parser.add_argument('-op', '--opti', help='type of optimizator [adam, sgd]', default='adam', type=str)
    parser.add_argument('-glr','--genlr',help='generator learning rate', default=1e-3, type=float)
    parser.add_argument('-dlr','--dislr',help='discriminator rate', default=1e-3, type=float)
    parser.add_argument('-bz', '--batch_size', help='batch_size', default=16, type=int)
    parser.add_argument('-wforce', '--num_workers', help='workers number', default=1, type=int)
    args = parser.parse_args()

    netparams = basic_parameters(w = args.w, 
                            h = args.h, 
                            latent_dim=args.ld, 
                            history_length=args.hl, 
                            future_length=args.fl,
                            enc_layers=args.enc,
                            lstm_dim=args.lstm,
                            out_layers=args.out,
                            attention=args.at)                            

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_transforms = tfs.Compose([tfs.Resize((320, 239)),
                               tfs.ToTensor(),
                               tfs.Normalize([0.5], [0.5])])

    train_set = RobotDataset('train', 
                            args.ld, 
                            args.hl, 
                            data_transforms, 
                            train=True)
    valid_set = RobotDataset('valid', 
                            args.ld, 
                            args.hl, 
                            data_transforms, 
                            train=False)
    test__set = RobotDataset('test', 
                            args.ld, 
                            args.hl, 
                            data_transforms, 
                            train=False)

    train_loader = DataLoader(train_set, batch_size=args.bz, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=args.bz, shuffle=False)
    test__loader = DataLoader(test__set, batch_size=args.bz, shuffle=False)

    gen = Generator(netparams).to(device)
    dis = Discriminator(netparams).to(device)

    if args.opti == 'adam':
        gen_opti = torch.optim.Adam(gen.parameters(), lr=args.glr)
        dis_opti = torch.optim.Adam(dis.parameters(), lr=args.dlr)
    else:
        gen_opti = torch.optim.SGD(gen.parameters(), lr=args.glr)
        dis_opti = torch.optim.SGD(dis.parameters(), lr=args.dlr)

    train_gan(args.e, gen, dis, train_loader, valid_loader, gen_opti, dis_opti)
    test_gan(gen, dis, test__loader, gen_opti, dis_opti)