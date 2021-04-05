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
    parser = argparse.ArgumentParser(description='for testing')
    parser.add_argument('--gen_weights', help='Name of generator weigths file', default='gen.pth', type=str)
    parser.add_argument('--latent_dim', help='Z latent dimension', default=512, type=int)
    parser.add_argument('--history_length', help='history window', default=8, type=int)
    parser.add_argument('--future_length', help='prediction steps', default=12, type=int)
    parser.add_argument('--width', help='image width', default=320, type=int)
    parser.add_argument('--height', help='image height', default=239, type=int)
    parser.add_argument('--enc_layers', help='encoder layers', default=2, type=int)
    parser.add_argument('--lstm_dim', help='lstm latent dimension', default=256, type=int)
    parser.add_argument('--output_dim', help='generator ouptut dimension', default=2, type=int)
    parser.add_argument('--backbone', help='CNN backbone [CNN_own, resnet18]',default='CNN_own', type=str)
    parser.add_argument('--attention', help='type of attention [add, mult]', default='add', type=str)
    parser.add_argument('--gopti', help='type of optimizator for generator [adam, sgd]', default='adam', type=str)
    parser.add_argument('--dopti', help='type of optimizator for discriminator [adam, sgd]', default='sgd', type=str)
    parser.add_argument('--genlr', help='generator learning rate', default=1e-3, type=float)
    parser.add_argument('--dislr', help='discriminator rate', default=1e-3, type=float)
    parser.add_argument('--batch_size', help='batch_size', default=32, type=int)
    parser.add_argument('--num_workers', help='workers number', default=1, type=int)
    args = parser.parse_args()

    seq_len = max([dataset_explore('train'), dataset_explore('valid'), dataset_explore('test')]) #Max sequence length in the data set

    netparams = hyperparameters(w=args.width, 
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
                               tfs.Normalize([0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5])])

    test__set = RobotDataset('test', 
                            args.latent_dim, 
                            seq_len,
                            data_transforms)

    test__loader = DataLoader(test__set, batch_size=args.batch_size, shuffle=False)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    gen = Generator(netparams, device).to(device)
    checkpoint = torch.load(args.gen_weights)
    gen.load_state_dict(checkpoint)

    dis = Discriminator(netparams, device).to(device)

    if args.gopti == 'adam':
        gen_opti = torch.optim.Adam(gen.parameters(), lr=args.genlr)
    else:
        gen_opti = torch.optim.SGD(gen.parameters(), lr=args.genlr)

    if args.dopti == 'adam':
        dis_opti = torch.optim.Adam(dis.parameters(), lr=args.dislr)
    else:
        dis_opti = torch.optim.SGD(dis.parameters(), lr=args.dislr)

    mini_log = test_gan(gen, 
            dis, 
            test__loader, 
            gen_opti, 
            dis_opti, 
            netparams, 
            device)

#    log['testg_loss'] = mini_log[0]
#    log['testd_loss'] = mini_log[1]
#    log['testd_ADE'] = mini_log[2]
#    log['testd_FDE'] = mini_log[3]
#
#    import json 
#    with open('Training_log_' + args.name + '.txt', 'w') as json_file:
#        json.dump(log, json_file)        