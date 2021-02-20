import torch
from torchvision import datasets
from torchvision import transforms as tfs
from torch.utils.data import Dataset
import os
import numpy as np 
from PIL import Image
import csv
from math import sqrt, acos, asin, pi

class RobotDataset(Dataset):
    def __init__(self, path, latent_dim, sequence_length, transforms):
        scenes = os.listdir(path)
        self.imgspaths = []
        self.trajpaths = []
        self.sequence_length = sequence_length
        self.transforms = transforms

        for maindir in scenes:
            samples = os.listdir(path + '/' + maindir)
            samples = sorted(samples)
            for sample in samples:
                if sample[0] == 'R':
                    self.imgspaths.append(path + '/' + maindir + '/' + sample)
                if sample[0] == 'T':
                    self.trajpaths.append(path + '/' + maindir + '/' + sample)

        # self.noise = torch.randn((len(self.trajpaths), latent_dim))
        self.noise = torch.randn((len(self.trajpaths), sequence_length, latent_dim))

    def __getitem__(self, idx):
        imgs = sorted(os.listdir(self.imgspaths[idx]))
        if len(imgs) == 0:
            raise Exception("Empty folder with", self.imgspaths[idx])
        for i in range(len(imgs)): 
            img = Image.open(self.imgspaths[idx] + '/' + imgs[i])
            if (i == 0):
                sec_tensor = self.transforms(img)
                c, w, h = sec_tensor.shape
                sec_tensor = sec_tensor.view(1, c, w, h)
            else:
                next_frame = self.transforms(img)
                next_frame = next_frame.view(1, c, w, h)
                sec_tensor = torch.cat((sec_tensor, next_frame))

        for j in range(len(imgs), self.sequence_length): #Zero padding to fit sequence length
            next_frame = torch.zeros((1, c, w, h))
            sec_tensor = torch.cat((sec_tensor, next_frame))

        trajectory = []
        velocity = []
        target = []
        header = True
        curr_row = 0

        with open(self.trajpaths[idx], newline='') as File:  
            reader = csv.reader(File)
            for row in reader:
                if (header):
                    header = False
                    continue
                curr_row += 1
                x = float(row[0]) #Cambiar a sistema en polares
                y = float(row[1])
                rho = sqrt(x * x + y * y)
                #theta = acos(x/rho) - pi/2 #Polar angle of robot
                ang = (float(row[4]) - 180) * pi/180 #Orientation of robot
                vx = float(row[2]) #Cambiar a sistema en polares
                vy = float(row[3])
                vrho = sqrt(vx * vx + vy * vy)
                vang = float(row[5]) * pi/180
                rho_target = float(row[6])
                theta_target = float(row[7])
                trajectory.append([x, y, ang])
                velocity.append([vx, vy, vang])
                target.append([rho_target, theta_target])

            for i in range(curr_row, self.sequence_length): #Zero padding to fit sequence length
                trajectory.append([0.0, 0.0, 0.0])
                velocity.append([0.0, 0.0, 0.0])
                target.append([0.0, 0.0])

        trajectory = torch.tensor(np.array(trajectory, dtype=np.float32))
        velocity = torch.tensor(np.array(velocity, dtype=np.float32))
        target = torch.tensor(np.array(target, dtype=np.float32))

        # noise = self.noise[idx].view(1, -1).clone()
        # for i in range(1, self.past_length):
        #     noise = torch.cat((noise, self.noise[idx].view(1, -1).clone()))

        noise = self.noise[idx]

        batch = {
            'noise' : noise,
            'imgs' : sec_tensor, 
            'trajectory' : trajectory,
            'velocity' : velocity,
            'target' : target
        }

        return batch

    def __len__(self):
        return len(self.trajpaths)

def dataset_explore(path):
    scenes = os.listdir(path)
    longest = 0
    for maindir in scenes:
        samples = os.listdir(path + '/' + maindir)
        samples = sorted(samples)
        for sample in samples:
            if sample[0] == 'R':
                imgspath = path + '/' + maindir + '/' + sample
                countimgs = os.listdir(imgspath)
                if len(countimgs) > longest:
                    longest = len(countimgs)

    return longest
