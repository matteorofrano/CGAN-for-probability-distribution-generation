import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
import datetime
import os, sys
from matplotlib.pyplot import imshow, imsave
from typing import Callable


def get_sample_image(G, n_noise=100):
    """
        save sample 100 images
    """
    device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    img = np.zeros([280, 280])
    for j in range(10): #class j
        c = torch.zeros([10, 10]).to(device) #10x10 grid
        c[:, j] = 1
        z = torch.randn(10, n_noise).to(device)
        y_hat = G(z,c).view(10, 28, 28) #generator takes in input z rand vector and c class
        result = y_hat.cpu().data.numpy()
        img[j*28:(j+1)*28] = np.concatenate([x for x in result], axis=-1) #produce final grid
    return img

def to_onehot(x, num_classes=10):
    assert isinstance(x, int) or isinstance(x, (torch.LongTensor, torch.cuda.LongTensor))
    if isinstance(x, int):
        c = torch.zeros(1, num_classes).long() # vectors of zeros with d=num_classes
        c[0][x] = 1 #at position x set value to 1
    else:
        x = x.cpu()
        c = torch.LongTensor(x.size(0), num_classes) # create a tensor of size (batch_size, num_classes) where batch size = size of x along dimension 0
        c.zero_()
        c.scatter_(1, x, 1) # dim, index, src value -> at second dimension at index x set value to 1
    return c


class Discriminator(nn.Module):
    """
        Simple Discriminator w/ MLP
    """
    # num_classes = 1 because output is real or fake
    def __init__(self, input_size=784, condition_size=10, num_classes=1):
        super(Discriminator, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_size+condition_size, 512), # z input size + c size
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, num_classes),
            nn.Sigmoid(),
        )
    
    def forward(self, x, c):        
        x, c = x.view(x.size(0), -1), c.view(c.size(0), -1).float()
        v = torch.cat((x, c), 1) # v: [input, label] concatenated vector
        y_ = self.layer(v)
        return y_
    



class Generator(nn.Module):
    """
        Simple Generator w/ MLP
    """
    def __init__(self, input_size=100, condition_size=10, num_classes=784):
        super(Generator, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_size+condition_size, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, num_classes),
            nn.Tanh()
        )
        
    def forward(self, x, c):
        x, c = x.view(x.size(0), -1), c.view(c.size(0), -1).float()
        v = torch.cat((x, c), 1) # v: [input, label] concatenated vector
        y_ = self.layer(v)
        y_ = y_.view(x.size(0), 1, 28, 28)
        return y_
    




class CGAN():

    MODEL_NAME = 'ConditionalGAN'
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    """
    Cgan architecture 
    """
    def __init__(self, max_epoch:int = 100, batch_size = 64, n_critic:int = 1, z_noise_dim:int = 100, loss_fn:Callable =  nn.BCELoss()):
        """
        """

        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.n_critic = n_critic
        self.z_dim = z_noise_dim
        self.loss_fn = loss_fn

        #architecture
        self.G=None
        self.D=None

    # generator architecture
    def set_generator(self, input_size=100, condition_size=10, num_classes=784, **generator_params):
        """
        create the generator
        """
        
        generator=Generator(input_size=input_size, condition_size=condition_size, num_classes=num_classes, **generator_params)
        self.G=generator

    #discriminator architecture 
    def set_discriminator(self, input_size=784, condition_size=10, num_classes=1, **discriminator_params):
        """
        create the discriminator
        """

        discriminator = Discriminator(input_size=input_size, condition_size=condition_size, num_classes=num_classes, **discriminator_params)
        self.D=discriminator


    def train(self, data):
        """
        train process
        """

        if self.D is None or self.G is None:
            raise Exception("Discriminator or Generator is not defined. Use set_discriminator or set_generator to initialize them")
        
        MODEL_NAME = 'ConditionalGAN'
        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        data_loader = DataLoader(dataset=data, batch_size=self.batch_size, shuffle=True, drop_last=True)
        D_labels = torch.ones([self.batch_size, 1]).to(DEVICE) # Discriminator Label to real
        D_fakes = torch.zeros([self.batch_size, 1]).to(DEVICE) # Discriminator Label to fake

        D_opt = torch.optim.Adam(self.D.parameters(), lr=0.0002, betas=(0.5, 0.999))
        G_opt = torch.optim.Adam(self.G.parameters(), lr=0.0002, betas=(0.5, 0.999))

        step=0
        for epoch in range(self.max_epoch):
            for idx, (images, labels) in enumerate(data_loader):
                # Training Discriminator
                x = images.to(DEVICE) # observed trajectory - toy model BS
                y = labels.view(self.batch_size, 1) # n-days probability distribution pdf (parameters S0, sigma) 
                y = to_onehot(y).to(DEVICE)
                x_outputs = self.D(x, y)
                D_x_loss = self.loss_fn(x_outputs, D_labels) #kl?

                z = torch.randn(self.batch_size, self.z_dim).to(DEVICE)
                z_outputs = self.D(self.G(z, y), y)
                D_z_loss = self.loss_fn(z_outputs, D_fakes)
                D_loss = D_x_loss + D_z_loss
                
                self.D.zero_grad()
                D_loss.backward()
                D_opt.step()
                
                if step % self.n_critic == 0:
                    # Training Generator
                    z = torch.randn(self.batch_size, self.z_dim).to(DEVICE)
                    z_outputs = self.D(self.G(z, y), y)
                    G_loss = self.loss_fn(z_outputs, D_labels)

                    self.G.zero_grad()
                    G_loss.backward()
                    G_opt.step()
                
                if step % 500 == 0:
                    print('Epoch: {}/{}, Step: {}, D Loss: {}, G Loss: {}'.format(epoch, self.max_epoch, step, D_loss.item(), G_loss.item()))
                
                if step % 1000 == 0:
                    self.G.eval()
                    img = get_sample_image(self.G, self.z_dim)
                    imsave('samples/{}_step{}.jpg'.format(MODEL_NAME, str(step).zfill(3)), img, cmap='gray')
                    self.G.train()
                step += 1


    





if __name__=="__main__":
    transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5],
                                std=[0.5])]
                                )
    mnist = datasets.MNIST(root='../data/', train=True, transform=transform, download=True)


    conditional_gan = CGAN()
    conditional_gan.set_generator()
    conditional_gan.set_discriminator()
    conditional_gan.train(mnist)
