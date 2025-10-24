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
from utilities import TensorDataset, DataSimulator, prepare_data, pd


def get_generated_data(G, trajectory, G_noise=100):
    device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    z = torch.randn(trajectory.size(0), G_noise).to(device)
    y_hat = G(z,trajectory) #generator takes in input z rand vector and c condition
    result = y_hat.cpu().data.numpy()

    return result



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
                y = to_onehot(y).to(DEVICE) #type: ignore
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
                    print('Epoch: {}/{}, Step: {}, D Loss: {}, G Loss: {}'.format(epoch, self.max_epoch, step, D_loss.item(), G_loss.item())) #type:ignore
                
                if step % 1000 == 0:
                    self.G.eval()
                    img = get_sample_image(self.G, self.z_dim) #type: ignore
                    imsave('samples/{}_step{}.jpg'.format(MODEL_NAME, str(step).zfill(3)), img, cmap='gray')
                    self.G.train()
                step += 1

# MY IMPLEMENTATION 
##########################################################################################################################################
##########################################################################################################################################

class MyDiscriminator(nn.Module):
    """
        Simple Discriminator w/ MLP
    """
    # num_classes = 1 because output is real or fake
    def __init__(self, input_size=260, condition_size=22, num_classes=1):
        super(MyDiscriminator, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_size+condition_size, 256), # x input size + y size
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, num_classes),
            nn.Sigmoid(),
        )
    
    def forward(self, x, c):        
        x, c = x.view(x.size(0), -1), c.view(c.size(0), -1).float()
        v = torch.cat((x, c), 1) # v: [input, label] concatenated vector
        y_ = self.layer(v)
        return y_
    



class MyGenerator(nn.Module):
    """
        Simple Generator w/ MLP
    """
    def __init__(self, latent_size=260, condition_size=22, num_classes=2):
        super(MyGenerator, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(latent_size+condition_size, 128), # z noise vector + y size
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x, c):
        x, c = x.view(x.size(0), -1), c.view(c.size(0), -1).float()
        v = torch.cat((x, c), 1) # v: [input, label] concatenated vector
        y_ = self.layer(v)
        #y_ = y_.view(x.size(0), 1, 28, 28)
        return y_
    


class MyCGAN():

    

    """
    Cgan architecture 
    """
    def __init__(self, max_epoch:int = 100, batch_size = 32, n_critic:int = 1, z_noise_dim:int = 252, loss_fn:Callable =  nn.BCELoss()):
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

        #device specific
        self.MODEL_NAME = 'ConditionalGAN'
        self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # generator architecture
    def set_generator(self, condition_size=100, num_classes=2, **generator_params):
        """
        create the generator
        """
        
        self.G=MyGenerator(latent_size=self.z_dim, condition_size=condition_size, num_classes=num_classes, **generator_params)

    #discriminator architecture 
    def set_discriminator(self, input_size=784, condition_size=10, num_classes=1, **discriminator_params):
        """
        create the discriminator
        """

        self.D = MyDiscriminator(input_size=input_size, condition_size=condition_size, num_classes=num_classes, **discriminator_params)


    def train(self, data: TensorDataset):
        """
        train process
        """

        if self.D is None or self.G is None:
            raise Exception("Discriminator or Generator is not defined. Use set_discriminator or set_generator to initialize them")
        
        if isinstance(data, TensorDataset)==False:
            raise Exception(f"invalid input data format. A TensorDataset should be provided. Provided {type(data)}")
        
        data_loader = DataLoader(dataset=data, batch_size=self.batch_size, shuffle=True)
        D_labels = torch.ones([self.batch_size, 1]).to(self.DEVICE) # Discriminator Label to real
        D_fakes = torch.zeros([self.batch_size, 1]).to(self.DEVICE) # Discriminator Label to fake

        D_opt = torch.optim.Adam(self.D.parameters(), lr=0.0005, betas=(0.5, 0.999))
        G_opt = torch.optim.Adam(self.G.parameters(), lr=0.0005, betas=(0.5, 0.999))

        df = None
        step=0
        for epoch in range(self.max_epoch):
            predictions_list = []
            targets_list = []
            condition_list = []
            D_loss_list = []
            G_loss_list = []
            for idx, (prob_dist, trajectory) in enumerate(data_loader):   
                x=prob_dist
                y=trajectory

                x_outputs = self.D(x, y) # is the observed trajectory from t0 to t1 given the next n days probability distribution real? problem 2
                D_x_loss = self.loss_fn(x_outputs, D_labels) #kl?

                z = torch.randn(self.batch_size, self.z_dim).to(self.DEVICE)
                z_outputs = self.D(self.G(z, y), y)
                D_z_loss = self.loss_fn(z_outputs, D_fakes)
                D_loss = D_x_loss + D_z_loss
                
                self.D.zero_grad()
                D_loss.backward()
                D_opt.step()
                
                if step % self.n_critic == 0:
                    # Training Generator
                    z = torch.randn(self.batch_size, self.z_dim).to(self.DEVICE)
                    z_outputs = self.D(self.G(z, y), y)
                    G_loss = self.loss_fn(z_outputs, D_labels)

                    self.G.zero_grad()
                    G_loss.backward()
                    G_opt.step()
                
                if step % 500 == 0:
                    print('Epoch: {}/{}, Step: {}, D Loss: {}, G Loss: {}'.format(epoch, self.max_epoch, step, D_loss.item(), G_loss.item())) #type:ignore
                
                if step % 1000 == 0:
                    self.G.eval()
                    generated = get_generated_data(self.G, y, self.z_dim) #batch_size number of generated data
                    for i, row in enumerate(generated):
                        predictions_list.append(row.tolist())
                        targets_list.append(x[i, :].cpu().data.tolist())
                        condition_list.append(y[i, :])
                        D_loss_list.append(round(float(D_loss), 4))
                        G_loss_list.append(round(float(G_loss), 4)) #type:ignore


                    self.G.train()
                step += 1

            # Stack all batches together
            # Build a DataFrame where each list becomes a column
            if epoch == 0:
                distance = np.linalg.norm(np.array(predictions_list) - np.array(targets_list))
                df = pd.DataFrame({
                    "epoch": [int(epoch)]*len(predictions_list),
                    "generated": predictions_list,
                    "true": targets_list,
                    "D_loss": D_loss_list,
                    "G_loss" : G_loss_list,
                    "distance": distance

                    #"condition": condition_list
                })

            else: #update
                distance = np.linalg.norm(np.array(predictions_list) - np.array(targets_list))
                new_entries = pd.DataFrame({
                "epoch":[int(epoch)]*len(predictions_list),
                "generated": predictions_list,
                "true": targets_list,
                "D_loss": D_loss_list,
                "G_loss" : G_loss_list,
                "distance": distance
                #"condition": condition_list
                })

                df = pd.concat([df, new_entries], ignore_index=True)

        if df is not None:
            df.to_csv("generated_vs_true.csv", index=False)
        else: 
            raise Exception("df is null")


        





if __name__=="__main__":
    # data simulation 
    # example
    X0_range = (0.0,1.0)
    mu_range = (0.0, 0.0)
    sigma_range = (0.001, 1.0)
    T = 1.0        # Time horizon (1 year)
    N = 252        # Number of time steps (trading days in a year)
    J = 100000        # Number of paths to simulate
    SEED=42

    # --- Run the Simulation ---
    sim = DataSimulator(X0_range=X0_range, mu_range=mu_range, sigma_range=sigma_range, 
                                T=T, N=N, n_simulations=J, seed=SEED)

    paths = sim.get_BS_paths()
    pdfs= sim.get_BS_pdf(n_steps_ahead=10)

    #train myCGAN
    mydata = prepare_data(pdfs, paths)


    conditional_gan = MyCGAN(max_epoch=200)
    conditional_gan.set_generator(condition_size=paths.shape[1])
    conditional_gan.set_discriminator(input_size=pdfs.shape[1], condition_size=paths.shape[1])
    conditional_gan.train(mydata)
