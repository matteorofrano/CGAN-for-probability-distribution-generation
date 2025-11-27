import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
from matplotlib.pyplot import imshow, imsave
from typing import Callable, Optional, List, Dict
import os
import json
from utilities import TensorDataset, DataSimulator, prepare_data, pd


def get_generated_data(G, trajectory, G_noise=100):
    device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    z = torch.randn(trajectory.size(0), G_noise).to(device)
    y_hat = G(z,trajectory) #generator takes in input z rand vector and c condition
    result = y_hat.cpu().data.numpy()

    return result



class MyDiscriminator(nn.Module):
    """
        Simple Discriminator w/ MLP
    """
    # num_classes = 1 because output is real or fake
    def __init__(self, input_size=260, condition_size=22, output_dim=1, 
                 hidden_dims:List[int] = [256, 128], use_batch_norm:bool = False,
                 activation:str = 'leaky_relu', dropout:float = 0.0):
        
        super(MyDiscriminator, self).__init__()

        # Store configuration for saving/loading
        self.input_size = input_size
        self.condition_size = condition_size
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.use_batch_norm = use_batch_norm
        self.activation = activation
        self.dropout = dropout
        self.act_fn = self._get_activation(activation)

        input_dim = input_size + condition_size
        layers = []
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(input_dim, hidden_dim))

            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            layers.append(self.act_fn)

            if dropout>0.0:
                layers.append(nn.Dropout(dropout))

            input_dim = hidden_dim

        
        layers.append(nn.Linear(input_dim, output_dim))
        layers.append(nn.Sigmoid())
        self.layer = nn.Sequential(*layers)


    
    def forward(self, x, c):        
        x, c = x.view(x.size(0), -1), c.view(c.size(0), -1).float()
        v = torch.cat((x, c), 1) # v: [input, label] concatenated vector
        y_ = self.layer(v)
        return y_
    

    
    def get_config(self) -> dict:
        
        return {
            'input_size': self.input_size,
            'condition_size': self.condition_size,
            'output_dim': self.output_dim,
            'hidden_dims': self.hidden_dims,
            'use_batch_norm': self.use_batch_norm,
            'activation': self.activation,
            'dropout': self.dropout
        }
    


    def save(self, filepath: Optional[str] = None):
        
        if filepath is None:
            filepath = "discriminator.pth"

        save_dict = {
            'model_state_dict': self.state_dict(),
            'model_architecture': self.__class__.__name__,
            'architecture_params': self.get_config()
        }

        torch.save(save_dict, filepath)
        print(f"Discriminator saved to {filepath}")
    


    @classmethod
    def load(cls, filepath: str, device: Optional[torch.device] = None):
        """
        Load a saved discriminator model
            
        Returns:
            Loaded discriminator instance
        """
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Discriminator file not found at {filepath}")

        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(filepath, map_location=device)

        # Get architecture parameters
        architecture_params = checkpoint.get('architecture_params', {})

        if not architecture_params:
            raise ValueError(
                "No architecture parameters found in checkpoint. "
                "Cannot reconstruct the model."
            )

        # Create new instance with saved parameters
        model = cls(**architecture_params)

        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()

        print(f"Discriminator loaded from {filepath}")
        print(f"Architecture: {checkpoint.get('model_architecture', 'Unknown')}")
        print(f"Parameters: {architecture_params}")

        return model
    


    def _get_activation(self, activation: str):
        
        activations = {
            'leaky_relu': nn.LeakyReLU(0.2),
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'elu': nn.ELU(),
            'sigmoid': nn.Sigmoid()
            }
        
        return activations.get(activation.lower(), nn.LeakyReLU(0.2))




class MyGenerator(nn.Module):
    """
        Generator
    """

    def __init__(self, latent_size:int=260, condition_size:int=22, output_dim:int=2, 
                 hidden_dims:List[int] = [128, 256, 128], use_batch_norm:bool = True, 
                 activation:str = 'leaky_relu', dropout:float = 0.0):
        

        
        super(MyGenerator, self).__init__()

        # Store configuration for saving/loading
        self.latent_size = latent_size
        self.condition_size = condition_size
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.use_batch_norm = use_batch_norm
        self.activation = activation
        self.dropout = dropout
        self.act_fn = self._get_activation(activation)

        #build network
        input_dim = latent_size+condition_size
        layers = []
        for i, hidden_dim in enumerate(hidden_dims):
            # for now just linear layer
            layers.append(nn.Linear(input_dim, hidden_dim))

            if use_batch_norm and i > 0:  
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            layers.append(self.act_fn)
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            
            input_dim = hidden_dim
            
        layers.append(nn.Linear(input_dim, output_dim))

        #Sequential model
        self.layer = nn.Sequential(*layers)
        
    def forward(self, x, c):
        x, c = x.view(x.size(0), -1), c.view(c.size(0), -1).float()
        v = torch.cat((x, c), 1) # v: [input, label] concatenated vector
        y_ = self.layer(v)
        return y_    


    def get_config(self):
        return {
            'latent_size': self.latent_size,
            'condition_size': self.condition_size,
            'output_dim': self.output_dim,
            'hidden_dims': self.hidden_dims,
            'use_batch_norm': self.use_batch_norm,
            'activation': self.activation,
            'dropout': self.dropout
            }
    
    def save(self, filepath: str|None = None):
        """
        Save the generator model with complete architecture information
        
        Args:
            filepath: Path to save the generator. If None, uses default naming
            **architecture_params: Architecture parameters used when creating the generator
                                (condition_size, output_dim, hidden_dims, etc.)
        """
        
        if filepath is None:
            filepath = f"generator.pth"
        
        # Store all architecture parameters for perfect reconstruction
        save_dict = {
            'model_state_dict': self.state_dict(),
            'model_architecture': self.__class__.__name__,
            'architecture_params': self.get_config()  # Save ALL architecture parameters
        }
        
        torch.save(save_dict, filepath)
        print(f"Generator saved to {filepath}")

    
    @classmethod
    def load(cls, filepath: str, device: Optional[torch.device] = None):
        """
        Load a saved generator model
            
        Returns:
            Loaded generator instance
        """
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Generator file not found at {filepath}")
        
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        checkpoint = torch.load(filepath, map_location=device)
        
        # Get architecture parameters
        architecture_params = checkpoint.get('architecture_params', {})
        
        if not architecture_params:
            raise ValueError(
                "No architecture parameters found in checkpoint. "
                "Cannot reconstruct the model."
            )
        
        # Create new instance with saved parameters
        model = cls(**architecture_params)
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        print(f"Generator loaded from {filepath}")
        print(f"Architecture: {checkpoint.get('model_architecture', 'Unknown')}")
        print(f"Parameters: {architecture_params}")
        
        return model


    def _get_activation(self, activation):
        """Get activation function by name"""
        activations = {
            'leaky_relu': nn.LeakyReLU(0.2),
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'elu': nn.ELU(),
            'sigmoid': nn.Sigmoid()
        }
        return activations.get(activation.lower(), nn.LeakyReLU(0.2))
    


class MyCGAN():

    

    """
    Cgan architecture 
    """
    def __init__(self, max_epoch:int = 100, batch_size = 32, n_critic:int = 1,
                  z_noise_dim:int = 252, loss_fn:Callable =  nn.BCELoss(), name:str = 'ConditionalGAN'):
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
        self.MODEL_NAME = name
        self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # generator architecture
    def set_generator(self, condition_size=100, output_dim=2, **generator_params):
        """
        create the generator
        """
        
        self.G=MyGenerator(latent_size=self.z_dim, condition_size=condition_size, output_dim=output_dim, **generator_params)

    #discriminator architecture 
    def set_discriminator(self, input_size=784, condition_size=10, output_dim=1, **discriminator_params):
        """
        create the discriminator
        """

        self.D = MyDiscriminator(input_size=input_size, condition_size=condition_size, output_dim=output_dim, **discriminator_params)


    def train(self, data: TensorDataset, save_history:bool = False):
        """
        train process
        """

        if self.D is None or self.G is None:
            raise Exception("Discriminator or Generator is not defined. Use set_discriminator or set_generator to initialize them")
        
        if isinstance(data, TensorDataset)==False:
            raise Exception(f"invalid input data format. A TensorDataset should be provided. Provided {type(data)}")
        
        # Move models to device
        self.G.to(self.DEVICE)
        self.D.to(self.DEVICE)
        
        data_loader = DataLoader(dataset=data, batch_size=self.batch_size, shuffle=True, drop_last=True)
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
                x=prob_dist.to(self.DEVICE)
                y=trajectory.to(self.DEVICE)
                current_batch_size = x.size(0)

                # Create labels with correct batch size
                D_labels = torch.ones([current_batch_size, 1]).to(self.DEVICE) # Discriminator Label to real
                D_fakes = torch.zeros([current_batch_size, 1]).to(self.DEVICE) # Discriminator Label to fake

                #TRAIN DISCRIMINATOR
                #real samples
                x_outputs = self.D(x, y) # is the observed trajectory from t0 to t1 given the next n days probability distribution real? problem 2
                D_x_loss = self.loss_fn(x_outputs, D_labels) #kl?

                #fake samples
                z = torch.randn(current_batch_size, self.z_dim).to(self.DEVICE)
                z_outputs = self.D(self.G(z, y), y)
                D_z_loss = self.loss_fn(z_outputs, D_fakes)

                #backpropagation
                D_loss = D_x_loss + D_z_loss 
                self.D.zero_grad()
                D_loss.backward()
                D_opt.step()
                
                if step % self.n_critic == 0:
                    # TRAIN GENERATOR
                    z = torch.randn(current_batch_size, self.z_dim).to(self.DEVICE)
                    z_outputs = self.D(self.G(z, y), y)
                    G_loss = self.loss_fn(z_outputs, D_labels)
                    
                    #backpropagation
                    self.G.zero_grad()
                    G_loss.backward()
                    G_opt.step()
                
                if step % 500 == 0:
                    print('Epoch: {}/{}, Step: {}, D Loss: {}, G Loss: {}'.format(epoch, self.max_epoch, step, D_loss.item(), G_loss.item())) #type:ignore
                

                #store history for each middle batch of the epoch
                if save_history and idx == int(len(data_loader)/2): 
                    self.G.eval()
                    generated = get_generated_data(self.G, y, self.z_dim) #batch_size number of generated data
                    for i, row in enumerate(generated):
                        pred = row.tolist()
                        true = x[i, :].cpu().tolist()
                        predictions_list.append(pred)
                        targets_list.append(true)

                        condition_list.append(y[i, :].cpu().tolist())
                        D_loss_list.append(round(float(D_loss), 4))
                        G_loss_list.append(round(float(G_loss), 4)) #type:ignore
                        
                    self.G.train()

                step += 1

            # Build a DataFrame where each list becomes a column
            if len(predictions_list)>1:
                distance = np.linalg.norm(np.array(predictions_list) - np.array(targets_list))
                entries = pd.DataFrame({
                    "epoch": [int(epoch)]*len(predictions_list),
                    "generated": [json.dumps(pred) for pred in predictions_list],
                    "true": [json.dumps(true) for true in targets_list],
                    "D_loss": D_loss_list,
                    "G_loss" : G_loss_list,
                    "distance": distance
                    })
                
                if df is None:
                    df = entries
                else:
                    df = pd.concat([df, entries], ignore_index=True)

        if df is not None:
            df.to_csv("generated_vs_true.csv", index=False)
        


    def predict(self, data: TensorDataset):
        """
        Generate predictions using the trained Generator
        
        Args:
            data: TensorDataset containing conditions (trajectory data)

        Returns:
            predictions: Generated probability distributions
            conditions: Corresponding trajectory conditions
        """
        
        if self.G is None:
            raise Exception("Generator is not defined. Train the model first or use set_generator to initialize it")
        
        if isinstance(data, TensorDataset) == False:
            raise Exception(f"Invalid input data format. A TensorDataset should be provided. Provided {type(data)}")
        
        self.G.eval()
        self.G.to(self.DEVICE)
        
        data_loader = DataLoader(dataset=data, batch_size=self.batch_size, shuffle=False)
        
        predictions_list = []
        conditions_list = []
        
        with torch.no_grad():
            for idx, batch in enumerate(data_loader):
                # Handle both (prob_dist, trajectory) and (trajectory,) formats
                if len(batch) == 2:
                    _, trajectory = batch
                else:
                    trajectory = batch[0]
                
                y = trajectory.to(self.DEVICE)
                current_batch_size = y.size(0)
                
                # Generate samples
                z = torch.randn(current_batch_size, self.z_dim).to(self.DEVICE)
                generated = self.G(z, y)
                
                predictions_list.append(generated.cpu())
                conditions_list.append(y.cpu())
        
        # Concatenate all predictions
        predictions = torch.cat(predictions_list, dim=0)
        conditions = torch.cat(conditions_list, dim=0)
        
        return predictions, conditions
    

    def evaluate_error_distribution(self, data: TensorDataset, save_to:str|None = None) -> dict:
        """
        Compute element-wise error distribution between generated and true samples.
        
        Args:
            data: TensorDataset containing (true_samples, conditions)
            save_path: Optional path to save results as CSV
        
        Returns:
            Dictionary containing errors and statistics
        """
        # Get true values from dataset
        true = data.tensors[0].numpy()
        
        # Use existing predict method
        generated, conditions = self.predict(data)
        generated = generated.numpy()
        conditions = conditions.numpy()
        
        # Compute errors
        errors = generated - true
        
        stats = {
            "errors": errors,
            "generated": generated,
            "true": true,
            "conditions": conditions,
            "mean": np.mean(errors, axis=0),
            "std": np.std(errors, axis=0),
            "median": np.median(errors, axis=0)
        }
        
        # Save to CSV if path provided
        if save_to:
            n_dims = errors.shape[1]
            
            df_dict = {}
            for dim in range(n_dims):
                df_dict[f"generated_{dim}"] = generated[:, dim]
                df_dict[f"true_{dim}"] = true[:, dim]
                df_dict[f"error_{dim}"] = errors[:, dim]
                

            save_path = os.path.join("./data/results/", save_to)
            pd.DataFrame(df_dict).to_csv(save_path, index=False)
            print(f"Results saved to {save_path}")
        
        return stats
    

        

    def save_generator(self, filepath: Optional[str] = None):
        """Save the generator using its built-in save method"""
        if self.G is None:
            raise Exception("Generator is not defined. Nothing to save.")
        
        if filepath is None:
            filepath = f"{self.MODEL_NAME}_generator.pth"
        
        self.G.save(filepath)



    def save_discriminator(self, filepath: Optional[str] = None):
        """Save the discriminator using its built-in save method"""
        if self.D is None:
            raise Exception("Discriminator is not defined. Nothing to save.")
        
        if filepath is None:
            filepath = f"{self.MODEL_NAME}_discriminator.pth"
        
        self.D.save(filepath)


    def save_models(self, save_dir: str = "./models"):
        """
        Save both generator and discriminator models
        
        Args:
            save_dir: Directory to save the models
        """
        os.makedirs(save_dir, exist_ok=True)
        
        gen_path = os.path.join(save_dir, f"{self.MODEL_NAME}_generator.pth")
        disc_path = os.path.join(save_dir, f"{self.MODEL_NAME}_discriminator.pth")
        
        self.save_generator(gen_path)
        self.save_discriminator(disc_path)
        
        # Also save CGAN-level config
        cgan_config = {
            'max_epoch': self.max_epoch,
            'batch_size': self.batch_size,
            'n_critic': self.n_critic,
            'z_dim': self.z_dim,
            'model_name': self.MODEL_NAME
        }
        config_path = os.path.join(save_dir, f"{self.MODEL_NAME}_config.json")
        with open(config_path, 'w') as f:
            json.dump(cgan_config, f, indent=2)
        print(f"CGAN config saved to {config_path}")



    def load_generator(self, filepath: str):
        """
        Load a saved generator model
        
        Args:
            filepath: Path to the saved generator file
        """
        self.G = MyGenerator.load(filepath, device=self.DEVICE)
        
        # Update z_dim from loaded model
        self.z_dim = self.G.latent_size



    def load_discriminator(self, filepath: str):
        """
        Load a saved discriminator model
        
        """
        self.D = MyDiscriminator.load(filepath, device=self.DEVICE)

        


    def load_models(self, load_dir: str = "./models"):
        """
        Load both generator and discriminator models
        
        Args:
            load_dir: Directory containing the saved models
        """
        gen_path = os.path.join(load_dir, f"{self.MODEL_NAME}_generator.pth")
        disc_path = os.path.join(load_dir, f"{self.MODEL_NAME}_discriminator.pth")
        config_path = os.path.join(load_dir, f"{self.MODEL_NAME}_config.json")
        
        # Load CGAN config if exists
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                cgan_config = json.load(f)

            self.max_epoch = cgan_config.get('max_epoch', self.max_epoch)
            self.batch_size = cgan_config.get('batch_size', self.batch_size)
            self.n_critic = cgan_config.get('n_critic', self.n_critic)
            self.z_dim = cgan_config.get('z_dim', self.z_dim)
            print(f"CGAN config loaded from {config_path}")
        
        self.load_generator(gen_path)
        self.load_discriminator(disc_path)


    def get_config(self) -> Dict:
        """Get complete configuration of the CGAN"""

        config = {
            'cgan': {
                'max_epoch': self.max_epoch,
                'batch_size': self.batch_size,
                'n_critic': self.n_critic,
                'z_dim': self.z_dim,
                'model_name': self.MODEL_NAME
            },
            'generator': self.G.get_config() if self.G else None,
            'discriminator': self.D.get_config() if self.D else None
        }
        return config
            





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

    paths = sim.get_paths()
    pdfs= sim.get_pdf(n_steps_ahead=10)

    #train myCGAN
    mydata, mean, std = prepare_data(pdfs, paths)


    conditional_gan = MyCGAN(max_epoch=200)
    conditional_gan.set_generator(condition_size=paths.shape[1])
    conditional_gan.set_discriminator(input_size=pdfs.shape[1], condition_size=paths.shape[1])
    conditional_gan.train(mydata)
