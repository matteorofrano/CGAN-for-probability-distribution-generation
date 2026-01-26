
from typing import Optional, List
import torch.nn as nn
import torch
import os


class MyDiscriminator(nn.Module):
    """
        Simple Discriminator w/ MLP
    """
    # num_classes = 1 because output is real or fake
    def __init__(self, input_size=260, condition_size=22, output_dim=1, 
                 hidden_dims:List[int] = [256, 128], use_layer_norm:bool = False,
                 activation:str = 'leaky_relu', dropout:float = 0.0):
        
        super(MyDiscriminator, self).__init__()

        # Store configuration for saving/loading
        self.input_size = input_size
        self.condition_size = condition_size
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.use_batch_norm = use_layer_norm
        self.activation = activation
        self.dropout = dropout
        self.act_fn = self._get_activation(activation)

        input_dim = input_size + condition_size
        layers = []
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(input_dim, hidden_dim))

            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))

            layers.append(self.act_fn)

            if dropout>0.0:
                layers.append(nn.Dropout(dropout))

            input_dim = hidden_dim

        
        layers.append(nn.Linear(input_dim, output_dim))
        #layers.append(nn.Sigmoid()) not used if loss_fn is BCEwithLogitLoss or if Wasserstein distance is used

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
            'use_layer_norm': self.use_batch_norm,
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
                 activation:str = 'relu', dropout:float = 0.0, is_prob:bool = False):
        

        
        super(MyGenerator, self).__init__()

        # Store configuration for saving/loading
        self.latent_size = latent_size
        self.condition_size = condition_size
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.use_batch_norm = use_batch_norm
        self.is_prob = is_prob
        self.activation = activation
        self.dropout = dropout
        self.act_fn = self._get_activation(activation)

        #build network
        input_dim = latent_size+condition_size
        layers = []
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(input_dim, hidden_dim))

            if use_batch_norm and i > 0:  
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            layers.append(self.act_fn)
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            
            input_dim = hidden_dim
            
        layers.append(nn.Linear(input_dim, output_dim))
        if is_prob:
            layers.append(nn.LogSoftmax(dim=1)) # dim=1 for batch dimension

        #Sequential model
        self.layer = nn.Sequential(*layers)
        
    def forward(self, x, c):
        x, c = x.view(x.size(0), -1), c.view(c.size(0), -1).float()
        v = torch.cat((x, c), 1) # v: [trajectory, noise] concatenated vector
        y_ = self.layer(v)
        return y_    


    def get_config(self):
        return {
            'latent_size': self.latent_size,
            'condition_size': self.condition_size,
            'output_dim': self.output_dim,
            'hidden_dims': self.hidden_dims,
            'is_prob':self.is_prob,
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
    