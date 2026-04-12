
from layers import *
from typing import Optional, List
import torch.nn as nn
import torch
import os

def xavier_init_weights(m):
    """
    Apply Xavier (Glorot) initialization to linear layers
    
    Xavier initialization sets weights with variance scaled by fan_in and fan_out,
    which helps maintain gradient magnitudes across layers.
    """
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

class MyDiscriminator(nn.Module):
    """
        Simple Discriminator w/ MLP
    """
    # num_classes = 1 because output is real or fake
    def __init__(self, input_size=260, condition_size=22, output_dim=1, 
                 hidden_dims:List[int] = [256, 128], use_layer_norm:bool = False,
                 activation:str = 'leaky_relu', dropout:float = 0.0):
        
        super().__init__()

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

        self.layers = nn.Sequential(*layers)

        # Apply Xavier initialization
        self.apply(xavier_init_weights)

    def forward(self, x, c):        
        x, c = x.view(x.size(0), -1), c.view(c.size(0), -1).float()
        v = torch.cat((x, c), 1) # v: [input, condition] concatenated vector
        y_ = self.layers(v)
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
                 activation:str = 'leaky_relu', dropout:float = 0.0, is_prob:bool = False):
        

        
        super().__init__()

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
        self.layers = nn.Sequential(*layers)

        # Apply Xavier initialization
        self.apply(xavier_init_weights)

    def forward(self, c, z):
        c, z = c.view(c.size(0), -1), z.view(z.size(0), -1).float()
        v = torch.cat((c, z), 1) # v: [trajectory, noise] concatenated vector
        y_ = self.layers(v)
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
    




class RnnGenerator(MyGenerator):
    """
    A generator based on RNN style layers 
    """

    def __init__(self, latent_size:int=260, condition_size:int=22, output_dim:int=1, 
                 hidden_dim:int = 100, n_layers:int = 1, activation:str = 'leaky_relu', 
                 dropout:float = 0.0, rnn_layer:str = 'lstm'):
        

        
        nn.Module.__init__(self)

        # Store configuration for saving/loading
        self.latent_size = latent_size
        self.condition_size = condition_size
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.activation = activation
        self.dropout = dropout
        self.act_fn = self._get_activation(activation)
        self.rnn_layer = rnn_layer

        #build network
        if rnn_layer == 'lstm':
            self.sequential_model = nn.LSTM(output_dim, hidden_dim, n_layers,
                                        dropout=dropout, batch_first=True)
        elif rnn_layer == 'gru':
            self.sequential_model = nn.GRU(output_dim, hidden_dim, n_layers,
                                        dropout=dropout, batch_first=True)
        else:
            raise ValueError(f'Available rnn architectures are the "lstm" and "gru". {rnn_layer} provided instead')
        
        input_dense1 = latent_size+hidden_dim
        self.dense1 = nn.Linear(input_dense1, input_dense1)
        self.dense2 = nn.Linear(input_dense1, 1) 

        # Apply Xavier initialization to dense layers
        # RNN layers use orthogonal initialization by default which is also good
        self.dense1.apply(xavier_init_weights)
        self.dense2.apply(xavier_init_weights) 
        
        
    def forward(self, c, z):
        #x, z = x.view(x.size(0), -1), z.view(z.size(0), -1).float()
        c = c.unsqueeze(-1)
        h_out, _ = self.sequential_model(c)
        
        c = h_out[:, -1, :] #takes the condition representation of the last hidden state
        combined = torch.cat((z, c), dim=1) # v: [trajectory, noise] concatenated vector
        y = self.act_fn(self.dense1(combined))
        y_ = self.dense2(y)

        return y_  


    def get_config(self):
        return {
            'latent_size': self.latent_size,
            'condition_size': self.condition_size,
            'n_layers':self.n_layers,
            'output_dim': self.output_dim,
            'hidden_dim': self.hidden_dim,
            'activation': self.activation,
            'dropout': self.dropout,
            'rnn_layer':self.rnn_layer
            } 
    



class RnnDiscriminator(MyDiscriminator):
    """
    A discriminator based on RNN style layers 
    """

    def __init__(self, input_size=260, condition_size=22, output_dim=1, 
                    hidden_dim:int = 24, use_layer_norm:bool = False,
                    activation:str = 'leaky_relu', dropout:float = 0.0,
                    rnn_layer:str = 'lstm', n_layers:int = 1):

        nn.Module.__init__(self)

        # Store configuration for saving/loading
        self.input_size = input_size
        self.condition_size = condition_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.use_layer_norm = use_layer_norm
        self.activation = activation
        self.dropout = dropout
        self.act_fn = self._get_activation(activation)
        self.rnn_layer = rnn_layer

        #input_dim = input_size + condition_size
        #build network
        if use_layer_norm:
            if rnn_layer == 'lstm':
                self.sequential_model = MultiLayerNormLSTM(input_size, hidden_dim, n_layers, dropout)
            elif rnn_layer == 'gru':
                self.sequential_model = MultiLayerNormGRU(input_size, hidden_dim, n_layers, dropout)
            else:
                raise ValueError(f'Available rnn architectures are the "lstm" and "gru". {rnn_layer} provided instead')
        else:
            if rnn_layer == 'lstm':
                self.sequential_model = nn.LSTM(condition_size, hidden_dim, n_layers,
                                            dropout=dropout, batch_first=True)
            elif rnn_layer == 'gru':
                self.sequential_model = nn.GRU(condition_size, hidden_dim, n_layers,
                                            dropout=dropout, batch_first=True)
            else:
                raise ValueError(f'Available rnn architectures are the "lstm" and "gru". {rnn_layer} provided instead')
            
        self.dense = nn.Linear(hidden_dim, output_dim)

        # Apply Xavier initialization to dense layer
        self.dense.apply(xavier_init_weights)
        
    def forward(self, x, c):        
        #x, c = x.view(x.size(0), -1), c.view(c.size(0), -1).float()
        v = torch.cat((c, x), 1) # v: [condition, input] concatenated vector
        v = v.unsqueeze(-1)

        if self.rnn_layer == 'lstm':
            h_out, _ = self.sequential_model(v)
        elif self.rnn_layer =='gru':
            h_out = self.sequential_model(v)
        else:
            raise ValueError('The RNN based layer is not specified')
        
        y_ = self.dense(h_out[-1])
        return y_
    

    def get_config(self):
        return {
            'input_size': self.input_size,
            'condition_size': self.condition_size,
            'n_layers':self.n_layers,
            'use_layer_norm': self.use_layer_norm,
            'hidden_dim': self.hidden_dim,
            'activation': self.activation,
            'dropout': self.dropout,
            'act_fn':self.act_fn,
            'rnn_layer':self.rnn_layer
            } 



