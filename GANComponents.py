
from layers import *
from typing import Optional, List
import torch.nn as nn
import torch
import os

def _xavier_init_weights(m):
    """
    Apply Xavier (Glorot) initialization to linear layers
    
    Xavier initialization sets weights with variance scaled by fan_in and fan_out,
    which helps maintain gradient magnitudes across layers.
    """
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

def _unwrap(model: nn.Module) -> nn.Module:
    """Return model.module if wrapped in DataParallel, else model itself."""
    return model.module if isinstance(model, nn.DataParallel) else model
 

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
        self.apply(_xavier_init_weights)

    def forward(self, x, c):        
        x, c = x.view(x.size(0), -1), c.view(c.size(0), -1)
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

        core = _unwrap(self)
        save_dict = {
            'model_state_dict': core.state_dict(),
            'model_architecture': core.__class__.__name__,
            'architecture_params': core.get_config(),
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
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(filepath, map_location=device, weights_only=True)

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
        self.apply(_xavier_init_weights)

    def forward(self, c, z):
        c, z = c.view(c.size(0), -1), z.view(z.size(0), -1)
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
        core = _unwrap(self)
        save_dict = {
            'model_state_dict': core.state_dict(),
            'model_architecture': core.__class__.__name__,
            'architecture_params': core.get_config(),
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
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        checkpoint = torch.load(filepath, map_location=device, weights_only=True)
        
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


#-------------------RNN COMPONENTS-----------------------------------------

class RnnGenerator(MyGenerator):
    """
    ForGAN generator built around an RNN encoder (LSTM or GRU).
    -------------------------------------------------------------
    1. The condition window c  [batch, condition_size]  is treated as a
       univariate time-series: each of the `condition_size` time steps
       carries exactly 1 scalar feature.  After unsqueeze(-1) it becomes
       [batch, condition_size, 1], which is the correct (batch, seq, features)
       layout for PyTorch RNNs with batch_first=True.
 
    2. The RNN (LSTM or GRU, optionally multi-layer with LayerNorm) encodes
       the sequence.  Only the hidden state of the **last time step** of the
       **last layer** is kept as the condition representation
       h_c  [batch, hidden_dim].
 
    3. h_c is concatenated with the noise vector z  [batch, latent_size]
       to form  [batch, latent_size + hidden_dim].
 
    4. Two dense layers map the concatenation to the scalar (or vector)
       forecast  [batch, output_dim].
 
    Parameters
    ----------
    latent_size      : dimension of the noise vector z
    condition_size   : length of the look-back window (number of time steps)
    n_input_features : number of features per time step in c (1 for univariate)
    output_dim       : dimension of the forecast  x_{t+1}
    hidden_dim       : RNN hidden-state size  (R_G in the paper)
    n_layers         : number of stacked RNN layers
    activation       : activation between the two dense layers
    dropout          : inter-layer dropout (only active when n_layers > 1)
    rnn_layer        : 'lstm' | 'gru'
    use_layer_norm   : if True, use the custom LayerNorm cells from layers.py
    """
 
    def __init__(
        self,
        latent_size: int = 260,
        condition_size: int = 22,
        n_input_features: int = 1,
        output_dim: int = 1,
        hidden_dim: int = 100,
        n_layers: int = 1,
        activation: str = 'leaky_relu',
        dropout: float = 0.0,
        rnn_layer: str = 'lstm',
        use_layer_norm: bool = False,
    ):
        nn.Module.__init__(self)
 
        # ── stored for get_config / save / load ──────────────────────────────
        self.latent_size      = latent_size
        self.condition_size   = condition_size
        self.n_input_features = n_input_features
        self.output_dim       = output_dim
        self.hidden_dim       = hidden_dim
        self.n_layers         = n_layers
        self.activation       = activation
        self.dropout          = dropout
        self.rnn_layer        = rnn_layer
        self.use_layer_norm   = use_layer_norm
        self.act_fn           = self._get_activation(activation)
 
        # ── RNN encoder ───────────────────────────────────────────────────────
        # input_size = n_input_features (1 for a univariate series).
        if use_layer_norm:
            if rnn_layer == 'lstm':
                self.sequential_model = MultiLayerNormLSTM(
                    n_input_features, hidden_dim, n_layers, dropout
                )
            elif rnn_layer == 'gru':
                self.sequential_model = MultiLayerNormGRU(
                    n_input_features, hidden_dim, n_layers, dropout
                )
            else:
                raise ValueError(
                    f'rnn_layer must be "lstm" or "gru". Got "{rnn_layer}".'
                )
        else:
            if rnn_layer == 'lstm':
                self.sequential_model = nn.LSTM(
                    n_input_features, hidden_dim, n_layers,
                    dropout=dropout if n_layers > 1 else 0.0,
                    batch_first=True,
                )
            elif rnn_layer == 'gru':
                self.sequential_model = nn.GRU(
                    n_input_features, hidden_dim, n_layers,
                    dropout=dropout if n_layers > 1 else 0.0,
                    batch_first=True,
                )
            else:
                raise ValueError(
                    f'rnn_layer must be "lstm" or "gru". Got "{rnn_layer}".'
                )
 
        # ── Dense head ────────────────────────────────────────────────────────
        # Input  : [z | h_c]  →  (latent_size + hidden_dim)
        # Output : output_dim  
        input_dense = latent_size + hidden_dim
        self.dense1 = nn.Linear(input_dense, input_dense)
        self.dense2 = nn.Linear(input_dense, output_dim) 
 
        self.dense1.apply(_xavier_init_weights)
        self.dense2.apply(_xavier_init_weights)
 
    def forward(self, c: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        c : (batch, condition_size)          — look-back window
        z : (batch, latent_size)             — noise vector
 
        Returns
        -------
        y : (batch, output_dim)              — probabilistic forecast x_{t+1}
        """
        # Reshape c to (batch, seq_len, n_input_features)
        # For univariate: (batch, condition_size) → (batch, condition_size, 1)
        c = c.unsqueeze(-1)   # (batch, condition_size, n_input_features=1)
 
        if self.use_layer_norm:
            # Custom cells return (h_list, c_list) for LSTM, h_list for GRU.
            # h_list[-1] is the last-layer hidden state: (batch, hidden_dim).
            if self.rnn_layer == 'lstm':
                h_list, _ = self.sequential_model(c)
            else:
                h_list = self.sequential_model(c)
            h_c = h_list[-1]                           # (batch, hidden_dim)
        else:
            h_out, _ = self.sequential_model(c)
            h_c = h_out[:, -1, :]                      # (batch, hidden_dim)
 
        # Concatenate condition representation with noise
        combined = torch.cat((z, h_c), dim=1)          # (batch, latent+hidden)
 
        y = self.act_fn(self.dense1(combined))
        return self.dense2(y)                           # (batch, output_dim)
 
    def get_config(self) -> dict:
        return {
            'latent_size':      self.latent_size,
            'condition_size':   self.condition_size,
            'n_input_features': self.n_input_features,
            'output_dim':       self.output_dim,
            'hidden_dim':       self.hidden_dim,
            'n_layers':         self.n_layers,
            'activation':       self.activation,
            'dropout':          self.dropout,
            'rnn_layer':        self.rnn_layer,
            'use_layer_norm':   self.use_layer_norm,
        }
    

class RnnDiscriminator(MyDiscriminator):
    """
    ForGAN discriminator built around an RNN encoder (LSTM or GRU).
    -------------------------------------------------------------
    The discriminator judges whether a candidate value x_{t+1} is a plausible
    continuation of the condition window c = {x_0, ..., x_t}.  It does so by
    appending x_{t+1} at the **end** of the condition sequence to form the
    extended window  {x_0, ..., x_t, x_{t+1}}, then running an RNN over this
    combined sequence and projecting the final hidden state to a scalar
    real / fake score.
 
    Parameters
    ----------
    input_size       : dimension of x_{t+1}  (1 for scalar target)
    condition_size   : length of the look-back window
    n_input_features : features per time step in the *combined* sequence
                       (1 for a univariate series; multivariate not yet tested)
    output_dim       : discriminator output dimension (1 = real/fake logit)
    hidden_dim       : RNN hidden-state size  (R_D in the paper)
    n_layers         : number of stacked RNN layers
    use_layer_norm   : if True, use the custom LayerNorm cells from layers.py
    activation       : activation name (kept for API compatibility; not used
                       inside forward — the final dense layer has no activation)
    dropout          : inter-layer dropout (only active when n_layers > 1)
    rnn_layer        : 'lstm' | 'gru'
    """
 
    def __init__(
        self,
        input_size: int = 1,
        condition_size: int = 22,
        n_input_features: int = 1,
        output_dim: int = 1,
        hidden_dim: int = 24,
        n_layers: int = 1,
        use_layer_norm: bool = False,
        activation: str = 'leaky_relu',
        dropout: float = 0.0,
        rnn_layer: str = 'lstm',
    ):
        nn.Module.__init__(self)
 
        # ── stored for get_config / save / load ──────────────────────────────
        self.input_size       = input_size
        self.condition_size   = condition_size
        self.n_input_features = n_input_features
        self.output_dim       = output_dim
        self.hidden_dim       = hidden_dim
        self.n_layers         = n_layers
        self.use_layer_norm   = use_layer_norm
        self.activation       = activation
        self.dropout          = dropout
        self.rnn_layer        = rnn_layer
        self.act_fn           = self._get_activation(activation)
 
        # ── RNN encoder ───────────────────────────────────────────────────────
        if use_layer_norm:
            if rnn_layer == 'lstm':
                self.sequential_model = MultiLayerNormLSTM(
                    n_input_features, hidden_dim, n_layers, dropout
                )
            elif rnn_layer == 'gru':
                self.sequential_model = MultiLayerNormGRU(
                    n_input_features, hidden_dim, n_layers, dropout
                )
            else:
                raise ValueError(
                    f'rnn_layer must be "lstm" or "gru". Got "{rnn_layer}".'
                )
        else:
            if rnn_layer == 'lstm':
                self.sequential_model = nn.LSTM(
                    n_input_features, hidden_dim, n_layers,
                    dropout=dropout if n_layers > 1 else 0.0,
                    batch_first=True,
                )
            elif rnn_layer == 'gru':
                self.sequential_model = nn.GRU(
                    n_input_features, hidden_dim, n_layers,
                    dropout=dropout if n_layers > 1 else 0.0,
                    batch_first=True,
                )
            else:
                raise ValueError(
                    f'rnn_layer must be "lstm" or "gru". Got "{rnn_layer}".'
                )
 
        # ── Output projection ─────────────────────────────────────────────────
        self.dense = nn.Linear(hidden_dim, output_dim)
        self.dense.apply(_xavier_init_weights)
 
    # ──────────────────────────────────────────────────────────────────────────
 
    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, input_size)       — candidate next value x_{t+1}
        c : (batch, condition_size)   — look-back window
 
        Returns
        -------
        score : (batch, output_dim)   — real/fake logit (no sigmoid applied)
        """
        # Append x_{t+1} at the end of the condition sequence, then treat
        # each element of the combined vector as one time step (univariate).
        v = torch.cat((c, x), dim=1)   # (batch, condition_size + input_size)
        v = v.unsqueeze(-1)            # (batch, condition_size + input_size, 1)
 
        if self.use_layer_norm:
            # Custom cells return (h_list, c_list) for LSTM, h_list for GRU.
            # h_list[-1] → last-layer hidden state: (batch, hidden_dim).
            if self.rnn_layer == 'lstm':
                h_list, _ = self.sequential_model(v)
            else:
                h_list = self.sequential_model(v)
            h_last = h_list[-1]                        # (batch, hidden_dim)
        else:
            h_out, _ = self.sequential_model(v)
            h_last = h_out[:, -1, :]                   # (batch, hidden_dim)
 
        return self.dense(h_last)                      # (batch, output_dim)
 
    # ──────────────────────────────────────────────────────────────────────────
 
    def get_config(self) -> dict:
        return {
            'input_size':       self.input_size,
            'condition_size':   self.condition_size,
            'n_input_features': self.n_input_features,
            'output_dim':       self.output_dim,
            'hidden_dim':       self.hidden_dim,
            'n_layers':         self.n_layers,
            'use_layer_norm':   self.use_layer_norm,
            'activation':       self.activation,
            'dropout':          self.dropout,
            'rnn_layer':        self.rnn_layer,
        }
