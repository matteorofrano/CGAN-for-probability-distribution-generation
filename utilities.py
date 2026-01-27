from typing import Union, Tuple, Text, List
import pandas as pd 
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import torch 
from torch.utils.data import TensorDataset
from scipy.stats import norm, kstwobign
import struct
import ast
import math
import os
import json 
from pathlib import Path
import seaborn as sns
from scipy.stats import ttest_1samp
from scipy.spatial.distance import jensenshannon



def get_data_yf(ticker: str, start: str, end: Union[str, None] = None)-> pd.DataFrame|None:
    
    #session = requests.Session(impersonate="chrome") #type: ignore
    if end is not None:
        stock_df=yf.download(ticker, start=start, end=end)#session=session
    else:
        stock_df=yf.download(ticker, start=start)#session=session

    if stock_df is not None:
        stock_df.columns = [col[0].lower()+'_'+col[1].lower() if col[1] else col[0].lower() for col in stock_df.columns] #we flatten the multi-index structure
        stock_df.index.name = "date"
        stock_df.reset_index(inplace=True)
    
    return stock_df



def prepare_data(X:np.ndarray,C:np.ndarray, eps=1e-9, preprocess:str|None = None):
    """
    An auxiliary function to normalize data and load them into a dataloader
    X is the noise vector or true probability distribution
    C is the condition vector
    """
    #tensorization
    X_tensor = torch.tensor(X, dtype=torch.float32)
    C_tensor = torch.tensor(C, dtype=torch.float32)
    X_mean = None
    X_std = None

    if preprocess:       
        if preprocess == 'standardization':
            X_mean = X_tensor.mean(dim=0)
            X_std = X_tensor.std(dim=0) + eps

            #standardization
            X_tensor = (X_tensor - X_mean) / X_std
            C_tensor = (C_tensor - C_tensor.mean(dim=0)) / (C_tensor.std(dim=0) + eps)
        
        elif preprocess == 'log':
            X_tensor = torch.log(X_tensor + eps) # only target if probabilities
        else:
            raise Exception('Select an available preprocess step. Available are \"standardization\", \"log\"')
    
    dataset = TensorDataset(X_tensor, C_tensor)
    return dataset, X_mean, X_std
        
        


def plot_learning_curve(df_csv:str):
        """
        A function used to plot the learning curve of the trained model
        """

        # Load the CSV file
        df = pd.read_csv(df_csv)

        # Plot distance vs epoch
        plt.figure()
        plt.plot(df["epoch"], df["distance"])
        plt.xlabel("Epoch")
        plt.ylabel("Distance")
        plt.title("Distance over Epochs")
        plt.show()


def plot_bin_dist(trues:np.ndarray, preds:np.ndarray,
                   bins_values:np.ndarray, X_T: List[float]|None = None, ncols=3):

    n = len(trues)
    nrows = math.ceil(n/ncols)
    if n!= len(preds):
        raise ValueError('trues and genereted must have the same length')
    
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5 * ncols, 4 * nrows),
        sharex=True,
        sharey=True
    )

    axes = axes.flatten()
    for i, (true, pred) in enumerate(zip(trues, preds)):
        if len(true)!=len(pred):
            raise Exception(f'true and pred have different shapes. true ={true.shape}, pred = {pred.shape}')

        ax = axes[i]
        ax.plot(bins_values[:-1], true, label="True histogram", linewidth=2)
        ax.plot(bins_values[:-1], pred, label="Generated histogram", linewidth=2)

        if X_T is not None:
            ax.axvline(
                x=X_T[i],
                linestyle=":",
                linewidth=2,
                label=f"X_T={round(X_T[i], 3)}"
            )

        ax.set_title(f"Histogram {i}")
        ax.set_xlabel("Bin values")
        ax.set_ylabel("Probability")
        ax.legend()

    # Remove unused axes
    for j in range(n, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

    



def manage_csv_results(csv:str):

    df = pd.read_csv(csv)
    df["generated"] = df["generated"].apply(ast.literal_eval)
    df["true"] = df["true"].apply(ast.literal_eval)

    return df



def analyze_error_distribution(csv:str):

    df = pd.read_csv(csv)

    errors = [c for c in df.columns if c.startswith("error_")]
    means = df[errors].mean()
    stds = df[errors].std()

    if means.isna().any():
        raise Exception(f'there is a NaN value when computing means in {means[means.isna()]}')
    
    tests = []
    ci = []

    for col in errors:
        values = df[col].dropna()
        result = ttest_1samp(values, 0.0)
        tests.append(result.pvalue >0.05) #type:ignore
        ci.append((result.confidence_interval()))

    summary = pd.DataFrame({
        "mean":     means,
        "std":      stds,
        "median":   df[errors].median(),
        "skew":     df[errors].skew(),
        "kurtosis": df[errors].kurt(),
        "is_zero_test": tests,
        "confidence_interval": ci
    })

    if len(errors)<20:
        plt.figure(figsize=(10, 5))
    else:
        plt.figure(figsize=(30, 8))

    #distribution
    sns.violinplot(data=df[errors], inner=None)

    #mean markers
    plt.scatter(range(len(errors)), means.values, color="black", zorder=3, label="Mean")#type:ignore

    #std bars
    plt.errorbar(range(len(errors)),means.values,yerr=stds.values,fmt="none",ecolor="black",capsize=6,zorder=2) #type:ignore

    plt.title("Error Distributions with Mean ± Std")
    plt.xticks(range(len(errors)), errors, rotation=45, ha="right")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return means, stds, summary
    

def compute_js(generated_arr:np.ndarray, true_arr:np.ndarray, is_log:bool = True):
    """
    Compute the Jenson-Shannon measure 
    """
    js_distances = []
    epsilon = 1e-10 # Small epsilon for numerical stability

    for p, t in zip(generated_arr, true_arr): 
        if is_log:
            p = np.exp(p)
            t = np.exp(t)

        #normalization
        p = p / (np.sum(p) + epsilon)
        t = t / (np.sum(t) + epsilon)
        
        # compute JSD
        js_distances.append(jensenshannon(p, t))

    return js_distances


def ks_test_gan_cdf(generated_probs, true_probs):
    """
    Perform KS test comparing generated and true probability distributions.
    
    generated_probs : array of shape (n_bins,)
    true_probs : array of shape (n_bins,)
    """
    # Convert probabilities to CDFs
    generated_cdf = np.cumsum(generated_probs)
    true_cdf = np.cumsum(true_probs)
    
    # Compute KS statistic
    ks_statistic = np.max(np.abs(generated_cdf - true_cdf))
    
    # Compute p-value
    n = len(generated_probs)
    p_value = kstwobign.sf(ks_statistic * np.sqrt(n))
    
    return ks_statistic, p_value



class DataSimulator():
    """
    Simulation class object for J trajectories 

    Input:
        X0_range (tuple): The initial value in [0.0,10.0] of trajectory j at time t=0
        mu_range (tuple): The drift term in [0.0, 1.0] for trajectory j.
        sigma_range (tuple): The diffusion term in (0.0,1.0] for trajectory j.
        T (float): The total time horizon for the simulation (in years).
        N (int): The number of time steps in the simulation.
        n_simulations (int): The number J of independent trajectories to simulate.

    Returns:
        np.ndarray: A 2D NumPy array of shape (n_simulations, N + 1) where each row
                    represents a simulated trajectory of the log stock price.
    """

    def __init__(self,
                  X0_range: Union[Tuple[float, float], List[float]],
                  mu_range: Union[Tuple[float, float], List[float]],
                  sigma_range: Union[Tuple[float, float], List[float]],
                  T: float, N: int, n_simulations: int, seed:int|None = None):
        
        # Create one generator when the object is created
        self.rng = np.random.default_rng(seed)

        #class parameters
        self.X0_range = X0_range
        self.mu_range = mu_range
        self.sigma_range = sigma_range
        self.T=T
        self.N = N
        self.n_simulations = n_simulations
        self.dt = T/N
        self.sampling_strategies = ['uniform', 'log_uniform']

        #to be sampled
        self.X0 = None
        self.mu = None
        self.sigma = None

        #trajectories, probability density functions and bins
        self.paths=None
        self.X_T = None
        self.pdf=None
        self.bins = None

    def sample_parameter(self, strategy:Text, range: tuple):
        """
        Function used to sample J initial values, mu and sigma for each trajectory j
        """

        if strategy not in self.sampling_strategies:
            raise Exception(f'The provided strategy: {strategy} is not available. The available strategies are {self.sampling_strategies}')
        
        low = range[0]
        high = range[1]
        if not isinstance(high, float) or not isinstance(low, float):
            raise Exception(f'The provided range values should be floats. Provided {type(low), type(high)}.')
        

        
        if strategy == 'uniform':
            sampled_values = self.rng.uniform(low, high, size=self.n_simulations)

        elif strategy == 'loguniform':
            sampled_values = np.exp(self.rng.uniform(low, high, self.n_simulations))

        else:
            sampled_values = np.ndarray(shape=1)

        return sampled_values
    
    def get_paths(self):
        """
        Simulates log prices with Brownian Motion.
        This function simulates M trajectories of the log stock price process over the time horizon [0, T] using N time steps.
        dX_t = (mu - 0.5 * sigma^2) dt + sigma dW_t 

        """

        if isinstance(self.X0_range, Tuple):  
            self.X0 = self.sample_parameter('uniform', self.X0_range)
        elif isinstance(self.X0_range, List):
            self.X0 = np.array(self.X0_range)
        else:
            raise ValueError('X0_range parameter require either a tuple of the form (start,end) or a list of floating values')

        if isinstance(self.mu_range, Tuple):  
            self.mu = self.sample_parameter('uniform', self.mu_range)
        elif isinstance(self.mu_range, List):
            self.mu = np.array(self.mu_range)
        else:
            raise ValueError('mu_range parameter require either a tuple of the form (start,end) or a list of floating values')
        
        if isinstance(self.sigma_range, Tuple):  
            self.sigma = self.sample_parameter('uniform', self.sigma_range)
        elif isinstance(self.sigma_range, List):
            self.sigma = np.array(self.sigma_range)
        else:
            raise ValueError('sigma_range parameter require either a tuple of the form (start,end) or a list of floating values')
        

        for i, sampling_parameter in enumerate([self.X0, self.mu, self.sigma]):
            if len(sampling_parameter)<1:
                dict_map = {0:"X0", 1:"mu", 2:"sigma"}
                raise Exception(f'No sampled parameter available for {dict_map[i]}')
            
        #cumsum to build the paths and add the initial value
        #initialize an empty numpy array
        paths = np.zeros((self.n_simulations, self.N + 1)) #shape (M, N + 1)
        paths[:, 0] = self.X0 # Set the initial value for all paths

        
        #drift --- (mu - 0.5 * sigma^2) * dt 
        drift = (self.mu - 0.5 * self.sigma**2) * self.dt #type: ignore

        #shocks --- sigma * Z * sqrt(dt) with Z distributed as N(0,1)
        Z = self.rng.standard_normal(size=(self.n_simulations, self.N)) # shape (n_simulations, N)
        shocks = (self.sigma.reshape(-1,1) * Z) * np.sqrt(self.dt)

        #Calculate the increments for each step -> (mu - 0.5 * sigma^2) * dt  + sigma * Z * sqrt(dt) with Z distributed as N(0,1)
        increments = drift.reshape(-1,1) + shocks

        # cumulative sum of increments with start value X0
        paths[:, 1:] = self.X0.reshape(-1,1) + np.cumsum(increments, axis=1)
        self.paths = paths
        self.X_T = paths[:,-1]

        return paths
    
    def get_pdf(self, n_steps_ahead:int, n_bins:int|None = None, verbose:bool = False):
        """
        compute the analytical parameters of the normal distribution from BS paths
        args: 
            n_steps_ahead:int -> represent the lenght of the future period in terms of dt. For instance 10 times dt
            n_bins:int -> if None or 0 then just compute the analytical mean and std. If greater than 0 compute bins of the distribution
            bins: -> 1D array of custome bins. Usually used in inference time to load training bins
        """

        if self.X_T is None or self.mu is None or self.sigma is None or self.dt is None:
            raise Exception('Bad initialization of inputs of trajectories')

        
        # analytical parameters of the step ahead distribution
        delta_t = n_steps_ahead * self.dt
        mean = self.X_T - ((self.mu - 0.5 * self.sigma**2) * delta_t)
        std = self.sigma * np.sqrt(delta_t)

        if mean.shape != std.shape:
            raise Exception(f'Shapes does not match. Mean\'s array shape {mean.shape} while std\'s array shape {std.shape}')
        

        if verbose:
            print(f"Mean values - unique: {np.unique(mean).shape}, min: {mean.min()}, max: {mean.max()}")
            print(f"Std values - unique: {np.unique(std).shape}, min: {std.min()}, max: {std.max()}")
            print(f"First 5 means: {mean[:5]}")
            print(f"First 5 stds: {std[:5]}")


        # if bins already created
        if self.bins is not None:
            common_bins = self.bins
        # if not created yet 
        else:
            # use distribution parameters
            if n_bins is None or n_bins<1:
                pdf = np.column_stack((mean, std))
                self.pdf = pdf
                return self.pdf
            # compute bins
            else:
                x_min = mean - 4*std
                x_max = mean + 4*std

                # use the SAME absolute bin edges for all simulations
                global_x_min = x_min.min()
                global_x_max = x_max.max()
                common_bins = np.linspace(global_x_min, global_x_max, n_bins + 1)
                self.bins = common_bins

        # evaluate all distributions on the same bins  ?
        cdf_values = norm.cdf(common_bins, loc=mean.reshape(self.n_simulations, 1), 
                            scale=std.reshape(self.n_simulations, 1))
        probabilities = np.diff(cdf_values, axis=1)

        # zero-out small values and renormalize
        probabilities[probabilities < 1e-7] = 0.0
        row_sums = probabilities.sum(axis=1, keepdims=True)
        probabilities = probabilities / row_sums
        self.pdf = probabilities

        return self.pdf
            

    def save_configuration(self, filepath: str):
        """
        Save the bins to a JSON file for later use in inference. 
        """

        if self.bins is None:
            raise ValueError("No bins available to save. Run get_pdf() with n_bins > 0 first.")
        
        filepath = Path(filepath)
        bins_list = self.bins.tolist()
        json_dict = {'bins': bins_list, 'dt':self.dt}
        
        # save as JSON
        with open(filepath, 'w') as f:
            json.dump(json_dict, f)
        
        print(f"Bins saved to {filepath}")



    def load_configuration(self, filepath: str):
        """
        Load the bins from a saved JSON file.
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        
        # load JSON
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.bins = np.array(data['bins'])
        #self.dt = data['dt']
        print(f"Bins loaded from {filepath}")
        print(f"Loaded {len(self.bins)} bins")
        
        return self.bins
    

    def save_binary_file(self, file_name:str):
        if self.pdf is None or self.paths is None:
            raise Exception("Trajectory data or probability distribution are not available. " \
            "First generate those data")
        
        if self.pdf.shape[0] != self.paths.shape[0]:
            raise Exception("Number of generated trajectories and distributions should match." \
                            f"number of trajectories is {self.pdf.shape[0]} and number of distribution is {self.paths.shape[0]}")
        
        dtype_paths = self.paths.dtype.str.encode('utf-8')
        dtype_pdf   = self.pdf.dtype.str.encode('utf-8')

        file_path = file_name + '.bin'
        if not os.path.exists(file_path):
            mode = 'wb'
            write_header = True
        else:
            mode = 'ab'
            write_header = False

            #check datatype match
            with open(file_path, 'rb') as file:
                n1 = struct.unpack('<I', file.read(4))[0] #now the pointer is in position 4
                stored_dtype1 = file.read(n1) 

                n2 = struct.unpack('<I', file.read(4))[0]
                stored_dtype2 = file.read(n2)

                if stored_dtype1 != dtype_paths:
                        raise ValueError(
                        f"Incompatible dtype for paths: file has {stored_dtype1!r}\n "
                        f"new data has {dtype_paths!r}"
                        )
                
                if stored_dtype2 != dtype_pdf:
                        raise ValueError(
                        f"Incompatible dtype for paths: file has {stored_dtype2!r}\n "
                        f"new data has {dtype_pdf!r}"
                        )
                
        with open(file_path, mode) as f:
            if write_header:
                #write datatypes header
                f.write(struct.pack('<I', len(dtype_paths))) #writes a 4-byte integer giving the length of the upcoming dtype string
                f.write(dtype_paths) #writes that dtype
                f.write(struct.pack('<I', len(dtype_pdf)))
                f.write(dtype_pdf)

            #write rows 
            for i, path in enumerate(self.paths):
                f.write(struct.pack("<I", path.size)) # H used for max length of 65535
                f.write(struct.pack("<I", self.pdf[i, :].size))
                path.tofile(f)
                self.pdf[i, :].tofile(f)
            
        print(f'file stored in {file_path}')


    def load_binary_file(self, file_name:str):

        paths = []
        pdfs = []
        with open(file_name, 'rb') as f:
            
            # read dtype headers
            n = struct.unpack('<I', f.read(4))[0] # how many bytes to read next
            dtype_paths = f.read(n).decode('utf-8') # read exactly that many bytes

            n = struct.unpack('<I', f.read(4))[0]
            dtype_pdf = f.read(n).decode('utf-8')

            dt_path = np.dtype(dtype_paths)
            dt_pdf  = np.dtype(dtype_pdf)

            itemsize_path = dt_path.itemsize #Length of one array element in bytes.
            itemsize_pdf  = dt_pdf.itemsize

            while True:
                header = f.read(8)  # two 4-byte unsigned ints
                if not header:
                    break
                if len(header) < 8:
                    raise EOFError("corrupt or truncated file")

                len_path, len_pdf = struct.unpack('<II', header) #read integers containg path and pdf lengths

                nbytes_path = len_path * itemsize_path
                nbytes_pdf  = len_pdf  * itemsize_pdf

                path = f.read(nbytes_path)
                if len(path) != nbytes_path:
                    raise EOFError("truncated path data")

                pdf = f.read(nbytes_pdf)
                if len(pdf) != nbytes_pdf:
                    raise EOFError("truncated pdf data")

                arr_path = np.frombuffer(path, dtype=dt_path).copy()
                arr_pdf  = np.frombuffer(pdf, dtype=dt_pdf).copy()

                paths.append(arr_path)
                pdfs.append(arr_pdf)

        self.paths = np.vstack(paths)
        self.pdf = np.vstack(pdfs)
        print('processed binary file')

        return self.paths, self.pdf
            


            
    
    def plot(self):
        if self.paths is None:
            raise Exception("No paths has been generated yet")

        # Convert log-price paths back to price paths for a more intuitive plot
        price_paths = np.exp(self.paths)

        # --- Plotting the Results ---
        # Create a time array for the x-axis
        time_grid = np.linspace(0, self.T, self.N + 1)

        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot the log-price paths
        # We transpose the paths matrix so that each column is a path
        ax.plot(time_grid, self.paths.T, lw=0.8, alpha=0.7)

        # --- Formatting the Plot ---
        ax.set_title(f'{self.n_simulations} Simulated Log-Price Trajectories', fontsize=16)
        ax.set_xlabel('Time (Years)', fontsize=12)
        ax.set_ylabel('Log Stock Price $X_t = \ln(S_t)$', fontsize=12)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        # Add a horizontal line for the initial log price
        ax.legend()
        plt.tight_layout()
        plt.show()

        #Plot the actual stock price paths
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.plot(time_grid, price_paths.T, lw=0.8, alpha=0.7)
        ax2.set_title(f'{self.n_simulations} Simulated Stock Price Trajectories', fontsize=16)
        ax2.set_xlabel('Time (Years)', fontsize=12)
        ax2.set_ylabel('Stock Price $S_t$', fontsize=12)
        plt.tight_layout()
        plt.show()


        # plot histogram of final values
        plt.figure()
        plt.hist(self.paths[:, -1], bins=100)
        plt.title("Histogram of terminal prices")
        plt.xlabel("S(T)")
        plt.ylabel("frequency")
        plt.show()













if __name__ == '__main__':
    # data simulation 
    X0_range = (0.0,1.0)
    mu_range = (0.0, 0.0)
    sigma_range = (0.001, 1.0)
    T = 1.0        # Time horizon (1 year)
    N = 3        # Number of time steps
    J = 2        # Number of paths to simulate
    SEED=42

    # --- Run the Simulation ---
    sim = DataSimulator(X0_range=X0_range, mu_range=mu_range, sigma_range=sigma_range, 
                                T=T, N=N, n_simulations=J, seed=SEED)

    sim.get_paths()
    sim.get_pdf(n_steps_ahead=10, n_bins=3)
    sim.save_binary_file('data/demo')
    print(sim.paths)
    print(sim.pdf)

    file_paths, file_pdf = sim.load_binary_file('data/inputs/demo.bin')
    print(file_paths)
    print(file_pdf)