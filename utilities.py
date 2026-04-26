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
from scipy.stats import ttest_1samp, wasserstein_distance
from scipy.spatial.distance import jensenshannon



def manage_csv_results(csv:str):

    df = pd.read_csv(csv)
    df["generated"] = df["generated"].apply(ast.literal_eval)
    df["true"] = df["true"].apply(ast.literal_eval)

    return df


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
            # Row-wise standardization 
            C_mean = C_tensor.mean(dim=1, keepdim=True)
            C_std = C_tensor.std(dim=1, keepdim=True) + eps
            C_tensor = (C_tensor - C_mean) / C_std
            
            print(f"X tensor shape last axis {X_tensor.shape[1]}")
            if X_tensor.shape[1] == 1:
                # Use C_tensor statistics for standardization
                X_mean = C_mean 
                X_std = C_std    
                X_tensor = (X_tensor - X_mean) / X_std
            else:
                X_mean = X_tensor.mean(dim=1, keepdim=True)
                X_std = X_tensor.std(dim=1, keepdim=True) + eps
                X_tensor = (X_tensor - X_mean) / X_std
        
        elif preprocess == 'log':
            X_tensor = torch.log(X_tensor + eps) # only target if probabilities
        else:
            raise Exception('Select an available preprocess step. Available are \"standardization\", \"log\"')
    
    dataset = TensorDataset(X_tensor, C_tensor)
    return dataset, X_mean, X_std

def freedman_diaconis_bins(data: np.ndarray) -> tuple:
    """
    Compute optimal histogram bins using the Freedman-Diaconis Rule.
    
    Bin width: h = 2 * IQR * n^(-1/3)
    Number of bins: k = ceil((max - min) / h)
    
    Parameters:
        data: array-like of numeric values
    
    Returns:
        bin_width (float), num_bins (int), bin_edges (np.ndarray)
    """
    data = np.asarray(data, dtype=float)
    data = data[~np.isnan(data)]  # remove NaNs
    n = len(data)

    if n < 2:
        raise ValueError("Need at least 2 data points.")

    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25

    if iqr == 0:
        raise ValueError("IQR is zero — data may be too concentrated. Consider a different binning rule.")

    # Freedman-Diaconis bin width
    bin_width = 2 * iqr * n ** (-1/3)

    data_range = data.max() - data.min()
    num_bins = int(np.ceil(data_range / bin_width))

    # Build explicit bin edges for full control
    bin_edges = np.linspace(data.min(), data.max(), num_bins + 1)

    return bin_width, num_bins, bin_edges

def compare_simulated_pdfs(true_pdfs: np.ndarray,
                            generated_pdfs: np.ndarray, 
                            use_opt_binning:bool = False) -> tuple:
    """
    Function used to compare the result from two simulated pdfs
    params: true_pdfs: a 2D numpy array containing for each row the simulations which approximate the true pdf 
    params: generated_pdfs: a 2D numpy array containing for each row the simulations which approximate the generated pdf
    params: use_opt_binning: if true the freedman_diaconis_bins algorithm is used to provide number of bins and bin edges
    """

    if true_pdfs.shape != generated_pdfs.shape:
        raise ValueError(f"the shape of true_pdf and generated_pdf must match."
                          f"current true_pdf shape is{true_pdfs.shape}" 
                          f"while current generated_pdf shape is {generated_pdfs.shape}")
    
    true_discretized_pdfs = []
    generated_discretized_pdfs = []
    bin_edges_list = []
    for i, simulated_pdf in enumerate(true_pdfs):
        if use_opt_binning:
            _, _, bin_edges = freedman_diaconis_bins(simulated_pdf)
            hist_true, _ = np.histogram(simulated_pdf, bins=bin_edges)
            hist_generated, _ = np.histogram(generated_pdfs[i], bins=bin_edges)
        else:
            n_bins = int(np.ceil(np.sqrt(simulated_pdf.shape[0])))
            hist_true, bin_edges = np.histogram(simulated_pdf, bins=n_bins)
            hist_generated, _ = np.histogram(generated_pdfs[i], bins=bin_edges)    
            
        if hist_true.sum() == 0 or hist_generated.sum() == 0:
            continue

        bin_edges_list.append(bin_edges)
        true_discretized_pdfs.append(hist_true / hist_true.sum())
        generated_discretized_pdfs.append(hist_generated/hist_generated.sum())
    
    return true_discretized_pdfs, generated_discretized_pdfs, bin_edges_list

def compute_js(generated_arr:np.ndarray|list, true_arr:np.ndarray|list, is_log:bool = True):
    """
    Compute the Jenson-Shannon measure
    generated_arr: np.array -> an array of generated probability distributions 
    """
    js_distances = []

    for p, t in zip(generated_arr, true_arr): 
        if is_log:
            p = np.exp(p)
            t = np.exp(t)

        #normalization
        p = p / np.sum(p)
        t = t / np.sum(t)
        
        # compute JSD
        js_distances.append(jensenshannon(p, t, base=2.0))

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

def get_error_metrics(true: np.ndarray|list, generated: np.ndarray|list) -> dict:
        """
        Compute errors metrics for probability distributions
        
        Args:
            true: array containing true samples
            generated: array containing generated samples
        
        Returns:
            Dictionary containing errors and statistics
             total_variance_distance -> Maximum probability difference over all events. bounded [0,1]
             hellinger_distance -> bounded [0,1]
             jensen_shannon_distance -> sqrt(jensen-shannon divergence). bounded [0,1]
             emd_distance -> Earth mover distance. [0, infty]

        """
        tv_distance = []
        hellinger_distance = []
        js_distance = []
        emd_distance = []

        if isinstance(true, np.ndarray) and isinstance(generated, np.ndarray):
            tv_distance = 0.5*np.sum(np.abs(true-generated), axis=1)
            hellinger_distance = np.sqrt(0.5*np.sum((np.sqrt(true) - np.sqrt(generated))**2, axis=1))
            js_distance = compute_js(generated, true, is_log=False)
            emd_distance = np.mean([wasserstein_distance(t, g) for t, g in zip(true, generated)])
        else:
            for i, true_pdf in enumerate(true):
                ith_tv_distance = 0.5*np.sum(np.abs(true_pdf-generated[i]))
                ith_hellinger_distance = np.sqrt(0.5*np.sum((np.sqrt(true_pdf) - np.sqrt(generated[i]))**2))
                ith_js_distance = compute_js(generated, true, is_log=False)
                ith_emd_distance = wasserstein_distance(true_pdf, generated[i])
                
                tv_distance.append(ith_tv_distance)
                hellinger_distance.append(ith_hellinger_distance)
                js_distance.append(ith_js_distance)
                emd_distance.append(ith_emd_distance)
            
            tv_distance = np.mean(tv_distance)
            hellinger_distance = np.mean(hellinger_distance)
            js_distance = np.mean(js_distance)
            emd_distance = np.mean(emd_distance)



        stats = {
            "js_distance": js_distance,
            "hellinger_distance": hellinger_distance,
            "tv_distance": tv_distance,
            "emd_distance": emd_distance
            }
        
        return stats

def analyze_error_distribution(csv:str):

    """
    Produce a distribution of errors from a csv obtained using evaluate_error_distribution method of myCGAN
    
    Args:
        csv: path to the csv file
    """

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
    sns.violinplot(data=df[errors], inner=None, cut=0)
    
    #mean markers
    plt.scatter(range(len(errors)), means.values, color="black", zorder=3, label="Mean") #type:ignore

    #std bars
    plt.errorbar(range(len(errors)),means.values,yerr=stds.values,fmt="none",ecolor="black",capsize=6,zorder=2) #type:ignore

    plt.title("Error Distributions with Mean ± Std")
    plt.xticks(range(len(errors)), errors, rotation=45, ha="right")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return means, stds, summary
    
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



def plot_bin_dist(trues:np.ndarray|list,
                   preds:np.ndarray|list,
                   bins_values:np.ndarray|list,
                   X_T: List[float]|None = None,
                   means: np.ndarray | list | None = None,
                   stds: np.ndarray | list | None = None,
                   ncols=3,
                   zoom:bool = False):

    n = len(trues)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5 * ncols, 4 * nrows),
        sharey=True
    )
    axes = axes.flatten()

    for i, (true, pred) in enumerate(zip(trues, preds)):
        if len(true) != len(pred):
            raise Exception(f'true and pred have different shapes. true ={np.shape(true)}, pred = {np.shape(pred)}')

        # Handle bins_values: could be a single array, 2D array, or list of arrays
        if isinstance(bins_values, (list, tuple)) and len(bins_values) == n:
            bins = bins_values[i]
        else:
            bins = bins_values

        # Compute bin centers for this row
        bin_centers_row = 0.5 * (np.array(bins[:-1]) + np.array(bins[1:]))

        # find indices to zoom the distribution
        if zoom:
            indices = np.where(np.array(true) > 1e-7)[0]
            if len(indices) > 0:
                first_idx = indices[0]
                last_idx = indices[-1]
                start = int(first_idx - np.ceil(0.1 * first_idx))
                end = int(last_idx + np.ceil(0.1 * (len(true) - last_idx)))
                true = np.array(true)[start:end]
                pred = np.array(pred)[start:end]
                bin_centers_row = bin_centers_row[start:end]
            else:
                raise ValueError('the true distribution does not have any positive probability')

        ax = axes[i]
        width = bin_centers_row[1] - bin_centers_row[0]  # bin width
        ax.bar(bin_centers_row, true, width=width, alpha=0.5, label="True histogram")
        ax.bar(bin_centers_row, pred, width=width, alpha=0.5, label="Generated histogram")

        # Overlay theoretical normal distribution if mean and std are provided
        if means is not None and stds is not None:
            mu = means[i]
            sigma = stds[i]
            # Scale the PDF to match the probability mass (PDF * bin_width = probability per bin)
            normal_pdf = norm.pdf(bin_centers_row, loc=mu, scale=sigma) * width
            ax.plot(bin_centers_row, normal_pdf, color='black', linewidth=2,
                    linestyle='--', label=f"N({round(mu, 3)}, {round(sigma, 3)})")

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
        self.Z = None
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
    
    def _montecarlo_steps(self, n_mc_simulations:int, n_steps:int, start_values:np.ndarray ):

        if self.sigma is None or self.mu is None:
            raise ValueError("sigma and mu arrays are not initialized")

        #initialize an empty numpy array
        mc_paths = np.zeros((start_values.shape[0], n_steps + 1, n_mc_simulations)) #shape (M, N + 1, S)
        mc_paths[:, 0, :] = start_values.reshape(-1,1) # Set the initial value for all paths

        
        #drift --- (mu - 0.5 * sigma^2) * dt 
        drift = (self.mu - 0.5 * self.sigma**2) * self.dt #type: ignore
        mc_drift = np.broadcast_to(drift.reshape(drift.shape[0], 1, 1),
                                    (drift.shape[0], n_steps, n_mc_simulations))

        #shocks --- sigma * Z * sqrt(dt) with Z distributed as N(0,1)
        Z = self.rng.standard_normal(size=(start_values.shape[0], n_steps, n_mc_simulations)) # shape (M, N, S)
        mc_sigma = np.broadcast_to(self.sigma.reshape(self.sigma.shape[0], 1, 1),
                                    (self.sigma.shape[0], n_steps, n_mc_simulations))
        mc_shocks = (mc_sigma * Z) * np.sqrt(self.dt)

        # cumulative sum of increments with start value X0
        increments = mc_drift + mc_shocks
        start_values_mc = np.broadcast_to(start_values.reshape(start_values.shape[0], 1, 1),
                                    (start_values.shape[0], n_steps, n_mc_simulations))
       
        mc_paths[:, 1:, :] = start_values_mc + np.cumsum(increments, axis=1)
        mc_distributions = mc_paths[:,-1, :]
        return mc_distributions

    
    def get_paths(self, get_proxy_n:int = 0):
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
            
        
        #initialize an empty numpy array
        paths = np.zeros((self.n_simulations, self.N + 1)) #shape (M, N + 1)
        paths[:, 0] = self.X0 # Set the initial value for all paths

        
        #drift --- (mu - 0.5 * sigma^2) * dt 
        drift = (self.mu - 0.5 * self.sigma**2) * self.dt #type: ignore
        self.drift = drift

        #shocks --- sigma * Z * sqrt(dt) with Z distributed as N(0,1)
        Z = self.rng.standard_normal(size=(self.n_simulations, self.N)) # shape (n_simulations, N)
        self.Z = Z
        shocks = (self.sigma.reshape(-1,1) * Z) * np.sqrt(self.dt)
        self.diffusion = shocks

        #Calculate the increments for each step -> (mu - 0.5 * sigma^2) * dt  + sigma * Z * sqrt(dt) with Z distributed as N(0,1)
        increments = drift.reshape(-1,1) + shocks

        # cumulative sum of increments with start value X0
        paths[:, 1:] = self.X0.reshape(-1,1) + np.cumsum(increments, axis=1)
        self.paths = paths
        self.X_T = paths[:,-1]
        result = paths

        if get_proxy_n>0:
            result = np.column_stack([self.sigma, self.paths[:, -get_proxy_n:]])

        return result
    
    def get_pdf(self, n_steps_ahead:int, n_bins:int|None = None,
                 P:np.ndarray|None=None, mc_sims:int = 0, verbose:bool = False):
        """
        compute the analytical parameters of the normal distribution from BS paths
        args: 
            n_steps_ahead:int -> represent the lenght of the future period in terms of dt. For instance 10 times dt
            n_bins:int -> if None or 0 then just compute the analytical mean and std. If greater than 0 compute bins of the distribution
            P: np.ndarray -> array of latest price information
            mc_sims: int -> number of montecarlo simulations. If less than 2 the theoretical distribution is used 
            bins: -> 1D array of custome bins. Usually used in inference time to load training bins
        """

        if self.X_T is None or self.mu is None or self.sigma is None or self.dt is None:
            raise Exception('Bad initialization of inputs of trajectories')

        
        # analytical parameters of the step ahead distribution
        XT = P if P is not None else self.X_T
        delta_t = n_steps_ahead * self.dt
        mean = XT + ((self.mu - 0.5 * self.sigma**2) * delta_t)
        std = self.sigma * np.sqrt(delta_t)
        self.means = mean
        self.stds = std

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
        else:
            # use distribution parameters
            if n_bins is None:
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

        # evaluate all distributions on the same bins
        #cdf_values = norm.cdf(common_bins, loc=mean.reshape(self.n_simulations, 1), scale=std.reshape(self.n_simulations, 1))
        if mc_sims<2:
            cdf_values = norm.cdf(common_bins, loc=mean[:, None], scale=std[:, None])
            probabilities = np.diff(cdf_values, axis=1)
        else:
            montecarlo_simulations = self._montecarlo_steps(n_mc_simulations=1000, n_steps=n_steps_ahead, start_values=XT).squeeze()
            if n_bins is not None and n_bins>10:
                probabilities = np.zeros((mean.shape[0], n_bins))
                for i in range(montecarlo_simulations.shape[0]):
                    hist, _ = np.histogram(montecarlo_simulations[i], bins=common_bins)
                    probabilities[i] = hist / hist.sum()
            else:
                return montecarlo_simulations
                

        # zero-out small values and renormalize
        probabilities[probabilities < 1e-7] = 0.0
        row_sums = probabilities.sum(axis=1, keepdims=True)
        probabilities = probabilities / row_sums
        probabilities = probabilities.astype(np.float32)  
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

        if file_name.endswith('.bin'):
            file_path = file_name
        else:
            file_path = file_name + '.bin'

        paths = []
        pdfs = []
        with open(file_path, 'rb') as f:
            
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

class BinaryDataset(torch.utils.data.Dataset):
    """
    A memory-efficient, random-access PyTorch Dataset backed by a DataSimulator
    binary file.  Designed for training ForGAN on datasets that exceed available RAM.
 
    How random access works
    -----------------------
    Every record in the binary file has an identical byte size (fixed N and n_bins),
    so the byte offset of record i is:
 
        offset(i) = header_end + i * record_size
 
    This gives O(1) seek-and-read per sample, exactly like a numpy memmap.
 
    DataLoader worker safety
    ------------------------
    File handles are opened lazily inside each worker process via ``_get_fh()``.
    The handle is deliberately excluded from pickling (``__getstate__``) so that
    each spawned/forked worker starts with ``_fh = None`` and opens its own handle
    on first access — no shared-descriptor races.
 
    Item format
    -----------
    ``__getitem__`` returns ``(target, condition)`` to match the TensorDataset
    convention produced by ``prepare_data(targets, paths)``:
 
        target    – path[-1:]  shape (1,)   the next-step log-price X_T
        condition – path[:-1]  shape (N,)   the observed path history
 
    Batch helpers
    -------------
    ``load_subset(indices)``      → TensorDataset  (for fold scoring in tuner)
    ``load_subset_pdfs(indices)`` → np.ndarray      (analytical PDFs from file,
                                                      for fold JS comparison)
    """
 
    def __init__(self, file_path: str):
        if not file_path.endswith('.bin'):
            file_path = file_path + '.bin'
        self.file_path = file_path
 
        # ── parse binary header ───────────────────────────────────────────
        with open(self.file_path, 'rb') as f:
            n = struct.unpack('<I', f.read(4))[0]
            self.dtype_paths = np.dtype(f.read(n).decode('utf-8'))
            n = struct.unpack('<I', f.read(4))[0]
            self.dtype_pdf   = np.dtype(f.read(n).decode('utf-8'))
            self._header_end = f.tell()
 
            # Read the first record header to discover fixed path / pdf lengths
            rec_hdr = f.read(8)
            if len(rec_hdr) < 8:
                raise ValueError(f"Binary file '{file_path}' contains no data records.")
            self._len_path, self._len_pdf = struct.unpack('<II', rec_hdr)
 
        # ── fixed record layout ───────────────────────────────────────────
        # [4 B len_path][4 B len_pdf][path bytes][pdf bytes]
        self._path_nbytes   = self._len_path * self.dtype_paths.itemsize
        self._pdf_nbytes    = self._len_pdf  * self.dtype_pdf.itemsize
        self._record_size   = 8 + self._path_nbytes + self._pdf_nbytes
 
        # ── count records from file size ──────────────────────────────────
        file_size = os.path.getsize(self.file_path)
        data_size = file_size - self._header_end
        if data_size % self._record_size != 0:
            raise ValueError(
                f"File size is inconsistent with the computed record size "
                f"({self._record_size} bytes). The file may be corrupted or truncated."
            )
        self._n_records = data_size // self._record_size
 
        # ── lazy per-worker file handle (never pickled) ───────────────────
        self._fh = None
 
    # ── pickling support for DataLoader multiprocessing ──────────────────────
 
    def __getstate__(self):
        state = self.__dict__.copy()
        state['_fh'] = None      # exclude file handle — each worker opens its own
        return state
 
    def __setstate__(self, state):
        self.__dict__.update(state)
        self._fh = None
 
    def _get_fh(self):
        """Return an open file handle, creating one for this worker if needed."""
        if self._fh is None or self._fh.closed:
            self._fh = open(self.file_path, 'rb')
        return self._fh
 
    def __del__(self):
        if self._fh is not None and not self._fh.closed:
            self._fh.close()
 
    # ── Dataset protocol ─────────────────────────────────────────────────────
 
    def __len__(self) -> int:
        return self._n_records
 
    def __getitem__(self, idx: int):
        """
        Returns (target, condition) for sample ``idx``:
            target    – torch.float32 tensor shape (1,)   = path[-1]  (X_T)
            condition – torch.float32 tensor shape (N,)   = path[:-1] (history)
 
        The PDF stored in the file is NOT returned here; it is only used by
        ``load_subset_pdfs`` for evaluation inside the tuner.
        """
        if idx < 0 or idx >= self._n_records:
            raise IndexError(f"Index {idx} out of range [0, {self._n_records}).")
 
        # Byte offset: skip record-header (8 bytes) to reach path data directly
        offset = self._header_end + idx * self._record_size + 8
        fh = self._get_fh()
        fh.seek(offset)
 
        path = np.frombuffer(fh.read(self._path_nbytes), dtype=self.dtype_paths).astype(np.float32)
        # PDF bytes are skipped — not needed for per-sample training
 
        # Convention: target = last step, condition = preceding history
        return (
            torch.tensor(path[-1:], dtype=torch.float32),   # (1,)
            torch.tensor(path[:-1], dtype=torch.float32),   # (N,)
        )
 
    # ── Batch helpers ─────────────────────────────────────────────────────────
 
    def _sorted_read(self, indices: np.ndarray, load_pdf: bool = True, load_path: bool = True):
        """
        Internal helper: read path and/or pdf bytes for a set of indices.
        Reads are issued in ascending file-offset order for sequential I/O,
        then results are placed back into their original output positions.
        """
        n = len(indices)
        paths = np.empty((n, self._len_path), dtype=np.float32) if load_path else None
        pdfs  = np.empty((n, self._len_pdf),  dtype=np.float32) if load_pdf  else None
 
        sorted_pairs = sorted(enumerate(indices), key=lambda x: x[1])
        with open(self.file_path, 'rb') as f:
            for out_i, file_idx in sorted_pairs:
                # +8 to skip the per-record [len_path, len_pdf] header
                f.seek(self._header_end + file_idx * self._record_size + 8)
                if load_path:
                    paths[out_i] = np.frombuffer(f.read(self._path_nbytes), dtype=self.dtype_paths)
                else:
                    f.seek(self._path_nbytes, 1)   # skip path bytes (seek relative to current pos)
                if load_pdf:
                    pdfs[out_i] = np.frombuffer(f.read(self._pdf_nbytes), dtype=self.dtype_pdf)
        return paths, pdfs
 
    def load_subset(self, indices: np.ndarray) -> TensorDataset:
        """
        Load a subset of records as a TensorDataset in (target, condition) format.
        Used by the tuner to materialise a validation fold for ``model.generate()``.
 
        Memory cost: ``len(indices) × (1 + N) × 4`` bytes  (≈ 9 MB for 100 K rows, N=22)
        """
        paths, _ = self._sorted_read(indices, load_pdf=False, load_path=True)
        targets    = torch.tensor(paths[:, -1:], dtype=torch.float32)   # (n, 1)
        conditions = torch.tensor(paths[:, :-1], dtype=torch.float32)   # (n, N)
        return TensorDataset(targets, conditions)
 
    def load_subset_pdfs(self, indices: np.ndarray) -> np.ndarray:
        """
        Load only the analytical PDF vectors for a subset of indices.
        Used by ``BinaryGANHyperparameterTuner`` to retrieve per-fold ground-truth
        distributions without recomputing them from ``sim.mu`` / ``sim.sigma``.
 
        Returns
        -------
        np.ndarray of shape (len(indices), n_bins), dtype float32
        """
        _, pdfs = self._sorted_read(indices, load_pdf=True, load_path=False)
        return pdfs






if __name__ == '__main__':
    # data simulation 
    X0_range = (0.0,1.0)
    mu_range = (0.0, 0.0)
    sigma_range = (0.1, 1.0)
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

    file_paths, file_pdf = sim.load_binary_file('data/inputs/demo')
    print(file_paths)
    print(file_pdf)