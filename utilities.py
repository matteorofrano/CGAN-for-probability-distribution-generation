from typing import Union, Tuple, Text
import pandas as pd 
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt



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




class TrajectorySimulator():
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
                  X0_range: Tuple[float, float], mu_range: Tuple[float, float], sigma_range: Tuple[float, float],
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
        self.strategies = ['uniform', 'log_uniform']

        #to be sampled
        self.X0 = None
        self.mu = None
        self.sigma = None

        #trajectories
        self.paths=None

    def sample_parameters(self, strategy:Text, range: tuple):
        """
        Function used to sample J initial values, mu and sigma for each trajectory j
        """

        if strategy not in self.strategies:
            raise Exception(f'The provided strategy: {strategy} is not available. The available strategies are {self.strategies}')
        
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
    
    def simulate_BS_paths(self):
        """
        Simulates log prices with Brownian Motion.
        This function simulates M trajectories of the log stock price process over the time horizon [0, T] using N time steps.
        dX_t = (mu - 0.5 * sigma^2) dt + sigma dW_t 

        """

        self.X0 = self.sample_parameters('uniform', self.X0_range)
        self.mu = self.sample_parameters('uniform', self.mu_range)
        self.sigma = self.sample_parameters('uniform', self.sigma_range)

        for i, sampling_parameter in enumerate([self.X0, self.mu, self.sigma]):
            if len(sampling_parameter)<1:
                dict_map = {0:"X0", 1:"mu", 2:"sigma"}
                raise Exception(f'No sampled parameter available for {dict_map[i]}')
            
        #cumsum to build the paths and add the initial value
        #initialize an empty numpy array
        paths = np.zeros((self.n_simulations, N + 1)) #shape (M, N + 1)
        paths[:, 0] = self.X0 # Set the initial value for all paths

        dt = self.T / self.N
        #drift --- (mu - 0.5 * sigma^2) * dt 
        drift = (self.mu - 0.5 * self.sigma**2) * dt #type: ignore

        #shocks --- sigma * Z * sqrt(dt) with Z distributed as N(0,1)
        Z = self.rng.standard_normal(size=(self.n_simulations, self.N)) # shape (n_simulations, N)
        shocks = (self.sigma.reshape(-1,1) * Z) * np.sqrt(dt)

        #Calculate the increments for each step -> (mu - 0.5 * sigma^2) * dt  + sigma * Z * sqrt(dt) with Z distributed as N(0,1)
        increments = drift.reshape(-1,1) + shocks

        # cumulative sum of increments with start value X0
        paths[:, 1:] = self.X0.reshape(-1,1) + np.cumsum(increments, axis=1)
        self.paths = paths
        return paths
    
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
        ax.set_title(f'{J} Simulated Log-Price Trajectories', fontsize=16)
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
        ax2.set_title(f'{J} Simulated Stock Price Trajectories', fontsize=16)
        ax2.set_xlabel('Time (Years)', fontsize=12)
        ax2.set_ylabel('Stock Price $S_t$', fontsize=12)
        ax2.legend()
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
    # example
    X0_range = (0.0,1.0)
    mu_range = (0.0, 0.0)
    sigma_range = (0.001, 1.0)
    T = 1.0        # Time horizon (1 year)
    N = 252        # Number of time steps (trading days in a year)
    J = 50         # Number of paths to simulate
    SEED=42

    # --- Run the Simulation ---
    sim = TrajectorySimulator(X0_range=X0_range, mu_range=mu_range, sigma_range=sigma_range, 
                              T=T, N=N, n_simulations=J, seed=SEED)
    
    paths = sim.simulate_BS_paths()
    sim.plot()
    
    
    print(paths)