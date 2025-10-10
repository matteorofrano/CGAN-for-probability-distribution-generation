import numpy as np
import matplotlib.pyplot as plt


def simulate_log_price_trajectories(
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    N: int,
    n_simulations: int
) -> np.ndarray:
    """
    Simulates Black-Scholes log stock price trajectories based on Geometric Brownian Motion.

    This function simulates M trajectories of the log stock price process
    dX_t = (mu - 0.5 * sigma^2) dt + sigma dW_t
    over the time horizon [0, T] using N time steps.

    Args:
        S0 (float): The initial stock price at time t=0.
        mu (float): The expected return of the stock (drift).
        sigma (float): The volatility of the stock (diffusion).
        T (float): The total time horizon for the simulation (in years).
        N (int): The number of time steps in the simulation.
        n_simulations (int): The number of independent trajectories (paths) to simulate.

    Returns:
        np.ndarray: A 2D NumPy array of shape (n_simulations, N + 1) where each row
                    represents a simulated trajectory of the log stock price.
    """
    # Calculate time step and initial log price
    dt = T / N
    X0 = np.log(S0)

    # Define the constant drift and diffusion terms for the discrete process
    # The increment is: (mu - 0.5 * sigma^2) * dt + sigma * Z * sqrt(dt)
    # where Z is a standard normal random variable.
    drift = (mu - 0.5 * sigma**2) * dt
    
    #Generate all random shocks for all paths and all steps at once
    # Z has shape (n_simulations, N)
    Z = np.random.standard_normal(size=(n_simulations, N))
    shocks = sigma * Z * np.sqrt(dt)

    # 4. Calculate the increments for each step
    increments = drift + shocks

    # 5. Use cumsum to build the paths and add the initial value
    # Create an array to hold the paths, with shape (M, N + 1)
    paths = np.zeros((n_simulations, N + 1))
    paths[:, 0] = X0 # Set the initial value for all paths
    
    # Calculate the cumulative sum of increments and add the initial value
    paths[:, 1:] = X0 + np.cumsum(increments, axis=1)

    return paths



if __name__ == '__main__':
    # --- Simulation Parameters ---
    S0 = 100.0     # Initial stock price
    mu = 0.05      # Expected return (5% per year)
    sigma = 0.20   # Volatility (20% per year)
    T = 1.0        # Time horizon (1 year)
    N = 252        # Number of time steps (e.g., trading days in a year)
    M = 50         # Number of paths to simulate

    # --- Run the Simulation ---
    log_paths = simulate_log_price_trajectories(S0, mu, sigma, T, N, M)

    # Convert log-price paths back to price paths for a more intuitive plot
    price_paths = np.exp(log_paths)

    # --- Plotting the Results ---
    # Create a time array for the x-axis
    time_grid = np.linspace(0, T, N + 1)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the log-price paths
    # We transpose the paths matrix so that each column is a path
    ax.plot(time_grid, log_paths.T, lw=0.8, alpha=0.7)

    # --- Formatting the Plot ---
    ax.set_title(f'{M} Simulated Black-Scholes Log-Price Trajectories', fontsize=16)
    ax.set_xlabel('Time (Years)', fontsize=12)
    ax.set_ylabel('Log Stock Price $X_t = \ln(S_t)$', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Add a horizontal line for the initial log price
    ax.axhline(y=np.log(S0), color='black', linestyle='-.', lw=1.5, label=f'Initial Log Price: $\ln({S0})$')
    
    ax.legend()
    plt.tight_layout()
    plt.show()

    # Optional: Plot the actual stock price paths
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(time_grid, price_paths.T, lw=0.8, alpha=0.7)
    ax2.set_title(f'{M} Simulated Black-Scholes Stock Price Trajectories', fontsize=16)
    ax2.set_xlabel('Time (Years)', fontsize=12)
    ax2.set_ylabel('Stock Price $S_t$', fontsize=12)
    ax2.axhline(y=S0, color='black', linestyle='-.', lw=1.5, label=f'Initial Price: ${S0}')
    ax2.legend()
    plt.tight_layout()
    plt.show()