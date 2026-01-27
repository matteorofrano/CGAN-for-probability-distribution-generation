import os
import json
import time
import numpy as np
from typing import Callable, Optional, Dict
import torch
from torch.utils.data import DataLoader
from utilities import TensorDataset, DataSimulator, prepare_data, compute_js, pd
from GANComponents import MyDiscriminator, MyGenerator


class MyCGAN():

    

    """
    Cgan architecture 
    """
    def __init__(self, max_epoch:int = 100, batch_size = 32,
                  n_critic:int = 1, z_noise_dim:int = 252,
                  loss_fn:Callable|None =  torch.nn.BCEWithLogitsLoss(),
                  name:str = 'ConditionalGAN'):
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
        
        if self.loss_fn is None:
            raise ValueError('loss_fn parameter has not been initialized')
        
        # Move models to device
        self.G.to(self.DEVICE)
        self.D.to(self.DEVICE)
        
        data_loader = DataLoader(dataset=data, batch_size=self.batch_size, shuffle=True, drop_last=True)
        D_opt = torch.optim.Adam(self.D.parameters(), lr=0.0005, betas=(0.5, 0.999))
        G_opt = torch.optim.Adam(self.G.parameters(), lr=0.0005, betas=(0.5, 0.999))

        df = None
        step=0
        self.D.train()
        start_time = time.time()
        for epoch in range(self.max_epoch):
            predictions_list = []
            targets_list = []
            condition_list = []
            D_loss_list = []
            G_loss_list = []
            
            for idx, (prob_dist, trajectory) in enumerate(data_loader):   
                x=prob_dist.to(self.DEVICE)
                c=trajectory.to(self.DEVICE)
                current_batch_size = x.size(0)

                # Create labels with correct batch size
                D_labels = torch.ones([current_batch_size, 1]).to(self.DEVICE) # Discriminator Label to real
                D_fakes = torch.zeros([current_batch_size, 1]).to(self.DEVICE) # Discriminator Label to fake

                #TRAIN DISCRIMINATOR
                D_opt.zero_grad()
                
                #real samples
                x_outputs = self.D(x, c) 
                D_x_loss = self.loss_fn(x_outputs, D_labels) 

                #fake samples
                z = torch.randn((current_batch_size, self.z_dim)).to(self.DEVICE)
                z_outputs = self.D(self.G(z, c).detach(), c)
                D_z_loss = self.loss_fn(z_outputs, D_fakes)

                #backpropagation
                D_loss = D_x_loss + D_z_loss 
                D_loss.backward()
                D_opt.step()
                
                if step % self.n_critic == 0:
                    # TRAIN GENERATOR
                    G_opt.zero_grad()
                    z = torch.randn((current_batch_size, self.z_dim)).to(self.DEVICE)
                    z_outputs = self.D(self.G(z, c), c)
                    G_loss = self.loss_fn(z_outputs, D_labels)
                    
                    #backpropagation
                    G_loss.backward()
                    G_opt.step()
                
                if step % 500 == 0:
                    print('Epoch: {}/{}, Step: {}, D Loss: {}, G Loss: {}'.format(epoch, self.max_epoch, step, D_loss.item(), G_loss.item())) #type:ignore
                

                #store history for each middle batch of the epoch
                if save_history and idx == int(len(data_loader)/2): 
                    self.G.eval()
                    with torch.no_grad():
                        z = torch.randn((current_batch_size, self.z_dim)).to(self.DEVICE)
                        generated = self.G(z, c).cpu().numpy()
                    for i, row in enumerate(generated):
                        pred = row.tolist()
                        true = x[i, :].cpu().tolist()
                        predictions_list.append(pred)
                        targets_list.append(true)
                        condition_list.append(c[i, :].cpu().tolist())

                        D_loss_list.append(round(float(D_loss), 4))
                        G_loss_list.append(round(float(G_loss), 4)) #type:ignore
                        
                    self.G.train()

                step += 1

            # Build a DataFrame where each list becomes a column
            if len(predictions_list)>1:
                js_divergence = compute_js(np.array(predictions_list), np.array(targets_list))
                distance = np.mean(js_divergence)  #np.linalg.norm(np.array(predictions_list) - np.array(targets_list))
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

        end_time = time.time()
        if df is not None:
            df.to_csv("generated_vs_true.csv", index=False)
        


    def generate(self, data: TensorDataset, get_pdf:bool = False, bins: np.ndarray|None = None):
        """
        Generate predictions using the trained Generator
        
        Args:
            data: TensorDataset containing true values and conditions (pdf, trajectory)
            get_pdf: Boolean -> define if multiple generation using different noise vector is needed to compute the probability distribution of the outcome

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
        simulation_list = []
        
        with torch.no_grad():
            for idx, batch in enumerate(data_loader):
                # Handle both (prob_dist, trajectory) and (trajectory,) formats
                if len(batch) == 2:
                    _, trajectory = batch
                else:
                    trajectory = batch[0]
                
                c = trajectory.to(self.DEVICE)
                current_batch_size = c.size(0)
                
                # Generate samples
                if get_pdf:
                    sample_values = []
                    for _ in range(1000):
                        z = torch.randn((current_batch_size, self.z_dim)).to(self.DEVICE)
                        generated = self.G(z, c)
                        sample_values.append(generated.cpu())
                   
                    simulation_list.append(sample_values)
                    conditions_list.append(c.cpu())
                else:
                    z = torch.randn((current_batch_size, self.z_dim)).to(self.DEVICE)
                    generated = self.G(z, c)
                    
                    predictions_list.append(generated.cpu())
                    conditions_list.append(c.cpu())

        
        conditions = torch.cat(conditions_list, dim=0).numpy()
        # IF GENERATED SAMPLES ARE AVAILABLE THEN BUILD THE PDF 
        if simulation_list:
            sim_array = np.array(simulation_list).squeeze(-1)
            sim_array = sim_array.transpose(0, 2, 1) # shape (n_batch, batch_samples, 1000)
            print(f'shape array {sim_array.shape}')
            means =sim_array.mean(axis=2) # mean over the 1000 samples
            simulations= np.concatenate(sim_array, axis=0)

            # compute the pdf using available bins 
            if bins is not None:
                predictions = np.zeros((simulations.shape[0], bins.shape[0]-1))
                for i in range(simulations.shape[0]):
                    hist, _ = np.histogram(simulations[i], bins = bins)
                    predictions[i] = hist/hist.sum()
            else:
                predictions = simulations

        # IF PDF IS AVAILABLE THEN JUST RETURN IT
        else:
            predictions = torch.cat(predictions_list, dim=0).numpy()
        
        return conditions, predictions
    

    def evaluate_error_distribution(self, data: TensorDataset, save_to:str|None = None, eps:float = 1e-6) -> dict:
        """
        Compute element-wise error distribution between generated and true samples.
        
        Args:
            data: TensorDataset containing (true_samples, conditions)
            save_path: Optional path to save results as CSV
        
        Returns:
            Dictionary containing errors and statistics
        """

        true = data.tensors[0].numpy()
        conditions, generated = self.generate(data)
        true = np.clip(true, eps, 1.0)
        generated = np.clip(generated, eps, 1.0)

        errors = 2 * np.abs(true - generated) / (true + generated + eps)
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
