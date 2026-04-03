import torch
import time
import json
import numpy as np
from myCGAN import MyCGAN
from torch.utils.data import DataLoader
from utilities import TensorDataset, DataSimulator, prepare_data, compute_js, pd


class MyCWGAN(MyCGAN):
    """
    Wasserstein Conditional GAN - inherits from MyCGAN
    
    Key theoretical differences from parent CGAN:
    - Uses Wasserstein distance instead of JS divergence
    - Critic outputs unbounded scores, not probabilities
    - Gradient penalty enforces Lipschitz constraint
    - Different loss formulation
    """

    def __init__(self, max_epoch: int = 100, batch_size: int = 32, n_critic: int = 5,
                 early_stopping_patience: int = 10, early_stopping_min_delta: float = 0.001,
                 z_noise_dim: int = 252, lambda_gp: float = 10.0, name: str = 'WassersteinConditionalGAN'):
        """
        Initialize WCGAN by calling parent constructor but with modified defaults
        
        Args:
            max_epoch: Number of training epochs
            batch_size: Batch size for training
            n_critic: Number of critic updates per generator update (5 for WGAN vs 1 for GAN)
            z_noise_dim: Dimension of noise vector
            lambda_gp: Weight for gradient penalty term (typically 10)
            early_stopping_patience: Number of epochs to wait for improvement before stopping
            early_stopping_min_delta: Minimum change in W-distance to qualify as improvement
            name: Model name
        """

        super().__init__(
            max_epoch=max_epoch,
            batch_size=batch_size,
            n_critic=n_critic,
            z_noise_dim=z_noise_dim,
            loss_fn=None,
            name=name 
        )

        self.lambda_gp = lambda_gp

        # Early stopping parameters
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.best_w_distance = float('inf') 
        self.patience_counter = 0
        self.best_generator_state = None
        self.best_critic_state = None

    
    def set_critic(self, input_size:int = 784, condition_size:int = 10,
                    hidden_dim_rnn:None|int = None, **critic_params):
        '''
        Alias for set_discriminator with more appropriate naming for WGAN
        '''
        
        self.set_discriminator(input_size=input_size,
                               condition_size=condition_size,
                               output_dim=1,
                               hidden_dim_rnn=hidden_dim_rnn,
                               **critic_params)
        

    def compute_gradient_penalty(self, true_samples, generated_samples, c):
        """
        Compute gradient penalty for WGAN-GP
        
        Theory: The Wasserstein distance requires the critic to be 1-Lipschitz continuous.
        Instead of weight clipping (which can cause problems), we use gradient penalty.
        
        We penalize the critic when ||∇_x̂ C(x̂)||₂ deviates from 1, where x̂ is an
        interpolation between real and fake samples.
        
        Mathematical formulation:
        GP = λ * E[(||∇_x̂ C(x̂)||₂ - 1)²]
        where x̂ = ε*real + (1-ε)*fake, ε ~ U(0,1)
        
        This ensures the critic's gradients have norm close to 1 everywhere,
        satisfying the Lipschitz constraint.
        
        Args:
            ture_samples: Real data batch
            generated_samples: Generated fake data batch
            c: Conditioning information
            
        Returns:
            gradient_penalty: Scalar penalty value
        """

        if self.D is None:
            raise ValueError('The Discriminator (Critic) has not been initialized. ' \
            'Use first one of the following methods: set_discriminator, set_critic ')
        
        if true_samples.shape != generated_samples.shape:
            raise ValueError(f'true samples shape {true_samples.shape} do not match generated samples shape {generated_samples.shape}')
        
        batch_size = true_samples.size(0)

        # Sample random interpolation weight
        alpha = torch.rand(batch_size, 1).to(self.DEVICE)
        alpha = alpha.expand_as(true_samples)
        
        # Create interpolated samples
        interpolates = (alpha * true_samples + (1 - alpha) * generated_samples).requires_grad_(True)
        
        # Get critic scores on interpolated samples
        critic_interpolates = self.D(interpolates, c)
        
        # Compute gradients of critic output w.r.t. interpolated samples
        gradients = torch.autograd.grad(
            outputs=critic_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(critic_interpolates),
            create_graph=True,  # Allow backprop through this computation
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # Compute gradient norm
        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        
        # Penalize deviation from norm of 1
        gradient_penalty = ((gradient_norm - 1) ** 2).mean()
        
        return gradient_penalty
    

    def early_stop_check(self, current_w_distance, epoch):
        """
        Determine if training should stop early based on Wasserstein distance
        
        Theory: The Wasserstein distance is a meaningful metric - lower means
        the generator's distribution is closer to the real data distribution.
        Unlike BCE loss in standard GANs, W-distance directly measures the
        "cost" of transforming one distribution into another.
        
        When W-distance stops improving (plateaus), it suggests:
        1. Generator has learned the distribution well
        2. Further training may lead to overfitting
        3. Computational resources are better spent elsewhere
        
        Args:
            current_w_distance: Current epoch's Wasserstein distance
            epoch: Current epoch number
            
        Returns:
            bool: True if training should stop
        """
    
        if current_w_distance < (self.best_w_distance - self.early_stopping_min_delta):
            # Significant improvement found
            self.best_w_distance = current_w_distance
            self.patience_counter = 0
            
            # Save best model states
            self.best_generator_state = self.G.state_dict().copy() if self.G else None
            self.best_critic_state = self.D.state_dict().copy() if self.D else None
            
            print(f"  → New best W-distance: {current_w_distance:.4f}")
            return False
        else:
            # No improvement
            self.patience_counter += 1
            print(f"  No improvement. Patience: {self.patience_counter}/{self.early_stopping_patience}")
            
            if self.patience_counter >= self.early_stopping_patience:
                print(f"\n Early stopping triggered at epoch {epoch}")
                print(f" Best W-distance: {self.best_w_distance:.4f}")
                
                # Restore best model
                if self.best_generator_state:
                    if self.G:
                        self.G.load_state_dict(self.best_generator_state)
                        print("   Restored best generator weights")
                    else:
                        raise ValueError('Generator is None')      
    
                if self.best_critic_state:
                    if self.D:
                        self.D.load_state_dict(self.best_critic_state)
                        print("   Restored best critic weights")
                    else:
                        raise ValueError('Critic is None')
                
                return True
            
            return False
        

    def compute_epoch_wasserstein_distance(self, data_loader):
        """
        Compute average Wasserstein distance over entire dataset
        
        This gives a more stable metric than single-batch measurements
        for early stopping decisions.
        
        Args:
            data_loader: DataLoader for the dataset
            
        Returns:
            float: Average Wasserstein distance
        """
        if self.D and self.G:
            self.D.eval()
            self.G.eval()
        else:
            raise ValueError('Generator and Critic are not initiliazed. Use set_generator and set_discriminator/set_critic methods')
            
        total_w_dist = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for output, trajectory in data_loader:
                x = output.to(self.DEVICE)
                c = trajectory.to(self.DEVICE)
                current_batch_size = x.size(0)
                
                # Real samples score
                critic_real = self.D(x, c).mean()
                
                # Fake samples score
                z = torch.randn((current_batch_size, self.z_dim)).to(self.DEVICE)
                fake_samples = self.G(c, z)
                critic_fake = self.D(fake_samples, c).mean()
                
                # Wasserstein distance approximation
                w_dist = critic_real - critic_fake
                total_w_dist += w_dist.item()
                num_batches += 1
        
        self.D.train()
        self.G.train()
        
        return total_w_dist / num_batches if num_batches > 0 else float('inf')


    def train(self, data: TensorDataset, save_history: bool = False,
               distance_metric:str = 'js_divergence', early_stopping_waiting: int = 0):
        """
        Override training method with Wasserstein loss and gradient penalty

        args:
            early_stopping_waiting :int -> how many epochs are needed before checking if early stopping is needed
        
        Key differences from parent MyCGAN.train():
        1. No binary labels (D_labels, D_fakes) - we use raw scores
        2. Critic loss: -E[C(real)] + E[C(fake)] + λ*GP
        3. Generator loss: -E[C(fake)]
        4. More frequent critic updates (n_critic typically 5 vs 1)
        5. Lower learning rates (0.0001 vs 0.0005)
        
        Theoretical insight:
        Wasserstein distance = sup_{||f||_L≤1} E[f(real)] - E[f(fake)]
        We approximate this by training a critic to maximize this difference
        while enforcing the Lipschitz constraint via gradient penalty.
        """

        if self.D is None or self.G is None:
            raise Exception("Critic or Generator is not defined. Use set_critic/set_discriminator or set_generator")
        
        if not isinstance(data, TensorDataset):
            raise Exception(f"Invalid input data format. A TensorDataset should be provided. Provided {type(data)}")
        
        if early_stopping_waiting<0:
            raise ValueError(f'The waiting factor for early stopping calculation should be positive. {early_stopping_waiting} provided.')
        

        # Move models to device
        self.G.to(self.DEVICE)
        self.D.to(self.DEVICE)

        data_loader = DataLoader(dataset=data, batch_size=self.batch_size, shuffle=True, drop_last=True)
        C_opt = torch.optim.Adam(self.D.parameters(), lr=0.0001, betas=(0.5, 0.999))
        G_opt = torch.optim.Adam(self.G.parameters(), lr=0.0001, betas=(0.5, 0.999))
        
        df = None
        step = 1
        self.D.train()
        self.G.train()
        G_loss = float('inf')
        C_loss = float('inf')
        start_time = time.time()
        
        for epoch in range(self.max_epoch):
            predictions_list = []
            targets_list = []
            condition_list = []
            C_loss_list = []
            G_loss_list = []
            wasserstein_dist_list = []
            epoch_start_time = time.time()

            for idx, (prob_dist, trajectory) in enumerate(data_loader):
                x = prob_dist.to(self.DEVICE)
                c = trajectory.to(self.DEVICE)
                current_batch_size = x.size(0)

                # -----TRAINING THE CRITIC-----
                C_opt.zero_grad()

                # real samples
                critic_real = self.D(x,c)
                critic_real_mean = critic_real.mean()

                # generated samples
                z = torch.randn((current_batch_size, self.z_dim)).to(self.DEVICE)
                fake_samples = self.G(c,z).detach()
                critic_fake = self.D(fake_samples, c)
                critic_fake_mean = critic_fake.mean()

                # backpropagation -> critic minimize: -E[C(real)] + E[C(fake)] + λ*GP
                gradient_penalty = self.compute_gradient_penalty(x, fake_samples, c)
                C_loss = -critic_real_mean + critic_fake_mean + self.lambda_gp*gradient_penalty
                C_loss.backward()
                C_opt.step()


                # -----TRAINING THE GENERATOR-----
                if step % self.n_critic == 0:
                    G_opt.zero_grad()

                    #generated samples
                    z = torch.randn(current_batch_size, self.z_dim).to(self.DEVICE)
                    fake_samples_g = self.G(c,z)
                    critic_fake_g = self.D(fake_samples_g, c)

                    #backpropagation -> generator minimize -E[C(fake)]
                    G_loss = -critic_fake_g.mean()
                    G_loss.backward()
                    G_opt.step()

                    # # clean up to prevent memory accumulation
                    del fake_samples_g, critic_fake_g


                # LOGGING
                wasserstein_dist = critic_real_mean.item() - critic_fake_mean.item()
                if step%500==0:
                    print(f'Epoch: {epoch}/{self.max_epoch}, Step: {step}, '
                          f'C_loss: {C_loss.item():.4f}, G_loss: {G_loss.item():.4f}, '
                          f'W_dist: {wasserstein_dist:.4f}, GP: {gradient_penalty.item():.4f}')
                    
                # clean up to prevent memory accumulation
                del fake_samples, gradient_penalty
                
                #save history
                if save_history and idx==int(len(data_loader)/2):
                    self.G.eval()
                    with torch.no_grad():
                        z = torch.randn(current_batch_size, self.z_dim).to(self.DEVICE)
                        generated = self.G(c, z).cpu().numpy()

                    for i, row in enumerate(generated):
                        pred = row.tolist()
                        true = x[i, :].cpu().tolist()
                        predictions_list.append(pred)
                        targets_list.append(true)
                        condition_list.append(c[i, :].cpu().tolist())
                        C_loss_list.append(round(float(C_loss), 4))
                        G_loss_list.append(round(float(G_loss), 4))
                        wasserstein_dist_list.append(round(float(wasserstein_dist),4))
                    
                    self.G.train()
                
                step+=1

            epoch_time = time.time() - epoch_start_time
            # -----EARLY STOP CHECK-----
            if epoch > early_stopping_waiting:
                avg_w_distance = self.compute_epoch_wasserstein_distance(data_loader) 
                print(f"\n Epoch {epoch} Summary:")
                print(f"   Avg W-distance: {avg_w_distance:.4f}")
                print(f"   Epoch time: {epoch_time:.2f}s")
                
                # Check early stopping
                if self.early_stop_check(avg_w_distance, epoch):
                    print(f"\n Training stopped early after {epoch + 1} epochs")
                    break
            
            # Clear CUDA cache periodically to prevent memory buildup
            if torch.cuda.is_available() and epoch % 5 == 0:
                torch.cuda.empty_cache()

            # Build DataFrame
            if len(predictions_list) > 1:
                js_divergence = compute_js(np.array(predictions_list), np.array(targets_list))
                distance = np.mean(js_divergence)
                entries = pd.DataFrame({
                    "epoch": [int(epoch)] * len(predictions_list),
                    "generated": [json.dumps(pred) for pred in predictions_list],
                    "true": [json.dumps(true) for true in targets_list],
                    "C_loss": C_loss_list,
                    "G_loss": G_loss_list,
                    "wasserstein distance": wasserstein_dist_list,
                    "distance": distance
                })
                
                if df is None:
                    df = entries
                else:
                    df = pd.concat([df, entries], ignore_index=True)
        
        end_time = time.time()
        print(f"\n  Total training time: {end_time - start_time:.2f} seconds")
        
        if df is not None:
            df.to_csv("wcgan_generated_vs_true.csv", index=False)

                
            


            
        