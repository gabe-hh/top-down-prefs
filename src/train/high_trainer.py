import torch
import wandb
import tqdm
import src.utils.utils as utils
from torch.utils.data import Subset, random_split
from torch.utils.data import DataLoader
from src.model.world_model import high_tick
from src.utils.eval import plot_distribution_comparison, plot_img_comparison_batch, generate_high_trajectory_evaluation, generate_high_transition_predictions, generate_high_transition_predictions_replay, generate_high_trajectory_evaluation_replay

class HighTrainerOnline():
    def __init__(self, 
                 optimizer,
                 batch_size,
                 trajectory_length,
                 beta=1.0,
                 device='cuda',
                 kl_alpha=0.8,
                 eval_img_root='.',
                 eval_every_n_epochs=20,
                 bptt_truncate=None):
        
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.trajectory_length = trajectory_length
        self.beta = beta
        self.device = device
        self.kl_alpha = kl_alpha
        if self.kl_alpha is None: 
            self.kl_balancing = False
        else:
            self.kl_balancing = True
        
        self.img_root = eval_img_root
        self.eval_every_n_epochs = eval_every_n_epochs 
        self.bptt_truncate = bptt_truncate

        self.wandb_enabled = wandb.run is not None
        if not self.wandb_enabled:
            print("Wandb not enabled, printing training logs to console")

    def compute_loss(self, model_high, model_low, x, x_hat, dist, dist_prior, beta):
        recon_loss = model_low.latent_handler.sample_reconstruction_loss(x, x_hat) / self.batch_size

        if self.kl_balancing:
            kl_loss = model_high.latent_handler.balanced_kl_divergence(dist, dist_prior, alpha=self.kl_alpha)
        else:
            kl_loss = model_high.latent_handler.kl_divergence(dist, dist_prior)
        kl_loss /= self.batch_size

        loss = recon_loss + beta * kl_loss
        return loss, recon_loss, kl_loss
    
    def high_tick(self, model_low, env, h_low, obs):
        low_action_dim = model_low.action_dim
        for t in range(model_low.steps):
            o_tensor = utils.obs2tensor(obs['image']).to(self.device)
            terminal = model_low.is_terminal(t)

            if not terminal:
                action, a_tensor = utils.get_random_action(low_action_dim, self.batch_size)
                a_tensor = a_tensor.to(self.device)

            with torch.no_grad():
                _, z_low, dist_low = model_low(o_tensor, h_low)
            
            if not terminal:
                _,_, h_low = model_low.transition(z_low, a_tensor, h_low)
            
            if t == 0:
                initial_z_low = z_low.clone()
                initial_dist_low = dist_low
                initial_h_low = h_low.clone()
            elif terminal:
                final_z_low = z_low.clone()
                final_dist_low = dist_low
                final_h_low = h_low.clone()
                h_low = model_low.process_hidden_state(t, h_low, self.batch_size)
                return {
                    'initial_z_low': initial_z_low,
                    'initial_dist_low': initial_dist_low,
                    'initial_h_low': initial_h_low,
                    'final_z_low': final_z_low,
                    'final_dist_low': final_dist_low,
                    'final_h_low': final_h_low,
                    'h_low': h_low,
                    'obs': obs
                }
            
            obs,_,_,_,_ = env.step(action)

    def train(self, model_high, model_low, model_action, env, num_epochs, model_root):
        best_loss = float('inf')

        low_trajectory_length = model_low.steps

        model_high = model_high.to(self.device)
        model_low = model_low.to(self.device)
        model_action = model_action.to(self.device)
        zero_prior = model_high.zero_prior(self.batch_size, device=self.device)
        zero_prior = tuple(d.detach() for d in zero_prior)

        for epoch in tqdm.trange(num_epochs, desc="Training"):
            obs,_ = env.reset()

            low_action_dim = model_low.action_dim

            h_high = model_high.zero_hidden(self.batch_size, device=self.device)
            h_low = model_low.zero_hidden(self.batch_size, device=self.device)

            epoch_loss = 0
            recon_loss_total = 0
            kl_loss_total = 0
            loss_denom = 0

            accumulated_loss = torch.tensor(0.0, device=self.device)

            beta = self.beta(epoch) if callable(self.beta) else self.beta

            dist_prior = tuple(d.clone() for d in zero_prior)

            for T in range(self.trajectory_length):
                # Nested low-level trajectory
                with torch.no_grad():
                    low_data = high_tick(model_low, env, h_low, obs, self.device, self.batch_size)
                initial_z_low = low_data['initial_z']
                initial_dist_low = low_data['initial_dist']
                initial_h_low = low_data['initial_h']
                final_z_low = low_data['final_z']
                final_dist_low = low_data['final_dist']
                final_h_low = low_data['final_h']
                h_low = low_data['h']
                obs = low_data['obs']

                combined_low_state = torch.cat([initial_z_low.view(self.batch_size, -1), initial_h_low], dim=-1)
                x_hat, z_high, dist_high = model_high(combined_low_state, h_high)
                x_hat = model_low.latent_handler.reshape_latent(x_hat, model_low.latent_size)
                with torch.no_grad():
                    la_z_hat,_,a_high, dist_action = model_action(initial_z_low, final_z_low, initial_h_low, final_h_low)
                _, dist_high_next, h_high = model_high.transition(z_high, a_high, h_high)
                
                loss, recon_loss, kl_loss = self.compute_loss(model_high, model_low, initial_z_low, x_hat, dist_high, dist_prior, beta)

                accumulated_loss = accumulated_loss + loss
                epoch_loss += loss.item()
                recon_loss_total += recon_loss.item()
                kl_loss_total += kl_loss.item()
                loss_denom += 1

                if self.bptt_truncate and self.bptt_truncate > 0 and (T + 1) % self.bptt_truncate == 0:
                    self.optimizer.zero_grad()
                    accumulated_loss.backward()
                    self.optimizer.step()
                    accumulated_loss = torch.tensor(0.0, device=self.device)
                    h_high = h_high.detach()

                dist_prior = dist_high_next
            
            if accumulated_loss > 0:
                self.optimizer.zero_grad()
                accumulated_loss.backward()
                self.optimizer.step()
            
            avg_loss = epoch_loss / loss_denom

            if self.wandb_enabled:
                wandb.log({ "epoch": epoch+1,
                            "loss": avg_loss,
                            "recon_loss": recon_loss_total / loss_denom,
                            "kl_loss": kl_loss_total / loss_denom})
            else:
                print(f"Epoch {epoch+1}: loss {avg_loss}, recon_loss {recon_loss_total / loss_denom}, kl_loss {kl_loss_total / loss_denom}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                if self.wandb_enabled:
                    wandb.log({"min_val_loss": best_loss})
                    model_high.save_model(model_root, 'best.pt')
            
            if (epoch + 1) % self.eval_every_n_epochs == 0:
                if not self.wandb_enabled:
                    print("Evaluating model")
                with torch.no_grad():
                    logits, _ = initial_dist_low
                    plot_distribution_comparison(logits, x_hat, 'Low Posterior', 'Reconstruction', name='high-reconstructions', root=self.img_root, wandb_log=self.wandb_enabled, limit=8)
                    plot_distribution_comparison(final_z_low, la_z_hat, 'Final Low Posterior', 'Latent Action Recon', name='la-recon', root=self.img_root, wandb_log=self.wandb_enabled, limit=8)
                    decoded_img = model_low.decode(initial_z_low, initial_h_low)
                    sampled_x_hat = model_low.latent_handler.sample_if_logits(x_hat)
                    decoded_pred = model_low.decode(sampled_x_hat, initial_h_low)
                    plot_img_comparison_batch(decoded_img, decoded_pred, 'Decoded Low Posterior', 'Decoded High Recon', name='decoded-high-recon', root=self.img_root, wandb_log=self.wandb_enabled, limit=8)

                    generate_high_trajectory_evaluation(model_high, model_low, model_action, env,
                                                        device=self.device, name='high-transition-predictions', root=self.img_root, wandb_log=self.wandb_enabled, qty=8)

                    a_indices, a_onehot = utils.get_random_action_sequence((model_low.steps-1) * 30, model_low.action_dim, self.batch_size, device=self.device)
                    generate_high_transition_predictions(model_high, model_low, model_action, env, a_indices, a_onehot, device=self.device, name='high-multistep-predictions', root=self.img_root, wandb_log=self.wandb_enabled, qty=16)

        model_high.save_model(model_root, 'final.pt')

class HighTrainerReplay():
    def __init__(self, 
                 optimizer,
                 batch_size,
                 trajectory_length,
                 beta=1.0,
                 device='cuda',
                 kl_alpha=0.8,
                 eval_img_root='.',
                 eval_every_n_epochs=20,
                 bptt_truncate=None):
        
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.trajectory_length = trajectory_length
        self.beta = beta
        self.device = device
        self.kl_alpha = kl_alpha
        if self.kl_alpha is None: 
            self.kl_balancing = False
        else:
            self.kl_balancing = True
        
        self.img_root = eval_img_root
        self.eval_every_n_epochs = eval_every_n_epochs 
        self.bptt_truncate = bptt_truncate

        self.wandb_enabled = wandb.run is not None
        if not self.wandb_enabled:
            print("Wandb not enabled, printing training logs to console")

    def compute_loss(self, model_high, model_low, x, x_hat, dist, dist_prior, beta):
        recon_loss = model_low.latent_handler.sample_reconstruction_loss(x, x_hat) / self.batch_size

        if self.kl_balancing:
            kl_loss = model_high.latent_handler.balanced_kl_divergence(dist, dist_prior, alpha=self.kl_alpha)
        else:
            kl_loss = model_high.latent_handler.kl_divergence(dist, dist_prior)
        kl_loss /= self.batch_size

        loss = recon_loss + beta * kl_loss
        return loss, recon_loss, kl_loss
    
    def split_dataset(self, dataset, train_ratio=0.8, eval_size=0):
        all_indices = torch.randperm(len(dataset))
        example_indices = all_indices[:eval_size]
        remaining_indices = all_indices[eval_size:]
        
        example_dataset = Subset(dataset, example_indices)
        remaining_dataset = Subset(dataset, remaining_indices)
        
        # Split remaining data into train and val
        train_size = int(0.8 * len(remaining_dataset))
        val_size = len(remaining_dataset) - train_size
        train_dataset, val_dataset = random_split(remaining_dataset, [train_size, val_size])
        
        return train_dataset, val_dataset, example_dataset

    def train(self, model_high, model_low, model_action, dataset, num_epochs, model_root):
        best_loss = float('inf')

        low_trajectory_length = model_low.steps

        torch.manual_seed(42)

        train_dataset, val_dataset, example_dataset = self.split_dataset(dataset, eval_size=32)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        example_loader = DataLoader(example_dataset, batch_size=self.batch_size, shuffle=False)

        model_high = model_high.to(self.device)
        model_low = model_low.to(self.device)
        model_action = model_action.to(self.device)

        for epoch in tqdm.trange(num_epochs, desc="Training"): #TODO: Move inner loops to separate function for both train and val

            epoch_loss = 0
            recon_loss_total = 0
            kl_loss_total = 0
            val_epoch_loss = 0
            val_recon_loss_total = 0
            val_kl_loss_total = 0
            loss_denom = 0

            beta = self.beta(epoch) if callable(self.beta) else self.beta
            torch.autograd.set_detect_anomaly(True)
            
            for trajectory in train_loader:
                batch_size = trajectory[0]['initial_z'].shape[0]
                
                zero_prior = model_high.zero_prior(batch_size, device=self.device)
                zero_prior = tuple(d.detach() for d in zero_prior)
                accumulated_loss = torch.tensor(0.0, device=self.device)
                dist_prior = tuple(d.clone() for d in zero_prior)
                
                h_high = model_high.zero_hidden(batch_size, device=self.device)
                for i,T in enumerate(trajectory):
                    #batch_size = T['initial_z'].shape[0]
                    initial_z_low = T['initial_z'].to(self.device)
                    initial_dist_low = T['initial_dist']
                    initial_h_low = T['initial_h'].to(self.device)
                    final_z_low = T['final_z']
                    final_dist_low = T['final_dist']
                    final_h_low = T['final_h']
                    action_dist = T['action_dist']
                    a_high = T['sampled_action'].to(self.device)

                    #print(f"Batch size: {batch_size}, initial z: {initial_z_low.shape}, initial h: {initial_h_low.shape}, final z: {final_z_low.shape}, final h: {final_h_low.shape}, action: {a_high.shape}")
                    combined_low_state = torch.cat([initial_z_low.view(batch_size, -1), initial_h_low], dim=-1)
                    x_hat, z_high, dist_high = model_high(combined_low_state, h_high)
                    x_hat = model_low.latent_handler.reshape_latent(x_hat, model_low.latent_size)
                    # with torch.no_grad():
                    #     la_z_hat,_,a_high, dist_action = model_action(initial_z_low, final_z_low, initial_h_low, final_h_low)
                    a_high = a_high.clone().detach()
                    _, dist_high_next, h_high = model_high.transition(z_high, a_high, h_high)
                    
                    loss, recon_loss, kl_loss = self.compute_loss(model_high, model_low, initial_z_low, x_hat, dist_high, dist_prior, beta)

                    accumulated_loss = accumulated_loss + loss
                    epoch_loss += recon_loss.item() + kl_loss.item()
                    recon_loss_total += recon_loss.item()
                    kl_loss_total += kl_loss.item()
                    loss_denom += 1

                    if self.bptt_truncate and self.bptt_truncate > 0 and (i + 1) % self.bptt_truncate == 0:
                        self.optimizer.zero_grad()
                        accumulated_loss.backward()
                        self.optimizer.step()
                        accumulated_loss = torch.tensor(0.0, device=self.device)
                        h_high = h_high.detach().clone()

                    dist_prior = dist_high_next
                #break # For testing purposes
            
            if accumulated_loss > 0:
                self.optimizer.zero_grad()
                accumulated_loss.backward()
                self.optimizer.step()
            
            avg_loss = epoch_loss / loss_denom
            avg_recon_loss = recon_loss_total / loss_denom
            avg_kl_loss = kl_loss_total / loss_denom

            loss_denom = 0
            with torch.no_grad():
                for trajectory in val_loader:
                    batch_size = trajectory[0]['initial_z'].shape[0]
                    
                    zero_prior = model_high.zero_prior(batch_size, device=self.device)
                    zero_prior = tuple(d.detach() for d in zero_prior)
                    accumulated_loss = torch.tensor(0.0, device=self.device)
                    dist_prior = tuple(d.clone() for d in zero_prior)
                    
                    h_high = model_high.zero_hidden(batch_size, device=self.device)
                    for i,T in enumerate(trajectory):
                        #batch_size = T['initial_z'].shape[0]
                        initial_z_low = T['initial_z'].to(self.device)
                        initial_dist_low = T['initial_dist']
                        initial_h_low = T['initial_h'].to(self.device)
                        final_z_low = T['final_z']
                        final_dist_low = T['final_dist']
                        final_h_low = T['final_h']
                        action_dist = T['action_dist']
                        a_high = T['sampled_action'].to(self.device)

                        #print(f"Batch size: {batch_size}, initial z: {initial_z_low.shape}, initial h: {initial_h_low.shape}, final z: {final_z_low.shape}, final h: {final_h_low.shape}, action: {a_high.shape}")
                        combined_low_state = torch.cat([initial_z_low.view(batch_size, -1), initial_h_low], dim=-1)
                        x_hat, z_high, dist_high = model_high(combined_low_state, h_high)
                        x_hat = model_low.latent_handler.reshape_latent(x_hat, model_low.latent_size)
                        # with torch.no_grad():
                        #     la_z_hat,_,a_high, dist_action = model_action(initial_z_low, final_z_low, initial_h_low, final_h_low)
                        _, dist_high_next, h_high = model_high.transition(z_high, a_high, h_high)
                        
                        loss, recon_loss, kl_loss = self.compute_loss(model_high, model_low, initial_z_low, x_hat, dist_high, dist_prior, beta)

                        val_epoch_loss += recon_loss.item() + kl_loss.item()
                        val_recon_loss_total += recon_loss.item()
                        val_kl_loss_total += kl_loss.item()
                        loss_denom += 1

                        dist_prior = dist_high_next
            
            val_avg_loss = val_epoch_loss / loss_denom
            val_avg_recon_loss = val_recon_loss_total / loss_denom
            val_avg_kl_loss = val_kl_loss_total / loss_denom

            if self.wandb_enabled:
                wandb.log({ "epoch": epoch+1,
                            "loss": avg_loss,
                            "recon_loss": avg_recon_loss,
                            "kl_loss": avg_kl_loss,
                            "val_loss": val_avg_loss,
                            "val_recon_loss": val_avg_recon_loss,
                            "val_kl_loss": val_avg_kl_loss})
            else:
                print(f"Epoch {epoch+1}: loss {avg_loss}, recon_loss {recon_loss_total / loss_denom}, kl_loss {kl_loss_total / loss_denom}")

            if val_avg_loss < best_loss:
                best_loss = val_avg_loss
                if self.wandb_enabled:
                    wandb.log({"min_val_loss": best_loss})
                    model_high.save_model(model_root, 'best.pt')

            if (epoch + 1) % self.eval_every_n_epochs == 0:
                if not self.wandb_enabled:
                    print("Evaluating model")
                with torch.no_grad():
                    logits, _ = initial_dist_low
                    plot_distribution_comparison(logits, x_hat, 'Low Posterior', 'Reconstruction', name='high-reconstructions', root=self.img_root, wandb_log=self.wandb_enabled, limit=8)
                    #plot_distribution_comparison(final_z_low, la_z_hat, 'Final Low Posterior', 'Latent Action Recon', name='la-recon', root=self.img_root, wandb_log=self.wandb_enabled, limit=8)
                    decoded_img = model_low.decode(initial_z_low, initial_h_low)
                    sampled_x_hat = model_low.latent_handler.sample_if_logits(x_hat)
                    decoded_pred = model_low.decode(sampled_x_hat, initial_h_low)
                    plot_img_comparison_batch(decoded_img, decoded_pred, 'Decoded Low Posterior', 'Decoded High Recon', name='decoded-high-recon', root=self.img_root, wandb_log=self.wandb_enabled, limit=8)

                    for ex_traj in example_loader:
                        generate_high_trajectory_evaluation_replay(model_high, model_low, trajectory,
                                                            device=self.device, name='high-transition-predictions', root=self.img_root, wandb_log=self.wandb_enabled, qty=8)

                        generate_high_transition_predictions_replay(model_high, model_low, trajectory, device=self.device, name='high-multistep-predictions', root=self.img_root, wandb_log=self.wandb_enabled, qty=16)

        model_high.save_model(model_root, 'final.pt')