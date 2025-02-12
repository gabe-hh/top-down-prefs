import torch
from torch import nn, optim, Tensor
import torch.nn.functional as F
import wandb
import os

import src.model.encoder as encoder
import src.model.decoder as decoder
import src.model.transition as transition
from src.model.model_low import ModelLow
import src.utils.latent_handler as latent_handler
from src.utils.loss import mse_loss
#import src.train.utils as train_utils
import src.utils.utils as utils
from src.utils.eval import plot_img_comparison_batch, generate_low_transition_predictions

class LowTrainerOnline():
    def __init__(self,
                 optimizer,
                 batch_size,
                 trajectory_length,
                 beta=1.0,
                 early_stopper=None,
                 device='cuda',
                 kl_alpha=0.8,
                 eval_img_root='.',
                 eval_every_n_epochs=20):
        
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.trajectory_length = trajectory_length
        self.beta = beta
        self.early_stopper = early_stopper
        self.device = device
        self.kl_alpha = kl_alpha
        if self.kl_alpha is None: 
            self.kl_balancing = False
        else:
            self.kl_balancing = True

        self.img_root = eval_img_root
        self.eval_every_n_epochs = eval_every_n_epochs
        
        self.wandb_enabled = wandb.run is not None
        if not self.wandb_enabled:
            print("Wandb not enabled, printing training logs to console")

    def compute_loss(self, model, x, x_hat, dist, dist_prior):
        recon_loss = mse_loss(x_hat, x) / self.batch_size

        if self.kl_balancing:
            kl_loss = model.latent_handler.balanced_kl_divergence(dist, dist_prior, alpha=self.kl_alpha)
        else:
            kl_loss = model.latent_handler.kl_divergence(dist, dist_prior)
        kl_loss /= self.batch_size

        loss = recon_loss + self.beta * kl_loss
        return loss, recon_loss, kl_loss

    def train(self, model, env, num_epochs, model_root):
        model = model.to(self.device)
        action_dim = model.action_dim
        zero_prior = model.zero_prior(self.batch_size, device=self.device)
        best_loss = float('inf')
        for epoch in range(num_epochs):
            obs,_ = env.reset()

            dist_prior = zero_prior
            h = model.zero_hidden(self.batch_size).to(self.device)

            epoch_loss = 0

            recon_loss_total = 0
            kl_loss_total = 0
            loss_denom = 0

            for t in range(self.trajectory_length):
                o_tensor = utils.obs2tensor(obs['image']).to(self.device)
                action, a_tensor = utils.get_random_action(action_dim, self.batch_size)
                a_tensor = a_tensor.to(self.device)

                x_hat, z, dist = model(o_tensor, h)
                z_next, dist_next, h = model.transition(z, a_tensor, h)

                # Compute loss
                loss, recon_loss, kl_loss = self.compute_loss(model, o_tensor, x_hat, dist, dist_prior)

                epoch_loss += loss

                recon_loss_total += recon_loss.item()
                kl_loss_total += kl_loss.item()
                loss_denom += 1

                # Update prior
                dist_prior = dist_next
                obs,_,_,_,_ = env.step(action)

            self.optimizer.zero_grad()
            epoch_loss.backward()
            self.optimizer.step()

            avg_loss = (recon_loss_total + kl_loss_total) / loss_denom
            if self.wandb_enabled:
                wandb.log({'epoch': epoch+1, 'loss': avg_loss, 
                       'recon_loss': recon_loss_total / loss_denom, 
                       'kl_loss': kl_loss_total / loss_denom})
            else:
                print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss}, Recon Loss: {recon_loss_total / loss_denom}, KL Loss: {kl_loss_total / loss_denom}')
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                model.save_model(model_root, 'best.pt')
                if self.wandb_enabled:
                    if epoch > 500:
                        plot_img_comparison_batch(o_tensor, x_hat, 'Observation', 'Reconstruction', name='best_lo_recon', root=self.img_root, wandb_log=self.wandb_enabled, limit=8)
                    wandb.log({'best_loss': best_loss})
                    wandb.run.summary['best_loss'] = best_loss

            if (epoch + 1) % self.eval_every_n_epochs == 0:
                print(f'Plotting evaluation images at epoch {epoch+1}')
                plot_img_comparison_batch(o_tensor, x_hat, 'Observation', 'Reconstruction', name='eval_lo_recon', root=self.img_root, wandb_log=self.wandb_enabled, limit=8)
                self.eval_transitions(model, env, 8, self.img_root, wandb_log=self.wandb_enabled)
                model.save_model(model_root, 'latest.pt')

        model.save_model(model_root, 'final.pt')
        #self.eval_transitions(model, env, 8, wandb_log=self.wandb_enabled)

    def eval_transitions(self, model, env, num_transitions, path, wandb_log=True):
        a_indices, a_onehot, = utils.get_random_action_sequence(self.trajectory_length-1, model.action_dim, self.batch_size, device=self.device)
        a_onehot = a_onehot.to(self.device)
        generate_low_transition_predictions(model, env, a_indices, a_onehot, device=self.device, wandb_log=wandb_log, qty=num_transitions, name=f'eval_lo_transitions', root=path)