import torch
from torch import nn, optim, Tensor
import torch.nn.functional as F
import wandb
import os
import tqdm

from src.utils.loss import mse_loss, bernoulli_nll
#import src.train.utils as train_utils
import src.utils.utils as utils
from src.utils.eval import plot_img_comparison_batch, generate_low_transition_predictions
from src.utils.utils import add_goal_mask, separate_goal_mask, generate_goal_mask, get_goal_mask

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
                 eval_every_n_epochs=20,
                 bptt_truncate=None):
        
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
        self.bptt_truncate = bptt_truncate
        
        self.wandb_enabled = wandb.run is not None
        if not self.wandb_enabled:
            print("Wandb not enabled, printing training logs to console")

    def compute_loss(self, model, x, x_hat, dist, dist_prior, x_mask=None, x_hat_mask=None):
        recon_loss = mse_loss(x_hat, x) / self.batch_size

        if self.kl_balancing:
            kl_loss = model.latent_handler.balanced_kl_divergence(dist, dist_prior, alpha=self.kl_alpha)
        else:
            kl_loss = model.latent_handler.kl_divergence(dist, dist_prior)
        kl_loss /= self.batch_size

        if x_mask is not None:
            goal_loss = bernoulli_nll(x_hat_mask, x_mask) / self.batch_size
            loss = recon_loss + self.beta * kl_loss + goal_loss
            goal_loss = goal_loss.item()
        else:
            goal_loss = 0
            loss = recon_loss + self.beta * kl_loss
        recon_loss = recon_loss.item()
        kl_loss = kl_loss.item()
        return loss, recon_loss, kl_loss, goal_loss

    def train(self, model, env, num_epochs, model_root, goal_mask=False):
        model = model.to(self.device)
        action_dim = model.action_dim
        zero_prior = model.zero_prior(self.batch_size, device=self.device)
        zero_prior = tuple(d.detach() for d in zero_prior)
        best_loss = float('inf')
        for epoch in tqdm.trange(num_epochs, desc="Training"):
            obs,_ = env.reset()

            dist_prior = zero_prior
            h = model.zero_hidden(self.batch_size).to(self.device)

            epoch_loss = 0
            recon_loss_total = 0
            kl_loss_total = 0
            goal_loss_total = 0
            loss_denom = 0

            # Track accumulated loss for truncated BPTT
            accumulated_loss = torch.tensor(0.0, device=self.device)

            for t in range(self.trajectory_length):
                unmasked_o_tensor = utils.obs2tensor(obs['image']).to(self.device)
                if goal_mask:
                    o_tensor = add_goal_mask(unmasked_o_tensor.clone())
                    mask_o_tensor = get_goal_mask(o_tensor)
                else:
                    o_tensor = unmasked_o_tensor
                    mask_o_tensor = None

                terminal = model.is_terminal(t)
                if not terminal:
                    action, a_tensor = utils.get_random_action(action_dim, self.batch_size)
                    a_tensor = a_tensor.to(self.device)

                decoded, z, dist = model(o_tensor, h)
                x_hat, x_hat_mask = decoded

                if not terminal:
                    z_next, dist_next, h = model.transition(z, a_tensor, h)

                loss, recon_loss, kl_loss, goal_loss = self.compute_loss(model, unmasked_o_tensor, x_hat, dist, dist_prior, x_mask=mask_o_tensor, x_hat_mask=x_hat_mask)

                accumulated_loss = accumulated_loss + loss
                epoch_loss += loss.item()
                recon_loss_total += recon_loss
                kl_loss_total += kl_loss
                goal_loss_total += goal_loss
                loss_denom += 1

                # Perform truncated BPTT if conditions are met
                if self.bptt_truncate and self.bptt_truncate > 0 and (t + 1) % self.bptt_truncate == 0:
                    self.optimizer.zero_grad()
                    accumulated_loss.backward()
                    self.optimizer.step()
                    accumulated_loss = torch.tensor(0.0, device=self.device)
                    h = h.detach()  # Detach hidden state

                # Update prior
                if not terminal:
                    dist_prior = dist_next
                    obs,_,_,_,_ = env.step(action)
                else: #TODO: We might want to keep the prior, or set it to the current distribution
                    h = model.process_hidden_state(t, h, self.batch_size)
                    dist_prior = model.zero_prior(self.batch_size, device=self.device)

            # Handle remaining steps if any
            if accumulated_loss > 0:
                self.optimizer.zero_grad()
                accumulated_loss.backward()
                self.optimizer.step()

            avg_loss = epoch_loss / loss_denom
            if self.wandb_enabled:
                wandb.log({'epoch': epoch+1, 'loss': avg_loss, 
                       'recon_loss': recon_loss_total / loss_denom, 
                       'kl_loss': kl_loss_total / loss_denom,
                       'goal_loss': goal_loss_total / loss_denom})
            else:
                print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss}, Recon Loss: {recon_loss_total / loss_denom}, KL Loss: {kl_loss_total / loss_denom}')
            
            #unmasked_x_hat, mask_x_hat = utils.separate_goal_mask(x_hat)

            if avg_loss < best_loss:
                best_loss = avg_loss
                model.save_model(model_root, 'best.pt')
                if self.wandb_enabled:
                    if epoch > 500:
                        plot_img_comparison_batch(unmasked_o_tensor, x_hat, 'Observation', 'Reconstruction', name='best_lo_recon', root=self.img_root, wandb_log=self.wandb_enabled, limit=8)
                        if goal_mask:
                            plot_img_comparison_batch(mask_o_tensor, x_hat_mask, 'Goal Mask', 'Reconstruction Goal Mask', name='eval_lo_recon_goal_mask', root=self.img_root, wandb_log=self.wandb_enabled, limit=8)
                    wandb.log({'best_loss': best_loss})
                    wandb.run.summary['best_loss'] = best_loss

            if (epoch + 1) % self.eval_every_n_epochs == 0:
                if not self.wandb_enabled:
                    print(f'Plotting evaluation images at epoch {epoch+1}')
                plot_img_comparison_batch(unmasked_o_tensor, x_hat, 'Observation', 'Reconstruction', name='eval_lo_recon', root=self.img_root, wandb_log=self.wandb_enabled, limit=8)
                if goal_mask:
                    plot_img_comparison_batch(mask_o_tensor, x_hat_mask, 'Goal Mask', 'Reconstruction Goal Mask', name='eval_lo_recon_goal_mask', root=self.img_root, wandb_log=self.wandb_enabled, limit=8)
                self.eval_transitions(model, env, 8, self.img_root, wandb_log=self.wandb_enabled, goal_mask=goal_mask)
                model.save_model(model_root, 'latest.pt')

        model.save_model(model_root, 'final.pt')
        #self.eval_transitions(model, env, 8, wandb_log=self.wandb_enabled)

    def eval_transitions(self, model, env, num_transitions, path, wandb_log=True, goal_mask=False):
        a_indices, a_onehot, = utils.get_random_action_sequence(model.steps-1, model.action_dim, self.batch_size, device=self.device)
        a_onehot = a_onehot.to(self.device)
        generate_low_transition_predictions(model, env, a_indices, a_onehot, goal_mask=goal_mask, device=self.device, wandb_log=wandb_log, qty=num_transitions, name=f'eval_lo_transitions', root=path)