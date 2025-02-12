import torch
from torch import nn, optim, Tensor
import torch.nn.functional as F
import wandb
import os
from torch.utils.data import DataLoader, Subset, random_split
from src.model.latent_action import LatentActionModel
from src.utils.eval import plot_img_comparison_batch

class LatentActionTrainer():
    def __init__(self,
                 optimizer,
                 batch_size,
                 beta=1.0,
                 early_stopper=None,
                 device='cuda',
                 eval_img_root='.',
                 eval_every_n_epochs=20):
        
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.beta = beta
        self.early_stopper = early_stopper
        self.device = device

        self.img_root = eval_img_root
        self.eval_every_n_epochs = eval_every_n_epochs
        
        self.wandb_enabled = wandb.run is not None
        if not self.wandb_enabled:
            print("Wandb not enabled, printing training logs to console")

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
    
    def decode_states(model, low_model, pred, target, h, h_pred=None, name='state_reconstructions'):
        if h_pred is None:
            print("Using target hidden state for decoding")
            h_pred = h
        x_hat = low_model.decode(pred, h_pred)
        x = low_model.decode(target, h)
        plot_img_comparison_batch(x, x_hat, title1='Target', title2='Reconstruction', name=name, root=model.img_root)

    def compute_loss(self, model, x, x_hat, dist, d=None, d_hat=None):
        kl_loss = model.latent_handler.kl_div_fixed_prior(dist)
        if model.state_type == 'continuous':
            recon_loss = F.mse_loss(x_hat, x, reduction='sum')
        else:
            recon_loss = F.cross_entropy(x_hat, x, reduction='sum')
        
        if d_hat is not None:
            d_recon_loss = F.mse_loss(d_hat, d, reduction='sum')
        else:
            d_recon_loss = 0
        loss = recon_loss + d_recon_loss + self.beta * kl_loss
        return loss, recon_loss, kl_loss, d_recon_loss
    
    def process_batch(self, model, data, example=False):
        initial_z = data['initial_z'].to(self.device)
        initial_dist = data['initial_dist']
        initial_dist = tuple(d.to(self.device) for d in initial_dist)
        final_z = data['final_z'].to(self.device)
        final_dist = data['final_dist']
        final_dist = tuple(d.to(self.device) for d in final_dist)
        final_h = data['final_h'].to(self.device)
        
        z_hat, d_hat, a, dist = model(initial_z, final_z, final_h)
        
        loss, recon_loss, kl_loss, d_recon_loss = self.compute_loss(model, final_z, z_hat, dist, final_h, d_hat)
        
        if example:
            return loss, recon_loss, kl_loss, d_recon_loss, z_hat, d_hat, a, dist, final_z, final_dist, final_h
        else:
            return loss, recon_loss, kl_loss, d_recon_loss

    def train(self, model, dataset, num_epochs, model_root, train_ratio=0.8, eval_size=32, low_model=None):
        torch.manual_seed(42)

        train_dataset, val_dataset, example_dataset = self.split_dataset(dataset, train_ratio, eval_size)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        example_loader = DataLoader(example_dataset, batch_size=eval_size, shuffle=False)
        
        min_val_loss = float('inf')

        for epoch in range(num_epochs):
            model.train()

            epoch_loss = 0
            recon_loss_total = 0
            kl_loss_total = 0
            d_recon_loss_total = 0
            loss_denom = 0

            for i, data in enumerate(train_loader):
                self.optimizer.zero_grad()

                loss, recon_loss, kl_loss, d_recon_loss = self.process_batch(model, data)
                
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                recon_loss_total += recon_loss.item()
                kl_loss_total += kl_loss.item()
                d_recon_loss_total += d_recon_loss
                loss_denom += 1


            if self.wandb_enabled:
                wandb.log({ "epoch": epoch+1,
                            "train_loss": epoch_loss / loss_denom,
                            "train_recon_loss": recon_loss_total / loss_denom,
                            "train_kl_loss": kl_loss_total / loss_denom})
                if d_recon_loss > 0:
                    wandb.log({"train_d_recon_loss": d_recon_loss_total / loss_denom})
            
            epoch_loss = 0
            recon_loss_total = 0
            kl_loss_total = 0
            d_recon_loss_total = 0
            loss_denom = 0

            model.eval()
            for i, data in enumerate(val_loader):
                with torch.no_grad():
                    loss, recon_loss, kl_loss, d_recon_loss = self.process_batch(model, data)
                    
                    epoch_loss += loss.item()
                    recon_loss_total += recon_loss.item()
                    kl_loss_total += kl_loss.item()
                    d_recon_loss_total += d_recon_loss
                    loss_denom += 1

            avg_loss = epoch_loss / loss_denom
            if self.wandb_enabled:
                wandb.log({ "epoch": epoch+1,
                            "val_loss":avg_loss,
                            "val_recon_loss": recon_loss_total / loss_denom,
                            "val_kl_loss": kl_loss_total / loss_denom})
                if d_recon_loss > 0:
                    wandb.log({"val_d_recon_loss": d_recon_loss_total / loss_denom})

            if avg_loss < min_val_loss:
                min_val_loss = avg_loss
                if self.wandb_enabled:
                    wandb.log({"min_val_loss": min_val_loss})
                    model.save_model(model_root, 'best.pt')

            if (epoch + 1) % self.eval_every_n_epochs == 0:
                if low_model is not None:
                    low_model.eval()
                    with torch.no_grad():
                        for i, data in enumerate(example_loader):
                            loss, recon_loss, kl_loss, d_recon_loss, z_hat, d_hat, a, dist, final_z, final_dist, final_h = self.process_batch(model, data, example=True)
                            self.decode_states(low_model, z_hat, final_z, final_h, d_hat, name='state_reconstructions')

                if self.early_stopper is not None:
                    if self.early_stopper.step(loss.item()):
                        break
                
                model.save_model(model_root, 'latest.pt')
        model.save_model(model_root, 'final.pt')