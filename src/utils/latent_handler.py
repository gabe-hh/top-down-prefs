import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from src.utils.loss import kld_gaussian, kld_categorical

class BaseLatentHandler:
    def reshape_latent(self, latent, dist_params):
        raise NotImplementedError()

    def reparameterize(self, params):
        raise NotImplementedError()
    
    def sample_if_logits(self, input):
        raise NotImplementedError()

    def kl_divergence(self, posterior, prior):
        raise NotImplementedError()
    
    def balanced_kl_divergence(self, posterior, prior, alpha=0.8):
        raise NotImplementedError()
    
    def kl_div_fixed_prior(self, posterior):
        raise NotImplementedError()
    
    def sample_reconstruction_loss(self, x, x_hat):
        raise NotImplementedError()
    
    def zero_latent(self, batch_size, params):
        raise NotImplementedError()
    
    def zero_prior(self, batch_size, params):
        raise NotImplementedError()
    
# A Gaussian latent handler
class GaussianLatentHandler(BaseLatentHandler):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim

    def sample_if_logits(self, input):
        return input

    def reshape_latent(self, latent, dist_params):
        return latent

    def reparameterize(self, params):
        mu, logvar = params
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def kl_divergence(self, posterior, prior):
        mu1, logvar1 = posterior
        mu2, logvar2 = prior
        return kld_gaussian(mu1, logvar1, mu2, logvar2)
    
    def balanced_kl_divergence(self, posterior, prior, alpha=0.8):
        mu1, logvar1 = posterior
        mu2, logvar2 = prior
        gradless_mu1 = mu1.clone().detach()
        gradless_mu2 = mu2.clone().detach()
        gradless_logvar1 = logvar1.clone().detach()
        gradless_logvar2 = logvar2.clone().detach()
        kld = alpha * kld_gaussian(gradless_mu1, gradless_logvar1, mu2, logvar2) + (1 - alpha) * kld_gaussian(mu1, logvar1, gradless_mu2, gradless_logvar2)
        return kld
    
    def kl_div_fixed_prior(self, posterior):
        mu, logvar = posterior
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    def sample_reconstruction_loss(self, x, x_hat):
        return F.mse_loss(x_hat, x, reduction='sum')

    def zero_latent(self, batch_size, latent_size, device='cpu'):
        if isinstance(latent_size, tuple):
            raise ValueError("latent_size must be an integer for Gaussian latent space")
        return torch.zeros(batch_size, latent_size, requires_grad=False).to(device)
    
    def zero_prior(self, batch_size, latent_size, device='cpu'):
        if isinstance(latent_size, tuple):
            raise ValueError("latent_size must be an integer for Gaussian latent space")
        return (torch.zeros(batch_size, latent_size, requires_grad=False).to(device), 
                torch.ones(batch_size, latent_size, requires_grad=False).to(device))
    
class CategoricalLatentHandler(BaseLatentHandler):
    def __init__(self, temperature=1.0, straight_through=True):
        super().__init__()
        self.temperature = temperature
        self.straight_through = straight_through
        
    def reshape_latent(self, latent, dist_params):
        latent_dim, num_classes = dist_params
        return latent.view(latent.size(0), latent_dim, num_classes)
    
    def sample_if_logits(self, input):
        return self.reparameterize((input, None))

    def reparameterize(self, params):
        logits,_ = params
        if self.straight_through:
            p = D.OneHotCategoricalStraightThrough(logits=logits)
        else:
            p = D.RelaxedOneHotCategorical(self.temperature, logits=logits)
        return p.rsample()
    
    def kl_divergence(self, posterior, prior):
        logits, _ = posterior
        prior_logits, _ = prior
        return kld_categorical(logits, prior_logits)
    
    def balanced_kl_divergence(self, posterior, prior, alpha=0.8):
        logits, _ = posterior
        prior_logits, _ = prior
        gradless_logits = logits.clone().detach()
        gradless_prior_logits = prior_logits.clone().detach()
        kld = alpha * kld_categorical(gradless_logits, prior_logits) + (1 - alpha) * kld_categorical(logits, gradless_prior_logits)
        return kld
    
    def kl_div_fixed_prior(self, posterior):
        logits, _ = posterior
        prior_logits = torch.zeros_like(logits)
        return kld_categorical(logits, prior_logits)
    
    def kl_reconstruction_loss(self, x, x_hat):
        return kld_categorical(x_hat, x)
    
    def sample_reconstruction_loss(self, x, x_hat):
        return F.cross_entropy(x_hat.permute(0,2,1), torch.argmax(x,dim=2), reduction='sum')

    def zero_latent(self, batch_size, latent_size, device='cpu'):
        if not isinstance(latent_size, tuple) or len(latent_size) != 2:
            raise ValueError("latent_size must be a tuple of (latent_dim, num_classes) for Categorical latent space")
        
        latent_dim, num_classes = latent_size
        return torch.zeros(batch_size, latent_dim, num_classes, requires_grad=False).to(device)
    
    def zero_prior(self, batch_size, latent_size, device='cpu'):
        if not isinstance(latent_size, tuple) or len(latent_size) != 2:
            raise ValueError("latent_size must be a tuple of (latent_dim, num_classes) for Categorical latent space")
        
        latent_dim, num_classes = latent_size
        logits = torch.zeros(batch_size, latent_dim, num_classes, requires_grad=False).to(device)
        p = F.softmax(logits, dim=-1)
        return logits, p
