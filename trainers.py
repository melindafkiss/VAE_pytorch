import numpy as np

import gc

import torch
import torch.nn.functional as F

from torch.autograd import Variable

import gin
import gin.torch
import gin.torch.external_configurables

import time

@gin.configurable
class VAETrainer:
    def __init__(self, model, device, batch_size,
                 optimizer=gin.REQUIRED, train_loader=None,
                 test_loader=None, trail_label_idx=0, rec_cost =
                 'sum'):
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.optimizer = optimizer(self.model.parameters())
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.trail_label_idx = trail_label_idx

        assert rec_cost in {'mean', 'sum'}
        self.rec_cost = rec_cost
        
        #In the local, no monitoring case, generate video and covered area from fixed batch.
        _, (self.trail_batch, self.trail_labels, _) = enumerate(self.train_loader).__next__()
        self.trail_batch = self.trail_batch.to(self.device)
        self.trail_labels = self.trail_labels
        label_type = self.trail_labels.dtype
        self.trail_labels = self.trail_labels.cpu().detach().numpy()
        
        self.all_labels = torch.zeros(torch.Size([len(train_loader.dataset)]), dtype=label_type)
        for _, (_, y, idx) in enumerate(self.train_loader):
            with torch.no_grad():
                self.all_labels[idx] = y if len(self.trail_labels.shape) < 2 else y[:, self.trail_label_idx]
        self.all_labels = self.all_labels.cpu().detach().numpy()

        # If there are multiple labels, use trail_label_idx
        if len(self.trail_labels.shape) == 2:
            self.trail_labels = self.trail_labels[self.trail_label_idx]


    def sample_pz(self, n=100):
        base_dist = torch.distributions.normal.Normal(torch.zeros(self.model.z_dim), torch.ones(self.model.z_dim))
        dist = torch.distributions.independent.Independent(base_dist, 1)

        sample = dist.sample(torch.Size([n]))
        return sample

        
    def kl_loss(self, z_mean, z_log_var):
        size_loss = 0.5 * torch.sum(torch.mul(z_mean, z_mean))
        variance_loss = 0.5 * torch.sum(-1 - z_log_var + torch.exp(z_log_var))
        return size_loss + variance_loss

    
    def decode_batch(self, z):
        with torch.no_grad():
            z = z.to(self.device)
            gen_x = self.model._decode(z)
            return {
                'decode': gen_x
            }

            
    def reconstruct(self, x):
        with torch.no_grad():
            x = x.to(self.device)
            recon_x = self.model(x)
            del x
            return recon_x
        
    def rec_loss_on_test(self, x_test):
        with torch.no_grad():
            x = x_test.to(self.device)
            recon_x = self.model(x)
            rec_loss = F.mse_loss(recon_x,x, reduction=self.rec_cost)
            del x
            return {
                'rec_loss': rec_loss
                }
        
    def reg_loss_on_test(self):
        with torch.no_grad():
            test_kl = torch.zeros(torch.Size([len(self.test_loader.dataset)])).to(self.device).detach()
            for batch_idx, (x, y, idx) in enumerate(self.test_loader):
                x = x.to(self.device)
                mu = self.model._encode(x)[..., 0]
                std = self.model._encode(x)[..., 1]
                test_kl[idx] = self.kl_loss(mu, std)
                del x
            return torch.mean(test_kl)

            
    def loss_on_batch(self, x):
        x = x.to(self.device)
        recon_x = self.model(x)
        mu = self.model._encode(x)[..., 0]
        std = self.model._encode(x)[..., 1]
        
        bce = F.mse_loss(recon_x, x, reduction=self.rec_cost)
        
        reg_loss = self.kl_loss(mu, std)
        loss = bce + reg_loss

        result = {
            'loss': loss,
            'reg_loss': reg_loss,
            'rec_loss': bce,
            'decode': recon_x,
        }
        return result

    def train_on_batch(self, x):

        self.optimizer.zero_grad()
        result = self.loss_on_batch(x)
        result['loss'].backward()
        self.optimizer.step()
        return result
