import numpy as np

import gc

import torch
import torch.nn.functional as F

from torch.autograd import Variable

import gin
import gin.torch
import gin.torch.external_configurables

import time

import math

@gin.configurable
class VAETrainer:
    def __init__(self, model, funcmodel, learn_func, baseline, device, batch_size,
                 optimizer=gin.REQUIRED, train_loader=None,
                 test_loader=None, trail_label_idx=0, rec_cost =
                 'sum'):
        self.model = model
        self.funcmodel = funcmodel
        self.learn_func = learn_func
        self.baseline = baseline
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
        size_loss = torch.mean(0.5 * torch.sum(torch.mul(z_mean, z_mean), axis = 1))
        variance_loss = torch.mean(0.5 * torch.sum(-1 - z_log_var + torch.exp(z_log_var), axis = 1))
        return size_loss + variance_loss


    # Assuming that the variance of the decoder is 1
    def fdiv_loss(self, x, N = 10):
        inp = torch.zeros(N)
        latent_samples = self.sample_pz(N)

        if self.learn_func:
            with torch.no_grad():
                x = x.to(self.device)
                enc_mean = self.model._encode(x)[..., 0]
                enc_sigma = self.model._encode(x)[..., 1]
        else:
            x = x.to(self.device)
            enc_mean = self.model._encode(x)[..., 0]
            enc_sigma = self.model._encode(x)[..., 1]
        
        base_dist_1 = torch.distributions.normal.Normal(torch.zeros(self.model.z_dim), torch.ones(self.model.z_dim))
        dist_1 = torch.distributions.independent.Independent(base_dist, 1)
        base_dist_2 = torch.distributions.normal.Normal(enc_mean, enc_sigma)
        dist_2 = torch.distributions.independent.Independent(base_dist, 1)
        
        for i in range(N):
            z = latent_samples[i]
            if self.learn_func:
                with torch.no_grad():
                    dec_mean = self.model._decode(z)
            else:
                dec_mean = self.model._decode(z)
            
            base_dist_3 = torch.distributions.normal.Normal(dec_mean, torch.ones(np.prod(self.model.input_dims)))
            dist_3 = torch.distributions.independent.Independent(base_dist, 1)

            v = torch.exp(dist_1.log_prob(z)) * torch.exp(dist_3.log_prob(x)) / torch.exp(dist_2.log_prob(z))
            
            if self.learn_func:
                inp[i] = self.funcmodel(v)
            else:
                with torch.no_grad():
                    inp[i] = self.funcmodel(v)
        
        value = torch.mean(inp)
        value = self.funcmodel.f_inverse(value)
        value = torch.log(value)

        return value

    def f_log_loss(self, M = 100):
        dist = 0
        for i in range(M):
            x = 0.5 + i / M
            dist += ((self.func_model(x) - math.log(x)) ** 2) / M
        return dist
        

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
        if self.baseline:
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

        else:
            loss = self.fdiv_loss(x, 10) + self.f_log_loss(100)
            result = {
                'loss': loss,
                'decode': recon_x,
            }

            
        return result

    def train_on_batch(self, x):

        self.optimizer.zero_grad()
        result = self.loss_on_batch(x)
        result['loss'].backward()
        self.optimizer.step()
        return result
