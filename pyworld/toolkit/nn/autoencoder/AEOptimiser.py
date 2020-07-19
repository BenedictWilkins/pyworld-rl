#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 14:35:59 2019

author: Benedict Wilkins
"""

import torch
import torch.nn.functional as F
import numpy as np
from collections import namedtuple

from ..optimise import TorchOptimiser

class AEOptimiser(TorchOptimiser):
    
    def __init__(self, ae, loss=F.binary_cross_entropy_with_logits, lr=0.0005):
        super(AEOptimiser, self).__init__(ae, base_optimiser=torch.optim.Adam(ae.parameters(), lr=lr))
        self.loss = loss
       
    def step(self, x, y=None):
        if y is None:
            y = x
        loss = self.loss(self.model(x), y.to(x.device))
        return loss

class VAEOptimiser(TorchOptimiser):

    def __init__(self, vae, loss, beta=1., lr=0.0005):
        super(VAEOptimiser, self).__init__(vae, base_optimiser=torch.optim.Adam(vae.parameters(), lr=lr))
        self.beta = beta
        self.__loss = losss

    def step(self, x):
        x, mu_z, logvar_z = self.model(x)
        x_target = x.to(self.model.device)
        kld_loss = self.beta * -0.5 * (1. + logvar_z - mu_z.pow(2) - logvar_z.exp()).sum()#.view(batch_size, -1).mean()
        x_loss = self.__loss(x, x_target)
        loss = x_loss + kld_loss
        return loss

class DAEOptimiser(TorchOptimiser):

    def noise_gaussian(x, p = 0.3):
        return x + np.random.randn(*x.shape)
    
    def noise_pepper(x, p = 0.3):
        return x * (np.random.uniform(size=x.shape) > p)

    def noise_saltpepper(x, p = 0.3):
        i = np.random.uniform(size=x.shape) < p
        x[i] = (np.random.uniform(size=np.sum(i)) > 0.5)
        return x

    def __init__(self, ae, loss, noise_source = noise_gaussian, lr=0.0005):
        super(DAEOptimiser, self).__init__(ae, base_optimiser=torch.optim.Adam(ae.parameters(), lr=lr))
        self.__loss = loss
       
    def step(self, x):
        loss = self.__loss(self.model(x), x.to(self.model.device))
        return loss


'''
class AAE(Optimiser):
    
    def __init__(self, aae, lr=3e-4, lr_disc=3e-6, beta = 0.5, logits=True):
        super(AAE, self).__init__(aae)
        self.optim = torch.optim.Adam(aae.parameters(), lr=lr)
        self.beta = beta
        #self.real_loss = lambda p : - p.log()
        #self.fake_loss = lambda p : - (1.-p).log()
        self.cma = CMA('loss', 'pixel_loss', 'adversarial_loss')
        self.real = torch.as_tensor(np.array([], dtype=np.float))
        self.fake = torch.as_tensor(np.array([], dtype=np.float))
        if logits:
            self.adv_loss_fun = F.binary_cross_entropy_with_logits
        else:
            self.adv_loss_fun = F.binary_cross_entropy
        self.pixel_loss_fun = F.mse_loss
        
    def loss(self, x_target, x, p_real, p_fake):
        x_target = x_target.to(self.model.device)
        if not self.real.shape == p_real.shape:
            self.real = torch.as_tensor(np.ones(p_real.shape), dtype=p_real.dtype, device=self.model.device)
            self.fake = torch.as_tensor(np.zeros(p_fake.shape), dtype=p_fake.dtype, device=self.model.device)
             
        adv_loss = (self.adv_loss_fun(p_real, self.real) + self.adv_loss_fun(p_fake, self.fake))
        
        return self.pixel_loss_fun(x, x_target), adv_loss
    
    #(0.5 * (self.real_loss(p_real) + self.fake_loss(p_fake))).mean()
        
    def step(self, x):
        self.optim.zero_grad()
        pixel_loss, adv_loss = self.loss(x, *self.model(x))
        loss = (self.beta * pixel_loss + (1 - self.beta) * adv_loss)
        self.cma.push(np.array([loss.item(), pixel_loss.item(), adv_loss.item()]))
        loss.backward()
        self.optim.step()

        
    def info(self):
        return {'model':type(self.model).__name__, 
                'adversarial_loss': self.adv_loss_fun.__name__, 
                'pixel_loss':self.pixel_loss_fun.__name__, 
                'optimiser':str(self.optim), 
                'beta':self.beta}

        
class VAEGAN(Optimiser):
    
    def __init__(self, vae, beta=1., lr=3e-4, ema_n = 100):
        super(VAEGAN, self).__init__(vae)
        self.beta = beta
        self.ema = EMA(ema_n) #track losses
        self.optim_encoder = torch.optim.RMSprop(vae.encoder.parameters(), alpha=0.9,eps=1e-8, lr=lr)
        self.optim_decoder = torch.optim.RMSprop(vae.decoder.parameters(), alpha=0.9,eps=1e-8, lr=lr)
        self.optim_disc = torch.optim.RMSprop(vae.disc.parameters(), alpha=0.9,eps=1e-8, lr=lr)
        
        self.train_dis = True
        self.train_dec = True
        
    def loss(self, x_target, x, prior, gan, disl):
        batch_size = x.shape[0]
        
        #the reconstruction error may be useful for debugging however it is not actually used in the loss computation!
        #x_target = x_target.to(self.model.device)
        #_recon_loss = ((x_target - x) ** 2).view(batch_size, -1).sum(1)
        
        #L_prior, we are using a unit gaussian prior?
        mu_z, logvar_z = prior
        _kld_loss = self.beta * -0.5 * (1. + logvar_z - mu_z.pow(2) - logvar_z.exp()).view(batch_size, -1).sum(1)
        
        #L_disl, using a Gaussian observation model for the intermediate layer of disc (difference between real x and fake x)
        p1_inter, p2_inter = disl
        _disl_loss = ((p1_inter - p2_inter) ** 2).view(batch_size, -1).sum(1)
        
        #L_gan
        p1, p2, p3 = gan
        #p1,p2,p3 are the probability of a fake given by disc for x, dec(enc(x)) and dec(N(z|0,I)) respectively
        #print("prob real is real:", p1.mean().item(), "prob fake is real:", p2.mean().item())


        margin = 0.2
        equilibrium = 0.68
        if p1.mean().item() < equilibrium-margin or p2.mean().item() < equilibrium-margin:
            self.train_dis = False
        if p1.mean().item() > equilibrium+margin or p2.mean().item() > equilibrium+margin:
            self.train_dec = False
        if  self.train_dec is False and  self.train_dis is False:
            self.train_dis = True
            self.train_dec = True
           
                
        _gan_loss = - ((p1 + 1e-6).log() + (1 - p2 + 1e-6).log() + (1 - p3 + 1e-6).log()).sum(1)

        return _kld_loss.mean(), _disl_loss.mean(), _gan_loss.mean()
    
    def step(self, x, lambda_dec=1e-6):
        self.model.zero_grad()
        kld,disl,gan = self.loss(x, *self.model(x))
        #logging
        self.ema.push(np.array([kld.item(), disl.item(), gan.item()]))
        
        #update encoder
        loss_encoder = kld + disl
        loss_encoder.backward(retain_graph=True)
        self.optim_encoder.step()
        self.model.zero_grad()
        
        if self.train_dec:
            loss_decoder = lambda_dec * disl - (1-lambda_dec) * gan 
            loss_decoder.backward(retain_graph=True)
            self.optim_decoder.step()
            
        if self.train_dis:
            self.model.disc.zero_grad()
            gan.backward()
            self.optim_disc.step()
'''          
