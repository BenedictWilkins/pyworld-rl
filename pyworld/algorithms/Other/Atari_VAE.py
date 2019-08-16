#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 13:31:55 2019

@author: ben
"""

#!/usr/bin/env python

#CREDIT: Deep-Reinforcement-Learning-Hands-On PacktPublishing

import random
import argparse
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

import torchvision.transforms as transforms
import torchvision.utils as vutils


import gym
import gym.spaces

import numpy as np

from PIL import Image

log = gym.logger
log.set_level(gym.logger.INFO)

LATENT_VECTOR_SIZE = 40
FILTERS = 64
BATCH_SIZE = 16

# dimension input image will be rescaled
IMAGE_SIZE = 64

LEARNING_RATE = 0.0001
REPORT_EVERY_ITER = 100
SAVE_IMAGE_EVERY_ITER = 100


class InputWrapper(gym.ObservationWrapper):
    """
    Preprocessing of input numpy array:
    1. resize image into predefined size
    2. move color channel axis to a first place
    """
    def __init__(self, *args):
        super(InputWrapper, self).__init__(*args)
        assert isinstance(self.observation_space, gym.spaces.Box)
        old_space = self.observation_space
        self.observation_space = gym.spaces.Box(self.observation(old_space.low), self.observation(old_space.high),
                                                dtype=np.float32)

    def observation(self, observation):
        # resize image
        new_obs = cv2.resize(observation, (IMAGE_SIZE, IMAGE_SIZE))
        # transform (210, 160, 3) -> (3, 210, 160)
        new_obs = np.moveaxis(new_obs, 2, 0)
        return new_obs.astype(np.float32)


class VAE(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(VAE, self).__init__()
        # this pipe converges image into the single number
        self.conv_pipe = nn.Sequential(
            nn.Conv2d(in_channels=input_shape[0], out_channels=FILTERS,
                      kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=FILTERS, out_channels=FILTERS*2,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(FILTERS*2),
            nn.ReLU(),
            nn.Conv2d(in_channels=FILTERS * 2, out_channels=FILTERS * 4,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(FILTERS * 4),
            nn.ReLU(),
            nn.Conv2d(in_channels=FILTERS * 4, out_channels=FILTERS * 8,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(FILTERS * 8),
            nn.ReLU(),
            nn.Conv2d(in_channels=FILTERS * 8, out_channels=LATENT_VECTOR_SIZE,
                      kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
        )
        self.mu_layer = nn.Linear(LATENT_VECTOR_SIZE, LATENT_VECTOR_SIZE)
        self.sig_layer = nn.Linear(LATENT_VECTOR_SIZE, LATENT_VECTOR_SIZE)
        self.deconv_pipe = nn.Sequential(
            nn.ConvTranspose2d(in_channels=LATENT_VECTOR_SIZE, out_channels=FILTERS * 8,
                               kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(FILTERS * 8),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=FILTERS * 8, out_channels=FILTERS * 4,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(FILTERS * 4),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=FILTERS * 4, out_channels=FILTERS * 2,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(FILTERS * 2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=FILTERS * 2, out_channels=FILTERS,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(FILTERS),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=FILTERS, out_channels=output_shape[0],
                               kernel_size=4, stride=2, padding=1),
            nn.Tanh())
        
            
    def encoder(self, x):
        en = self.conv_pipe(x).view(-1, LATENT_VECTOR_SIZE)
        return self.mu_layer(en), self.sig_layer(en)
        
    
    def decoder(self, z):
        z = z.view(-1, LATENT_VECTOR_SIZE, 1, 1)
        return self.deconv_pipe(z) #deconv pipe expects 4d input
        
    def reparam(self, mu, logsig):
        std = 0.5 * torch.exp(logsig) #to avoid numerical problems the network learns log variance
        gaus = torch.randn(LATENT_VECTOR_SIZE) #normal distribution with mean 0 variance 1
        return mu + gaus * std

    def forward(self, x):
        mu, logsig = self.encoder(x)
        z = self.reparam(mu, logsig)
        y = self.decoder(z)
        return y, mu, logsig
    
    def gaussian_reg_loss(self, y, x, mu, logsig):
        xyl = nn.MSELoss()
        # solution to kl divergence with a standard gaussian prior |integral N(z;mu,sig) log N(z;0,I)
        kld = -0.5 * torch.sum(1 + logsig - mu.pow(2) - logsig.exp()) 
        return xyl(y, x) + kld
    
writer = SummaryWriter()

def iterate_batches(envs, batch_size=BATCH_SIZE):
    batch = [e.reset() for e in envs]
    env_gen = iter(lambda: random.choice(envs), None)

    while True:
        e = next(env_gen)
        obs, reward, is_done, _ = e.step(e.action_space.sample())
        if np.mean(obs) > 0.01:
            batch.append(obs)
        if len(batch) == batch_size:
            # Normalising input between -1 to 1
            batch_np = np.array(batch, dtype=np.float32) * 2.0 / 255.0 - 1.0
            yield torch.tensor(batch_np)
            batch.clear()
        if is_done:
            e.reset()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action='store_true', help="Enable cuda computation")
    args = parser.parse_args()

    device = torch.device("cuda" if args.cuda else "cpu")
    envs = [InputWrapper(gym.make(name)) for name in ['FishingDerby-v0', 'MsPacman-v0']]
    input_shape = envs[0].observation_space.shape

    vae = VAE(input_shape=input_shape,output_shape=input_shape).to(device)

    objective = vae.gaussian_reg_loss
    optimizer = optim.Adam(params=vae.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    
    writer = SummaryWriter()

    losses = []
    iter_no = 0
    
    batch_v = next(iterate_batches(envs))
    writer.add_graph(vae, batch_v)

    for batch_v in iterate_batches(envs):
        
        batch_v = batch_v.to(device)
        
        # train
        optimizer.zero_grad()
        output_v, mu, logsig = vae(batch_v)
        loss = objective(output_v, batch_v, mu, logsig)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        if iter_no % REPORT_EVERY_ITER == 0:
            log.info("Iter %d: loss=%.3e", iter_no, np.mean(losses))
            writer.add_scalar("loss", np.mean(losses), iter_no)
            losses = []
        if iter_no % SAVE_IMAGE_EVERY_ITER == 0:  

            gen_img  = vutils.make_grid(output_v.data[:64], normalize=True)
            real_img = vutils.make_grid(batch_v.data[:64], normalize=True)
            gen_img = transforms.ToTensor()(transforms.Resize(256 + 128, Image.NEAREST)(transforms.ToPILImage()(gen_img)))
            real_img = transforms.ToTensor()(transforms.Resize(256 + 128, Image.NEAREST)(transforms.ToPILImage()(real_img)))
            
            writer.add_image("fake", gen_img, iter_no)
            writer.add_image("real", real_img, iter_no)
        
        iter_no += 1
        
        

    