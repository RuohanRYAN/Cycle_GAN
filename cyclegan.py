import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import transforms
from dataset import ImageDataset
import os
import torch.nn as nn
from utils import *
import torch
from network import Generator, Discriminator, get_cycle_consistency_loss, get_disc_loss, get_gen_adversarial_loss, get_gen_loss, get_identity_loss
from torch.utils.data import Dataset, DataLoader

def weights_init(m): # a function to initialise the weights of our model
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)


def train(save_model=False):
    mean_generator_loss = 0
    mean_discriminator_loss = 0
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    cur_step = 0

    scheduler_discP = torch.optim.lr_scheduler.ReduceLROnPlateau(disc_P_opt, factor=0.1, patience=5,
                                                                 verbose=True)
    scheduler_discM = torch.optim.lr_scheduler.ReduceLROnPlateau(disc_M_opt, factor=0.1, patience=5,
                                                                 verbose=True)
    scheduler_gen = torch.optim.lr_scheduler.ReduceLROnPlateau(gen_opt, factor=0.1, patience=5,
                                                               verbose=True)

    for epoch in range(n_epochs):
        losses_P = []
        losses_M = []
        losses_gen = []
        for real_P, real_M in tqdm(dataloader):
            real_P = nn.functional.interpolate(real_P, size=target_shape)
            real_M = nn.functional.interpolate(real_M, size=target_shape)
            cur_batch_size = len(real_P)
            real_P = real_P
            real_M = real_M

            ### Update discriminator A ###
            disc_P_opt.zero_grad()  # Zero out the gradient before backpropagation
            with torch.no_grad():
                fake_P = gen_MP(real_M)
            disc_P_loss = get_disc_loss(real_P, fake_P, disc_P, adv_criterion)
            disc_P_loss.backward(retain_graph=True)  # Update gradients
            disc_P_opt.step()  # Update optimizer
            losses_P.append(disc_P_loss.item())

            ### Update discriminator B ###
            disc_M_opt.zero_grad()  # Zero out the gradient before backpropagation
            with torch.no_grad():
                fake_M = gen_PM(real_P)
            disc_M_loss = get_disc_loss(real_M, fake_M, disc_M, adv_criterion)
            disc_M_loss.backward(retain_graph=True)  # Update gradients
            disc_M_opt.step()  # Update optimizer
            losses_M.append(disc_M_loss.item())

            ### Update generator ###
            gen_opt.zero_grad()
            gen_loss, fake_P, fake_M = get_gen_loss(
                real_P, real_M, gen_PM, gen_MP, disc_P, disc_M, adv_criterion, recon_criterion, recon_criterion
            )
            gen_loss.backward()  # Update gradients
            gen_opt.step()  # Update optimizer
            losses_gen.append(gen_loss.item())
        mean_loss_disc_M = sum(losses_M) / len(losses_M)
        mean_loss_disc_P = sum(losses_P) / len(losses_P)
        mean_loss_gen = sum(losses_gen) / len(losses_gen)
        scheduler_discP.step(mean_loss_disc_P)
        scheduler_discM.step(mean_loss_disc_M)
        scheduler_gen.step(mean_loss_gen)

BASE_PATH = ".\\gan-getting-started\\"
MONET_PATH, PHOTO_PATH = prepare_Paths(BASE_PATH)
MONET_FILENAMES, PHOTO_FILENAMES = Prepare_list_names(MONET_PATH, PHOTO_PATH)
target_shape = 256
load_shape = 286
transform = transforms.Compose([
    transforms.Resize(load_shape),
    transforms.RandomCrop(target_shape),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
dataset = ImageDataset(MONET_FILENAMES, PHOTO_FILENAMES, transform=transform)

adv_criterion = nn.MSELoss()
recon_criterion = nn.L1Loss()

n_epochs = 1
dim_PHOTO = 3
dim_MONET = 3
display_step = 200
batch_size = 1
lr = 0.0002

gen_PM = Generator(dim_PHOTO, dim_MONET)
gen_MP = Generator(dim_MONET, dim_PHOTO)
gen_opt = torch.optim.Adam(list(gen_PM.parameters()) + list(gen_MP.parameters()), lr=lr, betas=(0.5, 0.999))
disc_P = Discriminator(dim_PHOTO)
disc_P_opt = torch.optim.Adam(disc_P.parameters(), lr=lr, betas=(0.5, 0.999))
disc_M = Discriminator(dim_MONET)
disc_M_opt = torch.optim.Adam(disc_M.parameters(), lr=lr, betas=(0.5, 0.999))
gen_PM = gen_PM.apply(weights_init)
gen_MP = gen_MP.apply(weights_init)
disc_P = disc_P.apply(weights_init)
disc_M = disc_M.apply(weights_init)

train()



