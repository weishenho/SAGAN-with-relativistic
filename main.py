import argparse
import os
import numpy as np

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

from model import GeneratorResnet, DiscriminatorResnet, Discriminator

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')

parser.add_argument('--resnet_generator', type=bool, default=True, help='resnet generator or DC generator')
parser.add_argument('--resnet_discriminator', type=bool, default=True, help='resnet discriminator or DC discriminator')

parser.add_argument('--g_lr', type=float, default=0.0001)
parser.add_argument('--d_lr', type=float, default=0.0004)
parser.add_argument('--lr_decay', type=float, default=0.95)

parser.add_argument('--b1', type=float, default=0.0, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.9, help='adam: decay of first order momentum of gradient')
parser.add_argument('--n_cpu', type=int, default=2, help='number of cpu threads to use during batch generation')
parser.add_argument('--latent_dim', type=int, default=128, help='dimensionality of the latent space')
parser.add_argument('--img_size', type=int, default=64, help='size of each image dimension [512, 256, 128, 64, 32]')
parser.add_argument('--channels', type=int, default=3, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=400, help='interval between image sampling')
parser.add_argument('--display_loss_interval', type=int, default=10, help='display_loss_interval')

parser.add_argument('--input_folder', default='data/anime', help='input folder')
parser.add_argument('--checkpoint_interval', type=int, default=1, help='interval between model checkpoints')
parser.add_argument('--dataset_name', type=str, default="anime", help='name of the dataset')
parser.add_argument('--epoch', type=int, default=0, help='epoch to start training from')

opt = parser.parse_args()
print(opt)

os.makedirs('images/%s' % opt.dataset_name, exist_ok=True)
os.makedirs('saved_models/%s' % opt.dataset_name, exist_ok=True)

cuda = True if torch.cuda.is_available() else False


def train():
    # Loss function
    adversarial_loss = torch.nn.BCELoss()

    # Initialize generator and discriminator
    if opt.resnet_generator:
        generator = GeneratorResnet(opt)

    if opt.resnet_discriminator:
        discriminator = DiscriminatorResnet(opt)
    else:
        discriminator = Discriminator(opt)

    print(generator)
    print()
    print(discriminator)

    if cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()

    if opt.epoch != 0:
        # Load pretrained models
        generator.load_state_dict(torch.load('saved_models/%s/generator_%d.pth' % (opt.dataset_name, opt.epoch)))
        discriminator.load_state_dict(torch.load('saved_models/%s/discriminator_%d.pth' % (opt.dataset_name, opt.epoch)))

    trans = transforms.Compose([
            transforms.Resize(size=(opt.img_size, opt.img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
    dataset = datasets.ImageFolder(root=opt.input_folder, transform=trans)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.g_lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.d_lr, betas=(opt.b1, opt.b2))

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    fixed_z = torch.randn(opt.batch_size, opt.latent_dim).type(Tensor)

    # ----------
    #  Training
    # ----------

    def reset_grad():
        optimizer_D.zero_grad()
        optimizer_G.zero_grad()


    for epoch in range(opt.epoch, opt.n_epochs):
        for i, (imgs, _) in enumerate(dataloader):
            
            bs = imgs.shape[0]
            # Adversarial ground truths
            y = Variable(Tensor(bs).fill_(1.0), requires_grad=False)

            # Configure input
            real_imgs = Variable(imgs.type(Tensor))

            # ---------------------
            #  Train Discriminator
            # ---------------------
            #Real
            d_out_real = discriminator(real_imgs)

            #Fake
            z = torch.randn(bs, opt.latent_dim).type(Tensor)
            fake_images = generator(z)
            d_out_fake = discriminator(fake_images)

            d_loss = (torch.mean((d_out_real - torch.mean(d_out_fake) - y) ** 2) + torch.mean((d_out_fake - torch.mean(d_out_real) + y) ** 2))/2
            reset_grad()
            d_loss.backward()
            optimizer_D.step()

            # -----------------
            #  Train Generator
            # -----------------

            #Real
            d_out_real = discriminator(real_imgs)

            #Fake
            z = torch.randn(bs, opt.latent_dim).type(Tensor)
            fake_images = generator(z)
            g_out_fake = discriminator(fake_images)


            g_loss_fake = (torch.mean((d_out_real - torch.mean(g_out_fake) + y) ** 2) + torch.mean((g_out_fake - torch.mean(d_out_real) - y) ** 2))/2

            reset_grad()
            g_loss_fake.backward()
            optimizer_G.step()


            batches_done = epoch * len(dataloader) + i

            if batches_done % opt.display_loss_interval == 0:
                print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, opt.n_epochs, i, len(dataloader),
                                                                    d_loss.item(), g_loss_fake.item()))
            if batches_done % opt.sample_interval == 0:
                fake_images = generator(fixed_z)
                save_image(fake_images, 'images/%s/%d.png' % (opt.dataset_name, batches_done), nrow=8, normalize=True)
                
        if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
            # Save model checkpoints
            torch.save(generator.state_dict(), 'saved_models/%s/generator_%d.pth' % (opt.dataset_name, epoch))
            torch.save(discriminator.state_dict(), 'saved_models/%s/discriminator_%d.pth' % (opt.dataset_name, epoch))


if __name__ == '__main__':
    train()
