import torch.nn as nn
import numpy as np
import torch
from csp import CSP,  matrix_assignment_consistency
from solver import CSPSolver
import math
import time

def weights_init(m):
    """
        Initialize the network: convolutional and batchnorm layers are initialized with 
        values coming from a Normal distribution with mean 0 and variance 0.02. Biases are set to 0.

        Args:
            m: layer to initialize.
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    """
        Generator model

    """

    def __init__(self, input_size, csp_shape, dropout_prob= 0):
        """
            Initialize the generator model

            Args:
                input size: dimension of the input vector
                csp_shape: dictionary containing the shape of the considered CSP
                dropout_prob: dropout probability.
        """

        super().__init__()

        self.d = csp_shape['d']
        self.n = csp_shape['n']
        self.sat_matrix_size = csp_shape['n']*csp_shape['d']

        self.conv = nn.Sequential(
            nn.ConvTranspose2d( 1, out_channels=32, kernel_size=7, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.5),
            nn.ConvTranspose2d( 32, out_channels=128, kernel_size=5, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.5),
            nn.ConvTranspose2d( 128, out_channels=128, kernel_size=5, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.5),
            nn.Linear(142, 64),
            
            nn.LeakyReLU(0.2, True),
            nn.Linear(64, 32),
        )

        self.dense = nn.Sequential(

            nn.Linear(581632, 256),
            nn.LeakyReLU(0.2, True),
            nn.Linear(256, self.sat_matrix_size*self.sat_matrix_size + csp_shape['n'] + 1)
        )

        self.relu = nn.LeakyReLU(0.2, True)
        self.sigmoid = nn.Tanh()


    def forward(self, x):
        """
            Forward pass of the model

            Args:
                x: input vector.
        """
        x = self.conv(x)

        # flattening
        x = x.view(x.size()[0], -1)

        x = self.dense(x)

        matrix = torch.narrow(x, 1, 0, self.sat_matrix_size*self.sat_matrix_size)
        matrix = matrix.view(matrix.size()[0], self.sat_matrix_size, -1)
        matrix = self.sigmoid(matrix)


        assignments = torch.narrow(x, 1, self.sat_matrix_size * self.sat_matrix_size, self.n)
        assignments = assignments.view(assignments.size()[0], -1, self.n)
        assignments = self.relu(assignments)


        sat_label = torch.narrow(x, 1, self.sat_matrix_size * self.sat_matrix_size + self.n, 1)
        sat_label = sat_label.view(sat_label.size()[0], 1, -1)
        sat_label = self.sigmoid(sat_label)

        labeled_assignments = torch.cat((assignments.type(torch.float), sat_label.type(torch.float)), dim=-1)

        return matrix, labeled_assignments

class Discriminator(nn.Module):

    """
        Discriminator model
    """


    def __init__(self, input_size, out_size, dropout_prob= 0):
        """
            Initialize the generator model

            Args:
                input size: dimension of the input vector.
                out_size: shape of the output vector.
                dropout_prob: dropout probability.
        """
        super().__init__()

        self.input_size = input_size

        self.model = nn.Sequential(
            nn.Conv1d(1, 16, 5, 2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(16, 32, 3, 2, bias=False),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=True),
        
            nn.Conv1d(32, 64, 3, 2,bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
    
            nn.Conv1d(64, 128, 1, 2, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(128, 1, 1, 2, bias=False),


            nn.Sigmoid()
        )


    def forward(self, x):
        """
            Forward pass of the model

            Args:
                x: input vector.
        """
        x = self.model(x)
        out = torch.squeeze(x, dim=-1)

        return out


def train_batch(gen, discr, batch, csp_size, loss_fn, optimizer):
    """
    Train the Model on a batch

    Args:
        gen: generative model
        discr: discriminative model
        batch: samples batch
        csp_size: dictionary with shape of the considered csp
        loss_fn: loss function
        optimizer: optimization procedure
    
    Returns:
        float indicating the batch loss
    """

    ############################
    # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
    ###########################
    discr_optimizer = optimizer['discr']
    discr_optimizer.zero_grad()

    # flipped labels and smoothing
    real = torch.empty((batch.shape[0],1)).uniform_(0, 0.1)
    fake = torch.empty((batch.shape[0],1)).uniform_(0.9, 1)

    # noisy labels
    noisy = torch.empty((batch.shape[0],1)).uniform_(0.9, 1)
    random = torch.rand(*real.shape)
    real = torch.where(random <= 0.05, noisy, real)

    noisy = torch.empty((batch.shape[0],1)).uniform_(0, 0.1)
    random = torch.rand(*fake.shape)
    fake = torch.where(random <= 0.05, noisy, fake)


    # computing the loss
    output = discr(batch)

    real_loss = loss_fn(output, real)
    D_x = output.mean().item()
    start = time.time()
    real_loss.backward()
    end = time.time()

    rnd_assgn = torch.randn((batch.shape[0], 1, 128, 128))

    start = time.time()
    fake_csp, fake_batch = gen(rnd_assgn)
    end = time.time()

    # let's solve each problem with a solution
    n = int(csp_size['n'])
    d = int(csp_size['d'])
    
    output = discr(fake_batch.detach())
    fake_loss = loss_fn(output, fake)
    start = time.time()
    fake_loss.backward()
    end = time.time()
    D_G_z1 = output.mean().item()

    discr_top = discr.model[0].weight.grad.norm()
    discr_bottom = discr.model[-2].weight.grad.norm()

    discr_loss = (real_loss + fake_loss)/2

    start = time.time()
    discr_optimizer.step()
    end = time.time()

    ############################
    # (2) Update G network: maximize log(D(G(z)))
    ###########################
    gen_optimizer = optimizer['gen']
    gen_optimizer.zero_grad()


    output = discr(fake_batch)
    gen_loss = loss_fn(output, real)
    

    D_G_z2 = output.mean().item()
    gen_loss.backward()

    gen_top = gen.conv[0].weight.grad.norm()
    gen_bottom = gen.dense[-1].weight.grad.norm()

    gen_optimizer.step()

    return gen_loss.item(), discr_loss.item(), D_x, D_G_z1, D_G_z2, discr_top.item(), discr_bottom.item(), gen_top.item(), gen_bottom.item()


from csp import CSP
from dataset import CSPDataset, ToTensor
from torch.utils.data import DataLoader, Dataset
import torch

if __name__ == "__main__":
    
    seed = 17
    torch.manual_seed(seed)

    k=4
    dataset = CSPDataset(size=1000, k=k, n=20, alpha=0.4, r=1.4, p=0.5, transform=ToTensor())
    
    # retrieving data
    n = dataset.csp.n
    m = dataset.csp.m
    n_bad = dataset.csp.n_bad_assgn
    csp_shape = {'k':k, 'n':n, 'd':dataset.csp.d, 'm':m, 'n_bad':n_bad}

    dataloader = DataLoader(dataset, batch_size=6, shuffle=True)

    gen_input_size = 128
    hidden_units = 256
    layers_num = 4

    generator = Generator(gen_input_size, csp_shape)
    discriminator = Discriminator(dataset.csp.n, 2)

    adversarial_loss = torch.nn.BCELoss()

    g_optimizer = torch.optim.Adam(generator.parameters())
    d_optimizer = torch.optim.Adam(discriminator.parameters())
    optimizer = {'gen': g_optimizer, 'discr':d_optimizer}

    for i_batch, sample_batched in enumerate(dataloader):

        # observe 4th batch and stop.
        if i_batch == 2:
            
            gen_loss, discr_loss, D_x, D_G_z1, D_G_z2 = train_batch(generator, discriminator, sample_batched, csp_shape, adversarial_loss, optimizer)

            break

