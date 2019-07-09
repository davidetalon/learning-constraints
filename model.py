import torch.nn as nn
import numpy as np
import torch
from CSP import CSP

class Generator(nn.Module):

    def block(input_size, out_size):
            layers =[nn.Linear(input_size, out_size)]
            layers.append(nn.ReLU())
            return layers

    def __init__(self, input_size, hidden_units, layers_num, csp_shape, dropout_prob= 0):

        super().__init__()

        # def block(input_size, out_size):
        #     layers =[nn.Linear(input_size, out_size)]
        #     layers.append(nn.ReLU())
        #     return layers

        self.d = csp_shape['d']
        self.sat_matrix_size = csp_shape['n']*csp_shape['d']

        self.conv = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(1, out_channels=16, kernel_size=3, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(16, 16, kernel_size=3, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        self.dense = nn.Sequential(
            nn.Linear(4096576,self.sat_matrix_size),
            nn.ReLU(),
            nn.Linear(self.sat_matrix_size, (self.sat_matrix_size**2)*self.d),
            nn.Sigmoid()
        )


    def forward(self, x):
        x = self.conv(x)

        # flattening
        x = x.view(x.size()[0], -1)

        x = self.dense(x)
        x = x.view(x.shape[0], self.sat_matrix_size, -1)

        out = torch.where(x < 0.5, torch.zeros(x.shape), torch.ones(x.shape))
        
        return out

class Discriminator(nn.Module):


    def __init__(self, input_size, hidden_units, layers_num, out_size, dropout_prob= 0):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv1d(1, out_channels=32, kernel_size=7, stride=1),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(kernel_size=3,stride=1),
            nn.ReLU(),
            nn.Conv1d(32, out_channels=16, kernel_size=3, stride=1),
            nn.BatchNorm1d(16),
            nn.MaxPool1d(kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(16, out_channels=8, kernel_size=3, stride=1),
            nn.BatchNorm1d(8),
            nn.MaxPool1d(kernel_size=3),
            nn.ReLU(),
        )

        self.dense = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid() 
        )

    def forward(self, x):
        x = self.model(x)

        # flattening
        x = x.view(x.shape[0], -1)

        out = self.dense(x)
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


    # labels
    real = torch.ones(batch.shape[0], 1)
    fake = torch.zeros(batch.shape[0], 1)

    # generate the CSP
    mean = torch.zeros(batch.shape[0], 1, 256, 256)
    variance = torch.ones(batch.shape[0], 1, 256, 256)
    rnd_assgn = torch.normal(mean, variance)

    # -----------------
    #  Train Generator
    # -----------------

    gen_optimizer = optimizer['gen']
    gen_optimizer.zero_grad()

    # generate the fake csp
    fake_csp = gen(rnd_assgn)

    # get random assignment and check for consistency
    assignments = torch.randint(0, csp_size['d'], (batch.shape[0], csp_size['n']))

    # check consistency
    consistency = CSP.matrix_assignment_consistency(assignments, fake_csp, csp_size['d'], csp_size['n'])

    # label each assignment with its satisfiability
    fake_batch = torch.cat((assignments.type(torch.float), consistency.type(torch.float)),1)

    print('Real batch sat: ', batch[:, 0, 20])
    print('Fake batch sat: ', fake_batch[:, 20])

    # computing the loss
    fake_batch.unsqueeze_(1)

    gen_loss = loss_fn(discr(fake_batch), real)

    # optimizing
    gen_loss.backward()
    gen_optimizer.step()

    # ---------------------
    #  Train Discriminator
    # ---------------------

    discr_optimizer = optimizer['discr']
    discr_optimizer.zero_grad()

    # computing the loss
    real_loss = loss_fn(discr(batch), real)
    fake_loss = loss_fn(discr(fake_batch), fake)
    discr_loss = (real_loss + fake_loss) /2

    # optimizing
    discr_loss.backward()
    discr_optimizer.step()

    return gen_loss, discr_loss


from CSP import CSP
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

    generator = Generator(gen_input_size, hidden_units, layers_num,  csp_shape)
    discriminator = Discriminator(dataset.csp.n, hidden_units, layers_num, 2)

    adversarial_loss = torch.nn.BCELoss()

    g_optimizer = torch.optim.Adam(generator.parameters())
    d_optimizer = torch.optim.Adam(discriminator.parameters())
    optimizer = {'gen': g_optimizer, 'discr':d_optimizer}

    for i_batch, sample_batched in enumerate(dataloader):

        # observe 4th batch and stop.
        if i_batch == 2:
            
            train_batch(generator, discriminator, sample_batched, csp_shape, adversarial_loss, optimizer)

            break

