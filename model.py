import torch.nn as nn
import numpy as np
import torch
from csp import CSP,  matrix_assignment_consistency
from solver import CSPSolver


class Generator(nn.Module):

    # def block(input_size, out_size):
    #         layers =[nn.Linear(input_size, out_size)]
    #         layers.append(nn.ReLU())
    #         return layers

    def __init__(self, input_size, hidden_units, layers_num, csp_shape, dropout_prob= 0):

        super().__init__()

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
            nn.Linear(1000000,self.sat_matrix_size),
            nn.ReLU(),
            nn.Linear(self.sat_matrix_size, self.sat_matrix_size*self.sat_matrix_size),
            nn.Sigmoid()
        )


    def forward(self, x):
        x = self.conv(x)

        # flattening
        x = x.view(x.size()[0], -1)

        x = self.dense(x)

        x = x.view(x.size()[0], self.sat_matrix_size, -1)

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

    ############################
    # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
    ###########################
    discr_optimizer = optimizer['discr']
    discr_optimizer.zero_grad()

    # labels
    real = torch.ones(batch.shape[0], 1)
    fake = torch.zeros(batch.shape[0], 1)

    # computing the loss
    output = discr(batch)
    real_loss = loss_fn(output, real)

    D_x = output.mean().item()
    
    real_loss.backward()

    # generate the CSP
    mean = torch.zeros(batch.shape[0], 1, 128, 128)
    variance = torch.ones(batch.shape[0], 1, 128, 128)
    rnd_assgn = torch.normal(mean, variance)

    fake_csp = gen(rnd_assgn)

    # let's solve each problem with a solution
    n = int(csp_size['n'])
    d = int(csp_size['d'])

    assignment = np.empty((fake_csp.shape[0], n+1))
    for csp in range(fake_csp.shape[0]):
        solver = CSPSolver(n, d, fake_csp[csp, :, :], limit=1)
        if solver.solution_count()>=1:
            assgn = solver.get_satisfying_assignments(n=1)
            assgn = np.append(assgn, 1)

        else:
            assgn = np.random.randint(0, d-1, n)
            assgn = np.append(assgn, 0)
        assignment[csp, :] = assgn[:]

    fake_batch = torch.from_numpy(assignment)
    fake_batch.unsqueeze_(1)
    fake_batch = fake_batch.type(torch.float)
    
    output = discr(fake_batch.detach())
    fake_loss = loss_fn(output, fake)

    fake_loss.backward()
    D_G_z1 = output.mean().item()

    discr_loss = real_loss + fake_loss

    discr_optimizer.step()

    ############################
    # (2) Update G network: maximize log(D(G(z)))
    ###########################
    gen_optimizer = optimizer['gen']
    gen_optimizer.zero_grad()

    output = discr(fake_batch)
    gen_loss = loss_fn(output, real)

    D_G_z2 = output.mean().item()
    gen_loss.backward()

    gen_optimizer.step()

    return gen_loss, discr_loss, D_x, D_G_z1, D_G_z2


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

    generator = Generator(gen_input_size, hidden_units, layers_num,  csp_shape)
    discriminator = Discriminator(dataset.csp.n, hidden_units, layers_num, 2)

    adversarial_loss = torch.nn.BCELoss()

    g_optimizer = torch.optim.Adam(generator.parameters())
    d_optimizer = torch.optim.Adam(discriminator.parameters())
    optimizer = {'gen': g_optimizer, 'discr':d_optimizer}

    for i_batch, sample_batched in enumerate(dataloader):

        # observe 4th batch and stop.
        if i_batch == 2:
            
            gen_loss, discr_loss, D_x, D_G_z1, D_G_z2 = train_batch(generator, discriminator, sample_batched, csp_shape, adversarial_loss, optimizer)

            break

