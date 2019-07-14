import torch.nn as nn
import numpy as np
import torch
from csp import CSP,  matrix_assignment_consistency
from solver import CSPSolver
import math

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):

    def __init__(self, input_size, csp_shape, dropout_prob= 0):

        super().__init__()

        self.d = csp_shape['d']
        self.n = csp_shape['n']
        self.sat_matrix_size = csp_shape['n']*csp_shape['d']

        self.conv = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.ConvTranspose2d( 1, out_channels=8, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d( 8, out_channels=16, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d( 16, out_channels=32, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d( 32, out_channels=64, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d( 64, out_channels=128, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.dense = nn.Sequential(
            nn.Linear(2437632, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, self.sat_matrix_size*self.sat_matrix_size + csp_shape['n']*self.d + 1)
        )

        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()



    def forward(self, x):
        x = self.conv(x)

        # flattening
        x = x.view(x.size()[0], -1)

        x = self.dense(x)

        print(x.shape)
        matrix = torch.narrow(x, 1, 0, self.sat_matrix_size*self.sat_matrix_size)
        print(matrix.shape)
        matrix = matrix.view(matrix.size()[0], self.sat_matrix_size, -1)
        matrix = torch.where(matrix < 0.5, torch.zeros(matrix.shape), torch.ones(matrix.shape))
        print('matrix', matrix.shape)

        assignments = torch.narrow(x, 1, self.sat_matrix_size * self.sat_matrix_size, self.n * self.d)
        assignments = assignments.view(assignments.size()[0], -1, self.n,)
        assignments = self.softmax(assignments)
        assignments = torch.argmax(assignments, dim=1, keepdim=True)
        print('assignment', assignments.shape)

        sat_label = torch.narrow(x, 1, self.sat_matrix_size * self.sat_matrix_size + self.n * self.d, 1)
        sat_label = sat_label.view(sat_label.size()[0], 1, -1)
        sat_label = self.sigmoid(sat_label)
        sat_label = torch.where(sat_label < 0.5, torch.zeros(sat_label.shape), torch.ones(sat_label.shape))

        print('sat_labels', sat_label.shape)

        print('ass ', type(assignments), ' sat', type(sat_label))
        assignments = torch.cat((assignments.type(torch.float), sat_label.type(torch.float)), dim=-1)

        print('cat', assignments.shape)
        return matrix, assignments

class Discriminator(nn.Module):


    def __init__(self, input_size, out_size, dropout_prob= 0):
        super().__init__()

        self.input_size = input_size
        self.model = nn.Sequential(
            nn.Conv1d(1, out_channels=16, kernel_size=5, stride=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(16, out_channels=32, kernel_size=3, stride=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3)

        )

        size = 32 * math.floor((math.floor(((input_size-4)/2) + 1) - 3)/2 + 1)
        print(size)
        self.dense = nn.Sequential(
            nn.Linear(size , 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 1),
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

    fake_csp, fake_batch = gen(rnd_assgn)

    # let's solve each problem with a solution
    n = int(csp_size['n'])
    d = int(csp_size['d'])

    # assignment = np.empty((fake_csp.shape[0], n+1))
    # for csp in range(fake_csp.shape[0]):
    #     solver = CSPSolver(n, d, fake_csp[csp, :, :], limit=1)
    #     if solver.solution_count()>=1:
    #         assgn = solver.get_satisfying_assignments(n=1)
    #         assgn = np.append(assgn, 1)

    #     else:
    #         assgn = np.random.randint(0, d-1, n)
    #         assgn = np.append(assgn, 0)
    #     assignment[csp, :] = assgn[:]
    
    # TODO:add non-sat assignments
    # random_assignment = self.csp.generate_rnd_assignment(size)

    #     # add satisfying assingnments
    #     if solv.solution_count()>0:
    #         satisfying_assignments = solv.get_satisfying_assignments()
    #         print(satisfying_assignments.shape, self.assignments.shape)
    #         self.assignments = np.concatenate((self.assignments, satisfying_assignments), axis=0)
    # assignment = ass
    # fake_batch = torch.from_numpy(assignment)
    # fake_batch.unsqueeze_(1)
    # fake_batch = fake_batch.type(torch.float)
    
    print(fake_batch.shape)
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

