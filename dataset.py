from csp import CSP, matrix_assignment_consistency
import torch
from torch.utils.data import Dataset
from solver import CSPSolver, solutionGatherer
import numpy as np
import time

class CSPDataset(Dataset):
    """
    Generate a CSP Dataset
    """

    def __init__(self, size =100, k=4, n=20, alpha=0.4, r=1.4, p=0.5, transform=False):
    
        """
            Initialize the CSPDataset object

            Args:
                k: arity of each constraint.
                n: number of variables of the CSP.
                alpha: parameter that indicates the domain sizes d = n^a
                r: parameter that indicates the number of constraints m=nr ln n
                p: parameter for the tightness of each constraint
                transform: Optional transform to be applied 
                    on a sample.
        """

        self.transform = transform
        self.csp = CSP(k, n, alpha, r, p)
        solv = CSPSolver(self.csp.n, self.csp.d, self.csp.matrix, limit=50)

        if solv.solution_count() == 0:
            raise Exception('No solutions found')

        satisfying_assignments = solv.get_satisfying_assignments()

        

        # labeling sat assignments
        sat_labels = np.ones((solv.solution_count(), 1))
        sat_assignments =np.concatenate((satisfying_assignments, sat_labels), axis=1)
        print('sat assgn', satisfying_assignments.shape, sat_labels.shape, sat_assignments.shape)

        # getting random generated assignments
        random_assignments = self.csp.generate_rnd_assignment(solv.solution_count()*8)

        # labeling rnd assignments
        consistency = consistency = matrix_assignment_consistency(torch.from_numpy(random_assignments), torch.from_numpy(self.csp.matrix), self.csp.d, self.csp.n)
        rnd_assignments = np.concatenate((random_assignments, consistency.numpy()), 1)

        print('rnd assgn', random_assignments.shape, consistency.shape, rnd_assignments.shape)
        self.assignments = np.concatenate((sat_assignments, rnd_assignments), 0)

        print('final assignments: ', self.assignments.shape)
    def __len__(self):
        """
        Compute the length of the dataset

        Returns:
            returns the size of the dataset
        """
        return self.assignments.shape[0]



    def __getitem__(self, idx):
        """
        Get item of index idx from the dataset
        
        Returns:
            returns the element of the dataset of index idx
        """
        sample = self.assignments[idx]

        if self.transform:
            sample = self.transform(sample)


        sample_unsqueezed = torch.unsqueeze(sample, 0)

        # # check consistency
        # consistency = matrix_assignment_consistency(sample_unsqueezed.type(torch.int64), torch.from_numpy(self.csp.matrix), self.csp.d)

        # # label each sample with its consistency
        # sample = torch.cat((sample_unsqueezed, consistency.type(torch.float)),1)
        # sample.squeeze()

        return sample_unsqueezed

class ToTensor(object):
    """
    Convert CSP assignments to tensors

    """
    def __call__(self, sample):
        return torch.from_numpy(sample).float()

from torch.utils.data import DataLoader

if __name__ == "__main__":

    random_seed = 15

    dataset = CSPDataset(size=10, k=4, n=20, alpha=0.4, r=1.4, p=0.5, transform=ToTensor())
    print(dataset.assignments.shape)

    dataloader = DataLoader(dataset, batch_size=3,
                        shuffle=True, num_workers=4)

    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched.shape)

        # observe 4th batch and stop.
        if i_batch == 2:
            print (sample_batched)
            break