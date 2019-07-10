import numpy as np
import math
import CSP
import torch

class CSP():

    def __init__(self, k, n, alpha, r, p):

        """
            Initialize the CSP following the RB Model

            Args:
                k: arity of each constraint.
                n: number of variables of the CSP.
                alpha: parameter that indicates the domain sizes d = n^a
                r: parameter that indicates the number of constraints m=nr ln n
                p: parameter for the tightness of each constraint

        """
        self.k = k
        self.n = n
        self.d = int(math.ceil(math.pow(n, alpha)))
        self.m = int(r * n * math.log(n))
        self.n_bad_assgn = int(math.ceil(p * math.pow(self.d, self.k)))

        constraints = self.generate_CSP(alpha, r, p)

        matrix_csp = constraints2matrix(constraints, self.n, self.d)

        self.constraints = constraints
        self.matrix = matrix_csp
    
    def check_consistent_assgn(self, assignment, csp):
        """
            Check if the given assignment assignment is consistent with the CSP csp

            Args:
                assignment: assignment to check for consistency
                csp: csp considered for the assignment

            Returns:
                boolean value that is False if the assignment is not consistent with the
                csp, True otherwise
        """

        const_scopes = self.constraints['scopes']
        const_assignment = self.constraints['values']
        # check if each non-valid assignment of the constraint is present in the given assignment
        inconsistent = True
        for const in range(self.m):
            for assgn in range(self.n_bad_assgn):
                for var in range(self.k):
                    if assignment[const_scopes[const, var]] == const_assignment[const, assgn, var]:
                        inconsistent = False
                        break
        
        return inconsistent
    
    def generate_rnd_assignment(self, n_assignments=1):
        """
        Generate an assignment for the CSP

        Args:
            size: int indicating the number of assignments to generate
        
        Returns:
            Matrix n x k of assignments for the CSP 
        """

        return np.random.randint(0, self.d, (n_assignments, self.n))


    
    def generate_CSP(self, alpha, r, p):

        """
        Generate the CSP following the RB Model[Xu and Li, 2000].

        Args:
            k: arity of each constraint.
            n: number of variables of the CSP.
            alpha: parameter that indicates the domain sizes d = n^a
            r: parameter that indicates the number of constraints m=nr ln n
            p: parameter for the tightness of each constraint

        Returns:
            Returns a CSP of n variables, domains of size d = n^alpha, m= nr ln n constraints which have dp^k 
            unallowed assignments.
        """

        # domains have values 0, 1, 2,..., d-1
        # select m=rn ln n constraints

        # all constraints have the same number of possible assignments d^k
        # let's see them as numbers on a d-digits
        tot_assgn = int(math.pow(self.d, self.k))
        print("n", self.n, ", k", self.k, ", d", self.d,", m", self.m, ", n_bad", self.n_bad_assgn)
        values = np.zeros([self.m, self.n_bad_assgn, self.k], dtype='int64')
        scopes = np.zeros([self.m, self.k], dtype='int64')

        for ic in range(0,self.m):
            c_scope = choose_without_rep(self.n, self.k)
            scopes[ic, :] = c_scope
            assgn = choose_without_rep(tot_assgn, self.n_bad_assgn)
            iv = 0
            for assgn_idx in assgn:
                notsat_assgn = number2assignment(assgn_idx, self.d, self.k)
                values[ic, iv, :] = notsat_assgn      
                iv += 1
            
        return {'scopes':scopes, 'values': values}

    def matrix_assignment_consistency(assignment, matrix, d, n):

        """
        Check if the given CSP matrix is satisfied by the assignment

        Args:
            assignment: tensor containing the batch of assignments (Batch, Variables)
            matrix: tensor representing the CSP (Batch, Domain*Variables, Domain*Variables)

        Returns:
            tensor of size (Batch, 1) which has 1 in i-th position if the i-th assignment satisfies the CSP
        """
        # TODO:CHECK CONSISTENCY nonzero returns two arrays with indexes of non zero elements - not pairs of elements
        disallawed =torch.nonzero()

        consistency = torch.ones(assignment.shape[0], 1, dtype=torch.int64)

        for assgn in range(assignment.shape[0]):
            for x in disallawed:
                if (assignment[assgn][x[0]//d] == x[0]%d and assignment[assgn][x[1]//d]==x[1]%d):
                    consistency[assgn] = 0
                    break
        return consistency

        
def choose_without_rep(n, k):

    """
    Choose k variables over n without repetitions.

    Args:
        n: total number of elements.
        k: number of elements to choose

    Returns:
        Returns the index of k random elements chosen over n ones without repetitions
    """

    indexes = np.arange(n)
    for x in range(k):
        idx = np.random.randint(0, n - x)
        chosen = indexes[idx]
        indexes[idx] = indexes[n-x-1]
        indexes[n-x-1] = chosen
    return indexes[-k:]


def number2assignment(number, base, n_digits):

    """
    Transform assignment from an identifier notation to an assignment where each value is associated
    to the column variable. Transform from a 10-base number to a d-base one

    Args:
        number: identifier to transform
        base: base of the transformed number

    Returns:
        Return the assignment as a vector
    """

    integer = number
    digits = []
    while integer != 0:
        decimal = integer % base
        digits.append(int(decimal))
        integer //= base
    for i in range(n_digits - len(digits)):
        digits.append(0)
    return digits[::-1]


def constraints2matrix(assignments, n, d):
    """
    Transform the assignements in the form of a dictionary with 'bad_values'
    and 'scopes' into a matrix of n*d n*d values such that 0 for consistent assignment
    and 1 for non consistent ones

    Args:
        assignments: dictionary with values and scopes of the csp
        n: number of variables of the csp
        d: domain dimension of the csp
    """

    # empty matrix
    matrix = np.zeros((d*n, d*n), dtype='int32')
    values = assignments['values']
    scopes = assignments['scopes']

    # scan all the constraints
    for c in range(values.shape[0]):

        # scan all disallawed assignments
        for assgn in range(values.shape[1]):
            matrix[scopes[c][0]*d + values[c][assgn][0]][scopes[c][1]*d + values[c][assgn][1]] = 1
    return matrix


# testing functions
if __name__ == "__main__":
    rnd_seed = 890
    np.random.seed(rnd_seed)

    # (k, n, alpha, r, p)
    csp = CSP(2, 6, 0.4, 0.3, 0.5)
    
    assignment = np.random.randint(0,csp.d, 20)
    print(assignment)
    print(csp.check_consistent_assgn(assignment, csp))







    
