import numpy as np
import math
import CSP

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

            Returns:
                boolean value that is False if the assignment is not consistent with the
                csp, True otherwise
        """

        self.k = k
        self.n = n
        self.d = int(math.ceil(math.pow(n, alpha)))
        self.m = int(r * n * math.log(n))
        self.n_bad_assgn = int(math.ceil(p * math.pow(self.d, k)))


        self.constraints = generate_CSP(k, n, alpha, r, p)
    
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
    
    def generate_rnd_assignment(self, n=1):
        """
        Generate an assignment for the CSP

        Args:
            size: int indicating the number of assignments to generate
        
        Returns:
            Matrix n x k of assignments for the CSP 
        """

        return np.random.randint(0, self.d, (n, self.k))


    
def generate_CSP(k, n, alpha, r, p):

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
    d = int(math.ceil(math.pow(n, alpha)))
    m = int(r * n * math.log(n))
    n_bad_assgn = int(math.ceil(p * math.pow(d, k)))

    # all constraints have the same number of possible assignments d^k
    # let's see them as numbers on a d-digits
    tot_assgn = int(math.pow(d, k))
    print("d", d,"m", m, "n_bad", n_bad_assgn)
    values = np.zeros([m, n_bad_assgn, k], dtype='int64')
    scopes = np.zeros([m, k], dtype='int64')

    for ic in range(0,m):
        c_scope = choose_without_rep(n, k)
        scopes[ic, :] = c_scope
        assgn = choose_without_rep(tot_assgn, n_bad_assgn)
        iv = 0
        for assgn_idx in assgn:
            notsat_assgn = number2assignment(assgn_idx, d, k)
            values[ic, iv, :] = notsat_assgn      
            iv += 1
        
    return {'scopes':scopes, 'values': values}

        
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

# testing functions
if __name__ == "__main__":
    rnd_seed = 14
    np.random.seed(rnd_seed)

    # (k, n, alpha, r, p)
    csp = CSP(4, 20, 0.4, 1.4, 0.5)

    assignment = np.random.randint(0,csp.k, 20)
    print(assignment)
    print(csp.check_consistent_assgn(assignment, csp))







    
