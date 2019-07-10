from CSP import CSP
from ortools.sat.python.cp_model import CpModel, CpSolver, CpSolverSolutionCallback
import numpy as np

np.random.seed(seed=10)

class solutionGatherer(CpSolverSolutionCallback):
    """Print intermediate solutions."""

    def __init__(self, variables, limit):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__variables = variables
        self.__solution_count = 0
        self.__solution_limit = limit
        self.solutions = []

    def on_solution_callback(self):

        # a new solution has been found: one more 
        self.__solution_count += 1

        # appending the consistent assignment
        self.solutions.append([self.Value(v) for v in self.__variables])
        if self.__solution_count >= self.__solution_limit:
            self.StopSearch()


    def solution_count(self):
        return self.__solution_count
    
    def get_solutions(self):
        return np.array(self.solutions)

class Solver(object):

    def __init__(self, csp, limit=40):

        # getting csp
        self.csp = csp

        # initializing the model
        self.model = CpModel()

        # adding variables to the model
        self.variables = self.add_variables()

        # adding constraints
        self.add_disallawed_assignments(self.csp.matrix)

        solver = cp_model.CpSolver()
        self.gatherer = solutionGatherer(self.variables, limit)
        self.status = solver.SearchForAllSolutions(self.model, self.gatherer)


    def add_variables(self):
        """
        Add variables to the CpModel

        Returns:
            returns the variables added to the CpModel
        """
        variables = [None] * self.csp.n
        for i in range(self.csp.n):
            variables[i] = self.model.NewIntVar(0, self.csp.d-1, 'x'+str(i))
            
        return variables



    def add_disallawed_assignments(self, matrix):
        """
        Add constraints to the CpModel

        Args:
            matrix: matrix with forbidden assignments
        """

        d = self.csp.d
        disallowed = np.nonzero(matrix)
        for ix in range(disallowed[0].shape[0]):
            scope = [self.variables[disallowed[0][ix]//d], self.variables[disallowed[1][ix]//d]]
            assignment = [(int(disallowed[0][ix]%d), int(disallowed[1][ix]%d))]
            self.model.AddForbiddenAssignments(scope, assignment)

    def has_solutions(self):
        """
        Indicates if the model is feasible or not
        
        Returns:
            True if the model is FEASIBLE, False otherwise
        """
        if self.status==cp_model.FEASIBLE:
            return False
        return True

    def solution_count(self):
        """
        Indicates if the number of solutions found
        
        Returns:
            Return the number of satisfying assignments
        """
        return self.gatherer.solution_count()

        
    def get_satisfying_assignments(self, n = None):
        """
        Gives n satisfing assignments for the given matrix
        
        Args:
            n: number of assignments required
        
        Returns:
            n satisfying assignments, if n is not specified all assignments are returned

        Raises:
            Exception if the number of required assignments is greater than the number of satisfying assignments
        """
        if (n != None) and (n > self.gatherer.solution_count()):
            raise Exception('The number of solution required is greater than the number of satisfying assingments')
        solutions = self.gatherer.get_solutions()

        if n != None:
                solutions = solutions[:n]

        return solutions

# testing 
from ortools.sat.python import cp_model
if __name__=='__main__':

    # creating the csp
    csp = CSP(k=2, n=24, alpha=1.0, r=1.4, p=0.5)
 
    # initiate Solver
    solv = Solver(csp, 4)

    print(solv.solution_count())

    print(solv.get_satisfying_assignments(None))
    

