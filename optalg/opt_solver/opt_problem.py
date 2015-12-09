#****************************************************#
# This file is part of OPTALG.                       #
#                                                    #
# Copyright (c) 2015, Tomas Tinoco De Rubira.        #
#                                                    #
# OPTALG is released under the BSD 2-clause license. #
#****************************************************#

class OptProblem:
    """
    Optimization problem class. 
    It represents problem of the form
    
    minimize    phi(x)

    subject to  Ax = 0       : lam
                f(x) = 0     : nu
                x <= u       : mu
                x >= l       : pi
    """

    def __init__(self):
        """
        Constructor.
        """
        
        # Objective
        self.phi = 0     # value
        self.gphi = None # gradient
        self.Hphi = None # Hessian (lower triangular)
        
        # Linear constraints
        self.A = None
        self.b = None

        # Nonlinear constraints
        self.f = None     # violations
        self.J = None     # Jacobian
        self.Hcomb = None # linear combination of constraint Hessians
        
        # Bounds
        self.u = None # upper bound
        self.l = None # lower bound

        # Initial point
        self.x = None   # primal
        self.lam = None # dual (lin constr)
        self.nu = None  # dual (nonlin constr)
        self.mu = None  # dual (upper bounds)
        self.pi = None  # dual (lower bounds)
        
    def eval(self,x):
        """
        Evaluates the objective value and constraints
        at the give point.
        """

        pass

    def show(self):
        """
        Displays information about the problem.
        """

        pass
