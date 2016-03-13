#****************************************************#
# This file is part of OPTALG.                       #
#                                                    #
# Copyright (c) 2015-2016, Tomas Tinoco De Rubira.   #
#                                                    #
# OPTALG is released under the BSD 2-clause license. #
#****************************************************#

class OptProblem:

    def __init__(self):
        """
        Class for representing general optimization problems.
        """
        
        #: Objective function value
        self.phi = 0     

        #: Objective function gradient
        self.gphi = None 

        #: Objective function Hessian (lower triangular)
        self.Hphi = None 
        
        #: Matrix for linear equality constraints
        self.A = None

        #: Right-hand side for linear equality constraints
        self.b = None

        #: Nonlinear equality constraint function
        self.f = None    

        #: Jacobian of nonlinear constraints
        self.J = None    

        #: Linear combination of Hessians of nonlinear constraints
        self.H_combined = None
        
        #: Upper limits 
        self.u = None

        #: Lower limits
        self.l = None

        #: Initial point
        self.x = None

        #: Lagrande multipliers for linear equality constraints
        self.lam = None

        #: Lagrande multipliers for nonlinear equality constraints
        self.nu = None 

        #: Lagrande multipliers for upper limits
        self.mu = None 

        #: Lagrande multipliers for lower limits 
        self.pi = None 
        
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
