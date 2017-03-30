#****************************************************#
# This file is part of OPTALG.                       #
#                                                    #
# Copyright (c) 2015-2017, Tomas Tinoco De Rubira.   #
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

        #: Integer flags (boolean array)
        self.P = None

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

    def get_num_primal_variables(self):
        """
        Gets number of primal variables.

        Returns
        -------
        num : int
        """

        if self.x is not None:
            return self.x.size
        if self.gphi is not None:
            return self.gphi.size
        if self.Hphi is not None:
            return self.Hphi.shape[0]
        if self.A is not None:
            return self.A.shape[1]
        if self.J is not None:
            return self.J.shape[1]
        if self.u is not None:
            return self.u.size
        if self.l is not None:
            return self.l.size
        return 0

    def get_num_linear_equality_constraints(self):
        """
        Gets number of linear equality constraints.

        Returns
        -------
        num : int
        """

        if self.A is not None:
            return self.A.shape[0]
        return 0

    def get_num_nonlinear_equality_constraints(self):
        """
        Gets number of nonlinear equality constraints.

        Returns
        -------
        num : int
        """

        if self.f is not None:
            return self.f.size
        return 0

    def combine_H(self, coeff, ensure_psd=False):
        """
        Forms and saves a linear combination of the individual constraint Hessians.

        Parameters
        ----------
        coeff : vector
        ensure_psd : {``True``,``False``}
        """
        
        pass
        
    def eval(self,x):
        """
        Evaluates the objective value and constraints
        at the give point.

        Parameters
        ----------
        x : vector
        """

        pass

    def show(self):
        """
        Displays information about the problem.
        """

        pass
