#****************************************************#
# This file is part of OPTALG.                       #
#                                                    #
# Copyright (c) 2019, Tomas Tinoco De Rubira.        #
#                                                    #
# OPTALG is released under the BSD 2-clause license. #
#****************************************************#

import numpy as np
from types import MethodType
from scipy.sparse import eye, bmat, triu, coo_matrix

class OptProblem(object):
    """
    Class for representing general optimization problems.
    """

    # Properties
    PROP_CURV_LINEAR = 'linear'
    PROP_CURV_QUADRATIC = 'quadratic'
    PROP_CURV_NONLINEAR = 'nonlinear'
    PROP_VAR_INTEGER = 'integer'
    PROP_VAR_CONTINUOUS = 'continuous'
    PROP_TYPE_FEASIBILITY = 'feasibility'
    PROP_TYPE_OPTIMIZATION = 'optimization'

    def __init__(self):
        """
        Class for representing general optimization problems.

        Parameters
        ----------
        problem : Object
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
        
        #: Wrapped problem
        self.wrapped_problem = None

    def recover_primal_variables(self, x):
        """
        Recovers primal variables for original problem.

        Parameters
        ----------
        x : ndarray
        """

        return x

    def recover_dual_variables(self, lam, nu, mu, pi):
        """
        Recovers dual variables for original problem.

        Parameters
        ----------
        lam : ndarray
        nu : ndarray
        mu : ndarray
        pi : ndarray
        """

        return lam, nu, mu, pi
        
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
        
    def eval(self, x):
        """
        Evaluates the objective value and constraints
        at the give point.

        Parameters
        ----------
        x : vector
        """

        pass

    def show(self, inf=1e8):
        """
        Displays information about the problem.
        """

        print('\nProblem info')
        print('------------')
        print('vars: %d' %self.gphi.size)
        print('integers: %d' %np.sum(self.P == True))
        print('A: rows %d cols %d nnz %d' %(self.A.shape[0], self.A.shape[1], self.A.nnz))
        print('J: rows %d cols %d nnz %d' %(self.J.shape[0], self.J.shape[1], self.J.nnz))
        print('u: %d' %(np.sum(self.u < inf)))
        print('l: %d' %(np.sum(self.l > -inf)))

    def to_lin(self):
        """
        Converts problem to linear problem.

        Returns
        -------
        p : |LinProblem|
        """

        from .problem_lin import LinProblem

        self.eval(self.x)

        c = self.gphi.copy()

        return LinProblem(c, self.A, self.b, self.l, self.u, self.x)

    def to_quad(self):
        """
        Converts problem to quadratic problem.

        Returns
        -------
        p : |QuadProblem|
        """

        from .problem_quad import QuadProblem

        self.eval(self.x)

        H = self.Hphi + self.Hphi.T - triu(self.Hphi)
        g = self.gphi - H*self.x

        return QuadProblem(H, g, self.A, self.b, self.l, self.u, self.x)

    def to_mixintlin(self):
        """
        Converts problem to mixed integer linear problem.

        Returns
        -------
        p : |MixIntLinProblem|
        """

        from .problem_mixintlin import MixIntLinProblem

        self.eval(self.x)

        c = self.gphi.copy()

        if self.P is None:
            self.P = np.array([False]*self.x.size, dtype=bool)

        return MixIntLinProblem(c, self.A, self.b, self.l, self.u, self.P, self.x)

def cast_problem(problem):
    """
    Casts problem object with known interface as OptProblem.

    Parameters
    ----------
    problem : Object
    """
    
    # Optproblem
    if isinstance(problem, OptProblem):
        return problem
    
    # Other
    else:
        
        # Type Base
        if (not hasattr(problem,'G') or 
            (problem.G.shape[0] == problem.G.shape[1] and
             problem.G.shape[0] == problem.G.nnz and
             np.all(problem.G.row == problem.G.col) and 
             np.all(problem.G.data == 1.))):
            return create_problem_from_type_base(problem)

        # Type A
        else:
            return create_problem_from_type_A(problem)

def create_problem_from_type_base(problem):
    """
    Creates OptProblem from type-base problem.

    Parameters
    ----------
    problem : Object
    """

    p = OptProblem()

    # Init attributes
    p.phi = problem.phi
    p.gphi = problem.gphi 
    p.Hphi = problem.Hphi 
    p.A = problem.A
    p.b = problem.b
    p.f = problem.f
    p.J = problem.J
    p.H_combined = problem.H_combined
    p.u = problem.u
    p.l = problem.l
    p.x = problem.x
    
    p.P = None
    p.lam = None
    p.nu = None
    p.mu = None
    p.pi = None
    
    p.wrapped_problem = problem
    
    # Methods
    def eval(cls, x):
        cls.wrapped_problem.eval(x)
        cls.phi = cls.wrapped_problem.phi
        cls.gphi = cls.wrapped_problem.gphi
        cls.Hphi = cls.wrapped_problem.Hphi
        cls.f = cls.wrapped_problem.f
        cls.J = cls.wrapped_problem.J
        
    def combine_H(cls, coeff, ensure_psd=False):
        cls.wrapped_problem.combine_H(coeff, ensure_psd)
        cls.H_combined = cls.wrapped_problem.H_combined
        
    p.eval = MethodType(eval, p)
    p.combine_H = MethodType(combine_H, p)

    # Return
    return p

def create_problem_from_type_A(problem):
    """
    Creates OptProblem from type-A problem.

    Parameters
    ----------
    problem : Object
    """
    
    p = OptProblem()
    
    nx = problem.get_num_primal_variables()
    nz = problem.G.shape[0]

    p.phi = problem.phi
    p.gphi = np.hstack((problem.gphi,np.zeros(nz)))
    p.Hphi = coo_matrix((problem.Hphi.data,(problem.Hphi.row,problem.Hphi.col)),shape=(nx+nz,nx+nz))    
    p.A = bmat([[problem.A,None],[problem.G,-eye(nz)]],format='coo')
    p.b = np.hstack((problem.b,np.zeros(nz)))
    p.f = problem.f
    p.J = coo_matrix((problem.J.data,(problem.J.row,problem.J.col)),shape=(problem.J.shape[0],nx+nz))
    p.H_combined = coo_matrix((problem.H_combined.data,(problem.H_combined.row,problem.H_combined.col)),shape=(nx+nz,nx+nz))
    p.u = np.hstack((problem.get_upper_limits(),problem.u))                
    p.l = np.hstack((problem.get_lower_limits(),problem.l))
    p.x = np.hstack((problem.x,np.zeros(nz)))

    p.P = None
    p.lam = None
    p.nu = None
    p.mu = None
    p.pi = None
    
    p.wrapped_problem = problem

    def eval(cls, xz):
        x = xz[:nx]
        z = xz[nx:]
        prob = cls.wrapped_problem
        prob.eval(x)
        cls.phi = prob.phi
        cls.gphi = np.hstack((prob.gphi,np.zeros(nz)))
        cls.Hphi = coo_matrix((prob.Hphi.data,(prob.Hphi.row,prob.Hphi.col)),shape=(nx+nz,nx+nz))
        cls.f = prob.f
        cls.J = coo_matrix((prob.J.data,(prob.J.row,prob.J.col)),shape=(prob.J.shape[0],nx+nz))

    def combine_H(cls, coeff, ensure_psd=False):
        prob = cls.wrapped_problem
        prob.combine_H(coeff,ensure_psd=ensure_psd)
        cls.H_combined = coo_matrix((prob.H_combined.data,(prob.H_combined.row,prob.H_combined.col)),shape=(nx+nz,nx+nz))
            
    def recover_primal_variables(cls, x):
        return x[:nx]
    
    def recover_dual_variables(cls, lam, nu, mu, pi):
        prob = cls.wrapped_problem
        return lam[:prob.A.shape[0]],nu,mu[nx:],pi[nx:]
                
    p.eval = MethodType(eval, p)
    p.combine_H = MethodType(combine_H, p)
    p.recover_primal_variables = MethodType(recover_primal_variables, p)
    p.recover_dual_variables = MethodType(recover_dual_variables, p)

    # Return
    return p
