#****************************************************#
# This file is part of OPTALG.                       #
#                                                    #
# Copyright (c) 2015-2016, Tomas Tinoco De Rubira.   #
#                                                    #
# OPTALG is released under the BSD 2-clause license. #
#****************************************************#

class StochObjMS_Problem:
    """
    Represents a stochastic multi-stage optimization problem 
    of the form
    
    minimize(x)   F(x_t,w_t) + E[ Q_{t+1}(x_t,W_{t+1}) | W_t ]
    subject to    x_t in X(x_{t-1},w_t),

    where Q_{t+1} captures the optimal objective value of
    the next stage for a particular realization of the uncertainty.
    """

    def eval_F(self,x,w,tol=None):
        """
        Evaluate objective function for specific realization
        of uncertainty.

        Parameters
        ----------
        x : vector
        w : vector
        tol : float

        Returns
        -------
        F : float
        gF : vector (gradient or subgradient)
        """

        pass

    def eval_F_approx(self,x,tol=None):
        """
        Evaluates deterministic approximation of objective funciton.
        
        Parameters
        ----------
        x : vector
        tol : float

        Returns
        -------
        F : float
        gF : vector (gradient or subgradient)
        """

        pass

    def eval_EF(self,x,samples=100,tol=None):
        """
        Evaluates objective funciton.

        Parameters
        ----------
        x : vector
        samples : int
        tol : float
        
        Returns
        -------
        EF : float
        gEF : vector (gradient or subgradient)
        """

        pass

    def get_size_x(self):
        """
        Gets dimension of x.

        Returns
        -------
        dim : int
        """
        
        return 0

    def get_prop_x(self,x):
        """
        Gets some scalar property of x for 
        debugging/monitoring purposes.

        Parameters
        ----------
        x : vector

        Returns
        -------
        prop : float
        """

        return 0.

    def project_x(self,x):
        """
        Projects x onto feasible set.
        
        Parameters
        ----------
        x : vector

        Returns
        -------
        y : vector
        """

        pass

    def sample_w(self):
        """
        Samples realization of random vector.

        Returns
        -------
        w : vector
        """

        pass
        
    def show(self):
        """ 
        Shows problem properties.
        """

        pass

    def solve_approx(self,g_corr=None,quiet=False,tol=1e-4):
        """
        Solves slope-corrected problem
        
        minimize(x)   F_approx(x) + g^Tx (sloped correction)
        subject to    x in X.
        """
        
        pass

    def get_strong_convexity_constant(self):

        return 1.
