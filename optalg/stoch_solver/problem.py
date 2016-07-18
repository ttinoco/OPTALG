#****************************************************#
# This file is part of OPTALG.                       #
#                                                    #
# Copyright (c) 2015-2016, Tomas Tinoco De Rubira.   #
#                                                    #
# OPTALG is released under the BSD 2-clause license. #
#****************************************************#

class StochProblem:
    """
    Represents a stochastic optimization problem 
    of the form
    
    minimize(x)   E[F(x,w)]
    subject to    x in X.
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
        Evaluates deterministic approximation of objective function.
        
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
        Evaluates objective function.

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
        xp : vector
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
        Solves slope-corrected approximate problem
        
        minimize(x)   F_approx(x) + g^Tx (sloped correction)
        subject to    x in X.
        """
        
        pass

    def get_strong_convexity_constant(self):

        return 1.
