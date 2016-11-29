#****************************************************#
# This file is part of OPTALG.                       #
#                                                    #
# Copyright (c) 2015-2016, Tomas Tinoco De Rubira.   #
#                                                    #
# OPTALG is released under the BSD 2-clause license. #
#****************************************************#

class StochProblemC:
    """
    Represents a stochastic optimization problem 
    of the form
    
    minimize(x)   E[F(x,w)]
    subject to    E[G(x,w)] <= 0
                  x in X
    """

    def eval_FG(self,x,w):
        """
        Evaluates objective function and constraints
        for a given realization of the uncertainty.
        
        Parameters
        ----------

        Returns
        -------
        """

        pass

    def eval_FG_approx(self,x):
        """
        Evaluates deterministic approximation of 
        objective function and constraints.

        Parameters
        ----------

        Returns
        -------
        """

        pass

    def eval_EFG(self,x):
        """
        Evaluates objective function and
        constraints.
        
        Parameters
        ----------

        Returns
        -------
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

    def get_init_x(self):
        """
        Gets initial point.

        Returns
        -------
        x : vector
        """

        return None

    def get_size_lam(self):
        """
        Gets dimension of lambda, which is the
        vector of Lagrange multipliers associated
        with the stochastic constraints.

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

    def project_lam(self,lam):
        """
        Projects lam onto nonnegative orthand.
        
        Parameters
        ----------
        lam : vector

        Returns
        -------
        lamp : vector
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

    def solve_Lrelaxed_approx(self,lam,g_corr=None,J_corr=None,quiet=False):
        """
        Solves slope-corrected Lagrangian-relaxed problem
        
        minimize(x)   F_approx(x) + lam^TG_approx(x) + g^Tx + lam^TJx (slope correction)
        subject to    x in X
        """

        pass

    def get_strong_convexity_constant(self):

        return 1.
