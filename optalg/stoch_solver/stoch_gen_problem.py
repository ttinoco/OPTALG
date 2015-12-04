#****************************************************#
# This file is part of OPTALG.                       #
#                                                    #
# Copyright (c) 2015, Tomas Tinoco De Rubira.        #
#                                                    #
# OPTALG is released under the BSD 2-clause license. #
#****************************************************#

class StochGen_Problem:
    """
    Represents a stochastic optimization problem 
    of the form
    
    minimize(x)   E[F(x,w)]
    subject to    E[G(x,w)] <= 0
                  x in X
    """

    def eval_FG(self,x,w):

        pass

    def eval_FG_approx(self,x):

        pass

    def eval_EFG(self,x):

        pass

    def eval_L(self,x,lam,w):

        pass

    def eval_L_approx(self,x,lam):

        pass

    def eval_EL(self,x,lam):

        pass

    def get_size_x(self):

        return 0

    def get_size_lam(self):

        return 0

    def get_prop_x(self,x):
        
        return 0.

    def project_x(self,x):
        
        pass

    def project_lam(self,lam):

        pass

    def sample_w(self):

        pass
        
    def show(self):

        pass

    def solve_Lrelaxed_approx(self,lam,g_corr=None,J_corr=None,tol=1e-4,quiet=False):
        """
        Solves
        
        minimize(x)   F_approx + lam^TG_approx(x) + g^Tx + lam^TJx (slope correction)
        subject to    x in X
        """

        pass
