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

    def eval_EFG(self,x):

        pass

    def eval_L(self,x,lam,w):

        pass

    def eval_EL(self,x,lam):

        pass

    def get_size_x(self):

        pass

    def get_prop_x(self,x):
        
        return 0.

    def project_on_X(self,x):
        
        pass

    def sample_w(self):

        pass
        
    def show(self):

        pass

    def solve_Lrelaxed_certainty_equivalent(self,lam,g_corr=None,J_corr=None,Ew=None,tol=1e-4,quiet=False,samples=500):
        """
        Solves
        
        minimize(x)   F(x,Ew) + lam^TG(x,Ew) + g^Tx + lam^TJx (slope corrected)
        subject to    x in X
        """

        pass
