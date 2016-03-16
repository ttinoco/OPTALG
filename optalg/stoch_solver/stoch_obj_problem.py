#****************************************************#
# This file is part of OPTALG.                       #
#                                                    #
# Copyright (c) 2015-2016, Tomas Tinoco De Rubira.   #
#                                                    #
# OPTALG is released under the BSD 2-clause license. #
#****************************************************#

class StochObj_Problem:
    """
    Represents a stochastic optimization problem 
    of the form
    
    minimize(x)   E[F(x,w)]
    subject to    x in X.
    """

    def eval_F(self,x,w,tol=None):

        pass

    def eval_F_approx(self,x,tol=None):

        pass

    def eval_EF(self,x,tol=None):

        pass

    def get_size_x(self):

        return 0

    def get_prop_x(self,x):
        
        return 0.

    def project_x(self,x):
        
        pass

    def sample_w(self):

        pass
        
    def show(self):

        pass

    def solve_approx(self,g_corr=None,quiet=False,tol=1e-4):
        """
        Solves
        
        minimize(x)   F_approx(x) + g^Tx (sloped correction)
        subject to    x in X.
        """
        
        pass

    def get_strong_convexity_constant(self):

        return 1.
