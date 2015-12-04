#****************************************************#
# This file is part of OPTALG.                       #
#                                                    #
# Copyright (c) 2015, Tomas Tinoco De Rubira.        #
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

    def eval_F(self,x,w):

        pass

    def eval_EF(self,x):

        pass

    def get_size_x(self):

        return 0

    def get_prop_x(self,x):
        
        return 0.

    def project_on_X(self,x):
        
        pass

    def sample_w(self):

        pass
        
    def show(self):

        pass
