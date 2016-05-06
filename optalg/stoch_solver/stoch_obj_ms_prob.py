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

    pass
