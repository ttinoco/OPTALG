#****************************************************#
# This file is part of OPTALG.                       #
#                                                    #
# Copyright (c) 2015, Tomas Tinoco De Rubira.        #
#                                                    #
# OPTALG is released under the BSD 2-clause license. #
#****************************************************#

import numpy as np

class StochasticSolver:
    """"
    Parent class for stochastic optimnization solvers.
    """

    def __init__(self,problem):
        
        self.problem = problem

    def solve(self,x=None,maxiters=100,callback=None):

        pass
        
    
        

    
