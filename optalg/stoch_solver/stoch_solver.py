#****************************************************#
# This file is part of OPTALG.                       #
#                                                    #
# Copyright (c) 2015-2017, Tomas Tinoco De Rubira.   #
#                                                    #
# OPTALG is released under the BSD 2-clause license. #
#****************************************************#

class StochSolver:
    """"
    Parent class for stochastic optimnization solvers.
    """

    def __init__(self):
        
        self.parameters = {}

        self.x = None

    def get_primal_variables(self):
        """
        Gets primal variables.
        
        Returns
        -------
        variables : ndarray
        """

        return self.x

    def set_parameters(self,parameters):
        """
        Sets solver parameters.
        
        Parameters
        ----------
        parameters : dict
        """
        
        for key,value in list(parameters.items()):
            if key in self.parameters:
                self.parameters[key] = value

    def solve(self,problem):
        """
        Solves stochatic optimization problem.

        Parameters
        ----------
        problem : StochProblem, StochProblemC, StochProblemMS
        """

        pass
    
        

    
