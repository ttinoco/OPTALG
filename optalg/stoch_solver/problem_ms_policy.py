#****************************************************#
# This file is part of OPTALG.                       #
#                                                    #
# Copyright (c) 2015-2017, Tomas Tinoco De Rubira.   #
#                                                    #
# OPTALG is released under the BSD 2-clause license. #
#****************************************************#

import numpy as np
from .problem_ms import StochProblemMS

class StochProblemMS_Policy:
    """
    Operation policy for multi-stage stochatic
    optimization problem.
    """

    def __init__(self,problem,name='',data=None,construction_time=0.,param1=None,param2=None):
        """
        Operation policy for multistage stochastic
        optimization problem.
        
        Parameters
        ----------
        problem : StochProblemMS
        name : string
        data : Object
        construction_time : float (minutes)
        param1 : 
        param2 :
        """

        assert(isinstance(problem,StochProblemMS))

        self.name = name
        self.data = data
        self.problem = problem
        self.construction_time = construction_time
        self.param1 = param1
        self.param2 = param2

    def get_name(self):
        """
        Gets policy name.

        Return
        ------
        name : str
        """
        
        return self.name

    def get_problem(self):
        """
        Gets problem.

        Return
        ------
        problem : StochProblemMS
        """
        
        return self.problem

    def get_construction_time(self):
        """
        Gets construction time.

        Return
        ------
        time : float
        """

        return self.construction_time

    def get_param1(self):
        """
        Gets policy parameter 1.

        Return
        ------
        param : 
        """

        return self.param1

    def get_param2(self):
        """
        Gets policy parameter 2.

        Return
        ------
        param : 
        """

        return self.param2

    def apply(self,t,x_prev,W):
        """
        Applies operation policy at stage t
        given operation details of the previous stage
        and observations up to the current time.

        Parameters
        ----------
        t : {0,...,T-1}
        x_prev : vector
        W : list (length t+1)

        Returns
        -------
        x : vector
        """

        return None
