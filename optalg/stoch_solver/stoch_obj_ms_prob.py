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

    def get_num_stages(self):
        """
        Gets number of stages.

        Returns
        -------
        num : int
        """
        
        return 0

    def get_size_x(self):
        """
        Gets size of stage vector x.

        Returns
        -------
        size : int
        """

        return 0

    def eval_stage_approx(self,t,w_list,x_prev,g_corr=[],quiet=False,tol=1e-4):
        """
        Evaluates approximate optimal stage cost.

        Parameters
        ----------
        t : int (stage)
        x_prev : vector
        w_list : list of random vectors for stage t,...,T
        g_corr : list of slope corrections for stage t,...,T
        quiet : {True,False}

        Returns
        -------
        x : stage solution
        Q : stage cost
        gQ : stage cost subgradient wrt x_prev
        """

        return None,None,None

    def sample_w(self,t,observations):
        """
        Samples realization of renewable powers for the given stage
        given the observations.

        Parameters
        ----------
        t : int (stage)
        observations : list

        Parameters
        ----------
        w : vector
        """

        return None

    def predict_w(self,t,observations):
        """
        Prodicts renewable powers for the given stage
        given the observations.

        Parameters
        ----------
        t : int (stage)
        observations : list

        Returns
        -------
        w : vector
        """

        return None

    def show(self):
        """
        Shows problem information.
        """

        pass
