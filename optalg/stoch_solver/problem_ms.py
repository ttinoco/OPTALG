#****************************************************#
# This file is part of OPTALG.                       #
#                                                    #
# Copyright (c) 2015-2017, Tomas Tinoco De Rubira.   #
#                                                    #
# OPTALG is released under the BSD 2-clause license. #
#****************************************************#

class StochProblemMS:
    """
    Represents a stochastic multi-stage optimization problem 
    of the form
    
    minimize(x_t)  F_t(x_t,w_t) + E[ H_{t+1}(x_t,W_{t+1}) | W_t ]
    subject to     x_t in X(x_{t-1},w_t),

    for t = 0,...,T-1, where H_{t+1} captures the optimal objective value of
    the next stage for a particular realization of the uncertainty.
    """

    def eval_F(self,t,x,w):
        """
        Evaluates F_t(x_t,w_t).

        Parameters
        ----------
        t : {0,...,T-1}
        x : vector
        w : vector
        
        Returns
        -------
        F : float
        """

        return None

    def get_num_stages(self):
        """
        Gets number of stages.
        
        Returns
        -------
        num : int
        """
        
        return 0

    def get_size_x(self,t):
        """
        Gets size of vector x for a given stage.

        Parameters
        ----------
        t : {0,...,T-1}

        Returns
        -------
        size : int
        """

        return 0

    def get_x_prev(self):
        """
        Gets constant x for time before t=0.
        
        Returns
        -------
        x_prev : vector
        """
        
        return None

    def solve_stage_with_cuts(self,t,w,x_prev,A,b,tol=1e-4,init_data=None,quiet=False):
        """
        Solves approximate stage problem for given realization of
        uncertainty and cuts that approximate cost-to-go function.
        More specifically, problem solved is

        minimize(x_t,y_t)   F_t(x_t,w_t) + y_t
        subject to          x_t in X(x_{t-1},w_t)
                            Ax_t + b + 1y_t >= 0.

        Parameters
        ----------
        t : {0,...,T-1}
        w : random vector
        x_prev : vector of previous stage variables
        A : matrix for constructing cuts
        b : vector for contructing cuts
        tol : solver tolerance
        init_data : dict (for warm start)
        quiet : {True,False}        

        Results
        -------
        x : stage-t solution
        H : stage-t cost
        gH : stage-t cost subgradient wrt x_prev
        """
        
        return None,None,None

    def solve_stages(self,t,w_list,x_prev,g_list=[],tf=None,tol=1e-4,quiet=False,next=False):
        """
        Solves stages using given realizations of uncertainty 
        and cost-to-go slope corrections.

        Parameters
        ----------
        t : {0,...,T-1}
        w_list : list of random vectors for stage t,...,tf
        x_prev : vector of previous stage variables
        g_list : list of slope corrections of objective functions for stage t,...,tf
        tf : {0,...,T-1} (T-1 by default)
        tol : float
        quiet : {True,False}
        next : {True,False}

        Returns
        -------
        x : stage-t solution
        H : stage-t cost
        gH : stage-t cost subgradient wrt x_prev
        gHnext : stage-(t+1) cost subgradient wrt x
        """

        return None,None,None

    def is_point_feasible(self,t,x,x_prev,w):
        """
        Checks wether point is feasible for the given stage.

        Parameters
        ----------
        t : {0,...,T-1}
        x : vector
        x_prev : vector
        w : vector

        Returns
        -------
        flag : {True,False}
        """
        
        return False

    def sample_w(self,t,observations):
        """
        Samples realization of uncertainty for the given stage
        given the observations.

        Parameters
        ----------
        t : {0,...,T-1}
        observations : list (length t)

        Parameters
        ----------
        w : vector
        """

        return None

    def sample_W(self,t):
        """
        Samples realization of uncertainty up
        to the given stage.

        Parameters
        ----------
        t : {0,...,T-1}

        Parameters
        ----------
        W : list (length t+1)
        """

        return None

    def predict_w(self,t,observations):
        """
        Predicts realizatoin of uncertainty for the given stage
        given the observations.

        Parameters
        ----------
        t : {0,...,T-1}
        observations : list (length t)

        Returns
        -------
        w : vector
        """

        return None

    def predict_W(self,t):
        """
        Predicts realizations of uncertainty up to the
        given stage.

        Parameters
        ----------
        t : {0,...,T-1}

        Returns
        -------
        W : list (length t+1)
        """

        return None

    def show(self):
        """
        Shows problem information.
        """

        pass
