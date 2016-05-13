#****************************************************#
# This file is part of OPTALG.                       #
#                                                    #
# Copyright (c) 2015-2016, Tomas Tinoco De Rubira.   #
#                                                    #
# OPTALG is released under the BSD 2-clause license. #
#****************************************************#

class StochObjMS_Policy:
    """
    Operation policy for multi-stage stochatic
    optimization problem (StochObjMS_Problem).
    """

    def apply(self,t,x_prev,Wt):
        """
        Applies operation policy at stage t
        given operation details of the previous stage
        and observations up to the current time.

        Parameters
        ----------
        t : int
        x_prev : vector
        Wt : list

        Returns
        -------
        x : vector
        """

        return None
