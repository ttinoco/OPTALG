#****************************************************#
# This file is part of OPTALG.                       #
#                                                    #
# Copyright (c) 2015-2016, Tomas Tinoco De Rubira.   #
#                                                    #
# OPTALG is released under the BSD 2-clause license. #
#****************************************************#

from lin_solver import LinSolver

def new_linsolver(name,prop):
    
    if name == 'mumps':
        from mumps import LinSolverMUMPS
        return LinSolverMUMPS(prop)
    elif name == 'superlu':
        from superlu import LinSolverSUPERLU
        return LinSolverSUPERLU(prop)
    elif name == 'default':
        try:
            return new_linsolver('mumps',prop)
        except ImportError:
            return new_linsolver('superlu',prop)            
    else:
        raise ValueError('invalid linear solver name')

