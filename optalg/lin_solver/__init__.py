#****************************************************#
# This file is part of OPTALG.                       #
#                                                    #
# Copyright (c) 2015, Tomas Tinoco De Rubira.        #
#                                                    #
# OPTALG is released under the BSD 2-clause license. #
#****************************************************#

from lin_solver import LinSolver
from mumps import LinSolverMUMPS
from superlu import LinSolverSUPERLU

def new_linsolver(name,prop):
    
    if name == 'mumps':
        return LinSolverMUMPS(prop)
    elif name == 'superlu':
        return LinSolverSUPERLU(prop)
    else:
        raise ValueError('invalid linear solver name')

