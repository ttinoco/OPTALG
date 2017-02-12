#****************************************************#
# This file is part of OPTALG.                       #
#                                                    #
# Copyright (c) 2015-2017, Tomas Tinoco De Rubira.   #
#                                                    #
# OPTALG is released under the BSD 2-clause license. #
#****************************************************#

import numpy as np
cimport numpy as np

np.import_array()

EPS = 1e-4

cdef class AugLBounds:

    cdef int n
    cdef float du
    cdef object umin
    cdef object umax
    cdef object f
    cdef object Jrow
    cdef object Jcol
    cdef object Jdata
    cdef object Hcomb_row
    cdef object Hcomb_col
    cdef object Hcomb_data
    
    def __init__(self,umin,umax):
        
        assert(umin.size == umax.size)

        self.n = umin.size
        self.umin = umin
        self.umax = umax
        self.du = np.maximum(umax-umin,EPS)

        self.f = np.zeros(2*self.n)
        self.Jrow = np.array(range(2*self.n))
        self.Jcol = np.concatenate((range(self.n),range(self.n)))
        self.Jdata = np.zeros(2*self.n)
    
    def eval(self,u):

        a1 = self.umax-u
        a2 = u-self.umin
        b = EPS*EPS/self.du

        
        
        

