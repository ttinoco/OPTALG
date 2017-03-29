#****************************************************#
# This file is part of OPTALG.                       #
#                                                    #
# Copyright (c) 2015-2017, Tomas Tinoco De Rubira.   #
#                                                    #
# OPTALG is released under the BSD 2-clause license. #
#****************************************************#

import numpy as np
cimport numpy as np

cimport cclp

np.import_array()

from scipy.sparse import csc_matrix

class ClpContextError(Exception):
    """
    Clp context error exception.
    """
    def __init__(self,value):
        self.value = value
    def __str__(self):
        return repr(self.value)

cdef class ClpContext:
    """
    Clp context class.
    """
    
    cdef cclp.Clp_Simplex* model
    
    def __cinit__(self):

        self.model = cclp.Clp_newModel()

    def __dealloc__(self):

        if self.model != NULL:
            cclp.Clp_deleteModel(self.model)
        self.model = NULL

    def loadProblem(self,numcols,numrows,A,collb,colub,obj,rowlb,rowub):
        
        A = csc_matrix(A)
        
        cdef np.ndarray[int,mode='c'] _start = A.indptr
        cdef np.ndarray[int,mode='c'] _index = A.indices
        cdef np.ndarray[double,mode='c'] _value = A.data
        cdef np.ndarray[double,mode='c'] _collb = collb
        cdef np.ndarray[double,mode='c'] _colub = colub
        cdef np.ndarray[double,mode='c'] _obj = obj
        cdef np.ndarray[double,mode='c'] _rowlb = rowlb
        cdef np.ndarray[double,mode='c'] _rowub = rowub

        assert(_value.size == _index.size)
        assert(A.shape == (numrows,numcols))
        assert(A.indices.size == (A.shape[1]+1))

        cclp.Clp_loadProblem(self.model,
                             numcols,
                             numrows,
                             <int*>(_start.data),
                             <int*>(_index.data),
                             <double*>(_value.data),
                             <double*>(_collb.data),
                             <double*>(_colub.data),
                             <double*>(_obj.data),
                             <double*>(_rowlb.data),
                             <double*>(_rowub.data))
        
    
        
        
        
