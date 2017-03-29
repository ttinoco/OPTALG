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

cdef extern from "numpy/arrayobject.h":
     void PyArray_ENABLEFLAGS(np.ndarray arr, int flags)
     void PyArray_CLEARFLAGS(np.ndarray arr, int flags)

cdef ArrayDouble(double* a, int size):
     cdef np.npy_intp shape[1]
     shape[0] = <np.npy_intp> size
     arr = np.PyArray_SimpleNewFromData(1,shape,np.NPY_DOUBLE,a)
     PyArray_CLEARFLAGS(arr,np.NPY_OWNDATA)
     return arr

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

    def loadProblem(self,n,A,collb,colub,obj,rowlb,rowub):
        
        A = csc_matrix(A)
        
        cdef np.ndarray[int,mode='c'] _start = A.indptr
        cdef np.ndarray[int,mode='c'] _index = A.indices
        cdef np.ndarray[double,mode='c'] _value = A.data
        cdef np.ndarray[double,mode='c'] _collb = collb
        cdef np.ndarray[double,mode='c'] _colub = colub
        cdef np.ndarray[double,mode='c'] _obj = obj
        cdef np.ndarray[double,mode='c'] _rowlb = rowlb
        cdef np.ndarray[double,mode='c'] _rowub = rowub

        assert(A.shape[1] == n)
        assert(_start.size == (n+1))
        assert(_start[n] == A.nnz)
        assert(_index.size == A.nnz)
        assert(_value.size == A.nnz)
        assert(_collb.size == n)
        assert(_colub.size == n)
        assert(_obj.size == n)
        assert(_rowlb.size == A.shape[0])
        assert(_rowub.size == A.shape[0])

        cclp.Clp_loadProblem(self.model,
                             n,
                             A.shape[0],
                             <int*>(_start.data),
                             <int*>(_index.data),
                             <double*>(_value.data),
                             <double*>(_collb.data),
                             <double*>(_colub.data),
                             <double*>(_obj.data),
                             <double*>(_rowlb.data),
                             <double*>(_rowub.data))

    def setlogLevel(self,value):

        cclp.Clp_setLogLevel(self.model,value)

    def status(self):

        return cclp.Clp_status(self.model)

    def initialSolve(self):
        
        return cclp.Clp_initialSolve(self.model)

    def primalColumnSolution(self):

        n = cclp.Clp_numberColumns(self.model)
        return ArrayDouble(cclp.Clp_primalColumnSolution(self.model),n)

    def primalRowSolution(self):

        m = cclp.Clp_numberRows(self.model)
        return ArrayDouble(cclp.Clp_primalRowSolution(self.model),m)

    def dualColumnSolution(self):

        n = cclp.Clp_numberColumns(self.model)
        return ArrayDouble(cclp.Clp_dualColumnSolution(self.model),n)

    def dualRowSolution(self):

        m = cclp.Clp_numberRows(self.model)
        return ArrayDouble(cclp.Clp_dualRowSolution(self.model),m)

        
        
        
        
