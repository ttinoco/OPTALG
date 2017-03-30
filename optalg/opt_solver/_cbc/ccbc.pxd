#****************************************************#
# This file is part of OPTALG.                       #
#                                                    #
# Copyright (c) 2015-2017, Tomas Tinoco De Rubira.   #
#                                                    #
# OPTALG is released under the BSD 2-clause license. #
#****************************************************#

cdef extern from "coin/Cbc_C_Interface.h":

    ctypedef void Cbc_Model

    Cbc_Model* Cbc_newModel()
    void Cbc_deleteModel(Cbc_Model* model)

    void Cbc_loadProblem(Cbc_Model* model, int numcols, int numrows, int* start, int* index, double* value,
                         double* collb, double* colub, double* obj, double* rowlb, double* rowub)

    void Cbc_setLogLevel(Cbc_Model* model, int value)
    
    int Cbc_status(Cbc_Model* model)
    int Cbc_branchAndBound(Cbc_Model* model)
    
    int Cbc_getNumRows(Cbc_Model* model)
    int Cbc_getNumCols(Cbc_Model* model)
    
    void Cbc_copyInIntegerInformation(Cbc_Model* model, char* information)

    double* Cbc_getColSolution(Cbc_Model* model)
    
    int Cbc_isProvenOptimal(Cbc_Model* model)
    
