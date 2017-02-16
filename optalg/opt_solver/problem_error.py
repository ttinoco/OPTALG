#****************************************************#
# This file is part of OPTALG.                       #
#                                                    #
# Copyright (c) 2015-2017, Tomas Tinoco De Rubira.   #
#                                                    #
# OPTALG is released under the BSD 2-clause license. #
#****************************************************#

class OptProblemError(Exception):
    
    def __init__(self,value):
        self.value = value
        
    def __str__(self):
        return str(self.value)

class OptProblemError_InvalidDataDimensions(OptProblemError):    
    def __init__(self):
        OptProblemError.__init__(self,'invalid data dimemnesions')


