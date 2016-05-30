#*****************************************************#
# This file is part of OPTALG.                        #
#                                                     #
# Copyright (c) 2015-2016, Tomas Tinoco De Rubira.    #
#                                                     #
# OPTALG is released under the BSD 2-clause license.  #
#*****************************************************#

def ApplyFunc(args):

    cls = args[0]
    fnc = args[1]
    args = args[2:]
    
    return getattr(cls,fnc)(*args)
