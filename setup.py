#****************************************************#
# This file is part of OPTALG.                       #
#                                                    #
# Copyright (c) 2015-2017, Tomas Tinoco De Rubira.   #
#                                                    #
# OPTALG is released under the BSD 2-clause license. #
#****************************************************#

import sys
import argparse
from distutils.core import setup,Extension

ext_modules = []

parser = argparse.ArgumentParser()
parser.add_argument('--extensions',nargs='*',default=[])
args, unknown = parser.parse_known_args()
sys.argv = [sys.argv[0]] + unknown

# mumps
if 'all' in args.extensions or 'mumps' in args.extensions:
    from Cython.Build import cythonize
    ext_modules += cythonize([Extension(name='optalg.lin_solver._mumps._dmumps',
                                        sources=['./optalg/lin_solver/_mumps/_dmumps.pyx'])])

# ipopt
if 'all' in args.extensions or 'ipopt' in args.extensions:
    from Cython.Build import cythonize
    ext_modules += cythonize([Extension(name='optalg.opt_solver._ipopt.cipopt',
                                        sources=['./optalg/opt_solver/_ipopt/cipopt.pyx'])])

# clp
if 'all' in args.extensions or 'clp' in args.extensions:
    from Cython.Build import cythonize 
    ext_modules += cythonize([Extension(name='optalg.opt_solver._clp.cclp',
                                        sources=['./optalg/opt_solver/_clp/cclp.pyx'])])

# cbc
if 'all' in args.extensions or 'cbc' in args.extensions:
    from Cython.Build import cythonize 
    ext_modules += cythonize([Extension(name='optalg.opt_solver._cbc.ccbc',
                                        sources=['./optalg/opt_solver/_cbc/ccbc.pyx'])])
 
setup(name='OPTALG',
      version='1.1.2',
      description='Optimization Algorithms',
      author='Tomas Tinoco De Rubira',
      author_email='ttinoco5687@gmail.com',
      ext_modules=ext_modules,
      packages=['optalg',
                'optalg.lin_solver',
                'optalg.lin_solver._mumps',
                'optalg.opt_solver',
                'optalg.opt_solver._ipopt',
                'optalg.opt_solver._clp',
                'optalg.opt_solver._cbc',
                'optalg.stoch_solver'],
      requires=['scipy',
                'numpy',
                'dill'])
