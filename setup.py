#****************************************************#
# This file is part of OPTALG.                       #
#                                                    #
# Copyright (c) 2015-2016, Tomas Tinoco De Rubira.   #
#                                                    #
# OPTALG is released under the BSD 2-clause license. #
#****************************************************#

import sys
from setuptools import setup,Extension

ext_modules = []

# mumps
if '--no_mumps' in sys.argv:
    sys.argv.remove('--no_mumps')
else:
    ext_modules.append(Extension(name='optalg.lin_solver._mumps._dmumps',
                                 sources=['./optalg/lin_solver/_mumps/_dmumps.c'],
                                 libraries=['dmumps_seq'],
                                 library_dirs=[],
                                 extra_link_args=[]))

setup(name='OPTALG',
      version='1.0',
      description='Optimization Algorithms',
      author='Tomas Tinoco De Rubira',
      author_email='ttinoco5687@gmail.com',
      ext_modules=ext_modules,
      packages=['optalg',
                'optalg.lin_solver',
                'optalg.lin_solver._mumps',
                'optalg.opt_solver',
                'optalg.stoch_solver'],
      requires=['scipy',
                'numpy',
                'multiprocess'])
