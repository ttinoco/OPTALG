#****************************************************#
# This file is part of OPTALG.                       #
#                                                    #
# Copyright (c) 2015, Tomas Tinoco De Rubira.        #
#                                                    #
# OPTALG is released under the BSD 2-clause license. #
#****************************************************#

from distutils.core import setup,Extension

ext_modules = [Extension(name='optalg.lin_solver._mumps._dmumps',
                         sources=['./optalg/lin_solver/_mumps/_dmumps.c'],
                         libraries=['dmumps_seq'],
                         library_dirs=[],
                         extra_link_args=[])]

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
                'numpy'])
