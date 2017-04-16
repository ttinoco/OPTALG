.. _start:

***************
Getting Started
***************

This section describes how to get started with OPTALG. In particular, it covers dependencies, installation, and provides a quick example showing how to use this package.

.. _start_dependencies:

Dependencies
============

OPTALG has the following dependencies:

* `Numpy`_ (>=1.11.2)
* `Scipy`_ (>=0.18.1)
* `Dill`_ (>=0.2.5)
* `Cython`_ (>=0.20.1)
* `MUMPS`_ (==4.10.0) (optional)
* `IPOPT`_ (>=3.12.6) (optional)
* `CLP`_ (>=1.15.5) (optional)
* `CBC`_ (>=2.8.7) (optional)

.. _start_download:

Download
========

The latest version of OPTALG can be downloaded from `<https://github.com/ttinoco/OPTALG>`_.

.. _start_installation:

Installation
============

The OPTALG Python module can be installed using::

  sudo pip install -r requirements.txt
  sudo python setup.py install

from the root directory of the package. By default, no wrappers are built for any of the optional dependencies. To build wrappers, the command ``build_ext`` should be added along with the option ``--with`` followed by the key word ``all`` or by a list of specific wrappers to be built, *e.g.*, ``mumps ipopt clp cbc``. For example, the following command builds wrappers for `MUMPS`_ and `CBC`_::

  sudo python setup.py build_ext --with "mumps cbc" install 

To make this work, the options ``--libraries``, ``--include-dirs``, ``--library-dirs``, and ``--rpath`` of the command ``build_ext`` should be provided as needed either on the command line or through a configuration file `setup.cfg`_. A sample configuration file for OPTALG can be found :download:`here <../setup.cfg>`, which corresponds to an installation of all wrappers in a system with `IPOPT`_ installed in a non-standard location. 

To test OPTLAG, first execute the command ``build_ext`` with the option ``--inplace`` and then use `Nose`_, as in the following example::

  python setup.py build_ext --inplace --with "mumps cbc"
  nosetests -s -v

Example
=======

As a quick example of how to use OPTALG, consider the task of solving a quadratic program. This can be done as follows::

  >>> coming soon

.. _Numpy: http://www.numpy.org
.. _Scipy: http://www.scipy.org
.. _Dill: https://pypi.python.org/pypi/dill
.. _Cython: http://cython.org/
.. _MUMPS: http://mumps.enseeiht.fr/
.. _IPOPT: https://projects.coin-or.org/Ipopt
.. _CLP: https://projects.coin-or.org/Clp
.. _CBC: https://projects.coin-or.org/Cbc
.. _Nose: http://nose.readthedocs.io/en/latest/
.. _setup.cfg : https://docs.python.org/2/distutils/configfile.html

