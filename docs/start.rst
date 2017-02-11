.. _start:

***************
Getting Started
***************

This section describes how to get started with OPTALG. In particular, it covers required packages, installation, and provides a quick example showing how to use this package.

.. _start_requirements:

* `Numpy`_ (>=1.8.2)
* `Scipy`_ (>=0.13.3)
* `Dill`_ (>=0.2.5)
* `MUMPS`_ (==4.10) (optional)
* `IPOPT`_ (>=3.12.6) (optional)

.. _start_download:

Download
========

The latest version of OPTALG can be downloaded from `<https://github.com/ttinoco/OPTALG>`_.

.. _start_installation:

Installation
============

The OPTALG Python module can be installed using::

  > sudo -E python setup.py install

from the root directory of the package. If `MUMPS`_ is not available, then the option ``--no_mumps`` should be added to the above command. If `IPOPT`_ is not available, then the option ``--no_ipopt`` should be added to the above command. Otherwise, you need to define an environment variable ``IPOPT`` such that the directories ``IPOPT/lib`` and ``IPOPT/include/coin`` contain the libraries and header files, respectively, needed by `IPOPT`_. 

Example
=======

As a quick example of how to use OPTALG, consider the task of solving a quadratic program. This can be done as follows::

  >>> coming soon

.. _Numpy: http://www.numpy.org
.. _Scipy: http://www.scipy.org
.. _Dill: https://pypi.python.org/pypi/dill
.. _MUMPS: http://mumps.enseeiht.fr/
.. _IPOPT: https://projects.coin-or.org/Ipopt

