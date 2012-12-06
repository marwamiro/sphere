Introduction
============
The *sphere*-software is used for three-dimensional discrete element method (DEM) particle simulations. The source code is written in C++, CUDA C and Python, and is compiled by the user. The main computations are performed on the graphics processing unit (GPU) using NVIDIA's general purpose parallel computing architecture, CUDA. Simulation setup and data analysis is performed with the included Python API.
The ultimate aim of the *sphere* software is to simulate soft-bedded subglacial conditions, while retaining the flexibility to perform simulations of granular material in other environments.

The purpose of this documentation is to provide the user with a thorough walk-through of the installation, work-flow, data-analysis and visualization methods of *sphere*. In addition, the *sphere* internals are exposed to provide a way of understanding of the discrete element method numerical routines taking place.

.. note:: Command examples in this document starting with the symbol ``$`` are meant to be executed in the shell of the operational system, and ``>>>`` means execution in Python. 

All numerical values in this document, the source code, and the configuration files are typeset with strict respect to the SI unit system.

Requirements
------------
The build requirements are:
  * A Nvidia CUDA-supported version of Linux or Mac OS X (see the `CUDA toolkit release notes <http://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html>`_ for more information)
  * `GNU Make <https://www.gnu.org/software/make/>`_
  * The `GNU Compiler Collection <http://gcc.gnu.org/>`_ (GCC)
  * The `Nvidia CUDA toolkit and SDK <https://developer.nvidia.com/cuda-downloads>`_

The runtime requirements are:
  * A `CUDA-enabled GPU <http://www.nvidia.com/object/cuda_gpus.html>`_ with compute capability 1.1 or greater.
  * A Nvidia CUDA-enabled GPU and device driver

Optional tools, required for simulation setup and data processing:
  * `Python 2.7 <http://www.python.org/getit/releases/2.7/>`_
  * `Numpy <http://numpy.scipy.org>`_
  * `Matplotlib <http://matplotlib.org>`_
  * `Imagemagick <http://www.imagemagick.org/script/index.php>`_
  * `ffmpeg <http://ffmpeg.org/>`_

Optional tools, required for building the documentation:
  * `Sphinx <http://sphinx-doc.org>`_
  * `Doxygen <http://www.stack.nl/~dimitri/doxygen/>`_
  * `Breathe <http://michaeljones.github.com/breathe/>`_

`Git <http://git-scm.com>`_ is used as the distributed version control system platform, and the source code is maintained at `Github <https://github.com/anders-dc/sphere/>`_. *sphere* is licensed under the `GNU Public License, v.3 <https://www.gnu.org/licenses/gpl.html>`_.


Building *sphere*
-----------------
All instructions required for building *sphere* are provided in a number of ``Makefiles``. To generate the main *sphere* command-line executable, go to the source code directory, and invoke GNU Make::

 $ cd src
 $ make

If successfull, the GNU Makefile will create the required data folders, object files, as well as the *sphere* executable in the root folder. The executable will be named after the host architecture, e.g. ``sphere_darwin_i386`` on a 32-bit OS X system, or ``sphere_linux_x86_64`` on a 64-bit linux system. Issue the following command to check the executable::

 $ ./sphere_* --version

The documentation can be read in the `reStructuredText <http://docutils.sourceforge.net/docs/ref/rst/restructuredtext.html>`_-format in the ``doc/sphinx/`` folder, or build into e.g. HTML or PDF format with the following commands::

 $ cd doc/sphinx
 $ make html
 $ make latexpdf

To see all available output formats, execute::

 $ make help

