=============
SPHERE readme
=============
Sphere is a 3D discrete element method algorithm utilizing CUDA.

Sphere is licensed under the GNU General Public License, v.3.
See license.txt for more information.

See the ``doc/`` folder for general reference.

Requirements
============
The build requirements are:
- A Nvidia CUDA-supported version of Linux or Mac OS X
- The GNU Compiler Collection (GCC)
- The Nvidia CUDA toolkit and SDK

The runtime requirements are:
- A Nvidia CUDA-enabled GPU and device driver

Optional tools, required for simulation setup and data processing:
- Python 2.7
- Numpy
- Matplotlib
- Imagemagick
- ffmpeg

Optional tools, required for building the documentation:
- Sphinx
- Doxygen
- Breathe

Obtaining SPHERE
================
The best way to keep up to date with subsequent updates, bugfixes
and development, is to use the GIT version control system.

To obtain a local copy, execute:
  ``git clone https://github.com/anders-dc/sphere.git``

Build instructions
==================
 cd src/ && make

This will generate a command-line executable in the root folder.
