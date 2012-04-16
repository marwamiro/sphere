=============
SPHERE readme
=============
Sphere is a 3D discrete element method algorithm utilizing CUDA.

License
=======
Sphere is licensed under the GNU General Public License, v.3.
See license.txt for more information.

Documentation
=============
See the ``doc/`` folder for general reference.

Requirements
============
The build requirements are:
 - A Nvidia CUDA-supported version of Linux or Mac OS X
 - The GNU Compiler Collection (GNU)
 - The Nvidia CUDA toolkit and SDK
The runtime requirements are:
 - A Nvidia CUDA-enabled GPU and device driver
Optional tools, required for simulation setup and data processing:
 - Python 2
 - Numpy
 - Matplotlib
 - Imagemagick
 - ffmpeg

Obtaining SPHERE
================
The best way to keep up to date with subsequent updates, bugfixes
and development, is to use the GIT version control system.
To obtain a local copy, execute:
  ``git clone git://github.com/anders-dc/sphere.git``

Build instructions
==================
  ``cd src/``
  ``make``
The compiler will generate a command-line executable in the root 
folder. The SPHERE raytracer must be built seperately:
  ``cd raytracer/``
  ``make``
