#!/usr/bin/env python
import subprocess
import numpy
import matplotlib.pyplot as plt

# Import sphere functionality
from sphere import *

# New class
init = Spherebin(np = 2, nd = 3)

init.radius = numpy.ones(init.np, dtype=numpy.float64) * 0.5;

init.defaultparams()

# The world should never be less than 3 cells in ANY direction, due to contact searcg algorithm
init.initsetup(gridnum = numpy.array([12,4,4]), periodic = 1, shearmodel = 2, g = numpy.array([0.0, 0.0, 0.0]))

init.x[0] = numpy.array([1, 2, 2])
init.x[1] = numpy.array([6, 2, 2])
#init.x[2] = numpy.array([7, 2, 2])
#init.x[3] = numpy.array([8, 2, 2])

# Set fraction of critical damping (0 = elastic, 1 = completely inelastic)
damping_fraction = 1.0
init.nu = numpy.ones(init.np, dtype=numpy.float64) \
          * damping_fraction * 2.0 * math.sqrt(4.0/3.0 * math.pi * init.radius.min()**3 \
	  * init.rho[0] * init.k_n[0])


#for i in range(init.np):
#  init.x[i] = numpy.array([4+i*init.radius[i]*2, 2, 2])

init.vel[0] = numpy.array([10.0, 0.0, 0.0])


init.initTemporal(total = 1.0)

init.writebin("../input/nc-test.bin")
#render("../input/nc-test.bin", out = "~/Desktop/nc-test")

subprocess.call("cd ..; rm output/*; ./sphere_darwin_X86_64 nc-test", shell=True)

visualize("nc-test", "energy", savefig=True, outformat='png')
#visualize("nc-test", "walls", savefig=True)

subprocess.call("rm ../img_out/*; cd ../raytracer; ./render_all_outputs_GPU_clever.sh", shell=True)


