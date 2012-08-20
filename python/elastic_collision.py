#!/usr/bin/env python
import subprocess
import numpy
import matplotlib.pyplot as plt

# Import sphere functionality
from sphere import *

# New class
init = Spherebin(np = 2, nd = 3)

#init.generateRadii(radius_mean = 0.5, radius_variance = 1e-15, histogram = 0)
init.radius[0] = numpy.ones(1, dtype=numpy.float64) * 0.5;
init.radius[1] = numpy.ones(1, dtype=numpy.float64) * 0.52;

init.defaultparams()

# The world should never be less than 3 cells in ANY direction, due to contact searcg algorithm
init.initsetup(gridnum = numpy.array([12,4,4]), periodic = 1, shearmodel = 2, g = numpy.array([0.0, 0.0, 0.0]))

init.x[0] = numpy.array([1, 2, 2])
init.x[1] = numpy.array([6, 2, 2])
#init.x[2] = numpy.array([7, 2, 2])
#init.x[3] = numpy.array([8, 2, 2])

init.nu = numpy.zeros(init.np, dtype=numpy.float64)

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


