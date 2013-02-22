#!/usr/bin/env python
from pytestutils import *

#### Input/output tests ####
print("### Memory tests ###")

# Generate data in python
orig = Spherebin(np = 100, nw = 1, sid = "test-initgrid")
orig.generateRadii(histogram = False)
orig.defaultParams()
orig.initRandomGridPos(g = numpy.zeros(orig.nd))
orig.initTemporal(current = 0.0, total = 0.0)
orig.time_total = 2.0*orig.time_dt;
orig.time_file_dt = orig.time_dt;
orig.writebin(verbose=False)

# Test C++ routines
print("Valgrind: C++ routines")
orig.run(verbose=False, hideinputfile=True, valgrind=True)


# Test CUDA routines
print("cuda-memcheck: CUDA routines")
orig.run(verbose=False, hideinputfile=True, cudamemcheck=True)

# Remove temporary files
cleanup(orig)
