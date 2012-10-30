#!/usr/bin/env python
from sphere import *

def compare(first, second, string):
  if (first == second):
    print(string + ":\tPassed")
  else:
    print(string + ":\tFailed")


#### Input/output tests ####
print("### Input/output tests ###")

# Generate data in python
orig = Spherebin(100)
orig.generateRadii()
orig.defaultParams()
orig.initRandomGridPos()
orig.initTemporal(total = 0.0, current = 0.0)
orig.xysum = numpy.ones(orig.np*2, dtype=numpy.float64).reshape(orig.np, 2) * 1
orig.vel = numpy.ones(orig.np*orig.nd, dtype=numpy.float64).reshape(orig.np, orig.nd) * 2
orig.force = numpy.ones(orig.np*orig.nd, dtype=numpy.float64).reshape(orig.np, orig.nd) * 3
orig.angpos = numpy.ones(orig.np*orig.nd, dtype=numpy.float64).reshape(orig.np, orig.nd) * 4
orig.angvel = numpy.ones(orig.np*orig.nd, dtype=numpy.float64).reshape(orig.np, orig.nd) * 5
orig.torque = numpy.ones(orig.np*orig.nd, dtype=numpy.float64).reshape(orig.np, orig.nd) * 6
orig.writebin("orig.bin", verbose=False)

# Test Python IO routines
py = Spherebin()
py.readbin("orig.bin", verbose=False)
compare(orig, py, "Python IO")

# Test C++ IO routines
run("python/orig.bin")
cpp = Spherebin()
cpp.readbin("../output/orig.output0.bin", verbose=False)
compare(orig, cpp, "C++ IO   ")


