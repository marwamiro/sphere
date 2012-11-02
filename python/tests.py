#!/usr/bin/env python
from sphere import *

def compare(first, second, string):
  if (first == second):
    print(string + "\tPassed")
  else:
    print(string + "\tFailed")


#### Input/output tests ####
print("### Input/output tests ###")

# Generate data in python
orig = Spherebin(np = 100, nw = 0)
orig.generateRadii()
orig.defaultParams()
orig.initRandomGridPos(g = numpy.zeros(orig.nd))
orig.initTemporal(current = 0.0, total = 0.0)
orig.time_total = 2.0*orig.time_dt;
orig.time_file_dt = orig.time_dt;
orig.writebin("orig.bin", verbose=False)

# Test Python IO routines
py = Spherebin()
py.readbin("orig.bin", verbose=False)
compare(orig, py, "Python IO:")

# Test C++ IO routines
run("python/orig.bin", verbose=False, hideinputfile=True)
cpp = Spherebin()
cpp.readbin("../output/orig.output0.bin", verbose=False)
compare(orig, cpp, "C++ IO:   ")

# Test CUDA IO routines
cuda = Spherebin()
cuda.readbin("../output/orig.output1.bin", verbose=False)
cuda.time_current = orig.time_current
cuda.time_step_count = orig.time_step_count
compare(orig, cuda, "CUDA IO:  ")

