#!/usr/bin/env python
from sphere import *
import subprocess

def compare(first, second, string):
  if (first == second):
    print(string + "\tPassed")
  else:
    print(string + "\tFailed")


#### Input/output tests ####
print("### Input/output tests ###")

# Generate data in python
orig = Spherebin(np = 100, nw = 1, sid = "test")
orig.generateRadii(histogram = False)
orig.defaultParams()
orig.initRandomGridPos(g = numpy.zeros(orig.nd))
orig.initTemporal(current = 0.0, total = 0.0)
orig.time_total = 2.0*orig.time_dt;
orig.time_file_dt = orig.time_dt;
orig.writebin(verbose=False)

# Test Python IO routines
py = Spherebin()
py.readbin("../input/test.bin", verbose=False)
compare(orig, py, "Python IO:")

# Test C++ IO routines
orig.run(verbose=False, hideinputfile=True)
#orig.run()
cpp = Spherebin()
cpp.readbin("../output/test.output00000.bin", verbose=False)
compare(orig, cpp, "C++ IO:   ")

# Test CUDA IO routines
cuda = Spherebin()
cuda.readbin("../output/test.output00001.bin", verbose=False)
cuda.time_current = orig.time_current
cuda.time_step_count = orig.time_step_count
compare(orig, cuda, "CUDA IO:  ")

# Remove temporary files
subprocess.call("rm ../input/" + orig.sid + ".bin", shell=True)
subprocess.call("rm ../output/" + orig.sid + ".*.bin", shell=True)

