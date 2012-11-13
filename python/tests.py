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
for N in [1, 2, 10, 1e2, 1e3, 1e4]:
  print("{} particle(s)".format(int(N)))
  orig = Spherebin(np = int(N), nw = 0, sid = "test")
  orig.generateRadii(histogram = False)
  orig.defaultParams()
  orig.initRandomGridPos(g = numpy.zeros(orig.nd), gridnum=numpy.array([N*N+3,N*N+3,N*N+3]))
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
  cpp = Spherebin()
  cpp.readbin("../output/test.output0.bin", verbose=False)
  compare(orig, cpp, "C++ IO:   ")

  # Test CUDA IO routines
  cuda = Spherebin()
  cuda.readbin("../output/test.output1.bin", verbose=False)
  cuda.time_current = orig.time_current
  cuda.time_step_count = orig.time_step_count
  compare(orig, cuda, "CUDA IO:  ")

# Remove temporary files
subprocess.call("rm ../{input,output}/test*bin", shell=True)

