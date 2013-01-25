#!/usr/bin/env python
from sphere import *
import subprocess

def passed():
    return "\tPassed"

def failed():
    return "\tFailed"

def compare(first, second, string):
  if (first == second):
    print(string + passed())
  else:
    print(string + failed())

def compareFloats(first, second, string, criterion=1e-5):
    if abs(first-second) < criterion:
        print(string + passed())
    else :
        print(string + failed())

def cleanup(spherebin):
    'Remove temporary files'
    subprocess.call("rm -f ../input/" + spherebin.sid + ".bin", shell=True)
    subprocess.call("rm -f ../output/" + spherebin.sid + ".*.bin", shell=True)
    print("")


#### Input/output tests ####
print("### Input/output tests ###")

# Generate data in python
orig = Spherebin(np = 100, nw = 1, sid = "test-initgrid")
orig.generateRadii(histogram = False)
orig.defaultParams()
orig.initRandomGridPos(g = numpy.zeros(orig.nd))
orig.initTemporal(current = 0.0, total = 0.0)
orig.time_total = 2.0*orig.time_dt;
orig.time_file_dt = orig.time_dt;
orig.writebin(verbose=False)

# Test Python IO routines
py = Spherebin()
py.readbin("../input/" + orig.sid + ".bin", verbose=False)
compare(orig, py, "Python IO:")

# Test C++ IO routines
orig.run(verbose=False, hideinputfile=True)
#orig.run()
cpp = Spherebin()
cpp.readbin("../output/" + orig.sid + ".output00000.bin", verbose=False)
compare(orig, cpp, "C++ IO:   ")

# Test CUDA IO routines
cuda = Spherebin()
cuda.readbin("../output/" + orig.sid + ".output00001.bin", verbose=False)
cuda.time_current = orig.time_current
cuda.time_step_count = orig.time_step_count
compare(orig, cuda, "CUDA IO:  ")

# Remove temporary files
cleanup(orig)


#### Porosity tests ####
print("### porosity tests ###")

def testPorosities(spherebin):

    # Number of vertical slices
    slicevals = [1, 2, 4]
    i = 1   # iterator var
    for slices in slicevals:

        # Find correct value of bulk porosity
        n_bulk = spherebin.bulkPorosity()
        #print("Bulk: " + str(n_bulk))

        porosity = spherebin.porosity(slices = slices)[0]
        #print("Avg: " + str(numpy.average(porosity)))
        #print(porosity)

        # Check if average of porosity function values matches the bulk porosity
        compareFloats(n_bulk, numpy.average(porosity), \
                spherebin.sid + ": Porosity average to bulk porosity ("\
                + str(i) + "/" + str(len(slicevals)) + "):")
        i += 1

# Test data from previous test
testPorosities(orig)

# Simple cubic packing of uniform spheres
# The theoretical porosity is (4/3*pi*r^3)/(2r)^3 = 0.476
sidelen = 10
cubic = Spherebin(np = sidelen**3, sid='cubic')
radius = 1.0
cubic.generateRadii(psd='uni', radius_mean=radius, radius_variance=0.0, histogram=False)
for ix in range(sidelen):
    for iy in range(sidelen):
        for iz in range(sidelen):
            i = ix + sidelen * (iy + sidelen * iz) # linear index
            cubic.x[i,0] = ix*radius*2.0 + radius
            cubic.x[i,1] = iy*radius*2.0 + radius
            cubic.x[i,2] = iz*radius*2.0 + radius
cubic.L[:] = 2.0 * radius * sidelen

cubic.initTemporal(0.2)
cubic.initGrid()

testPorosities(cubic)

cleanup(cubic)

