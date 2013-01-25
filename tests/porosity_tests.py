#!/usr/bin/env python
from pytestutils import *

#### Porosity tests ####
print("### porosity tests ###")

# Generate data in python
orig = Spherebin(np = 100, nw = 1, sid = "test-initgrid")
orig.generateRadii(histogram = False)
orig.defaultParams()
orig.initRandomGridPos(g = numpy.zeros(orig.nd))
orig.initTemporal(current = 0.0, total = 0.0)
orig.time_total = 2.0*orig.time_dt;
orig.time_file_dt = orig.time_dt;
orig.writebin(verbose=False)

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

