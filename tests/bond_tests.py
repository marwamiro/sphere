#!/usr/bin/env python
from pytestutils import *
from sphere import *

def printKinematics(sb):
    print('bonds_delta_n'); print(sb.bonds_delta_n)
    print('bonds_delta_t'); print(sb.bonds_delta_t)
    print('bonds_omega_n'); print(sb.bonds_omega_n)
    print('bonds_omega_t'); print(sb.bonds_omega_t)
    print('force'); print(sb.force)
    print('torque'); print(sb.torque)
    print('vel'); print(sb.vel)
    print('angvel'); print(sb.angvel)

#### Bond tests ####
print("### Bond tests ###")

# Zero arrays
z2_1 = numpy.zeros((2,1))
z2_2 = numpy.zeros((2,2))
z1_3 = numpy.zeros((1,3))
z2_3 = numpy.zeros((2,3))

# Small value arrays
smallval = 1e-8
s2_1 = numpy.ones((2,1))*smallval

# Inter-particle distances to try (neg. for overlap)
#distances = [0.2, 0.0, -0.2]
#distances = [0.2, 0.0]
distances = []
#distances = [0.2]

for d in distances:

    radii = 0.5
    print("## Inter-particle distance: " + str(d/radii) + " radii")

    sb = Spherebin(np=2, sid='bondtest')
    cleanup(sb)

    # setup particles, bond, and simulation
    sb.x[0,:] = numpy.array((10.0, 10.0, 10.0))
    sb.x[1,:] = numpy.array((10.0+2.0*radii+d, 10.0, 10.0))
    sb.radius = numpy.ones(sb.np)*radii
    sb.initGridAndWorldsize(margin = 10, periodic = 1, contactmodel = 2, g = numpy.array([0.0, 0.0, 0.0]))
    sb.bond(0, 1)
    sb.defaultParams(gamma_n = 0.0, gamma_t = 0.0)
    #sb.initTemporal(total=0.5, file_dt=0.01)
    #sb.render(verbose=False)
    #visualize(sb.sid, "energy")


    print("# Stability test")
    sb.x[0,:] = numpy.array((10.0, 10.0, 10.0))
    sb.x[1,:] = numpy.array((10.0+2.0*radii+d, 10.0, 10.0))
    sb.zeroKinematics()
    sb.initTemporal(total=0.2, file_dt=0.01)
    #sb.initTemporal(total=0.01, file_dt=0.0001)
    sb.writebin(verbose=False)
    sb.run(verbose=False)
    #sb.run()
    sb.readlast(verbose=False)
    compareFloats(0.0, sb.energy("kin") + sb.energy("rot") + sb.energy("bondpot"), "Energy cons.")
    compareNumpyArrays(sb.vel, z2_3, "vel\t")
    compareNumpyArrays(sb.angvel, z2_3, "angvel\t")
    #printKinematics(sb)
    #visualize(sb.sid, "energy")
    #sb.readbin('../output/' + sb.sid + '.output00001.bin')
    #printKinematics(sb)

    print("# Normal expansion")
    sb.x[0,:] = numpy.array((10.0, 10.0, 10.0))
    sb.x[1,:] = numpy.array((10.0+2.0*radii+d, 10.0, 10.0))
    sb.zeroKinematics()
    sb.initTemporal(total=0.2, file_dt=0.01)
    sb.vel[1,0] = 1e-4
    Ekinrot0 = sb.energy("kin") + sb.energy("rot")
    sb.writebin(verbose=False)
    sb.run(verbose=False)
    sb.readlast(verbose=False)
    compareFloats(Ekinrot0, sb.energy("kin") + sb.energy("rot") + sb.energy("bondpot"), "Energy cons.")
    print("vel[:,0]"),
    if ((sb.vel[0,0] <= 0.0) or (sb.vel[1,0] <= 0.0)):
        print(failed())
    else :
        print(passed())
    compareNumpyArrays(sb.vel[:,1:2], z2_2, "vel[:,1:2]")
    compareNumpyArrays(sb.angvel, z2_3, "angvel\t")
    printKinematics(sb)
    #visualize(sb.sid, "energy")
    continue

    print("# Normal contraction")
    sb.zeroKinematics()
    sb.initTemporal(total=0.2, file_dt=0.01)
    sb.vel[1,0] = -1e-4
    Ekinrot0 = sb.energy("kin") + sb.energy("rot")
    sb.writebin(verbose=False)
    sb.run(verbose=False)
    sb.readlast(verbose=False)
    compareFloats(Ekinrot0, sb.energy("kin") + sb.energy("rot") + sb.energy("bondpot"), "Energy cons.")
    print("vel[:,0]"),
    if ((sb.vel[0,0] >= 0.0) or (sb.vel[1,0] >= 0.0)):
        print(failed())
    else :
        print(passed())
    compareNumpyArrays(sb.vel[:,1:2], z2_2, "vel[:,1:2]")
    compareNumpyArrays(sb.angvel, z2_3, "angvel\t")
    #printKinematics(sb)

    print("# Shear")
    sb.x[0,:] = numpy.array((10.0, 10.0, 10.0))
    sb.x[1,:] = numpy.array((10.0+2.0*radii+d, 10.0, 10.0))
    sb.zeroKinematics()
    sb.initTemporal(total=0.2, file_dt=0.01)
    sb.vel[1,2] = 1e-4
    Ekinrot0 = sb.energy("kin") + sb.energy("rot")
    sb.writebin(verbose=False)
    sb.run(verbose=False)
    sb.readlast(verbose=False)
    compareFloats(Ekinrot0, sb.energy("kin") + sb.energy("rot") + sb.energy("bondpot"), "Energy cons.")
    #print("vel[:,0]"),
    #if ((sb.vel[0,0] <= 0.0) or (sb.vel[1,0] >= 0.0)):
    #    print(failed())
    #else :
    #    print(passed())
    compareNumpyArrays(sb.vel[:,1], z2_1, "vel[:,1]")
    print("vel[:,2]"),
    if ((sb.vel[0,2] <= 0.0) or (sb.vel[1,2] <= 0.0)):
        print(failed())
    else :
        print(passed())
    #compareNumpyArrays(sb.angvel[:,0:2:2], z2_2, "angvel[:,0:2:2]")
    #print("angvel[:,1]"),
    #if ((sb.angvel[0,1] >= 0.0) or (sb.angvel[1,1] >= 0.0)):
    #    print(failed())
    #else :
    #    print(passed())
    compareNumpyArrays(sb.angvel, z2_3, "angvel\t")
    #printKinematics(sb)
    #visualize(sb.sid, "energy")


    #'''
    print("# Twist")
    sb.x[0,:] = numpy.array((10.0, 10.0, 10.0))
    sb.x[1,:] = numpy.array((10.0+2.0*radii+d, 10.0, 10.0))
    sb.zeroKinematics()
    sb.initTemporal(total=0.2, file_dt=0.01)
    #sb.initTemporal(total=0.001, file_dt=0.00001)
    sb.angvel[1,0] = 1e-4
    Ekinrot0 = sb.energy("kin") + sb.energy("rot")
    sb.writebin(verbose=False)
    sb.run(verbose=False)
    sb.readlast(verbose=False)
    compareFloats(Ekinrot0, sb.energy("kin") + sb.energy("rot") + sb.energy("bondpot"), "Energy cons.")
    #compareNumpyArrays(sb.vel, z2_3, "vel\t")
    print("angvel[:,0]"),
    if ((sb.angvel[0,0] <= 0.0) or (sb.angvel[1,0] <= 0.0)):
        raise Exception("Failed")
        print(failed())
    else :
        print(passed())
    compareNumpyArrays(sb.angvel[:,1:2], z2_2, "angvel[:,1:2]")
    #printKinematics(sb)
    #visualize(sb.sid, "energy")
    

    #'''
    print("# Bend")
    sb.x[0,:] = numpy.array((10.0, 10.0, 10.0))
    sb.x[1,:] = numpy.array((10.0+2.0*radii+d, 10.0, 10.0))
    sb.zeroKinematics()
    sb.initTemporal(total=0.2, file_dt=0.01)
    sb.angvel[0,1] = -1e-4
    sb.angvel[1,1] = 1e-4
    Ekinrot0 = sb.energy("kin") + sb.energy("rot")
    sb.writebin(verbose=False)
    sb.run(verbose=False)
    sb.readlast(verbose=False)
    compareFloats(Ekinrot0, sb.energy("kin") + sb.energy("rot") + sb.energy("bondpot"), "Energy cons.")
    #compareNumpyArrays(sb.vel, z2_3, "vel\t")
    #compareNumpyArrays(sb.angvel[:,0:2:2], z2_2, "angvel[:,0:2:2]")
    print("angvel[:,1]"),
    if ((sb.angvel[0,1] == 0.0) or (sb.angvel[1,1] == 0.0)):
        raise Exception("Failed")
        print(failed())
    else :
        print(passed())
    #printKinematics(sb)
    #visualize(sb.sid, "energy")
    #'''

