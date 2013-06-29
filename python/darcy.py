#!/usr/bin/env python

# Import sphere functionality
from sphere import *
import sys

### EXPERIMENT SETUP ###
initialization = True
consolidation  = True
#shearing       = True
rendering      = False
#plots	       = False



# Number of particles
#np = 1e2
np = 1e4

# Common simulation id
sim_id = "darcy"

# Deviatoric stress [Pa]
#devs = 10e3
devslist = [10.0e3]

### INITIALIZATION ###

# New class
init = Spherebin(np = np, nd = 3, nw = 0, sid = sim_id + "-init")

# Save radii
init.generateRadii(radius_mean = 0.05)

# Use default params
init.defaultParams(mu_s = 0.4, mu_d = 0.4, nu = 8.9e-4)

# Initialize positions in random grid (also sets world size)
#init.initRandomGridPos(gridnum = numpy.array([9, 9, 1000]), periodic = 1, contactmodel = 2)
#init.initRandomGridPos(gridnum = numpy.array([32, 32, 1000]), periodic = 1, contactmodel = 2)
init.initRandomGridPos(gridnum = numpy.array([32, 32, 1000]), periodic = 1, contactmodel = 1)

# Bond ~30% of the particles
#init.random2bonds(spacing=0.1)

# Set duration of simulation
init.initTemporal(total = 7.0)
init.time_file_dt[0] = 0.05
#init.time_file_dt[0] = init.time_dt[0]*0.99
#init.time_total[0] = init.time_dt[0]*2.0
#init.initTemporal(total = 0.5)
#init.time_file_dt[0] = init.time_total[0]/5.0

#init.f_rho[2,2,4] = 5.0
#init.f_rho[6,6,10] = 1.1
#init.f_rho[:,:,-1] = 1.0001

if (initialization == True):

    # Write input file for sphere
    init.writebin()

    # Run sphere
    init.run(dry=True)
    init.run(darcyflow=True)


### CONSOLIDATION ###

for devs in devslist:
    # New class
    cons = Spherebin(np = init.np, nw = 1, sid = sim_id + "-cons-devs{}".format(devs))

    # Read last output file of initialization step
    lastf = status(sim_id + "-init")
    cons.readbin("../output/" + sim_id + "-init.output{:0=5}.bin".format(lastf), verbose=False)

    # Setup consolidation experiment
    cons.consolidate(deviatoric_stress = devs, periodic = init.periodic)


    # Set duration of simulation
    cons.initTemporal(total = 1.5)
    #cons.initTemporal(total = 0.0019, file_dt = 0.00009)
    #cons.initTemporal(total = 0.0019, file_dt = 1e-6)
    #cons.initTemporal(total = 0.19, file_dt = 0.019)

    cons.w_m[0] *= 0.001



    if (consolidation == True):
        # Write input file for sphere
        cons.writebin()

        # Run sphere
        cons.run(dry=True) # show values, don't run
        cons.run(darcyflow=True) # run

        if (plots == True):
            # Make a graph of energies
            visualize(cons.sid, "energy", savefig=True, outformat='png')
            visualize(cons.sid, "walls", savefig=True, outformat='png')

        if (rendering == True):
            # Render images with raytracer
            cons.render(method = "pres", max_val = 2.0*devs, verbose = False)

        project = cons.sid
        lastfile = status(cons.sid)
        sb = Spherebin()
        for i in range(lastfile+1):
            fn = "../output/{0}.output{1:0=5}.bin".format(project, i)
            sb.sid = project + ".output{:0=5}".format(i)
            sb.readbin(fn, verbose = False)
            for y in range(0,sb.num[1]):
                sb.plotFluidDensities(y = y)
                sb.plotFluidVelocities(y = y)
