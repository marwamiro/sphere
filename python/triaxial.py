#!/usr/bin/env python

# Import sphere functionality
from sphere import *

### EXPERIMENT SETUP ###
initialization = False
consolidation  = True
rendering      = True
plots	       = True

# Number of particles
np = 2e3

# Common simulation id
sim_id = "triaxial-test"

# Normal stress (sigma_3)
devs = 10e3


### INITIALIZATION ###

# New class
init = Spherebin(np = np, nd = 3, nw = 0, sid = sim_id + "-init")

# Save radii
init.generateRadii(radius_mean = 0.05)

# Use default params
init.defaultParams(gamma_n = 0.0, mu_s = 0.4, mu_d = 0.4)

# Initialize positions in random grid (also sets world size)
init.initRandomGridPos(gridnum = numpy.array([12, 12, 1000]), periodic = 0, contactmodel = 2)

# Set duration of simulation
init.initTemporal(total = 5.0)

if (initialization == True):
  # Write input file for sphere
  init.writebin()

  # Run sphere
  init.run()

  if (plots == True):
    # Make a graph of energies
    visualize(init.sid, "energy", savefig=True, outformat='png')

  #if (rendering == True):
    # Render images with raytracer
    #init.render(method = "angvel", max_val = 0.3, verbose = False)


### CONSOLIDATION ###

# New class
cons = Spherebin(np = np, nw = 1, sid = sim_id + "-cons")

# Read last output file of initialization step
lastf = status(sim_id + "-init")
cons.readbin("../output/" + sim_id + "-init.output{:0=5}.bin".format(lastf), verbose=False)

# Setup triaxial experiment
cons.triaxial(wvel = -cons.L[2]*0.01, deviatoric_stress = devs) # two percent of height per second

# Set duration of simulation
cons.initTemporal(total = 5.0)

cons.w_m[:] *= 0.001

if (consolidation == True):
  # Write input file for sphere
  cons.writebin()

  # Run sphere
  cons.run(dry=True) # show values, don't run
  cons.run() # run

  if (plots == True):
    # Make a graph of energies
    visualize(cons.sid, "energy", savefig=True, outformat='png')
    visualize(cons.sid, "walls", savefig=True, outformat='png')

  if (rendering == True):
    # Render images with raytracer
    cons.render(method = "pres", max_val = 1e4, verbose = False)

