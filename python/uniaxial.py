#!/usr/bin/env python

# Import sphere functionality
from sphere import *

### EXPERIMENT SETUP ###
consolidation  = True
rendering      = True
plots	       = True

# Number of particles
np = 2e3

# Common simulation id
sim_id = "uniaxial-test"


### CONSOLIDATION ###

# New class
cons = Spherebin(np = np, nw = 1, sid = sim_id + "-cons")

# Read last output file of initialization step
lastf = status("shear-test-init")
cons.readbin("../output/shear-test-init.output{:0=5}.bin".format(lastf), verbose=False)

# Setup consolidation experiment
cons.uniaxialStrainRate(wvel = -cons.L[2]*0.05) # five percent of height per second

# Set duration of simulation
cons.initTemporal(total = 3.0)
#cons.initTemporal(total = 0.0019, file_dt = 0.00009)
#cons.initTemporal(total = 0.0019, file_dt = 1e-6)
#cons.initTemporal(total = 0.19, file_dt = 0.019)

cons.w_m[0] *= 0.001



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

