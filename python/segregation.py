#!/usr/bin/env python

# Import sphere functionality
from sphere import *

### EXPERIMENT SETUP ###
initialization = True
consolidation  = True
shearing       = True
rendering      = True
plots          = True

# Number of particles
np = 2e4

# Common simulation id
sim_id = "segregation"

# Deviatoric stress [Pa]
#devslist = [80e3, 10e3, 20e3, 40e3, 60e3, 120e3]
devslist = [80e3]
#devs = 0

### INITIALIZATION ###

# New class
init = Spherebin(np = np, nd = 3, nw = 0, sid = sim_id + "-init")

# Save radii
init.generateRadii(radius_mean = 0.08)

# Use default params
init.defaultParams(gamma_n = 0.0, mu_s = 0.3, mu_d = 0.3)

# Initialize positions in random grid (also sets world size)
hcells = np**(1.0/3.0)
init.initRandomGridPos(gridnum = numpy.array([hcells, hcells, 1e9]), periodic = 1, contactmodel = 2)

# Decrease size of top particles
fraction = 0.80
z_min = numpy.min(init.x[:,2] - init.radius)
z_max = numpy.max(init.x[:,2] + init.radius)
I = numpy.nonzero(init.x[:,2] > (z_max - (z_max-z_min)*fraction))
init.radius[I] = init.radius[I] * 0.5

# Set duration of simulation
init.initTemporal(total = 10.0)

if (initialization == True):
  # Write input file for sphere
  init.writebin()

  # Run sphere
  init.run(dry=True)
  init.run()

  if (plots == True):
    # Make a graph of energies
    visualize(init.sid, "energy", savefig=True, outformat='png')

  if (rendering == True):
    # Render images with raytracer
    init.render(method = "angvel", max_val = 0.3, verbose = False)


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
    cons.run() # run

    if (plots == True):
      # Make a graph of energies
      visualize(cons.sid, "energy", savefig=True, outformat='png')
      visualize(cons.sid, "walls", savefig=True, outformat='png')

    if (rendering == True):
      # Render images with raytracer
      cons.render(method = "pres", max_val = 2.0*devs, verbose = False)


### SHEARING ###

  # New class
  shear = Spherebin(np = cons.np, nw = cons.nw, sid = sim_id + "-shear-devs{}".format(devs))

  # Read last output file of initialization step
  lastf = status(sim_id + "-cons-devs{}".format(devs))
  shear.readbin("../output/" + sim_id + "-cons-devs{}.output{:0=5}.bin".format(devs, lastf), verbose = False)

  # Setup shear experiment
  shear.shear(shear_strain_rate = 0.05, periodic = init.periodic)

  # Set duration of simulation
  #shear.initTemporal(total = 20.0)
  shear.initTemporal(total = 400.0)

  if (shearing == True):
    # Write input file for sphere
    shear.writebin()

    # Run sphere
    shear.run(dry=True)
    shear.run()

    if (plots == True):
      # Make a graph of energies
      visualize(shear.sid, "energy", savefig=True, outformat='png')
      visualize(shear.sid, "shear", savefig=True, outformat='png')

    if (rendering == True):
      # Render images with raytracer
      shear.render(method = "pres", max_val = 2.0*devs, verbose = False)

