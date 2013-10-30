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
devslist = [120e3]
#devs = 0

### INITIALIZATION ###

# New class
init = Spherebin(np = np, nd = 3, nw = 0, sid = sim_id + "-init")

# Save radii
init.generateRadii(radius_mean = 0.08)

# Use default params
init.defaultParams(gamma_n = 0.0, mu_s = 0.4, mu_d = 0.4)

# Initialize positions in random grid (also sets world size)
hcells = np**(1.0/3.0)*0.6
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
  #shear.shear(shear_strain_rate = 0.05, periodic = init.periodic)
  shear_strain_rate = 0.05

  ## Custom shear function
  # Find lowest and heighest point
  z_min = numpy.min(shear.x[:,2] - shear.radius)
  z_max = numpy.max(shear.x[:,2] + shear.radius)

  # the grid cell size is equal to the max. particle diameter
  cellsize = shear.L[0] / shear.num[0]

  # make grid one cell heigher to allow dilation
  shear.num[2] += 1
  shear.L[2] = shear.num[2] * cellsize

  # zero kinematics
  shear.zeroKinematics()

  # set friction coefficients
  shear.mu_s[0] = 0.5
  shear.mu_d[0] = 0.5

  # set the thickness of the horizons of fixed particles
  #fixheight = 2*cellsize
  #fixheight = cellsize

  # Fix horizontal velocity to 0.0 of lowermost particles
  d_max_below = numpy.max(shear.radius[numpy.nonzero(shear.x[:,2] <
      (z_max-z_min)*0.3)])*2.0
  #I = numpy.nonzero(shear.x[:,2] < (z_min + fixheight))
  I = numpy.nonzero(shear.x[:,2] < (z_min + d_max_below))
  shear.fixvel[I] = 1
  shear.angvel[I,0] = 0.0
  shear.angvel[I,1] = 0.0
  shear.angvel[I,2] = 0.0
  shear.vel[I,0] = 0.0 # x-dim
  shear.vel[I,1] = 0.0 # y-dim

  # Copy bottom fixed particles to top
  z_offset = z_max-z_min
  shearvel = (z_max-z_min)*shear_strain_rate
  for i in I[0]:
      x = shear.x[i,:] + numpy.array([0.0, 0.0, z_offset])
      vel = numpy.array([shearvel, 0.0, 0.0])
      shear.addParticle(x = x, radius = shear.radius[i], fixvel = 1, vel = vel)
      

  # Set wall viscosities to zero
  shear.gamma_wn[0] = 0.0
  shear.gamma_wt[0] = 0.0

  # Set wall friction coefficients to zero
  shear.mu_ws[0] = 0.0
  shear.mu_wd[0] = 0.0

  # Readjust top wall
  shear.adjustUpperWall()

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

