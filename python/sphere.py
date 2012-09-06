#!/usr/bin/env python2.7
import math
import numpy
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import subprocess

numpy.seterr(all='warn', over='raise')

# Class declarations
class Spherebin:
  """ Class containing all data SPHERE data.
      Contains functions for reading and writing
      binaries.
  """

  # Constructor - Initialize arrays
  def __init__(self, nd=3, np=1, nw=1):
    self.nd = numpy.ones(1, dtype=numpy.int32) * nd
    self.np = numpy.ones(1, dtype=numpy.uint32) * np

    # Time parameters
    self.time_dt         = numpy.zeros(1, dtype=numpy.float64)
    self.time_current    = numpy.zeros(1, dtype=numpy.float64)
    self.time_total      = numpy.zeros(1, dtype=numpy.float64)
    self.time_file_dt    = numpy.zeros(1, dtype=numpy.float64)
    self.time_step_count = numpy.zeros(1, dtype=numpy.uint32)

    # World dimensions and grid data
    self.origo   = numpy.zeros(self.nd, dtype=numpy.float64)
    self.L       = numpy.zeros(self.nd, dtype=numpy.float64)
    self.num     = numpy.zeros(self.nd, dtype=numpy.uint32)

    # Particle data
    self.x       = numpy.zeros(self.np*self.nd, dtype=numpy.float64).reshape(self.np,self.nd)
    self.vel     = numpy.zeros(self.np*self.nd, dtype=numpy.float64).reshape(self.np,self.nd)
    self.angvel  = numpy.zeros(self.np*self.nd, dtype=numpy.float64).reshape(self.np,self.nd)
    self.force   = numpy.zeros(self.np*self.nd, dtype=numpy.float64).reshape(self.np,self.nd)
    self.torque  = numpy.zeros(self.np*self.nd, dtype=numpy.float64).reshape(self.np,self.nd)
    self.fixvel  = numpy.zeros(self.np, dtype=numpy.float64)
    self.xsum    = numpy.zeros(self.np, dtype=numpy.float64)
    self.radius  = numpy.zeros(self.np, dtype=numpy.float64)
    self.rho     = numpy.zeros(self.np, dtype=numpy.float64)
    self.k_n     = numpy.zeros(self.np, dtype=numpy.float64)
    self.k_t     = numpy.zeros(self.np, dtype=numpy.float64)
    self.k_r	 = numpy.zeros(self.np, dtype=numpy.float64)
    self.gamma_n = numpy.zeros(self.np, dtype=numpy.float64)
    self.gamma_t = numpy.zeros(self.np, dtype=numpy.float64)
    self.gamma_r = numpy.zeros(self.np, dtype=numpy.float64)
    self.mu_s    = numpy.zeros(self.np, dtype=numpy.float64)
    self.mu_d    = numpy.zeros(self.np, dtype=numpy.float64)
    self.mu_r    = numpy.zeros(self.np, dtype=numpy.float64)
    self.es_dot  = numpy.zeros(self.np, dtype=numpy.float64)
    self.es	 = numpy.zeros(self.np, dtype=numpy.float64)
    self.p	 = numpy.zeros(self.np, dtype=numpy.float64)

    self.bonds   = numpy.ones(self.np*4, dtype=numpy.uint32).reshape(self.np,4) * self.np

    # Constant, single-value physical parameters
    self.globalparams = numpy.zeros(1, dtype=numpy.int32)
    self.g            = numpy.zeros(self.nd, dtype=numpy.float64)
    self.kappa        = numpy.zeros(1, dtype=numpy.float64)
    self.db           = numpy.zeros(1, dtype=numpy.float64)
    self.V_b          = numpy.zeros(1, dtype=numpy.float64)
    self.shearmodel   = numpy.zeros(1, dtype=numpy.uint32)

    # Wall data
    self.nw 	 = numpy.ones(1, dtype=numpy.uint32) * nw
    self.wmode   = numpy.zeros(self.nw, dtype=numpy.int32)
    self.w_n     = numpy.zeros(self.nw*self.nd, dtype=numpy.float64).reshape(self.nw,self.nd)
    self.w_x     = numpy.zeros(self.nw, dtype=numpy.float64)
    self.w_m     = numpy.zeros(self.nw, dtype=numpy.float64)
    self.w_vel   = numpy.zeros(self.nw, dtype=numpy.float64)
    self.w_force = numpy.zeros(self.nw, dtype=numpy.float64)
    self.w_devs  = numpy.zeros(self.nw, dtype=numpy.float64)
    
    # x- and y-boundary behavior
    self.periodic = numpy.zeros(1, dtype=numpy.uint32)
    self.gamma_wn = numpy.zeros(1, dtype=numpy.float64)
    self.gamma_ws = numpy.zeros(1, dtype=numpy.float64)
    self.gamma_wr = numpy.zeros(1, dtype=numpy.float64)


  # Read binary data
  def readbin(self, targetbin, verbose = True):
    """ Reads a target SPHERE binary file and returns data.
    """
    fh = None
    try:
      if (verbose == True):
	print("Input file: {0}".format(targetbin))
      fh = open(targetbin, "rb")

      # Read the number of dimensions and particles
      self.nd = numpy.fromfile(fh, dtype=numpy.int32, count=1)
      self.np = numpy.fromfile(fh, dtype=numpy.uint32, count=1)
       
      # Read the time variables
      self.time_dt 	   = numpy.fromfile(fh, dtype=numpy.float64, count=1)
      self.time_current    = numpy.fromfile(fh, dtype=numpy.float64, count=1)
      self.time_total 	   = numpy.fromfile(fh, dtype=numpy.float64, count=1)
      self.time_file_dt    = numpy.fromfile(fh, dtype=numpy.float64, count=1)
      self.time_step_count = numpy.fromfile(fh, dtype=numpy.uint32, count=1)

      # Allocate array memory for particles
      self.x       = numpy.zeros(self.np*self.nd, dtype=numpy.float64).reshape(self.np,self.nd)
      self.vel     = numpy.zeros(self.np*self.nd, dtype=numpy.float64).reshape(self.np,self.nd)
      self.angvel  = numpy.zeros(self.np*self.nd, dtype=numpy.float64).reshape(self.np,self.nd)
      self.force   = numpy.zeros(self.np*self.nd, dtype=numpy.float64).reshape(self.np,self.nd)
      self.torque  = numpy.zeros(self.np*self.nd, dtype=numpy.float64).reshape(self.np,self.nd)
      self.fixvel  = numpy.zeros(self.np, dtype=numpy.float64)
      self.xsum    = numpy.zeros(self.np, dtype=numpy.float64)
      self.radius  = numpy.zeros(self.np, dtype=numpy.float64)
      self.rho     = numpy.zeros(self.np, dtype=numpy.float64)
      self.k_n     = numpy.zeros(self.np, dtype=numpy.float64)
      self.k_t     = numpy.zeros(self.np, dtype=numpy.float64)
      self.k_r	   = numpy.zeros(self.np, dtype=numpy.float64)
      self.gamma_n = numpy.zeros(self.np, dtype=numpy.float64)
      self.gamma_t = numpy.zeros(self.np, dtype=numpy.float64)
      self.gamma_r = numpy.zeros(self.np, dtype=numpy.float64)
      self.mu_s    = numpy.zeros(self.np, dtype=numpy.float64)
      self.mu_d    = numpy.zeros(self.np, dtype=numpy.float64)
      self.mu_r    = numpy.zeros(self.np, dtype=numpy.float64)
      self.es_dot  = numpy.zeros(self.np, dtype=numpy.float64)
      self.es	   = numpy.zeros(self.np, dtype=numpy.float64)
      self.p	   = numpy.zeros(self.np, dtype=numpy.float64)
      self.bonds   = numpy.zeros(self.np, dtype=numpy.float64)

      # Read remaining data from binary
      self.origo   = numpy.fromfile(fh, dtype=numpy.float64, count=self.nd)
      self.L       = numpy.fromfile(fh, dtype=numpy.float64, count=self.nd)
      self.num     = numpy.fromfile(fh, dtype=numpy.uint32, count=self.nd)
  
      # Per-particle vectors
      for j in range(self.np):
        for i in range(self.nd):
  	  self.x[j,i]      = numpy.fromfile(fh, dtype=numpy.float64, count=1)
	  self.vel[j,i]    = numpy.fromfile(fh, dtype=numpy.float64, count=1)
	  self.angvel[j,i] = numpy.fromfile(fh, dtype=numpy.float64, count=1)
	  self.force[j,i]  = numpy.fromfile(fh, dtype=numpy.float64, count=1)
	  self.torque[j,i] = numpy.fromfile(fh, dtype=numpy.float64, count=1)
 
      # Per-particle single-value parameters
      for j in range(self.np):
	self.fixvel[j]  = numpy.fromfile(fh, dtype=numpy.float64, count=1)
	self.xsum[j]    = numpy.fromfile(fh, dtype=numpy.float64, count=1)
        self.radius[j]  = numpy.fromfile(fh, dtype=numpy.float64, count=1)
        self.rho[j]     = numpy.fromfile(fh, dtype=numpy.float64, count=1)
        self.k_n[j]     = numpy.fromfile(fh, dtype=numpy.float64, count=1)
        self.k_t[j]     = numpy.fromfile(fh, dtype=numpy.float64, count=1)
	self.k_r[j]     = numpy.fromfile(fh, dtype=numpy.float64, count=1)
	self.gamma_n[j] = numpy.fromfile(fh, dtype=numpy.float64, count=1)
	self.gamma_t[j] = numpy.fromfile(fh, dtype=numpy.float64, count=1)
	self.gamma_r[j] = numpy.fromfile(fh, dtype=numpy.float64, count=1)
        self.mu_s[j]    = numpy.fromfile(fh, dtype=numpy.float64, count=1) 
        self.mu_d[j]    = numpy.fromfile(fh, dtype=numpy.float64, count=1) 
        self.mu_r[j]    = numpy.fromfile(fh, dtype=numpy.float64, count=1)
        self.es_dot[j]  = numpy.fromfile(fh, dtype=numpy.float64, count=1)
        self.es[j]      = numpy.fromfile(fh, dtype=numpy.float64, count=1)
        self.p[j]       = numpy.fromfile(fh, dtype=numpy.float64, count=1)

      # Constant, single-value physical parameters
      self.globalparams = numpy.fromfile(fh, dtype=numpy.int32, count=1)
      self.g            = numpy.fromfile(fh, dtype=numpy.float64, count=self.nd)
      self.kappa        = numpy.fromfile(fh, dtype=numpy.float64, count=1)
      self.db           = numpy.fromfile(fh, dtype=numpy.float64, count=1)
      self.V_b          = numpy.fromfile(fh, dtype=numpy.float64, count=1)
      self.shearmodel   = numpy.fromfile(fh, dtype=numpy.uint32, count=1)

      # Wall data
      self.nw 	   = numpy.fromfile(fh, dtype=numpy.uint32, count=1)
      self.wmode   = numpy.zeros(self.nw, dtype=numpy.int32)
      self.w_n     = numpy.zeros(self.nw*self.nd, dtype=numpy.float64).reshape(self.nw,self.nd)
      self.w_x     = numpy.zeros(self.nw, dtype=numpy.float64)
      self.w_m     = numpy.zeros(self.nw, dtype=numpy.float64)
      self.w_vel   = numpy.zeros(self.nw, dtype=numpy.float64)
      self.w_force = numpy.zeros(self.nw, dtype=numpy.float64)
      self.w_devs  = numpy.zeros(self.nw, dtype=numpy.float64)

      for j in range(self.nw):
	self.wmode[j] = numpy.fromfile(fh, dtype=numpy.int32, count=1)
        for i in range(self.nd):
  	  self.w_n[j,i] = numpy.fromfile(fh, dtype=numpy.float64, count=1)

        self.w_x[j]     = numpy.fromfile(fh, dtype=numpy.float64, count=1)
        self.w_m[j]     = numpy.fromfile(fh, dtype=numpy.float64, count=1)
        self.w_vel[j]   = numpy.fromfile(fh, dtype=numpy.float64, count=1)
        self.w_force[j] = numpy.fromfile(fh, dtype=numpy.float64, count=1)
        self.w_devs[j]  = numpy.fromfile(fh, dtype=numpy.float64, count=1)
    
      # x- and y-boundary behavior
      self.periodic = numpy.fromfile(fh, dtype=numpy.int32, count=1)
      self.gamma_wn = numpy.fromfile(fh, dtype=numpy.float64, count=1)
      self.gamma_ws = numpy.fromfile(fh, dtype=numpy.float64, count=1)
      self.gamma_wr = numpy.fromfile(fh, dtype=numpy.float64, count=1)

      # Read interparticle bond list
      self.bonds = numpy.zeros(self.np*4, dtype=numpy.uint32).reshape(self.np,4)
      for j in range(self.np):
	for i in range(4):
	  self.bonds[j,i] = numpy.fromfile(fh, dtype=numpy.uint32, count=1)

      fh.close()
      
    finally:
      if fh is not None:
        fh.close()

  # Write binary data
  def writebin(self, targetbin):
    """ Reads a target SPHERE binary file and returns data.
    """
    fh = None
    try:
      print("Output file: {0}".format(targetbin))
      fh = open(targetbin, "wb")

      # Write the number of dimensions and particles
      fh.write(self.nd.astype(numpy.int32))
      fh.write(self.np.astype(numpy.uint32))
       
      # Write the time variables
      fh.write(self.time_dt.astype(numpy.float64))
      fh.write(self.time_current.astype(numpy.float64))
      fh.write(self.time_total.astype(numpy.float64))
      fh.write(self.time_file_dt.astype(numpy.float64))
      fh.write(self.time_step_count.astype(numpy.uint32))

      # Read remaining data from binary
      fh.write(self.origo.astype(numpy.float64))
      fh.write(self.L.astype(numpy.float64))
      fh.write(self.num.astype(numpy.uint32))
  
      # Per-particle vectors
      for j in range(self.np):
        for i in range(self.nd):
	  fh.write(self.x[j,i].astype(numpy.float64))
	  fh.write(self.vel[j,i].astype(numpy.float64))
	  fh.write(self.angvel[j,i].astype(numpy.float64))
	  fh.write(self.force[j,i].astype(numpy.float64))
	  fh.write(self.torque[j,i].astype(numpy.float64))
 
      # Per-particle single-value parameters
      for j in range(self.np):
	fh.write(self.fixvel[j].astype(numpy.float64))
	fh.write(self.xsum[j].astype(numpy.float64))
        fh.write(self.radius[j].astype(numpy.float64))
        fh.write(self.rho[j].astype(numpy.float64))
        fh.write(self.k_n[j].astype(numpy.float64))
        fh.write(self.k_t[j].astype(numpy.float64))
	fh.write(self.k_r[j].astype(numpy.float64))
	fh.write(self.gamma_n[j].astype(numpy.float64))
	fh.write(self.gamma_t[j].astype(numpy.float64))
	fh.write(self.gamma_r[j].astype(numpy.float64))
        fh.write(self.mu_s[j].astype(numpy.float64))
        fh.write(self.mu_d[j].astype(numpy.float64))
        fh.write(self.mu_r[j].astype(numpy.float64))
        fh.write(self.es_dot[j].astype(numpy.float64))
        fh.write(self.es[j].astype(numpy.float64))
        fh.write(self.p[j].astype(numpy.float64))

      # Constant, single-value physical parameters
      fh.write(self.globalparams.astype(numpy.int32))
      fh.write(self.g.astype(numpy.float64))
      fh.write(self.kappa.astype(numpy.float64))
      fh.write(self.db.astype(numpy.float64))
      fh.write(self.V_b.astype(numpy.float64))
      fh.write(self.shearmodel.astype(numpy.uint32))

      fh.write(self.nw.astype(numpy.uint32))
      for j in range(self.nw):
	fh.write(self.wmode[j].astype(numpy.int32))
        for i in range(self.nd):
  	  fh.write(self.w_n[j,i].astype(numpy.float64))

        fh.write(self.w_x[j].astype(numpy.float64))
        fh.write(self.w_m[j].astype(numpy.float64))
        fh.write(self.w_vel[j].astype(numpy.float64))
        fh.write(self.w_force[j].astype(numpy.float64))
        fh.write(self.w_devs[j].astype(numpy.float64))
    
      # x- and y-boundary behavior
      fh.write(self.periodic.astype(numpy.uint32))
      fh.write(self.gamma_wn.astype(numpy.float64))
      fh.write(self.gamma_ws.astype(numpy.float64))
      fh.write(self.gamma_wr.astype(numpy.float64))

      # Read interparticle bond list
      for j in range(self.np):
	for i in range(4):
	  fh.write(self.bonds[j,i].astype(numpy.uint32))

      fh.close()
      
    finally:
      if fh is not None:
        fh.close()

  def generateRadii(self, psd = 'logn',
      			  radius_mean = 440e-6,
			  radius_variance = 8.8e-9,
			  histogram = 1):
    """ Draw random particle radii from the selected probability distribution.
    	Specify mean radius and variance. The variance should be kept at a
	very low value.
    """

    if psd == 'logn': # Log-normal probability distribution
      mu = math.log((radius_mean**2)/math.sqrt(radius_variance+radius_mean**2))
      sigma = math.sqrt(math.log(radius_variance/(radius_mean**2)+1))
      self.radius = numpy.random.lognormal(mu, sigma, self.np)
    if psd == 'uni':  # Uniform distribution
      radius_min = radius_mean - radius_variance
      radius_max = radius_mean + radius_variance
      self.radius = numpy.random.uniform(radius_min, radius_max, self.np)

    # Show radii as histogram
    if histogram == 1:
      fig = plt.figure(figsize=(15,10), dpi=300)
      figtitle = 'Particle size distribution, {0} particles'.format(self.np[0])
      fig.text(0.5,0.95,figtitle,horizontalalignment='center',fontproperties=FontProperties(size=18))
      bins = 20
      # Create histogram
      plt.hist(self.radius, bins)
      # Plot
      plt.xlabel('Radii [m]')
      plt.ylabel('Count')
      plt.axis('tight')
      fig.savefig('psd.png')
      fig.clf()
 

  # Initialize particle positions to completely random, non-overlapping configuration.
  # This method is very compute intensive at high particle numbers.
  def initRandomPos(self, g = numpy.array([0.0, 0.0, -9.80665]), 
      		          gridnum = numpy.array([12, 12, 36]),
		          periodic = 1,
		          shearmodel = 2):
    """ Initialize particle positions in loose, cubic configuration.
        Radii must be set beforehand.
	xynum is the number of rows in both x- and y- directions.
    """
    self.g = g
    self.periodic[0] = periodic

    # Calculate cells in grid
    self.num = gridnum

    # World size
    r_max = numpy.amax(self.radius)
    cellsize = 2 * r_max
    self.L = self.num * cellsize

    # Particle positions randomly distributed without overlap
    for i in range(self.np):
      overlaps = True
      while overlaps == True:
	overlaps = False

	# Draw random position
	for d in range(self.nd):
	  self.x[i,d] = (self.L[d] - self.origo[d] - 2*r_max) \
                        * numpy.random.random_sample() \
		        + self.origo[d] + r_max
        
	# Check other particles for overlaps
	for j in range(i-1):
	  delta = self.x[i] - self.x[j]
	  delta_len = math.sqrt(numpy.dot(delta,delta)) \
	              - (self.radius[i] + self.radius[j])
	  if (delta_len < 0.0):
	    overlaps = True
      print "\rFinding non-overlapping particle positions, {0} % complete".format(numpy.ceil(i/self.np[0]*100)),
   
    print " "
    self.shearmodel[0] = shearmodel

    # Initialize upper wall
    self.wmode[0] = 0	# 0: devs, 1: vel
    self.w_n[0,2] = -1.0
    self.w_x[0] = self.L[2]
    self.w_m[0] = self.rho[0] * self.np * math.pi * r_max**3
    self.w_vel[0] = 0.0
    self.w_force[0] = 0.0
    self.w_devs[0] = 0.0
    #self.nw[0] = numpy.ones(1, dtype=numpy.uint32) * 1
    self.nw = numpy.ones(1, dtype=numpy.uint32) * 1

  # Initialize particle positions to regular, grid-like, non-overlapping configuration
  def initGridPos(self, g = numpy.array([0.0, 0.0, -9.80665]), 
      		        gridnum = numpy.array([12, 12, 36]),
		        periodic = 1,
		        shearmodel = 2):
    """ Initialize particle positions in loose, cubic configuration.
        Radii must be set beforehand.
	xynum is the number of rows in both x- and y- directions.
    """
    self.g = g
    self.periodic[0] = periodic

    # Calculate cells in grid
    self.num = gridnum

    # World size
    r_max = numpy.amax(self.radius)
    cellsize = 2.1 * r_max
    self.L = self.num * cellsize

    # Check whether there are enough grid cells 
    if ((self.num[0]*self.num[1]*self.num[2]-(2**3)) < self.np):
      print "Error! The grid is not sufficiently large."
      raise NameError('Error! The grid is not sufficiently large.')

    gridpos = numpy.zeros(self.nd, dtype=numpy.uint32)

    # Make sure grid is sufficiently large if every second level is moved
    if (self.periodic[0] == 1):
      self.num[0] -= 1
      self.num[1] -= 1
      
      # Check whether there are enough grid cells 
      if ((self.num[0]*self.num[1]*self.num[2]-(2*3*3)) < self.np):
        print "Error! The grid is not sufficiently large."
        raise NameError('Error! The grid is not sufficiently large.')


    # Particle positions randomly distributed without overlap
    for i in range(self.np):

      # Find position in 3d mesh from linear index
      gridpos[0] = (i % (self.num[0]))
      gridpos[1] = numpy.floor(i/(self.num[0])) % (self.num[0])
      gridpos[2] = numpy.floor(i/((self.num[0])*(self.num[1]))) #\
	           #% ((self.num[0])*(self.num[1]))
	
      for d in range(self.nd):
        self.x[i,d] = gridpos[d] * cellsize + 0.5*cellsize

      if (self.periodic[0] == 1): # Allow pushing every 2.nd level out of lateral boundaries
        # Offset every second level
        if (gridpos[2] % 2):
	  self.x[i,0] += 0.5*cellsize
	  self.x[i,1] += 0.5*cellsize

    self.shearmodel[0] = shearmodel

    # Readjust grid to correct size
    if (self.periodic[0] == 1):
      self.num[0] += 1
      self.num[1] += 1

    # Initialize upper wall
    self.wmode[0] = 0
    self.w_n[0,2] = -1.0
    self.w_x[0] = self.L[2]
    self.w_m[0] = self.rho[0] * self.np * math.pi * r_max**3
    self.w_vel[0] = 0.0
    self.w_force[0] = 0.0
    self.w_devs[0] = 0.0
    self.nw = numpy.ones(1, dtype=numpy.uint32) * 1


  # Initialize particle positions to non-overlapping configuration
  # in grid, with a certain element of randomness
  def initRandomGridPos(self, g = numpy.array([0.0, 0.0, -9.80665]), 
      		              gridnum = numpy.array([12, 12, 32]),
			      periodic = 1,
			      shearmodel = 2):
    """ Initialize particle positions in loose, cubic configuration.
        Radii must be set beforehand.
	xynum is the number of rows in both x- and y- directions.
    """
    self.g = g
    self.periodic[0] = periodic

    # Calculate cells in grid
    coarsegrid = numpy.floor(gridnum/2) 

    # World size 
    r_max = numpy.amax(self.radius)
    cellsize = 2.1 * r_max * 2 # Cells in grid 2*size to make space for random offset

    # Check whether there are enough grid cells 
    if (((coarsegrid[0]-1)*(coarsegrid[1]-1)*(coarsegrid[2]-1)) < self.np):
      print "Error! The grid is not sufficiently large."
      raise NameError('Error! The grid is not sufficiently large.')

    gridpos = numpy.zeros(self.nd, dtype=numpy.uint32)

    # Particle positions randomly distributed without overlap
    for i in range(self.np):

      # Find position in 3d mesh from linear index
      gridpos[0] = (i % (coarsegrid[0]))
      gridpos[1] = numpy.floor(i/(coarsegrid[0])) % (coarsegrid[0])
      gridpos[2] = numpy.floor(i/((coarsegrid[0])*(coarsegrid[1])))
	
      # Place particles in grid structure, and randomly adjust the positions
      # within the oversized cells (uniform distribution)
      for d in range(self.nd):
	r = self.radius[i]*1.05
        self.x[i,d] = gridpos[d] * cellsize \
		      + ((cellsize-r) - r) * numpy.random.random_sample() + r

    self.shearmodel[0] = shearmodel

    # Calculate new grid with cell size equal to max. particle diameter
    x_max = numpy.max(self.x[:,0] + self.radius)
    y_max = numpy.max(self.x[:,1] + self.radius)
    z_max = numpy.max(self.x[:,2] + self.radius)
    cellsize = 2.1 * r_max
    # Adjust size of world
    self.num[0] = numpy.ceil(x_max/cellsize)
    self.num[1] = numpy.ceil(y_max/cellsize)
    self.num[2] = numpy.ceil(z_max/cellsize)
    self.L = self.num * cellsize

    # Initialize upper wall
    self.wmode[0] = 0
    self.w_n[0,2] = -1.0
    self.w_x[0] = self.L[2]
    self.w_m[0] = self.rho[0] * self.np * math.pi * r_max**3
    self.w_vel[0] = 0.0
    self.w_force[0] = 0.0
    self.w_devs[0] = 0.0
    self.nw = numpy.ones(1, dtype=numpy.uint32) * 1

  # Adjust grid and upper wall for consolidation under deviatoric stress
  def consolidate(self, deviatoric_stress = 10e3, 
      			periodic = 1):
    """ Setup consolidation experiment. Specify the upper wall 
        deviatoric stress in Pascal, default value is 10 kPa.
    """

    # Compute new grid, scaled to fit max. and min. particle positions
    z_min = numpy.min(self.x[:,2] - self.radius)
    z_max = numpy.max(self.x[:,2] + self.radius)
    cellsize = self.L[0] / self.num[0]
    z_adjust = 1.1	# Overheightening of grid. 1.0 = no overheightening
    self.num[2] = numpy.ceil((z_max-z_min)*z_adjust/cellsize)
    self.L[2] = (z_max-z_min)*z_adjust

    # Initialize upper wall
    self.wmode[0] = 0
    self.w_n[0,2] = -1.0
    self.w_x[0] = self.L[2]
    self.w_m[0] = self.rho[0] * self.np * math.pi * (cellsize/2.0)**3
    self.w_vel[0] = 0.0
    self.w_force[0] = 0.0
    self.w_devs[0] = deviatoric_stress
    self.nw = numpy.ones(1, dtype=numpy.uint32) * 1

  # Adjust grid and upper wall for consolidation under fixed upper wall velocity
  def uniaxialStrainRate(self, wvel = -0.001,
      			periodic = 1):
    """ Setup consolidation experiment. Specify the upper wall 
        deviatoric stress in Pascal, default value is 10 kPa.
    """

    # Compute new grid, scaled to fit max. and min. particle positions
    z_min = numpy.min(self.x[:,2] - self.radius)
    z_max = numpy.max(self.x[:,2] + self.radius)
    cellsize = self.L[0] / self.num[0]
    z_adjust = 1.1	# Overheightening of grid. 1.0 = no overheightening
    self.num[2] = numpy.ceil((z_max-z_min)*z_adjust/cellsize)
    self.L[2] = (z_max-z_min)*z_adjust

    # Initialize upper wall
    self.wmode[0] = 1
    self.w_n[0,2] = -1.0
    self.w_x[0] = self.L[2]
    self.w_m[0] = self.rho[0] * self.np * math.pi * (cellsize/2.0)**3
    self.w_vel[0] = wvel
    self.w_force[0] = 0.0
    self.nw = numpy.ones(1, dtype=numpy.uint32) * 1


  # Adjust grid and upper wall for shear, and fix boundary particle velocities
  def shear(self, deviatoric_stress = 10e3, 
      		  shear_strain_rate = 1,
      		  periodic = 1):
    """ Setup shear experiment. Specify the upper wall 
        deviatoric stress in Pascal, default value is 10 kPa.
	The shear strain rate is the shear length divided by the
	initial height per second.
    """

    # Compute new grid, scaled to fit max. and min. particle positions
    z_min = numpy.min(self.x[:,2] - self.radius)
    z_max = numpy.max(self.x[:,2] + self.radius)
    cellsize = self.L[0] / self.num[0]
    z_adjust = 1.3	# Overheightening of grid. 1.0 = no overheightening
    self.num[2] = numpy.ceil((z_max-z_min)*z_adjust/cellsize)
    self.L[2] = (z_max-z_min)*z_adjust

    # Initialize upper wall
    self.w_devs[0] = deviatoric_stress

    # Zero kinematics
    self.vel     = numpy.zeros(self.np*self.nd, dtype=numpy.float64).reshape(self.np,self.nd)
    self.angvel  = numpy.zeros(self.np*self.nd, dtype=numpy.float64).reshape(self.np,self.nd)

    fixheight = 2*cellsize
    #fixheight = cellsize
 
    # Fix horizontal velocity to 0.0 of lowermost particles
    I = numpy.nonzero(self.x[:,2] < (z_min + fixheight)) # Find indices of lowermost 10%
    self.fixvel[I] = 1
    self.angvel[I,0] = 0.0
    self.angvel[I,1] = 0.0
    self.angvel[I,2] = 0.0
    self.vel[I,0] = 0.0 # x-dim
    self.vel[I,1] = 0.0 # y-dim

    # Fix horizontal velocity to specific value of uppermost particles
    I = numpy.nonzero(self.x[:,2] > (z_max - fixheight)) # Find indices of lowermost 10%
    self.fixvel[I] = 1
    self.angvel[I,0] = 0.0
    self.angvel[I,1] = 0.0
    self.angvel[I,2] = 0.0
    self.vel[I,0] = (z_max-z_min)*shear_strain_rate
    self.vel[I,1] = 0.0 # y-dim

    # Zero x-axis displacement
    self.xsum = numpy.zeros(self.np, dtype=numpy.float64)

    # Set wall viscosities to zero
    self.gamma_wn[0] = 0.0
    self.gamma_ws[0] = 0.0
    self.gamma_wr[0] = 0.0


 
  def initTemporal(self, total,
      			 current = 0.0,
			 file_dt = 0.01,
			 step_count = 0):
    """ Set temporal parameters for the simulation.
        Particle radii and physical parameters need to be set
	prior to these.
    """
    
    r_min = numpy.amin(self.radius)

    # Computational time step (O'Sullivan et al, 2003)
    #self.time_dt[0] = 0.17 * math.sqrt((4.0/3.0 * math.pi * r_min**3 * self.rho[0]) / numpy.amax([self.k_n[:], self.k_t[:]]) )
    self.time_dt[0] = 0.12 * math.sqrt((4.0/3.0 * math.pi * r_min**3 * self.rho[0]) / numpy.amax([self.k_n[:], self.k_t[:]]) )
    
    # Time at start
    self.time_current[0] = current
    self.time_total[0] = total
    self.time_file_dt[0] = file_dt
    self.time_step_count[0] = 0

  def defaultParams(self, ang_s = 20,
      			  ang_d = 15,
      			  ang_r = 0,
			  rho = 2660,
			  k_n = 1e9,
			  k_t = 1e9,
			  k_r = 4e6,
			  gamma_n = 1e3,
			  gamma_t = 1e3,
			  gamma_r = 2e3,
			  gamma_wn = 1e3,
			  gamma_ws = 1e3,
			  gamma_wr = 2e3,
			  capillaryCohesion = 0):
    """ Initialize particle parameters to default values.
        Radii must be set prior to calling this function.
    """
    # Particle material density, kg/m^3
    self.rho = numpy.ones(self.np, dtype=numpy.float64) * rho

    
    ### Dry granular material parameters

    # Contact normal elastic stiffness, N/m
    self.k_n = numpy.ones(self.np, dtype=numpy.float64) * k_n

    # Contact shear elastic stiffness (for shearmodel = 2), N/m
    self.k_t = numpy.ones(self.np, dtype=numpy.float64) * k_t

    # Contact rolling elastic stiffness (for shearmodel = 2), N/m
    self.k_r = numpy.ones(self.np, dtype=numpy.float64) * k_r

    # Contact normal viscosity. Critical damping: 2*sqrt(m*k_n).
    # Normal force component elastic if nu = 0.0.
    #self.gamma_n = numpy.ones(self.np, dtype=numpy.float64) \
    #	      * nu_frac * 2.0 * math.sqrt(4.0/3.0 * math.pi * numpy.amin(self.radius)**3 \
    #	      * self.rho[0] * self.k_n[0])
    self.gamma_n = numpy.ones(self.np, dtype=numpy.float64) * gamma_n
		      
    # Contact shear viscosity, Ns/m
    self.gamma_t = numpy.ones(self.np, dtype=numpy.float64) * gamma_t

    # Contact rolling viscosity, Ns/m?
    self.gamma_r = numpy.ones(self.np, dtype=numpy.float64) * gamma_r

    # Contact static shear friction coefficient
    self.mu_s = numpy.ones(self.np, dtype=numpy.float64) * numpy.tan(numpy.radians(ang_s))

    # Contact dynamic shear friction coefficient
    self.mu_d = numpy.ones(self.np, dtype=numpy.float64) * numpy.tan(numpy.radians(ang_d))

    # Contact rolling friction coefficient
    self.mu_r = numpy.ones(self.np, dtype=numpy.float64) * numpy.tan(numpy.radians(ang_r))

    # Global parameters
    # if 1 = all particles have the same values for the physical parameters
    self.globalparams[0] = 1

    # Wall viscosities
    self.gamma_wn[0] = gamma_wn # normal
    self.gamma_ws[0] = gamma_ws # sliding
    self.gamma_wr[0] = gamma_wr # rolling


    ### Parameters related to capillary bonds

    # Wettability, 0=perfect
    theta = 0.0;
    if (capillaryCohesion == 1):
      self.kappa[0] = 2.0 * math.pi * gamma_t * numpy.cos(theta)  # Prefactor
      self.V_b[0] = 1e-12  # Liquid volume at bond
    else :
      self.kappa[0] = 0.0;  # Zero capillary force
      self.V_b[0] = 0.0;    # Zero liquid volume at bond

    # Debonding distance
    self.db[0] = (1.0 + theta/2.0) * self.V_b**(1.0/3.0)

  def energy(self, method):
    """ Calculate the sum of the energy components of all particles.
    """

    if method == 'pot':
      m = numpy.ones(self.np) * 4.0/3.0 * math.pi * self.radius**3 * self.rho
      return numpy.sum(m * math.sqrt(numpy.dot(self.g,self.g)) * self.x[:,2])

    elif method == 'kin':
      m = numpy.ones(self.np) * 4.0/3.0 * math.pi * self.radius**3 * self.rho
      esum = 0.0
      for i in range(self.np):
	esum += 0.5 * m[i] * math.sqrt(numpy.dot(self.vel[i,:],self.vel[i,:]))**2
      return esum

    elif method == 'rot':
      m = numpy.ones(self.np) * 4.0/3.0 * math.pi * self.radius**3 * self.rho
      esum = 0.0
      for i in range(self.np):
        esum += 0.5 * 2.0/5.0 * m[i] * self.radius[i]**2 \
	       * math.sqrt(numpy.dot(self.angvel[i,:],self.angvel[i,:]))**2
      return esum

    elif method == 'shear':
      return numpy.sum(self.es)

    elif method == 'shearrate':
      return numpy.sum(self.es_dot)

  def voidRatio(self):
    """ Return the current void ratio
    """

    # Find the bulk volume
    V_t = (self.L[0] - self.origo[0]) \
    	 *(self.L[1] - self.origo[1]) \
	 *(self.w_x[0] - self.origo[2])

    # Find the volume of solids
    V_s = numpy.sum(4.0/3.0 * math.pi * self.radius**3)

    # Return the void ratio
    e = (V_t - V_s)/V_s
    return e


  def porosity(self, lower_corner, 
      		     upper_corner, 
		     grid = numpy.array([10,10,10], int), 
		     precisionfactor = 10):
    """ Calculate the porosity inside each grid cell.
	Specify the lower and upper corners of the volume to evaluate.
        A good starting point for the grid vector is self.num.
	The precision factor determines how much precise the grid porosity is.
	A larger value will result in better precision, but more computations O(n^3).
    """

    # Cell size length
    csl = numpy.array([(upper_corner[0]-lower_corner[0]) / grid[0], \
	  	       (upper_corner[1]-lower_corner[1]) / grid[1], \
		       (upper_corner[2]-lower_corner[2]) / grid[2] ])

    # Create 3d vector of porosity values
    porosity_grid = numpy.ones((grid[0], grid[1], grid[2]), float) * csl[0]*csl[1]*csl[2]

    # Fine grid, to be interpolated to porosity_grid. The fine cells are
    # assumed to be either completely solid- (True), or void space (False).
    fine_grid = numpy.zeros((grid[0]*precisionfactor, \
			     grid[1]*precisionfactor, \
			     grid[2]*precisionfactor), bool)

    # Side length of fine grid cells
    csl_fine = numpy.array([(upper_corner[0]-lower_corner[0]) / (grid[0]*precisionfactor), \
			    (upper_corner[1]-lower_corner[1]) / (grid[1]*precisionfactor), \
			    (upper_corner[2]-lower_corner[2]) / (grid[2]*precisionfactor) ])
      
    # Volume of fine grid vells
    Vc_fine = csl_fine[0] * csl_fine[1] * csl_fine[2]

    # Iterate over fine grid cells
    for ix in range(grid[0]*precisionfactor):
      for iy in range(grid[1]*precisionfactor):
	for iz in range(grid[2]*precisionfactor):

	  # Coordinates of cell centre
	  cpos = numpy.array([ix*csl_fine[0] + 0.5*csl_fine[0], \
	  		      iy*csl_fine[1] + 0.5*csl_fine[1], \
			      iz*csl_fine[2] + 0.5*csl_fine[2] ])


	  # Find out if the cell centre lies within a particle
	  for i in range(self.np):
	    p = self.x[i,:] 	# Particle position
	    r = self.radius[i]	# Particle radius

	    delta = numpy.linalg.norm(cpos - p) - r

	    if (delta < 0.0): # Cell lies within a particle
	      fine_grid[ix,iy,iz] = True
	      break # No need to check more particles

	  fine_grid[ix,iy,iz] = False

    # Interpolate fine grid to coarse grid by looping
    # over the fine grid, and subtracting the fine cell volume
    # if it is marked as inside a particle
    for ix in range(fine_grid[0]):
      for iy in range(fine_grid[1]):
	for iz in range(fine_grid[2]):
	  if (fine_grid[ix,iy,iz] == True):
	    porosity_grid[floor(ix/precisionfactor), \
			  floor(iy/precisionfactor), \
			  floor(iz/precisionfactor) ] -= Vc_fine


    return porosity_grid



def render(binary,
           out = '~/img_out/rt-out',
	   graphicsformat = 'jpg',
	   resolution = numpy.array([800, 800]),
	   workhorse = 'GPU',
	   method = 'pressure',
	   max_val = 4e3,
	   rt_bin = '../raytracer/rt'):
  """ Render target binary using the sphere raytracer.
  """

  # Use raytracer to render the scene into a temporary PPM file
  if workhorse == 'GPU':
    subprocess.call(rt_bin + " GPU {0} {1} {2} {3}.ppm {4} {5}" \
	.format(binary, resolution[0], resolution[1], out, method, max_val), shell=True)
  if workhorse == 'CPU':
    subprocess.call(rt_bin + " CPU {0} {1} {2} {3}.ppm" \
	.format(binary, resolution[0], resolution[1], out), shell=True)

  # Use ImageMagick's convert command to change the graphics format
  subprocess.call("convert {0}.ppm {0}.{1}" \
      .format(out, graphicsformat), shell=True)

  # Delete temporary PPM file
  subprocess.call("rm {0}.ppm".format(out), shell=True)
  
  
def visualize(project, method = 'energy', savefig = False, outformat = 'png'):
  """ Visualize output from the target project,
      where the temporal progress is of interest.
  """

  lastfile = status(project)

  ### Plotting
  fig = plt.figure(figsize=(15,10),dpi=300)
  figtitle = "{0}, simulation {1}".format(method, project)
  fig.text(0.5,0.95,figtitle,horizontalalignment='center',fontproperties=FontProperties(size=18))


  if method == 'energy':

    # Allocate arrays
    Epot = numpy.zeros(lastfile+1)
    Ekin = numpy.zeros(lastfile+1)
    Erot = numpy.zeros(lastfile+1)
    Es  = numpy.zeros(lastfile+1)
    Es_dot = numpy.zeros(lastfile+1)
    Esum = numpy.zeros(lastfile+1)

    # Read energy values from project binaries
    sb = Spherebin()
    for i in range(lastfile+1):
      fn = "../output/{0}.output{1}.bin".format(project, i)
      sb.readbin(fn, verbose = False)

      Epot[i] = sb.energy("pot")
      Ekin[i] = sb.energy("kin")
      Erot[i] = sb.energy("rot")
      Es[i]   = sb.energy("shear")
      Es_dot[i] = sb.energy("shearrate")
      Esum[i] = Epot[i] + Ekin[i] + Erot[i] + Es[i]
    
    t = numpy.linspace(0.0, sb.time_current, lastfile+1)

    # Potential energy
    ax1 = plt.subplot2grid((2,3),(0,0))
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Total potential energy [J]')
    ax1.plot(t, Epot, '+-')

    # Kinetic energy
    ax2 = plt.subplot2grid((2,3),(0,1))
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Total kinetic energy [J]')
    ax2.plot(t, Ekin, '+-')

    # Rotational energy
    ax3 = plt.subplot2grid((2,3),(0,2))
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Total rotational energy [J]')
    ax3.plot(t, Erot, '+-')

    # Shear energy rate
    ax4 = plt.subplot2grid((2,3),(1,0))
    ax4.set_xlabel('Time [s]')
    ax4.set_ylabel('Shear energy rate [W]')
    ax4.plot(t, Es_dot, '+-')
    
    # Shear energy
    ax5 = plt.subplot2grid((2,3),(1,1))
    ax5.set_xlabel('Time [s]')
    ax5.set_ylabel('Total shear energy [J]')
    ax5.plot(t, Es, '+-')

    # Total energy
    #ax6 = plt.subplot2grid((2,3),(1,2))
    #ax6.set_xlabel('Time [s]')
    #ax6.set_ylabel('Total energy [J]')
    #ax6.plot(t, Esum, '+-')

    # Combined view
    ax6 = plt.subplot2grid((2,3),(1,2))
    ax6.set_xlabel('Time [s]')
    ax6.set_ylabel('Energy [J]')
    ax6.plot(t, Epot, '+-g')
    ax6.plot(t, Ekin, '+-b')
    ax6.plot(t, Erot, '+-r')
    ax6.legend(('$\sum E_{pot}$','$\sum E_{kin}$','$\sum E_{rot}$'), 'upper right', shadow=True)

  elif method == 'walls':

    # Read energy values from project binaries
    sb = Spherebin()
    for i in range(lastfile+1):
      fn = "../output/{0}.output{1}.bin".format(project, i)
      sb.readbin(fn, verbose = False)

      # Allocate arrays on first run
      if (i == 0):
	wforce = numpy.zeros((lastfile+1)*sb.nw[0], dtype=numpy.float64).reshape((lastfile+1), sb.nw[0])
	wvel   = numpy.zeros((lastfile+1)*sb.nw[0], dtype=numpy.float64).reshape((lastfile+1), sb.nw[0])
	wpos   = numpy.zeros((lastfile+1)*sb.nw[0], dtype=numpy.float64).reshape((lastfile+1), sb.nw[0])
	wdevs  = numpy.zeros((lastfile+1)*sb.nw[0], dtype=numpy.float64).reshape((lastfile+1), sb.nw[0])

      wforce[i] = sb.w_force[0]
      wvel[i]   = sb.w_vel[0]
      wpos[i]   = sb.w_x[0]
      wdevs[i]  = sb.w_devs[0]
    
    t = numpy.linspace(0.0, sb.time_current, lastfile+1)

    # Plotting
    ax1 = plt.subplot2grid((2,2),(0,0))
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Position [m]')
    ax1.plot(t, wpos, '+-')

    ax2 = plt.subplot2grid((2,2),(0,1))
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Velocity [m/s]')
    ax2.plot(t, wvel, '+-')

    ax3 = plt.subplot2grid((2,2),(1,0))
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Force [N]')
    ax3.plot(t, wforce, '+-')

    ax4 = plt.subplot2grid((2,2),(1,1))
    ax4.set_xlabel('Time [s]')
    ax4.set_ylabel('Deviatoric stress [Pa]')
    ax4.plot(t, wdevs, '+-')


  elif method == 'shear':

    sb = Spherebin()
    # Read stress values from project binaries
    for i in range(lastfile+1):

      fn = "../output/{0}.output{1}.bin".format(project, i)
      sb.readbin(fn, verbose = False)

      # First iteration: Allocate arrays and find constant values
      if (i == 0):
	xdisp    = numpy.zeros(lastfile+1, dtype=numpy.float64)  # Shear displacement
	sigma    = numpy.zeros(lastfile+1, dtype=numpy.float64)  # Normal stress
	tau      = numpy.zeros(lastfile+1, dtype=numpy.float64)  # Shear stress
	dilation = numpy.zeros(lastfile+1, dtype=numpy.float64)  # Upper wall position

	fixvel = numpy.nonzero(sb.fixvel > 0.0)
	#fixvel_upper = numpy.nonzero(sb.vel[fixvel,0] > 0.0)
	shearvel = sb.vel[fixvel,0].max()
	w_x0 = sb.w_x[0]
	A = sb.L[0] * sb.L[1]   # Upper surface area

      # Summation of shear stress contributions
      for j in fixvel[0]:
	if (sb.vel[j,0] > 0.0):
	  tau[i] += -sb.force[j,0]

      xdisp[i]    = sb.time_current[0] * shearvel
      sigma[i]    = sb.w_force[0] / A
      sigma[i]    = sb.w_devs[0]
      #tau[i]      = sb.force[fixvel_upper,0].sum() / A
      dilation[i] = sb.w_x[0] - w_x0

    # Plot stresses
    ax1 = plt.subplot2grid((2,1),(0,0))
    ax1.set_xlabel('Shear distance [m]')
    ax1.set_ylabel('Stress [Pa]')
    ax1.plot(xdisp, sigma, '+-g')
    ax1.plot(xdisp, tau, '+-r')
    #plt.legend('$\sigma`$','$\tau$')

    # Plot dilation
    ax2 = plt.subplot2grid((2,1),(1,0))
    ax2.set_xlabel('Shear distance [m]')
    ax2.set_ylabel('Dilation [m]')
    ax2.plot(xdisp, dilation, '+-')


  # Optional save of figure
  if (savefig == True):
    fig.savefig("{0}-{1}.{2}".format(project, method, outformat))
    fig.clf()
  else:
    plt.show()


def status(project):
  """ Check the status.dat file for the target project,
      and return the last file numer.
  """
  fh = None
  try:
    filepath = "../output/{0}.status.dat".format(project)
    #print(filepath)
    fh = open(filepath)
    data = fh.read()
    #print(data)
    return int(data.split()[2])  # Return last file number
  except (EnvironmentError, ValueError, KeyError) as err:
    print("status.py: import error: {0}".format(err))
    return False
  finally:
    if fh is not None:
      fh.close()


