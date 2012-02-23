#!/usr/bin/env python2.7
import math
import numpy
import matplotlib.pyplot as plt
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
    self.time_dt         = numpy.zeros(1, dtype=numpy.float32)
    self.time_current    = numpy.zeros(1, dtype=numpy.float32)
    self.time_total      = numpy.zeros(1, dtype=numpy.float64)
    self.time_file_dt    = numpy.zeros(1, dtype=numpy.float32)
    self.time_step_count = numpy.zeros(1, dtype=numpy.uint32)

    # World dimensions and grid data
    self.origo   = numpy.zeros(self.nd, dtype=numpy.float32)
    self.L       = numpy.zeros(self.nd, dtype=numpy.float32)
    self.num     = numpy.zeros(self.nd, dtype=numpy.uint32)

    # Particle data
    self.x       = numpy.zeros(self.np*self.nd, dtype=numpy.float32).reshape(self.np,self.nd)
    self.vel     = numpy.zeros(self.np*self.nd, dtype=numpy.float32).reshape(self.np,self.nd)
    self.angvel  = numpy.zeros(self.np*self.nd, dtype=numpy.float32).reshape(self.np,self.nd)
    self.force   = numpy.zeros(self.np*self.nd, dtype=numpy.float32).reshape(self.np,self.nd)
    self.torque  = numpy.zeros(self.np*self.nd, dtype=numpy.float32).reshape(self.np,self.nd)
    self.fixvel  = numpy.zeros(self.np, dtype=numpy.float32)
    self.xsum    = numpy.zeros(self.np, dtype=numpy.float32)
    self.radius  = numpy.zeros(self.np, dtype=numpy.float32)
    self.rho     = numpy.zeros(self.np, dtype=numpy.float32)
    self.k_n     = numpy.zeros(self.np, dtype=numpy.float32)
    self.k_s     = numpy.zeros(self.np, dtype=numpy.float32)
    self.k_r	 = numpy.zeros(self.np, dtype=numpy.float32)
    self.gamma_s = numpy.zeros(self.np, dtype=numpy.float32)
    self.gamma_r = numpy.zeros(self.np, dtype=numpy.float32)
    self.mu_s    = numpy.zeros(self.np, dtype=numpy.float32)
    self.mu_r    = numpy.zeros(self.np, dtype=numpy.float32)
    self.C       = numpy.zeros(self.np, dtype=numpy.float32)
    self.E       = numpy.zeros(self.np, dtype=numpy.float32)
    self.K       = numpy.zeros(self.np, dtype=numpy.float32)
    self.nu      = numpy.zeros(self.np, dtype=numpy.float32)
    self.es_dot  = numpy.zeros(self.np, dtype=numpy.float32)
    self.es	 = numpy.zeros(self.np, dtype=numpy.float32)
    self.p	 = numpy.zeros(self.np, dtype=numpy.float32)

    self.bonds   = numpy.ones(self.np*4, dtype=numpy.uint32).reshape(self.np,4) * self.np

    # Constant, single-value physical parameters
    self.globalparams = numpy.zeros(1, dtype=numpy.int32)
    self.g            = numpy.zeros(self.nd, dtype=numpy.float32)
    self.kappa        = numpy.zeros(1, dtype=numpy.float32)
    self.db           = numpy.zeros(1, dtype=numpy.float32)
    self.V_b          = numpy.zeros(1, dtype=numpy.float32)
    self.shearmodel   = numpy.zeros(1, dtype=numpy.uint32)

    # Wall data
    self.nw 	 = numpy.ones(1, dtype=numpy.uint32)
    self.w_n     = numpy.zeros(self.nw*self.nd, dtype=numpy.float32).reshape(self.nw,self.nd)
    self.w_x     = numpy.zeros(self.nw, dtype=numpy.float32)
    self.w_m     = numpy.zeros(self.nw, dtype=numpy.float32)
    self.w_vel   = numpy.zeros(self.nw, dtype=numpy.float32)
    self.w_force = numpy.zeros(self.nw, dtype=numpy.float32)
    self.w_devs  = numpy.zeros(self.nw, dtype=numpy.float32)
    
    # x- and y-boundary behavior
    self.periodic = numpy.zeros(1, dtype=numpy.uint32)


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
      self.time_dt 	   = numpy.fromfile(fh, dtype=numpy.float32, count=1)
      self.time_current    = numpy.fromfile(fh, dtype=numpy.float64, count=1)
      self.time_total 	   = numpy.fromfile(fh, dtype=numpy.float64, count=1)
      self.time_file_dt    = numpy.fromfile(fh, dtype=numpy.float32, count=1)
      self.time_step_count = numpy.fromfile(fh, dtype=numpy.uint32, count=1)

      # Allocate array memory for particles
      self.x       = numpy.zeros(self.np*self.nd, dtype=numpy.float32).reshape(self.np,self.nd)
      self.vel     = numpy.zeros(self.np*self.nd, dtype=numpy.float32).reshape(self.np,self.nd)
      self.angvel  = numpy.zeros(self.np*self.nd, dtype=numpy.float32).reshape(self.np,self.nd)
      self.force   = numpy.zeros(self.np*self.nd, dtype=numpy.float32).reshape(self.np,self.nd)
      self.torque  = numpy.zeros(self.np*self.nd, dtype=numpy.float32).reshape(self.np,self.nd)
      self.fixvel  = numpy.zeros(self.np, dtype=numpy.float32)
      self.xsum    = numpy.zeros(self.np, dtype=numpy.float32)
      self.radius  = numpy.zeros(self.np, dtype=numpy.float32)
      self.rho     = numpy.zeros(self.np, dtype=numpy.float32)
      self.k_n     = numpy.zeros(self.np, dtype=numpy.float32)
      self.k_s     = numpy.zeros(self.np, dtype=numpy.float32)
      self.k_r	   = numpy.zeros(self.np, dtype=numpy.float32)
      self.gamma_s = numpy.zeros(self.np, dtype=numpy.float32)
      self.gamma_r = numpy.zeros(self.np, dtype=numpy.float32)
      self.mu_s    = numpy.zeros(self.np, dtype=numpy.float32)
      self.mu_r    = numpy.zeros(self.np, dtype=numpy.float32)
      self.C       = numpy.zeros(self.np, dtype=numpy.float32)
      self.E       = numpy.zeros(self.np, dtype=numpy.float32)
      self.K       = numpy.zeros(self.np, dtype=numpy.float32)
      self.nu      = numpy.zeros(self.np, dtype=numpy.float32)
      self.es_dot  = numpy.zeros(self.np, dtype=numpy.float32)
      self.es	   = numpy.zeros(self.np, dtype=numpy.float32)
      self.p	   = numpy.zeros(self.np, dtype=numpy.float32)
      self.bonds   = numpy.zeros(self.np, dtype=numpy.float32)

      # Read remaining data from binary
      self.origo   = numpy.fromfile(fh, dtype=numpy.float32, count=self.nd)
      self.L       = numpy.fromfile(fh, dtype=numpy.float32, count=self.nd)
      self.num     = numpy.fromfile(fh, dtype=numpy.uint32, count=self.nd)
  
      # Per-particle vectors
      for j in range(self.np):
        for i in range(self.nd):
  	  self.x[j,i]      = numpy.fromfile(fh, dtype=numpy.float32, count=1)
	  self.vel[j,i]    = numpy.fromfile(fh, dtype=numpy.float32, count=1)
	  self.angvel[j,i] = numpy.fromfile(fh, dtype=numpy.float32, count=1)
	  self.force[j,i]  = numpy.fromfile(fh, dtype=numpy.float32, count=1)
	  self.torque[j,i] = numpy.fromfile(fh, dtype=numpy.float32, count=1)
 
      # Per-particle single-value parameters
      for j in range(self.np):
	self.fixvel[j]  = numpy.fromfile(fh, dtype=numpy.float32, count=1)
	self.xsum[j]    = numpy.fromfile(fh, dtype=numpy.float32, count=1)
        self.radius[j]  = numpy.fromfile(fh, dtype=numpy.float32, count=1)
        self.rho[j]     = numpy.fromfile(fh, dtype=numpy.float32, count=1)
        self.k_n[j]     = numpy.fromfile(fh, dtype=numpy.float32, count=1)
        self.k_s[j]     = numpy.fromfile(fh, dtype=numpy.float32, count=1)
	self.k_r[j]     = numpy.fromfile(fh, dtype=numpy.float32, count=1)
	self.gamma_s[j] = numpy.fromfile(fh, dtype=numpy.float32, count=1)
	self.gamma_r[j] = numpy.fromfile(fh, dtype=numpy.float32, count=1)
        self.mu_s[j]    = numpy.fromfile(fh, dtype=numpy.float32, count=1) 
        self.mu_r[j]    = numpy.fromfile(fh, dtype=numpy.float32, count=1)
        self.C[j]       = numpy.fromfile(fh, dtype=numpy.float32, count=1)
        self.E[j]       = numpy.fromfile(fh, dtype=numpy.float32, count=1)
        self.K[j]       = numpy.fromfile(fh, dtype=numpy.float32, count=1)
        self.nu[j]      = numpy.fromfile(fh, dtype=numpy.float32, count=1)
        self.es_dot[j]  = numpy.fromfile(fh, dtype=numpy.float32, count=1)
        self.es[j]      = numpy.fromfile(fh, dtype=numpy.float32, count=1)
        self.p[j]       = numpy.fromfile(fh, dtype=numpy.float32, count=1)

      # Constant, single-value physical parameters
      self.globalparams = numpy.fromfile(fh, dtype=numpy.int32, count=1)
      self.g            = numpy.fromfile(fh, dtype=numpy.float32, count=self.nd)
      self.kappa        = numpy.fromfile(fh, dtype=numpy.float32, count=1)
      self.db           = numpy.fromfile(fh, dtype=numpy.float32, count=1)
      self.V_b          = numpy.fromfile(fh, dtype=numpy.float32, count=1)
      self.shearmodel   = numpy.fromfile(fh, dtype=numpy.uint32, count=1)

      # Wall data
      self.nw 	   = numpy.fromfile(fh, dtype=numpy.uint32, count=1)
      self.w_n     = numpy.zeros(self.nw*self.nd, dtype=numpy.float32).reshape(self.nw,self.nd)
      self.w_x     = numpy.zeros(self.nw, dtype=numpy.float32)
      self.w_m     = numpy.zeros(self.nw, dtype=numpy.float32)
      self.w_vel   = numpy.zeros(self.nw, dtype=numpy.float32)
      self.w_force = numpy.zeros(self.nw, dtype=numpy.float32)
      self.w_devs  = numpy.zeros(self.nw, dtype=numpy.float32)

      for j in range(self.nw):
        for i in range(self.nd):
  	  self.w_n[j,i] = numpy.fromfile(fh, dtype=numpy.float32, count=1)

        self.w_x[j]     = numpy.fromfile(fh, dtype=numpy.float32, count=1)
        self.w_m[j]     = numpy.fromfile(fh, dtype=numpy.float32, count=1)
        self.w_vel[j]   = numpy.fromfile(fh, dtype=numpy.float32, count=1)
        self.w_force[j] = numpy.fromfile(fh, dtype=numpy.float32, count=1)
        self.w_devs[j]  = numpy.fromfile(fh, dtype=numpy.float32, count=1)
    
      # x- and y-boundary behavior
      self.periodic = numpy.fromfile(fh, dtype=numpy.int32, count=1)

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
      fh.write(self.time_dt.astype(numpy.float32))
      fh.write(self.time_current.astype(numpy.float64))
      fh.write(self.time_total.astype(numpy.float64))
      fh.write(self.time_file_dt.astype(numpy.float32))
      fh.write(self.time_step_count.astype(numpy.uint32))

      # Read remaining data from binary
      fh.write(self.origo.astype(numpy.float32))
      fh.write(self.L.astype(numpy.float32))
      fh.write(self.num.astype(numpy.uint32))
  
      # Per-particle vectors
      for j in range(self.np):
        for i in range(self.nd):
	  fh.write(self.x[j,i].astype(numpy.float32))
	  fh.write(self.vel[j,i].astype(numpy.float32))
	  fh.write(self.angvel[j,i].astype(numpy.float32))
	  fh.write(self.force[j,i].astype(numpy.float32))
	  fh.write(self.torque[j,i].astype(numpy.float32))
 
      # Per-particle single-value parameters
      for j in range(self.np):
	fh.write(self.fixvel[j].astype(numpy.float32))
	fh.write(self.xsum[j].astype(numpy.float32))
        fh.write(self.radius[j].astype(numpy.float32))
        fh.write(self.rho[j].astype(numpy.float32))
        fh.write(self.k_n[j].astype(numpy.float32))
        fh.write(self.k_s[j].astype(numpy.float32))
	fh.write(self.k_r[j].astype(numpy.float32))
	fh.write(self.gamma_s[j].astype(numpy.float32))
	fh.write(self.gamma_r[j].astype(numpy.float32))
        fh.write(self.mu_s[j].astype(numpy.float32))
        fh.write(self.mu_r[j].astype(numpy.float32))
        fh.write(self.C[j].astype(numpy.float32))
        fh.write(self.E[j].astype(numpy.float32))
        fh.write(self.K[j].astype(numpy.float32))
        fh.write(self.nu[j].astype(numpy.float32))
        fh.write(self.es_dot[j].astype(numpy.float32))
        fh.write(self.es[j].astype(numpy.float32))
        fh.write(self.p[j].astype(numpy.float32))

      # Constant, single-value physical parameters
      fh.write(self.globalparams.astype(numpy.int32))
      fh.write(self.g.astype(numpy.float32))
      fh.write(self.kappa.astype(numpy.float32))
      fh.write(self.db.astype(numpy.float32))
      fh.write(self.V_b.astype(numpy.float32))
      fh.write(self.shearmodel.astype(numpy.uint32))

      fh.write(self.nw.astype(numpy.uint32))
      for j in range(self.nw):
        for i in range(self.nd):
  	  fh.write(self.w_n[j,i].astype(numpy.float32))

        fh.write(self.w_x[j].astype(numpy.float32))
        fh.write(self.w_m[j].astype(numpy.float32))
        fh.write(self.w_vel[j].astype(numpy.float32))
        fh.write(self.w_force[j].astype(numpy.float32))
        fh.write(self.w_devs[j].astype(numpy.float32))
    
      # x- and y-boundary behavior
      fh.write(self.periodic.astype(numpy.uint32))

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
      bins = 20
      # Create histogram
      plt.hist(self.radius, bins)
      # Plot
      plt.title('Particle size distribution, {0} particles'.format(self.np))
      plt.xlabel('Radii [m]')
      plt.ylabel('Count')
      plt.axis('tight')
      plt.show()
 

  # Initialize particle positions to non-overlapping configuration
  def initsetup(self, g = numpy.array([0.0, 0.0, -9.80665]), 
      		      gridnum = numpy.array([12, 12, 36]),
		      periodic = 1,
		      shearmodel = 1):
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
	  if (delta_len < 0):
	    overlaps = True
      print "\rFinding non-overlapping particle positions, {0} % complete".format(numpy.ceil(i/self.np[0]*100)),
   
    print " "
    self.shearmodel[0] = shearmodel

    # Initialize upper wall
    self.w_n[0,2] = -1.0
    self.w_x[0] = self.L[2]
    self.w_m[0] = self.rho[0] * self.np * math.pi * r_max**3
    self.w_vel[0] = 0.0
    self.w_force[0] = 0.0
    self.w_devs[0] = 0.0
    self.nw[0] = numpy.ones(1, dtype=numpy.uint32) * 1


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
    self.time_dt[0] = 0.17 * math.sqrt((4.0/3.0 * math.pi * r_min**3 * self.rho[0]) / self.k_n[0])
    
    # Time at start
    self.time_current[0] = current
    self.time_total[0] = total
    self.time_file_dt[0] = file_dt
    self.time_step_count[0] = 0

  def defaultparams(self, ang_s = 25,
      			  ang_r = 35,
			  rho = 3600,
			  k_n = 4e5,
			  k_s = 4e5,
			  k_r = 4e6,
			  gamma_s = 4e2,
			  gamma_r = 4e2,
			  capillaryCohesion = 0):
    """ Initialize particle parameters to default values.
        Radii must be set prior to calling this function.
    """
    # Particle material density, kg/m^3
    self.rho = numpy.ones(self.np, dtype=numpy.float32) * rho

    
    ### Dry granular material parameters

    # Contact normal elastic stiffness, N/m
    self.k_n = numpy.ones(self.np, dtype=numpy.float32) * k_n

    # Contact shear elastic stiffness (for shearmodel = 2), N/m
    self.k_s = numpy.ones(self.np, dtype=numpy.float32) * k_s

    # Contact rolling elastic stiffness (for shearmodel = 2), N/m
    self.k_r = numpy.ones(self.np, dtype=numpy.float32) * k_r

    # Contact shear viscosity (for shearmodel = 1), Ns/m
    self.gamma_s = numpy.ones(self.np, dtype=numpy.float32) * gamma_s

    # Contact rolling visscosity (for shearmodel = 1), Ns/m?
    self.gamma_r = numpy.ones(self.np, dtype=numpy.float32) * gamma_r

    # Contact shear friction coefficient
    self.mu_s = numpy.ones(self.np, dtype=numpy.float32) * numpy.tan(numpy.radians(ang_s))

    # Contact rolling friction coefficient
    self.mu_r = numpy.ones(self.np, dtype=numpy.float32) * numpy.tan(numpy.radians(ang_r))

    r_min = numpy.amin(self.radius)
    
    # Poisson's ratio. Critical damping: 2*sqrt(m*k_n).
    # Normal force component elastic if nu = 0.0.
    self.nu = numpy.ones(self.np, dtype=numpy.float32) \
	      * 0.1 * 2.0 * math.sqrt(4.0/3.0 * math.pi * r_min**3 \
	      * self.rho[0] * self.k_n[0])
    
    # Global parameters
    # if 1 = all particles have the same values for the physical parameters
    self.globalparams[0] = 1

    ### Parameters related to capillary bonds

    # Wettability, 0=perfect
    theta = 0.0;
    if (capillaryCohesion == 1):
      self.kappa[0] = 2.0 * math.pi * gamma_s * numpy.cos(theta)  # Prefactor
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


def render(binary,
           out = '~/img_out/rt-out',
	   graphicsformat = 'jpg',
	   resolution = numpy.array([800, 800]),
	   workhorse = 'GPU',
	   method = 'pressure',
	   max_val = 4e3):
  """ Render target binary using the sphere raytracer.
  """

  # Use raytracer to render the scene into a temporary PPM file
  if workhorse == 'GPU':
    subprocess.call("../raytracer/rt GPU {0} {1} {2} {3}.ppm {4} {5}" \
	.format(binary, resolution[0], resolution[1], out, method, max_val), shell=True)
  if workhorse == 'CPU':
    subprocess.call("../raytracer/rt CPU {0} {1} {2} {3}.ppm" \
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

    # Plotting
    plt.subplot(2,3,1)
    plt.xlabel('Time [s]')
    plt.ylabel('Total potential energy [J]')
    plt.plot(t, Epot, '+-')

    plt.subplot(2,3,2)
    plt.xlabel('Time [s]')
    plt.ylabel('Total kinetic energy [J]')
    plt.plot(t, Ekin, '+-')

    plt.subplot(2,3,3)
    plt.xlabel('Time [s]')
    plt.ylabel('Total rotational energy [J]')
    plt.plot(t, Erot, '+-')

    plt.subplot(2,3,4)
    plt.xlabel('Time [s]')
    plt.ylabel('Shear energy rate [W]')
    plt.plot(t, Es_dot, '+-')

    plt.subplot(2,3,5)
    plt.xlabel('Time [s]')
    plt.ylabel('Total shear energy [J]')
    plt.plot(t, Es, '+-')

    plt.subplot(2,3,6)
    plt.xlabel('Time [s]')
    plt.ylabel('Total energy [J]')
    plt.plot(t, Esum, '+-')

    #plt.show()

  elif method == 'walls':

    # Read energy values from project binaries
    sb = Spherebin()
    for i in range(lastfile+1):
      fn = "../output/{0}.output{1}.bin".format(project, i)
      sb.readbin(fn, verbose = False)

      # Allocate arrays on first run
      if (i == 0):
	wforce = numpy.zeros(lastfile+1, sb.nw[0])
	wvel   = numpy.zeros(lastfile+1, sb.nw[0])
	wpos   = numpy.zeros(lastfile+1, sb.nw[0])
	wdevs  = numpy.zeros(lastfile+1, sb.nw[0])

      wforce[i] = sb.w_force
      wvel[i]   = sb.w_vel
      wpos[i]   = sb.w_x
      wdevs[i]  = sb.w_devs
    
    t = numpy.linspace(0.0, sb.time_current, lastfile+1)

    # Plotting
    plt.subplot(2,2,1)
    plt.xlabel('Time [s]')
    plt.ylabel('Position [m]')
    plt.plot(t, wpos, '+-')

    plt.subplot(2,2,2)
    plt.xlabel('Time [s]')
    plt.ylabel('Velocity [m/s]')
    plt.plot(t, wvel, '+-')

    plt.subplot(2,2,3)
    plt.xlabel('Time [s]')
    plt.ylabel('Force [N]')
    plt.plot(t, wforce, '+-')

    plt.subplot(2,2,4)
    plt.xlabel('Time [s]')
    plt.ylabel('Deviatoric stress [Pa]')
    plt.plot(t, wdevs, '+-')


  # Optional save of figure
  if (savefig == True):
    plt.savefig("{0}-{1}.{2}".format(project, method, outformat))
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


