#!/usr/bin/env python2.7
import math
import numpy
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import subprocess

numpy.seterr(all='warn', over='raise')

class Spherebin:
    """ 
    Class containing all data SPHERE data.
    
    Contains functions for reading and writing binaries, as well as simulation
    setup and data analysis.
    """

    def __init__(self, np = 1, nd = 3, nw = 1, sid = 'unnamed'):
        """
        Constructor - initializes arrays

        :param np: the number of particles to allocate memory for
        :param nd: the number of spatial dimensions
        :param nw: the number of dynamic walls
        :param sid: the simulation id
        :type np: int
        :type nd: int
        :type nw: int
        :type sid: string

        """

        self.nd = numpy.ones(1, dtype=numpy.int32) * nd
        self.np = numpy.ones(1, dtype=numpy.uint32) * np
        self.sid = sid

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
        self.periodic = numpy.zeros(1, dtype=numpy.uint32)

        # Particle data
        self.x       = numpy.zeros((self.np, self.nd), dtype=numpy.float64)
        self.radius  = numpy.ones(self.np, dtype=numpy.float64)
        self.xysum   = numpy.zeros((self.np, 2), dtype=numpy.float64)
        self.vel     = numpy.zeros((self.np, self.nd), dtype=numpy.float64)
        self.fixvel  = numpy.zeros(self.np, dtype=numpy.float64)
        self.force   = numpy.zeros((self.np, self.nd), dtype=numpy.float64)
        self.angpos  = numpy.zeros((self.np, self.nd), dtype=numpy.float64)
        self.angvel  = numpy.zeros((self.np, self.nd), dtype=numpy.float64)
        self.torque  = numpy.zeros((self.np, self.nd), dtype=numpy.float64)

        self.es_dot  = numpy.zeros(self.np, dtype=numpy.float64)
        self.es      = numpy.zeros(self.np, dtype=numpy.float64)
        self.ev_dot  = numpy.zeros(self.np, dtype=numpy.float64)
        self.ev      = numpy.zeros(self.np, dtype=numpy.float64)
        self.p       = numpy.zeros(self.np, dtype=numpy.float64)

        self.g        = numpy.array([0.0, 0.0, 0.0], dtype=numpy.float64)
        self.k_n      = numpy.ones(1, dtype=numpy.float64) * 1.16e9
        self.k_t      = numpy.ones(1, dtype=numpy.float64) * 1.16e9
        self.k_r      = numpy.zeros(1, dtype=numpy.float64)
        self.gamma_n  = numpy.zeros(1, dtype=numpy.float64)
        self.gamma_t  = numpy.zeros(1, dtype=numpy.float64)
        self.gamma_r  = numpy.zeros(1, dtype=numpy.float64)
        self.mu_s     = numpy.ones(1, dtype=numpy.float64)
        self.mu_d     = numpy.ones(1, dtype=numpy.float64)
        self.mu_r     = numpy.zeros(1, dtype=numpy.float64)
        self.gamma_wn = numpy.ones(1, dtype=numpy.float64) * 1.0e3
        self.gamma_wt = numpy.ones(1, dtype=numpy.float64) * 1.0e3
        self.mu_ws    = numpy.ones(1, dtype=numpy.float64)
        self.mu_wd    = numpy.ones(1, dtype=numpy.float64)
        self.rho      = numpy.ones(1, dtype=numpy.float64) * 2600.0
        self.contactmodel = numpy.ones(1, dtype=numpy.uint32) * 2    # contactLinear default
        self.kappa        = numpy.zeros(1, dtype=numpy.float64)
        self.db           = numpy.zeros(1, dtype=numpy.float64)
        self.V_b          = numpy.zeros(1, dtype=numpy.float64)

        # Wall data
        self.nw      = numpy.ones(1, dtype=numpy.uint32) * nw
        self.wmode   = numpy.zeros(self.nw, dtype=numpy.int32)

        self.w_n     = numpy.zeros((self.nw, self.nd), dtype=numpy.float64)
        if (self.nw > 0):
            self.w_n[0,2] = -1.0
        self.w_x     = numpy.ones(self.nw, dtype=numpy.float64)
        self.w_m     = numpy.zeros(self.nw, dtype=numpy.float64)
        self.w_vel   = numpy.zeros(self.nw, dtype=numpy.float64)
        self.w_force = numpy.zeros(self.nw, dtype=numpy.float64)
        self.w_devs  = numpy.zeros(self.nw, dtype=numpy.float64)

        self.lambda_bar = numpy.ones(1, dtype=numpy.float64)
        self.nb0 = numpy.zeros(1, dtype=numpy.uint32)
        self.bonds = numpy.zeros((self.nb0, 2), dtype=numpy.uint32)
        self.bonds_delta_n = numpy.zeros(self.nb0, dtype=numpy.float64)
        self.bonds_delta_t = numpy.zeros((self.nb0, self.nd), dtype=numpy.float64)
        self.bonds_omega_n = numpy.zeros(self.nb0, dtype=numpy.float64)
        self.bonds_omega_t = numpy.zeros((self.nb0, self.nd), dtype=numpy.float64)

    def __cmp__(self, other):
        """ Called when to Spherebin objects are compared.
            Returns 0 if the values are identical """
        if ( (\
                self.nd == other.nd and\
                self.np == other.np and\
                self.time_dt == other.time_dt and\
                self.time_current == other.time_current and\
                self.time_total == other.time_total and\
                self.time_file_dt == other.time_file_dt and\
                self.time_step_count == other.time_step_count and\
                (self.origo == other.origo).all() and\
                (self.L == other.L).all() and\
                (self.num == other.num).all() and\
                self.periodic == other.periodic and\
                (self.x == other.x).all() and\
                (self.radius == other.radius).all() and\
                (self.xysum == other.xysum).all() and\
                (self.vel == other.vel).all() and\
                (self.fixvel == other.fixvel).all() and\
                (self.force == other.force).all() and\
                (self.angpos == other.angpos).all() and\
                (self.angvel == other.angvel).all() and\
                (self.torque == other.torque).all() and\
                (self.es_dot == other.es_dot).all() and\
                (self.es == other.es).all() and\
                (self.ev_dot == other.ev_dot).all() and\
                (self.ev == other.ev).all() and\
                (self.p == other.p).all() and\
                (self.g == other.g).all() and\
                self.k_n == other.k_n and\
                self.k_t == other.k_t and\
                self.k_r == other.k_r and\
                self.gamma_n == other.gamma_n and\
                self.gamma_t == other.gamma_t and\
                self.gamma_r == other.gamma_r and\
                self.mu_s == other.mu_s and\
                self.mu_d == other.mu_d and\
                self.mu_r == other.mu_r and\
                self.rho == other.rho and\
                self.contactmodel == other.contactmodel and\
                self.kappa == other.kappa and\
                self.db == other.db and\
                self.V_b == other.V_b and\
                self.nw == other.nw and\
                (self.wmode == other.wmode).all() and\
                (self.w_n == other.w_n).all() and\
                (self.w_x == other.w_x).all() and\
                (self.w_m == other.w_m).all() and\
                (self.w_vel == other.w_vel).all() and\
                (self.w_force == other.w_force).all() and\
                (self.w_devs == other.w_devs).all() and\
                self.gamma_wn == other.gamma_wn and\
                self.gamma_wt == other.gamma_wt and\
                self.lambda_bar == other.lambda_bar and\
                self.nb0 == other.nb0 and\
                self.bonds == other.bonds and\
                self.bonds_delta_n == other.bonds_delta_n and\
                self.bonds_delta_t == other.bonds_delta_t and\
                self.bonds_omega_n == other.bonds_omega_n and\
                self.bonds_omega_t == other.bonds_omega_t\
                ).all() == True):
                    return 0 # All equal
        else :
            return 1



    def readbin(self, targetbin, verbose = True):
        'Reads a target SPHERE binary file'

        fh = None
        try :
            if (verbose == True):
                print("Input file: {0}".format(targetbin))
            fh = open(targetbin, "rb")

            # Read the number of dimensions and particles
            self.nd = numpy.fromfile(fh, dtype=numpy.int32, count=1)
            self.np = numpy.fromfile(fh, dtype=numpy.uint32, count=1)

            # Read the time variables
            self.time_dt         = numpy.fromfile(fh, dtype=numpy.float64, count=1)
            self.time_current    = numpy.fromfile(fh, dtype=numpy.float64, count=1)
            self.time_total      = numpy.fromfile(fh, dtype=numpy.float64, count=1)
            self.time_file_dt    = numpy.fromfile(fh, dtype=numpy.float64, count=1)
            self.time_step_count = numpy.fromfile(fh, dtype=numpy.uint32, count=1)

            # Allocate array memory for particles
            self.x       = numpy.zeros((self.np, self.nd), dtype=numpy.float64)
            self.radius  = numpy.zeros(self.np, dtype=numpy.float64)
            self.xysum   = numpy.zeros((self.np, 2), dtype=numpy.float64)
            self.vel     = numpy.zeros((self.np, self.nd), dtype=numpy.float64)
            self.fixvel  = numpy.zeros(self.np, dtype=numpy.float64)
            #self.force   = numpy.zeros((self.np, self.nd), dtype=numpy.float64)
            #self.angpos  = numpy.zeros((self.np, self.nd), dtype=numpy.float64)
            #self.angvel  = numpy.zeros((self.np, self.nd), dtype=numpy.float64)
            #self.torque  = numpy.zeros((self.np, self.nd), dtype=numpy.float64)
            self.es_dot  = numpy.zeros(self.np, dtype=numpy.float64)
            self.es      = numpy.zeros(self.np, dtype=numpy.float64)
            self.ev_dot  = numpy.zeros(self.np, dtype=numpy.float64)
            self.ev      = numpy.zeros(self.np, dtype=numpy.float64)
            self.p       = numpy.zeros(self.np, dtype=numpy.float64)

            # Read remaining data from binary
            self.origo    = numpy.fromfile(fh, dtype=numpy.float64, count=self.nd)
            self.L        = numpy.fromfile(fh, dtype=numpy.float64, count=self.nd)
            self.num      = numpy.fromfile(fh, dtype=numpy.uint32, count=self.nd)
            self.periodic = numpy.fromfile(fh, dtype=numpy.int32, count=1)

            # Per-particle vectors
            for i in range(self.np):
                self.x[i,:]    = numpy.fromfile(fh, dtype=numpy.float64, count=self.nd)
                self.radius[i] = numpy.fromfile(fh, dtype=numpy.float64, count=1)

            self.xysum = numpy.fromfile(fh, dtype=numpy.float64, count=self.np*2).reshape(self.np,2)

            for i in range(self.np):
                self.vel[i,:]  = numpy.fromfile(fh, dtype=numpy.float64, count=self.nd)
                self.fixvel[i] = numpy.fromfile(fh, dtype=numpy.float64, count=1)

            self.force = numpy.fromfile(fh, dtype=numpy.float64, count=self.np*self.nd).reshape(self.np, self.nd)

            self.angpos = numpy.fromfile(fh, dtype=numpy.float64, count=self.np*self.nd).reshape(self.np, self.nd)
            self.angvel = numpy.fromfile(fh, dtype=numpy.float64, count=self.np*self.nd).reshape(self.np, self.nd)
            self.torque = numpy.fromfile(fh, dtype=numpy.float64, count=self.np*self.nd).reshape(self.np, self.nd)

            # Per-particle single-value parameters
            self.es_dot  = numpy.fromfile(fh, dtype=numpy.float64, count=self.np)
            self.es      = numpy.fromfile(fh, dtype=numpy.float64, count=self.np)
            self.ev_dot  = numpy.fromfile(fh, dtype=numpy.float64, count=self.np)
            self.ev      = numpy.fromfile(fh, dtype=numpy.float64, count=self.np)
            self.p       = numpy.fromfile(fh, dtype=numpy.float64, count=self.np)

            # Constant, global physical parameters
            self.g            = numpy.fromfile(fh, dtype=numpy.float64, count=self.nd)
            self.k_n          = numpy.fromfile(fh, dtype=numpy.float64, count=1)
            self.k_t          = numpy.fromfile(fh, dtype=numpy.float64, count=1)
            self.k_r          = numpy.fromfile(fh, dtype=numpy.float64, count=1)
            self.gamma_n      = numpy.fromfile(fh, dtype=numpy.float64, count=1)
            self.gamma_t      = numpy.fromfile(fh, dtype=numpy.float64, count=1)
            self.gamma_r      = numpy.fromfile(fh, dtype=numpy.float64, count=1)
            self.mu_s         = numpy.fromfile(fh, dtype=numpy.float64, count=1) 
            self.mu_d         = numpy.fromfile(fh, dtype=numpy.float64, count=1) 
            self.mu_r         = numpy.fromfile(fh, dtype=numpy.float64, count=1)
            self.gamma_wn     = numpy.fromfile(fh, dtype=numpy.float64, count=1)
            self.gamma_wt     = numpy.fromfile(fh, dtype=numpy.float64, count=1)
            self.mu_ws        = numpy.fromfile(fh, dtype=numpy.float64, count=1) 
            self.mu_wd        = numpy.fromfile(fh, dtype=numpy.float64, count=1) 
            self.rho          = numpy.fromfile(fh, dtype=numpy.float64, count=1)
            self.contactmodel = numpy.fromfile(fh, dtype=numpy.uint32, count=1)
            self.kappa        = numpy.fromfile(fh, dtype=numpy.float64, count=1)
            self.db           = numpy.fromfile(fh, dtype=numpy.float64, count=1)
            self.V_b          = numpy.fromfile(fh, dtype=numpy.float64, count=1)

            # Wall data
            self.nw      = numpy.fromfile(fh, dtype=numpy.uint32, count=1)
            self.wmode   = numpy.zeros(self.nw, dtype=numpy.int32)
            self.w_n     = numpy.zeros(self.nw*self.nd, dtype=numpy.float64).reshape(self.nw,self.nd)
            self.w_x     = numpy.zeros(self.nw, dtype=numpy.float64)
            self.w_m     = numpy.zeros(self.nw, dtype=numpy.float64)
            self.w_vel   = numpy.zeros(self.nw, dtype=numpy.float64)
            self.w_force = numpy.zeros(self.nw, dtype=numpy.float64)
            self.w_devs  = numpy.zeros(self.nw, dtype=numpy.float64)

            self.wmode   = numpy.fromfile(fh, dtype=numpy.int32, count=self.nw)
            for i in range(self.nw):
                self.w_n[i,:] = numpy.fromfile(fh, dtype=numpy.float64, count=self.nd)
                self.w_x[i]   = numpy.fromfile(fh, dtype=numpy.float64, count=1)
            for i in range(self.nw):
                self.w_m[i]     = numpy.fromfile(fh, dtype=numpy.float64, count=1)
                self.w_vel[i]   = numpy.fromfile(fh, dtype=numpy.float64, count=1)
                self.w_force[i] = numpy.fromfile(fh, dtype=numpy.float64, count=1)
                self.w_devs[i]  = numpy.fromfile(fh, dtype=numpy.float64, count=1)

            # Inter-particle bonds
            self.lambda_bar = numpy.fromfile(fh, dtype=numpy.float64, count=1)
            self.nb0 = numpy.fromfile(fh, dtype=numpy.uint32, count=1)
            self.bonds = numpy.zeros((self.nb0, 2), dtype=numpy.uint32)
            for i in range(self.nb0):
                self.bonds[i,0] = numpy.fromfile(fh, dtype=numpy.uint32, count=1)
                self.bonds[i,1] = numpy.fromfile(fh, dtype=numpy.uint32, count=1)
            #self.bonds_delta_n = numpy.zeros(self.nb0, dtype=numpy.float64)
            #self.bonds_delta_t = numpy.zeros((self.nb0, seld.nd), dtype=numpy.float64)
            #self.bonds_omega_n = numpy.zeros(self.nb0, dtype=numpy.float64)
            #self.bonds_omega_t = numpy.zeros((self.nb0, seld.nd), dtype=numpy.float64)
            self.bonds_delta_n = numpy.fromfile(fh, dtype=numpy.float64, count=self.nb0)
            self.bonds_delta_t = numpy.fromfile(fh, dtype=numpy.float64, count=self.nb0*self.nd).reshape(self.nb0*self.nd)
            self.bonds_omega_n = numpy.fromfile(fh, dtype=numpy.float64, count=self.nb0)
            self.bonds_omega_t = numpy.fromfile(fh, dtype=numpy.float64, count=self.nb0*self.nd).reshape(self.nb0*self.nd)

        finally:
            if fh is not None:
                fh.close()

    def writebin(self, folder = "../input/", verbose = True):
        'Writes to a target SPHERE binary file'

        fh = None
        try :
            targetbin = folder + "/" + self.sid + ".bin"
            if (verbose == True):
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
            fh.write(self.periodic.astype(numpy.uint32))

            # Per-particle vectors
            for i in range(self.np):
                fh.write(self.x[i,:].astype(numpy.float64))
                fh.write(self.radius[i].astype(numpy.float64))

            fh.write(self.xysum.astype(numpy.float64))

            for i in range(self.np):
                fh.write(self.vel[i,:].astype(numpy.float64))
                fh.write(self.fixvel[i].astype(numpy.float64))

            fh.write(self.force.astype(numpy.float64))

            fh.write(self.angpos.astype(numpy.float64))
            fh.write(self.angvel.astype(numpy.float64))
            fh.write(self.torque.astype(numpy.float64))

            # Per-particle single-value parameters
            fh.write(self.es_dot.astype(numpy.float64))
            fh.write(self.es.astype(numpy.float64))
            fh.write(self.ev_dot.astype(numpy.float64))
            fh.write(self.ev.astype(numpy.float64))
            fh.write(self.p.astype(numpy.float64))

            fh.write(self.g.astype(numpy.float64))
            fh.write(self.k_n.astype(numpy.float64))
            fh.write(self.k_t.astype(numpy.float64))
            fh.write(self.k_r.astype(numpy.float64))
            fh.write(self.gamma_n.astype(numpy.float64))
            fh.write(self.gamma_t.astype(numpy.float64))
            fh.write(self.gamma_r.astype(numpy.float64))
            fh.write(self.mu_s.astype(numpy.float64))
            fh.write(self.mu_d.astype(numpy.float64))
            fh.write(self.mu_r.astype(numpy.float64))
            fh.write(self.gamma_wn.astype(numpy.float64))
            fh.write(self.gamma_wt.astype(numpy.float64))
            fh.write(self.mu_ws.astype(numpy.float64))
            fh.write(self.mu_wd.astype(numpy.float64))
            fh.write(self.rho.astype(numpy.float64))
            fh.write(self.contactmodel.astype(numpy.uint32))
            fh.write(self.kappa.astype(numpy.float64))
            fh.write(self.db.astype(numpy.float64))
            fh.write(self.V_b.astype(numpy.float64))

            fh.write(self.nw.astype(numpy.uint32))
            for i in range(self.nw):
                fh.write(self.wmode[i].astype(numpy.int32))
            for i in range(self.nw):
                fh.write(self.w_n[i,:].astype(numpy.float64))
                fh.write(self.w_x[i].astype(numpy.float64))

            for i in range(self.nw):
                fh.write(self.w_m[i].astype(numpy.float64))
                fh.write(self.w_vel[i].astype(numpy.float64))
                fh.write(self.w_force[i].astype(numpy.float64))
                fh.write(self.w_devs[i].astype(numpy.float64))

            fh.write(self.lambda_bar.astype(numpy.float64))
            fh.write(self.nb0.astype(numpy.uint32))
            for i in range(self.nb0):
                fh.write(self.bonds[i,0].astype(numpy.uint32))
                fh.write(self.bonds[i,1].astype(numpy.uint32))
            fh.write(self.bonds_delta_n.astype(numpy.float64))
            fh.write(self.bonds_delta_t.astype(numpy.float64))
            fh.write(self.bonds_omega_n.astype(numpy.float64))
            fh.write(self.bonds_omega_t.astype(numpy.float64))


        finally:
            if fh is not None:
                fh.close()

    def readlast(self, verbose=True):
        lastfile = status(self.sid)
        fn = "../output/{0}.output{1:0=5}.bin".format(self.sid, lastfile)
        self.readbin(fn, verbose)

    def generateRadii(self, psd = 'logn',
            radius_mean = 440e-6,
            radius_variance = 8.8e-9,
            histogram = True):
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
        if (histogram == True):
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


    def initRandomPos(self, g = numpy.array([0.0, 0.0, -9.80665]), 
            gridnum = numpy.array([12, 12, 36]),
            periodic = 1,
            contactmodel = 2):
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
            print("\rFinding non-overlapping particle positions, {0} % complete".format(numpy.ceil(i/self.np[0]*100)))

        # Print newline
        print()

        self.contactmodel[0] = contactmodel


    def initGrid(self):
        """ Initialize grid suitable for the particle positions set previously.
            The margin parameter adjusts the distance (in no. of max. radii)
            from the particle boundaries.
        """

        # Cell configuration
        cellsize_min = 2.1 * numpy.amax(self.radius)
        self.num[0] = numpy.ceil((self.L[0]-self.origo[0])/cellsize_min)
        self.num[1] = numpy.ceil((self.L[1]-self.origo[1])/cellsize_min)
        self.num[2] = numpy.ceil((self.L[2]-self.origo[2])/cellsize_min)

        if (self.num[0] < 4 or self.num[1] < 4 or self.num[2] < 4):
            print("Error: The grid must be at least 3 cells in each direction")
            print(" Grid: x={}, y={}, z={}".format(self.num[0], self.num[1], self.num[2]))

        # Put upper wall at top boundary
        if (self.nw > 0):
            self.w_x[0] = self.L[0]


    # Generate grid based on particle positions
    def initGridAndWorldsize(self, g = numpy.array([0.0, 0.0, -9.80665]),
            margin = 2.0,
            periodic = 1,
            contactmodel = 2):
        """ Initialize grid suitable for the particle positions set previously.
            The margin parameter adjusts the distance (in no. of max. radii)
            from the particle boundaries.
        """

        self.g = g
        self.periodic[0] = periodic

        # Cell configuration
        r_max = numpy.amax(self.radius)

        # Max. and min. coordinates of world
        self.origo = numpy.array([numpy.amin(self.x[:,0] - self.radius[:]),
            numpy.amin(self.x[:,1] - self.radius[:]),
            numpy.amin(self.x[:,2] - self.radius[:])]) \
                    - margin*r_max
        self.L = numpy.array([numpy.amax(self.x[:,0] + self.radius[:]),
            numpy.amax(self.x[:,1] + self.radius[:]),
            numpy.amax(self.x[:,2] + self.radius[:])]) \
                    + margin*r_max

        cellsize_min = 2.1 * r_max    
        self.num[0] = numpy.ceil((self.L[0]-self.origo[0])/cellsize_min)
        self.num[1] = numpy.ceil((self.L[1]-self.origo[1])/cellsize_min)
        self.num[2] = numpy.ceil((self.L[2]-self.origo[2])/cellsize_min)

        if (self.num[0] < 4 or self.num[1] < 4 or self.num[2] < 4):
            print("Error: The grid must be at least 3 cells in each direction")
            print(self.num)

        self.contactmodel[0] = contactmodel

        # Put upper wall at top boundary
        if (self.nw > 0):
            self.w_x[0] = self.L[0]


    # Initialize particle positions to regular, grid-like, non-overlapping configuration
    def initGridPos(self, g = numpy.array([0.0, 0.0, -9.80665]), 
            gridnum = numpy.array([12, 12, 36]),
            periodic = 1,
            contactmodel = 2):
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
            print("Error! The grid is not sufficiently large.")
            raise NameError('Error! The grid is not sufficiently large.')

        gridpos = numpy.zeros(self.nd, dtype=numpy.uint32)

        # Make sure grid is sufficiently large if every second level is moved
        if (self.periodic[0] == 1):
            self.num[0] -= 1
            self.num[1] -= 1

        # Check whether there are enough grid cells 
        if ((self.num[0]*self.num[1]*self.num[2]-(2*3*3)) < self.np):
            print("Error! The grid is not sufficiently large.")
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

        self.contactmodel[0] = contactmodel

        # Readjust grid to correct size
        if (self.periodic[0] == 1):
            self.num[0] += 1
            self.num[1] += 1


    def initRandomGridPos(self, g = numpy.array([0.0, 0.0, -9.80665]), 
            gridnum = numpy.array([12, 12, 32]),
            periodic = 1,
            contactmodel = 2):
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
            print("Error! The grid is not sufficiently large.")
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

        self.contactmodel[0] = contactmodel

        # Calculate new grid with cell size equal to max. particle diameter
        x_max = numpy.max(self.x[:,0] + self.radius)
        y_max = numpy.max(self.x[:,1] + self.radius)
        z_max = numpy.max(self.x[:,2] + self.radius)
        # Adjust size of world
        self.num[0] = numpy.ceil(x_max/cellsize)
        self.num[1] = numpy.ceil(y_max/cellsize)
        self.num[2] = numpy.ceil(z_max/cellsize)
        self.L = self.num * cellsize

    def createBondPair(self, i, j, spacing=-0.1):
        """ Bond particles i and j. Particle j is moved adjacent to particle i,
        and oriented randomly. 
        @param spacing (float) The inter-particle distance prescribed. Positive 
        values result in a inter-particle distance, negative equal an overlap.
        The value is relative to the sum of the two radii.
        """

        x_i = self.x[i]
        r_i = self.radius[i]
        r_j = self.radius[j]
        dist_ij = (r_i + r_j)*(1.0 + spacing)

        dazi = numpy.random.rand(1) * 360.0  # azimuth
        azi = numpy.radians(dazi)
        dang = numpy.random.rand(1) * 180.0 - 90.0 # angle
        ang = numpy.radians(dang)

        x_j = numpy.copy(x_i)
        x_j[0] = x_j[0] + dist_ij * numpy.cos(azi) * numpy.cos(ang)
        x_j[1] = x_j[1] + dist_ij * numpy.sin(azi) * numpy.cos(ang)
        x_j[2] = x_j[2] + dist_ij * numpy.sin(ang) * numpy.cos(azi)
        self.x[j] = x_j

        self.bond(i,j)     # register bond
        
        # Check that the spacing is correct
        x_ij = self.x[i] - self.x[j]
        x_ij_length = numpy.sqrt(x_ij.dot(x_ij))
        if ((x_ij_length - dist_ij) > dist_ij*0.01):
            print(x_i); print(r_i)
            print(x_j); print(r_j)
            print(x_ij_length); print(dist_ij)
            raise Exception("Error, something went wrong in createBondPair")



   

    def zeroKinematics(self):
        'Zero kinematics of particles'

        self.vel = numpy.zeros(self.np*self.nd, dtype=numpy.float64)\
                .reshape(self.np, self.nd)
        self.angvel = numpy.zeros(self.np*self.nd, dtype=numpy.float64)\
                .reshape(self.np, self.nd)
        self.angpos = numpy.zeros(self.np*self.nd, dtype=numpy.float64)\
                .reshape(self.np, self.nd)
        self.es = numpy.zeros(self.np, dtype=numpy.float64)
        self.ev = numpy.zeros(self.np, dtype=numpy.float64)
        self.xysum = numpy.zeros(self.np*2, dtype=numpy.float64)\
                .reshape(self.np, 2)


    def adjustUpperWall(self, z_adjust = 1.1):
        'Adjust grid and dynamic upper wall to max. particle height'

        # Compute new grid, scaled to fit max. and min. particle positions
        z_min = numpy.min(self.x[:,2] - self.radius)
        z_max = numpy.max(self.x[:,2] + self.radius)
        cellsize = self.L[0] / self.num[0]
        self.num[2] = numpy.ceil(((z_max-z_min)*z_adjust + z_min)/cellsize)
        self.L[2] = (z_max-z_min)*z_adjust + z_min

        # Initialize upper wall
        self.nw = numpy.ones(1)
        self.wmode = numpy.zeros(1) # fixed BC
        self.w_n = numpy.zeros(self.nw*self.nd, dtype=numpy.float64).reshape(self.nw,self.nd)
        self.w_n[0,2] = -1.0
        self.w_x = numpy.array([z_max])
        self.w_m = numpy.array([self.rho[0] * self.np * math.pi * (cellsize/2.0)**3])
        self.w_vel = numpy.zeros(1)
        self.w_force = numpy.zeros(1)
        self.w_devs = numpy.zeros(1) 



    def consolidate(self, deviatoric_stress = 10e3, 
            periodic = 1):
        """ Setup consolidation experiment. Specify the upper wall 
            deviatoric stress in Pascal, default value is 10 kPa.
        """

        # Zero the kinematics of all particles
        self.zeroKinematics()

        # Adjust grid and placement of upper wall
        self.adjustUpperWall()

        # Set the top wall BC to a value of deviatoric stress
        self.wmode = numpy.array([1])
        self.w_devs = numpy.ones(1) * deviatoric_stress


    def uniaxialStrainRate(self, wvel = -0.001,
            periodic = 1):
        """ Setup consolidation experiment. Specify the upper wall 
            velocity in m/s, default value is -0.001 m/s (i.e. downwards).
        """

        # zero kinematics
        self.zeroKinematics()

        # Initialize upper wall
        self.adjustUpperWall()
        self.wmode = numpy.array([2]) # strain rate BC
        self.w_vel = numpy.array([wvel])


    def shear(self,
            shear_strain_rate = 1,
            periodic = 1):
        """ Setup shear experiment. Specify the upper wall 
            deviatoric stress in Pascal, default value is 10 kPa.
            The shear strain rate is the shear length divided by the
            initial height per second.
        """

        # Find lowest and heighest point
        z_min = numpy.min(self.x[:,2] - self.radius)
        z_max = numpy.max(self.x[:,2] + self.radius)

        # the grid cell size is equal to the max. particle diameter
        cellsize = self.L[0] / self.num[0]

        # make grid one cell heigher to allow dilation
        self.num[2] += 1
        self.L[2] = self.num[2] * cellsize

        # zero kinematics
        self.zeroKinematics()

        # set the thickness of the horizons of fixed particles
        #fixheight = 2*cellsize
        fixheight = cellsize

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


        # Set wall viscosities to zero
        self.gamma_wn[0] = 0.0
        self.gamma_wt[0] = 0.0

        # Set wall friction coefficients to zero
        self.mu_ws[0] = 0.0
        self.mu_wd[0] = 0.0


    def initTemporal(self, total,
            current = 0.0,
            file_dt = 0.05,
            step_count = 0):
        """ Set temporal parameters for the simulation.
            Particle radii and physical parameters need to be set
            prior to these.
        """

        r_min = numpy.amin(self.radius)

        # Computational time step (O'Sullivan et al, 2003)
        #self.time_dt[0] = 0.17 * math.sqrt((4.0/3.0 * math.pi * r_min**3 * self.rho[0]) / numpy.amax([self.k_n[:], self.k_t[:]]) )
        # Computational time step (Zhang and Campbell, 1992)
        self.time_dt[0] = 0.075 * math.sqrt((V_sphere(r_min) * self.rho[0]) / numpy.amax([self.k_n[:], self.k_t[:]]) )

        # Time at start
        self.time_current[0] = current
        self.time_total[0] = total
        self.time_file_dt[0] = file_dt
        self.time_step_count[0] = 0


    def defaultParams(self, 
	    mu_s = 0.4,
            mu_d = 0.4,
            mu_r = 0.0,
            rho = 2600,
            k_n = 1.16e9,
            k_t = 1.16e9,
            k_r = 0,
            gamma_n = 0,
            gamma_t = 0,
            gamma_r = 0,
            gamma_wn = 1e4,
            gamma_wt = 1e4,
            capillaryCohesion = 0):
        """ Initialize particle parameters to default values.
            Radii must be set prior to calling this function.
        """

        # Particle material density, kg/m^3
        self.rho = numpy.ones(1, dtype=numpy.float64) * rho


        ### Dry granular material parameters

        # Contact normal elastic stiffness, N/m
        self.k_n = numpy.ones(1, dtype=numpy.float64) * k_n

        # Contact shear elastic stiffness (for contactmodel = 2), N/m
        self.k_t = numpy.ones(1, dtype=numpy.float64) * k_t

        # Contact rolling elastic stiffness (for contactmodel = 2), N/m
        self.k_r = numpy.ones(1, dtype=numpy.float64) * k_r

        # Contact normal viscosity. Critical damping: 2*sqrt(m*k_n).
        # Normal force component elastic if nu = 0.0.
        #self.gamma_n = numpy.ones(self.np, dtype=numpy.float64) \
                #          * nu_frac * 2.0 * math.sqrt(4.0/3.0 * math.pi * numpy.amin(self.radius)**3 \
                #          * self.rho[0] * self.k_n[0])
        self.gamma_n = numpy.ones(1, dtype=numpy.float64) * gamma_n

        # Contact shear viscosity, Ns/m
        self.gamma_t = numpy.ones(1, dtype=numpy.float64) * gamma_t

        # Contact rolling viscosity, Ns/m?
        self.gamma_r = numpy.ones(1, dtype=numpy.float64) * gamma_r

        # Contact static shear friction coefficient
        #self.mu_s = numpy.ones(1, dtype=numpy.float64) * numpy.tan(numpy.radians(ang_s))
        self.mu_s = numpy.ones(1, dtype=numpy.float64) * mu_s

        # Contact dynamic shear friction coefficient
        #self.mu_d = numpy.ones(1, dtype=numpy.float64) * numpy.tan(numpy.radians(ang_d))
        self.mu_d = numpy.ones(1, dtype=numpy.float64) * mu_d

        # Contact rolling friction coefficient
        #self.mu_r = numpy.ones(1, dtype=numpy.float64) * numpy.tan(numpy.radians(ang_r))
        self.mu_r = numpy.ones(1, dtype=numpy.float64) * mu_r

        # Wall viscosities
        self.gamma_wn[0] = gamma_wn # normal
        self.gamma_wt[0] = gamma_wt # sliding

        # Wall friction coefficients
        self.mu_ws = self.mu_s  # static
        self.mu_wd = self.mu_d  # dynamic

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


    def bond(self, i, j):
        """ Create a bond between particles i and j """

        self.lambda_bar[0] = 1.0 # Radius multiplier to parallel-bond radii
        
        if (hasattr(self, 'bonds') == False):
            self.bonds = numpy.array([[i,j]], dtype=numpy.uint32)
        else :
            self.bonds = numpy.vstack((self.bonds, [i,j]))

        if (hasattr(self, 'bonds_delta_n') == False):
            self.bonds_delta_n = numpy.array([0.0], dtype=numpy.uint32)
        else :
            #self.bonds_delta_n = numpy.vstack((self.bonds_delta_n, [0.0]))
            self.bonds_delta_n = numpy.append(self.bonds_delta_n, [0.0])

        if (hasattr(self, 'bonds_delta_t') == False):
            self.bonds_delta_t = numpy.array([[0.0, 0.0, 0.0]], dtype=numpy.uint32)
        else :
            self.bonds_delta_t = numpy.vstack((self.bonds_delta_t, [0.0, 0.0, 0.0]))

        if (hasattr(self, 'bonds_omega_n') == False):
            self.bonds_omega_n = numpy.array([0.0], dtype=numpy.uint32)
        else :
            #self.bonds_omega_n = numpy.vstack((self.bonds_omega_n, [0.0]))
            self.bonds_omega_n = numpy.append(self.bonds_omega_n, [0.0])

        if (hasattr(self, 'bonds_omega_t') == False):
            self.bonds_omega_t = numpy.array([[0.0, 0.0, 0.0]], dtype=numpy.uint32)
        else :
            self.bonds_omega_t = numpy.vstack((self.bonds_omega_t, [0.0, 0.0, 0.0]))

        # Increment the number of bonds with one
        self.nb0 += 1


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

        elif method == 'visc_n':
            return numpy.sum(self.ev)

        elif method == 'visc_n_rate':
            return numpy.sum(self.ev_dot)

        elif method == 'bondpot':
            R_bar = self.lambda_bar * numpy.min(self.radius[self.bonds[:,0]], self.radius[self.bonds[:,1]])
            A = numpy.pi * R_bar**2
            I = 0.25 * numpy.pi * R_bar**4
            J = I*2.0
            bondpot_fn = numpy.sum(0.5 * A * self.k_n * numpy.abs(self.bonds_delta_n)**2)
            bondpot_ft = numpy.sum(0.5 * A * self.k_t * numpy.linalg.norm(self.bonds_delta_t)**2)
            bondpot_tn = numpy.sum(0.5 * J * self.k_t * numpy.abs(self.bonds_omega_n)**2)
            bondpot_tt = numpy.sum(0.5 * I * self.k_n * numpy.linalg.norm(self.bonds_omega_t)**2)
            return bondpot_fn + bondpot_ft + bondpot_tn + bondpot_tt

    def voidRatio(self):
        'Returns the current void ratio'

        # Find the bulk volume
        V_t = (self.L[0] - self.origo[0]) \
                *(self.L[1] - self.origo[1]) \
                *(self.w_x[0] - self.origo[2])

        # Find the volume of solids
        V_s = numpy.sum(4.0/3.0 * math.pi * self.radius**3)

        # Return the void ratio
        e = (V_t - V_s)/V_s
        return e


    def bulkPorosity(self):
        """ Calculate and return the bulk porosity """

        if (self.nw == 0):
            V_total = self.L[0] * self.L[1] * self.L[2]
        elif (self.nw == 1):
            V_total = self.L[0] * self.L[1] * self.w_x[0]
            if (V_total <= 0.0):
                raise Exception("Could not determine total volume")

        # Find the volume of solids
        V_solid = numpy.sum(V_sphere(self.radius))
        return (V_total - V_solid) / V_total
   

    def porosity(self,
            slices = 10,
            verbose = False):
        """ Calculate the porosity as a function of depth, by averaging values
            in horizontal slabs.
            Returns porosity values and depth
        """

        # Write data as binary
        self.writebin(verbose=False)

        # Run porosity program on binary
        pipe = subprocess.Popen(\
                ["../porosity",\
                "-s","{}".format(slices),\
                "../input/" + self.sid + ".bin"],\
                stdout=subprocess.PIPE)
        output, err = pipe.communicate()

        if (err):
            print(err)
            raise Exception("Could not run external 'porosity' program")

        # read one line of output at a time
        s2 = output.split('\n')
        depth = []
        porosity = []
        for row in s2:
            if (row != '\n' or row != '' or row != ' '): # skip blank lines
                s3 = row.split('\t')
                if (s3 != '' and len(s3) == 2): # make sure line has two vals
                    depth.append(float(s3[0]))
                    porosity.append(float(s3[1]))

        return numpy.array(porosity), numpy.array(depth)


    def run(self, verbose=True, hideinputfile=False, dry=False, valgrind=False, cudamemcheck=False):
        'Execute sphere with target project'

        quiet = ""
        stdout = ""
        dryarg = ""
        valgrindbin = ""
        cudamemchk = ""
        if (verbose == False):
            quiet = "-q "
        if (hideinputfile == True):
            stdout = " > /dev/null"
        if (dry == True):
            dryarg = "--dry "
        if (valgrind == True):
            valgrindbin = "valgrind -q "
        if (cudamemcheck == True):
            cudamemchk = "cuda-memcheck "


        subprocess.call("cd ..; " + valgrindbin + cudamemchk + "./sphere " + quiet + dryarg + "input/" + self.sid + ".bin " + stdout, shell=True)


    def torqueScript(self, 
            email="adc@geo.au.dk", 
            email_alerts="ae",
            walltime="8:00:00"):
        'Create job script for the Torque queue manager for the binary'

        filename = self.sid + ".sh"
        fh = None
        try :
            fh = open(filename, "w")

            # write stuff

        finally :
            if fh is not None:
                fh.close()


    def render(self,
            method = "pres",
            max_val = 1e3,
            lower_cutoff = 0.0,
            graphicsformat = "png",
            verbose=True):
        'Render all output files that belong to the simulation, determined by sid.'

        quiet = ""
        if (verbose == False):
            quiet = "-q"

        # Render images using sphere raytracer
        if (method == "normal"):
            subprocess.call("cd ..; for F in `ls output/" + self.sid + "*.bin`; do ./sphere " + quiet \
                    + " --render $F; done" \
                    , shell=True)
        else :
            subprocess.call("cd ..; for F in `ls output/" + self.sid + "*.bin`; do ./sphere " + quiet \
                    + " --method " + method + " {}".format(max_val) \
                    + " -l {}".format(lower_cutoff) \
                    + " --render $F; done" \
                    , shell=True)

        # Convert images to compressed format
        convert()

    def shearvel(self):
        'Calculates and returns the shear velocity (gamma_dot) of the experiment'

        # Find the fixed particles
        fixvel = numpy.nonzero(self.fixvel > 0.0)

        # The shear velocity is the x-axis velocity value of the upper particles
        return self.vel[fixvel,0].max()


    def shearstrain(self):
        'Calculates and returns the shear strain (gamma) value of the experiment'

        # Current height
        w_x0 = self.w_x[0]

        # Displacement of the upper, fixed particles in the shear direction
        xdisp = self.time_current[0] * self.shearvel()

        # Return shear strain
        return xdisp/w_x0

    def thinsection_x1x3(self,
            x2 = 'center',
            graphicsformat = 'png',
            cbmax = None,
            arrowscale = 0.01,
            velarrowscale = 1.0,
            slipscale = 1.0,
            verbose = False):
        ''' Produce a 2D image of particles on a x1,x3 plane, intersecting the second axis at x2.
            Output is saved as '<sid>-ts-x1x3.txt' in the current folder.

            An upper limit to the pressure color bar range can be set by the cbmax parameter.

            The data can be plotted in gnuplot with:
                gnuplot> set size ratio -1
                gnuplot> set palette defined (0 "blue", 0.5 "gray", 1 "red")
                gnuplot> plot '<sid>-ts-x1x3.txt' with circles palette fs transparent solid 0.4 noborder
        '''

        if (x2 == 'center') :
            x2 = (self.L[1] - self.origo[1]) / 2.0

        # Initialize plot circle positionsr, radii and pressures
        ilist = []
        xlist = []
        ylist = []
        rlist = []
        plist = []
        pmax = 0.0
        rmax = 0.0
        axlist = []
        aylist = []
        daxlist = []
        daylist = []
        dvxlist = []
        dvylist = []

        # Loop over all particles, find intersections
        for i in range(self.np):
            
            delta = abs(self.x[i,1] - x2)   # distance between centre and plane

            if (delta < self.radius[i]): # if the sphere intersects the plane

                # Store particle index
                ilist.append(i)
                
                # Store position on plane
                xlist.append(self.x[i,0])
                ylist.append(self.x[i,2])

                # Store radius of intersection
                r_circ = math.sqrt(self.radius[i]**2 - delta**2)
                if (r_circ > rmax):
                    rmax = r_circ
                rlist.append(r_circ)

                # Store pressure
                pval = self.p[i]
                if (cbmax != None):
                    if (pval > cbmax):
                        pval = cbmax
                plist.append(pval)

                # Store rotational velocity data for arrows
                # Save two arrows per particle
                axlist.append(self.x[i,0]) # x starting point of arrow
                axlist.append(self.x[i,0]) # x starting point of arrow
                aylist.append(self.x[i,2] + r_circ*0.5) # y starting point of arrow
                aylist.append(self.x[i,2] - r_circ*0.5) # y starting point of arrow
                daxlist.append(self.angvel[i,1]*arrowscale) # delta x for arrow end point
                daxlist.append(-self.angvel[i,1]*arrowscale) # delta x for arrow end point
                daylist.append(0.0) # delta y for arrow end point
                daylist.append(0.0) # delta y for arrow end point

                # Store linear velocity data
                dvxlist.append(self.vel[i,0]*velarrowscale) # delta x for arrow end point
                dvylist.append(self.vel[i,2]*velarrowscale) # delta y for arrow end point

                if (r_circ > self.radius[i]):
                    raise Exception("Error, circle radius is larger than the particle radius")
                if (self.p[i] > pmax):
                    pmax = self.p[i]

        if (verbose == True):
            print("Max. pressure of intersecting spheres: " + str(pmax) + " Pa")
            if (cbmax != None):
                print("Value limited to: " + str(cbmax) + " Pa")

        # Save circle data
        filename = '../gnuplot/data/' + self.sid + '-ts-x1x3.txt'
        fh = None
        try :
            fh = open(filename, 'w')

            for (x, y, r, p) in zip(xlist, ylist, rlist, plist):
                fh.write("{}\t{}\t{}\t{}\n".format(x, y, r, p))
        
        finally :
            if fh is not None:
                fh.close()

        # Save angular velocity data. The arrow lengths are normalized to max. radius
        #   Output format: x, y, deltax, deltay
        #   gnuplot> plot '-' using 1:2:3:4 with vectors head filled lt 2
        filename = '../gnuplot/data/' + self.sid + '-ts-x1x3-arrows.txt'
        fh = None
        try :
            fh = open(filename, 'w')

            for (ax, ay, dax, day) in zip(axlist, aylist, daxlist, daylist):
                fh.write("{}\t{}\t{}\t{}\n".format(ax, ay, dax, day))
        
        finally :
            if fh is not None:
                fh.close()

        # Save linear velocity data
        #   Output format: x, y, deltax, deltay
        #   gnuplot> plot '-' using 1:2:3:4 with vectors head filled lt 2
        filename = '../gnuplot/data/' + self.sid + '-ts-x1x3-velarrows.txt'
        fh = None
        try :
            fh = open(filename, 'w')

            for (x, y, dvx, dvy) in zip(xlist, ylist, dvxlist, dvylist):
                fh.write("{}\t{}\t{}\t{}\n".format(x, y, dvx, dvy))
        
        finally :
            if fh is not None:
                fh.close()



        # Check whether there are slips between the particles intersecting the plane
        sxlist = []
        sylist = []
        dsxlist = []
        dsylist = []
        anglelist = [] # angle of the slip vector
        slipvellist = [] # velocity of the slip
        for i in ilist:

            # Loop through other particles, and check whether they are in contact
            for j in ilist:
                #if (i < j):
                if (i != j):

                    # positions
                    x_i = self.x[i,:]
                    x_j = self.x[j,:]

                    # radii
                    r_i = self.radius[i]
                    r_j = self.radius[j]

                    # Inter-particle vector
                    x_ij = x_i - x_j
                    x_ij_length = numpy.sqrt(x_ij.dot(x_ij))

                    # Check for overlap
                    if (x_ij_length - (r_i + r_j) < 0.0):

                        # contact plane normal vector
                        n_ij = x_ij / x_ij_length

                        vel_i = self.vel[i,:]
                        vel_j = self.vel[j,:]
                        angvel_i = self.angvel[i,:]
                        angvel_j = self.angvel[j,:]

                        # Determine the tangential contact surface velocity in the x,z plane
                        dot_delta = (vel_i - vel_j) \
                                + r_i * numpy.cross(n_ij, angvel_i) \
                                + r_j * numpy.cross(n_ij, angvel_j)

                        # Subtract normal component to get tangential velocity
                        dot_delta_n = n_ij * numpy.dot(dot_delta, n_ij)
                        dot_delta_t = dot_delta - dot_delta_n

                        # Save slip velocity data for gnuplot
                        if (dot_delta_t[0] != 0.0 or dot_delta_t[2] != 0.0):

                            # Center position of the contact
                            cpos = x_i - x_ij * 0.5

                            sxlist.append(cpos[0])
                            sylist.append(cpos[2])
                            dsxlist.append(dot_delta_t[0] * slipscale)
                            dsylist.append(dot_delta_t[2] * slipscale)
                            #anglelist.append(math.degrees(math.atan(dot_delta_t[2]/dot_delta_t[0])))
                            anglelist.append(math.atan(dot_delta_t[2]/dot_delta_t[0]))
                            slipvellist.append(numpy.sqrt(dot_delta_t.dot(dot_delta_t)))


        # Write slip lines to text file
        filename = '../gnuplot/data/' + self.sid + '-ts-x1x3-slips.txt'
        fh = None
        try :
            fh = open(filename, 'w')

            for (sx, sy, dsx, dsy) in zip(sxlist, sylist, dsxlist, dsylist):
                fh.write("{}\t{}\t{}\t{}\n".format(sx, sy, dsx, dsy))
        
        finally :
            if fh is not None:
                fh.close()

        # Plot thinsection with gnuplot script
        gamma = self.shearstrain()
        subprocess.call("""cd ../gnuplot/scripts && gnuplot -e "sid='{}'; gamma='{:.4}'; xmin='{}'; xmax='{}'; ymin='{}'; ymax='{}'" plotts.gp""".format(\
                self.sid, self.shearstrain(), self.origo[0], self.L[0], self.origo[2], self.L[2]), shell=True)

        # Find all particles who have a slip velocity higher than slipvel
        slipvellimit = 0.01
        slipvels = numpy.nonzero(numpy.array(slipvellist) > slipvellimit)


        # Bin slip angle data for histogram
        binno = 36/2
        hist_ang, bins_ang = numpy.histogram(numpy.array(anglelist)[slipvels], bins=binno, density=False)
        center_ang = (bins_ang[:-1] + bins_ang[1:]) / 2.0

        center_ang_mirr = numpy.concatenate((center_ang, center_ang + math.pi))
        hist_ang_mirr = numpy.tile(hist_ang, 2)

        # Write slip angles to text file
        #numpy.savetxt(self.sid + '-ts-x1x3-slipangles.txt', zip(center_ang, hist_ang), fmt="%f\t%f")

        fig = plt.figure()
        ax = fig.add_subplot(111, polar=True)
        ax.bar(center_ang_mirr, hist_ang_mirr, width=30.0/180.0)
        fig.savefig('../img_out/' + self.sid + '-ts-x1x3-slipangles.png')
        fig.clf()


def convert(graphicsformat = "png",
        folder = "../img_out"):
    'Converts all PPM images in img_out to graphicsformat, using ImageMagick'

    #quiet = " > /dev/null"
    quiet = ""
    # Convert images
    subprocess.call("for F in " + folder \
            + "/*.ppm ; do BASE=`basename $F .ppm`; convert $F " \
            + folder + "/$BASE." + graphicsformat + " " \
            + quiet + " ; done", shell=True)

    # Remove PPM files
    subprocess.call("rm " + folder + "/*.ppm", shell=True)


def render(binary,
        method = "pres",
        max_val = 1e3,
        lower_cutoff = 0.0,
        graphicsformat = "png",
        verbose=True):
    'Render target binary using the sphere raytracer.'

    quiet = ""
    if (verbose == False):
        quiet = "-q"

    # Render images using sphere raytracer
    if (method == "normal"):
        subprocess.call("cd .. ; ./sphere " + quiet + \
                " --render " + binary, shell=True)
    else :
        subprocess.call("cd .. ; ./sphere " + quiet + \
                " --method " + method + " {}".format(max_val) + \
                " -l {}".format(lower_cutoff) + \
                " --render " + binary, shell=True)

    # Convert images to compressed format
    convert()


def video(project,
        out_folder = "./",
        video_format = "mp4",
        graphics_folder = "../img_out/",
        graphics_format = "png",
        fps = 25,
        qscale = 1,
        bitrate = 1800,
        verbose = False):
    'Use ffmpeg to combine images to animation. All images should be rendered beforehand.'

    # Possible loglevels: quiet, panic, fatal, error, warning, info, verbose, debug
    loglevel = "info" # verbose = True
    if (verbose == False):
        loglevel = "error"

    subprocess.call(\
            "ffmpeg -qscale {0} -r {1} -b {2} -y ".format(qscale, fps, bitrate) \
            + "-loglevel " + loglevel + " " \
            + "-i " + graphics_folder + project + ".output%05d." + graphics_format + " " \
            + out_folder + "/" + project + "." + video_format, shell=True)

def thinsectionVideo(project,
        out_folder = "./",
        video_format = "mp4",
        fps = 25,
        qscale = 1,
        bitrate = 1800,
        verbose = False):
    ''' Use ffmpeg to combine thin section images to animation.
        This function will start off by rendering the images.
    '''

    # Render thin section images (png)
    lastfile = status(project)
    sb = Spherebin()
    for i in range(lastfile+1):
        fn = "../output/{0}.output{1:0=5}.bin".format(project, i)
        sb.sid = project + ".output{:0=5}".format(i)
        sb.readbin(fn, verbose = False)
        sb.thinsection_x1x3(cbmax = sb.w_devs[0]*4.0)

    # Combine images to animation
    # Possible loglevels: quiet, panic, fatal, error, warning, info, verbose, debug
    loglevel = "info" # verbose = True
    if (verbose == False):
        loglevel = "error"

    subprocess.call(\
            "ffmpeg -qscale {0} -r {1} -b {2} -y ".format(qscale, fps, bitrate) \
            + "-loglevel " + loglevel + " " \
            + "-i ../img_out/" + project + ".output%05d-ts-x1x3.png " \
            + "-vf 'crop=((in_w/2)*2):((in_h/2)*2)' " \
            + out_folder + "/" + project + "-ts-x1x3." + video_format, \
            shell=True)

def visualize(project, method = 'energy', savefig = True, outformat = 'png'):
    """ Visualize output from the target project,
        where the temporal progress is of interest.
    """

    lastfile = status(project)

    ### Plotting
    if (outformat != 'txt'):
        fig = plt.figure(figsize=(15,10),dpi=300)
        figtitle = "{0}, simulation {1}".format(method, project)
        fig.text(0.5,0.95,figtitle,horizontalalignment='center',fontproperties=FontProperties(size=18))


    if method == 'energy':

        # Allocate arrays
        Epot = numpy.zeros(lastfile+1)
        Ekin = numpy.zeros(lastfile+1)
        Erot = numpy.zeros(lastfile+1)
        Es  = numpy.zeros(lastfile+1)
        Ev  = numpy.zeros(lastfile+1)
        Es_dot = numpy.zeros(lastfile+1)
        Ev_dot = numpy.zeros(lastfile+1)
        Ebondpot = numpy.zeros(lastfile+1)
        Esum = numpy.zeros(lastfile+1)

        # Read energy values from project binaries
        sb = Spherebin()
        for i in range(lastfile+1):
            fn = "../output/{0}.output{1:0=5}.bin".format(project, i)
            sb.readbin(fn, verbose = False)

            Epot[i] = sb.energy("pot")
            Ekin[i] = sb.energy("kin")
            Erot[i] = sb.energy("rot")
            Es[i]   = sb.energy("shear")
            Ev[i]   = sb.energy("visc_n")
            Es_dot[i] = sb.energy("shearrate")
            Ev_dot[i] = sb.energy("visc_n_rate")
            Ebondpot[i] = sb.energy("bondpot")
            Esum[i] = Epot[i] + Ekin[i] + Erot[i] + Es[i] + Ev[i] + Ebondpot[i]

            t = numpy.linspace(0.0, sb.time_current, lastfile+1)

        if (outformat != 'txt'):
            # Potential energy
            ax1 = plt.subplot2grid((2,5),(0,0))
            ax1.set_xlabel('Time [s]')
            ax1.set_ylabel('Total potential energy [J]')
            ax1.plot(t, Epot, '+-')
            ax1.grid()

            # Kinetic energy
            ax2 = plt.subplot2grid((2,5),(0,1))
            ax2.set_xlabel('Time [s]')
            ax2.set_ylabel('Total kinetic energy [J]')
            ax2.plot(t, Ekin, '+-')
            ax2.grid()

            # Rotational energy
            ax3 = plt.subplot2grid((2,5),(0,2))
            ax3.set_xlabel('Time [s]')
            ax3.set_ylabel('Total rotational energy [J]')
            ax3.plot(t, Erot, '+-')
            ax3.grid()

            # Bond energy
            ax4 = plt.subplot2grid((2,5),(0,3))
            ax4.set_xlabel('Time [s]')
            ax4.set_ylabel('Bond energy [J]')
            ax4.plot(t, Ebondpot, '+-')
            ax4.grid()

            # Total energy
            ax5 = plt.subplot2grid((2,5),(0,4))
            ax5.set_xlabel('Time [s]')
            ax5.set_ylabel('Total energy [J]')
            ax5.plot(t, Esum, '+-')
            ax5.grid()

            # Shear energy rate
            ax6 = plt.subplot2grid((2,5),(1,0))
            ax6.set_xlabel('Time [s]')
            ax6.set_ylabel('Frictional dissipation rate [W]')
            ax6.plot(t, Es_dot, '+-')
            ax6.grid()

            # Shear energy
            ax7 = plt.subplot2grid((2,5),(1,1))
            ax7.set_xlabel('Time [s]')
            ax7.set_ylabel('Total frictional dissipation [J]')
            ax7.plot(t, Es, '+-')
            ax7.grid()

            # Visc_n energy rate
            ax8 = plt.subplot2grid((2,5),(1,2))
            ax8.set_xlabel('Time [s]')
            ax8.set_ylabel('Viscous dissipation rate [W]')
            ax8.plot(t, Ev_dot, '+-')
            ax8.grid()

            # Visc_n energy
            ax9 = plt.subplot2grid((2,5),(1,3))
            ax9.set_xlabel('Time [s]')
            ax9.set_ylabel('Total viscous dissipation [J]')
            ax9.plot(t, Ev, '+-')
            ax9.grid()


            # Combined view
            ax10 = plt.subplot2grid((2,5),(1,4))
            ax10.set_xlabel('Time [s]')
            ax10.set_ylabel('Energy [J]')
            ax10.plot(t, Epot, '+-g')
            ax10.plot(t, Ekin, '+-b')
            ax10.plot(t, Erot, '+-r')
            ax10.legend(('$\sum E_{pot}$','$\sum E_{kin}$','$\sum E_{rot}$'), 'upper right', shadow=True)
            ax10.grid()

    elif method == 'walls':

        # Read energy values from project binaries
        sb = Spherebin()
        for i in range(lastfile+1):
            fn = "../output/{0}.output{1:0=5}.bin".format(project, i)
            sb.readbin(fn, verbose = False)

            # Allocate arrays on first run
            if (i == 0):
                wforce = numpy.zeros((lastfile+1)*sb.nw[0], dtype=numpy.float64).reshape((lastfile+1), sb.nw[0])
                wvel   = numpy.zeros((lastfile+1)*sb.nw[0], dtype=numpy.float64).reshape((lastfile+1), sb.nw[0])
                wpos   = numpy.zeros((lastfile+1)*sb.nw[0], dtype=numpy.float64).reshape((lastfile+1), sb.nw[0])
                wdevs  = numpy.zeros((lastfile+1)*sb.nw[0], dtype=numpy.float64).reshape((lastfile+1), sb.nw[0])
                maxpos = numpy.zeros((lastfile+1), dtype=numpy.float64)
                logstress = numpy.zeros((lastfile+1), dtype=numpy.float64)
                voidratio = numpy.zeros((lastfile+1), dtype=numpy.float64)

            wforce[i] = sb.w_force[0]
            wvel[i]   = sb.w_vel[0]
            wpos[i]   = sb.w_x[0]
            wdevs[i]  = sb.w_devs[0]
            maxpos[i] = numpy.max(sb.x[:,2]+sb.radius)
            logstress[i] = numpy.log((sb.w_force[0]/(sb.L[0]*sb.L[1]))/1000.0)
            voidratio[i] = sb.voidRatio()


        t = numpy.linspace(0.0, sb.time_current, lastfile+1)

        # Plotting
        if (outformat != 'txt'):
            # linear plot of time vs. wall position
            ax1 = plt.subplot2grid((2,2),(0,0))
            ax1.set_xlabel('Time [s]')
            ax1.set_ylabel('Position [m]')
            ax1.plot(t, wpos, '+-', label="upper wall")
            ax1.plot(t, maxpos, '+-', label="heighest particle")
            ax1.legend()
            ax1.grid()

            #ax2 = plt.subplot2grid((2,2),(1,0))
            #ax2.set_xlabel('Time [s]')
            #ax2.set_ylabel('Force [N]')
            #ax2.plot(t, wforce, '+-')

            # semilog plot of log stress vs. void ratio
            ax2 = plt.subplot2grid((2,2),(1,0))
            ax2.set_xlabel('log deviatoric stress [kPa]')
            ax2.set_ylabel('Void ratio [-]')
            ax2.plot(logstress, voidratio, '+-')
            ax2.grid()

            # linear plot of time vs. wall velocity
            ax3 = plt.subplot2grid((2,2),(0,1))
            ax3.set_xlabel('Time [s]')
            ax3.set_ylabel('Velocity [m/s]')
            ax3.plot(t, wvel, '+-')
            ax3.grid()

            # linear plot of time vs. deviatoric stress
            ax4 = plt.subplot2grid((2,2),(1,1))
            ax4.set_xlabel('Time [s]')
            ax4.set_ylabel('Deviatoric stress [Pa]')
            ax4.plot(t, wdevs, '+-', label="$\sigma_0$")
            ax4.plot(t, wforce/(sb.L[0]*sb.L[1]), '+-', label="$\sigma'$")
            ax4.legend(loc=4)
            ax4.grid()


    elif method == 'shear':

        sb = Spherebin()
        # Read stress values from project binaries
        for i in range(lastfile+1):
            fn = "../output/{0}.output{1:0=5}.bin".format(project, i)
            sb.readbin(fn, verbose = False)

            # First iteration: Allocate arrays and find constant values
            if (i == 0):
                xdisp     = numpy.zeros(lastfile+1, dtype=numpy.float64)  # Shear displacement
                sigma_eff = numpy.zeros(lastfile+1, dtype=numpy.float64)  # Normal stress
                sigma_def = numpy.zeros(lastfile+1, dtype=numpy.float64)  # Normal stress
                tau       = numpy.zeros(lastfile+1, dtype=numpy.float64)  # Shear stress
                dilation  = numpy.zeros(lastfile+1, dtype=numpy.float64)  # Upper wall position
                tau_u = 0.0             # Peak shear stress
                tau_u_shearstrain = 0.0 # Shear strain value of peak shear stress

                fixvel = numpy.nonzero(sb.fixvel > 0.0)
                #fixvel_upper = numpy.nonzero(sb.vel[fixvel,0] > 0.0)
                shearvel = sb.vel[fixvel,0].max()
                w_x0 = sb.w_x[0]        # Original height
                A = sb.L[0] * sb.L[1]   # Upper surface area

            # Summation of shear stress contributions
            for j in fixvel[0]:
                if (sb.vel[j,0] > 0.0):
                    tau[i] += -sb.force[j,0]

            xdisp[i]    = sb.time_current[0] * shearvel
            sigma_eff[i] = sb.w_force[0] / A
            sigma_def[i] = sb.w_devs[0]
            dilation[i] = sb.w_x[0] - w_x0                 # dilation in meters
            #dilation[i] = (sb.w_x[0] - w_x0)/w_x0 * 100.0   # dilation in percent

            # Test if this was the max. shear stress
            if (tau[i] > tau_u):
                tau_u = tau[i]
                tau_u_shearstrain = xdisp[i]/w_x0


        # Plot stresses
        if (outformat != 'txt'):
            shearinfo = "$\\tau_u$ = {:.3} Pa at $\gamma$ = {:.3}".format(tau_u, tau_u_shearstrain)
            fig.text(0.5,0.03,shearinfo,horizontalalignment='center',fontproperties=FontProperties(size=14))
            ax1 = plt.subplot2grid((2,1),(0,0))
            ax1.set_xlabel('Shear strain [-]')
            ax1.set_ylabel('Stress [Pa]')
            ax1.plot(xdisp/w_x0, sigma_eff, '+-g', label="$\sigma'$")
            ax1.plot(xdisp/w_x0, sigma_def, '+-b', label="$\sigma_0$")
            ax1.plot(xdisp/w_x0, tau, '+-r', label="$\\tau$")
            ax1.legend(loc = 4)
            ax1.grid()

            # Plot dilation
            ax2 = plt.subplot2grid((2,1),(1,0))
            ax2.set_xlabel('Shear strain [-]')
            ax2.set_ylabel('Dilation [m]')
            ax2.plot(xdisp/w_x0, dilation, '+-')
            ax2.grid()

        else : 
            # Write values to textfile
            filename = "shear-stresses-{0}.txt".format(project)
            #print("Writing stress data to " + filename)
            fh = None
            try :
                fh = open(filename, "w")
                L = sb.L[2] - sb.origo[2] # Initial height
                for i in range(lastfile+1):
                    # format: shear distance [mm], sigma [kPa], tau [kPa], Dilation [%]
                    fh.write("{0}\t{1}\t{2}\t{3}\n".format(xdisp[i],
                    sigma_eff[i]/1000.0,
                    tau[i]/1000.0,
                    dilation[i]))
            finally :
                if fh is not None:
                    fh.close()

    else :
        print("Visualization type '" + method + "' not understood")


    # Optional save of figure
    if (outformat != 'txt'):
        if (savefig == True):
            fig.savefig("{0}-{1}.{2}".format(project, method, outformat))
            fig.clf()
        else :
            plt.show()

def run(binary, verbose=True, hideinputfile=False):
    'Execute sphere with target binary as input'

    quiet = ""
    stdout = ""
    if (verbose == False):
        quiet = "-q"
    if (hideinputfile == True):
        stdout = " > /dev/null"
    subprocess.call("cd ..; ./sphere " + quiet + " " + binary + " " + stdout, shell=True)



def status(project):
    """ Check the status.dat file for the target project,
        and return the last file numer.
    """
    
    fh = None
    try :
        filepath = "../output/{0}.status.dat".format(project)
        #print(filepath)
        fh = open(filepath)
        data = fh.read()
        #print(data)
        return int(data.split()[2])  # Return last file number
    finally :
        if fh is not None:
            fh.close()

def cleanup(spherebin):
    'Remove input/output files and images from simulation'
    subprocess.call("rm -f ../input/" + spherebin.sid + ".bin", shell=True)
    subprocess.call("rm -f ../output/" + spherebin.sid + ".*.bin", shell=True)
    subprocess.call("rm -f ../img_out/" + spherebin.sid + ".*", shell=True)

def V_sphere(r):
    """ Returns the volume of a sphere with radius r
    """
    return 4.0/3.0 * math.pi * r**3.0

# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
