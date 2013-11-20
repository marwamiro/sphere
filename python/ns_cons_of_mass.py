#!/usr/bin/env python

# Import sphere functionality
import sphere
from sphere import visualize, status
import sys
import numpy
import pylab

### EXPERIMENT SETUP ###
initialization = True
consolidation  = True
#shearing       = True

figformat = 'pdf'

# Number of particles
#np = 1e4
np = 1e2

# Common simulation id
sim_id = "ns"

# Deviatoric stress [Pa]
#devs = 10e3
devslist = [10.0e3]

### INITIALIZATION ###

# New class
init = sphere.Spherebin(np = np, nd = 3, nw = 0, sid = sim_id + "-init")


# Save radii
init.generateRadii(radius_mean = 0.05, histogram=False)


# Use default params
init.defaultParams(mu_s = 0.4, mu_d = 0.4, nu = 8.9e-4)

# Initialize positions in random grid (also sets world size)
#init.initRandomGridPos(gridnum = numpy.array([80, 80, 1000]), periodic = 1, contactmodel = 1)
init.initRandomGridPos(gridnum = numpy.array([6, 6, 1000]), periodic = 1, contactmodel = 1)

# Set duration of simulation
#init.initTemporal(total = 2.5)
#init.time_file_dt[0] = 0.05
init.initTemporal(total = 0.05)
init.time_file_dt[0] = 0.005

# Small pertubation
#init.p_f[init.num[0]/2,init.num[1]/2,init.num[2]/2] = 2.0


# Write input file for sphere
init.writebin()

# Run sphere
init.run(dry=True)
init.run(cfd=True)

init.writeVTKall()

#if (plots == True):
    # Make a graph of energies
    #visualize(init.sid, "energy", savefig=True, outformat='png')

#if (rendering == True):
    # Render images with raytracer
    #init.render(method = "pres", max_val = 2.0*devs, verbose = False)

project = init.sid
lastfile = status(init.sid)
sb = sphere.Spherebin()
time = numpy.zeros(lastfile+1)
sum_op_f = numpy.zeros(lastfile+1)
for i in range(lastfile+1):
    fn = "../output/{0}.output{1:0=5}.bin".format(project, i)
    sb.sid = project + ".output{:0=5}".format(i)
    sb.readbin(fn, verbose = False)
    #for y in range(0,sb.num[1]):
        #sb.plotFluidDensities(y = y)
        #sb.plotFluidVelocities(y = y)

    time[i] = sb.time_current[0]
    sum_op_f[i] = sb.p_f.sum()

#stack = numpy.vstack((time,sum_op_f))
#numpy.savetxt("sum_op_f", stack)

# Set figure parameters
fig_width_pt = 246.0  # Get this from LaTeX using \showthe\columnwidth
#fig_width_pt = 50.0  # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27               # Convert pt to inch
golden_mean = (pylab.sqrt(5)-1.0)/2.0         # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # width in inches
fig_height = fig_width*golden_mean      # height in inches
fig_size =  [fig_width,fig_height]
params = {'backend': 'ps',
    'axes.labelsize': 10,
    'text.fontsize': 10,
    'legend.fontsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'text.usetex':
    True,
    'figure.figsize':
    fig_size}
pylab.rcParams.update(params)

# generate data
#x = pylab.arange(-2*pylab.pi,2*pylab.pi,0.01)
#y1 = pylab.sin(x)
#y2 = pylab.cos(x)


y = init.num[1]/2

t = 0
fn = "../output/{0}.output{1:0=5}.bin".format(project, t)
sb.sid = project + ".output{:0=5}".format(i)
sb.readbin(fn, verbose = False)
pylab.figure(1)
pylab.clf()
pylab.axes([0.125,0.2,0.95-0.125,0.95-0.2])
pylab.imshow(sb.p_f[:,y,:].T, origin='lower', interpolation='nearest',
        cmap=pylab.cm.Blues)
#imgplt.set_interpolation('nearest')
#pylab.set_interpolation('bicubic')
#imgplt.set_cmap('hot')
pylab.xlabel('$i_x$')
pylab.ylabel('$i_z$')
pylab.title('$t = {}$ s'.format(t*sb.time_file_dt[0]))
#cb = pylab.colorbar(orientation = 'horizontal', shrink=0.8, pad=0.23)
cb = pylab.colorbar(orientation = 'horizontal', shrink=0.8, pad=0.23, ticks=[sb.p_f[:,y,:].min(), sb.p_f[:,y,:].max()])
cb.ax.set_xticklabels([1.0, sb.p_f[:,y,:].max()])
cb.set_label('H [m]')

pylab.savefig('ns_cons_of_mass1.' + figformat)
pylab.clf()

t = 1
fn = "../output/{0}.output{1:0=5}.bin".format(project, t)
sb.sid = project + ".output{:0=5}".format(i)
sb.readbin(fn, verbose = False)
pylab.figure(1)
pylab.clf()
pylab.axes([0.125,0.2,0.95-0.125,0.95-0.2])
pylab.imshow(sb.p_f[:,y,:].T, origin='lower', interpolation='nearest',
        cmap=pylab.cm.Blues)
#pylab.set_interpolation('bicubic')
pylab.xlabel('$i_x$')
pylab.ylabel('$i_z$')
pylab.title('$t = {}$ s'.format(t*sb.time_file_dt[0]))
cb = pylab.colorbar(orientation = 'horizontal', shrink=0.8, pad=0.23, ticks=[sb.p_f[:,y,:].min(), sb.p_f[:,y,:].max()])
#cb.ax.set_xticklabels([1.0, sb.p_f[:,y,:].max()])
cb.ax.set_xticklabels([sb.p_f[:,y,:].min(), sb.p_f[:,y,:].max()])
cb.set_label('H [m]')
#pylab.tight_layout()
pylab.savefig('ns_cons_of_mass2.' + figformat)
pylab.clf()

t = init.status()
fn = "../output/{0}.output{1:0=5}.bin".format(project, t)
sb.sid = project + ".output{:0=5}".format(i)
sb.readbin(fn, verbose = False)
pylab.figure(1)
pylab.clf()
pylab.axes([0.125,0.2,0.95-0.125,0.95-0.2])
pylab.imshow(sb.p_f[:,y,:].T, origin='lower', interpolation='nearest',
        cmap=pylab.cm.Blues)
#pylab.set_interpolation('bicubic')
pylab.xlabel('$i_x$')
pylab.ylabel('$i_z$')
pylab.title('$t = {}$ s'.format(t*sb.time_file_dt[0]))
cb = pylab.colorbar(orientation = 'horizontal', shrink=0.8, pad=0.23, ticks=[sb.p_f[:,y,:].min(), sb.p_f[:,y,:].max()])
#cb.ax.set_xticklabels([1.0, sb.p_f[:,y,:].max()])
cb.ax.set_xticklabels([sb.p_f[:,y,:].min(), sb.p_f[:,y,:].max()])
cb.set_label('H [m]')
#pylab.tight_layout()
pylab.savefig('ns_cons_of_mass3.' + figformat)
pylab.clf()

#pylab.axes([0.125,0.2,0.95-0.125,0.95-0.2])
pylab.axes([0.20,0.2,0.95-0.20,0.95-0.2])
pylab.plot(time, sum_op_f, '-k')
pylab.xlabel('$t$ [s]')
pylab.ylabel('$\sum H_i$ [m]')
pylab.xlim([0,time.max()])
pylab.ylim([0,sum_op_f.max()*1.1])
#pylab.legend()
#pylab.tight_layout()
pylab.grid()
pylab.savefig('ns_cons_of_mass4.' + figformat)
pylab.clf()

#pylab.tight_layout(h_pad=6.0)
#pylab.savefig('ns_cons_of_mass.eps')

#fig = matplotlib.pyplot.figure(figsize=(10,5),dpi=300)
#ax = matplotlib.pyplot.subplot2grid((1, 1), (0, 0))
#ax.plot(time, sum_op_f)
#fig.savefig("ns_cons_of_mass.png")
#fig.clf()
