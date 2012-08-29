#!/usr/bin/env python

# Import sphere functionality
from sphere import *
import subprocess

# Determine system hostname
import os; 
hostname = os.uname()[1]

# Simulation ID
sid = "1e4-largesize-uniaxial"

# Delete previous output
subprocess.call("rm ../{img_out,output}/" + sid + "*", shell=True);


#### GRAVITATIONAL CONSOLIDATION

# New class
init = Spherebin(np = 1e4)

# Generate random radii
init.generateRadii(psd = 'logn', histogram = 1, radius_mean = 0.02)

# Initialize particle parameters
# Values from Yohannes et al. 2012
init.defaultParams(ang_s = 28,
    		   ang_d = 28,
		   ang_r = 0,
		   rho = 2600,
		   k_n = 1.16e9,
		   k_t = 1.16e9,
		   gamma_n = 0.0,
		   gamma_t = 0.0,
		   gamma_r = 0.0)

# Place particles in grid-like arrangement
#   periodic: 0 = frictional x- and y-walls
#   periodic: 1 = periodic x- and y-boundaries
#   periodic: 2 = periodic x boundaries, frictional y-walls
#   shearmodel: 1 = viscous, frictional shear force model
#   shearmodel: 2 = elastic, frictional shear force model
init.initRandomGridPos(gridnum = numpy.array([20,10,600]), periodic = 1, shearmodel = 2)

# Initialize temporal parameters
init.initTemporal(total = 5.0, file_dt = 0.10)

# Write output binary for sphere
init.writebin("../input/" + sid + "-initgrid.bin".format(sid))

# Render start configuration
render("../input/" + sid + "-initgrid.bin", out = sid + "-initgrid")

# Run simulation
subprocess.call("cd ..; ./sphere_*_X86_64 " + sid + "-initgrid", shell=True)

# Plot energy
visualize(sid + "-initgrid", "energy", savefig=True, outformat='png')


#### CONSOLIDATION UNDER DEVIATORIC STRESS

# New class
cons = Spherebin(np = init.np, nd = init.nd)

# Find out which output file was the last generated during gravitational consolidation
lastfile = status(sid + "-initgrid", verbose = False)
filename = "../output" + sid + "-initgrid.output{0}.bin".format(lastfile)


## Uniaxial compression loop:
#    1. Read last experiment binary
#    2. Set new devs and zero current time
#    3. Run sphere
#    4. Visualize wall data and find void ratio (e)
#    5. Save name of last binary written

# Define deviatoric stresses to test
stresses = numpy.array([10e3, 20e3, 40e3])
voidratio = numpy.zeros(1, length(stresses))

i = 0
for devs in stresses:

  # Simulation ID for consolidation run
  cid = sid + "-uniaxial-{0}Pa".format(devs)

  # Read the last output file of from the gravitational consolidation experiment
  cons.readbin(filename)

  # Setup consolidation experiment
  cons.consolidate(deviatoric_stress = devs, periodic = init.periodic)

  # Zero time variables and set new total time
  cons.initTemporal(total = 3.0, file_dt = 0.03)

  # Write output binary for sphere
  cons.writebin("../input/" + cid + ".bin")

  # Run simulation
  subprocess.call("cd ..; ./sphere_*_X86_64 " + cid, shell=True)

  # Plot energy and wall data
  visualize(cid, "energy", savefig=True, outformat='png')
  visualize(cid, "walls", savefig=True, outformat='png')

  # Find void ratio
  lastfile = status(cid, verbose = False)
  cons.readbin(cid)
  voidratio[i] = voidRatio()
  i = i+1

  # Name of last output file
  filename = "../output" + cid + ".output{0}.bin".format(lastfile)


# Save plot of measured compression curve
fig = plt.figure(figsize(15,10),dpi=300)
figtitle = "{0}, measured compression curve".format(sid)
fig.text(0.5,0.95,figtitle,horizontalalignment='center',fontproperties=FontProperties(size=18))
ax1 = plt.subplot2grid((1,1),(0,0))
ax1.set_xlabel('Stress [kPa]')
ax1.set_ylabel('Void ratio')
ax1.semilogx(stresses, voidratio, '+-')

# Save values to text file
fh = None
try:
  fh = open(sid + "-uniaxial.txt", "w")
  for i in range(length(stresses)):
    fh.write("{0}\t{1}\n".format(stresses[i], voidratio[i]))
  finally:
    if fh is not None:
      fh.close()

