#!/usr/bin/env python
"""
Example of two particles colliding.
Place script in sphere/python folder, and invoke with `python 2.7 collision.py`
"""

# Import the sphere module for setting up, running, and analyzing the
# experiment. We also need the numpy module when setting arrays in the sphere
# object.
import sphere
import numpy


### SIMULATION SETUP

# Create a sphere object with two preallocated particles and a simulation ID
SB = sphere.Spherebin(np = 2, sid = 'collision')

SB.radius[:] = 0.3 # set radii to 0.3 m

# Define the positions of the two particles
SB.x[0, :] = numpy.array([0.0, 0.0, 0.0])   # particle 1 (idx 0)
SB.x[1, :] = numpy.array([1.0, 0.0, 0.0])   # particle 2 (idx 1)

# The default velocity is [0,0,0]. Slam particle 1 into particle 2 by defining
# a positive x velocity for particle 1.
SB.vel[0, 0] = 1.0

# let's disable gravity in this simulation
GRAVITY = numpy.array([0.0, 0.0, 0.0])

# Set the world limits and the particle sorting grid. The particles need to stay
# within the world limits for the entire simulation, otherwise it will stop!
SB.initGridAndWorldsize(g = GRAVITY, margin = 10.0)

# Define the temporal parameters, e.g. the total time (total) and the file
# output interval (file_dt), both in seconds
SB.initTemporal(total = 2.0, file_dt = 0.1)

# Save the simulation as a input file for sphere
SB.writebin()

# Using a 'dry' run, the sphere main program will display important parameters.
# sphere will end after displaying these values.
SB.run(dry = True)


### RUNNING THE SIMULATION

# Start the simulation on the GPU from the sphere program
SB.run()


### ANALYSIS OF SIMULATION RESULTS

# Plot the system energy through time, image saved as collision-energy.png
sphere.visualize(SB.sid, method = 'energy')

# Render the particles using the built-in raytracer
SB.render()
