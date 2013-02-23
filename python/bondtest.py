#!/usr/bin/env python

from sphere import *

sb = Spherebin(np=2, sid='bondtest')

cleanup(sb)

sb.x[0,:] = numpy.array((2,2,2))
sb.x[1,:] = numpy.array((3.5,2,2))
sb.radius = numpy.ones(sb.np)*0.5

#sb.vel[1,2] = 1
sb.angvel[1,1] = 0.01


sb.initGridAndWorldsize(margin = 10, periodic = 1, contactmodel = 2, g = numpy.array([0.0, 0.0, 0.0]))

sb.bond(0, 1)

sb.defaultParams()
#sb.gamma_n[0] = 10000
#sb.initTemporal(total=4.0)
sb.initTemporal(total=0.01, file_dt=2e-4)

sb.writebin()

sb.run(dry=True)
sb.run()

sb.render(verbose=False)

visualize(sb.sid, "energy")

sb.readlast()
print(sb.bonds_delta_n)
print(sb.bonds_delta_t)
print(sb.bonds_omega_n)
print(sb.bonds_omega_t)
print()
print(sb.force)
print(sb.torque)
print(sb.vel)
print(sb.angvel)
