#!/usr/bin/env python

# Import sphere functionality
from sphere import *

#render("../input/1e3-init-pyout.bin")

# New class
init = Spherebin(np = 1e3, nd = 3)
init.generateRadii(psd = 'uni', histogram = 0)
init.defaultparams()
init.initsetup()
init.initTemporal(total = 1.5)
init.writebin("../input/1e3-pyinit.bin")
render("../input/1e3-pyinit.bin", out = "~/Desktop/init")
