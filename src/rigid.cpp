// rigid.cpp -- Rigid body solver using the Bullet Physics engine
#include <iostream>
#include <string>
#include <cstdio>

// Bullet is installed using the Debian packages.
// Documentation is located in /usr/share/doc/libbullet*
#include "btBulletDynamicsCommon.h"

#include "sphere.h"
#include "datatypes.h"
#include "utility.cuh"
#include "utility.h"
#include "constants.cuh"
#include "debug.h"


