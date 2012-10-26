#ifndef CONSTANTS_CUH_
#define CONSTANTS_CUH_

#include "datatypes.h"

// Most constant memory variables are stored in
// structures, see datatypes.cuh

// Constant memory size: 64 kb
__constant__ unsigned int devC_nd;   // Number of dimensions
__constant__ unsigned int devC_np;   // Number of particles
__constant__ unsigned int devC_nw;   // Number of dynamic walls
__constant__ int	  devC_nc;   // Max. number of contacts a particle can have
__constant__ Float	  devC_dt;   // Computational time step length

// Device constant memory structures
__constant__ Params devC_params;
__constant__ Grid   devC_grid;
 
#endif
