// datatypes.h -- Structure templates and function prototypes

// Avoiding multiple inclusions of header file
#ifndef DATATYPES_H_
#define DATATYPES_H_

#include <math.h>
#include "vector_functions.h"
//#include "vector_arithmetic.h"


// Enable profiling of kernel runtimes?
// 0: No (default)
// 1: Yes
#define PROFILING 1

// Output information about contacts to stdout?
// 0: No (default)
// 1: Yes
#define CONTACTINFO 0


//////////////////////
// TYPE DEFINITIONS //
//////////////////////

// REMEMBER: When changing the precision below,
// change values in typedefs.h accordingly.

// Uncomment all five lines below for single precision
/*
typedef Float Float;
typedef Float3 Float3;
typedef Float4 Float4;
#define MAKE_FLOAT3(x, y, z) make_Float3(x, y, z)
#define MAKE_FLOAT4(x, y, z, w) make_Float4(x, y, z, w)
*/


// Uncomment all five lines below for double precision
///*
typedef double Float;
typedef double2 Float2;
typedef double3 Float3;
typedef double4 Float4;
#define MAKE_FLOAT3(x, y, z) make_double3(x, y, z)
#define MAKE_FLOAT4(x, y, z, w) make_double4(x, y, z, w)
//*/


////////////////////////
// SYMBOLIC CONSTANTS //
////////////////////////

// Define the max. number of walls
#define MAXWALLS 6


const Float PI = 3.14159265358979;

// Number of dimensions (1 and 2 NOT functional)
const unsigned int ND = 3;

// Define source code version
const Float VERS = 0.25;

// Max. number of contacts per particle
//const int NC = 16;
const int NC = 32;


///////////////////////////
// STRUCTURE DECLARATION //
///////////////////////////

// Structure containing variable particle parameters
struct Particles {
  Float *radius;
  Float *k_n;
  Float *k_t;
  Float *k_r;
  Float *gamma_n;
  Float *gamma_t;
  Float *gamma_r;
  Float *mu_s;
  Float *mu_d;
  Float *mu_r;
  Float *rho;
  Float *es_dot;
  Float *ev_dot;
  Float *es;
  Float *ev;
  Float *p;
  Float *m;
  Float *I;
  unsigned int np;
};

// Structure containing grid parameters
struct Grid {
  unsigned int nd;
  Float origo[ND];
  Float L[ND];
  unsigned int num[ND];
};

// Structure containing time parameters
struct Time {
  Float dt;
  double current;
  double total;
  Float file_dt;
  unsigned int step_count;
};

// Structure containing constant, global physical parameters
struct Params {
  int global;
  Float g[ND];
  Float dt;
  unsigned int np;
  unsigned int nw;
  int wmode[MAXWALLS];
  Float k_n;
  Float k_t;
  Float k_r;
  Float gamma_n;
  Float gamma_t;
  Float gamma_r;
  Float gamma_wn;
  Float gamma_wt;
  Float gamma_wr;
  Float mu_s; 
  Float mu_d;
  Float mu_r;
  Float rho;
  Float kappa;
  Float db;
  Float V_b;
  int periodic;
  unsigned int shearmodel;
};


/////////////////////////
// PROTOTYPE FUNCTIONS //
/////////////////////////
int fwritebin(char *target, Particles *p, 
    	      Float4 *host_x, Float4 *host_vel, 
	      Float4 *host_angvel, Float4 *host_force, 
	      Float4 *host_torque, Float4 *host_angpos, 
	      uint4 *host_bonds,
	      Grid *grid, Time *time, Params *params,
	      Float4 *host_w_nx, Float4 *host_w_mvfd);

// device.cu
//extern "C"
void initializeGPU(void);

//extern "C"
void gpuMain(Float4* host_x,
    	     Float4* host_vel,
	     Float4* host_acc,
	     Float4* host_angvel,
	     Float4* host_angacc,
	     Float4* host_force,
	     Float4* host_torque,
	     Float4* host_angpos,
	     uint4*  host_bonds,
	     Particles p, Grid grid,
	     Time time, Params params,
	     Float4* host_w_nx,
	     Float4* host_w_mvfd,
	     const char* cwd,
	     const char* inputbin);

#endif
