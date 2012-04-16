// datatypes.h -- Structure templates and function prototypes

// Avoiding multiple inclusions of header file
#ifndef DATATYPES_H_
#define DATATYPES_H_

/////////////////////////////////
// STATIC VARIABLE DECLARATION //
/////////////////////////////////


#include "vector_functions.h"

//// Symbolic constants

const float PI = 3.14159265358979f;

// Number of dimensions (1 and 2 NOT functional)
const unsigned int ND = 3;

// Define source code version
const float VERS = 0.25;

// Max. number of contacts per particle
const char NC = 16;


///////////////////////////
// STRUCTURE DECLARATION //
///////////////////////////

// Structure containing variable particle parameters
struct Particles {
  float *radius;
  float *k_n;
  float *k_s;
  float *k_r;
  float *gamma_n;
  float *gamma_s;
  float *gamma_r;
  float *mu_s;
  float *mu_d;
  float *mu_r;
  float *rho;
  float *es_dot;
  float *es;
  float *p;
  float *m;
  float *I;
  unsigned int np;
};

// Structure containing grid parameters
struct Grid {
  unsigned int nd;
  float *origo;
  float *L;
  unsigned int *num;
};

// Structure containing time parameters
struct Time {
  float dt;
  double current;
  double total;
  float file_dt;
  unsigned int step_count;
};

// Structure containing constant, global physical parameters
struct Params {
  //bool global;
  int global;
  float *g;
  unsigned int np;
  unsigned int nw;
  float dt; 
  float k_n;
  float k_s;
  float k_r;
  float gamma_n;
  float gamma_s;
  float gamma_r;
  float mu_s; 
  float mu_d;
  float mu_r;
  float rho;
  float kappa;
  float db;
  float V_b;
  int periodic;
  unsigned int shearmodel;
};


/////////////////////////
// PROTOTYPE FUNCTIONS //
/////////////////////////
int fwritebin(char *target, Particles *p, 
    	      float4 *host_x, float4 *host_vel, 
	      float4 *host_angvel, float4 *host_force, 
	      float4 *host_torque, uint4 *host_bonds,
	      Grid *grid, Time *time, Params *params,
	      float4 *host_w_nx, float4 *host_w_mvfd);

// device.cu
//extern "C"
void initializeGPU(void);

//extern "C"
void gpuMain(float4* host_x,
    	     float4* host_vel,
	     float4* host_acc,
	     float4* host_angvel,
	     float4* host_angacc,
	     float4* host_force,
	     float4* host_torque,
	     uint4*  host_bonds,
	     Particles* p, Grid* grid,
	     Time* time, Params* params,
	     float4* host_w_nx,
	     float4* host_w_mvfd,
	     const char* cwd,
	     const char* inputbin);

#endif
