// Avoiding multiple inclusions of header file
#ifndef DATATYPES_H_
#define DATATYPES_H_

#include <math.h>
#include "vector_functions.h"
//#include "vector_arithmetic.h"
#include "typedefs.h"
#include "constants.h"


////////////////////////////
// STRUCTURE DECLARATIONS //
////////////////////////////

// Structure containing kinematic particle values
struct Kinematics {
  Float4 *x;		// Positions + radii (w)
  Float2 *xysum;	// Horizontal distance traveled
  Float4 *vel;		// Translational velocities + fixvels (w)
  Float4 *force;	// Sums of forces
  Float4 *angpos;	// Angular positions
  Float4 *angvel;	// Angular velocities
  Float4 *torque;	// Sums of torques
};

// Structure containing individual physical particle parameters
struct Energies {
  Float *es_dot;	// Frictional dissipation rates
  Float *es;		// Frictional dissipations
  Float *ev_dot;	// Viscous dissipation rates
  Float *ev;		// Viscous dissipations
  Float *p;		// Pressures
  //uint4 *bonds;		// Cohesive bonds
};

// Structure containing grid parameters
struct Grid {
  Float origo[ND];	// World coordinate system origo
  Float L[ND];		// World dimensions
  unsigned int num[ND];	// Neighbor-search cells along each axis
  int periodic;		// Behavior of boundaries at 1st and 2nd world edge
};

// Structure containing time parameters
struct Time {
  Float dt;		// Computational time step length
  double current;	// Current time
  double total;		// Total time (at the end of experiment)
  Float file_dt;	// Time between output files
  unsigned int step_count; // Number of steps taken at current time
};

// Structure containing constant, global physical parameters
struct Params {
  Float g[ND];		// Gravitational acceleration
  Float k_n;		// Normal stiffness
  Float k_t;		// Tangential stiffness
  Float k_r;		// Rotational stiffness
  Float gamma_n;	// Normal viscosity
  Float gamma_t;	// Tangential viscosity
  Float gamma_r;	// Rotational viscosity
  Float mu_s; 		// Static friction coefficient
  Float mu_d;		// Dynamic friction coefficient
  Float mu_r;		// Rotational friction coefficient
  Float rho;		// Material density
  unsigned int contactmodel; // Inter-particle contact model
  Float kappa;		// Capillary bond prefactor
  Float db;		// Capillary bond debonding distance
  Float V_b;		// Volume of fluid in capillary bond
};

// Structure containing wall parameters
struct Walls {
  unsigned int nw;	// Number of walls (<= MAXWALLS)
  int wmode[MAXWALLS];	// Wall modes
  Float4* nx;		// Wall normal and position
  Float4* mvfd;		// Wall mass, velocity, force and dev. stress
  Float gamma_wn;	// Wall normal viscosity
  Float gamma_wt;	// Wall tangential viscosity
  Float gamma_wr;	// Wall rolling viscosity
};

#endif
