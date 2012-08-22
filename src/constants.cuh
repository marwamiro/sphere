#ifndef CONSTANTS_CUH_
#define CONSTANTS_CUH_

// Constant memory size: 64 kb
__constant__ Float        devC_origo[ND];  // World coordinate system origo
__constant__ Float        devC_L[ND];	   // World length in dimension ND
__constant__ unsigned int devC_num[ND];	   // Number of cells in dimension ND
__constant__ Float        devC_dt;	   // Time step length
__constant__ int          devC_global;	   // Parameter properties, 1: global, 0: individual
__constant__ Float        devC_g[ND];	   // Gravitational acceleration vector
__constant__ unsigned int devC_np;	   // Number of particles
__constant__ char	  devC_nc;	   // Max. number of contacts a particle can have
__constant__ unsigned int devC_shearmodel; // Shear force model: 1: viscous, frictional, 2: elastic, frictional
__constant__ Float        devC_k_n;	   // Material normal stiffness
__constant__ Float        devC_k_s;	   // Material shear stiffness
__constant__ Float        devC_k_r;	   // Material rolling stiffness
__constant__ Float        devC_gamma_n;	   // Material normal viscosity
__constant__ Float        devC_gamma_s;	   // Material shear viscosity
__constant__ Float	  devC_gamma_r;	   // Material rolling viscosity
__constant__ Float        devC_gamma_wn;   // Wall normal viscosity
__constant__ Float        devC_gamma_ws;   // Wall shear viscosity
__constant__ Float	  devC_gamma_wr;   // Wall rolling viscosity
__constant__ Float        devC_mu_s;	   // Material static shear friction coefficient
__constant__ Float        devC_mu_d;	   // Material dynamic shear friction coefficient
__constant__ Float	  devC_mu_r;	   // Material rolling friction coefficient
__constant__ Float        devC_rho;	   // Material density
__constant__ Float	  devC_kappa;	   // Capillary bond prefactor
__constant__ Float	  devC_db;	   // Debonding distance
__constant__ Float	  devC_V_b;	   // Liquid volume of capillary bond
__constant__ unsigned int devC_nw;	   // Number of walls
__constant__ unsigned int devC_w_n;	   // Dimension of orthogonal wall surface normal
__constant__ int          devC_periodic;   // Behavior of x- and y boundaries: 0: walls, 1: periodic
#endif
