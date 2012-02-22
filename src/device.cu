// device.cu -- GPU specific operations utilizing the CUDA API.
#include <iostream>
#include <cstdio>
#include <cuda.h>
#include <cutil_math.h>

#include "thrust/device_ptr.h"
#include "thrust/sort.h"

#include "datatypes.h"
#include "datatypes.cuh"
#include "constants.cuh"

#include "cuPrintf.cu"


// Returns the cellID containing the particle, based cubic grid
// See Bayraktar et al. 2009
// Kernel is executed on the device, and is callable from the device only
__device__ int calcCellID(float3 x) 
{ 
  unsigned int i_x, i_y, i_z;

  // Calculate integral coordinates:
  i_x = floor((x.x - devC_origo[0]) / (devC_L[0]/devC_num[0]));
  i_y = floor((x.y - devC_origo[1]) / (devC_L[1]/devC_num[1]));
  i_z = floor((x.z - devC_origo[2]) / (devC_L[2]/devC_num[2]));

  // Integral coordinates are converted to 1D coordinate:
  return __umul24(__umul24(i_z, devC_num[1]),
      		  devC_num[0]) 
    	 + __umul24(i_y, devC_num[0]) + i_x;

} // End of calcCellID(...)


// Calculate hash value for each particle, based on position in grid.
// Kernel executed on device, and callable from host only.
__global__ void calcParticleCellID(unsigned int* dev_gridParticleCellID, 
    				   unsigned int* dev_gridParticleIndex, 
				   float4* dev_x) 
{
  //unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x; // Thread id
  unsigned int idx = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

  if (idx < devC_np) { // Condition prevents block size error

    //volatile float4 x = dev_x[idx]; // Ensure coalesced read
    float4 x = dev_x[idx]; // Ensure coalesced read

    unsigned int cellID = calcCellID(make_float3(x.x, x.y, x.z));

    // Store values    
    dev_gridParticleCellID[idx] = cellID;
    dev_gridParticleIndex[idx]  = idx;

  }
} // End of calcParticleCellID(...)


// Reorder particle data into sorted order, and find the start and end particle indexes
// of each cell in the sorted hash array.
// Kernel executed on device, and callable from host only.
__global__ void reorderArrays(unsigned int* dev_cellStart, unsigned int* dev_cellEnd,
			      unsigned int* dev_gridParticleCellID, 
			      unsigned int* dev_gridParticleIndex,
			      float4* dev_x, float4* dev_vel, 
			      float4* dev_angvel, float* dev_radius,
			      //uint4* dev_bonds,
			      float4* dev_x_sorted, float4* dev_vel_sorted,
			      float4* dev_angvel_sorted, float* dev_radius_sorted)
			      //uint4* dev_bonds_sorted)
{ 

  // Create hash array in shared on-chip memory. The size of the array 
  // (threadsPerBlock + 1) is determined at launch time (extern notation).
  extern __shared__ unsigned int shared_data[]; 

  // Thread index in block
  unsigned int tidx = threadIdx.x;

  // Thread index in grid
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x; 
  //unsigned int idx = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

  // CellID hash value of particle idx
  unsigned int cellID;

  // Read cellID data and store it in shared memory (shared_data)
  if (idx < devC_np) { // Condition prevents block size error
    cellID = dev_gridParticleCellID[idx];

    // Load hash data into shared memory, allowing access to neighbor particle cellID values
    shared_data[tidx+1] = cellID; 

    if (idx > 0 && tidx == 0) {
      // First thread in block must load neighbor particle hash
      shared_data[0] = dev_gridParticleCellID[idx-1];
    }
  }

  // Pause completed threads in this block, until all 
  // threads are done loading data into shared memory
  __syncthreads();

  // Find lowest and highest particle index in each cell
  if (idx < devC_np) { // Condition prevents block size error
    // If this particle has a different cell index to the previous particle, it's the first
    // particle in the cell -> Store the index of this particle in the cell.
    // The previous particle must be the last particle in the previous cell.
    if (idx == 0 || cellID != shared_data[tidx]) {
      dev_cellStart[cellID] = idx;
      if (idx > 0)
	dev_cellEnd[shared_data[tidx]] = idx;
    }

    // Check wether the thread is the last one
    if (idx == (devC_np - 1)) 
      dev_cellEnd[cellID] = idx + 1;


    // Use the sorted index to reorder the position and velocity data
    unsigned int sortedIndex = dev_gridParticleIndex[idx];

    // Fetch from global read
    float4 x      = dev_x[sortedIndex];
    float4 vel    = dev_vel[sortedIndex];
    float4 angvel = dev_angvel[sortedIndex];
    float  radius = dev_radius[sortedIndex];
    //uint4  bonds  = dev_bonds[sortedIndex];

    __syncthreads();
    // Write sorted data to global memory
    dev_x_sorted[idx]      = x;
    dev_vel_sorted[idx]    = vel;
    dev_angvel_sorted[idx] = angvel;
    dev_radius_sorted[idx] = radius;
    //dev_bonds_sorted[idx]  = bonds;
  }
} // End of reorderArrays(...)




// COLLISION LOCATION AND PROCESSING FUNCTIONS


// Linear viscoelastic contact model for particle-wall interactions
// with tangential friction and rolling resistance
__device__ float contactLinear_wall(float3* N, float3* T, float* es_dot, float* p,
				    unsigned int idx_a, float radius_a,
				    float4* dev_vel_sorted, float4* dev_angvel_sorted,
				    float3 n, float delta, float wvel)
{
  // Fetch velocities from global memory
  float4 linvel_a_tmp = dev_vel_sorted[idx_a];
  float4 angvel_a_tmp = dev_angvel_sorted[idx_a];

  // Convert velocities to three-component vectors
  float3 linvel_a = make_float3(linvel_a_tmp.x,
      				linvel_a_tmp.y,
				linvel_a_tmp.z);
  float3 angvel_a = make_float3(angvel_a_tmp.x,
      				angvel_a_tmp.y,
				angvel_a_tmp.z);

  // Store the length of the angular velocity for later use
  float  angvel_length = length(angvel_a);

  // Contact velocity is the sum of the linear and
  // rotational components
  float3 vel = linvel_a + radius_a * cross(n, angvel_a) + wvel;

  // Normal component of the contact velocity
  float  vel_n = dot(vel, n);

  // The tangential velocity is the contact velocity
  // with the normal component subtracted
  float3 vel_t = vel - n * (dot(vel, n));
  float  vel_t_length = length(vel_t);

  // Calculate elastic normal component
  //float3 f_n = -devC_k_n * delta * n;

  // Normal force component: Elastic - viscous damping
  float3 f_n = (-devC_k_n * delta - devC_nu * vel_n) * n;

  // Make sure the viscous damping doesn't exceed the elastic component,
  // i.e. the damping factor doesn't exceed the critical damping, 2*sqrt(m*k_n)
  if (dot(f_n, n) < 0.0f)
    f_n = make_float3(0.0f, 0.0f, 0.0f);

  float  f_n_length = length(f_n); // Save length for later use

  // Initialize vectors
  float3 f_s   = make_float3(0.0f, 0.0f, 0.0f);
  float3 T_res = make_float3(0.0f, 0.0f, 0.0f);

  // Check that the tangential velocity is high enough to avoid
  // divide by zero (producing a NaN)
  if (vel_t_length > 0.f) {

    // Shear force component
    // Limited by Mohr Coulomb failure criterion
    f_s = -1.0f * fmin(devC_gamma_s * vel_t_length,
		       devC_mu_s * f_n_length)
          * vel_t/vel_t_length;

    // Shear energy production rate [W]
    *es_dot += -dot(vel_t, f_s);
  }

  if (angvel_length > 0.f) {
    // Apply rolling resistance (Zhou et al. 1999)
    //T_res = -angvel_a/angvel_length * devC_mu_r * radius_a * f_n_length;

    // New rolling resistance model
    T_res = -1.0f * fmin(devC_gamma_r * radius_a * angvel_length,
			 devC_mu_r * radius_a * f_n_length)
            * angvel_a/angvel_length;
  }

  // Total force from wall
  *N += f_n + f_s;

  // Total torque from wall
  *T += -radius_a * cross(n, f_s) + T_res;

  // Pressure excerted onto particle from this contact
  *p += f_n_length / (4.0f * PI * radius_a*radius_a);

  // Return force excerted onto the wall
  //return -dot(*N, n);
  return dot(f_n, n);
}


// Linear vicoelastic contact model for particle-particle interactions
// with tangential friction and rolling resistance
__device__ void contactLinearViscous(float3* N, float3* T, float* es_dot, float* p,
    			      	     unsigned int idx_a, unsigned int idx_b, 
				     float4* dev_vel_sorted, 
				     float4* dev_angvel_sorted,
				     float radius_a, float radius_b, 
				     float3 x_ab, float x_ab_length, 
				     float delta_ab, float kappa) 
{

  // Allocate variables and fetch missing time=t values for particle A and B
  float4 vel_a     = dev_vel_sorted[idx_a];
  float4 vel_b     = dev_vel_sorted[idx_b];
  float4 angvel4_a = dev_angvel_sorted[idx_a];
  float4 angvel4_b = dev_angvel_sorted[idx_b];

  // Convert to float3's
  float3 angvel_a = make_float3(angvel4_a.x, angvel4_a.y, angvel4_a.z);
  float3 angvel_b = make_float3(angvel4_b.x, angvel4_b.y, angvel4_b.z);

  // Force between grain pair decomposed into normal- and tangential part
  float3 f_n, f_s, f_c, T_res;

  // Normal vector of contact
  float3 n_ab = x_ab/x_ab_length;

  // Relative contact interface velocity, w/o rolling
  float3 vel_ab_linear = make_float3(vel_a.x - vel_b.x, 
      				     vel_a.y - vel_b.y, 
				     vel_a.z - vel_b.z);

  // Relative contact interface velocity of particle surfaces at
  // the contact, with rolling (Hinrichsen and Wolf 2004, eq. 13.10)
  float3 vel_ab = vel_ab_linear
		  + radius_a * cross(n_ab, angvel_a)
		  + radius_b * cross(n_ab, angvel_b);

  // Relative contact interface rolling velocity
  float3 angvel_ab = angvel_a - angvel_b;
  float  angvel_ab_length = length(angvel_ab);

  // Normal component of the relative contact interface velocity
  float vel_n_ab = dot(vel_ab_linear, n_ab);

  // Tangential component of the relative contact interface velocity
  // Hinrichsen and Wolf 2004, eq. 13.9
  float3 vel_t_ab = vel_ab - (n_ab * dot(vel_ab, n_ab));
  float  vel_t_ab_length = length(vel_t_ab);

  // Compute the normal stiffness of the contact
  //float k_n_ab = k_n_a * k_n_b / (k_n_a + k_n_b);

  // Calculate rolling radius
  float R_bar = (radius_a + radius_b) / 2.0f;

  // Normal force component: linear-elastic approximation (Augier 2009, eq. 3)
  // with velocity dependant damping
  //   Damping coefficient: alpha = 0.8
  //f_n = (-k_n_ab * delta_ab + 2.0f * 0.8f * sqrtf(m_eff*k_n_ab) * vel_ab) * n_ab;

  // Linear spring for normal component (Renzo 2004, eq. 35)
  // Dissipation due to  plastic deformation is modelled by using a different
  // unloading spring constant (Walton and Braun 1986)
  // Here the factor in the second term determines the relative strength of the
  // unloading spring relative to the loading spring.
  /*  if (vel_n_ab > 0.0f) {	// Loading
      f_n = (-k_n_ab * delta_ab) * n_ab;
      } else {			// Unloading
      f_n = (-k_n_ab * 0.90f * delta_ab) * n_ab;
      } // f_n is OK! */

  // Normal force component: Elastic
  //f_n = -k_n_ab * delta_ab * n_ab;

  // Normal force component: Elastic - viscous damping
  f_n = (-devC_k_n * delta_ab - devC_nu * vel_n_ab) * n_ab;

  // Make sure the viscous damping doesn't exceed the elastic component,
  // i.e. the damping factor doesn't exceed the critical damping, 2*sqrt(m*k_n)
  if (dot(f_n, n_ab) < 0.0f)
    f_n = make_float3(0.0f, 0.0f, 0.0f);

  float f_n_length = length(f_n);

  // Add max. capillary force
  f_c = -kappa * sqrtf(radius_a * radius_b) * n_ab;

  // Initialize force vectors to zero
  f_s   = make_float3(0.0f, 0.0f, 0.0f);
  T_res = make_float3(0.0f, 0.0f, 0.0f);

  // Shear force component: Nonlinear relation
  // Coulomb's law of friction limits the tangential force to less or equal
  // to the normal force
  if (vel_t_ab_length > 0.f) {

    // Shear force
    f_s = -1.0f * fmin(devC_gamma_s * vel_t_ab_length, 
		       devC_mu_s * length(f_n-f_c)) 
          * vel_t_ab/vel_t_ab_length;

    // Shear friction production rate [W]
    *es_dot += -dot(vel_t_ab, f_s);
  }

  if (angvel_ab_length > 0.f) {
    // Apply rolling resistance (Zhou et al. 1999)
    //T_res = -angvel_ab/angvel_ab_length * devC_mu_r * R_bar * length(f_n);

    // New rolling resistance model
    T_res = -1.0f * fmin(devC_gamma_r * R_bar * angvel_ab_length,
			 devC_mu_r * R_bar * f_n_length)
            * angvel_ab/angvel_ab_length;
  }


  // Add force components from this collision to total force for particle
  *N += f_n + f_s + f_c; 
  *T += -R_bar * cross(n_ab, f_s) + T_res;

  // Pressure excerted onto the particle from this contact
  *p += f_n_length / (4.0f * PI * radius_a*radius_a);

} // End of contactLinearViscous()


// Linear elastic contact model for particle-particle interactions
__device__ void contactLinear(float3* N, float3* T, 
    			      float* es_dot, float* p,
			      unsigned int idx_a_orig,
			      unsigned int idx_b_orig, 
			      float4  vel_a, 
			      float4* dev_vel,
			      float3  angvel_a,
			      float4* dev_angvel,
			      float radius_a, float radius_b, 
			      float3 x_ab, float x_ab_length, 
			      float delta_ab, float4* dev_delta_t,
			      unsigned int mempos) 
{

  // Allocate variables and fetch missing time=t values for particle A and B
  float4 vel_b     = dev_vel[idx_b_orig];
  float4 angvel4_b = dev_angvel[idx_b_orig];

  // Fetch previous sum of shear displacement for the contact pair
  float4 delta_t0  = dev_delta_t[mempos];

  // Convert to float3
  float3 angvel_b = make_float3(angvel4_b.x, angvel4_b.y, angvel4_b.z);

  // Force between grain pair decomposed into normal- and tangential part
  float3 f_n, f_s, f_c, T_res;

  // Normal vector of contact
  float3 n_ab = x_ab/x_ab_length;

  // Relative contact interface velocity, w/o rolling
  float3 vel_ab_linear = make_float3(vel_a.x - vel_b.x, 
      				     vel_a.y - vel_b.y, 
				     vel_a.z - vel_b.z);

  // Relative contact interface velocity of particle surfaces at
  // the contact, with rolling (Hinrichsen and Wolf 2004, eq. 13.10)
  float3 vel_ab = vel_ab_linear
		  + radius_a * cross(n_ab, angvel_a)
		  + radius_b * cross(n_ab, angvel_b);

  // Relative contact interface rolling velocity
  float3 angvel_ab = angvel_a - angvel_b;
  float  angvel_ab_length = length(angvel_ab);

  // Normal component of the relative contact interface velocity
  float vel_n_ab = dot(vel_ab_linear, n_ab);

  // Tangential component of the relative contact interface velocity
  // Hinrichsen and Wolf 2004, eq. 13.9
  float3 vel_t_ab = vel_ab - (n_ab * dot(vel_ab, n_ab));

  // Add tangential displacement to total tangential displacement
  float3 delta_t  = make_float3(delta_t0.x, delta_t0.y, delta_t0.z) + vel_t_ab * devC_dt;
  float  delta_t_length = length(delta_t);

  // Compute the normal stiffness of the contact
  //float k_n_ab = k_n_a * k_n_b / (k_n_a + k_n_b);

  // Calculate rolling radius
  float R_bar = (radius_a + radius_b) / 2.0f;

  // Normal force component: Elastic
  //f_n = -devC_k_n * delta_ab * n_ab;

  // Normal force component: Elastic - viscous damping
  f_n = (-devC_k_n * delta_ab - devC_nu * vel_n_ab) * n_ab;

  // Make sure the viscous damping doesn't exceed the elastic component,
  // i.e. the damping factor doesn't exceed the critical damping, 2*sqrt(m*k_n)
  if (dot(f_n, n_ab) < 0.0f)
    f_n = make_float3(0.0f, 0.0f, 0.0f);

  float f_n_length = length(f_n);

  // Add max. capillary force
  f_c = -devC_kappa * sqrtf(radius_a * radius_b) * n_ab;

  // Initialize force vectors to zero
  f_s   = make_float3(0.0f, 0.0f, 0.0f);
  T_res = make_float3(0.0f, 0.0f, 0.0f);

  // Shear force component: Nonlinear relation
  // Coulomb's law of friction limits the tangential force to less or equal
  // to the normal force
  if (delta_t_length > 0.f) {

    // Shear force: Elastic, limited by Mohr-Coulomb
    f_s = -1.0f * fmin(devC_k_s * delta_t_length, 
		       devC_mu_s * length(f_n-f_c)) 
          * delta_t/delta_t_length;

    // Shear friction production rate [W]
    *es_dot += -dot(vel_t_ab, f_s);
  }

  /*if (angvel_ab_length > 0.f) {
    // Apply rolling resistance (Zhou et al. 1999)
    //T_res = -angvel_ab/angvel_ab_length * devC_mu_r * R_bar * length(f_n);

    // New rolling resistance model
    T_res = -1.0f * fmin(devC_gamma_r * R_bar * angvel_ab_length,
			 devC_mu_r * R_bar * f_n_length)
            * angvel_ab/angvel_ab_length;
  }*/

  // Add force components from this collision to total force for particle
  *N += f_n + f_s + f_c; 
  *T += -R_bar * cross(n_ab, f_s) + T_res;

  // Pressure excerted onto the particle from this contact
  *p += f_n_length / (4.0f * PI * radius_a*radius_a);

  // Store sum of tangential displacements
  dev_delta_t[mempos] = make_float4(delta_t, 0.0f);

} // End of contactLinear()


// Linear-elastic bond: Attractive force with normal- and shear components
// acting upon particle A in a bonded particle pair
__device__ void bondLinear(float3* N, float3* T, float* es_dot, float* p,
			   unsigned int idx_a, unsigned int idx_b, 
			   float4* dev_x_sorted, float4* dev_vel_sorted, 
			   float4* dev_angvel_sorted,
			   float radius_a, float radius_b, 
			   float3 x_ab, float x_ab_length, 
			   float delta_ab) 
{

  // If particles are not overlapping, apply bond force
  if (delta_ab > 0.0f) {

    // Allocate variables and fetch missing time=t values for particle A and B
    float4 vel_a     = dev_vel_sorted[idx_a];
    float4 vel_b     = dev_vel_sorted[idx_b];
    float4 angvel4_a = dev_angvel_sorted[idx_a];
    float4 angvel4_b = dev_angvel_sorted[idx_b];

    // Convert to float3's
    float3 angvel_a = make_float3(angvel4_a.x, angvel4_a.y, angvel4_a.z);
    float3 angvel_b = make_float3(angvel4_b.x, angvel4_b.y, angvel4_b.z);

    // Normal vector of contact
    float3 n_ab = x_ab/x_ab_length;

    // Relative contact interface velocity, w/o rolling
    float3 vel_ab_linear = make_float3(vel_a.x - vel_b.x, 
				       vel_a.y - vel_b.y, 
				       vel_a.z - vel_b.z);

    // Relative contact interface velocity of particle surfaces at
    // the contact, with rolling (Hinrichsen and Wolf 2004, eq. 13.10)
    float3 vel_ab = vel_ab_linear
		    + radius_a * cross(n_ab, angvel_a)
		    + radius_b * cross(n_ab, angvel_b);

    // Relative contact interface rolling velocity
    //float3 angvel_ab = angvel_a - angvel_b;
    //float  angvel_ab_length = length(angvel_ab);

    // Normal component of the relative contact interface velocity
    //float vel_n_ab = dot(vel_ab_linear, n_ab);

    // Tangential component of the relative contact interface velocity
    // Hinrichsen and Wolf 2004, eq. 13.9
    float3 vel_t_ab = vel_ab - (n_ab * dot(vel_ab, n_ab));
    //float  vel_t_ab_length = length(vel_t_ab);

    float3 f_n = make_float3(0.0f, 0.0f, 0.0f);
    float3 f_s = make_float3(0.0f, 0.0f, 0.0f);

    // Mean radius
    float R_bar = (radius_a + radius_b)/2.0f;

    // Normal force component: Elastic
    f_n = devC_k_n * delta_ab * n_ab;

    if (length(vel_t_ab) > 0.f) {
      // Shear force component: Viscous
      f_s = -1.0f * devC_gamma_s * vel_t_ab;

      // Shear friction production rate [W]
      *es_dot += -dot(vel_t_ab, f_s);
    }

    // Add force components from this bond to total force for particle
    *N += f_n + f_s;
    *T += -R_bar * cross(n_ab, f_s);

    // Pressure excerted onto the particle from this bond
    *p += length(f_n) / (4.0f * PI * radius_a*radius_a);

  }
} // End of bondLinear()


// Capillary cohesion after Richefeu et al. (2006)
__device__ void capillaryCohesion_exp(float3* N, float radius_a, 
    				      float radius_b, float delta_ab,
    				      float3 x_ab, float x_ab_length, 
				      float kappa)
{

  // Normal vector 
  float3 n_ab = x_ab/x_ab_length;

  float3 f_c;
  float lambda, R_geo, R_har, r, h;

  // Determine the ratio; r = max{Ri/Rj;Rj/Ri}
  if ((radius_a/radius_b) > (radius_b/radius_a))
    r = radius_a/radius_b;
  else
    r = radius_b/radius_a;

  // Exponential decay function
  h = -sqrtf(r);

  // The harmonic mean
  R_har = (2.0f * radius_a * radius_b) / (radius_a + radius_b);

  // The geometrical mean
  R_geo = sqrtf(radius_a * radius_b);

  // The exponential falloff of the capillary force with distance
  lambda = 0.9f * h * sqrtf(devC_V_b/R_har);

  // Calculate cohesional force
  f_c = -kappa * R_geo * expf(-delta_ab/lambda) * n_ab;

  // Add force components from this collision to total force for particle
  *N += f_c;

} // End of capillaryCohesion_exp



// Find overlaps between particle no. 'idx' and particles in cell 'gridpos'
// Kernel executed on device, and callable from device only.
__device__ void overlapsInCell(int3 targetCell, 
    			       unsigned int idx_a, 
			       float4 x_a, float radius_a,
			       float3* N, float3* T, 
			       float* es_dot, float* p,
			       float4* dev_x_sorted, 
			       float* dev_radius_sorted,
			       float4* dev_vel_sorted, 
			       float4* dev_angvel_sorted,
			       unsigned int* dev_cellStart, 
			       unsigned int* dev_cellEnd,
			       float4* dev_w_nx, 
			       float4* dev_w_mvfd)
			       //uint4 bonds)
{

  // Variable containing modifier for interparticle
  // vector, if it crosses a periodic boundary
  float3 distmod = make_float3(0.0f, 0.0f, 0.0f);

  // Check whether x- and y boundaries are to be treated as periodic
  // 1: x- and y boundaries periodic
  // 2: x boundaries periodic
  if (devC_periodic == 1) {

    // Periodic x-boundary
    if (targetCell.x < 0) {
      targetCell.x = devC_num[0] - 1;
      distmod += make_float3(devC_L[0], 0.0f, 0.0f);
    }
    if (targetCell.x == devC_num[0]) {
      targetCell.x = 0;
      distmod -= make_float3(devC_L[0], 0.0f, 0.0f);
    }

    // Periodic y-boundary
    if (targetCell.y < 0) {
      targetCell.y = devC_num[1] - 1;
      distmod += make_float3(0.0f, devC_L[1], 0.0f);
    }
    if (targetCell.y == devC_num[1]) {
      targetCell.y = 0;
      distmod -= make_float3(0.0f, devC_L[1], 0.0f);
    }

  } else if (devC_periodic == 2) {

    // Periodic x-boundary
    if (targetCell.x < 0) {
      targetCell.x = devC_num[0] - 1;
      distmod += make_float3(devC_L[0], 0.0f, 0.0f);
    }
    if (targetCell.x == devC_num[0]) {
      targetCell.x = 0;
      distmod -= make_float3(devC_L[0], 0.0f, 0.0f);
    }

    // Hande out-of grid cases on y-axis
    if (targetCell.y < 0 || targetCell.y == devC_num[1])
      return;

  } else {

    // Hande out-of grid cases on x- and y-axes
    if (targetCell.x < 0 || targetCell.x == devC_num[0])
      return;
    if (targetCell.y < 0 || targetCell.y == devC_num[1])
      return;
  }

  // Handle out-of-grid cases on z-axis
  if (targetCell.z < 0 || targetCell.z == devC_num[2])
    return;


  //// Check and process particle-particle collisions

  // Calculate linear cell ID
  unsigned int cellID = targetCell.x  
    			+ __umul24(targetCell.y, devC_num[0]) 
			+ __umul24(__umul24(devC_num[0], 
			      		    devC_num[1]), 
			 	   targetCell.z); 

  // Lowest particle index in cell
  unsigned int startIdx = dev_cellStart[cellID];

  // Make sure cell is not empty
  if (startIdx != 0xffffffff) {

    // Highest particle index in cell + 1
    unsigned int endIdx = dev_cellEnd[cellID];

    // Iterate over cell particles
    for (unsigned int idx_b = startIdx; idx_b<endIdx; ++idx_b) {
      if (idx_b != idx_a) { // Do not collide particle with itself

	// Fetch position and velocity of particle B.
	float4 x_b      = dev_x_sorted[idx_b];
	float  radius_b = dev_radius_sorted[idx_b];
	float  kappa 	= devC_kappa;

	// Distance between particle centers (float4 -> float3)
	float3 x_ab = make_float3(x_a.x - x_b.x, 
	    			  x_a.y - x_b.y, 
				  x_a.z - x_b.z);

	// Adjust interparticle vector if periodic boundary/boundaries
	// are crossed
	x_ab += distmod;

	float x_ab_length = length(x_ab);

	// Distance between particle perimeters
	float delta_ab = x_ab_length - (radius_a + radius_b); 

	// Check for particle overlap
	if (delta_ab < 0.0f) {
		  contactLinearViscous(N, T, es_dot, p, 
	      		       idx_a, idx_b,
			       dev_vel_sorted, 
			       dev_angvel_sorted,
			       radius_a, radius_b, 
			       x_ab, x_ab_length,
			       delta_ab, kappa);
	} else if (delta_ab < devC_db) { 
	  // Check wether particle distance satisfies the capillary bond distance
	  capillaryCohesion_exp(N, radius_a, radius_b, delta_ab, 
	      			x_ab, x_ab_length, kappa);
	}

	// Check wether particles are bonded together
	/*if (bonds.x == idx_b || bonds.y == idx_b ||
	    bonds.z == idx_b || bonds.w == idx_b) {
	  bondLinear(N, T, es_dot, p, 
	      	     idx_a, idx_b,
	   	     dev_x_sorted, dev_vel_sorted,
		     dev_angvel_sorted,
		     radius_a, radius_b,
		     x_ab, x_ab_length,
		     delta_ab);
	}*/

      } // Do not collide particle with itself end
    } // Iterate over cell particles end
  } // Check wether cell is empty end
} // End of overlapsInCell(...)

// Find overlaps between particle no. 'idx' and particles in cell 'gridpos'
// Write the indexes of the overlaps in array contacts[]
// Kernel executed on device, and callable from device only.
__device__ void findContactsInCell(int3 targetCell, 
    			           unsigned int idx_a, 
				   float4 x_a, float radius_a,
				   float4* dev_x_sorted, 
				   float* dev_radius_sorted,
				   unsigned int* dev_cellStart, 
				   unsigned int* dev_cellEnd,
				   unsigned int* dev_gridParticleIndex,
				   int* nc,
				   unsigned int* dev_contacts,
				   float4* dev_x_ab_r_b)
{
  // Variable containing modifier for interparticle
  // vector, if it crosses a periodic boundary
  float3 distmod = make_float3(0.0f, 0.0f, 0.0f);

  // Check whether x- and y boundaries are to be treated as periodic
  // 1: x- and y boundaries periodic
  // 2: x boundaries periodic
  if (devC_periodic == 1) {

    // Periodic x-boundary
    if (targetCell.x < 0) {
      targetCell.x = devC_num[0] - 1;
      distmod += make_float3(devC_L[0], 0.0f, 0.0f);
    }
    if (targetCell.x == devC_num[0]) {
      targetCell.x = 0;
      distmod -= make_float3(devC_L[0], 0.0f, 0.0f);
    }

    // Periodic y-boundary
    if (targetCell.y < 0) {
      targetCell.y = devC_num[1] - 1;
      distmod += make_float3(0.0f, devC_L[1], 0.0f);
    }
    if (targetCell.y == devC_num[1]) {
      targetCell.y = 0;
      distmod -= make_float3(0.0f, devC_L[1], 0.0f);
    }

  } else if (devC_periodic == 2) {

    // Periodic x-boundary
    if (targetCell.x < 0) {
      targetCell.x = devC_num[0] - 1;
      distmod += make_float3(devC_L[0], 0.0f, 0.0f);
    }
    if (targetCell.x == devC_num[0]) {
      targetCell.x = 0;
      distmod -= make_float3(devC_L[0], 0.0f, 0.0f);
    }

    // Hande out-of grid cases on y-axis
    if (targetCell.y < 0 || targetCell.y == devC_num[1])
      return;

  } else {

    // Hande out-of grid cases on x- and y-axes
    if (targetCell.x < 0 || targetCell.x == devC_num[0])
      return;
    if (targetCell.y < 0 || targetCell.y == devC_num[1])
      return;
  }

  // Handle out-of-grid cases on z-axis
  if (targetCell.z < 0 || targetCell.z == devC_num[2])
    return;


  //// Check and process particle-particle collisions

  // Calculate linear cell ID
  unsigned int cellID = targetCell.x  
    			+ __umul24(targetCell.y, devC_num[0]) 
			+ __umul24(__umul24(devC_num[0], 
			      		    devC_num[1]), 
			 	   targetCell.z); 

  // Lowest particle index in cell
  unsigned int startIdx = dev_cellStart[cellID];

  // Make sure cell is not empty
  if (startIdx != 0xffffffff) {

    // Highest particle index in cell + 1
    unsigned int endIdx = dev_cellEnd[cellID];

    // Read the original index of particle A
    unsigned int idx_a_orig = dev_gridParticleIndex[idx_a];

    // Iterate over cell particles
    for (unsigned int idx_b = startIdx; idx_b<endIdx; ++idx_b) {
      if (idx_b != idx_a) { // Do not collide particle with itself

	// Fetch position and radius of particle B.
	float4 x_b      = dev_x_sorted[idx_b];
	float  radius_b = dev_radius_sorted[idx_b];

	// Read the original index of particle B
	unsigned int idx_b_orig = dev_gridParticleIndex[idx_b];

	__syncthreads();

	// Distance between particle centers (float4 -> float3)
	float3 x_ab = make_float3(x_a.x - x_b.x, 
	    			  x_a.y - x_b.y, 
				  x_a.z - x_b.z);

	// Adjust interparticle vector if periodic boundary/boundaries
	// are crossed
	x_ab += distmod;

	float x_ab_length = length(x_ab);

	// Distance between particle perimeters
	float delta_ab = x_ab_length - (radius_a + radius_b); 

	// Check for particle overlap
	if (delta_ab < 0.0f) {
	  
	  // If the particle is not yet registered in the contact list,
	  // use the next position in the array
	  int cpos = *nc;
	  unsigned int cidx;

	  // Find out, if particle is already registered in contact list
	  for (int i=0; i<devC_nc; ++i) {
	    __syncthreads();
	    cidx = dev_contacts[(unsigned int)(idx_a_orig*devC_nc+i)];
	    if (cidx == idx_b_orig)
	      cpos = i;
	  }

	  __syncthreads();
	  //cuPrintf("\nOverlap: idx_a = %u (orig=%u), idx_b = %u (orig=%u), cpos = %d\n", 
	    //  	   idx_a, idx_a_orig, idx_b, idx_b_orig, cpos);

	  // Write the particle index to the relevant position,
	  // no matter if it already is there or not (concurrency of write)
	  dev_contacts[(unsigned int)(idx_a_orig*devC_nc+cpos)] = idx_b_orig;

	  // Write the interparticle vector and radius of particle B
	  dev_x_ab_r_b[(unsigned int)(idx_a_orig*devC_nc+cpos)] = make_float4(x_ab, radius_b);
	  
	  // Increment contact counter
	  ++*nc;
	}

	// Check wether particles are bonded together
	/*if (bonds.x == idx_b || bonds.y == idx_b ||
	    bonds.z == idx_b || bonds.w == idx_b) {
	  bondLinear(N, T, es_dot, p, 
	      	     idx_a, idx_b,
	   	     dev_x_sorted, dev_vel_sorted,
		     dev_angvel_sorted,
		     radius_a, radius_b,
		     x_ab, x_ab_length,
		     delta_ab);
	}*/

      } // Do not collide particle with itself end
    } // Iterate over cell particles end
  } // Check wether cell is empty end
} // End of findContactsInCell(...)


// For a single particle:
// Search for neighbors to particle 'idx' inside the 27 closest cells, 
// and save the contact pairs in global memory.
__global__ void topology(unsigned int* dev_cellStart, 
    			 unsigned int* dev_cellEnd, // Input: Particles in cell 
			 unsigned int* dev_gridParticleIndex, // Input: Unsorted-sorted key
			 float4* dev_x_sorted, float* dev_radius_sorted, 
			 unsigned int* dev_contacts, 
			 float4* dev_x_ab_r_b)
{
  // Thread index equals index of particle A
  unsigned int idx_a = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  if (idx_a < devC_np) {
    // Fetch particle data in global read
    float4 x_a      = dev_x_sorted[idx_a];
    float  radius_a = dev_radius_sorted[idx_a];

    // Count the number of contacts in this time step
    int nc = 0;

    // Grid position of host and neighbor cells in uniform, cubic grid
    int3 gridPos;
    int3 targetPos;

    // Calculate cell address in grid from position of particle
    gridPos.x = floor((x_a.x - devC_origo[0]) / (devC_L[0]/devC_num[0]));
    gridPos.y = floor((x_a.y - devC_origo[1]) / (devC_L[1]/devC_num[1]));
    gridPos.z = floor((x_a.z - devC_origo[2]) / (devC_L[2]/devC_num[2]));

    // Find overlaps between particle no. idx and all particles
    // from its own cell + 26 neighbor cells
    for (int z_dim=-1; z_dim<2; ++z_dim) { // z-axis
      for (int y_dim=-1; y_dim<2; ++y_dim) { // y-axis
	for (int x_dim=-1; x_dim<2; ++x_dim) { // x-axis
	  targetPos = gridPos + make_int3(x_dim, y_dim, z_dim);
	  findContactsInCell(targetPos, idx_a, x_a, radius_a,
	       		     dev_x_sorted, dev_radius_sorted,
			     dev_cellStart, dev_cellEnd,
			     dev_gridParticleIndex,
	    		     &nc, dev_contacts, dev_x_ab_r_b);
	}
      }
    }
  }
} // End of topology(...)


// For a single particle:
// Search for neighbors to particle 'idx' inside the 27 closest cells, 
// and compute the resulting normal- and shear force on it.
// Collide with top- and bottom walls, save resulting force on upper wall.
// Kernel is executed on device, and is callable from host only.
__global__ void interact(unsigned int* dev_gridParticleIndex, // Input: Unsorted-sorted key
			 unsigned int* dev_cellStart,
			 unsigned int* dev_cellEnd,
    			 float4* dev_x_sorted, float* dev_radius_sorted, 
			 float4* dev_vel_sorted, float4* dev_angvel_sorted,
			 float4* dev_vel, float4* dev_angvel,
			 float4* dev_force, float4* dev_torque,
			 float* dev_es_dot, float* dev_es, float* dev_p,
			 float4* dev_w_nx, float4* dev_w_mvfd, 
			 float* dev_w_force, //uint4* dev_bonds_sorted,
			 unsigned int* dev_contacts, 
			 float4* dev_x_ab_r_b,
			 float4* dev_delta_t)
{
  // Thread index equals index of particle A
  unsigned int idx_a = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

  if (idx_a < devC_np) {

    // Fetch particle data in global read
    unsigned int idx_a_orig = dev_gridParticleIndex[idx_a];
    float4 x_a      = dev_x_sorted[idx_a];
    float  radius_a = dev_radius_sorted[idx_a];

    // Fetch wall data in global read
    float4 w_up_nx   = dev_w_nx[0];
    float4 w_up_mvfd = dev_w_mvfd[0];

    // Fetch world dimensions in constant memory read
    float3 origo = make_float3(devC_origo[0], 
			       devC_origo[1], 
			       devC_origo[2]); 
    float3 L = make_float3(devC_L[0], 
			   devC_L[1], 
			   devC_L[2]);

    // Index of particle which is bonded to particle A.
    // The index is equal to the particle no (p.np)
    // if particle A is bond-less.
    //uint4 bonds = dev_bonds_sorted[idx_a];

    // Initiate shear friction loss rate at 0.0
    float es_dot = 0.0f; 

    // Initiate pressure on particle at 0.0
    float p = 0.0f;

    // Allocate memory for temporal force/torque vector values
    float3 N = make_float3(0.0f, 0.0f, 0.0f);
    float3 T = make_float3(0.0f, 0.0f, 0.0f);

    // Apply linear elastic, frictional contact model to registered contacts
    if (devC_shearmodel == 2) {
      unsigned int idx_b_orig, mempos;
      float delta_n, x_ab_length;
      float4 x_ab_r_b;
      float3 x_ab;
      float4 vel_a     = dev_vel_sorted[idx_a];
      float4 angvel4_a = dev_angvel_sorted[idx_a];
      float3 angvel_a  = make_float3(angvel4_a.x, angvel4_a.y, angvel4_a.z);

      // Loop over all possible contacts, and remove contacts
      // that no longer are valid (delta_n > 0.0)
      for (int i = 0; i<devC_nc; ++i) {
	mempos = idx_a_orig * devC_nc + i;
	idx_b_orig = dev_contacts[mempos];
	x_ab_r_b   = dev_x_ab_r_b[mempos];
	x_ab       = make_float3(x_ab_r_b.x,
	    			 x_ab_r_b.y,
				 x_ab_r_b.z);
	x_ab_length = length(x_ab);
	delta_n = x_ab_length - (radius_a + x_ab_r_b.w);

	// Process collision if the particles are overlapping
	if (delta_n < 0.0f) {
	  if (idx_b_orig != devC_np) {
	    //cuPrintf("\nProcessing contact, idx_a_orig = %u, idx_b_orig = %u, contact = %d, delta_n = %f\n",
	    //  idx_a_orig, idx_b_orig, i, delta_n);
	    contactLinear(&N, &T, &es_dot, &p, 
		idx_a_orig,
		idx_b_orig,
		vel_a,
		dev_vel,
		angvel_a,
		dev_angvel,
		radius_a, x_ab_r_b.w, 
		x_ab, x_ab_length,
		delta_n, dev_delta_t, 
		mempos);
	  }
	} else {
	  // Remove this contact (there is no particle with index=np)
	  dev_contacts[mempos] = devC_np; 
	  // Zero sum of shear displacement in this position
	  dev_delta_t[mempos]  = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	}
      }
    } else if (devC_shearmodel == 1) {

      int3 gridPos;
      int3 targetPos;

      // Calculate address in grid from position
      gridPos.x = floor((x_a.x - devC_origo[0]) / (devC_L[0]/devC_num[0]));
      gridPos.y = floor((x_a.y - devC_origo[1]) / (devC_L[1]/devC_num[1]));
      gridPos.z = floor((x_a.z - devC_origo[2]) / (devC_L[2]/devC_num[2]));

      // Find overlaps between particle no. idx and all particles
      // from its own cell + 26 neighbor cells.
      // Calculate resulting normal- and shear-force components and
      // torque for the particle on the base of contactLinearViscous()
      for (int z_dim=-1; z_dim<2; ++z_dim) { // z-axis
	for (int y_dim=-1; y_dim<2; ++y_dim) { // y-axis
	  for (int x_dim=-1; x_dim<2; ++x_dim) { // x-axis
	    targetPos = gridPos + make_int3(x_dim, y_dim, z_dim);
	    overlapsInCell(targetPos, idx_a, x_a, radius_a,
			   &N, &T, &es_dot, &p,
			   dev_x_sorted, dev_radius_sorted, 
			   dev_vel_sorted, dev_angvel_sorted,
			   dev_cellStart, dev_cellEnd,
			   dev_w_nx, dev_w_mvfd);
	  }
	}
      }

    }

    //// Interact with walls
    float delta_w; // Overlap distance
    float3 w_n;    // Wall surface normal
    float w_force = 0.0f; // Force on wall from particle A

    // Upper wall (idx 0)
    delta_w = w_up_nx.w - (x_a.z + radius_a);
    w_n = make_float3(0.0f, 0.0f, -1.0f);
    if (delta_w < 0.0f) {
      w_force = contactLinear_wall(&N, &T, &es_dot, &p, idx_a, radius_a,
	  			   dev_vel_sorted, dev_angvel_sorted,
				   w_n, delta_w, w_up_mvfd.y);
    }

    // Lower wall (force on wall not stored)
    delta_w = x_a.z - radius_a - origo.z;
    w_n = make_float3(0.0f, 0.0f, 1.0f);
    if (delta_w < 0.0f) {
      (void)contactLinear_wall(&N, &T, &es_dot, &p, idx_a, radius_a,
	  		       dev_vel_sorted, dev_angvel_sorted,
	  		       w_n, delta_w, 0.0f);
    }


    if (devC_periodic == 0) {

      // Left wall
      delta_w = x_a.x - radius_a - origo.x;
      w_n = make_float3(1.0f, 0.0f, 0.0f);
      if (delta_w < 0.0f) {
	(void)contactLinear_wall(&N, &T, &es_dot, &p, idx_a, radius_a,
	    			 dev_vel_sorted, dev_angvel_sorted,
				 w_n, delta_w, 0.0f);
      }

      // Right wall
      delta_w = L.x - (x_a.x + radius_a);
      w_n = make_float3(-1.0f, 0.0f, 0.0f);
      if (delta_w < 0.0f) {
	(void)contactLinear_wall(&N, &T, &es_dot, &p, idx_a, radius_a,
	    			 dev_vel_sorted, dev_angvel_sorted,
				 w_n, delta_w, 0.0f);
      }

      // Front wall
      delta_w = x_a.y - radius_a - origo.y;
      w_n = make_float3(0.0f, 1.0f, 0.0f);
      if (delta_w < 0.0f) {
	(void)contactLinear_wall(&N, &T, &es_dot, &p, idx_a, radius_a,
	    			 dev_vel_sorted, dev_angvel_sorted,
				 w_n, delta_w, 0.0f);
      }

      // Back wall
      delta_w = L.y - (x_a.y + radius_a);
      w_n = make_float3(0.0f, -1.0f, 0.0f);
      if (delta_w < 0.0f) {
	(void)contactLinear_wall(&N, &T, &es_dot, &p, idx_a, radius_a,
	    			 dev_vel_sorted, dev_angvel_sorted,
				 w_n, delta_w, 0.0f);
      }

    } else if (devC_periodic == 2) {

      // Front wall
      delta_w = x_a.y - radius_a - origo.y;
      w_n = make_float3(0.0f, 1.0f, 0.0f);
      if (delta_w < 0.0f) {
	(void)contactLinear_wall(&N, &T, &es_dot, &p, idx_a, radius_a,
	    			 dev_vel_sorted, dev_angvel_sorted,
				 w_n, delta_w, 0.0f);
      }

      // Back wall
      delta_w = L.y - (x_a.y + radius_a);
      w_n = make_float3(0.0f, -1.0f, 0.0f);
      if (delta_w < 0.0f) {
	(void)contactLinear_wall(&N, &T, &es_dot, &p, idx_a, radius_a,
	    			 dev_vel_sorted, dev_angvel_sorted,
				 w_n, delta_w, 0.0f);
      }
    }


    // Hold threads for coalesced write
    __syncthreads();

    // Write force to unsorted position
    unsigned int orig_idx = dev_gridParticleIndex[idx_a];
    dev_force[orig_idx]   = make_float4(N, 0.0f);
    dev_torque[orig_idx]  = make_float4(T, 0.0f);
    dev_es_dot[orig_idx]  = es_dot;
    dev_es[orig_idx]     += es_dot * devC_dt;
    dev_p[orig_idx]       = p;
    dev_w_force[orig_idx] = w_force;
  }
} // End of interact(...)





// FUNCTION FOR UPDATING TRAJECTORIES

// Second order integration scheme based on Taylor expansion of particle kinematics. 
// Kernel executed on device, and callable from host only.
__global__ void integrate(float4* dev_x_sorted, float4* dev_vel_sorted, // Input
			  float4* dev_angvel_sorted, float* dev_radius_sorted, // Input
			  float4* dev_x, float4* dev_vel, float4* dev_angvel, // Output
			  float4* dev_force, float4* dev_torque, // Input
			  unsigned int* dev_gridParticleIndex) // Input: Sorted-Unsorted key
{
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x; // Thread id

  if (idx < devC_np) { // Condition prevents block size error

    // Copy data to temporary arrays to avoid any potential read-after-write, 
    // write-after-read, or write-after-write hazards. 
    unsigned int orig_idx = dev_gridParticleIndex[idx];
    float4 force  = dev_force[orig_idx];
    float4 torque = dev_torque[orig_idx];

    // Initialize acceleration vectors to zero
    float4 acc    = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 angacc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    // Fetch particle position and velocity values from global read
    float4 x      = dev_x_sorted[idx];
    float4 vel    = dev_vel_sorted[idx];
    float4 angvel = dev_angvel_sorted[idx];
    float  radius = dev_radius_sorted[idx];

    // Coherent read from constant memory to registers
    float  dt    = devC_dt;
    float3 origo = make_float3(devC_origo[0], devC_origo[1], devC_origo[2]); 
    float3 L     = make_float3(devC_L[0], devC_L[1], devC_L[2]);
    float  rho   = devC_rho;

    // Particle mass
    float m = 4.0f/3.0f * PI * radius*radius*radius * rho;

    // Update linear acceleration of particle
    acc.x = force.x / m;
    acc.y = force.y / m;
    acc.z = force.z / m;

    // Update angular acceleration of particle 
    // (angacc = (total moment)/Intertia, intertia = 2/5*m*r^2)
    angacc.x = torque.x * 1.0f / (2.0f/5.0f * m * radius*radius);
    angacc.y = torque.y * 1.0f / (2.0f/5.0f * m * radius*radius);
    angacc.z = torque.z * 1.0f / (2.0f/5.0f * m * radius*radius);

    // Add gravity
    acc.x += devC_g[0];
    acc.y += devC_g[1];
    acc.z += devC_g[2];
    //acc.z -= 9.82f;

    // Only update velocity (and position) if the horizontal velocity is not fixed
    if (vel.w > 0.0f) {

      // Zero horizontal acceleration
      acc.x = 0.0f;
      acc.y = 0.0f;

      // Update vertical linear velocity
      vel.z += acc.z * dt;

      // Zero the angular acceleration and -velocity
      angacc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    } 

    // Update linear velocity
    vel.x += acc.x * dt;
    vel.y += acc.y * dt;
    vel.z += acc.z * dt;

    // Update angular velocity
    angvel.x += angacc.x * dt;
    angvel.y += angacc.y * dt;
    angvel.z += angacc.z * dt;

    // Update position. First-order Euler's scheme:
    //x.x += vel.x * dt;
    //x.y += vel.y * dt;
    //x.z += vel.z * dt;

    // Update position. Second-order scheme based on Taylor expansion 
    // (greater accuracy than the first-order Euler's scheme)
    x.x += vel.x * dt + (acc.x * dt*dt)/2.0f;
    x.y += vel.y * dt + (acc.y * dt*dt)/2.0f;
    x.z += vel.z * dt + (acc.z * dt*dt)/2.0f;


    // Add x-displacement for this time step to 
    // sum of x-displacements
    x.w += vel.x * dt + (acc.x * dt*dt)/2.0f;

    // Move particle across boundary if it is periodic
    if (devC_periodic == 1) {
      if (x.x < origo.x)
	x.x += L.x;
      if (x.x > L.x)
	x.x -= L.x;
      if (x.y < origo.y)
	x.y += L.y;
      if (x.y > L.y)
	x.y -= L.y;
    } else if (devC_periodic == 2) {
      if (x.x < origo.x)
	x.x += L.x;
      if (x.x > L.x)
	x.x -= L.x;
    }

    // Hold threads for coalesced write
    __syncthreads();

    // Store data in global memory at original, pre-sort positions
    dev_angvel[orig_idx] = angvel;
    dev_vel[orig_idx]    = vel;
    dev_x[orig_idx]      = x;
  } 
} // End of integrate(...)


// Reduce wall force contributions from particles to a single value per wall
__global__ void summation(float* in, float *out)
{
  __shared__ float cache[256];
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int cacheIdx = threadIdx.x;

  float temp = 0.0f;
  while (idx < devC_np) {
    temp += in[idx];
    idx += blockDim.x * gridDim.x;
  }

  // Set the cache values
  cache[cacheIdx] = temp;

  __syncthreads();

  // For reductions, threadsPerBlock must be a power of two
  // because of the following code
  unsigned int i = blockDim.x/2;
  while (i != 0) {
    if (cacheIdx < i)
      cache[cacheIdx] += cache[cacheIdx + i];
    __syncthreads();
    i /= 2;
  }

  // Write sum for block to global memory
  if (cacheIdx == 0)
    out[blockIdx.x] = cache[0];
}

// Update wall positions
__global__ void integrateWalls(float4* dev_w_nx, 
    			       float4* dev_w_mvfd,
			       float* dev_w_force_partial,
			       unsigned int blocksPerGrid)
{
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x; // Thread id

  if (idx < devC_nw) { // Condition prevents block size error

    // Copy data to temporary arrays to avoid any potential read-after-write, 
    // write-after-read, or write-after-write hazards. 
    float4 w_nx   = dev_w_nx[idx];
    float4 w_mvfd = dev_w_mvfd[idx];

    // Find the final sum of forces on wall
    w_mvfd.z = 0.0f;
    for (int i=0; i<blocksPerGrid; ++i) {
      w_mvfd.z += dev_w_force_partial[i];
    }

    float dt = devC_dt;

    // Normal load = Deviatoric stress times wall surface area,
    // directed downwards.
    float N = -w_mvfd.w*devC_L[0]*devC_L[1];

    // Calculate resulting acceleration of wall
    // (Wall mass is stored in w component of position float4)
    float acc = (w_mvfd.z+N)/w_mvfd.x;

    // Update linear velocity
    w_mvfd.y += acc * dt;

    // Update position. Second-order scheme based on Taylor expansion 
    // (greater accuracy than the first-order Euler's scheme)
    w_nx.w += w_mvfd.y * dt + (acc * dt*dt)/2.0f;

    // Store data in global memory
    dev_w_nx[idx]   = w_nx;
    dev_w_mvfd[idx] = w_mvfd;
  }
} // End of integrateWalls(...)




// Wrapper function for initializing the CUDA components.
// Called from main.cpp
//extern "C"
__host__ void initializeGPU(void)
{
  using std::cout;

  // Specify target device
  int cudadevice = 0;

  // Variables containing device properties
  cudaDeviceProp prop;
  int devicecount;
  int cudaDriverVersion;
  int cudaRuntimeVersion;


  // Register number of devices
  cudaGetDeviceCount(&devicecount);

  if(devicecount == 0) {
    cout << "\nERROR: No CUDA-enabled devices availible. Bye.\n";
    exit(EXIT_FAILURE);
  } else if (devicecount == 1) {
    cout << "\nSystem contains 1 CUDA compatible device.\n";
  } else {
    cout << "\nSystem contains " << devicecount << " CUDA compatible devices.\n";
  }

  cudaGetDeviceProperties(&prop, cudadevice);
  cudaDriverGetVersion(&cudaDriverVersion);
  cudaRuntimeGetVersion(&cudaRuntimeVersion);

  cout << "Using CUDA device ID: " << cudadevice << "\n";
  cout << "  - Name: " <<  prop.name << ", compute capability: " 
    << prop.major << "." << prop.minor << ".\n";
  cout << "  - CUDA Driver version: " << cudaDriverVersion/1000 
    << "." <<  cudaDriverVersion%100 
    << ", runtime version " << cudaRuntimeVersion/1000 << "." 
    << cudaRuntimeVersion%100 << "\n\n";

  // Comment following line when using a system only containing exclusive mode GPUs
  cudaChooseDevice(&cudadevice, &prop); 

  checkForCudaErrors("After initializing CUDA device");
}

// Copy selected constant components to constant device memory.
//extern "C"
__host__ void transferToConstantMemory(Particles* p,
    				       Grid* grid, 
				       Time* time, 
				       Params* params)
{
  using std::cout;

  cout << "\n  Transfering data to constant device memory:     ";

  cudaMemcpyToSymbol("devC_np", &p->np, sizeof(p->np));
  cudaMemcpyToSymbol("devC_nc", &NC, sizeof(char));
  cudaMemcpyToSymbol("devC_origo", grid->origo, sizeof(float)*ND);
  cudaMemcpyToSymbol("devC_L", grid->L, sizeof(float)*ND);
  cudaMemcpyToSymbol("devC_num", grid->num, sizeof(unsigned int)*ND);
  cudaMemcpyToSymbol("devC_dt", &time->dt, sizeof(float));
  cudaMemcpyToSymbol("devC_global", &params->global, sizeof(int));
  cudaMemcpyToSymbol("devC_g", params->g, sizeof(float)*ND);
  cudaMemcpyToSymbol("devC_nw", &params->nw, sizeof(unsigned int));
  cudaMemcpyToSymbol("devC_periodic", &params->periodic, sizeof(int));

  if (params->global == 1) {
    // If the physical properties of the particles are global (params.global == true),
    //   copy the values from the first particle into the designated constant memory. 
    //printf("(params.global == %d) ", params.global);
    params->k_n     = p->k_n[0];
    params->k_s	    = p->k_s[0];
    params->k_r	    = p->k_r[0];
    params->gamma_s = p->gamma_s[0];
    params->gamma_r = p->gamma_r[0];
    params->mu_s    = p->mu_s[0];
    params->mu_r    = p->mu_r[0];
    params->C       = p->C[0];
    params->rho     = p->rho[0];
    params->E       = p->E[0];
    params->K       = p->K[0];
    params->nu      = p->nu[0];
    cudaMemcpyToSymbol("devC_k_n", &params->k_n, sizeof(float));
    cudaMemcpyToSymbol("devC_k_s", &params->k_s, sizeof(float));
    cudaMemcpyToSymbol("devC_k_r", &params->k_r, sizeof(float));
    cudaMemcpyToSymbol("devC_gamma_s", &params->gamma_s, sizeof(float));
    cudaMemcpyToSymbol("devC_gamma_r", &params->gamma_r, sizeof(float));
    cudaMemcpyToSymbol("devC_mu_s", &params->mu_s, sizeof(float));
    cudaMemcpyToSymbol("devC_mu_r", &params->mu_r, sizeof(float));
    cudaMemcpyToSymbol("devC_C", &params->C, sizeof(float));
    cudaMemcpyToSymbol("devC_rho", &params->rho, sizeof(float));
    cudaMemcpyToSymbol("devC_E", &params->E, sizeof(float));
    cudaMemcpyToSymbol("devC_K", &params->K, sizeof(float));
    cudaMemcpyToSymbol("devC_nu", &params->nu, sizeof(float));
    cudaMemcpyToSymbol("devC_kappa", &params->kappa, sizeof(float));
    cudaMemcpyToSymbol("devC_db", &params->db, sizeof(float));
    cudaMemcpyToSymbol("devC_V_b", &params->V_b, sizeof(float));
    cudaMemcpyToSymbol("devC_shearmodel", &params->shearmodel, sizeof(unsigned int));
  } else {
    //printf("(params.global == %d) ", params.global);
    // Copy params structure with individual physical values from host to global memory
    //Params *dev_params;
    //HANDLE_ERROR(cudaMalloc((void**)&dev_params, sizeof(Params)));
    //HANDLE_ERROR(cudaMemcpyToSymbol(dev_params, &params, sizeof(Params)));
    //printf("Done\n");
    cout << "\n\nError: SPHERE is not yet ready for non-global physical variables.\nBye!\n";
    exit(EXIT_FAILURE); // Return unsuccessful exit status
  }
  checkForCudaErrors("After transferring to device constant memory");

  cout << "Done\n";
}


//extern "C"
__host__ void gpuMain(float4* host_x,
		      float4* host_vel,
		      float4* host_acc,
		      float4* host_angvel,
		      float4* host_angacc,
		      float4* host_force,
		      float4* host_torque,
		      uint4*  host_bonds,
		      Particles* p, 
		      Grid* grid, 
		      Time* time, 
		      Params* params,
		      float4* host_w_nx,
		      float4* host_w_mvfd,
		      const char* cwd, 
		      const char* inputbin)
{

  using std::cout;	// Namespace directive

  // Copy data to constant global device memory
  transferToConstantMemory(p, grid, time, params);

  // Declare pointers for particle variables on the device
  float4* dev_x;	// Particle position
  float4* dev_vel;	// Particle linear velocity
  float4* dev_angvel;	// Particle angular velocity
  float4* dev_acc;	// Particle linear acceleration
  float4* dev_angacc;	// Particle angular acceleration
  float4* dev_force;	// Sum of forces
  float4* dev_torque;	// Sum of torques
  float*  dev_radius;	// Particle radius
  float*  dev_es_dot;	// Current shear energy producion rate
  float*  dev_es;	// Total shear energy excerted on particle
  float*  dev_p;	// Pressure excerted onto particle
  //uint4*  dev_bonds;	// Particle bond pairs

  // Declare pointers for wall vectors on the device
  float4* dev_w_nx;            // Wall normal (x,y,z) and position (w)
  float4* dev_w_mvfd;          // Wall mass (x), velocity (y), force (z) 
  			       // and deviatoric stress (w)
  float*  dev_w_force;	       // Resulting force on wall per particle
  float*  dev_w_force_partial; // Partial sum from block of threads

  // Memory for sorted particle data
  float4* dev_x_sorted;
  float4* dev_vel_sorted;
  float4* dev_angvel_sorted;
  float*  dev_radius_sorted; 
  //uint4*  dev_bonds_sorted;

  // Grid-particle array pointers
  unsigned int* dev_gridParticleCellID;
  unsigned int* dev_gridParticleIndex;
  unsigned int* dev_cellStart;
  unsigned int* dev_cellEnd;

  // Particle contact bookkeeping
  unsigned int* dev_contacts;
  // x,y,z contains the interparticle vector, corrected if contact 
  // is across a periodic boundary. 
  // w contains radius of particle b.
  float4* dev_x_ab_r_b;
  float4* dev_delta_t; // Accumulated shear distance of contact

  // Particle memory size
  unsigned int memSizef  = sizeof(float) * p->np;
  unsigned int memSizef4 = sizeof(float4) * p->np;

  // Allocate device memory for particle variables,
  // tie to previously declared pointers
  cout << "  Allocating device memory:                       ";

  // Particle arrays
  cudaMalloc((void**)&dev_x, memSizef4);
  cudaMalloc((void**)&dev_x_sorted, memSizef4);
  cudaMalloc((void**)&dev_vel, memSizef4);
  cudaMalloc((void**)&dev_vel_sorted, memSizef4);
  cudaMalloc((void**)&dev_angvel, memSizef4);
  cudaMalloc((void**)&dev_angvel_sorted, memSizef4);
  cudaMalloc((void**)&dev_acc, memSizef4);
  cudaMalloc((void**)&dev_angacc, memSizef4);
  cudaMalloc((void**)&dev_force, memSizef4);
  cudaMalloc((void**)&dev_torque, memSizef4);
  cudaMalloc((void**)&dev_radius, memSizef);
  cudaMalloc((void**)&dev_radius_sorted, memSizef);
  cudaMalloc((void**)&dev_es_dot, memSizef);
  cudaMalloc((void**)&dev_es, memSizef);
  cudaMalloc((void**)&dev_p, memSizef);
  //cudaMalloc((void**)&dev_bonds, sizeof(uint4) * p->np);
  //cudaMalloc((void**)&dev_bonds_sorted, sizeof(uint4) * p->np);

  // Cell-related arrays
  cudaMalloc((void**)&dev_gridParticleCellID, sizeof(unsigned int)*p->np);
  cudaMalloc((void**)&dev_gridParticleIndex, sizeof(unsigned int)*p->np);
  cudaMalloc((void**)&dev_cellStart, sizeof(unsigned int)*grid->num[0]*grid->num[1]*grid->num[2]);
  cudaMalloc((void**)&dev_cellEnd, sizeof(unsigned int)*grid->num[0]*grid->num[1]*grid->num[2]);

  // Particle contact bookkeeping arrays
  cudaMalloc((void**)&dev_contacts, sizeof(unsigned int)*p->np*NC); // Max NC contacts per particle
  cudaMalloc((void**)&dev_x_ab_r_b, sizeof(float4)*p->np*NC);
  cudaMalloc((void**)&dev_delta_t, sizeof(float4)*p->np*NC);

  // Wall arrays
  cudaMalloc((void**)&dev_w_nx, sizeof(float)*params->nw*4);
  cudaMalloc((void**)&dev_w_mvfd, sizeof(float)*params->nw*4);
  cudaMalloc((void**)&dev_w_force, sizeof(float)*params->nw*p->np);

  checkForCudaErrors("Post device memory allocation");
  cout << "Done\n";

  // Transfer data from host to gpu device memory
  cout << "  Transfering data to the device:                 ";

  // Particle data
  cudaMemcpy(dev_x, host_x, memSizef4, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_vel, host_vel, memSizef4, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_acc, host_acc, memSizef4, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_angvel, host_angvel, memSizef4, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_angacc, host_angacc, memSizef4, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_force, host_force, memSizef4, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_torque, host_torque, memSizef4, cudaMemcpyHostToDevice);
  //cudaMemcpy(dev_bonds, host_bonds, sizeof(uint4) * p->np, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_radius, p->radius, memSizef, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_es_dot, p->es_dot, memSizef, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_es, p->es, memSizef, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_p, p->p, memSizef, cudaMemcpyHostToDevice);

  // Wall data (wall mass and number in constant memory)
  cudaMemcpy(dev_w_nx, host_w_nx, sizeof(float)*params->nw*4, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_w_mvfd, host_w_mvfd, sizeof(float)*params->nw*4, cudaMemcpyHostToDevice);
  
  // Initialize contacts lists to p.np
  unsigned int* npu = new unsigned int[p->np*NC];
  for (unsigned int i=0; i<(p->np*NC); ++i)
    npu[i] = p->np;
  cudaMemcpy(dev_contacts, npu, sizeof(unsigned int)*p->np*NC, cudaMemcpyHostToDevice);
  delete[] npu;

  // Create array of 0.0 values on the host and transfer these to the shear displacement array
  float4* zerosf4 = new float4[p->np*NC];
  for (unsigned int i=0; i<(p->np*NC); ++i)
    zerosf4[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  cudaMemcpy(dev_x_ab_r_b, zerosf4, sizeof(float4)*p->np*NC, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_delta_t, zerosf4, sizeof(float4)*p->np*NC, cudaMemcpyHostToDevice);
  delete[] zerosf4;

  checkForCudaErrors("Post memcopy");
  cout << "Done\n";

  // Synchronization point
  cudaThreadSynchronize();
  checkForCudaErrors("Start of mainLoop()");

  // Model world variables
  float tic, toc, filetimeclock, time_spent, dev_time_spent;

  // File output
  FILE* fp;
  char file[1000];  // Complete path+filename variable

  // Start CPU clock
  tic = clock();

  // GPU workload configuration
  unsigned int threadsPerBlock = 256; 
  // Create enough blocks to accomodate the particles
  unsigned int blocksPerGrid   = iDivUp(p->np, threadsPerBlock); 
  dim3 dimGrid(blocksPerGrid, 1, 1); // Blocks arranged in 1D grid
  dim3 dimBlock(threadsPerBlock, 1, 1); // Threads arranged in 1D block
  // Shared memory per block
  unsigned int smemSize = sizeof(unsigned int)*(threadsPerBlock+1);

  cudaMalloc((void**)&dev_w_force_partial, sizeof(float)*dimGrid.x);

  // Report to stdout
  cout << "\n  Device memory allocation and transfer complete.\n"
       << "  - Blocks per grid: "
       << dimGrid.x << "*" << dimGrid.y << "*" << dimGrid.z << "\n"
       << "  - Threads per block: "
       << dimBlock.x << "*" << dimBlock.y << "*" << dimBlock.z << "\n"
       << "  - Shared memory required per block: " << smemSize << " bytes\n";

  // Initialize counter variable values
  filetimeclock = 0.0;
  long iter = 0;

  // Create first status.dat
  sprintf(file,"%s/output/%s.status.dat", cwd, inputbin);
  fp = fopen(file, "w");
  fprintf(fp,"%2.4e %2.4e %d\n", 
      	  time->current, 
	  100.0*time->current/time->total, 
	  time->step_count);
  fclose(fp);

  // Write first output data file: output0.bin, thus testing writing of bin files
  sprintf(file,"%s/output/%s.output0.bin", cwd, inputbin);
  if (fwritebin(file, p, host_x, host_vel, 
		host_angvel, host_force, 
		host_torque,
		host_bonds,
		grid, time, params,
		host_w_nx, host_w_mvfd) != 0)  {
    cout << "\n Problem during fwritebin \n";
    exit(EXIT_FAILURE);
  }

  cout << "\n  Entering the main calculation time loop...\n\n"
       << "  IMPORTANT: Do not close this terminal, doing so will \n"
       << "             terminate this SPHERE process. Follow the \n"
       << "             progress in MATLAB using:\n"
       << "                >> status('" << inputbin << "')\n"
       << "             or in this directory by executing:\n"
       << "                $ ./sphere_status " << inputbin << "\n\n";

  // Enable cuPrintf()
  //cudaPrintfInit();

  // Start GPU clock
  cudaEvent_t dev_tic, dev_toc;
  cudaEventCreate(&dev_tic);
  cudaEventCreate(&dev_toc);
  cudaEventRecord(dev_tic, 0);

  cout << "  Current simulation time: " << time->current << " s.";


  // MAIN CALCULATION TIME LOOP
  while (time->current <= time->total) {

    // Increment iteration counter
    ++iter;

    // Print current step number to terminal
    //printf("Step: %d\n", time.step_count);


    // Routine check for errors
    checkForCudaErrors("Start of main while loop");


    // For each particle: 
    // Compute hash key (cell index) from position 
    // in the fine, uniform and homogenous grid.
    calcParticleCellID<<<dimGrid, dimBlock>>>(dev_gridParticleCellID, 
					      dev_gridParticleIndex, 
					      dev_x);

    // Synchronization point
    cudaThreadSynchronize();
    checkForCudaErrors("Post calcParticleCellID");


    // Sort particle (key, particle ID) pairs by hash key with Thrust radix sort
    thrust::sort_by_key(thrust::device_ptr<uint>(dev_gridParticleCellID),
			thrust::device_ptr<uint>(dev_gridParticleCellID + p->np),
			thrust::device_ptr<uint>(dev_gridParticleIndex));
    cudaThreadSynchronize(); // Needed? Does thrust synchronize threads implicitly?
    checkForCudaErrors("Post calcParticleCellID");


    // Zero cell array values by setting cellStart to its highest possible value,
    // specified with pointer value 0xffffffff, which for a 32 bit unsigned int
    // is 4294967295.
    cudaMemset(dev_cellStart, 0xffffffff, 
	       grid->num[0]*grid->num[1]*grid->num[2]*sizeof(unsigned int));
    cudaThreadSynchronize();
    checkForCudaErrors("Post cudaMemset");

    // Use sorted order to reorder particle arrays (position, velocities, radii) to ensure
    // coherent memory access. Save ordered configurations in new arrays (*_sorted).
    reorderArrays<<<dimGrid, dimBlock, smemSize>>>(dev_cellStart, 
						   dev_cellEnd,
						   dev_gridParticleCellID, 
						   dev_gridParticleIndex,
						   dev_x, dev_vel, 
						   dev_angvel, dev_radius, 
						   //dev_bonds,
						   dev_x_sorted, 
						   dev_vel_sorted, 
						   dev_angvel_sorted, 
						   dev_radius_sorted);
						   //dev_bonds_sorted);

    // Synchronization point
    cudaThreadSynchronize();
    checkForCudaErrors("Post reorderArrays", iter);

    // The contact search in topology() is only necessary for determining
    // the accumulated shear distance needed in the linear elastic
    // shear force model
    if (params->shearmodel == 2) {
      // For each particle: Search contacts in neighbor cells
      topology<<<dimGrid, dimBlock>>>(dev_cellStart, 
	  			      dev_cellEnd,
				      dev_gridParticleIndex,
				      dev_x_sorted, 
				      dev_radius_sorted, 
				      dev_contacts, 
				      dev_x_ab_r_b);

      // Empty cuPrintf() buffer to console
      //cudaThreadSynchronize();
      //cudaPrintfDisplay(stdout, true);

      // Synchronization point
      cudaThreadSynchronize();
      checkForCudaErrors("Post topology. Possibly caused by numerical instability. Is the computational time step too large?", iter);
    }

    // For each particle: Process collisions and compute resulting forces.
    interact<<<dimGrid, dimBlock>>>(dev_gridParticleIndex,
				    dev_cellStart,
				    dev_cellEnd,
				    dev_x_sorted, dev_radius_sorted,
				    dev_vel_sorted, dev_angvel_sorted,
				    dev_vel, dev_angvel,
				    dev_force, dev_torque,
				    dev_es_dot, dev_es, dev_p,
				    dev_w_nx, dev_w_mvfd, dev_w_force,
				    //dev_bonds_sorted,
				    dev_contacts, dev_x_ab_r_b, 
				    dev_delta_t);

    // Empty cuPrintf() buffer to console
    //cudaThreadSynchronize();
    //cudaPrintfDisplay(stdout, true);

    // Synchronization point
    cudaThreadSynchronize();
    checkForCudaErrors("Post interact - often caused if particles move outside the grid", iter);


    // Update particle kinematics
    integrate<<<dimGrid, dimBlock>>>(dev_x_sorted, dev_vel_sorted, 
				     dev_angvel_sorted, dev_radius_sorted,
				     dev_x, dev_vel, dev_angvel,
				     dev_force, dev_torque, 
				     dev_gridParticleIndex);

    // Summation of forces on wall
    summation<<<dimGrid, dimBlock>>>(dev_w_force, dev_w_force_partial);

    // Synchronization point
    cudaThreadSynchronize();
    checkForCudaErrors("Post integrate & wall force summation");

    // Update wall kinematics
    integrateWalls<<< 1, params->nw>>>(dev_w_nx, 
				       dev_w_mvfd,
				       dev_w_force_partial,
				       blocksPerGrid);

    // Synchronization point
    cudaThreadSynchronize();
    checkForCudaErrors("Post integrateWalls");

    // Update timers and counters
    time->current    += time->dt;
    filetimeclock    += time->dt;

    // Report time to console
    cout << "\r  Current simulation time: " << time->current << " s.        ";


    // Produce output binary if the time interval 
    // between output files has been reached
    if (filetimeclock > time->file_dt) {

      // Pause the CPU thread until all CUDA calls previously issued are completed
      cudaThreadSynchronize();
      checkForCudaErrors("Beginning of file output section");

      //// Copy device data to host memory

      // Particle data
      cudaMemcpy(host_x, dev_x, memSizef4, cudaMemcpyDeviceToHost);
      cudaMemcpy(host_vel, dev_vel, memSizef4, cudaMemcpyDeviceToHost);
      cudaMemcpy(host_acc, dev_acc, memSizef4, cudaMemcpyDeviceToHost);
      cudaMemcpy(host_angvel, dev_angvel, memSizef4, cudaMemcpyDeviceToHost);
      cudaMemcpy(host_force, dev_force, memSizef4, cudaMemcpyDeviceToHost);
      cudaMemcpy(host_torque, dev_torque, memSizef4, cudaMemcpyDeviceToHost);
      //cudaMemcpy(host_bonds, dev_bonds, sizeof(uint4) * p->np, cudaMemcpyDeviceToHost);
      cudaMemcpy(p->es_dot, dev_es_dot, memSizef, cudaMemcpyDeviceToHost);
      cudaMemcpy(p->es, dev_es, memSizef, cudaMemcpyDeviceToHost);
      cudaMemcpy(p->p, dev_p, memSizef, cudaMemcpyDeviceToHost);

      // Wall data
      cudaMemcpy(host_w_nx, dev_w_nx, 
	  	 sizeof(float)*params->nw*4, cudaMemcpyDeviceToHost);
      cudaMemcpy(host_w_mvfd, dev_w_mvfd, 
	  	 sizeof(float)*params->nw*4, cudaMemcpyDeviceToHost);

      // Pause the CPU thread until all CUDA calls previously issued are completed
      cudaThreadSynchronize();

      // Write binary output file
      time->step_count += 1;
      sprintf(file,"%s/output/%s.output%d.bin", cwd, inputbin, time->step_count);

      if (fwritebin(file, p, host_x, host_vel, 
	    	    host_angvel, host_force, 
		    host_torque, host_bonds,
		    grid, time, params,
	    	    host_w_nx, host_w_mvfd) != 0) {
	cout << "\n Error during fwritebin() in main loop\n";
	exit(EXIT_FAILURE);
      }

      // Update status.dat at the interval of filetime 
      sprintf(file,"%s/output/%s.status.dat", cwd, inputbin);
      fp = fopen(file, "w");
      fprintf(fp,"%2.4e %2.4e %d\n", 
	      time->current, 
	      100.0*time->current/time->total,
	      time->step_count);
      fclose(fp);

      filetimeclock = 0.0;
    }
  }

  // Stop clock and display calculation time spent
  toc = clock();
  cudaEventRecord(dev_toc, 0);
  cudaEventSynchronize(dev_toc);

  time_spent = (toc - tic)/(CLOCKS_PER_SEC);
  cudaEventElapsedTime(&dev_time_spent, dev_tic, dev_toc);

  cout << "\nSimulation ended. Statistics:\n"
       << "  - Last output file number: " 
       << time->step_count << "\n"
       << "  - GPU time spent: "
       << dev_time_spent/1000.0f << " s\n"
       << "  - CPU time spent: "
       << time_spent << " s\n"
       << "  - Mean duration of iteration:\n"
       << "      " << dev_time_spent/((double)iter*1000.0f) << " s\n"; 

  cudaEventDestroy(dev_tic);
  cudaEventDestroy(dev_toc);

  // Free memory allocated to cudaPrintfInit
  //cudaPrintfEnd();

  // Free GPU device memory  
  printf("\nLiberating device memory:                        ");

  // Particle arrays
  cudaFree(dev_x);
  cudaFree(dev_x_sorted);
  cudaFree(dev_vel);
  cudaFree(dev_vel_sorted);
  cudaFree(dev_angvel);
  cudaFree(dev_angvel_sorted);
  cudaFree(dev_acc);
  cudaFree(dev_angacc);
  cudaFree(dev_force);
  cudaFree(dev_torque);
  cudaFree(dev_radius);
  cudaFree(dev_radius_sorted);
  cudaFree(dev_es_dot);
  cudaFree(dev_es);
  cudaFree(dev_p);
  //cudaFree(dev_bonds);
  //cudaFree(dev_bonds_sorted);
  cudaFree(dev_contacts);
  cudaFree(dev_x_ab_r_b);
  cudaFree(dev_delta_t);

  // Cell-related arrays
  cudaFree(dev_gridParticleIndex);
  cudaFree(dev_cellStart);
  cudaFree(dev_cellEnd);

  // Wall arrays
  cudaFree(dev_w_nx);
  cudaFree(dev_w_mvfd);
  cudaFree(dev_w_force);
  cudaFree(dev_w_force_partial);

  printf("Done\n");
} /* EOF */
