#ifndef INTEGRATION_CUH_
#define INTEGRATION_CUH_

// integration.cuh
// Functions responsible for temporal integration


// Second order integration scheme based on Taylor expansion of particle kinematics. 
// Kernel executed on device, and callable from host only.
__global__ void integrate(Float4* dev_x_sorted, Float4* dev_vel_sorted, // Input
			  Float4* dev_angvel_sorted, Float* dev_radius_sorted, // Input
			  Float4* dev_x, Float4* dev_vel, Float4* dev_angvel, // Output
			  Float4* dev_force, Float4* dev_torque, // Input
			  unsigned int* dev_gridParticleIndex) // Input: Sorted-Unsorted key
{
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x; // Thread id

  if (idx < devC_np) { // Condition prevents block size error

    // Copy data to temporary arrays to avoid any potential read-after-write, 
    // write-after-read, or write-after-write hazards. 
    unsigned int orig_idx = dev_gridParticleIndex[idx];
    Float4 force  = dev_force[orig_idx];
    Float4 torque = dev_torque[orig_idx];

    // Initialize acceleration vectors to zero
    Float4 acc    = MAKE_FLOAT4(0.0f, 0.0f, 0.0f, 0.0f);
    Float4 angacc = MAKE_FLOAT4(0.0f, 0.0f, 0.0f, 0.0f);

    // Fetch particle position and velocity values from global read
    Float4 x      = dev_x_sorted[idx];
    Float4 vel    = dev_vel_sorted[idx];
    Float4 angvel = dev_angvel_sorted[idx];
    Float  radius = dev_radius_sorted[idx];

    // Coherent read from constant memory to registers
    Float  dt    = devC_dt;
    Float3 origo = MAKE_FLOAT3(devC_origo[0], devC_origo[1], devC_origo[2]); 
    Float3 L     = MAKE_FLOAT3(devC_L[0], devC_L[1], devC_L[2]);
    Float  rho   = devC_rho;

    // Particle mass
    Float m = 4.0f/3.0f * PI * radius*radius*radius * rho;

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

    // Check if particle has a fixed horizontal velocity
    if (vel.w > 0.0f) {

      // Zero horizontal acceleration and disable
      // gravity to counteract segregation.
      // Particles may move in the z-dimension,
      // to allow for dilation.
      acc.x = 0.0f;
      acc.y = 0.0f;
      acc.z -= devC_g[2];

      // Zero the angular acceleration
      angacc = MAKE_FLOAT4(0.0f, 0.0f, 0.0f, 0.0f);
    } 

    // Update angular velocity
    angvel.x += angacc.x * dt;
    angvel.y += angacc.y * dt;
    angvel.z += angacc.z * dt;

    // Update linear velocity
    vel.x += acc.x * dt;
    vel.y += acc.y * dt;
    vel.z += acc.z * dt;

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
__global__ void summation(Float* in, Float *out)
{
  __shared__ Float cache[256];
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int cacheIdx = threadIdx.x;

  Float temp = 0.0f;
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
__global__ void integrateWalls(Float4* dev_w_nx, 
    			       Float4* dev_w_mvfd,
			       Float* dev_w_force_partial,
			       unsigned int blocksPerGrid)
{
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x; // Thread id

  if (idx < devC_nw) { // Condition prevents block size error

    // Copy data to temporary arrays to avoid any potential read-after-write, 
    // write-after-read, or write-after-write hazards. 
    Float4 w_nx   = dev_w_nx[idx];
    Float4 w_mvfd = dev_w_mvfd[idx];
    int wmode = devC_wmode[idx];  // Wall BC, 0: devs, 1: vel
    Float acc;

    // Find the final sum of forces on wall
    w_mvfd.z = 0.0f;
    for (int i=0; i<blocksPerGrid; ++i) {
      w_mvfd.z += dev_w_force_partial[i];
    }

    Float dt = devC_dt;

    // If wall BC is controlled by deviatoric stress:
    //if (wmode == 0) {

      // Normal load = Deviatoric stress times wall surface area,
      // directed downwards.
      Float N = -w_mvfd.w*devC_L[0]*devC_L[1];

      // Calculate resulting acceleration of wall
      // (Wall mass is stored in w component of position Float4)
      acc = (w_mvfd.z + N)/w_mvfd.x;

      // Update linear velocity
      w_mvfd.y += acc * dt;
    
    // Wall BC is controlled by velocity, which should not change
    //} else if (wmode == 1) { 
    //  acc = 0.0f;
    //}
     
    // Update position. Second-order scheme based on Taylor expansion 
    // (greater accuracy than the first-order Euler's scheme)
    w_nx.w += w_mvfd.y * dt + (acc * dt*dt)/2.0f;

    // Store data in global memory
    dev_w_nx[idx]   = w_nx;
    dev_w_mvfd[idx] = w_mvfd;
  }
} // End of integrateWalls(...)


#endif
