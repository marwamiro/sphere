#ifndef CONTACTSEARCH_CUH_
#define CONTACTSEARCH_CUH_

// contactsearch.cuh
// Functions for identifying contacts and processing boundaries


// Calculate the distance modifier between a contact pair. The modifier
// accounts for periodic boundaries. If the target cell lies outside
// the grid, it returns -1.
// Function is called from overlapsInCell() and findContactsInCell().
__device__ int findDistMod(int3* targetCell, Float3* distmod)
{
  // Check whether x- and y boundaries are to be treated as periodic
  // 1: x- and y boundaries periodic
  // 2: x boundaries periodic
  if (devC_periodic == 1) {

    // Periodic x-boundary
    if (targetCell->x < 0) {
      targetCell->x = devC_num[0] - 1;
      *distmod += MAKE_FLOAT3(devC_L[0], 0.0f, 0.0f);
    }
    if (targetCell->x == devC_num[0]) {
      targetCell->x = 0;
      *distmod -= MAKE_FLOAT3(devC_L[0], 0.0f, 0.0f);
    }

    // Periodic y-boundary
    if (targetCell->y < 0) {
      targetCell->y = devC_num[1] - 1;
      *distmod += MAKE_FLOAT3(0.0f, devC_L[1], 0.0f);
    }
    if (targetCell->y == devC_num[1]) {
      targetCell->y = 0;
      *distmod -= MAKE_FLOAT3(0.0f, devC_L[1], 0.0f);
    }


  // Only x-boundaries are periodic
  } else if (devC_periodic == 2) {

    // Periodic x-boundary
    if (targetCell->x < 0) {
      targetCell->x = devC_num[0] - 1;
      *distmod += MAKE_FLOAT3(devC_L[0], 0.0f, 0.0f);
    }
    if (targetCell->x == devC_num[0]) {
      targetCell->x = 0;
      *distmod -= MAKE_FLOAT3(devC_L[0], 0.0f, 0.0f);
    }

    // Hande out-of grid cases on y-axis
    if (targetCell->y < 0 || targetCell->y == devC_num[1])
      return -1;


  // No periodic boundaries
  } else {

    // Hande out-of grid cases on x- and y-axes
    if (targetCell->x < 0 || targetCell->x == devC_num[0])
      return -1;
    if (targetCell->y < 0 || targetCell->y == devC_num[1])
      return -1;
  }

  // Handle out-of-grid cases on z-axis
  if (targetCell->z < 0 || targetCell->z == devC_num[2])
    return -1;

  // Return successfully
  return 0;
}



// Find overlaps between particle no. 'idx' and particles in cell 'gridpos'.
// Contacts are processed as soon as they are identified.
// Used for shearmodel=1, where contact history is not needed.
// Kernel executed on device, and callable from device only.
// Function is called from interact().
__device__ void overlapsInCell(int3 targetCell, 
    			       unsigned int idx_a, 
			       Float4 x_a, Float radius_a,
			       Float3* N, Float3* T, 
			       Float* es_dot, Float* p,
			       Float4* dev_x_sorted, 
			       Float* dev_radius_sorted,
			       Float4* dev_vel_sorted, 
			       Float4* dev_angvel_sorted,
			       unsigned int* dev_cellStart, 
			       unsigned int* dev_cellEnd,
			       Float4* dev_w_nx, 
			       Float4* dev_w_mvfd)
			       //uint4 bonds)
{

  // Get distance modifier for interparticle
  // vector, if it crosses a periodic boundary
  Float3 distmod = MAKE_FLOAT3(0.0f, 0.0f, 0.0f);
  if (findDistMod(&targetCell, &distmod) == -1)
    return; // Target cell lies outside the grid


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
	Float4 x_b      = dev_x_sorted[idx_b];
	Float  radius_b = dev_radius_sorted[idx_b];
	Float  kappa 	= devC_kappa;

	// Distance between particle centers (Float4 -> Float3)
	Float3 x_ab = MAKE_FLOAT3(x_a.x - x_b.x, 
	    			  x_a.y - x_b.y, 
				  x_a.z - x_b.z);

	// Adjust interparticle vector if periodic boundary/boundaries
	// are crossed
	x_ab += distmod;
	Float x_ab_length = length(x_ab);

	// Distance between particle perimeters
	Float delta_ab = x_ab_length - (radius_a + radius_b); 

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
// Write the indexes of the overlaps in array contacts[].
// Used for shearmodel=2, where bookkeeping of contact history is necessary.
// Kernel executed on device, and callable from device only.
// Function is called from topology().
__device__ void findContactsInCell(int3 targetCell, 
    			           unsigned int idx_a, 
				   Float4 x_a, Float radius_a,
				   Float4* dev_x_sorted, 
				   Float* dev_radius_sorted,
				   unsigned int* dev_cellStart, 
				   unsigned int* dev_cellEnd,
				   unsigned int* dev_gridParticleIndex,
				   int* nc,
				   unsigned int* dev_contacts,
				   Float4* dev_distmod)
{
  // Get distance modifier for interparticle
  // vector, if it crosses a periodic boundary
  Float3 distmod = MAKE_FLOAT3(0.0f, 0.0f, 0.0f);
  if (findDistMod(&targetCell, &distmod) == -1)
    return; // Target cell lies outside the grid


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
	Float4 x_b      = dev_x_sorted[idx_b];
	Float  radius_b = dev_radius_sorted[idx_b];

	// Read the original index of particle B
	unsigned int idx_b_orig = dev_gridParticleIndex[idx_b];

	__syncthreads();

	// Distance between particle centers (Float4 -> Float3)
	Float3 x_ab = MAKE_FLOAT3(x_a.x - x_b.x, 
	    			  x_a.y - x_b.y, 
				  x_a.z - x_b.z);

	// Adjust interparticle vector if periodic boundary/boundaries
	// are crossed
	x_ab += distmod;

	Float x_ab_length = length(x_ab);

	// Distance between particle perimeters
	Float delta_ab = x_ab_length - (radius_a + radius_b); 

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

	  // Write the particle index to the relevant position,
	  // no matter if it already is there or not (concurrency of write)
	  dev_contacts[(unsigned int)(idx_a_orig*devC_nc+cpos)] = idx_b_orig;


	  // Write the interparticle vector and radius of particle B
	 //dev_x_ab_r_b[(unsigned int)(idx_a_orig*devC_nc+cpos)] = make_Float4(x_ab, radius_b);
	  dev_distmod[(unsigned int)(idx_a_orig*devC_nc+cpos)] = MAKE_FLOAT4(distmod.x, distmod.y, distmod.z, radius_b);
	  
	  // Increment contact counter
	  ++*nc;
	}

	// Write the inter-particle position vector correction and radius of particle B
	//dev_distmod[(unsigned int)(idx_a_orig*devC_nc+cpos)] = make_Float4(distmod, radius_b);

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
// Function is called from mainGPU loop.
__global__ void topology(unsigned int* dev_cellStart, 
    			 unsigned int* dev_cellEnd, // Input: Particles in cell 
			 unsigned int* dev_gridParticleIndex, // Input: Unsorted-sorted key
			 Float4* dev_x_sorted, Float* dev_radius_sorted, 
			 unsigned int* dev_contacts,
			 Float4* dev_distmod)
{
  // Thread index equals index of particle A
  unsigned int idx_a = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  if (idx_a < devC_np) {
    // Fetch particle data in global read
    Float4 x_a      = dev_x_sorted[idx_a];
    Float  radius_a = dev_radius_sorted[idx_a];

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
	    		     &nc, dev_contacts, dev_distmod);
	}
      }
    }
  }
} // End of topology(...)


// For a single particle:
// If shearmodel=1:
//   Search for neighbors to particle 'idx' inside the 27 closest cells, 
//   and compute the resulting force and torque on it.
// If shearmodel=2:
//   Process contacts saved in dev_contacts by topology(), and compute
//   the resulting force and torque on it.
// For all shearmodels:
//   Collide with top- and bottom walls, save resulting force on upper wall.
// Kernel is executed on device, and is callable from host only.
// Function is called from mainGPU loop.
__global__ void interact(unsigned int* dev_gridParticleIndex, // Input: Unsorted-sorted key
			 unsigned int* dev_cellStart,
			 unsigned int* dev_cellEnd,
			 Float4* dev_x, Float* dev_radius,
    			 Float4* dev_x_sorted, Float* dev_radius_sorted, 
			 Float4* dev_vel_sorted, Float4* dev_angvel_sorted,
			 Float4* dev_vel, Float4* dev_angvel,
			 Float4* dev_force, Float4* dev_torque,
			 Float* dev_es_dot, Float* dev_es, Float* dev_p,
			 Float4* dev_w_nx, Float4* dev_w_mvfd, 
			 Float* dev_w_force, //uint4* dev_bonds_sorted,
			 unsigned int* dev_contacts, 
			 Float4* dev_distmod,
			 Float4* dev_delta_t)
{
  // Thread index equals index of particle A
  unsigned int idx_a = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

  if (idx_a < devC_np) {

    // Fetch particle data in global read
    unsigned int idx_a_orig = dev_gridParticleIndex[idx_a];
    Float4 x_a      = dev_x_sorted[idx_a];
    Float  radius_a = dev_radius_sorted[idx_a];

    // Fetch wall data in global read
    Float4 w_up_nx   = dev_w_nx[0];
    Float4 w_up_mvfd = dev_w_mvfd[0];

    // Fetch world dimensions in constant memory read
    Float3 origo = MAKE_FLOAT3(devC_origo[0], 
			       devC_origo[1], 
			       devC_origo[2]); 
    Float3 L = MAKE_FLOAT3(devC_L[0], 
			   devC_L[1], 
			   devC_L[2]);

    // Index of particle which is bonded to particle A.
    // The index is equal to the particle no (p.np)
    // if particle A is bond-less.
    //uint4 bonds = dev_bonds_sorted[idx_a];

    // Initiate shear friction loss rate at 0.0
    Float es_dot = 0.0f; 

    // Initiate pressure on particle at 0.0
    Float p = 0.0f;

    // Allocate memory for temporal force/torque vector values
    Float3 F = MAKE_FLOAT3(0.0f, 0.0f, 0.0f);
    Float3 T = MAKE_FLOAT3(0.0f, 0.0f, 0.0f);

    // Apply linear elastic, frictional contact model to registered contacts
    if (devC_shearmodel == 2) {
      unsigned int idx_b_orig, mempos;
      Float delta_n, x_ab_length, radius_b;
      Float3 x_ab;
      Float4 x_b, distmod;
      Float4 vel_a     = dev_vel_sorted[idx_a];
      Float4 angvel4_a = dev_angvel_sorted[idx_a];
      Float3 angvel_a  = MAKE_FLOAT3(angvel4_a.x, angvel4_a.y, angvel4_a.z);

      // Loop over all possible contacts, and remove contacts
      // that no longer are valid (delta_n > 0.0)
      for (int i = 0; i<(int)devC_nc; ++i) {
	mempos = (unsigned int)(idx_a_orig * devC_nc + i);
	__syncthreads();
	idx_b_orig = dev_contacts[mempos];
	distmod    = dev_distmod[mempos];
	x_b        = dev_x[idx_b_orig];
	//radius_b   = dev_radius[idx_b_orig];
	radius_b   = distmod.w;

	// Inter-particle vector, corrected for periodic boundaries
	x_ab = MAKE_FLOAT3(x_a.x - x_b.x + distmod.x,
	    		   x_a.y - x_b.y + distmod.y,
			   x_a.z - x_b.z + distmod.z);

	x_ab_length = length(x_ab);
	delta_n = x_ab_length - (radius_a + radius_b);


	if (idx_b_orig != (unsigned int)devC_np) {

	  // Process collision if the particles are overlapping
	  if (delta_n < 0.0f) {
	    //cuPrintf("\nProcessing contact, idx_a_orig = %u, idx_b_orig = %u, contact = %d, delta_n = %f\n",
	    //  idx_a_orig, idx_b_orig, i, delta_n);
	    /*contactLinearViscous(&F, &T, &es_dot, &p, 
	      		       idx_a_orig, idx_b_orig,
			       dev_vel, 
			       dev_angvel,
			       radius_a, radius_b, 
			       x_ab, x_ab_length,
			       delta_n, devC_kappa);
	    dev_delta_t[mempos] = MAKE_FLOAT4(0.0f, 0.0f, 0.0f, 0.0f);*/
	    contactLinear(&F, &T, &es_dot, &p, 
			  idx_a_orig,
			  idx_b_orig,
			  vel_a,
			  dev_vel,
			  angvel_a,
			  dev_angvel,
			  radius_a, radius_b, 
			  x_ab, x_ab_length,
			  delta_n, dev_delta_t, 
			  mempos);
	  } else {
	    __syncthreads();
	    // Remove this contact (there is no particle with index=np)
	    dev_contacts[mempos] = devC_np; 
	    // Zero sum of shear displacement in this position
	    dev_delta_t[mempos] = MAKE_FLOAT4(0.0f, 0.0f, 0.0f, 0.0f);
	  }
	} else {
	  __syncthreads();
	  dev_delta_t[mempos] = MAKE_FLOAT4(0.0f, 0.0f, 0.0f, 0.0f);
	}
      } // Contact loop end


    // Find contacts and process collisions immidiately for
    // shearmodel 1 (visco-frictional).
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
			   &F, &T, &es_dot, &p,
			   dev_x_sorted, dev_radius_sorted, 
			   dev_vel_sorted, dev_angvel_sorted,
			   dev_cellStart, dev_cellEnd,
			   dev_w_nx, dev_w_mvfd);
	  }
	}
      }

    }

    //// Interact with walls
    Float delta_w; // Overlap distance
    Float3 w_n;    // Wall surface normal
    Float w_force = 0.0f; // Force on wall from particle A

    // Upper wall (idx 0)
    delta_w = w_up_nx.w - (x_a.z + radius_a);
    w_n = MAKE_FLOAT3(0.0f, 0.0f, -1.0f);
    if (delta_w < 0.0f) {
      w_force = contactLinear_wall(&F, &T, &es_dot, &p, idx_a, radius_a,
	  			   dev_vel_sorted, dev_angvel_sorted,
				   w_n, delta_w, w_up_mvfd.y);
    }

    // Lower wall (force on wall not stored)
    delta_w = x_a.z - radius_a - origo.z;
    w_n = MAKE_FLOAT3(0.0f, 0.0f, 1.0f);
    if (delta_w < 0.0f) {
      (void)contactLinear_wall(&F, &T, &es_dot, &p, idx_a, radius_a,
	  		       dev_vel_sorted, dev_angvel_sorted,
	  		       w_n, delta_w, 0.0f);
    }


    if (devC_periodic == 0) {

      // Left wall
      delta_w = x_a.x - radius_a - origo.x;
      w_n = MAKE_FLOAT3(1.0f, 0.0f, 0.0f);
      if (delta_w < 0.0f) {
	(void)contactLinear_wall(&F, &T, &es_dot, &p, idx_a, radius_a,
	    			 dev_vel_sorted, dev_angvel_sorted,
				 w_n, delta_w, 0.0f);
      }

      // Right wall
      delta_w = L.x - (x_a.x + radius_a);
      w_n = MAKE_FLOAT3(-1.0f, 0.0f, 0.0f);
      if (delta_w < 0.0f) {
	(void)contactLinear_wall(&F, &T, &es_dot, &p, idx_a, radius_a,
	    			 dev_vel_sorted, dev_angvel_sorted,
				 w_n, delta_w, 0.0f);
      }

      // Front wall
      delta_w = x_a.y - radius_a - origo.y;
      w_n = MAKE_FLOAT3(0.0f, 1.0f, 0.0f);
      if (delta_w < 0.0f) {
	(void)contactLinear_wall(&F, &T, &es_dot, &p, idx_a, radius_a,
	    			 dev_vel_sorted, dev_angvel_sorted,
				 w_n, delta_w, 0.0f);
      }

      // Back wall
      delta_w = L.y - (x_a.y + radius_a);
      w_n = MAKE_FLOAT3(0.0f, -1.0f, 0.0f);
      if (delta_w < 0.0f) {
	(void)contactLinear_wall(&F, &T, &es_dot, &p, idx_a, radius_a,
	    			 dev_vel_sorted, dev_angvel_sorted,
				 w_n, delta_w, 0.0f);
      }

    } else if (devC_periodic == 2) {

      // Front wall
      delta_w = x_a.y - radius_a - origo.y;
      w_n = MAKE_FLOAT3(0.0f, 1.0f, 0.0f);
      if (delta_w < 0.0f) {
	(void)contactLinear_wall(&F, &T, &es_dot, &p, idx_a, radius_a,
	    			 dev_vel_sorted, dev_angvel_sorted,
				 w_n, delta_w, 0.0f);
      }

      // Back wall
      delta_w = L.y - (x_a.y + radius_a);
      w_n = MAKE_FLOAT3(0.0f, -1.0f, 0.0f);
      if (delta_w < 0.0f) {
	(void)contactLinear_wall(&F, &T, &es_dot, &p, idx_a, radius_a,
	    			 dev_vel_sorted, dev_angvel_sorted,
				 w_n, delta_w, 0.0f);
      }
    }


    // Hold threads for coalesced write
    __syncthreads();

    // Write force to unsorted position
    unsigned int orig_idx = dev_gridParticleIndex[idx_a];
    dev_force[orig_idx]   = MAKE_FLOAT4(F.x, F.y, F.z, 0.0f);
    dev_torque[orig_idx]  = MAKE_FLOAT4(T.x, T.y, T.z, 0.0f);
    dev_es_dot[orig_idx]  = es_dot;
    dev_es[orig_idx]     += es_dot * devC_dt;
    dev_p[orig_idx]       = p;
    dev_w_force[orig_idx] = w_force;
  }
} // End of interact(...)


#endif
