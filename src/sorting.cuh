// Returns the cellID containing the particle, based cubic grid
// See Bayraktar et al. 2009
// Kernel is executed on the device, and is callable from the device only
__device__ int calcCellID(Float3 x) 
{ 
  unsigned int i_x, i_y, i_z;

  // Calculate integral coordinates:
  i_x = floor((x.x - devC_grid.origo[0]) / (devC_grid.L[0]/devC_grid.num[0]));
  i_y = floor((x.y - devC_grid.origo[1]) / (devC_grid.L[1]/devC_grid.num[1]));
  i_z = floor((x.z - devC_grid.origo[2]) / (devC_grid.L[2]/devC_grid.num[2]));

  // Integral coordinates are converted to 1D coordinate:
  return (i_z * devC_grid.num[1]) * devC_grid.num[0] + i_y * devC_grid.num[0] + i_x;

} // End of calcCellID(...)


// Calculate hash value for each particle, based on position in grid.
// Kernel executed on device, and callable from host only.
__global__ void calcParticleCellID(unsigned int* dev_gridParticleCellID, 
    				   unsigned int* dev_gridParticleIndex, 
				   Float4* dev_x) 
{
  //unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x; // Thread id
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < devC_params.np) { // Condition prevents block size error

    //volatile Float4 x = dev_x[idx]; // Ensure coalesced read
    Float4 x = dev_x[idx]; // Ensure coalesced read

    unsigned int cellID = calcCellID(MAKE_FLOAT3(x.x, x.y, x.z));

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
			      Float4* dev_x, Float4* dev_vel, 
			      Float4* dev_angvel, Float* dev_radius,
			      //uint4* dev_bonds,
			      Float4* dev_x_sorted, Float4* dev_vel_sorted,
			      Float4* dev_angvel_sorted, Float* dev_radius_sorted)
			      //uint4* dev_bonds_sorted)
{ 

  // Create hash array in shared on-chip memory. The size of the array 
  // (threadsPerBlock + 1) is determined at launch time (extern notation).
  extern __shared__ unsigned int shared_data[]; 

  // Thread index in block
  unsigned int tidx = threadIdx.x;

  // Thread index in grid
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x; 

  // CellID hash value of particle idx
  unsigned int cellID;

  // Read cellID data and store it in shared memory (shared_data)
  if (idx < devC_params.np) { // Condition prevents block size error
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
  if (idx < devC_params.np) { // Condition prevents block size error
    // If this particle has a different cell index to the previous particle, it's the first
    // particle in the cell -> Store the index of this particle in the cell.
    // The previous particle must be the last particle in the previous cell.
    if (idx == 0 || cellID != shared_data[tidx]) {
      dev_cellStart[cellID] = idx;
      if (idx > 0)
	dev_cellEnd[shared_data[tidx]] = idx;
    }

    // Check wether the thread is the last one
    if (idx == (devC_params.np - 1)) 
      dev_cellEnd[cellID] = idx + 1;


    // Use the sorted index to reorder the position and velocity data
    unsigned int sortedIndex = dev_gridParticleIndex[idx];

    // Fetch from global read
    Float4 x      = dev_x[sortedIndex];
    Float4 vel    = dev_vel[sortedIndex];
    Float4 angvel = dev_angvel[sortedIndex];
    Float  radius = dev_radius[sortedIndex];
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



