// device.cu -- GPU specific operations utilizing the CUDA API.
#include <iostream>
#include <cstdio>
#include <cuda.h>

#include "vector_arithmetic.h"
#include "thrust/device_ptr.h"
#include "thrust/sort.h"

#include "sphere.h"
#include "datatypes.h"
#include "utility.cuh"
#include "constants.cuh"
#include "debug.h"

#include "sorting.cuh"	
#include "contactmodels.cuh"
#include "cohesion.cuh"
#include "contactsearch.cuh"
#include "integration.cuh"

//#include "cuPrintf.cu"

// Wrapper function for initializing the CUDA components.
// Called from main.cpp
//extern "C"
__host__ void DEM::initializeGPU(void)
{
  using std::cout; // stdout

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
    std::cerr << "\nERROR: No CUDA-enabled devices availible. Bye.\n";
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

// Start timer for kernel profiling
__host__ void startTimer(cudaEvent_t* kernel_tic)
{
  cudaEventRecord(*kernel_tic);
}

// Stop timer for kernel profiling and time to function sum
__host__ void stopTimer(cudaEvent_t *kernel_tic,
    			cudaEvent_t *kernel_toc,
			float *kernel_elapsed,
			double* sum)
{
    cudaEventRecord(*kernel_toc, 0);
    cudaEventSynchronize(*kernel_toc);
    cudaEventElapsedTime(kernel_elapsed, *kernel_tic, *kernel_toc);
    *sum += *kernel_elapsed;
}

// Check values of parameters in constant memory
__global__ void checkConstantValues(int* dev_equal,
    				    Grid* dev_grid,
				    Params* dev_params)
{

  // Values ok (0)
  *dev_equal = 0;

  // Compare values between global- and constant
  // memory structures
  if (dev_grid->nd != devC_grid.nd ||
      dev_grid->origo[0] != devC_grid.origo[0] ||
      dev_grid->origo[1] != devC_grid.origo[1] ||
      dev_grid->origo[2] != devC_grid.origo[2] ||
      dev_grid->L[0] != devC_grid.L[0] ||
      dev_grid->L[1] != devC_grid.L[1] ||
      dev_grid->L[2] != devC_grid.L[2] ||
      dev_grid->num[0] != devC_grid.num[0] ||
      dev_grid->num[1] != devC_grid.num[1] ||
      dev_grid->num[2] != devC_grid.num[2])
    *dev_equal = 1; // Not ok

  else if (dev_params->global != devC_params.global ||
      dev_params->g[0] != devC_params.g[0] ||
      dev_params->g[1] != devC_params.g[1] ||
      dev_params->g[2] != devC_params.g[2] ||
      dev_params->dt != devC_params.dt ||
      dev_params->np != devC_params.np ||
      dev_params->nw != devC_params.nw ||
      dev_params->wmode[0] != devC_params.wmode[0] ||
      dev_params->k_n != devC_params.k_n ||
      dev_params->k_t != devC_params.k_t ||
      dev_params->k_r != devC_params.k_r ||
      dev_params->gamma_n != devC_params.gamma_n ||
      dev_params->gamma_t != devC_params.gamma_t ||
      dev_params->gamma_r != devC_params.gamma_r ||
      dev_params->gamma_wn != devC_params.gamma_wn ||
      dev_params->gamma_wt != devC_params.gamma_wt ||
      dev_params->gamma_wr != devC_params.gamma_wr ||
      dev_params->mu_s != devC_params.mu_s ||
      dev_params->mu_d != devC_params.mu_d ||
      dev_params->mu_r != devC_params.mu_r ||
      dev_params->rho != devC_params.rho ||
      dev_params->kappa != devC_params.kappa ||
      dev_params->db != devC_params.db ||
      dev_params->V_b != devC_params.V_b ||
      dev_params->periodic != devC_params.periodic ||
      dev_params->shearmodel != devC_params.shearmodel)
    *dev_equal = 2; // Not ok

}


// Copy the constant data components to device memory,
// and check whether the values correspond to the 
// values in constant memory.
__host__ void DEM::checkConstantMemory()
{

  //cudaPrintfInit();

  // Allocate space in global device memory
  Grid* dev_grid;
  Params* dev_params;
  cudaMalloc((void**)&dev_grid, sizeof(Grid));
  cudaMalloc((void**)&dev_params, sizeof(Params));

  // Copy structure data from host to global device memory
  cudaMemcpy(dev_grid, grid, sizeof(Grid), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_params, params, sizeof(Params), cudaMemcpyHostToDevice);

  // Compare values between global and constant memory
  // structures on the device.
  int* equal = new int;	// The values are equal = 0, if not = 1
  *equal = 0;
  int* dev_equal;
  cudaMalloc((void**)&dev_equal, sizeof(int));
  checkConstantValues<<<1,1>>>(dev_equal, dev_grid, dev_params);
  checkForCudaErrors("After constant memory check");

  // Copy result to host
  cudaMemcpy(equal, dev_equal, sizeof(int), cudaMemcpyDeviceToHost);

  // Free global device memory
  cudaFree(dev_grid);
  cudaFree(dev_params);
  cudaFree(dev_equal);

  //cudaPrintfDisplay(stdout, true);

  // Are the values equal?
  if (*equal != 0) {
    std::cerr << "Error! The values in constant memory do not "
              << "seem to be correct (" << *equal << ").\n";
    exit(1);
  } else {
    std::cout << "  Constant values ok (" << *equal << ").\n";
  }
}

// Copy selected constant components to constant device memory.
__host__ void DEM::transferToConstantDeviceMemory()
{
  using std::cout;

  if (verbose == 1)
    cout << "\n  Transfering data to constant device memory:     ";

  cudaMemcpyToSymbol("devC_nd", &nd, sizeof(nd));
  cudaMemcpyToSymbol("devC_np", &np, sizeof(np));
  cudaMemcpyToSymbol("devC_nc", &NC, sizeof(int));
  cudaMemcpyToSymbol("devC_dt", &time.dt, sizeof(Float));
  cudaMemcpyToSymbol(devC_grid, grid, sizeof(Grid));
  cudaMemcpyToSymbol(devC_params, params, sizeof(Params));
  
  checkForCudaErrors("After transferring to device constant memory");
  
  if (verbose == 1)
    cout << "Done\n";

  //checkConstantMemory();
}


// Allocate device memory for particle variables,
// tied to previously declared pointers in structures
__host__ void DEM::allocateGlobalDeviceMemory(void)
{
  // Particle memory size
  unsigned int memSizeF  = sizeof(Float) * np;
  unsigned int memSizeF4 = sizeof(Float4) * np;

  if (verbose == 1)
    std::cout << "  Allocating device memory:                       ";

  // Particle arrays
  cudaMalloc((void**)&dev_k.x, memSizeF4);
  cudaMalloc((void**)&dev_sort.x_sorted, memSizeF4);
  cudaMalloc((void**)&dev_k.vel, memSizeF4);
  cudaMalloc((void**)&dev_sort.vel_sorted, memSizeF4);
  cudaMalloc((void**)&dev_k.angvel, memSizeF4);
  cudaMalloc((void**)&dev_sort.angvel_sorted, memSizeF4);
  cudaMalloc((void**)&dev_k.acc, memSizeF4);
  k.acc = new Float4[np];
  cudaMalloc((void**)&dev_k.angacc, memSizeF4);
  k.angacc = new Float4[np];
  cudaMalloc((void**)&dev_k.force, memSizeF4);
  cudaMalloc((void**)&dev_k.torque, memSizeF4);
  cudaMalloc((void**)&dev_k.angpos, memSizeF4);
  cudaMalloc((void**)&dev_e.es_dot, memSizeF);
  cudaMalloc((void**)&dev_e.ev_dot, memSizeF);
  cudaMalloc((void**)&dev_e.es, memSizeF);
  cudaMalloc((void**)&dev_e.ev, memSizeF);
  cudaMalloc((void**)&dev_e.p, memSizeF);

  // Cell-related arrays
  cudaMalloc((void**)&dev_sort.gridParticleCellID, sizeof(unsigned int)*np);
  cudaMalloc((void**)&dev_sort.gridParticleIndex, sizeof(unsigned int)*np);
  cudaMalloc((void**)&dev_sort.cellStart, sizeof(unsigned int)*grid.num[0]*grid.num[1]*grid.num[2]);
  cudaMalloc((void**)&dev_sort.cellEnd, sizeof(unsigned int)*grid.num[0]*grid.num[1]*grid.num[2]);

  // Particle contact bookkeeping arrays
  cudaMalloc((void**)&dev_k.contacts, sizeof(unsigned int)*np*NC); // Max NC contacts per particle
  cudaMalloc((void**)&dev_k.distmod, sizeof(Float4)*np*NC);
  cudaMalloc((void**)&dev_k.delta_t, sizeof(Float4)*np*NC);

  // Host contact bookkeeping arrays
  k.contacts = new unsigned int[np*NC];
  // Initialize contacts lists to np
  for (unsigned int i=0; i<(np*NC); ++i)
    k.contacts[i] = np;
  k.distmod = new Float4[np*NC];
  k.delta_t = new Float4[np*NC];

  // Wall arrays
  cudaMalloc((void**)&dev_walls.nx, sizeof(Float4)*walls.nw);
  cudaMalloc((void**)&dev_walls.mvfd, sizeof(Float4)*walls.nw);
  cudaMalloc((void**)&dev_walls.force, sizeof(Float)*walls.nw*np);
  // dev_w_force_partial allocated later

  checkForCudaErrors("End of allocateGlobalDeviceMemory");
  if (verbose == 1)
    std::cout << "Done\n";
}

__host__ void DEM::freeGlobalDeviceMemory()
{
  if (verbose == 1)
    printf("\nLiberating device memory:                        ");
  // Particle arrays
  cudaFree(dev_k.x);
  cudaFree(dev_sort.x_sorted);
  cudaFree(dev_k.vel);
  cudaFree(dev_sort.vel_sorted);
  cudaFree(dev_k.angvel);
  cudaFree(dev_sort.angvel_sorted);
  cudaFree(dev_k.acc);
  cudaFree(dev_k.angacc);
  cudaFree(dev_k.force);
  cudaFree(dev_k.torque);
  cudaFree(dev_k.angpos);
  cudaFree(dev_e.es_dot);
  cudaFree(dev_e.ev_dot);
  cudaFree(dev_e.es);
  cudaFree(dev_e.ev);
  cudaFree(dev_e.p);
  cudaFree(dev_k.contacts);
  cudaFree(dev_k.distmod);
  cudaFree(dev_k.delta_t);

  // Cell-related arrays
  cudaFree(dev_sort.gridParticleIndex);
  cudaFree(dev_sort.cellStart);
  cudaFree(dev_sort.cellEnd);

  // Wall arrays
  cudaFree(dev_walls.nx);
  cudaFree(dev_walls.mvfd);
  cudaFree(dev_walls.force);
  cudaFree(dev_w_force_partial);

  if (verbose == 1)
    printf("Done\n");
}


__host__ void DEM::transferToGlobalDeviceMemory()
{
  if (verbose == 1)
    std::cout << "  Transfering data to the device:                 ";

  // Copy structure data from host to global device memory
  /*cudaMemcpy(dev_k, k, sizeof(Kinematics), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_e, e, sizeof(Energies), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_time, time, sizeof(Time), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_walls, walls, sizeof(Walls), cudaMemcpyHostToDevice);*/
  cudaMemcpy(dev_k, k, sizeof(k), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_e, e, sizeof(e), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_time, time, sizeof(time), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_walls, walls, sizeof(walls), cudaMemcpyHostToDevice);

  checkForCudaErrors("End of transferToGlobalDeviceMemory");
  if (verbose == 1)
    std::cout << "Done\n";
}

__host__ void DEM::transferToGlobalDeviceMemory()
{
  cout << "  Transfering data to the device:                 ";

  // Copy structure data from host to global device memory
  cudaMemcpy(&k, dev_k, sizeof(k), cudaMemcpyDeviceToHost);
  cudaMemcpy(&e, dev_e, sizeof(e), cudaMemcpyDeviceToHost);
  cudaMemcpy(&time, dev_time, sizeof(time), cudaMemcpyDeviceToHost);
  cudaMemcpy(&walls, dev_walls, sizeof(walls), cudaMemcpyDeviceToHost);

  checkForCudaErrors("End of transferFromGlobalDeviceMemory");
  if (verbose == 1)
    std::cout << "Done\n";
}


// Iterate through time by explicit time integration
__host__ void DEM::startTime()
{

  using std::cout; // Namespace directive
  char file[200];  // Output filename
  FILE *fp;

  // Copy data to constant global device memory
  transferToConstantDeviceMemory();


  // Particle memory size
  //unsigned int memSizeF  = sizeof(Float) * np;
  //unsigned int memSizeF4 = sizeof(Float4) * np;

  // Allocate device memory for particle variables,
  // tied to previously declared pointers in structures
  allocateGlobalDeviceMemory();

  // Transfer data from host to gpu device memory
  transferToGlobalDeviceMemory();

  // Synchronization point
  cudaThreadSynchronize();
  checkForCudaErrors("Start of startTime()");

  // Model world variables
  float tic, toc, filetimeclock, time_spent, dev_time_spent;

  // Start CPU clock
  tic = clock();

  // GPU workload configuration
  unsigned int threadsPerBlock = 256; 
  // Create enough blocks to accomodate the particles
  unsigned int blocksPerGrid = iDivUp(np, threadsPerBlock); 
  dim3 dimGrid(blocksPerGrid, 1, 1); // Blocks arranged in 1D grid
  dim3 dimBlock(threadsPerBlock, 1, 1); // Threads arranged in 1D block
  // Shared memory per block
  unsigned int smemSize = sizeof(unsigned int)*(threadsPerBlock+1);

  Float* dev_w_force_partial;
  cudaMalloc((void**)&dev_w_force_partial, sizeof(Float)*dimGrid.x);

  // Report to stdout
  if (verbose == 1) {
    cout << "\n  Device memory allocation and transfer complete.\n"
      << "  - Blocks per grid: "
      << dimGrid.x << "*" << dimGrid.y << "*" << dimGrid.z << "\n"
      << "  - Threads per block: "
      << dimBlock.x << "*" << dimBlock.y << "*" << dimBlock.z << "\n"
      << "  - Shared memory required per block: " << smemSize << " bytes\n";
  }

  // Initialize counter variable values
  filetimeclock = 0.0;
  long iter = 0;

  // Create first status.dat
  sprintf(file,"output/%s.status.dat", inputbin);
  fp = fopen(file, "w");
  fprintf(fp,"%2.4e %2.4e %d\n", 
      	  time.current, 
	  100.0*time.current/time.total, 
	  time.step_count);
  fclose(fp);

  // Write first output data file: output0.bin, thus testing writing of bin files
  sprintf(file,"output/%s.output0.bin", inputbin);
  writebin(file);

  if (verbose == 1) {
    cout << "\n  Entering the main calculation time loop...\n\n"
      << "  IMPORTANT: Do not close this terminal, doing so will \n"
      << "             terminate this SPHERE process. Follow the \n"
      << "             progress by executing:\n"
      << "                $ ./sphere_status " << inputbin << "\n\n";
  }


  // Start GPU clock
  cudaEvent_t dev_tic, dev_toc;
  cudaEventCreate(&dev_tic);
  cudaEventCreate(&dev_toc);
  cudaEventRecord(dev_tic, 0);

  // If profiling is enabled, initialize timers for each kernel
  cudaEvent_t kernel_tic, kernel_toc;
  float kernel_elapsed;
  double t_calcParticleCellID = 0.0;
  double t_thrustsort = 0.0;
  double t_reorderArrays = 0.0;
  double t_topology = 0.0;
  double t_interact = 0.0;
  double t_integrate = 0.0;
  double t_summation = 0.0;
  double t_integrateWalls = 0.0;

  if (PROFILING == 1) {
    cudaEventCreate(&kernel_tic);
    cudaEventCreate(&kernel_toc);
  }

  cout << "  Current simulation time: " << time.current << " s.";


  // MAIN CALCULATION TIME LOOP
  while (time.current <= time.total) {

    // Increment iteration counter
    ++iter;

    // Print current step number to terminal
    //printf("Step: %d\n", time.step_count);


    // Routine check for errors
    checkForCudaErrors("Start of main while loop");


    // For each particle: 
    // Compute hash key (cell index) from position 
    // in the fine, uniform and homogenous grid.
    if (PROFILING == 1)
      startTimer(&kernel_tic);
    calcParticleCellID<<<dimGrid, dimBlock>>>(dev_sort.gridParticleCellID, 
					      dev_sort.gridParticleIndex, 
					      dev_k.x);

    // Synchronization point
    cudaThreadSynchronize();
    if (PROFILING == 1)
      stopTimer(&kernel_tic, &kernel_toc, &kernel_elapsed, &t_calcParticleCellID);
    checkForCudaErrors("Post calcParticleCellID");


    // Sort particle (key, particle ID) pairs by hash key with Thrust radix sort
    if (PROFILING == 1)
      startTimer(&kernel_tic);
    thrust::sort_by_key(thrust::device_ptr<uint>(dev_sort.gridParticleCellID),
			thrust::device_ptr<uint>(dev_sort.gridParticleCellID + np),
			thrust::device_ptr<uint>(dev_sort.gridParticleIndex));
    cudaThreadSynchronize(); // Needed? Does thrust synchronize threads implicitly?
    if (PROFILING == 1)
      stopTimer(&kernel_tic, &kernel_toc, &kernel_elapsed, &t_thrustsort);
    checkForCudaErrors("Post thrust::sort_by_key");


    // Zero cell array values by setting cellStart to its highest possible value,
    // specified with pointer value 0xffffffff, which for a 32 bit unsigned int
    // is 4294967295.
    cudaMemset(dev_sort.cellStart, 0xffffffff, 
	       grid.num[0]*grid.num[1]*grid.num[2]*sizeof(unsigned int));
    cudaThreadSynchronize();
    checkForCudaErrors("Post cudaMemset");

    // Use sorted order to reorder particle arrays (position, velocities, radii) to ensure
    // coherent memory access. Save ordered configurations in new arrays (*_sorted).
    if (PROFILING == 1)
      startTimer(&kernel_tic);
    reorderArrays<<<dimGrid, dimBlock, smemSize>>>(dev_sort.cellStart, 
						   dev_sort.cellEnd,
						   dev_sort.gridParticleCellID, 
						   dev_sort.gridParticleIndex,
						   dev_k.x, dev_k.vel, 
						   dev_k.angvel,
						   dev_sort.x_sorted, 
						   dev_sort.vel_sorted, 
						   dev_sort.angvel_sorted);

    // Synchronization point
    cudaThreadSynchronize();
    if (PROFILING == 1)
      stopTimer(&kernel_tic, &kernel_toc, &kernel_elapsed, &t_reorderArrays);
    checkForCudaErrors("Post reorderArrays", iter);

    // The contact search in topology() is only necessary for determining
    // the accumulated shear distance needed in the linear elastic
    // and nonlinear contact force model
    if (params.contactmodel == 2 || params.contactmodel == 3) {
      // For each particle: Search contacts in neighbor cells
      if (PROFILING == 1)
	startTimer(&kernel_tic);
      topology<<<dimGrid, dimBlock>>>(dev_sort.cellStart, 
				      dev_sort.cellEnd,
				      dev_sort.gridParticleIndex,
				      dev_sort.x_sorted, 
				      dev_k.contacts,
				      dev_k.distmod);


      // Synchronization point
      cudaThreadSynchronize();
      if (PROFILING == 1)
	stopTimer(&kernel_tic, &kernel_toc, &kernel_elapsed, &t_topology);
      checkForCudaErrors("Post topology: One or more particles moved outside the grid.\nThis could possibly be caused by a numerical instability.\nIs the computational time step too large?", iter);
    }


    // For each particle: Process collisions and compute resulting forces.
    if (PROFILING == 1)
      startTimer(&kernel_tic);
    interact<<<dimGrid, dimBlock>>>(dev_sort.gridParticleIndex,
				    dev_sort.cellStart,
				    dev_sort.cellEnd,
				    dev_k.x,
				    dev_sort.x_sorted,
				    dev_sort.vel_sorted,
				    dev_sort.angvel_sorted,
				    dev_k.vel,
				    dev_k.angvel,
				    dev_k.force, 
				    dev_k.torque,
				    dev_e.es_dot,
				    dev_e.ev_dot, 
				    dev_e.es, dev_e.ev, dev_e.p,
				    dev_walls.nx,
				    dev_walls.mvfd,
				    dev_walls.force,
				    dev_k.contacts,
				    dev_k.distmod,
				    dev_k.delta_t);


    // Synchronization point
    cudaThreadSynchronize();
    if (PROFILING == 1)
      stopTimer(&kernel_tic, &kernel_toc, &kernel_elapsed, &t_interact);
    checkForCudaErrors("Post interact - often caused if particles move outside the grid", iter);

    // Update particle kinematics
    if (PROFILING == 1)
      startTimer(&kernel_tic);
    integrate<<<dimGrid, dimBlock>>>(dev_sort.x_sorted, 
				     dev_sort.vel_sorted, 
				     dev_sort.angvel_sorted,
				     dev_k.x, 
				     dev_k.vel, 
				     dev_k.angvel,
				     dev_k.force,
				     dev_k.torque, 
				     dev_k.angpos,
				     dev_sort.gridParticleIndex);

    cudaThreadSynchronize();
    if (PROFILING == 1)
      stopTimer(&kernel_tic, &kernel_toc, &kernel_elapsed, &t_integrate);

    // Summation of forces on wall
    if (PROFILING == 1)
      startTimer(&kernel_tic);
    summation<<<dimGrid, dimBlock>>>(dev_walls.force, dev_w_force_partial);

    // Synchronization point
    cudaThreadSynchronize();
    if (PROFILING == 1)
      stopTimer(&kernel_tic, &kernel_toc, &kernel_elapsed, &t_summation);
    checkForCudaErrors("Post integrate & wall force summation");

    // Update wall kinematics
    if (PROFILING == 1)
      startTimer(&kernel_tic);
    integrateWalls<<< 1, walls.nw>>>(dev_walls.nx, 
				     dev_walls.mvfd,
				     dev_w_force_partial,
				     blocksPerGrid);

    // Synchronization point
    cudaThreadSynchronize();
    if (PROFILING == 1)
      stopTimer(&kernel_tic, &kernel_toc, &kernel_elapsed, &t_integrateWalls);
    checkForCudaErrors("Post integrateWalls");


    // Update timers and counters
    time.current  += time.dt;
    filetimeclock += time.dt;

    // Report time to console
    if (verbose == 1) {
      cout << "\r  Current simulation time: " 
	<< time.current << " s.        ";// << std::flush;
    }


    // Produce output binary if the time interval 
    // between output files has been reached
    if (filetimeclock > time.file_dt) {

      // Pause the CPU thread until all CUDA calls previously issued are completed
      cudaThreadSynchronize();
      checkForCudaErrors("Beginning of file output section");

      //// Copy device data to host memory
      transferFromGlobalDeviceMemory();

      // Pause the CPU thread until all CUDA calls previously issued are completed
      cudaThreadSynchronize();

      // Write binary output file
      time.step_count += 1;
      sprintf(file,"output/%s.output%d.bin", inputbin, time.step_count);
      writebin(file);


      if (CONTACTINFO == 1) {
	// Write contact information to stdout
	/*cout << "\n\n---------------------------\n"
	     << "t = " << time.current << " s.\n"
	     << "---------------------------\n";

	for (int n = 0; n < p.np; ++n) {
	  cout << "\n## Particle " << n << " ##\n";

	  cout  << "- contacts:\n";
	  for (int nc = 0; nc < NC; ++nc) 
	    cout << "[" << nc << "]=" << host_contacts[nc+NC*n] << '\n';

	  cout << "\n- delta_t:\n";
	  for (int nc = 0; nc < NC; ++nc) 
	    cout << host_delta_t[nc+NC*n].x << '\t'
		 << host_delta_t[nc+NC*n].y << '\t'
		 << host_delta_t[nc+NC*n].z << '\t'
		 << host_delta_t[nc+NC*n].w << '\n';

	  cout << "\n- distmod:\n";
	  for (int nc = 0; nc < NC; ++nc) 
	    cout << host_distmod[nc+NC*n].x << '\t'
		 << host_distmod[nc+NC*n].y << '\t'
		 << host_distmod[nc+NC*n].z << '\t'
		 << host_distmod[nc+NC*n].w << '\n';
	}
	cout << '\n';*/
      }

      // Update status.dat at the interval of filetime 
      sprintf(file,"output/%s.status.dat", inputbin);
      fp = fopen(file, "w");
      fprintf(fp,"%2.4e %2.4e %d\n", 
	      time.current, 
	      100.0*time.current/time.total,
	      time.step_count);
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
       << time.step_count << "\n"
       << "  - GPU time spent: "
       << dev_time_spent/1000.0f << " s\n"
       << "  - CPU time spent: "
       << time_spent << " s\n"
       << "  - Mean duration of iteration:\n"
       << "      " << dev_time_spent/((double)iter*1000.0f) << " s\n"; 

  cudaEventDestroy(dev_tic);
  cudaEventDestroy(dev_toc);

  cudaEventDestroy(kernel_tic);
  cudaEventDestroy(kernel_toc);

  // Report time spent on each kernel
  if (PROFILING == 1) {
    double t_sum = t_calcParticleCellID + t_thrustsort + t_reorderArrays
                 + t_topology + t_interact + t_summation + t_integrateWalls;
    cout << "\nKernel profiling statistics:\n"
         << "  - calcParticleCellID:\t" << t_calcParticleCellID/1000.0 << " s"
	 << "\t(" << 100.0*t_calcParticleCellID/t_sum << " %)\n"
         << "  - thrustsort:\t\t" << t_thrustsort/1000.0 << " s"
	 << "\t(" << 100.0*t_thrustsort/t_sum << " %)\n"
         << "  - reorderArrays:\t" << t_reorderArrays/1000.0 << " s"
	 << "\t(" << 100.0*t_reorderArrays/t_sum << " %)\n"
         << "  - topology:\t\t" << t_topology/1000.0 << " s"
	 << "\t(" << 100.0*t_topology/t_sum << " %)\n"
         << "  - interact:\t\t" << t_interact/1000.0 << " s"
	 << "\t(" << 100.0*t_interact/t_sum << " %)\n"
         << "  - integrate:\t\t" << t_integrate/1000.0 << " s"
	 << "\t(" << 100.0*t_integrate/t_sum << " %)\n"
         << "  - summation:\t\t" << t_summation/1000.0 << " s"
	 << "\t(" << 100.0*t_summation/t_sum << " %)\n"
         << "  - integrateWalls:\t" << t_integrateWalls/1000.0 << " s"
	 << "\t(" << 100.0*t_integrateWalls/t_sum << " %)\n";
  }


  // Free GPU device memory  
  freeGlobalDeviceMemory();

  // Contact info arrays
  delete[] k.contacts;
  delete[] k.distmod;
  delete[] k.delta_t;

} /* EOF */
