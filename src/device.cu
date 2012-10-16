// device.cu -- GPU specific operations utilizing the CUDA API.
#include <iostream>
#include <cstdio>
#include <cuda.h>

#include "vector_arithmetic.h"
#include "thrust/device_ptr.h"
#include "thrust/sort.h"

#include "datatypes.h"
#include "datatypes.cuh"
#include "constants.cuh"

#include "sorting.cuh"	
#include "contactmodels.cuh"
#include "cohesion.cuh"
#include "contactsearch.cuh"
#include "integration.cuh"


// Wrapper function for initializing the CUDA components.
// Called from main.cpp
//extern "C"
__host__ void initializeGPU(void)
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
  cudaMemcpyToSymbol("devC_nc", &NC, sizeof(int));
  cudaMemcpyToSymbol("devC_origo", grid->origo, sizeof(Float)*ND);
  cudaMemcpyToSymbol("devC_L", grid->L, sizeof(Float)*ND);
  cudaMemcpyToSymbol("devC_num", grid->num, sizeof(unsigned int)*ND);
  cudaMemcpyToSymbol("devC_dt", &time->dt, sizeof(Float));
  cudaMemcpyToSymbol("devC_global", &params->global, sizeof(int));
  cudaMemcpyToSymbol("devC_g", params->g, sizeof(Float)*ND);
  cudaMemcpyToSymbol("devC_nw", &params->nw, sizeof(unsigned int));
  cudaMemcpyToSymbol("devC_periodic", &params->periodic, sizeof(int));

  if (params->global == 1) {
    // If the physical properties of the particles are global (params.global == true),
    //   copy the values from the first particle into the designated constant memory. 
    //printf("(params.global == %d) ", params.global);
    params->k_n     = p->k_n[0];
    params->k_t	    = p->k_t[0];
    params->k_r	    = p->k_r[0];
    params->gamma_n = p->gamma_n[0];
    params->gamma_t = p->gamma_t[0];
    params->gamma_r = p->gamma_r[0];
    params->mu_s    = p->mu_s[0];
    params->mu_d    = p->mu_d[0];
    params->mu_r    = p->mu_r[0];
    params->rho     = p->rho[0];
    cudaMemcpyToSymbol("devC_k_n", &params->k_n, sizeof(Float));
    cudaMemcpyToSymbol("devC_k_t", &params->k_t, sizeof(Float));
    cudaMemcpyToSymbol("devC_k_r", &params->k_r, sizeof(Float));
    cudaMemcpyToSymbol("devC_gamma_n", &params->gamma_n, sizeof(Float));
    cudaMemcpyToSymbol("devC_gamma_t", &params->gamma_t, sizeof(Float));
    cudaMemcpyToSymbol("devC_gamma_r", &params->gamma_r, sizeof(Float));
    cudaMemcpyToSymbol("devC_gamma_wn", &params->gamma_wn, sizeof(Float));
    cudaMemcpyToSymbol("devC_gamma_ws", &params->gamma_ws, sizeof(Float));
    cudaMemcpyToSymbol("devC_gamma_wr", &params->gamma_wr, sizeof(Float));
    cudaMemcpyToSymbol("devC_mu_s", &params->mu_s, sizeof(Float));
    cudaMemcpyToSymbol("devC_mu_d", &params->mu_d, sizeof(Float));
    cudaMemcpyToSymbol("devC_mu_r", &params->mu_r, sizeof(Float));
    cudaMemcpyToSymbol("devC_rho", &params->rho, sizeof(Float));
    cudaMemcpyToSymbol("devC_kappa", &params->kappa, sizeof(Float));
    cudaMemcpyToSymbol("devC_db", &params->db, sizeof(Float));
    cudaMemcpyToSymbol("devC_V_b", &params->V_b, sizeof(Float));
    cudaMemcpyToSymbol("devC_shearmodel", &params->shearmodel, sizeof(unsigned int));
    cudaMemcpyToSymbol("devC_wmode", &params->wmode, sizeof(int)*MAXWALLS);
  } else {
    //printf("(params.global == %d) ", params.global);
    // Copy params structure with individual physical values from host to global memory
    //Params *dev_params;
    //HANDLE_ERROR(cudaMalloc((void**)&dev_params, sizeof(Params)));
    //HANDLE_ERROR(cudaMemcpyToSymbol(dev_params, &params, sizeof(Params)));
    //printf("Done\n");
    std::cerr << "\n\nError: SPHERE is not yet ready for non-global physical variables.\nBye!\n";
    exit(EXIT_FAILURE); // Return unsuccessful exit status
  }
  checkForCudaErrors("After transferring to device constant memory");

  cout << "Done\n";
}


//extern "C"
__host__ void gpuMain(Float4* host_x,
		      Float4* host_vel,
		      Float4* host_acc,
		      Float4* host_angvel,
		      Float4* host_angacc,
		      Float4* host_force,
		      Float4* host_torque,
		      Float4* host_angpos,
		      uint4*  host_bonds,
		      Particles* p, 
		      Grid* grid, 
		      Time* time, 
		      Params* params,
		      Float4* host_w_nx,
		      Float4* host_w_mvfd,
		      const char* cwd, 
		      const char* inputbin)
{

  using std::cout;	// Namespace directive

  // Copy data to constant global device memory
  transferToConstantMemory(p, grid, time, params);

  // Declare pointers for particle variables on the device
  Float4* dev_x;	// Particle position
  Float4* dev_vel;	// Particle linear velocity
  Float4* dev_angvel;	// Particle angular velocity
  Float4* dev_acc;	// Particle linear acceleration
  Float4* dev_angacc;	// Particle angular acceleration
  Float4* dev_force;	// Sum of forces
  Float4* dev_torque;	// Sum of torques
  Float4* dev_angpos;	// Particle angular position
  Float*  dev_radius;	// Particle radius
  Float*  dev_es_dot;	// Current shear energy producion rate
  Float*  dev_ev_dot;	// Current viscous energy producion rate
  Float*  dev_es;	// Total shear energy excerted on particle
  Float*  dev_ev;	// Total viscous energy excerted on particle
  Float*  dev_p;	// Pressure excerted onto particle
  //uint4*  dev_bonds;	// Particle bond pairs

  // Declare pointers for wall vectors on the device
  Float4* dev_w_nx;            // Wall normal (x,y,z) and position (w)
  Float4* dev_w_mvfd;          // Wall mass (x), velocity (y), force (z) 
  			       // and deviatoric stress (w)
  Float*  dev_w_force;	       // Resulting force on wall per particle
  Float*  dev_w_force_partial; // Partial sum from block of threads

  // Memory for sorted particle data
  Float4* dev_x_sorted;
  Float4* dev_vel_sorted;
  Float4* dev_angvel_sorted;
  Float*  dev_radius_sorted; 
  //uint4*  dev_bonds_sorted;

  // Grid-particle array pointers
  unsigned int* dev_gridParticleCellID;
  unsigned int* dev_gridParticleIndex;
  unsigned int* dev_cellStart;
  unsigned int* dev_cellEnd;

  // Particle contact bookkeeping
  unsigned int* dev_contacts;
  unsigned int* host_contacts = new unsigned int[p->np*NC];
  // Particle pair distance correction across periodic boundaries
  Float4* dev_distmod;
  Float4* host_distmod = new Float4[p->np*NC];
  // x,y,z contains the interparticle vector, corrected if contact 
  // is across a periodic boundary. 
  Float4* dev_delta_t; // Accumulated shear distance of contact
  Float4* host_delta_t = new Float4[p->np*NC];

  // Particle memory size
  unsigned int memSizeF  = sizeof(Float) * p->np;
  unsigned int memSizeF4 = sizeof(Float4) * p->np;

  // Allocate device memory for particle variables,
  // tie to previously declared pointers
  cout << "  Allocating device memory:                       ";

  // Particle arrays
  cudaMalloc((void**)&dev_x, memSizeF4);
  cudaMalloc((void**)&dev_x_sorted, memSizeF4);
  cudaMalloc((void**)&dev_vel, memSizeF4);
  cudaMalloc((void**)&dev_vel_sorted, memSizeF4);
  cudaMalloc((void**)&dev_angvel, memSizeF4);
  cudaMalloc((void**)&dev_angvel_sorted, memSizeF4);
  cudaMalloc((void**)&dev_acc, memSizeF4);
  cudaMalloc((void**)&dev_angacc, memSizeF4);
  cudaMalloc((void**)&dev_force, memSizeF4);
  cudaMalloc((void**)&dev_torque, memSizeF4);
  cudaMalloc((void**)&dev_angpos, memSizeF4);
  cudaMalloc((void**)&dev_radius, memSizeF);
  cudaMalloc((void**)&dev_radius_sorted, memSizeF);
  cudaMalloc((void**)&dev_es_dot, memSizeF);
  cudaMalloc((void**)&dev_ev_dot, memSizeF);
  cudaMalloc((void**)&dev_es, memSizeF);
  cudaMalloc((void**)&dev_ev, memSizeF);
  cudaMalloc((void**)&dev_p, memSizeF);
  //cudaMalloc((void**)&dev_bonds, sizeof(uint4) * p->np);
  //cudaMalloc((void**)&dev_bonds_sorted, sizeof(uint4) * p->np);

  // Cell-related arrays
  cudaMalloc((void**)&dev_gridParticleCellID, sizeof(unsigned int)*p->np);
  cudaMalloc((void**)&dev_gridParticleIndex, sizeof(unsigned int)*p->np);
  cudaMalloc((void**)&dev_cellStart, sizeof(unsigned int)*grid->num[0]*grid->num[1]*grid->num[2]);
  cudaMalloc((void**)&dev_cellEnd, sizeof(unsigned int)*grid->num[0]*grid->num[1]*grid->num[2]);

  // Particle contact bookkeeping arrays
  cudaMalloc((void**)&dev_contacts, sizeof(unsigned int)*p->np*NC); // Max NC contacts per particle
  cudaMalloc((void**)&dev_distmod, sizeof(Float4)*p->np*NC);
  cudaMalloc((void**)&dev_delta_t, sizeof(Float4)*p->np*NC);

  // Wall arrays
  cudaMalloc((void**)&dev_w_nx, sizeof(Float)*params->nw*4);
  cudaMalloc((void**)&dev_w_mvfd, sizeof(Float)*params->nw*4);
  cudaMalloc((void**)&dev_w_force, sizeof(Float)*params->nw*p->np);

  checkForCudaErrors("Post device memory allocation");
  cout << "Done\n";

  // Transfer data from host to gpu device memory
  cout << "  Transfering data to the device:                 ";

  // Particle data
  cudaMemcpy(dev_x, host_x, memSizeF4, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_vel, host_vel, memSizeF4, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_acc, host_acc, memSizeF4, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_angvel, host_angvel, memSizeF4, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_angacc, host_angacc, memSizeF4, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_force, host_force, memSizeF4, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_torque, host_torque, memSizeF4, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_angpos, host_angpos, memSizeF4, cudaMemcpyHostToDevice);
  //cudaMemcpy(dev_bonds, host_bonds, sizeof(uint4) * p->np, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_radius, p->radius, memSizeF, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_es_dot, p->es_dot, memSizeF, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_ev_dot, p->ev_dot, memSizeF, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_es, p->es, memSizeF, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_ev, p->ev, memSizeF, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_p, p->p, memSizeF, cudaMemcpyHostToDevice);

  // Wall data (wall mass and number in constant memory)
  cudaMemcpy(dev_w_nx, host_w_nx, sizeof(Float)*params->nw*4, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_w_mvfd, host_w_mvfd, sizeof(Float)*params->nw*4, cudaMemcpyHostToDevice);
  
  // Initialize contacts lists to p.np
  unsigned int* npu = new unsigned int[p->np*NC];
  for (unsigned int i=0; i<(p->np*NC); ++i)
    npu[i] = p->np;
  cudaMemcpy(dev_contacts, npu, sizeof(unsigned int)*p->np*NC, cudaMemcpyHostToDevice);
  delete[] npu;

  // Create array of 0.0 values on the host and transfer these to the distance 
  // modifier and shear displacement arrays
  Float4* zerosF4 = new Float4[p->np*NC];
  for (unsigned int i=0; i<(p->np*NC); ++i)
    zerosF4[i] = MAKE_FLOAT4(0.0f, 0.0f, 0.0f, 0.0f);
  cudaMemcpy(dev_distmod, zerosF4, sizeof(Float4)*p->np*NC, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_delta_t, zerosF4, sizeof(Float4)*p->np*NC, cudaMemcpyHostToDevice);
  delete[] zerosF4;

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

  cudaMalloc((void**)&dev_w_force_partial, sizeof(Float)*dimGrid.x);

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
		host_torque, host_angpos,
		host_bonds,
		grid, time, params,
		host_w_nx, host_w_mvfd) != 0)  {
    std::cerr << "\n Problem during fwritebin \n";
    exit(EXIT_FAILURE);
  }

  cout << "\n  Entering the main calculation time loop...\n\n"
       << "  IMPORTANT: Do not close this terminal, doing so will \n"
       << "             terminate this SPHERE process. Follow the \n"
       << "             progress by executing:\n"
       << "                $ ./sphere_status " << inputbin << "\n\n";

  // Enable cuPrintf()
  //cudaPrintfInit();

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
    if (PROFILING == 1)
      startTimer(&kernel_tic);
    calcParticleCellID<<<dimGrid, dimBlock>>>(dev_gridParticleCellID, 
					      dev_gridParticleIndex, 
					      dev_x);

    // Synchronization point
    cudaThreadSynchronize();
    if (PROFILING == 1)
      stopTimer(&kernel_tic, &kernel_toc, &kernel_elapsed, &t_calcParticleCellID);
    checkForCudaErrors("Post calcParticleCellID");


    // Sort particle (key, particle ID) pairs by hash key with Thrust radix sort
    if (PROFILING == 1)
      startTimer(&kernel_tic);
    thrust::sort_by_key(thrust::device_ptr<uint>(dev_gridParticleCellID),
			thrust::device_ptr<uint>(dev_gridParticleCellID + p->np),
			thrust::device_ptr<uint>(dev_gridParticleIndex));
    cudaThreadSynchronize(); // Needed? Does thrust synchronize threads implicitly?
    if (PROFILING == 1)
      stopTimer(&kernel_tic, &kernel_toc, &kernel_elapsed, &t_thrustsort);
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
    if (PROFILING == 1)
      startTimer(&kernel_tic);
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
    if (PROFILING == 1)
      stopTimer(&kernel_tic, &kernel_toc, &kernel_elapsed, &t_reorderArrays);
    checkForCudaErrors("Post reorderArrays", iter);

    // The contact search in topology() is only necessary for determining
    // the accumulated shear distance needed in the linear elastic
    // and nonlinear contact force model
    if (params->shearmodel == 2 || params->shearmodel == 3) {
      // For each particle: Search contacts in neighbor cells
      if (PROFILING == 1)
	startTimer(&kernel_tic);
      topology<<<dimGrid, dimBlock>>>(dev_cellStart, 
				      dev_cellEnd,
				      dev_gridParticleIndex,
				      dev_x_sorted, 
				      dev_radius_sorted, 
				      dev_contacts,
				      dev_distmod);

      // Empty cuPrintf() buffer to console
      //cudaThreadSynchronize();
      //cudaPrintfDisplay(stdout, true);

      // Synchronization point
      cudaThreadSynchronize();
      if (PROFILING == 1)
	stopTimer(&kernel_tic, &kernel_toc, &kernel_elapsed, &t_topology);
      checkForCudaErrors("Post topology: One or more particles moved outside the grid.\nThis could possibly be caused by a numerical instability.\nIs the computational time step too large?", iter);
    }


    // For each particle: Process collisions and compute resulting forces.
    if (PROFILING == 1)
      startTimer(&kernel_tic);
    interact<<<dimGrid, dimBlock>>>(dev_gridParticleIndex,
				    dev_cellStart,
				    dev_cellEnd,
				    dev_x, dev_radius,
				    dev_x_sorted, dev_radius_sorted,
				    dev_vel_sorted, dev_angvel_sorted,
				    dev_vel, dev_angvel,
				    dev_force, dev_torque,
				    dev_es_dot, dev_ev_dot, 
				    dev_es, dev_ev, dev_p,
				    dev_w_nx, dev_w_mvfd, dev_w_force,
				    //dev_bonds_sorted,
				    dev_contacts,
				    dev_distmod,
				    dev_delta_t);

    // Empty cuPrintf() buffer to console
    //cudaThreadSynchronize();
    //cudaPrintfDisplay(stdout, true);

    // Synchronization point
    cudaThreadSynchronize();
    if (PROFILING == 1)
      stopTimer(&kernel_tic, &kernel_toc, &kernel_elapsed, &t_interact);
    checkForCudaErrors("Post interact - often caused if particles move outside the grid", iter);

    // Update particle kinematics
    if (PROFILING == 1)
      startTimer(&kernel_tic);
    integrate<<<dimGrid, dimBlock>>>(dev_x_sorted, dev_vel_sorted, 
				     dev_angvel_sorted, dev_radius_sorted,
				     dev_x, dev_vel, dev_angvel,
				     dev_force, dev_torque, dev_angpos,
				     dev_gridParticleIndex);

    cudaThreadSynchronize();
    if (PROFILING == 1)
      stopTimer(&kernel_tic, &kernel_toc, &kernel_elapsed, &t_integrate);

    // Summation of forces on wall
    if (PROFILING == 1)
      startTimer(&kernel_tic);
    summation<<<dimGrid, dimBlock>>>(dev_w_force, dev_w_force_partial);

    // Synchronization point
    cudaThreadSynchronize();
    if (PROFILING == 1)
      stopTimer(&kernel_tic, &kernel_toc, &kernel_elapsed, &t_summation);
    checkForCudaErrors("Post integrate & wall force summation");

    // Update wall kinematics
    if (PROFILING == 1)
      startTimer(&kernel_tic);
    integrateWalls<<< 1, params->nw>>>(dev_w_nx, 
				       dev_w_mvfd,
				       dev_w_force_partial,
				       blocksPerGrid);

    // Synchronization point
    cudaThreadSynchronize();
    if (PROFILING == 1)
      stopTimer(&kernel_tic, &kernel_toc, &kernel_elapsed, &t_integrateWalls);
    checkForCudaErrors("Post integrateWalls");


    // Update timers and counters
    time->current    += time->dt;
    filetimeclock    += time->dt;

    // Report time to console
    cout << "\r  Current simulation time: " 
         << time->current << " s.        ";// << std::flush;


    // Produce output binary if the time interval 
    // between output files has been reached
    if (filetimeclock > time->file_dt) {

      // Pause the CPU thread until all CUDA calls previously issued are completed
      cudaThreadSynchronize();
      checkForCudaErrors("Beginning of file output section");

      //// Copy device data to host memory

      // Particle data
      cudaMemcpy(host_x, dev_x, memSizeF4, cudaMemcpyDeviceToHost);
      cudaMemcpy(host_vel, dev_vel, memSizeF4, cudaMemcpyDeviceToHost);
      cudaMemcpy(host_acc, dev_acc, memSizeF4, cudaMemcpyDeviceToHost);
      cudaMemcpy(host_angvel, dev_angvel, memSizeF4, cudaMemcpyDeviceToHost);
      cudaMemcpy(host_force, dev_force, memSizeF4, cudaMemcpyDeviceToHost);
      cudaMemcpy(host_torque, dev_torque, memSizeF4, cudaMemcpyDeviceToHost);
      cudaMemcpy(host_angpos, dev_angpos, memSizeF4, cudaMemcpyDeviceToHost);
      //cudaMemcpy(host_bonds, dev_bonds, sizeof(uint4) * p->np, cudaMemcpyDeviceToHost);
      cudaMemcpy(p->es_dot, dev_es_dot, memSizeF, cudaMemcpyDeviceToHost);
      cudaMemcpy(p->ev_dot, dev_ev_dot, memSizeF, cudaMemcpyDeviceToHost);
      cudaMemcpy(p->es, dev_es, memSizeF, cudaMemcpyDeviceToHost);
      cudaMemcpy(p->ev, dev_ev, memSizeF, cudaMemcpyDeviceToHost);
      cudaMemcpy(p->p, dev_p, memSizeF, cudaMemcpyDeviceToHost);

      // Wall data
      cudaMemcpy(host_w_nx, dev_w_nx, 
	  	 sizeof(Float)*params->nw*4, cudaMemcpyDeviceToHost);
      cudaMemcpy(host_w_mvfd, dev_w_mvfd, 
	  	 sizeof(Float)*params->nw*4, cudaMemcpyDeviceToHost);

  
      // Contact information
      if (CONTACTINFO == 1) {
	cudaMemcpy(host_contacts, dev_contacts, sizeof(unsigned int)*p->np*NC, cudaMemcpyDeviceToHost);
	cudaMemcpy(host_delta_t, dev_delta_t, memSizeF4*NC, cudaMemcpyDeviceToHost);
	cudaMemcpy(host_distmod, dev_distmod, memSizeF4*NC, cudaMemcpyDeviceToHost);
      }

      // Pause the CPU thread until all CUDA calls previously issued are completed
      cudaThreadSynchronize();

      // Write binary output file
      time->step_count += 1;
      sprintf(file,"%s/output/%s.output%d.bin", cwd, inputbin, time->step_count);

      if (fwritebin(file, p, host_x, host_vel, 
	    	    host_angvel, host_force, 
		    host_torque, host_angpos,
		    host_bonds,
		    grid, time, params,
	    	    host_w_nx, host_w_mvfd) != 0) {
	cout << "\n Error during fwritebin() in main loop\n";
	exit(EXIT_FAILURE);
      }

      if (CONTACTINFO == 1) {
	// Write contact information to stdout
	cout << "\n\n---------------------------\n"
	     << "t = " << time->current << " s.\n"
	     << "---------------------------\n";

	for (int n = 0; n < p->np; ++n) {
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
	cout << '\n';
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

  cudaEventDestroy(kernel_tic);
  cudaEventDestroy(kernel_toc);

  // Report time spent on each kernel
  if (PROFILING == 1) {
    double t_sum = t_calcParticleCellID + t_thrustsort + t_reorderArrays
                 + t_topology + t_interact + t_summation + t_integrateWalls;
    cout << "\nKernel profiling statistics:\n"
         << "  - calcParticleCellID:\t" << t_calcParticleCellID/1000.0 << " s"
	 << " (" << 100.0*t_calcParticleCellID/t_sum << " %)\n"
         << "  - thrustsort:\t\t" << t_thrustsort/1000.0 << " s"
	 << " (" << 100.0*t_thrustsort/t_sum << " %)\n"
         << "  - reorderArrays:\t" << t_reorderArrays/1000.0 << " s"
	 << " (" << 100.0*t_reorderArrays/t_sum << " %)\n"
         << "  - topology:\t\t" << t_topology/1000.0 << " s"
	 << " (" << 100.0*t_topology/t_sum << " %)\n"
         << "  - interact:\t\t" << t_interact/1000.0 << " s"
	 << " (" << 100.0*t_interact/t_sum << " %)\n"
         << "  - integrate:\t\t" << t_integrate/1000.0 << " s"
	 << " (" << 100.0*t_integrate/t_sum << " %)\n"
         << "  - summation:\t\t" << t_summation/1000.0 << " s"
	 << " (" << 100.0*t_summation/t_sum << " %)\n"
         << "  - integrateWalls:\t" << t_integrateWalls/1000.0 << " s"
	 << " (" << 100.0*t_integrateWalls/t_sum << " %)\n";
  }


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
  cudaFree(dev_angpos);
  cudaFree(dev_radius);
  cudaFree(dev_radius_sorted);
  cudaFree(dev_es_dot);
  cudaFree(dev_ev_dot);
  cudaFree(dev_es);
  cudaFree(dev_ev);
  cudaFree(dev_p);
  //cudaFree(dev_bonds);
  //cudaFree(dev_bonds_sorted);
  cudaFree(dev_contacts);
  cudaFree(dev_distmod);
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

  // Contact info arrays
  delete[] host_contacts;
  delete[] host_distmod;
  delete[] host_delta_t;

  printf("Done\n");
} /* EOF */
