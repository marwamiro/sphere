/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*  SPHERE source code by Anders Damsgaard Christensen, 2010-12,       */
/*  a 3D Discrete Element Method algorithm with CUDA GPU acceleration. */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
// Licence: Gnu Public License (GPL) v. 3. See license.txt.
// See doc/sphere-doc.pdf for full documentation.
// Compile with GNU make by typing 'make' in the src/ directory.               
// SPHERE is called from the command line with './sphere_<architecture> projectname' 


// Including library files
#include <iostream>
#include <cstdio>  // Standard library functions for file input and output
#include <stdlib.h> // Functions involving memory allocation, process control, conversions and others
#include <unistd.h> // UNIX only: For getcwd
#include <string.h> // For strerror and strcmp
#include <errno.h>  // For errno
#include <math.h>   // Basic Floating point mathematical operations
#include <time.h>   // Time and date functions for timer

// Including user files
#include "datatypes.h"


//////////////////
// MAIN ROUTINE //
//////////////////
// The main loop returns the value 0 to the shell, if the program terminated
// successfully, and 1 if an error occured which caused the program to crash.
int main(int argc, char *argv[]) 
{
  // Namespace declarations
  using std::cout;
  using std::cerr;

  // LOCAL VARIABLE DECLARATIONS
  if(!argv[1] || argc != 2) {
    cout << "Error: Specify simulation name from the input/ directory (input binary file), e.g. ./sphere input\n";
    return 1; // Return unsuccessful exit status
  }

  unsigned int i,j; // Counter variables
  char file[1000];  // Complete path+filename variable
  FILE *fp; 	    // Declare file pointer

  // Host particle variable structure
  Particles p;	    

  // Host grid structure
  Grid grid;

  // Host time structure
  Time time;

  // Host physical global constants structure
  Params params;

  // Read path to current working directory
  char *cwd;
  cwd = getcwd(0, 0);
  if (!cwd) {	// Terminate program execution if path is not obtained
    cerr << "Error: getcwd failed: " << strerror(errno) << '\n';
    return 1; // Return unsuccessful exit status
  }

  char *inputbin = argv[1]; // Input binary file read from command line argument

  // Opening cerenomy with fancy ASCII art
  cout << ".-------------------------------------.\n"
       << "|              _    Compiled for " << ND << "D   |\n" 
       << "|             | |                     |\n" 
       << "|    ___ _ __ | |__   ___ _ __ ___    |\n"
       << "|   / __| '_ \\| '_ \\ / _ \\ '__/ _ \\   |\n"
       << "|   \\__ \\ |_) | | | |  __/ | |  __/   |\n"
       << "|   |___/ .__/|_| |_|\\___|_|  \\___|   |\n"
       << "|       | |                           |\n"
       << "|       |_|           Version: " << VERS << "   |\n"           
       << "`-------------------------------------´\n"
       << " Simulation ID:\n"
       << " " << inputbin << "\n"
       << " -------------------------------------\n";
  
  initializeGPU();

  // Import binary data
  printf("Importing initial data from input/%s.bin:\n",inputbin);

  sprintf(file,"%s/input/%s.bin", cwd, inputbin);
  if ((fp = fopen(file,"rb")) == NULL) {
    cout << "Could not read input binary file " << cwd << "/input/" 
         << inputbin << ".bin. Bye.\n";
    exit(1); // Return unsuccessful exit status
  }

  // Read the number of dimensions and particles
  if (fread(&grid.nd, sizeof(grid.nd), 1, fp) != 1)
    exit(1); // Return unsuccessful exit status
  if (fread(&p.np, sizeof(p.np), 1, fp) != 1)
    exit(1); // Return unsuccessful exit status
  cout << "  - Number of dimensions: grid.nd         = " << grid.nd << "\n"
       << "  - Number of particles:  p.np            = " << p.np << "\n";

  if (grid.nd != ND) {
    cout << "Dimensionality mismatch between dataset and this SPHERE program.\n"
         << "The dataset is " << grid.nd 
	 << "D, this SPHERE binary is " << ND << "D.\n"
	 << "This execution is terminating.\n";
    exit(1); // Return unsuccessful exit status
  }

  // Report precision choice
  cout << "  - Compiled for ";
  if (sizeof(Float) == sizeof(float))
    cout << "single";
  else if (sizeof(Float) == sizeof(double))
    cout << "double";
  else {
    cerr << "Error! Chosen precision not available. Check datatypes.h\n";
    exit(1);
  }
  cout << " precision\n";

  // Read time parameters
  if (fread(&time.dt, sizeof(time.dt), 1, fp) != 1)
    exit(1); // Return unsuccessful exit status
  if (fread(&time.current, sizeof(time.current), 1, fp) != 1)
    exit(1); 
  if (fread(&time.total, sizeof(time.total), 1, fp) != 1)
    exit(1); 
  if (fread(&time.file_dt, sizeof(time.file_dt), 1, fp) != 1)
    exit(1); 
  if (fread(&time.step_count, sizeof(time.step_count), 1, fp) != 1)
    exit(1); 

  // Copy timestep length to constant memory structure
  params.dt = time.dt;

  // Copy number of particles to constant memory structure
  params.np = p.np;

  // Output display parameters to screen
  cout << "  - Timestep length:      time.dt         = " 
       << time.dt << " s\n"
       << "  - Start at time:        time.current    = " 
       << time.current << " s\n"
       << "  - Total sim. time:      time.total      = " 
       << time.total << " s\n"
       << "  - File output interval: time.file_dt    = " 
       << time.file_dt << " s\n"
       << "  - Start at step count:  time.step_count = " 
       << time.step_count << "\n";


  // For spatial vectors an array of Float4 vectors is chosen for best fit with 
  // GPU memory handling. Vector variable structure: ( x, y, z, <empty>).
  // Indexing starts from 0.

  // Allocate host arrays
  cout << "\n  Allocating host memory:                         ";
  //grid.origo   = new Float[ND];        // Coordinate system origo
  //grid.L       = new Float[ND];        // Model world dimensions
  //grid.num     = new unsigned int[ND]; // Number of cells in each dimension
  //params.g     = new Float[ND];	       // Gravitational acceleration vector
  p.radius     = new Float[p.np];      // Particle radii
  p.rho        = new Float[p.np];      // Particle densities
  p.m          = new Float[p.np];      // Particle masses
  p.I          = new Float[p.np];      // Particle moment of inertia
  p.k_n        = new Float[p.np];      // Particle normal stiffnesses
  p.k_t	       = new Float[p.np];      // Particle shear stiffnesses
  p.k_r	       = new Float[p.np];      // Particle rolling stiffnesses
  p.gamma_n    = new Float[p.np];      // Particle normal viscosity
  p.gamma_t    = new Float[p.np];      // Particle shear viscosity
  p.gamma_r    = new Float[p.np];      // Particle rolling viscosity
  p.mu_s       = new Float[p.np];      // Inter-particle static shear contact friction coefficients
  p.mu_d       = new Float[p.np];      // Inter-particle dynamic shear contact friction coefficients
  p.mu_r       = new Float[p.np];      // Inter-particle rolling contact friction coefficients
  p.es_dot     = new Float[p.np];      // Rate of shear energy dissipation
  p.ev_dot     = new Float[p.np];      // Rate of viscous energy dissipation
  p.es         = new Float[p.np];      // Total shear energy dissipation
  p.ev         = new Float[p.np];      // Total viscous energy dissipation
  p.p	       = new Float[p.np];      // Pressure excerted onto particle
  //params.wmode = new int[MAXWALLS];    // Wall BC's, 0: fixed, 1: devs, 2: vel

  // Allocate Float4 host arrays
  Float4 *host_x      = new Float4[p.np];  // Center coordinates for each particle (x)
  Float4 *host_vel    = new Float4[p.np];  // Particle velocities (dotx = v)
  Float4 *host_acc    = new Float4[p.np];  // Particle accellerations (dotdotx = a)
  Float4 *host_angvel = new Float4[p.np];  // Particle angular velocity vector (omega)
  Float4 *host_angacc = new Float4[p.np];  // Particle angular acceleration vector (dotomega)
  Float4 *host_force  = new Float4[p.np];  // Particle summed force
  Float4 *host_torque = new Float4[p.np];  // Particle summed torque
  Float4 *host_angpos = new Float4[p.np];  // Particle angular position


  uint4  *host_bonds  = new uint4[p.np];   // Bonds from particle [i] to two particles
  cout << "Done\n";

  cout << "  Reading remaining data from input binary:       ";
  // Read remaining data from input binary
  for (i=0; i<ND; ++i) {
    if (fread(&grid.origo[i], sizeof(grid.origo[i]), 1, fp) != 1)
      exit(1); // Return unsuccessful exit status
  }

  for (i=0; i<ND; ++i) {
    if (fread(&grid.L[i], sizeof(grid.L[i]), 1, fp) != 1)
      exit(1); // Return unsuccessful exit status
  }

  for (i=0; i<ND; ++i) {
    if (fread(&grid.num[i], sizeof(grid.num[i]), 1, fp) != 1)
      exit(1); // Return unsuccessful exit status
  }

  for (j=0; j<p.np; ++j) {
    if (fread(&host_x[j].x, sizeof(Float), 1, fp) != 1)
      exit(1); // Return unsuccessful exit status
    if (fread(&host_vel[j].x, sizeof(Float), 1, fp) != 1)
      exit(1);
    if (fread(&host_angvel[j].x, sizeof(Float), 1, fp) != 1)
      exit(1);
    if (fread(&host_force[j].x, sizeof(Float), 1, fp) != 1)
      exit(1);
    if (fread(&host_torque[j].x, sizeof(Float), 1, fp) != 1)
      exit(1);
    if (fread(&host_angpos[j].x, sizeof(Float), 1, fp) != 1)
      exit(1);


    if (fread(&host_x[j].y, sizeof(Float), 1, fp) != 1)
      exit(1); // Return unsuccessful exit status
    if (fread(&host_vel[j].y, sizeof(Float), 1, fp) != 1)
      exit(1);
    if (fread(&host_angvel[j].y, sizeof(Float), 1, fp) != 1)
      exit(1);
    if (fread(&host_force[j].y, sizeof(Float), 1, fp) != 1)
      exit(1);
    if (fread(&host_torque[j].y, sizeof(Float), 1, fp) != 1)
      exit(1);
    if (fread(&host_angpos[j].y, sizeof(Float), 1, fp) != 1)
      exit(1);

    if (fread(&host_x[j].z, sizeof(Float), 1, fp) != 1)
      exit(1); // Return unsuccessful exit status
    if (fread(&host_vel[j].z, sizeof(Float), 1, fp) != 1)
      exit(1);
    if (fread(&host_angvel[j].z, sizeof(Float), 1, fp) != 1)
      exit(1);
    if (fread(&host_force[j].z, sizeof(Float), 1, fp) != 1)
      exit(1);
    if (fread(&host_torque[j].z, sizeof(Float), 1, fp) != 1)
      exit(1);
    if (fread(&host_angpos[j].z, sizeof(Float), 1, fp) != 1)
      exit(1);
  }

  for (j=0; j<p.np; ++j) {
    if (fread(&host_vel[j].w, sizeof(Float), 1, fp) != 1) // Fixvel
      exit(1); // Return unsuccessful exit status
    if (fread(&host_x[j].w, sizeof(Float), 1, fp) != 1) // xsum
      exit(1);
    if (fread(&p.radius[j], sizeof(p.radius[j]), 1, fp) != 1)
      exit(1);
    if (fread(&p.rho[j], sizeof(p.rho[j]), 1, fp) != 1)
      exit(1);
    if (fread(&p.k_n[j], sizeof(p.k_n[j]), 1, fp) != 1)
      exit(1);
    if (fread(&p.k_t[j], sizeof(p.k_t[j]), 1, fp) != 1)
      exit(1);
    if (fread(&p.k_r[j], sizeof(p.k_r[j]), 1, fp) != 1)
      exit(1);
    if (fread(&p.gamma_n[j], sizeof(p.gamma_n[j]), 1, fp) != 1)
      exit(1);
    if (fread(&p.gamma_t[j], sizeof(p.gamma_t[j]), 1, fp) != 1)
      exit(1);
    if (fread(&p.gamma_r[j], sizeof(p.gamma_r[j]), 1, fp) != 1)
      exit(1);
    if (fread(&p.mu_s[j], sizeof(p.mu_s[j]), 1, fp) != 1)
      exit(1);
    if (fread(&p.mu_d[j], sizeof(p.mu_d[j]), 1, fp) != 1)
      exit(1);
    if (fread(&p.mu_r[j], sizeof(p.mu_r[j]), 1, fp) != 1)
      exit(1);
    if (fread(&p.es_dot[j], sizeof(p.es_dot[j]), 1, fp) != 1)
      exit(1);
    if (fread(&p.ev_dot[j], sizeof(p.ev_dot[j]), 1, fp) != 1)
      exit(1);
    if (fread(&p.es[j], sizeof(p.es[j]), 1, fp) != 1)
      exit(1);
    if (fread(&p.ev[j], sizeof(p.ev[j]), 1, fp) != 1)
      exit(1);
    if (fread(&p.p[j], sizeof(p.p[j]), 1, fp) != 1)
      exit(1);
  }

  if (fread(&params.global, sizeof(params.global), 1, fp) != 1)
    exit(1); // Return unsuccessful exit status
  for (i=0; i<ND; ++i) {
    if (fread(&params.g[i], sizeof(params.g[i]), 1, fp) != 1)
      exit(1);
  }

  if (fread(&params.kappa, sizeof(params.kappa), 1, fp) != 1)
    exit(1);
  if (fread(&params.db, sizeof(params.db), 1, fp) != 1)
    exit(1); 
  if (fread(&params.V_b, sizeof(params.V_b), 1, fp) != 1)
    exit(1); 
  if (fread(&params.shearmodel, sizeof(params.shearmodel), 1, fp) != 1)
    exit(1);


  // Number of dynamic walls
  if (fread(&params.nw, sizeof(params.nw), 1, fp) != 1)
    exit(1); 

  if (params.nw > MAXWALLS) {
    cerr << "Error; MAXWALLS (" << MAXWALLS << ") in datatypes.h "
         << "is smaller than the number of walls specified in the "
	 << "input file (" << params.nw << ").\n";
  }

  // Allocate host memory for walls
  // Wall normal (x,y,z), w: wall position on axis parallel to wall normal
  // Wall mass (x), velocity (y), force (z), and deviatoric stress (w)
  Float4 *host_w_nx   = new Float4[params.nw];
  Float4 *host_w_mvfd = new Float4[params.nw]; 

  // Read wall data
  for (j=0; j<params.nw; ++j) {
    // Wall condition mode: 0: fixed, 1: devs, 2: vel
    if (fread(&params.wmode[j], sizeof(params.wmode[j]), 1, fp) != 1)
      exit(1);

    // Wall normal, x-dimension
    if (fread(&host_w_nx[j].x, sizeof(Float), 1, fp) != 1)
      exit(1);
    // Wall normal, y-dimension
    if (fread(&host_w_nx[j].y, sizeof(Float), 1, fp) != 1)
      exit(1);
    // Wall normal, z-dimension
    if (fread(&host_w_nx[j].z, sizeof(Float), 1, fp) != 1)
      exit(1);

    // Wall position along axis parallel to wall normal
    if (fread(&host_w_nx[j].w, sizeof(Float), 1, fp) != 1)
      exit(1);

    // Wall mass
    if (fread(&host_w_mvfd[j].x, sizeof(Float), 1, fp) != 1)
      exit(1);

    // Wall velocity along axis parallel to wall normal
    if (fread(&host_w_mvfd[j].y, sizeof(Float), 1, fp) != 1)
      exit(1);

    // Wall force along axis parallel to wall normal
    if (fread(&host_w_mvfd[j].z, sizeof(Float), 1, fp) != 1)
      exit(1);

    // Wall deviatoric stress
    if (fread(&host_w_mvfd[j].w, sizeof(Float), 1, fp) != 1)
      exit(1);
  }
  // x- and y boundary behavior.
  // 0: walls, 1: periodic boundaries, 2: x boundary periodic, y frictional walls
  if (fread(&params.periodic, sizeof(params.periodic), 1, fp) != 1) {
    cout << "  - x- and y boundaries: Behavior not set, assuming frictional walls.";
    params.periodic = 0;
  }

  // Wall viscosities
  if (fread(&params.gamma_wn, sizeof(params.gamma_wn), 1, fp) != 1)
    exit(1);
  if (fread(&params.gamma_wt, sizeof(params.gamma_wt), 1, fp) != 1)
    exit(1);
  if (fread(&params.gamma_wr, sizeof(params.gamma_wr), 1, fp) != 1)
    exit(1);


  for (i=0; i<p.np; ++i) {
    if (fread(&host_bonds[i].x, sizeof(unsigned int), 1, fp) != 1)
      exit(1);
    if (fread(&host_bonds[i].y, sizeof(unsigned int), 1, fp) != 1)
      exit(1);
    if (fread(&host_bonds[i].z, sizeof(unsigned int), 1, fp) != 1)
      exit(1);
    if (fread(&host_bonds[i].w, sizeof(unsigned int), 1, fp) != 1)
      exit(1);
  }

  fclose(fp);

  cout << "Done\n";

  if (params.shearmodel == 1)
    cout << "  - Shear force model: Viscous, fricional\n";
  else if (params.shearmodel == 2)
    cout << "  - Shear force model: Linear elastic, viscous, frictional\n";
  else if (params.shearmodel == 3)
    cout << "  - Shear force model: Nonlinear (Hertzian) elastic, viscous, frictional\n";
  else {
    cerr << "Error: Shear model value not understood.\n";
    exit(1);
  }

  cout << "  - Number of dynamic walls: " << params.nw << "\n";

  if (params.periodic == 1)
    cout << "  - x- and y boundaries: Periodic\n";
  else if (params.periodic == 2)
    cout << "  - x boundaries: Periodic. y boundaries: Frictional walls\n";
  else 
    cout << "  - x- and y boundaries: Frictional walls\n";

  cout << "  - Top BC: ";
  if (params.wmode[0] == 0)
    cout << "Fixed\n";
  else if (params.wmode[0] == 1)
    cout << "Deviatoric stress\n";
  else if (params.wmode[0] == 2)
    cout << "Velocity\n";
  else {
    cerr << "Top boundary condition not recognized!\n";
    exit(1);
  }

  if (grid.nd == 1) {
    cout << "  - Grid: " 
	 << grid.num[0] << " cells (x)\n";
  } else if (grid.nd == 2) {
    cout << "  - Grid: " 
	 << grid.num[0] << " (x) * " 
	 << grid.num[1] << " (y) = " 
	 << grid.num[0]*grid.num[1] << " cells\n"; 
  } else if (grid.nd == 3) {
    cout << "  - Grid: " 
	 << grid.num[0] << " (x) * " 
	 << grid.num[1] << " (y) * " 
	 << grid.num[2] << " (z) = " 
	 << grid.num[0]*grid.num[1]*grid.num[2] << " cells\n";
  } else {
    cerr << "\nError; the number of dimensions must be 1, 2 or 3, and was\n"
	 << "defined as " << grid.nd << ". This SPHERE binary was compiled for " 
	 << ND << "D. Bye!\n";
    exit(1); // Return unsuccessful exit status
  }

  cout << "\nEntering the CUDA environment...\n";
  gpuMain(host_x,
	  host_vel,
	  host_acc,
	  host_angvel,
	  host_angacc,
	  host_force,
	  host_torque,
	  host_angpos,
	  host_bonds,
	  p, grid, 
	  time, params,
	  host_w_nx,
	  host_w_mvfd,
	  cwd, inputbin);


  // Free host memory. Delete pointers:
  printf("Liberating host memory:                          ");

  // Particle vectors
  delete[] host_x;
  delete[] host_vel;
  delete[] host_angvel;
  delete[] host_acc;
  delete[] host_angacc;
  delete[] host_force;
  delete[] host_torque;
  delete[] host_angpos;

  // Particle bonds
  delete[] host_bonds;

  // Particle single-value parameters
  //delete[] grid.origo;
  //delete[] grid.L;
  //delete[] grid.num;
  //delete[] params.g;
  delete[] p.radius;
  delete[] p.k_n;
  delete[] p.k_t;
  delete[] p.k_r;
  delete[] p.gamma_n;
  delete[] p.gamma_t;
  delete[] p.gamma_r;
  delete[] p.mu_s;
  delete[] p.mu_d;
  delete[] p.mu_r;
  delete[] p.rho;
  delete[] p.es_dot;
  delete[] p.ev_dot;
  delete[] p.es;
  delete[] p.ev;
  delete[] p.p;

  // Wall arrays
  delete[] host_w_nx;
  delete[] host_w_mvfd;

  // Free other dynamic host memory
  free(cwd);
  printf("Done\n");

  // Terminate execution
  printf("\nBye!\n");
  return 0; // Return successfull exit status
} 
// END OF FILE
