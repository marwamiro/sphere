#include <iostream>
#include <string>
#include <cstdio>
#include <cstdlib>

#include "typedefs.h"
#include "datatypes.h"
#include "constants.h"
#include "sphere.h"

// Constructor: Reads an input binary, and optionally checks
// and reports the values
DEM::DEM(const std::string inputbin, 
    const int verbosity,
    const int checkVals,
    const int render_img)
: verbose(verbosity)
{
  using std::cout;
  using std::cerr;

  // Extract sid from input binary filename 
  size_t dotpos = inputbin.rfind('.');
  size_t slashpos = inputbin.rfind('/');
  if (slashpos - dotpos < 1) {
    std::cerr << "Error! Unable to extract simulation id "
      << "from input file name.\n";
  }
  sid = inputbin.substr(slashpos+1, dotpos-slashpos-1);

  // Read target input binary
  readbin(inputbin.c_str());

  // Check numeric values of chosen parameters
  if (checkVals == 1)
    checkValues();

  // Report data values
  if (verbose == 1)
    reportValues();
    
  // Write initial data to output/<sid>.output0.bin
  writebin(("output/" + sid + ".output0.bin").c_str());

  // Initialize CUDA
  initializeGPU();

  // Render image using raytracer if requested
  if (render_img == 1) {
    float3 eye = make_float3(0.6f * grid.L[0],
			     -2.5f * grid.L[1],
			     0.52f * grid.L[2]);
    //float focalLength = 0.8f*grid.L[0];
    render(eye);
  }

}

// Destructor: Liberates dynamically allocated host memory
DEM::~DEM(void)
{
  delete[] k.x;
  delete[] k.xysum;
  delete[] k.vel;
  delete[] k.force;
  delete[] k.angpos;
  delete[] k.angvel;
  delete[] k.torque;
  delete[] e.es_dot;
  delete[] e.es;
  delete[] e.ev_dot;
  delete[] e.ev;
  delete[] e.p;
  delete[] walls.nx;
  delete[] walls.mvfd;
}


// Check numeric values of selected parameters
void DEM::checkValues(void)
{
  using std::cerr;

  unsigned int i;

  // Check the number of dimensions
  if (nd != ND) {
    cerr << "Error: nd = " << nd << ", ND = " << ND << '\n';
    exit(1);
  }

  // Check the number of possible contacts
  if (NC < 1) {
    cerr << "Error: NC = " << NC << '\n';
    exit(1);
  } else if (NC < 8) {
    cerr << "Warning: NC has a low value (" << NC << "). "
     << "Consider increasing it in 'constants.h'\n";
  }

  // Check that we have a positive number of particles
  if (np < 1) {
    cerr << "Error: np = " << np << '\n';
    exit(1);
  }

  // Check that the current time
  if (time.current < time.total || time.current < 0.0) {
    cerr << "Error: time.current = " << time.current
      << " s, time.total = " << time.total << " s\n";
    exit(1);
  }

  // Check world size
  if (grid.L[0] <= 0.0 || grid.L[1] <= 0.0 || grid.L[2] <= 0.0) {
    cerr << "Error: grid.L[0] = " << grid.L[0] << " m, "
      << "grid.L[1] = " << grid.L[1] << " m, "
      << "grid.L[2] = " << grid.L[2] << " m.\n";
    exit(1);
  }
   
  // Check grid size
  if (grid.num[0] <= 0 || grid.num[1] <= 0 || grid.num[2] <= 0) {
    cerr << "Error: grid.num[0] = " << grid.num[0] << ", "
      << "grid.num[1] = " << grid.num[1] << ", "
      << "grid.num[2] = " << grid.num[2] << ".\n";
    exit(1);
  }

  // Check grid size again
  if (grid.periodic == 2 && grid.num[0] < 3) {
    cerr << "Error: When 1st dimension boundaries are periodic, "
      << "there must be at least 3 cells in that dimension.";
    exit(1);
  }

  if (grid.periodic == 1 && (grid.num[0] < 3 || grid.num[1] < 3)) {
    cerr << "Error: When 1st and 2nd dimension boundaries are periodic, "
      << "there must be at least 3 cells in each of those dimensions.";
    exit(1);
  }

  // Check that radii are positive values
  for (i = 0; i < np; ++i) {
    if (k.x[i].w <= 0.0) {
      cerr << "Error: Particle " << i << " has a radius of "
	<< k.x[i].w << " m.";
      exit(1);
    }
  }

  // Check constant, global parameters
  if (params.k_n <= 0.0) {
    cerr << "Error: k_n = " << params.k_n << " N/m\n";
    exit(1);
  }
  
  if (params.rho <= 0.0) {
    cerr << "Error: rho = " << params.rho << " kg/m3\n";
    exit(1);
  }
}

// Report key parameter values to stdout
void DEM::reportValues()
{
  using std::cout;
  using std::cerr;
  using std::endl;

  cout << "  - Number of dimensions: nd = " << nd << "\n"
    << "  - Number of particles:  np = " << np << "\n";

  // Check precision choice
  cout << "  - Compiled for ";
  if (sizeof(Float) == sizeof(float)) {
    if (verbose == 1)
      cout << "single";
  } else if (sizeof(Float) == sizeof(double)) {
    if (verbose == 1)
      cout << "double";
  } else {
    cerr << "Error! Chosen precision not available. Check datatypes.h\n";
    exit(1);
  }
  cout << " precision\n";

  cout << "  - Timestep length:      time.dt         = " 
    << time.dt << " s\n"
    << "  - Start at time:        time.current    = " 
    << time.current << " s\n"
    << "  - Total sim. time:      time.total      = " 
    << time.total << " s\n"
    << "  - File output interval: time.file_dt    = " 
    << time.file_dt << " s\n"
    << "  - Start at step count:  time.step_count = " 
    << time.step_count << endl;

  if (params.contactmodel == 1)
    cout << "  - Contact model: Linear-elastic-viscous (n), visco-frictional (t)\n";
  else if (params.contactmodel == 2)
    cout << "  - Contact model: Linear-elastic-visco-frictional\n";
  else if (params.contactmodel == 3)
    cout << "  - Contact model: Nonlinear-elastic-visco-frictional\n";
  else {
    cerr << "Error: Contact model value not understood.\n";
    exit(1);
  }

  cout << "  - Number of dynamic walls: " << walls.nw << '\n';

  if (grid.periodic == 1)
    cout << "  - 1st and 2nd dim. boundaries: Periodic\n";
  else if (grid.periodic == 2)
    cout << "  - 1st dim. boundaries: Visco-frictional walls\n";
  else
    cout << "  - 1st and 2nd dim. boundaries: Visco-frictional walls\n";

  if (walls.nw > 0) {
    cout << "  - Top BC: ";
    if (walls.wmode[0] == 0)
      cout << "Fixed\n";
    else if (walls.wmode[0] == 1)
      cout << "Deviatoric stress\n";
    else if (walls.wmode[0] == 2)
      cout << "Velocity\n";
    else {
      cerr << "Top BC not recognized!\n";
      exit(1);
    }
  }

  cout << "  - Grid: ";
  if (nd == 1)
    cout << grid.num[0];
  else if (nd == 2)
    cout << grid.num[0] << " * " << grid.num[1];
  else 
    cout << grid.num[0] << " * " 
      << grid.num[1] << " * "
      << grid.num[2];
  cout << " cells\n";
}

