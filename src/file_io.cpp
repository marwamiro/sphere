#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstdlib>

#include "typedefs.h"
#include "datatypes.h"
#include "constants.h"
#include "sphere.h"

// Get the address of the first byte of an object's representation
// See Stroustrup (2008) p. 388
template<class T>
char* as_bytes(T& i)	// treat a T as a sequence of bytes
{
  // get the address of the first byte of memory used
  // to store the object
  void* addr = &i;

  // treat the object as bytes
  return static_cast<char*>(addr);
}

// Read DEM data from binary file
// Note: Static-size arrays can be bulk-read with e.g.
//   ifs.read(as_bytes(grid.L), sizeof(grid.L))
// while dynamic, and vector arrays (e.g. Float4) must
// be read one value at a time.
void DEM::readbin(const char *target)
{
  using std::cout;  // stdout
  using std::cerr;  // stderr
  using std::endl;  // endline. Implicitly flushes buffer
  unsigned int i;

  // Open input file
  // if target is string: std::ifstream ifs(target.c_str(), std::ios_base::binary);
  std::ifstream ifs(target, std::ios_base::binary);
  if (!ifs) {
    cerr << "Could not read input binary file '"
      << target << endl;
    exit(1);
  }

  ifs.read(as_bytes(nd), sizeof(nd));
  ifs.read(as_bytes(np), sizeof(np));
  if (verbose == 1) {
    cout << "  - Number of dimensions: nd = " << nd << "\n"
      << "  - Number of particles:  np = " << np << "\n";
  }

  if (nd != ND) {
    cerr << "Dimensionality mismatch between dataset and this SPHERE program.\n"
      << "The dataset is " << nd 
      << "D, this SPHERE binary is " << ND << "D.\n"
      << "This execution is terminating." << endl;
    exit(-1); // Return unsuccessful exit status
  }

  // Check precision choice
  if (verbose == 1)
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
  if (verbose == 1)
    cout << " precision\n";

  // Read time parameters
  ifs.read(as_bytes(time.dt), sizeof(time.dt));
  ifs.read(as_bytes(time.current), sizeof(time.current));
  ifs.read(as_bytes(time.total), sizeof(time.total));
  ifs.read(as_bytes(time.file_dt), sizeof(time.file_dt));
  ifs.read(as_bytes(time.step_count), sizeof(time.step_count));

  // Output display parameters to screen
  if (verbose == 1) {
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
  }

  // For spatial vectors an array of Float4 vectors is chosen for best fit with 
  // GPU memory handling. Vector variable structure: ( x, y, z, <empty>).
  // Indexing starts from 0.

  // Allocate host arrays
  if (verbose == 1)
    cout << "\n  Allocating host memory:                         ";
  // Allocate more host arrays
  k.x	   = new Float4[np];
  k.xysum  = new Float2[np];
  k.vel	   = new Float4[np];
  k.force  = new Float4[np];
  k.angpos = new Float4[np];
  k.angvel = new Float4[np];
  k.torque = new Float4[np];

  e.es_dot = new Float[np];
  e.es     = new Float[np];
  e.ev_dot = new Float[np];
  e.ev     = new Float[np];
  e.p	   = new Float[np];

  if (verbose == 1)
    cout << "Done\n";

  if (verbose == 1)
    cout << "  Reading remaining data from input binary:       ";

  // Read grid parameters
  ifs.read(as_bytes(grid.origo), sizeof(grid.origo));
  ifs.read(as_bytes(grid.L), sizeof(grid.L));
  ifs.read(as_bytes(grid.num), sizeof(grid.num));
  ifs.read(as_bytes(grid.periodic), sizeof(grid.periodic));

  // Read kinematic values
  for (i = 0; i<np; ++i) {
    ifs.read(as_bytes(k.x[i].x), sizeof(Float));
    ifs.read(as_bytes(k.x[i].y), sizeof(Float));
    ifs.read(as_bytes(k.x[i].z), sizeof(Float));
    ifs.read(as_bytes(k.x[i].w), sizeof(Float));
  }
  for (i = 0; i<np; ++i) {
    ifs.read(as_bytes(k.xysum[i].x), sizeof(Float));
    ifs.read(as_bytes(k.xysum[i].y), sizeof(Float));
  }
  for (i = 0; i<np; ++i) {
    ifs.read(as_bytes(k.vel[i].x), sizeof(Float));
    ifs.read(as_bytes(k.vel[i].y), sizeof(Float));
    ifs.read(as_bytes(k.vel[i].z), sizeof(Float));
    ifs.read(as_bytes(k.vel[i].w), sizeof(Float));
  }
  for (i = 0; i<np; ++i) {
    ifs.read(as_bytes(k.force[i].x), sizeof(Float));
    ifs.read(as_bytes(k.force[i].y), sizeof(Float));
    ifs.read(as_bytes(k.force[i].z), sizeof(Float));
    //ifs.read(as_bytes(k.force[i].w), sizeof(Float));
  }
  for (i = 0; i<np; ++i) {
    ifs.read(as_bytes(k.angpos[i].x), sizeof(Float));
    ifs.read(as_bytes(k.angpos[i].y), sizeof(Float));
    ifs.read(as_bytes(k.angpos[i].z), sizeof(Float));
    //ifs.read(as_bytes(k.angpos[i].w), sizeof(Float));
  }
  for (i = 0; i<np; ++i) {
    ifs.read(as_bytes(k.angvel[i].x), sizeof(Float));
    ifs.read(as_bytes(k.angvel[i].y), sizeof(Float));
    ifs.read(as_bytes(k.angvel[i].z), sizeof(Float));
    //ifs.read(as_bytes(k.angvel[i].w), sizeof(Float));
  }
  for (i = 0; i<np; ++i) {
    ifs.read(as_bytes(k.torque[i].x), sizeof(Float));
    ifs.read(as_bytes(k.torque[i].y), sizeof(Float));
    ifs.read(as_bytes(k.torque[i].z), sizeof(Float));
    //ifs.read(as_bytes(k.torque[i].w), sizeof(Float));
  }

  // Read energies
  for (i = 0; i<np; ++i)
    ifs.read(as_bytes(e.es_dot[i]), sizeof(Float));
  for (i = 0; i<np; ++i)
    ifs.read(as_bytes(e.es[i]), sizeof(Float));
  for (i = 0; i<np; ++i)
    ifs.read(as_bytes(e.ev_dot[i]), sizeof(Float));
  for (i = 0; i<np; ++i)
    ifs.read(as_bytes(e.ev[i]), sizeof(Float));
  for (i = 0; i<np; ++i)
    ifs.read(as_bytes(e.p[i]), sizeof(Float));

  // Read constant parameters
  ifs.read(as_bytes(params.g), sizeof(params.g));
  ifs.read(as_bytes(params.k_n), sizeof(params.k_n));
  ifs.read(as_bytes(params.k_t), sizeof(params.k_t));
  ifs.read(as_bytes(params.k_r), sizeof(params.k_r));
  ifs.read(as_bytes(params.gamma_n), sizeof(params.gamma_n));
  ifs.read(as_bytes(params.gamma_t), sizeof(params.gamma_t));
  ifs.read(as_bytes(params.gamma_r), sizeof(params.gamma_r));
  ifs.read(as_bytes(params.mu_s), sizeof(params.mu_s));
  ifs.read(as_bytes(params.mu_d), sizeof(params.mu_d));
  ifs.read(as_bytes(params.mu_r), sizeof(params.mu_r));
  ifs.read(as_bytes(params.gamma_wn), sizeof(params.gamma_wn));
  ifs.read(as_bytes(params.gamma_wt), sizeof(params.gamma_wt));
  ifs.read(as_bytes(params.mu_ws), sizeof(params.mu_s));
  ifs.read(as_bytes(params.mu_wd), sizeof(params.mu_d));
  ifs.read(as_bytes(params.rho), sizeof(params.rho));
  ifs.read(as_bytes(params.contactmodel), sizeof(params.contactmodel));
  ifs.read(as_bytes(params.kappa), sizeof(params.kappa));
  ifs.read(as_bytes(params.db), sizeof(params.db));
  ifs.read(as_bytes(params.V_b), sizeof(params.V_b));

  // Read wall parameters
  ifs.read(as_bytes(walls.nw), sizeof(walls.nw));
  if (walls.nw > MAXWALLS) {
    cerr << "Error; MAXWALLS (" << MAXWALLS << ") in datatypes.h "
      << "is smaller than the number of walls specified in the "
      << "input file (" << walls.nw << ").\n";
    exit(1);
  }

  // Allocate host memory for walls
  // Wall normal (x,y,z), w: wall position on axis parallel to wall normal
  // Wall mass (x), velocity (y), force (z), and deviatoric stress (w)
  walls.nx    = new Float4[walls.nw];
  walls.mvfd  = new Float4[walls.nw];
  walls.force = new Float[walls.nw*np];

  ifs.read(as_bytes(walls.wmode), sizeof(walls.wmode));
  for (i = 0; i<walls.nw; ++i) {
    ifs.read(as_bytes(walls.nx[i].x), sizeof(Float));
    ifs.read(as_bytes(walls.nx[i].y), sizeof(Float));
    ifs.read(as_bytes(walls.nx[i].z), sizeof(Float));
    ifs.read(as_bytes(walls.nx[i].w), sizeof(Float));
  }
  for (i = 0; i<walls.nw; ++i) {
    ifs.read(as_bytes(walls.mvfd[i].x), sizeof(Float));
    ifs.read(as_bytes(walls.mvfd[i].y), sizeof(Float));
    ifs.read(as_bytes(walls.mvfd[i].z), sizeof(Float));
    ifs.read(as_bytes(walls.mvfd[i].w), sizeof(Float));
  }

  // Close file if it is still open
  if (ifs.is_open())
    ifs.close();

  if (verbose == 1)
    cout << "Done\n";

}

// Write DEM data to binary file
void DEM::writebin(const char *target)
{
  unsigned int i;

  // Open output file
  std::ofstream ofs(target, std::ios_base::binary);
  if (!ofs) {
    std::cerr << "could create output binary file '"
      << target << std::endl;
    exit(1); // Return unsuccessful exit status
  }

  // If double precision: Values can be written directly
  if (sizeof(Float) == sizeof(double)) {

    ofs.write(as_bytes(nd), sizeof(nd));
    ofs.write(as_bytes(np), sizeof(np));

    // Write time parameters
    ofs.write(as_bytes(time.dt), sizeof(time.dt));
    ofs.write(as_bytes(time.current), sizeof(time.current));
    ofs.write(as_bytes(time.total), sizeof(time.total));
    ofs.write(as_bytes(time.file_dt), sizeof(time.file_dt));
    ofs.write(as_bytes(time.step_count), sizeof(time.step_count));

    // Write grid parameters
    ofs.write(as_bytes(grid.origo), sizeof(grid.origo));
    ofs.write(as_bytes(grid.L), sizeof(grid.L));
    ofs.write(as_bytes(grid.num), sizeof(grid.num));
    ofs.write(as_bytes(grid.periodic), sizeof(grid.periodic));

    // Write kinematic values
    for (i = 0; i<np; ++i) {
      ofs.write(as_bytes(k.x[i].x), sizeof(Float));
      ofs.write(as_bytes(k.x[i].y), sizeof(Float));
      ofs.write(as_bytes(k.x[i].z), sizeof(Float));
      ofs.write(as_bytes(k.x[i].w), sizeof(Float));
    }
    for (i = 0; i<np; ++i) {
      ofs.write(as_bytes(k.xysum[i].x), sizeof(Float));
      ofs.write(as_bytes(k.xysum[i].y), sizeof(Float));
    }
    for (i = 0; i<np; ++i) {
      ofs.write(as_bytes(k.vel[i].x), sizeof(Float));
      ofs.write(as_bytes(k.vel[i].y), sizeof(Float));
      ofs.write(as_bytes(k.vel[i].z), sizeof(Float));
      ofs.write(as_bytes(k.vel[i].w), sizeof(Float));
    }
    for (i = 0; i<np; ++i) {
      ofs.write(as_bytes(k.force[i].x), sizeof(Float));
      ofs.write(as_bytes(k.force[i].y), sizeof(Float));
      ofs.write(as_bytes(k.force[i].z), sizeof(Float));
      //ofs.write(as_bytes(k.force[i].w), sizeof(Float));
    }
    for (i = 0; i<np; ++i) {
      ofs.write(as_bytes(k.angpos[i].x), sizeof(Float));
      ofs.write(as_bytes(k.angpos[i].y), sizeof(Float));
      ofs.write(as_bytes(k.angpos[i].z), sizeof(Float));
      //ofs.write(as_bytes(k.angpos[i].w), sizeof(Float));
    }
    for (i = 0; i<np; ++i) {
      ofs.write(as_bytes(k.angvel[i].x), sizeof(Float));
      ofs.write(as_bytes(k.angvel[i].y), sizeof(Float));
      ofs.write(as_bytes(k.angvel[i].z), sizeof(Float));
      //ofs.write(as_bytes(k.angvel[i].w), sizeof(Float));
    }
    for (i = 0; i<np; ++i) {
      ofs.write(as_bytes(k.torque[i].x), sizeof(Float));
      ofs.write(as_bytes(k.torque[i].y), sizeof(Float));
      ofs.write(as_bytes(k.torque[i].z), sizeof(Float));
      //ofs.write(as_bytes(k.torque[i].w), sizeof(Float));
    }

    // Write energies
    for (i = 0; i<np; ++i)
      ofs.write(as_bytes(e.es_dot[i]), sizeof(Float));
    for (i = 0; i<np; ++i)
      ofs.write(as_bytes(e.es[i]), sizeof(Float));
    for (i = 0; i<np; ++i)
      ofs.write(as_bytes(e.ev_dot[i]), sizeof(Float));
    for (i = 0; i<np; ++i)
      ofs.write(as_bytes(e.ev[i]), sizeof(Float));
    for (i = 0; i<np; ++i)
      ofs.write(as_bytes(e.p[i]), sizeof(Float));

    // Write constant parameters
    ofs.write(as_bytes(params.g), sizeof(params.g));
    ofs.write(as_bytes(params.k_n), sizeof(params.k_n));
    ofs.write(as_bytes(params.k_t), sizeof(params.k_t));
    ofs.write(as_bytes(params.k_r), sizeof(params.k_r));
    ofs.write(as_bytes(params.gamma_n), sizeof(params.gamma_n));
    ofs.write(as_bytes(params.gamma_t), sizeof(params.gamma_t));
    ofs.write(as_bytes(params.gamma_r), sizeof(params.gamma_r));
    ofs.write(as_bytes(params.mu_s), sizeof(params.mu_s));
    ofs.write(as_bytes(params.mu_d), sizeof(params.mu_d));
    ofs.write(as_bytes(params.mu_r), sizeof(params.mu_r));
    ofs.write(as_bytes(params.gamma_wn), sizeof(params.gamma_wn));
    ofs.write(as_bytes(params.gamma_wt), sizeof(params.gamma_wt));
    ofs.write(as_bytes(params.mu_ws), sizeof(params.mu_ws));
    ofs.write(as_bytes(params.mu_wd), sizeof(params.mu_wd));
    ofs.write(as_bytes(params.rho), sizeof(params.rho));
    ofs.write(as_bytes(params.contactmodel), sizeof(params.contactmodel));
    ofs.write(as_bytes(params.kappa), sizeof(params.kappa));
    ofs.write(as_bytes(params.db), sizeof(params.db));
    ofs.write(as_bytes(params.V_b), sizeof(params.V_b));

    // Write wall parameters
    ofs.write(as_bytes(walls.nw), sizeof(walls.nw));
    ofs.write(as_bytes(walls.wmode), sizeof(walls.wmode));
    for (i = 0; i<walls.nw; ++i) {
      ofs.write(as_bytes(walls.nx[i].x), sizeof(Float));
      ofs.write(as_bytes(walls.nx[i].y), sizeof(Float));
      ofs.write(as_bytes(walls.nx[i].z), sizeof(Float));
      ofs.write(as_bytes(walls.nx[i].w), sizeof(Float));
    }
    for (i = 0; i<walls.nw; ++i) {
      ofs.write(as_bytes(walls.mvfd[i].x), sizeof(Float));
      ofs.write(as_bytes(walls.mvfd[i].y), sizeof(Float));
      ofs.write(as_bytes(walls.mvfd[i].z), sizeof(Float));
      ofs.write(as_bytes(walls.mvfd[i].w), sizeof(Float));
    }

    // Close file if it is still open
    if (ofs.is_open())
      ofs.close();

  } else {
    std::cerr << "Can't write output when in single precision mode.\n";
  }
}

