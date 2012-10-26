#include <iostream>
#include <cstdio>
#include <cstdlib>

#include "typedefs.h"
#include "datatypes.h"
#include "constants.h"
#include "sphere.h"

// Read DEM data from binary file
void DEM::readbin(const char *target)
{

  if (verbose == 1)
    std::cout << "reading binary: " << target << '\n';

  int err = 0;

  // Open input file
  FILE *fp;
  if ((fp = fopen(target, "rb")) == NULL) {
    std::cerr << "Could not read input binary file '"
      << target << "'\n";
    exit(++err);
  }

  // Read data
  if(fread(&nd, sizeof(nd), 1, fp) != 1)
    exit(++err); // Return unsuccessful exit status

  if (fread(&np, sizeof(np), 1, fp) != 1)
    exit(++err); // Return unsuccessful exit status
  if (verbose == 1) {
    std::cout << "  - Number of dimensions: nd = " << nd << "\n"
      << "  - Number of particles:  np = " << np << "\n";
  }

  if (nd != ND) {
    std::cerr << "Dimensionality mismatch between dataset and this SPHERE program.\n"
      << "The dataset is " << nd 
      << "D, this SPHERE binary is " << ND << "D.\n"
      << "This execution is terminating.\n";
    exit(++err); // Return unsuccessful exit status
  }

  // Check precision choice
  if (verbose == 1)
    std::cout << "  - Compiled for ";
  if (sizeof(Float) == sizeof(float)) {
    if (verbose == 1)
      std::cout << "single";
  } else if (sizeof(Float) == sizeof(double)) {
    if (verbose == 1)
      std::cout << "double";
  } else {
    std::cerr << "Error! Chosen precision not available. Check datatypes.h\n";
    exit(++err);
  }
  if (verbose == 1)
    std::cout << " precision\n";

  // Read time parameters
  if (fread(&time.dt, sizeof(time.dt), 1, fp) != 1)
    exit(++err); // Return unsuccessful exit status
  if (fread(&time.current, sizeof(time.current), 1, fp) != 1)
    exit(++err); 
  if (fread(&time.total, sizeof(time.total), 1, fp) != 1)
    exit(++err); 
  if (fread(&time.file_dt, sizeof(time.file_dt), 1, fp) != 1)
    exit(++err); 
  if (fread(&time.step_count, sizeof(time.step_count), 1, fp) != 1)
    exit(++err); 

  // Output display parameters to screen
  if (verbose == 1) {
    std::cout << "  - Timestep length:      time.dt         = " 
      << time.dt << " s\n"
      << "  - Start at time:        time.current    = " 
      << time.current << " s\n"
      << "  - Total sim. time:      time.total      = " 
      << time.total << " s\n"
      << "  - File output interval: time.file_dt    = " 
      << time.file_dt << " s\n"
      << "  - Start at step count:  time.step_count = " 
      << time.step_count << "\n";
  }

  // For spatial vectors an array of Float4 vectors is chosen for best fit with 
  // GPU memory handling. Vector variable structure: ( x, y, z, <empty>).
  // Indexing starts from 0.

  // Allocate host arrays
  if (verbose == 1)
    std::cout << "\n  Allocating host memory:                         ";
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
    std::cout << "Done\n";

  if (verbose == 1)
    std::cout << "  Reading remaining data from input binary:       ";

  // Read grid parameters
  if (fread(&grid.origo, sizeof(grid.origo[0]), nd, fp) != nd)
    exit(++err); // Return unsuccessful exit status
  if (fread(&grid.L, sizeof(grid.L[0]), nd, fp) != nd)
    exit(++err);
  if (fread(&grid.num, sizeof(grid.num[0]), nd, fp) != nd)
    exit(++err);
  if (fread(&grid.periodic, sizeof(grid.periodic), 1, fp) != 1)
    exit(++err);

  // Read kinematic values
  if (fread(&k.x, sizeof(Float4), np, fp) != np)
    exit(++err);
  if (fread(&k.xysum, sizeof(Float2), np, fp) != np)
    exit(++err);
  if (fread(&k.vel, sizeof(Float4), np, fp) != np)
    exit(++err);
  if (fread(&k.force, sizeof(Float4), np, fp) != np)
    exit(++err);
  if (fread(&k.angpos, sizeof(Float4), np, fp) != np)
    exit(++err);
  if (fread(&k.angvel, sizeof(Float4), np, fp) != np)
    exit(++err);
  if (fread(&k.torque, sizeof(Float4), np, fp) != np)
    exit(++err);
  // mass (m) and inertia (I) are calculated on device

  // Read energies
  if (fread(&e.es_dot, sizeof(e.es_dot[0]), np, fp) != np)
    exit(++err);
  if (fread(&e.es, sizeof(e.es[0]), np, fp) != np)
    exit(++err);
  if (fread(&e.ev_dot, sizeof(e.ev_dot[0]), np, fp) != np)
    exit(++err);
  if (fread(&e.ev, sizeof(e.ev[0]), np, fp) != np)
    exit(++err);
  if (fread(&e.p, sizeof(e.p[0]), np, fp) != np)
    exit(++err);

  // Read constant, global physical parameters
  if (fread(&params.g, sizeof(params.g[0]), nd, fp) != nd)
    exit(++err);
  if (fread(&params.k_n, sizeof(params.k_n), 1, fp) != 1)
    exit(++err);
  if (fread(&params.k_t, sizeof(params.k_t), 1, fp) != 1)
    exit(++err);
  if (fread(&params.k_r, sizeof(params.k_r), 1, fp) != 1)
    exit(++err);
  if (fread(&params.gamma_n, sizeof(params.gamma_n), 1, fp) != 1)
    exit(++err);
  if (fread(&params.gamma_t, sizeof(params.gamma_t), 1, fp) != 1)
    exit(++err);
  if (fread(&params.gamma_r, sizeof(params.gamma_r), 1, fp) != 1)
    exit(++err);
  if (fread(&params.mu_s, sizeof(params.mu_s), 1, fp) != 1)
    exit(++err);
  if (fread(&params.mu_d, sizeof(params.mu_d), 1, fp) != 1)
    exit(++err);
  if (fread(&params.mu_r, sizeof(params.mu_r), 1, fp) != 1)
    exit(++err);
  if (fread(&params.rho, sizeof(params.rho), 1, fp) != 1)
    exit(++err);
  if (fread(&params.contactmodel, sizeof(params.contactmodel), 1, fp) != 1)
    exit(++err);
  if (fread(&params.kappa, sizeof(params.kappa), 1, fp) != 1)
    exit(++err);
  if (fread(&params.db, sizeof(params.db), 1, fp) != 1)
    exit(++err); 
  if (fread(&params.V_b, sizeof(params.V_b), 1, fp) != 1)
    exit(++err); 

  // Read wall parameters
  if (fread(&walls.nw, sizeof(walls.nw), 1, fp) != 1)
    exit(++err); 
  // Allocate host memory for walls
  // Wall normal (x,y,z), w: wall position on axis parallel to wall normal
  // Wall mass (x), velocity (y), force (z), and deviatoric stress (w)
  walls.nx   = new Float4[walls.nw];
  walls.mvfd = new Float4[walls.nw]; 

  if (fread(&walls.wmode, sizeof(walls.wmode[0]), walls.nw, fp) != walls.nw)
    exit(++err);
  if (fread(&walls.nx, sizeof(Float4), walls.nw, fp) != 1)
    exit(++err);
  if (fread(&walls.mvfd, sizeof(Float4), walls.nw, fp) != 1)
    exit(++err);
  if (fread(&walls.gamma_wn, sizeof(walls.gamma_wn), 1, fp) != 1)
    exit(++err);
  if (fread(&walls.gamma_wt, sizeof(walls.gamma_wt), 1, fp) != 1)
    exit(++err);
  if (fread(&walls.gamma_wr, sizeof(walls.gamma_wr), 1, fp) != 1)
    exit(++err);


  if (walls.nw > MAXWALLS) {
    std::cerr << "Error; MAXWALLS (" << MAXWALLS << ") in datatypes.h "
      << "is smaller than the number of walls specified in the "
      << "input file (" << walls.nw << ").\n";
  }

  fclose(fp);

  if (verbose == 1)
    std::cout << "Done\n";

}

// Write DEM data to binary file
void DEM::writebin(const char *target)
{
  int err = 0;

  // Open output file
  FILE *fp;
  if ((fp = fopen(target, "wb")) == NULL) {
    std::cerr << "could create output binary file '"
      << target << "'.\n";
    exit(++err); // Return unsuccessful exit status
  }

  // If double precision: Values can be written directly
  if (sizeof(Float) == sizeof(double)) {

    fwrite(&nd, sizeof(nd), 1, fp);
    fwrite(&np, sizeof(np), 1, fp);

    // Write temporal parameters
    if (fwrite(&time.dt, sizeof(time.dt), 1, fp) != 1)
      exit(++err);
    if (fwrite(&time.current, sizeof(time.current), 1, fp) != 1)
      exit(++err);
    if (fwrite(&time.total, sizeof(time.total), 1, fp) != 1)
      exit(++err);
    if (fwrite(&time.file_dt, sizeof(time.file_dt), 1, fp) != 1)
      exit(++err);
    if (fwrite(&time.step_count, sizeof(time.step_count), 1, fp) != 1)
      exit(++err);

    // Write grid parameters
    if (fwrite(&grid.origo, sizeof(grid.origo[0]), nd, fp) != nd)
      exit(++err);
    if (fwrite(&grid.L, sizeof(grid.L[0]), nd, fp) != nd)
      exit(++err);
    if (fwrite(&grid.num, sizeof(grid.num[0]), nd, fp) != nd)
      exit(++err);
    if (fwrite(&grid.periodic, sizeof(grid.periodic), 1, fp) != 1)
      exit(++err);

    // Write kinematic values
    if (fwrite(&k.x, sizeof(Float4), np, fp) != np)
      exit(++err);
    if (fwrite(&k.xysum, sizeof(Float2), np, fp) != np)
      exit(++err);
    if (fwrite(&k.vel, sizeof(Float4), np, fp) != np)
      exit(++err);
    if (fwrite(&k.force, sizeof(Float4), np, fp) != np)
      exit(++err);
    if (fwrite(&k.angpos, sizeof(Float4), np, fp) != np)
      exit(++err);
    if (fwrite(&k.angvel, sizeof(Float4), np, fp) != np)
      exit(++err);
    if (fwrite(&k.torque, sizeof(Float4), np, fp) != np)
      exit(++err);

    // Write energies
    if (fwrite(&e.es_dot, sizeof(e.es_dot[0]), np, fp) != np)
      exit(++err);
    if (fwrite(&e.es, sizeof(e.es[0]), np, fp) != np)
      exit(++err);
    if (fwrite(&e.ev_dot, sizeof(e.ev_dot[0]), np, fp) != np)
      exit(++err);
    if (fwrite(&e.ev, sizeof(e.ev[0]), np, fp) != np)
      exit(++err);
    if (fwrite(&e.p, sizeof(e.p[0]), np, fp) != np)
      exit(++err);

    // Write constant, global physical parameters
    if (fwrite(&params.g, sizeof(params.g[0]), nd, fp) != nd)
      exit(++err);
    if (fwrite(&params.k_n, sizeof(params.k_n), 1, fp) != 1)
      exit(++err);
    if (fwrite(&params.k_t, sizeof(params.k_t), 1, fp) != 1)
      exit(++err);
    if (fwrite(&params.k_r, sizeof(params.k_r), 1, fp) != 1)
      exit(++err);
    if (fwrite(&params.gamma_n, sizeof(params.gamma_n), 1, fp) != 1)
      exit(++err);
    if (fwrite(&params.gamma_t, sizeof(params.gamma_t), 1, fp) != 1)
      exit(++err);
    if (fwrite(&params.gamma_r, sizeof(params.gamma_r), 1, fp) != 1)
      exit(++err);
    if (fwrite(&params.mu_s, sizeof(params.mu_s), 1, fp) != 1)
      exit(++err);
    if (fwrite(&params.mu_d, sizeof(params.mu_d), 1, fp) != 1)
      exit(++err);
    if (fwrite(&params.mu_r, sizeof(params.mu_r), 1, fp) != 1)
      exit(++err);
    if (fwrite(&params.rho, sizeof(params.rho), 1, fp) != 1)
      exit(++err);
    if (fwrite(&params.contactmodel, sizeof(params.contactmodel), 1, fp) != 1)
      exit(++err);
    if (fwrite(&params.kappa, sizeof(params.kappa), 1, fp) != 1)
      exit(++err);
    if (fwrite(&params.db, sizeof(params.db), 1, fp) != 1)
      exit(++err);
    if (fwrite(&params.V_b, sizeof(params.V_b), 1, fp) != 1)
      exit(++err);

    // Write walls parameters
    if (fwrite(&walls.nw, sizeof(walls.nw), 1, fp) != 1)
      exit(++err);
    if (fwrite(&walls.wmode, sizeof(walls.wmode[0]), walls.nw, fp) != walls.nw)
      exit(++err);
    if (fwrite(&walls.nx, sizeof(Float4), walls.nw, fp) != walls.nw)
      exit(++err);
    if (fwrite(&walls.mvfd, sizeof(Float4), walls.nw, fp) != walls.nw)
      exit(++err);
    if (fwrite(&walls.gamma_wn, sizeof(walls.gamma_wn), 1, fp) != 1)
      exit(++err);
    if (fwrite(&walls.gamma_wt, sizeof(walls.gamma_wt), 1, fp) != 1)
      exit(++err);
    if (fwrite(&walls.gamma_wr, sizeof(walls.gamma_wr), 1, fp) != 1)
      exit(++err);

  } else {
    std::cerr << "Can't write output when in single precision mode.\n";
  }
}

