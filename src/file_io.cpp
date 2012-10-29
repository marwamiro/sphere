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
  using std::cout;  // stdout
  using std::cerr;  // stderr
  using std::endl;  // endline. Implicitly flushes buffer

  if (verbose == 1)
    std::cout << "reading binary: " << target << '\n';

  int err = 0;

  // Open input file
  FILE *fp;
  ++err;
  if ((fp = fopen(target, "rb")) == NULL) {
    cerr << "Could not read input binary file '"
      << target << endl;
    exit(err);
  }

  // Read data
  ++err;
  if(fread(&nd, sizeof(nd), 1, fp) != 1) {
    cerr << "nd" << endl; exit(err); } // Return unsuccessful exit status
  ++err;
  if (fread(&np, sizeof(np), 1, fp) != 1) {
    cerr << "np" << endl; exit(err); } // Return unsuccessful exit status
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
    exit(err);
  }
  if (verbose == 1)
    cout << " precision\n";

  // Read time parameters
  ++err;
  if (fread(&time.dt, sizeof(time.dt), 1, fp) != 1) {
    cerr << "time.dt" << endl; exit(err); }
  ++err;
  if (fread(&time.current, sizeof(time.current), 1, fp) != 1) {
    cerr << "time.current" << endl; exit(err); }
  ++err;
  if (fread(&time.total, sizeof(time.total), 1, fp) != 1) {
    cerr << "time.total" << endl; exit(err); }
  ++err;
  if (fread(&time.file_dt, sizeof(time.file_dt), 1, fp) != 1) {
    cerr << "time.file_dt" << endl; exit(err); }
  ++err;
  if (fread(&time.step_count, sizeof(time.step_count), 1, fp) != 1) {
    cerr << "time.step_count" << endl; exit(err); }

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
  ++err;
  if (fread(&grid.origo, sizeof(grid.origo[0]), nd, fp) != nd) {
    cerr << "grid.origo" << endl; exit(err); }
  ++err;
  if (fread(&grid.L, sizeof(grid.L[0]), nd, fp) != nd) {
    cerr << "grid.L" << endl; exit(err); }
  ++err;
  if (fread(&grid.num, sizeof(grid.num[0]), nd, fp) != nd) {
    cerr << "grid.num" << endl; exit(err); }
  ++err;
  if (fread(&grid.periodic, sizeof(grid.periodic), 1, fp) != 1) {
    cerr << "grid.periodic" << endl; exit(err); }

  // Read kinematic values
  ++err;
  if (fread(&k.x, sizeof(Float4), np, fp) != np) {
    cerr << "k.x" << endl; exit(err); }
  ++err;
  if (fread(&k.xysum, sizeof(Float2), np, fp) != np) {
    cerr << "k.xysum" << endl; exit(err); }
  ++err;
  if (fread(&k.vel, sizeof(Float4), np, fp) != np) {
    cerr << "k.vel" << endl; exit(err); }
  ++err;
  if (fread(&k.force, sizeof(Float4), np, fp) != np) {
    cerr << "k.force" << endl; exit(err); }
  ++err;
  if (fread(&k.angpos, sizeof(Float4), np, fp) != np) {
    cerr << "k.angpos" << endl; exit(err); }
  ++err;
  if (fread(&k.angvel, sizeof(Float4), np, fp) != np) {
    cerr << "k.angvel" << endl; exit(err); }
  ++err;
  if (fread(&k.torque, sizeof(Float4), np, fp) != np) {
    cerr << "k.torque" << endl; exit(err); }
  // mass (m) and inertia (I) are calculated on device

  // Read energies
  ++err;
  if (fread(&e.es_dot, sizeof(e.es_dot[0]), np, fp) != np) {
    cerr << "e.es_dot" << endl; exit(err); }
  ++err;
  if (fread(&e.es, sizeof(e.es[0]), np, fp) != np) {
    cerr << "e.es" << endl; exit(err); }
  ++err;
  if (fread(&e.ev_dot, sizeof(e.ev_dot[0]), np, fp) != np) {
    cerr << "e.ev_dot" << endl; exit(err); }
  ++err;
  if (fread(&e.ev, sizeof(e.ev[0]), np, fp) != np) {
    cerr << "e.ev" << endl; exit(err); }
  ++err;
  if (fread(&e.p, sizeof(e.p[0]), np, fp) != np) {
    cerr << "e.p" << endl; exit(err); }

  // Read constant, global physical parameters
  ++err;
  if (fread(&params.g, sizeof(params.g[0]), nd, fp) != nd) {
    cerr << "params.g" << endl; exit(err); }
  ++err;
  if (fread(&params.k_n, sizeof(params.k_n), 1, fp) != 1) {
    cerr << "params.k_n" << endl; exit(err); }
  ++err;
  if (fread(&params.k_t, sizeof(params.k_t), 1, fp) != 1) {
    cerr << "params.k_t" << endl; exit(err); }
  ++err;
  if (fread(&params.k_r, sizeof(params.k_r), 1, fp) != 1) {
    cerr << "params.k_r" << endl; exit(err); }
  ++err;
  if (fread(&params.gamma_n, sizeof(params.gamma_n), 1, fp) != 1) {
    cerr << "params.gamma_n" << endl; exit(err); }
  ++err;
  if (fread(&params.gamma_t, sizeof(params.gamma_t), 1, fp) != 1) {
    cerr << "params.gamma_t" << endl; exit(err); }
  ++err;
  if (fread(&params.gamma_r, sizeof(params.gamma_r), 1, fp) != 1) {
    cerr << "params.gamma_r" << endl; exit(err); }
  ++err;
  if (fread(&params.mu_s, sizeof(params.mu_s), 1, fp) != 1) {
    cerr << "params.mu_s" << endl; exit(err); }
  ++err;
  if (fread(&params.mu_d, sizeof(params.mu_d), 1, fp) != 1) {
    cerr << "params.mu_d" << endl; exit(err); }
  ++err;
  if (fread(&params.mu_r, sizeof(params.mu_r), 1, fp) != 1) {
    cerr << "params.mu_r" << endl; exit(err); }
  ++err;
  if (fread(&params.rho, sizeof(params.rho), 1, fp) != 1) {
    cerr << "params.rho" << endl; exit(err); }
  ++err;
  if (fread(&params.contactmodel, sizeof(params.contactmodel), 1, fp) != 1) {
    cerr << "params.contactmodel" << endl; exit(err); }
  ++err;
  if (fread(&params.kappa, sizeof(params.kappa), 1, fp) != 1) {
    cerr << "params.kappa" << endl; exit(err); }
  ++err;
  if (fread(&params.db, sizeof(params.db), 1, fp) != 1) {
    cerr << "params.db" << endl; exit(err); }
  ++err;
  if (fread(&params.V_b, sizeof(params.V_b), 1, fp) != 1) {
  cerr << "params.V_b" << endl; exit(err); }

  // Read wall parameters
  ++err;
  if (fread(&walls.nw, sizeof(walls.nw), 1, fp) != 1) {
    cerr << "walls.nw" << endl; exit(err); }
  // Allocate host memory for walls
  // Wall normal (x,y,z), w: wall position on axis parallel to wall normal
  // Wall mass (x), velocity (y), force (z), and deviatoric stress (w)
  walls.nx   = new Float4[walls.nw];
  walls.mvfd = new Float4[walls.nw]; 

  ++err;
  if (fread(&walls.wmode, sizeof(walls.wmode[0]), walls.nw, fp) != walls.nw) {
    cerr << "walls.wmode" << endl; exit(err); }
  ++err;
  if (fread(&walls.nx, sizeof(Float4), walls.nw, fp) != 1) {
    cerr << "walls.nx" << endl; exit(err); }
  ++err;
  if (fread(&walls.mvfd, sizeof(Float4), walls.nw, fp) != 1) {
    cerr << "walls.mvfd" << endl; exit(err); }
  ++err;
  if (fread(&walls.gamma_wn, sizeof(walls.gamma_wn), 1, fp) != 1) {
    cerr << "walls.gamma_wn" << endl; exit(err); }
  ++err;
  if (fread(&walls.gamma_wt, sizeof(walls.gamma_wt), 1, fp) != 1) {
    cerr << "walls.gamma_wt" << endl; exit(err); }
  ++err;
  if (fread(&walls.gamma_wr, sizeof(walls.gamma_wr), 1, fp) != 1) {
    cerr << "walls.gamma_wr" << endl; exit(err); }

  if (walls.nw > MAXWALLS) {
    cerr << "Error; MAXWALLS (" << MAXWALLS << ") in datatypes.h "
      << "is smaller than the number of walls specified in the "
      << "input file (" << walls.nw << ").\n";
  }

  fclose(fp);

  if (verbose == 1)
    cout << "Done\n";

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
    exit(err); // Return unsuccessful exit status
  }

  // If double precision: Values can be written directly
  if (sizeof(Float) == sizeof(double)) {

    fwrite(&nd, sizeof(nd), 1, fp);
    fwrite(&np, sizeof(np), 1, fp);

    // Write temporal parameters
    ++err;
    if (fwrite(&time.dt, sizeof(time.dt), 1, fp) != 1)
      exit(err);
    ++err;
    if (fwrite(&time.current, sizeof(time.current), 1, fp) != 1)
      exit(err);
    ++err;
    if (fwrite(&time.total, sizeof(time.total), 1, fp) != 1)
      exit(err);
    ++err;
    if (fwrite(&time.file_dt, sizeof(time.file_dt), 1, fp) != 1)
      exit(err);
    ++err;
    if (fwrite(&time.step_count, sizeof(time.step_count), 1, fp) != 1)
      exit(err);

    // Write grid parameters
    ++err;
    if (fwrite(&grid.origo, sizeof(grid.origo[0]), nd, fp) != nd)
      exit(err);
    ++err;
    if (fwrite(&grid.L, sizeof(grid.L[0]), nd, fp) != nd)
      exit(err);
    ++err;
    if (fwrite(&grid.num, sizeof(grid.num[0]), nd, fp) != nd)
      exit(err);
    ++err;
    if (fwrite(&grid.periodic, sizeof(grid.periodic), 1, fp) != 1)
      exit(err);

    // Write kinematic values
    ++err;
    if (fwrite(&k.x, sizeof(Float4), np, fp) != np)
      exit(err);
    ++err;
    if (fwrite(&k.xysum, sizeof(Float2), np, fp) != np)
      exit(err);
    ++err;
    if (fwrite(&k.vel, sizeof(Float4), np, fp) != np)
      exit(err);
    ++err;
    if (fwrite(&k.force, sizeof(Float4), np, fp) != np)
      exit(err);
    ++err;
    if (fwrite(&k.angpos, sizeof(Float4), np, fp) != np)
      exit(err);
    ++err;
    if (fwrite(&k.angvel, sizeof(Float4), np, fp) != np)
      exit(err);
    ++err;
    if (fwrite(&k.torque, sizeof(Float4), np, fp) != np)
      exit(err);

    // Write energies
    ++err;
    if (fwrite(&e.es_dot, sizeof(e.es_dot[0]), np, fp) != np)
      exit(err);
    ++err;
    if (fwrite(&e.es, sizeof(e.es[0]), np, fp) != np)
      exit(err);
    ++err;
    if (fwrite(&e.ev_dot, sizeof(e.ev_dot[0]), np, fp) != np)
      exit(err);
    ++err;
    if (fwrite(&e.ev, sizeof(e.ev[0]), np, fp) != np)
      exit(err);
    ++err;
    if (fwrite(&e.p, sizeof(e.p[0]), np, fp) != np)
      exit(err);

    // Write constant, global physical parameters
    ++err;
    if (fwrite(&params.g, sizeof(params.g[0]), nd, fp) != nd)
      exit(err);
    ++err;
    if (fwrite(&params.k_n, sizeof(params.k_n), 1, fp) != 1)
      exit(err);
    ++err;
    if (fwrite(&params.k_t, sizeof(params.k_t), 1, fp) != 1)
      exit(err);
    ++err;
    if (fwrite(&params.k_r, sizeof(params.k_r), 1, fp) != 1)
      exit(err);
    ++err;
    if (fwrite(&params.gamma_n, sizeof(params.gamma_n), 1, fp) != 1)
      exit(err);
    ++err;
    if (fwrite(&params.gamma_t, sizeof(params.gamma_t), 1, fp) != 1)
      exit(err);
    ++err;
    if (fwrite(&params.gamma_r, sizeof(params.gamma_r), 1, fp) != 1)
      exit(err);
    ++err;
    if (fwrite(&params.mu_s, sizeof(params.mu_s), 1, fp) != 1)
      exit(err);
    ++err;
    if (fwrite(&params.mu_d, sizeof(params.mu_d), 1, fp) != 1)
      exit(err);
    ++err;
    if (fwrite(&params.mu_r, sizeof(params.mu_r), 1, fp) != 1)
      exit(err);
    ++err;
    if (fwrite(&params.rho, sizeof(params.rho), 1, fp) != 1)
      exit(err);
    ++err;
    if (fwrite(&params.contactmodel, sizeof(params.contactmodel), 1, fp) != 1)
      exit(err);
    ++err;
    if (fwrite(&params.kappa, sizeof(params.kappa), 1, fp) != 1)
      exit(err);
    ++err;
    if (fwrite(&params.db, sizeof(params.db), 1, fp) != 1)
      exit(err);
    ++err;
    if (fwrite(&params.V_b, sizeof(params.V_b), 1, fp) != 1)
      exit(err);

    // Write walls parameters
    ++err;
    if (fwrite(&walls.nw, sizeof(walls.nw), 1, fp) != 1)
      exit(err);
    ++err;
    if (fwrite(&walls.wmode, sizeof(walls.wmode[0]), walls.nw, fp) != walls.nw)
      exit(err);
    ++err;
    if (fwrite(&walls.nx, sizeof(Float4), walls.nw, fp) != walls.nw)
      exit(err);
    ++err;
    if (fwrite(&walls.mvfd, sizeof(Float4), walls.nw, fp) != walls.nw)
      exit(err);
    ++err;
    if (fwrite(&walls.gamma_wn, sizeof(walls.gamma_wn), 1, fp) != 1)
      exit(err);
    ++err;
    if (fwrite(&walls.gamma_wt, sizeof(walls.gamma_wt), 1, fp) != 1)
      exit(err);
    ++err;
    if (fwrite(&walls.gamma_wr, sizeof(walls.gamma_wr), 1, fp) != 1)
      exit(err);

  } else {
    std::cerr << "Can't write output when in single precision mode.\n";
  }
}

