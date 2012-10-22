#include <stdio.h>  // Standard library functions for file input and output
#include <stdlib.h> // Functions involving memory allocation, process control, conversions and others
#include <unistd.h> // UNIX only: For getcwd
#include <string.h> // For strerror and strcmp
#include <cuda.h>
#include "datatypes.h"


// Write host variables to target binary file
// The output format should ALWAYS be double precision,
// so this function will typecast the data before write
// if it is single precision.
int fwritebin(char *target, 
    Particles *p, 
    Float4 *host_x, 
    Float4 *host_vel, 
    Float4 *host_angvel, 
    Float4 *host_force, 
    Float4 *host_torque, 
    Float4 *host_angpos, 
    uint4 *host_bonds,
    Grid *grid, 
    Time *time, 
    Params *params,
    Float4 *host_w_nx, 
    Float4 *host_w_mvfd)
{

  FILE *fp;
  unsigned int u;
  unsigned int j;

  if ((fp = fopen(target,"wb")) == NULL) {
    printf("Could create output binary file. Bye.\n");
    return 1; // Return unsuccessful exit status
  }

  // If double precision: Values can be written directly
  if (sizeof(Float) == sizeof(double)) {

    // World dimensions
    fwrite(&grid->nd, sizeof(grid->nd), 1, fp);

    // Number of particles
    fwrite(&p->np, sizeof(p->np), 1, fp);

    // Temporal parameters
    fwrite(&time->dt, sizeof(time->dt), 1, fp);
    fwrite(&time->current, sizeof(time->current), 1, fp);
    fwrite(&time->total, sizeof(time->total), 1, fp);
    fwrite(&time->file_dt, sizeof(time->file_dt), 1, fp);
    fwrite(&time->step_count, sizeof(time->step_count), 1, fp);

    // World coordinate system origo
    for (u=0; u<grid->nd; ++u) {
      fwrite(&grid->origo[u], sizeof(grid->origo[u]), 1, fp);
    }

    // World dimensions
    for (u=0; u<grid->nd; ++u) {
      fwrite(&grid->L[u], sizeof(grid->L[u]), 1, fp);
    }

    // Grid cells along each dimension
    for (u=0; u<grid->nd; ++u) {
      fwrite(&grid->num[u], sizeof(grid->num[u]), 1, fp);
    }

    // Particle vectors
    for (j=0; j<p->np; ++j) {
      // x-axis
      fwrite(&host_x[j].x, sizeof(Float), 1, fp);
      fwrite(&host_vel[j].x, sizeof(Float), 1, fp);
      fwrite(&host_angvel[j].x, sizeof(Float), 1, fp);
      fwrite(&host_force[j].x, sizeof(Float), 1, fp);
      fwrite(&host_torque[j].x, sizeof(Float), 1, fp);
      fwrite(&host_angpos[j].x, sizeof(Float), 1, fp);

      // y-axis
      fwrite(&host_x[j].y, sizeof(Float), 1, fp);
      fwrite(&host_vel[j].y, sizeof(Float), 1, fp);
      fwrite(&host_angvel[j].y, sizeof(Float), 1, fp);
      fwrite(&host_force[j].y, sizeof(Float), 1, fp);
      fwrite(&host_torque[j].y, sizeof(Float), 1, fp);
      fwrite(&host_angpos[j].y, sizeof(Float), 1, fp);

      // z-axis
      fwrite(&host_x[j].z, sizeof(Float), 1, fp);
      fwrite(&host_vel[j].z, sizeof(Float), 1, fp);
      fwrite(&host_angvel[j].z, sizeof(Float), 1, fp);
      fwrite(&host_force[j].z, sizeof(Float), 1, fp);
      fwrite(&host_torque[j].z, sizeof(Float), 1, fp);
      fwrite(&host_angpos[j].z, sizeof(Float), 1, fp);
    } 

    // Individual particle values
    for (j=0; j<p->np; ++j) {
      fwrite(&host_vel[j].w, sizeof(Float), 1, fp);
      fwrite(&host_x[j].w, sizeof(Float), 1, fp);
      fwrite(&p->radius[j], sizeof(p->radius[j]), 1, fp);
      fwrite(&p->rho[j], sizeof(p->rho[j]), 1, fp);
      fwrite(&p->k_n[j], sizeof(p->k_n[j]), 1, fp);
      fwrite(&p->k_t[j], sizeof(p->k_t[j]), 1, fp);
      fwrite(&p->k_r[j], sizeof(p->k_r[j]), 1, fp);
      fwrite(&p->gamma_n[j], sizeof(p->gamma_n[j]), 1, fp);
      fwrite(&p->gamma_t[j], sizeof(p->gamma_t[j]), 1, fp);
      fwrite(&p->gamma_r[j], sizeof(p->gamma_r[j]), 1, fp);
      fwrite(&p->mu_s[j], sizeof(p->mu_s[j]), 1, fp);
      fwrite(&p->mu_d[j], sizeof(p->mu_d[j]), 1, fp);
      fwrite(&p->mu_r[j], sizeof(p->mu_r[j]), 1, fp);
      fwrite(&p->es_dot[j], sizeof(p->es_dot[j]), 1, fp);
      fwrite(&p->ev_dot[j], sizeof(p->ev_dot[j]), 1, fp);
      fwrite(&p->es[j], sizeof(p->es[j]), 1, fp);
      fwrite(&p->ev[j], sizeof(p->ev[j]), 1, fp);
      fwrite(&p->p[j], sizeof(p->p[j]), 1, fp);
    }

    // Singular parameters
    fwrite(&params->global, sizeof(params->global), 1, fp);
    for (u=0; u<grid->nd; ++u) {
      fwrite(&params->g[u], sizeof(params->g[u]), 1, fp);
    }
    fwrite(&params->kappa, sizeof(params->kappa), 1, fp);
    fwrite(&params->db, sizeof(params->db), 1, fp);
    fwrite(&params->V_b, sizeof(params->V_b), 1, fp);
    fwrite(&params->shearmodel, sizeof(params->shearmodel), 1, fp);

    // Walls
    fwrite(&params->nw, sizeof(params->nw), 1, fp); // No. of walls
    for (j=0; j<params->nw; ++j) {
      fwrite(&params->wmode[j], sizeof(params->wmode[j]), 1, fp);
      // Wall normal
      fwrite(&host_w_nx[j].x, sizeof(Float), 1, fp);
      fwrite(&host_w_nx[j].y, sizeof(Float), 1, fp);
      fwrite(&host_w_nx[j].z, sizeof(Float), 1, fp);

      fwrite(&host_w_nx[j].w, sizeof(Float), 1, fp);   // Wall position
      fwrite(&host_w_mvfd[j].x, sizeof(Float), 1, fp); // Wall mass
      fwrite(&host_w_mvfd[j].y, sizeof(Float), 1, fp); // Wall velocity
      fwrite(&host_w_mvfd[j].z, sizeof(Float), 1, fp); // Wall force
      fwrite(&host_w_mvfd[j].w, sizeof(Float), 1, fp); // Wall deviatoric stress
    }
    fwrite(&params->periodic, sizeof(params->periodic), 1, fp);
    fwrite(&params->gamma_wn, sizeof(params->gamma_wn), 1, fp);
    fwrite(&params->gamma_wt, sizeof(params->gamma_wt), 1, fp);
    fwrite(&params->gamma_wr, sizeof(params->gamma_wr), 1, fp);


    // Write bond pair values
    for (j=0; j<p->np; ++j) {
      fwrite(&host_bonds[j].x, sizeof(unsigned int), 1, fp);
      fwrite(&host_bonds[j].y, sizeof(unsigned int), 1, fp);
      fwrite(&host_bonds[j].z, sizeof(unsigned int), 1, fp);
      fwrite(&host_bonds[j].w, sizeof(unsigned int), 1, fp);
    }

  } else if (sizeof(Float) == sizeof(float)) {
    // Single precision: Type conversion required

    double d; // Double precision placeholder

    // World dimensions
    fwrite(&grid->nd, sizeof(grid->nd), 1, fp);

    // Number of particles
    fwrite(&p->np, sizeof(p->np), 1, fp);

    // Temporal parameters
    d = (double)time->dt;
    fwrite(&d, sizeof(d), 1, fp);
    fwrite(&time->current, sizeof(time->current), 1, fp);
    fwrite(&time->total, sizeof(time->total), 1, fp);
    d = (double)time->file_dt;
    fwrite(&d, sizeof(d), 1, fp);
    fwrite(&time->step_count, sizeof(time->step_count), 1, fp);

    // World coordinate system origo
    for (u=0; u<grid->nd; ++u) {
      d = (double)grid->origo[u];
      fwrite(&d, sizeof(d), 1, fp);
    }

    // World dimensions
    for (u=0; u<grid->nd; ++u) {
      d = (double)grid->L[u];
      fwrite(&d, sizeof(d), 1, fp);
    }

    // Grid cells along each dimension
    for (u=0; u<grid->nd; ++u) {
      fwrite(&grid->num[u], sizeof(grid->num[u]), 1, fp);
    }

    // Particle vectors
    for (j=0; j<p->np; ++j) {
      // x-axis
      d = (double)host_x[j].x;
      fwrite(&d, sizeof(d), 1, fp);
      d = (double)host_vel[j].x;
      fwrite(&d, sizeof(d), 1, fp);
      d = (double)host_angvel[j].x;
      fwrite(&d, sizeof(d), 1, fp);
      d = (double)host_force[j].x;
      fwrite(&d, sizeof(d), 1, fp);
      d = (double)host_torque[j].x;
      fwrite(&d, sizeof(d), 1, fp);
      d = (double)host_angpos[j].x;
      fwrite(&d, sizeof(d), 1, fp);

      // y-axis
      d = (double)host_x[j].y;
      fwrite(&d, sizeof(d), 1, fp);
      d = (double)host_vel[j].y;
      fwrite(&d, sizeof(d), 1, fp);
      d = (double)host_angvel[j].y;
      fwrite(&d, sizeof(d), 1, fp);
      d = (double)host_force[j].y;
      fwrite(&d, sizeof(d), 1, fp);
      d = (double)host_torque[j].y;
      fwrite(&d, sizeof(d), 1, fp);
      d = (double)host_angpos[j].y;
      fwrite(&d, sizeof(d), 1, fp);

      // z-axis
      d = (double)host_x[j].z;
      fwrite(&d, sizeof(d), 1, fp);
      d = (double)host_vel[j].z;
      fwrite(&d, sizeof(d), 1, fp);
      d = (double)host_angvel[j].z;
      fwrite(&d, sizeof(d), 1, fp);
      d = (double)host_force[j].z;
      fwrite(&d, sizeof(d), 1, fp);
      d = (double)host_torque[j].z;
      fwrite(&d, sizeof(d), 1, fp);
      d = (double)host_angpos[j].z;
      fwrite(&d, sizeof(d), 1, fp);
    } 

    // Individual particle values
    for (j=0; j<p->np; ++j) {
      d = (double)host_vel[j].w;
      fwrite(&d, sizeof(d), 1, fp);
      d = (double)host_x[j].w;
      fwrite(&d, sizeof(d), 1, fp);
      d = (double)p->radius[j];
      fwrite(&d, sizeof(d), 1, fp);
      d = (double)p->rho[j];
      fwrite(&d, sizeof(d), 1, fp);
      d = (double)p->k_n[j];
      fwrite(&d, sizeof(d), 1, fp);
      d = (double)p->k_t[j];
      fwrite(&d, sizeof(d), 1, fp);
      d = (double)p->k_r[j];
      fwrite(&d, sizeof(d), 1, fp);
      d = (double)p->gamma_n[j];
      fwrite(&d, sizeof(d), 1, fp);
      d = (double)p->gamma_t[j];
      fwrite(&d, sizeof(d), 1, fp);
      d = (double)p->gamma_r[j];
      fwrite(&d, sizeof(d), 1, fp);
      d = (double)p->mu_s[j];
      fwrite(&d, sizeof(d), 1, fp);
      d = (double)p->mu_d[j];
      fwrite(&d, sizeof(d), 1, fp);
      d = (double)p->mu_r[j];
      fwrite(&d, sizeof(d), 1, fp);
      d = (double)p->es_dot[j];
      fwrite(&d, sizeof(d), 1, fp);
      d = (double)p->ev_dot[j];
      fwrite(&d, sizeof(d), 1, fp);
      d = (double)p->es[j];
      fwrite(&d, sizeof(d), 1, fp);
      d = (double)p->ev[j];
      fwrite(&d, sizeof(d), 1, fp);
      d = (double)p->p[j];
      fwrite(&d, sizeof(d), 1, fp);
    }

    // Singular parameters
    fwrite(&params->global, sizeof(params->global), 1, fp);
    for (u=0; u<grid->nd; ++u) {
      d = (double)params->g[u];
      fwrite(&d, sizeof(d), 1, fp);
    }
    d = (double)params->kappa;
    fwrite(&d, sizeof(d), 1, fp);
    d = (double)params->db;
    fwrite(&d, sizeof(d), 1, fp);
    d = (double)params->V_b;
    fwrite(&d, sizeof(d), 1, fp);
    fwrite(&params->shearmodel, sizeof(params->shearmodel), 1, fp);

    // Walls
    fwrite(&params->nw, sizeof(params->nw), 1, fp); // No. of walls
    for (j=0; j<params->nw; ++j) {

      // Wall mode
      d = (double)params->wmode[j];
      fwrite(&d, sizeof(d), 1, fp);

      // Wall normal
      d = (double)host_w_nx[j].x;
      fwrite(&d, sizeof(d), 1, fp);
      d = (double)host_w_nx[j].y;
      fwrite(&d, sizeof(d), 1, fp);
      d = (double)host_w_nx[j].z;
      fwrite(&d, sizeof(d), 1, fp);

      d = (double)host_w_nx[j].w; 	// Wall position
      fwrite(&d, sizeof(d), 1, fp);
      d = (double)host_w_mvfd[j].x;	// Wall mass
      fwrite(&d, sizeof(d), 1, fp);
      d = (double)host_w_mvfd[j].y;	// Wall velocity
      fwrite(&d, sizeof(d), 1, fp);
      d = (double)host_w_mvfd[j].z;	// Wall force
      fwrite(&d, sizeof(d), 1, fp);
      d = (double)host_w_mvfd[j].w;	// Wall deviatoric stress
      fwrite(&d, sizeof(d), 1, fp);
    }
    fwrite(&params->periodic, sizeof(params->periodic), 1, fp);
    d = (double)params->gamma_wn;
    fwrite(&d, sizeof(d), 1, fp);
    d = (double)params->gamma_wt;
    fwrite(&d, sizeof(d), 1, fp);
    d = (double)params->gamma_wr;
    fwrite(&d, sizeof(d), 1, fp);

    // Write bond pair values
    for (j=0; j<p->np; ++j) {
      fwrite(&host_bonds[j].x, sizeof(unsigned int), 1, fp);
      fwrite(&host_bonds[j].y, sizeof(unsigned int), 1, fp);
      fwrite(&host_bonds[j].z, sizeof(unsigned int), 1, fp);
      fwrite(&host_bonds[j].w, sizeof(unsigned int), 1, fp);
    }

  } else {
    fprintf(stderr, "Error: Chosen floating-point precision is incompatible with the data file format.\n");
    exit(1);
  }

  fclose(fp);

  // This function returns 0 if it ended without problems,
  // and 1 if there were problems opening the target file.
  return 0;
} // End of fwritebin(...)


