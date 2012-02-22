#include <stdio.h>  // Standard library functions for file input and output
#include <stdlib.h> // Functions involving memory allocation, process control, conversions and others
#include <unistd.h> // UNIX only: For getcwd
#include <string.h> // For strerror and strcmp
#include <cuda.h>
#include "datatypes.h"


// Write host variables to target binary file
int fwritebin(char *target, Particles *p, float4 *host_x, float4 *host_vel, float4 *host_angvel, 
    float4 *host_force, float4 *host_torque, 
    uint4 *host_bonds,
    Grid *grid, Time *time, 
    Params *params,
    float4 *host_w_nx, float4 *host_w_mvfd)
{

  FILE *fp;
  unsigned int u;
  unsigned int j;

  if ((fp = fopen(target,"wb")) == NULL) {
    printf("Could create output binary file. Bye.\n");
    return 1; // Return unsuccessful exit status
  }

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
    fwrite(&host_x[j].x, sizeof(float), 1, fp);
    fwrite(&host_vel[j].x, sizeof(float), 1, fp);
    fwrite(&host_angvel[j].x, sizeof(float), 1, fp);
    fwrite(&host_force[j].x, sizeof(float), 1, fp);
    fwrite(&host_torque[j].x, sizeof(float), 1, fp);

    // y-axis
    fwrite(&host_x[j].y, sizeof(float), 1, fp);
    fwrite(&host_vel[j].y, sizeof(float), 1, fp);
    fwrite(&host_angvel[j].y, sizeof(float), 1, fp);
    fwrite(&host_force[j].y, sizeof(float), 1, fp);
    fwrite(&host_torque[j].y, sizeof(float), 1, fp);

    // z-axis
    fwrite(&host_x[j].z, sizeof(float), 1, fp);
    fwrite(&host_vel[j].z, sizeof(float), 1, fp);
    fwrite(&host_angvel[j].z, sizeof(float), 1, fp);
    fwrite(&host_force[j].z, sizeof(float), 1, fp);
    fwrite(&host_torque[j].z, sizeof(float), 1, fp);
  } 

  // Individual particle values
  for (j=0; j<p->np; ++j) {
    fwrite(&host_vel[j].w, sizeof(float), 1, fp);
    fwrite(&host_x[j].w, sizeof(float), 1, fp);
    fwrite(&p->radius[j], sizeof(p->radius[j]), 1, fp);
    fwrite(&p->rho[j], sizeof(p->rho[j]), 1, fp);
    fwrite(&p->k_n[j], sizeof(p->k_n[j]), 1, fp);
    fwrite(&p->k_s[j], sizeof(p->k_s[j]), 1, fp);
    fwrite(&p->k_r[j], sizeof(p->k_r[j]), 1, fp);
    fwrite(&p->gamma_s[j], sizeof(p->gamma_s[j]), 1, fp);
    fwrite(&p->gamma_r[j], sizeof(p->gamma_r[j]), 1, fp);
    fwrite(&p->mu_s[j], sizeof(p->mu_s[j]), 1, fp);
    fwrite(&p->mu_r[j], sizeof(p->mu_r[j]), 1, fp);
    fwrite(&p->C[j], sizeof(p->C[j]), 1, fp);
    fwrite(&p->E[j], sizeof(p->E[j]), 1, fp);
    fwrite(&p->K[j], sizeof(p->K[j]), 1, fp);
    fwrite(&p->nu[j], sizeof(p->nu[j]), 1, fp);
    fwrite(&p->es_dot[j], sizeof(p->es_dot[j]), 1, fp);
    fwrite(&p->es[j], sizeof(p->es[j]), 1, fp);
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
    // Wall normal
    fwrite(&host_w_nx[j].x, sizeof(float), 1, fp);
    fwrite(&host_w_nx[j].y, sizeof(float), 1, fp);
    fwrite(&host_w_nx[j].z, sizeof(float), 1, fp);

    fwrite(&host_w_nx[j].w, sizeof(float), 1, fp);   // Wall position
    fwrite(&host_w_mvfd[j].x, sizeof(float), 1, fp); // Wall mass
    fwrite(&host_w_mvfd[j].y, sizeof(float), 1, fp); // Wall velocity
    fwrite(&host_w_mvfd[j].z, sizeof(float), 1, fp); // Wall force
    fwrite(&host_w_mvfd[j].w, sizeof(float), 1, fp); // Wall deviatoric stress
  }
  fwrite(&params->periodic, sizeof(params->periodic), 1, fp);

  // Write bond pair values
  for (j=0; j<p->np; ++j) {
    fwrite(&host_bonds[j].x, sizeof(unsigned int), 1, fp);
    fwrite(&host_bonds[j].y, sizeof(unsigned int), 1, fp);
    fwrite(&host_bonds[j].z, sizeof(unsigned int), 1, fp);
    fwrite(&host_bonds[j].w, sizeof(unsigned int), 1, fp);
  }

  fclose(fp);

  // This function returns 0 if it ended without problems,
  // and 1 if there were problems opening the target file.
  return 0;
} // End of fwritebin(...)


