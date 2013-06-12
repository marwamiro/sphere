#ifndef LATTICEBOLTZMANN_CUH_
#define LATTICEBOLTZMANN_CUH_

#include "utility.h"

// Enable line below to perform lattice Boltzmann computations on the
// GPU, disable for CPU computation
//#define LBM_GPU

// latticeboltzmann.cuh
// Functions for solving the Navier-Stokes equations using the Lattice-Boltzmann
// method with D3Q19 stencils

// Calculate linear cell index from position (x,y,z)
// and fluid position vector (i).
// From A. Monitzer 2013
#ifdef LBM_GPU
__device__ 
#endif
unsigned int grid2index(
        unsigned int x, unsigned int y, unsigned int z,
        unsigned int i,
        unsigned int nx, unsigned int ny, unsigned int nz)
{
    return x + ((y + z*ny)*nx) + (nx*ny*nz*i);
}


// Equilibrium distribution
#ifdef LBM_GPU
__device__ 
#endif
Float feq(Float3 v, Float rho, Float3 e, Float w, Float dt, Float dx)
{
    // Monitzer 2010
    //return w*rho * (1.0 - 3.0/2.0 * dot(v,v) + 3.0*dot(e,v) +
            //9.0/2.0*dot(e,v)*dot(e,v));

    // Rinaldi 2012
    //return w*rho * (1.0 + 3.0*dot(e,v) + 9.0/2.0*dot(e,v)*dot(e,v)
            //- 3.0/2.0*dot(v,v));

    // Hecht 2010
    //Float c2_s = 1.0/sqrt(3);   // D3Q19 lattice speed of sound
    //c2_s *= c2_s;
    //return w*rho * (1.0 + dot(e,v)/c2_s 
            //+ (dot(e,v)*dot(e,v))/(2.0*c2_s*c2_s)
            //- dot(v,v)*dot(v,v)/(2.0*c2_s));

    // Chirila 2010
    //Float c2 = 1.0*grid.num[0]/devC_dt;
    //Float c2 = 1.0/sqrt(3.0);
    //c2 *= c2;   // Propagation speed on the lattice
    //return w*rho * (1.0 + 3.0*dot(e,v)/c2
            //+ 9.0/2.0 * dot(e,v)*dot(e,v)/(c2*c2)
            //- 3.0/2.0 * dot(v,v)/c2);

    // Habich 2011
    Float c2 = dx/dt * dx/dt;
    return rho*w
        * (1.0 + 3.0/c2*dot(e,v)
                + 9.0/(2.0*c2*c2) * dot(e,v)*dot(e,v)
                - 3.0/(2.0*c2) * dot(v,v));
}

// Collision operator
// Bhatnagar-Gross-Krook approximation (BGK), Thurey (2003).
#ifdef LBM_GPU
__device__
#endif
Float bgk(
        Float dt,
        Float dx,
        Float f,
        Float tau, 
        Float3 v,
        Float rho,
        Float3 e,
        Float w,
        Float3 extF)
{
    //Float feqval = feq(v, rho, e, w);
    //printf("feq(v, rho=%f, e, w=%f) = %f\n", rho, w, feqval);

    // Monitzer 2008
    //return dt / tau * (f - feq(v, rho, e, w))
        //- (1.0 - 1.0/(2.0*tau)) * 3.0/w * dot(extF, e);
    //return dt / tau * (f - feq(v, rho, e, w))
     //   + (2.0*tau - 1.0/(2.0*tau)) * 3.0/w * dot(extF, e);
    //return dt/tau * (f - feq(v, rho, e, w))
        //+ (2.0*tau - 1.0/(2.0*tau)) * 3.0/w * dot(extF, e);
    
    // Monitzer 2010
    //return dt/tau*(f - feq(v, rho, e, w))
        //+ (2.0*tau - 1.0)/(2.0*tau) * 3.0/w * dot(extF, e);

    // Rinaldi 2012
    //return 1.0/tau * (feq(v, rho, e, w) - f);

    // Habich 2011
    return (f - feq(v, rho, e, w, dt, dx))/tau
        + (2.0*tau - 1.0)/(2.0*tau) * 3.0/w * dot(extF, e);
}

// Initialize the fluid distributions on the base of the densities provided
#ifdef LBM_GPU
__global__ void initFluid(
        Float4* dev_v_rho,
        Float* dev_f)
#else
void initFluid(
        Float4* dev_v_rho,
        Float* dev_f,
        unsigned int nx,
        unsigned int ny,
        unsigned int nz)
#endif

{
#ifdef LBM_GPU
    // 3D thread index
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;

    // Grid dimensions
    const unsigned int nx = devC_grid.num[0];
    const unsigned int ny = devC_grid.num[1];
    const unsigned int nz = devC_grid.num[2];
#else
    for (unsigned int z = 0; z<nz; z++) {
        for (unsigned int y = 0; y<ny; y++) {
            for (unsigned int x = 0; x<nx; x++) {
#endif

    // Check that we are not outside the fluid grid
    if (x < nx && y < ny && z < nz) {

        // 1D thread index
        const unsigned int tidx = x + nx*y + nx*ny*z;

        // Read velocity and density, zero velocity
#ifdef LBM_GPU
        __syncthreads();
#endif
        Float4 v_rho = dev_v_rho[tidx];
        v_rho = MAKE_FLOAT4(0.0, 0.0, 0.0, v_rho.w);

        // Set values to equilibrium distribution (f_i = w_i * rho_0)
#ifdef LBM_GPU
        __syncthreads();
#endif
        dev_v_rho[tidx] = v_rho;
        dev_f[grid2index(x,y,z,0,nx,ny,nz)]  = 1.0/3.0  * v_rho.w;
        dev_f[grid2index(x,y,z,1,nx,ny,nz)]  = 1.0/18.0 * v_rho.w;
        dev_f[grid2index(x,y,z,2,nx,ny,nz)]  = 1.0/18.0 * v_rho.w;
        dev_f[grid2index(x,y,z,3,nx,ny,nz)]  = 1.0/18.0 * v_rho.w;
        dev_f[grid2index(x,y,z,4,nx,ny,nz)]  = 1.0/18.0 * v_rho.w;
        dev_f[grid2index(x,y,z,5,nx,ny,nz)]  = 1.0/18.0 * v_rho.w;
        dev_f[grid2index(x,y,z,6,nx,ny,nz)]  = 1.0/18.0 * v_rho.w;
        dev_f[grid2index(x,y,z,7,nx,ny,nz)]  = 1.0/36.0 * v_rho.w;
        dev_f[grid2index(x,y,z,8,nx,ny,nz)]  = 1.0/36.0 * v_rho.w;
        dev_f[grid2index(x,y,z,9,nx,ny,nz)]  = 1.0/36.0 * v_rho.w;
        dev_f[grid2index(x,y,z,10,nx,ny,nz)] = 1.0/36.0 * v_rho.w;
        dev_f[grid2index(x,y,z,11,nx,ny,nz)] = 1.0/36.0 * v_rho.w;
        dev_f[grid2index(x,y,z,12,nx,ny,nz)] = 1.0/36.0 * v_rho.w;
        dev_f[grid2index(x,y,z,13,nx,ny,nz)] = 1.0/36.0 * v_rho.w;
        dev_f[grid2index(x,y,z,14,nx,ny,nz)] = 1.0/36.0 * v_rho.w;
        dev_f[grid2index(x,y,z,15,nx,ny,nz)] = 1.0/36.0 * v_rho.w;
        dev_f[grid2index(x,y,z,16,nx,ny,nz)] = 1.0/36.0 * v_rho.w;
        dev_f[grid2index(x,y,z,17,nx,ny,nz)] = 1.0/36.0 * v_rho.w;
        dev_f[grid2index(x,y,z,18,nx,ny,nz)] = 1.0/36.0 * v_rho.w;
    }
#ifndef LBM_GPU
            }}}
#endif
}

// Combined streaming and collision step with particle coupling and optional
// periodic boundaries. Derived from A. Monitzer 2013
#ifdef LBM_GPU
__global__ void latticeBoltzmannD3Q19(
        Float* dev_f,
        Float* dev_f_new,
        Float4* dev_v_rho,        // fluid velocities and densities
        unsigned int* dev_cellStart, // first particle in cells
        unsigned int* dev_cellEnd,   // last particle in cells
        Float4* dev_x_sorted,   // particle positions + radii
        Float4* dev_vel_sorted, // particle velocities + fixvel
        Float4* dev_force,
        unsigned int* dev_gridParticleIndex)
#else
void latticeBoltzmannD3Q19(
        Float* dev_f,
        Float* dev_f_new,
        Float4* dev_v_rho,        // fluid velocities and densities
        Float devC_dt,
        Grid& grid,
        Params& params)

#endif
{
#ifdef LBM_GPU
    // 3D thread index
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;

    // Grid dimensions
    const unsigned int nx = devC_grid.num[0];
    const unsigned int ny = devC_grid.num[1];
    const unsigned int nz = devC_grid.num[2];
#else
    // Grid dimensions
    const unsigned int nx = grid.num[0];
    const unsigned int ny = grid.num[1];
    const unsigned int nz = grid.num[2];

    for (unsigned int z = 0; z<nz; z++) {
        for (unsigned int y = 0; y<ny; y++) {
            for (unsigned int x = 0; x<nx; x++) {
#endif

    // Check that we are not outside the fluid grid
    if (x < nx && y < ny && z < nz) {

        // 1D thread index
        const unsigned int tidx = x + nx*y + nx*ny*z;
        //printf("(x,y,x) = (%d,%d,%d), tidx = %d\n", x, y, z, tidx);

        // Load the fluid distribution into local registers
#ifdef LBM_GPU
        __syncthreads();
#endif
        Float f_0  = dev_f[grid2index(x,y,z,0,nx,ny,nz)];
        Float f_1  = dev_f[grid2index(x,y,z,1,nx,ny,nz)];
        Float f_2  = dev_f[grid2index(x,y,z,2,nx,ny,nz)];
        Float f_3  = dev_f[grid2index(x,y,z,3,nx,ny,nz)];
        Float f_4  = dev_f[grid2index(x,y,z,4,nx,ny,nz)];
        Float f_5  = dev_f[grid2index(x,y,z,5,nx,ny,nz)];
        Float f_6  = dev_f[grid2index(x,y,z,6,nx,ny,nz)];
        Float f_7  = dev_f[grid2index(x,y,z,7,nx,ny,nz)];
        Float f_8  = dev_f[grid2index(x,y,z,8,nx,ny,nz)];
        Float f_9  = dev_f[grid2index(x,y,z,9,nx,ny,nz)];
        Float f_10 = dev_f[grid2index(x,y,z,10,nx,ny,nz)];
        Float f_11 = dev_f[grid2index(x,y,z,11,nx,ny,nz)];
        Float f_12 = dev_f[grid2index(x,y,z,12,nx,ny,nz)];
        Float f_13 = dev_f[grid2index(x,y,z,13,nx,ny,nz)];
        Float f_14 = dev_f[grid2index(x,y,z,14,nx,ny,nz)];
        Float f_15 = dev_f[grid2index(x,y,z,15,nx,ny,nz)];
        Float f_16 = dev_f[grid2index(x,y,z,16,nx,ny,nz)];
        Float f_17 = dev_f[grid2index(x,y,z,17,nx,ny,nz)];
        Float f_18 = dev_f[grid2index(x,y,z,18,nx,ny,nz)];

        // Directional vectors to each lattice-velocity in D3Q19
        // Zero velocity: i = 0
        // Faces: i = 1..6
        // Edges: i = 7..18
        const Float3 e_0  = MAKE_FLOAT3( 0.0, 0.0, 0.0); // zero vel.
        const Float3 e_1  = MAKE_FLOAT3( 1.0, 0.0, 0.0); // face: +x
        const Float3 e_2  = MAKE_FLOAT3(-1.0, 0.0, 0.0); // face: -x
        const Float3 e_3  = MAKE_FLOAT3( 0.0, 1.0, 0.0); // face: +y
        const Float3 e_4  = MAKE_FLOAT3( 0.0,-1.0, 0.0); // face: -y
        const Float3 e_5  = MAKE_FLOAT3( 0.0, 0.0, 1.0); // face: +z
        const Float3 e_6  = MAKE_FLOAT3( 0.0, 0.0,-1.0); // face: -z
        const Float3 e_7  = MAKE_FLOAT3( 1.0, 1.0, 0.0); // edge: +x,+y
        const Float3 e_8  = MAKE_FLOAT3(-1.0,-1.0, 0.0); // edge: -x,-y
        const Float3 e_9  = MAKE_FLOAT3(-1.0, 1.0, 0.0); // edge: -x,+y
        const Float3 e_10 = MAKE_FLOAT3( 1.0,-1.0, 0.0); // edge: +x,-y
        const Float3 e_11 = MAKE_FLOAT3( 1.0, 0.0, 1.0); // edge: +x,+z
        const Float3 e_12 = MAKE_FLOAT3(-1.0, 0.0,-1.0); // edge: -x,-z
        const Float3 e_13 = MAKE_FLOAT3( 0.0, 1.0, 1.0); // edge: +y,+z
        const Float3 e_14 = MAKE_FLOAT3( 0.0,-1.0,-1.0); // edge: -y,-z
        const Float3 e_15 = MAKE_FLOAT3(-1.0, 0.0, 1.0); // edge: -x,+z
        const Float3 e_16 = MAKE_FLOAT3( 1.0, 0.0,-1.0); // edge: +x,-z
        const Float3 e_17 = MAKE_FLOAT3( 0.0,-1.0, 1.0); // edge: -y,+z
        const Float3 e_18 = MAKE_FLOAT3( 0.0, 1.0,-1.0); // edge: +y,-z


        //// Calculate the cell's macroproperties

        // Fluid density (rho = sum(f_i))
        const Float rho = f_0 + f_1 + f_2 + f_3 + f_4 + f_5 + f_6 + f_7 + f_8 +
            f_9 + f_10 + f_11 + f_12 + f_13 + f_14 + f_15 + f_16 + f_17 + f_18;

        // Fluid velocity (v = sum(f_i*e_i)/rho)
        const Float3 v = (f_0*e_0 + f_1*e_1 + f_2*e_2 + f_3*e_3 + f_4*e_4 +
                f_5*e_5 + f_6*e_6 + f_7*e_7 + f_8*e_8 + f_9*e_9 + f_10*e_10 +
                f_11*e_11 + f_12*e_12 + f_13*e_13 + f_14*e_14 + f_15*e_15 +
                f_16*e_16 + f_17*e_17 + f_18*e_18) / rho;

        //// Calculate the force transferred from the particles to the fluid
        /*Float3 f_particle;
        Float3 f_particles = MAKE_FLOAT3(0.0, 0.0, 0.0);
        Float4 x_particle4; // particle position + radius
        Float  r_particle;  // radius
        Float4 v_particle4; // particle velocity + fixvel
        Float3 v_particle;  // particle velocity


        // Lowest particle index in cell
        unsigned int startIdx = dev_cellStart[tidx];
        //unsigned int orig_idx;

        // Make sure cell is not empty
        if (startIdx != 0xffffffff) {

            // Highest particle index in cell + 1
            unsigned int endIdx = dev_cellEnd[tidx];
            
            // Iterate over cell particles
            for (unsigned int idx = startIdx; idx<endIdx; ++idx) {

                // Read particle radius and velocity
                __syncthreads();
                x_particle4 = dev_x_sorted[idx];
                v_particle4 = dev_vel_sorted[idx];

                r_particle = x_particle4.w;
                v_particle = MAKE_FLOAT3(
                        v_particle4.x,
                        v_particle4.y,
                        v_particle4.z);

                // Aerodynamic drag
                f_particle = (v - v_particle) * r_particle*r_particle;

                // Add the drag force to the sum of forces in the cell
                f_particles += f_particle;

                // The particle experiences the opposite drag force
                // Causes "unspecified launch failure"
                //orig_idx = dev_gridParticleIndex[idx];
                //dev_force[orig_idx] = MAKE_FLOAT4(
                        //-f_particle.x,
                        //-f_particle.y,
                        //-f_particle.z,
                        //0.0);
            }
        }

        // Scale the particle force 
        // 100: experimental value, depends on the grid size compared to the
        // particle size and the time step size
        f_particles *= 100.0 * rho * 6.0;
        */

#ifdef LBM_GPU
        Float dx = devC_grid.L[0]/devC_grid.num[0];

        // Fluid constant (Wei et al. 2004), nu: kinematic viscosity [Pa*s]
        //  nu = 1/6*(2*tau - 1) * dx * c
        //const Float tau = 0.5*(1.0 + 6.0*devC_params.nu);
        //const Float tau = 1.0/6.0*(2.0*devC_params.nu - 1.0) * dx*dx/devC_dt;
        const Float tau = (6.0*devC_params.nu * devC_dt/(dx*dx) + 1)/2.0;

        // Gravitational force (F = g * m)
        const Float3 f_gravity = MAKE_FLOAT3(
                devC_params.g[0]*dx*rho,
                devC_params.g[1] 
                * ((Float)devC_grid.L[1]/devC_grid.num[1]) * rho,
                devC_params.g[2] 
                * ((Float)devC_grid.L[2]/devC_grid.num[2]) * rho);
#else
        Float dx = grid.L[0]/grid.num[0];

        // Fluid constant (Wei et al. 2004), nu: kinematic viscosity [Pa*s]
        //const Float tau = 0.5*(1.0 + 6.0*params.nu);
        //const Float tau = 1.0/6.0*(2.0*params.nu - 1.0) * dx*dx/devC_dt;
        const Float tau = (6.0*params.nu * devC_dt/(dx*dx) + 1)/2.0;

        //if (tau <= 0.5) {
            //fprintf(stderr, "Error, tau <= 0.5\n");
            //exit(1);
        //}

        // Gravitational force (F = g * m)
        const Float3 f_gravity = MAKE_FLOAT3(
                params.g[0]*dx*rho,
                params.g[1] * ((Float)grid.L[1]/grid.num[1]) * rho,
                params.g[2] * ((Float)grid.L[2]/grid.num[2]) * rho);
#endif

        // The final external force
        //const Float3 f_ext = f_particles + f_gravity;
        const Float3 f_ext = f_gravity;
        //const Float3 f_ext = MAKE_FLOAT3(0.0, 0.0, 0.0);
        //printf("%d,%d,%d: f_ext = %f, %f, %f\n", x, y, z,
                //f_ext.x, f_ext.y, f_ext.z);

        //// Collide fluid
        // Weights corresponding to each e_i lattice-velocity in D3Q19, sum to 1.0
        f_0  -= bgk(devC_dt, dx, f_0,  tau, v, rho, e_0,  1.0/3.0,  f_ext);
        f_1  -= bgk(devC_dt, dx, f_1,  tau, v, rho, e_1,  1.0/18.0, f_ext);
        f_2  -= bgk(devC_dt, dx, f_2,  tau, v, rho, e_2,  1.0/18.0, f_ext);
        f_3  -= bgk(devC_dt, dx, f_3,  tau, v, rho, e_3,  1.0/18.0, f_ext);
        f_4  -= bgk(devC_dt, dx, f_4,  tau, v, rho, e_4,  1.0/18.0, f_ext);
        f_5  -= bgk(devC_dt, dx, f_5,  tau, v, rho, e_5,  1.0/18.0, f_ext);
        f_6  -= bgk(devC_dt, dx, f_6,  tau, v, rho, e_6,  1.0/18.0, f_ext);
        f_7  -= bgk(devC_dt, dx, f_7,  tau, v, rho, e_7,  1.0/36.0, f_ext);
        f_8  -= bgk(devC_dt, dx, f_8,  tau, v, rho, e_8,  1.0/36.0, f_ext);
        f_9  -= bgk(devC_dt, dx, f_9,  tau, v, rho, e_9,  1.0/36.0, f_ext);
        f_10 -= bgk(devC_dt, dx, f_10, tau, v, rho, e_10, 1.0/36.0, f_ext);
        f_11 -= bgk(devC_dt, dx, f_11, tau, v, rho, e_11, 1.0/36.0, f_ext);
        f_12 -= bgk(devC_dt, dx, f_12, tau, v, rho, e_12, 1.0/36.0, f_ext);
        f_13 -= bgk(devC_dt, dx, f_13, tau, v, rho, e_13, 1.0/36.0, f_ext);
        f_14 -= bgk(devC_dt, dx, f_14, tau, v, rho, e_14, 1.0/36.0, f_ext);
        f_15 -= bgk(devC_dt, dx, f_15, tau, v, rho, e_15, 1.0/36.0, f_ext);
        f_16 -= bgk(devC_dt, dx, f_16, tau, v, rho, e_16, 1.0/36.0, f_ext);
        f_17 -= bgk(devC_dt, dx, f_17, tau, v, rho, e_17, 1.0/36.0, f_ext);
        f_18 -= bgk(devC_dt, dx, f_18, tau, v, rho, e_18, 1.0/36.0, f_ext);
        //Float bgkval = bgk(devC_dt, f_1,  tau, v, rho, e_1,  1.0/18.0, f_ext);
        //Float feqval = feq(v, rho, e_1, 1.0/18.0);
        //printf("%d,%d,%d: dt %f, f %f, feq %f, tau %f, v %fx%fx%f, rho %f, e %fx%fx%f, f_ext %fx%fx%f, bgk %f\n",
                //x, y, z, devC_dt, f_1, feqval, tau, v.x, v.y, v.z, rho,
                //e_1.x, e_1.y, e_1.z, f_ext.x, f_ext.y, f_ext.z, bgkval);


        //// Stream fluid
        // Lower and upper boundaries: bounceback, sides: periodic

        
        // There may be a write conflict due to bounce backs
#ifdef LBM_GPU
        __syncthreads();
#endif
        
        // Write fluid velocity and density to global memory
        dev_v_rho[tidx] = MAKE_FLOAT4(v.x, v.y, v.z, rho); 
        //dev_v_rho[tidx] = MAKE_FLOAT4(x, y, z, rho); 

        // Face 0
        dev_f_new[grid2index(x,y,z,0,nx,ny,nz)] = fmax(0.0, f_0);

        //*

        // Face 1 (+x): Bounce back
        if (x < nx-1)
            dev_f_new[grid2index(x+1,  y,  z,  1,nx,ny,nz)] = fmax(0.0, f_1);
        else
            dev_f_new[grid2index(  x,  y,  z,  2,nx,ny,nz)] = fmax(0.0, f_1);

        // Face 2 (-x): Bounce back
        if (x > 0)
            dev_f_new[grid2index(x-1,  y,  z,  2,nx,ny,nz)] = fmax(0.0, f_2);
        else
            dev_f_new[grid2index(  x,  y,  z,  1,nx,ny,nz)] = fmax(0.0, f_2);

        // Face 3 (+y): Bounce back
        if (y < ny-1)
            dev_f_new[grid2index(  x,y+1,  z,  3,nx,ny,nz)] = fmax(0.0, f_3);
        else
            dev_f_new[grid2index(  x,  y,  z,  4,nx,ny,nz)] = fmax(0.0, f_3);

        // Face 4 (-y): Bounce back
        if (y > 0)
            dev_f_new[grid2index(  x,y-1,  z,  4,nx,ny,nz)] = fmax(0.0, f_4);
        else
            dev_f_new[grid2index(  x,  y,  z,  3,nx,ny,nz)] = fmax(0.0, f_4);

        // Face 5 (+z): Bounce back
        if (z < nz-1)
            dev_f_new[grid2index(  x,  y,z+1,  5,nx,ny,nz)] = fmax(0.0, f_5);
        else
            dev_f_new[grid2index(  x,  y,  z,  6,nx,ny,nz)] = fmax(0.0, f_5);

        // Face 6 (-z): Bounce back
        if (z > 0)
            dev_f_new[grid2index(  x,  y,z-1,  6,nx,ny,nz)] = fmax(0.0, f_6);
        else
            dev_f_new[grid2index(  x,  y,  z,  5,nx,ny,nz)] = fmax(0.0, f_6);
        
        // Edge 7 (+x,+y): Bounce back
        if (x < nx-1 && y < ny-1)
            dev_f_new[grid2index(x+1,y+1,  z,  7,nx,ny,nz)] = fmax(0.0, f_7);
        else if (x < nx-1)
            dev_f_new[grid2index(x+1,  y,  z,  9,nx,ny,nz)] = fmax(0.0, f_7);
        else if (y < ny-1)
            dev_f_new[grid2index(  x,y+1,  z, 10,nx,ny,nz)] = fmax(0.0, f_7);
        else
            dev_f_new[grid2index(  x,  y,  z,  8,nx,ny,nz)] = fmax(0.0, f_7);

        // Edge 8 (-x,-y): Bounce back
        if (x > 0 && y > 0)
            dev_f_new[grid2index(x-1,y-1,  z,  8,nx,ny,nz)] = fmax(0.0, f_8);
        else if (x > 0)
            dev_f_new[grid2index(x-1,  y,  z,  9,nx,ny,nz)] = fmax(0.0, f_8);
        else if (y > 0)
            dev_f_new[grid2index(  x,y-1,  z, 10,nx,ny,nz)] = fmax(0.0, f_8);
        else
            dev_f_new[grid2index(  x,  y,  z,  7,nx,ny,nz)] = fmax(0.0, f_8);

        // Edge 9 (-x,+y): Bounce back
        if (x > 0 && y < ny-1)
            dev_f_new[grid2index(x-1,y+1,  z,  9,nx,ny,nz)] = fmax(0.0, f_9);
        else if (x > 0)
            dev_f_new[grid2index(x-1,  y,  z,  8,nx,ny,nz)] = fmax(0.0, f_9);
        else if (y < ny-1)
            dev_f_new[grid2index(  x,y+1,  z,  7,nx,ny,nz)] = fmax(0.0, f_9);
        else
            dev_f_new[grid2index(  x,  y,  z, 10,nx,ny,nz)] = fmax(0.0, f_9);

        // Edge 10 (+x,-y): Bounce back
        if (x < nx-1 && y > 0)
            dev_f_new[grid2index(x+1,y-1,  z, 10,nx,ny,nz)] = fmax(0.0, f_10);
        else if (x < nx-1)
            dev_f_new[grid2index(x+1,  y,  z,  8,nx,ny,nz)] = fmax(0.0, f_10);
        else if (y > 0)
            dev_f_new[grid2index(  x,y-1,  z,  7,nx,ny,nz)] = fmax(0.0, f_10);
        else
            dev_f_new[grid2index(  x,  y,  z,  9,nx,ny,nz)] = fmax(0.0, f_10);

        // Edge 11 (+x,+z): Bounce back
        if (x < nx-1 && z < nz-1)
            dev_f_new[grid2index(x+1,  y,z+1, 11,nx,ny,nz)] = fmax(0.0, f_11);
        else if (x < nx-1)
            dev_f_new[grid2index(x+1,  y,  z, 16,nx,ny,nz)] = fmax(0.0, f_11);
        else if (z < nz-1)
            dev_f_new[grid2index(  x,  y,z+1, 15,nx,ny,nz)] = fmax(0.0, f_11);
        else
            dev_f_new[grid2index(  x,  y,  z, 12,nx,ny,nz)] = fmax(0.0, f_11);

        // Edge 12 (-x,-z): Bounce back
        if (x > 0 && z > 0)
            dev_f_new[grid2index(x-1,  y,z-1, 12,nx,ny,nz)] = fmax(0.0, f_12);
        else if (x > 0)
            dev_f_new[grid2index(x-1,  y,  z, 15,nx,ny,nz)] = fmax(0.0, f_12);
        else if (z > 0)
            dev_f_new[grid2index(  x,  y,z-1, 16,nx,ny,nz)] = fmax(0.0, f_12);
        else
            dev_f_new[grid2index(  x,  y,  z, 11,nx,ny,nz)] = fmax(0.0, f_12);

        // Edge 13 (+y,+z): Bounce back
        if (y < ny-1 && z < nz-1)
            dev_f_new[grid2index(  x,y+1,z+1, 13,nx,ny,nz)] = fmax(0.0, f_13);
        else if (y < ny-1)
            dev_f_new[grid2index(  x,y+1,  z, 18,nx,ny,nz)] = fmax(0.0, f_13);
        else if (z < nz-1)
            dev_f_new[grid2index(  x,  y,z+1, 17,nx,ny,nz)] = fmax(0.0, f_13);
        else
            dev_f_new[grid2index(  x,  y,  z, 14,nx,ny,nz)] = fmax(0.0, f_13);

        // Edge 14 (-y,-z): Bounce back
        if (y > 0 && z > 0)
            dev_f_new[grid2index(  x,y-1,z-1, 14,nx,ny,nz)] = fmax(0.0, f_14);
        else if (y > 0)
            dev_f_new[grid2index(  x,y-1,  z, 17,nx,ny,nz)] = fmax(0.0, f_14);
        else if (z > 0)
            dev_f_new[grid2index(  x,  y,z-1, 18,nx,ny,nz)] = fmax(0.0, f_14);
        else
            dev_f_new[grid2index(  x,  y,  z, 13,nx,ny,nz)] = fmax(0.0, f_14);

        // Edge 15 (-x,+z): Bounce back
        if (x > 0 && z < nz-1)
            dev_f_new[grid2index(x-1,  y,z+1, 15,nx,ny,nz)] = fmax(0.0, f_15);
        else if (x > 0)
            dev_f_new[grid2index(x-1,  y,  z, 12,nx,ny,nz)] = fmax(0.0, f_15);
        else if (z < nz-1)
            dev_f_new[grid2index(  x,  y,z+1, 11,nx,ny,nz)] = fmax(0.0, f_15);
        else
            dev_f_new[grid2index(  x,  y,  z, 16,nx,ny,nz)] = fmax(0.0, f_15);

        // Edge 16 (+x,-z)
        if (x < nx-1 && z > 0)
            dev_f_new[grid2index(x+1,  y,z-1, 16,nx,ny,nz)] = fmax(0.0, f_16);
        else if (x < nx-1)
            dev_f_new[grid2index(x+1,  y,  z, 11,nx,ny,nz)] = fmax(0.0, f_16);
        else if (z > 0)
            dev_f_new[grid2index(  x,  y,z-1, 12,nx,ny,nz)] = fmax(0.0, f_16);
        else
            dev_f_new[grid2index(  x,  y,  z, 15,nx,ny,nz)] = fmax(0.0, f_16);

        // Edge 17 (-y,+z)
        if (y > 0 && z < nz-1)
            dev_f_new[grid2index(  x,y-1,z+1, 17,nx,ny,nz)] = fmax(0.0, f_17);
        else if (y > 0)
            dev_f_new[grid2index(  x,y-1,  z, 14,nx,ny,nz)] = fmax(0.0, f_17);
        else if (z < nz-1)
            dev_f_new[grid2index(  x,  y,z+1, 13,nx,ny,nz)] = fmax(0.0, f_17);
        else
            dev_f_new[grid2index(  x,  y,  z, 18,nx,ny,nz)] = fmax(0.0, f_17);

        // Edge 18 (+y,-z)
        if (y < ny-1 && z > 0)
            dev_f_new[grid2index(  x,y+1,z-1, 18,nx,ny,nz)] = fmax(0.0, f_18);
        else if (y < ny-1)
            dev_f_new[grid2index(  x,y+1,  z, 13,nx,ny,nz)] = fmax(0.0, f_18);
        else if (z > 0)
            dev_f_new[grid2index(  x,  y,z-1, 14,nx,ny,nz)] = fmax(0.0, f_18);
        else
            dev_f_new[grid2index(  x,  y,  z, 17,nx,ny,nz)] = fmax(0.0, f_18);
        
        // */
        
        /*

        // Face 1 (+x): Periodic
        if (x < nx-1) // not at boundary
            dev_f_new[grid2index( x+1,   y,   z,  1,nx,ny,nz)] = fmax(0.0, f_1);
        else        // at boundary
            dev_f_new[grid2index(   0,   y,   z,  1,nx,ny,nz)] = fmax(0.0, f_1);

        // Face 2 (-x): Periodic
        if (x > 0)  // not at boundary
            dev_f_new[grid2index( x-1,   y,   z,  2,nx,ny,nz)] = fmax(0.0, f_2);
        else        // at boundary
            dev_f_new[grid2index(nx-1,   y,   z,  2,nx,ny,nz)] = fmax(0.0, f_2);
        
        // Face 3 (+y): Periodic
        if (y < ny-1) // not at boundary
            dev_f_new[grid2index(   x, y+1,   z,  3,nx,ny,nz)] = fmax(0.0, f_3);
        else        // at boundary
            dev_f_new[grid2index(   x,   0,   z,  3,nx,ny,nz)] = fmax(0.0, f_3);

        // Face 4 (-y): Periodic
        if (y > 0)  // not at boundary
            dev_f_new[grid2index(   x, y-1,   z,  4,nx,ny,nz)] = fmax(0.0, f_4);
        else        // at boundary
            dev_f_new[grid2index(   x,ny-1,   z,  4,nx,ny,nz)] = fmax(0.0, f_4);

        // Face 5 (+z): Bounce back, free slip
        if (z < nz-1) // not at boundary
            dev_f_new[grid2index(   x,   y, z+1,  5,nx,ny,nz)] = fmax(0.0, f_5);
        else        // at boundary
            dev_f_new[grid2index(   x,   y,   z,  6,nx,ny,nz)] = fmax(0.0, f_5);

        // Face 6 (-z): Bounce back, free slip
        if (z > 0)  // not at boundary
            dev_f_new[grid2index(   x,   y, z-1,  6,nx,ny,nz)] = fmax(0.0, f_6);
        else        // at boundary
            dev_f_new[grid2index(   x,   y,   z,  5,nx,ny,nz)] = fmax(0.0, f_6);
        
        // Edge 7 (+x,+y): Periodic
        if (x < nx-1 && y < ny-1)   // not at boundary
            dev_f_new[grid2index( x+1, y+1,   z,  7,nx,ny,nz)] = fmax(0.0, f_7);
        else if (x < nx-1)  // at +y boundary
            dev_f_new[grid2index( x+1,   0,   z,  7,nx,ny,nz)] = fmax(0.0, f_7);
        else if (y < ny-1)  // at +x boundary
            dev_f_new[grid2index(   0, y+1,   z,  7,nx,ny,nz)] = fmax(0.0, f_7);
        else    // at +x+y boundary
            dev_f_new[grid2index(   0,   0,   z,  7,nx,ny,nz)] = fmax(0.0, f_7);

        // Edge 8 (-x,-y): Periodic
        if (x > 0 && y > 0) // not at boundary
            dev_f_new[grid2index( x-1, y-1,   z,  8,nx,ny,nz)] = fmax(0.0, f_8);
        else if (x > 0) // at -y boundary
            dev_f_new[grid2index( x-1,ny-1,   z,  8,nx,ny,nz)] = fmax(0.0, f_8);
        else if (y > 0) // at -x boundary
            dev_f_new[grid2index(nx-1, y-1,   z,  8,nx,ny,nz)] = fmax(0.0, f_8);
        else    // at -x-y boundary
            dev_f_new[grid2index(nx-1,ny-1,   z,  8,nx,ny,nz)] = fmax(0.0, f_8);

        // Edge 9 (-x,+y): Periodic
        if (x > 0 && y < ny-1)  // not at boundary
            dev_f_new[grid2index( x-1, y+1,   z,  9,nx,ny,nz)] = fmax(0.0, f_9);
        else if (x > 0)     // at +y boundary
            dev_f_new[grid2index( x-1,   0,   z,  9,nx,ny,nz)] = fmax(0.0, f_9);
        else if (y < ny-1)  // at -x boundary
            dev_f_new[grid2index(nx-1, y+1,   z,  9,nx,ny,nz)] = fmax(0.0, f_9);
        else    // at -x+y boundary
            dev_f_new[grid2index(nx-1,   0,   z,  9,nx,ny,nz)] = fmax(0.0, f_9);

        // Edge 10 (+x,-y): Periodic
        if (x < nx-1 && y > 0)  // not at boundary
            dev_f_new[grid2index( x+1, y-1,   z, 10,nx,ny,nz)] = fmax(0.0, f_10);
        else if (x < nx-1)  // at -y boundary
            dev_f_new[grid2index( x+1,ny-1,   z, 10,nx,ny,nz)] = fmax(0.0, f_10);
        else if (y > 0)     // at +x boundary
            dev_f_new[grid2index(   0, y-1,   z, 10,nx,ny,nz)] = fmax(0.0, f_10);
        else    // at +x-y boundary
            dev_f_new[grid2index(   0,ny-1,   z, 10,nx,ny,nz)] = fmax(0.0, f_10);

        // Edge 11 (+x,+z): Periodic & bounce-back (free slip)
        if (x < nx-1 && z < nz-1)   // not at boundary
            dev_f_new[grid2index( x+1,   y, z+1, 11,nx,ny,nz)] = fmax(0.0, f_11);
        else if (x < nx-1)  // at +z boundary
            dev_f_new[grid2index( x+1,   y,   0, 12,nx,ny,nz)] = fmax(0.0, f_11);
        else if (z < nz-1)  // at +x boundary
            dev_f_new[grid2index(   0,   y, z+1, 11,nx,ny,nz)] = fmax(0.0, f_11);
        else    // at +x+z boundary
            dev_f_new[grid2index(   0,   y,   0, 12,nx,ny,nz)] = fmax(0.0, f_11);

        // Edge 12 (-x,-z): Periodic & bounce back (free slip)
        if (x > 0 && z > 0) // not at boundary
            dev_f_new[grid2index( x-1,   y, z-1, 12,nx,ny,nz)] = fmax(0.0, f_12);
        else if (x > 0) // at -z boundary
            dev_f_new[grid2index( x-1,   y,nz-1, 11,nx,ny,nz)] = fmax(0.0, f_12);
        else if (z > 0) // at -x boundary
            dev_f_new[grid2index(nx-1,   y, z-1, 12,nx,ny,nz)] = fmax(0.0, f_12);
        else    // at -x-z boundary
            dev_f_new[grid2index(nx-1,   y,nz-1, 11,nx,ny,nz)] = fmax(0.0, f_12);

        // Edge 13 (+y,+z): Periodic & bounce-back (free slip)
        if (y < ny-1 && z < nz-1)   // not at boundary
            dev_f_new[grid2index(   x, y+1, z+1, 13,nx,ny,nz)] = fmax(0.0, f_13);
        else if (y < ny-1)  // at +z boundary
            dev_f_new[grid2index(   x, y+1,   0, 14,nx,ny,nz)] = fmax(0.0, f_13);
        else if (z < nz-1)  // at +y boundary
            dev_f_new[grid2index(   x,   0, z+1, 13,nx,ny,nz)] = fmax(0.0, f_13);
        else    // at +y+z boundary
            dev_f_new[grid2index(   x,   0,   0, 14,nx,ny,nz)] = fmax(0.0, f_13);

        // Edge 14 (-y,-z): Periodic & bounce-back (free slip)
        if (y > 0 && z > 0) // not at boundary
            dev_f_new[grid2index(   x, y-1, z-1, 14,nx,ny,nz)] = fmax(0.0, f_14);
        else if (y > 0) // at -z boundary
            dev_f_new[grid2index(   x, y-1,nz-1, 13,nx,ny,nz)] = fmax(0.0, f_14);
        else if (z > 0) // at -y boundary
            dev_f_new[grid2index(   x,ny-1, z-1, 14,nx,ny,nz)] = fmax(0.0, f_14);
        else    // at -y-z boundary
            dev_f_new[grid2index(   x,ny-1,nz-1, 13,nx,ny,nz)] = fmax(0.0, f_14);

        // Edge 15 (-x,+z): Periodic & bounce-back (free slip)
        if (x > 0 && z < nz-1)  // not at boundary
            dev_f_new[grid2index( x-1,   y, z+1, 15,nx,ny,nz)] = fmax(0.0, f_15);
        else if (x > 0)     // at +z boundary
            dev_f_new[grid2index( x-1,   y,   0, 16,nx,ny,nz)] = fmax(0.0, f_15);
        else if (z < nz-1)  // at -x boundary
            dev_f_new[grid2index(nx-1,   y, z+1, 15,nx,ny,nz)] = fmax(0.0, f_15);
        else    // at -x+z boundary
            dev_f_new[grid2index(nx-1,   y,   0, 16,nx,ny,nz)] = fmax(0.0, f_15);

        // Edge 16 (+x,-z): Periodic & bounce-back (free slip)
        if (x < nx-1 && z > 0)  // not at boundary
            dev_f_new[grid2index( x+1,   y, z-1, 16,nx,ny,nz)] = fmax(0.0, f_16);
        else if (x < nx-1)  // at -z boundary
            dev_f_new[grid2index( x+1,   y,nz-1, 15,nx,ny,nz)] = fmax(0.0, f_16);
        else if (z > 0)     // at +x boundary
            dev_f_new[grid2index(   0,   y, z-1, 16,nx,ny,nz)] = fmax(0.0, f_16);
        else    // at +x-z boundary
            dev_f_new[grid2index(   0,   y,nz-1, 15,nx,ny,nz)] = fmax(0.0, f_16);

        // Edge 17 (-y,+z): Periodic & bounce-back (free slip)
        if (y > 0 && z < nz-1)  // not at boundary
            dev_f_new[grid2index(   x, y-1, z+1, 17,nx,ny,nz)] = fmax(0.0, f_17);
        else if (y > 0)     // at +z boundary
            dev_f_new[grid2index(   x, y-1,   0, 18,nx,ny,nz)] = fmax(0.0, f_17);
        else if (z < nz-1)  // at -y boundary
            dev_f_new[grid2index(   x,ny-1, z+1, 17,nx,ny,nz)] = fmax(0.0, f_17);
        else    // at -y+z boundary
            dev_f_new[grid2index(   x,ny-1,   0, 18,nx,ny,nz)] = fmax(0.0, f_17);

        // Edge 18 (+y,-z): Periodic & bounce-back (free slip)
        if (y < ny-1 && z > 0)    // not at boundary
            dev_f_new[grid2index(   x, y+1, z-1, 18,nx,ny,nz)] = fmax(0.0, f_18);
        else if (y < ny-1)  // at -z boundary
            dev_f_new[grid2index(   x, y+1,   0, 17,nx,ny,nz)] = fmax(0.0, f_18);
        else if (z > 0)     // at +y boundary
            dev_f_new[grid2index(   x,   0, z-1, 18,nx,ny,nz)] = fmax(0.0, f_18);
        else    // at +y-z boundary
            dev_f_new[grid2index(   x,   0,   0, 17,nx,ny,nz)] = fmax(0.0, f_18);
        // */
        

    }
#ifndef LBM_GPU
            }}}
#endif
}

#endif
// vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
