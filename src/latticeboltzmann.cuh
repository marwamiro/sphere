#ifndef LATTICEBOLTZMANN_CUH_
#define LATTICEBOLTZMANN_CUH_

// latticeboltzmann.cuh
// Functions for solving the Navier-Stokes equations using the Lattice-Boltzmann
// method with D3Q19 stencils

// Calculate linear cell index from position (x,y,z)
// and fluid position vector (i).
// From A. Monitzer 2013
__device__ unsigned int grid2index(
        unsigned int x, unsigned int y, unsigned int z,
        unsigned int i)
{
    return x + ((y + z*devC_grid.num[1])*devC_grid.num[0])
        + (devC_grid.num[0]*devC_grid.num[1]*devC_grid.num[2]*i);
}

// Equilibrium distribution
__device__ Float feq(Float3 v, Float rho, Float3 e, Float omega)
{
    return omega*rho * (1.0 - 3.0/2.0 * dot(v,v) + 3.0*dot(e,v) +
            9.0/2.0*dot(e,v)*dot(e,v));
}

// Collision operator
// Bhatnagar-Gross-Krook approximation (BGK), Thurey (2003).
__device__ Float bgk(
        Float f,
        Float tau, 
        Float3 v,
        Float rho,
        Float3 e,
        Float omega,
        Float3 extF)
{
    return devC_dt / tau * (f - feq(v, rho, e, omega))
        - (1.0 - 1.0/(2.0*tau)) * 3.0/omega * dot(extF, e);
}

// Initialize the fluid distributions on the base of the densities provided
__global__ void initfluid(
        Float4* dev_v_rho,
        Float* dev_f)
{
    // 3D thread index
    const unsigned int z = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int x = blockDim.z * blockIdx.z + threadIdx.z;

    // Grid dimensions
    const unsigned int nx = devC_grid.num[0];
    const unsigned int ny = devC_grid.num[1];
    const unsigned int nz = devC_grid.num[2];

    // Check that we are not outside the fluid grid
    if (x < nx && y < ny && z < nz) {

        // 1D thread index
        const unsigned long int tidx = x + nx*y + nx*ny*z;

        // Read velocity and density, zero velocity
        Float4 v_rho = dev_v_rho[tidx];
        v_rho = MAKE_FLOAT4(0.0, 0.0, 0.0, v_rho.w);

        // Set values to equilibrium distribution (f_i = omega_i * rho_0)
        __syncthreads();
        dev_v_rho[tidx] = v_rho;
        dev_f[grid2index(x,y,z,0)]  = 1.0/3.0  * v_rho.w;
        dev_f[grid2index(x,y,z,1)]  = 1.0/18.0 * v_rho.w;
        dev_f[grid2index(x,y,z,2)]  = 1.0/18.0 * v_rho.w;
        dev_f[grid2index(x,y,z,3)]  = 1.0/18.0 * v_rho.w;
        dev_f[grid2index(x,y,z,4)]  = 1.0/18.0 * v_rho.w;
        dev_f[grid2index(x,y,z,5)]  = 1.0/18.0 * v_rho.w;
        dev_f[grid2index(x,y,z,6)]  = 1.0/18.0 * v_rho.w;
        dev_f[grid2index(x,y,z,7)]  = 1.0/36.0 * v_rho.w;
        dev_f[grid2index(x,y,z,8)]  = 1.0/36.0 * v_rho.w;
        dev_f[grid2index(x,y,z,9)]  = 1.0/36.0 * v_rho.w;
        dev_f[grid2index(x,y,z,10)] = 1.0/36.0 * v_rho.w;
        dev_f[grid2index(x,y,z,11)] = 1.0/36.0 * v_rho.w;
        dev_f[grid2index(x,y,z,12)] = 1.0/36.0 * v_rho.w;
        dev_f[grid2index(x,y,z,13)] = 1.0/36.0 * v_rho.w;
        dev_f[grid2index(x,y,z,14)] = 1.0/36.0 * v_rho.w;
        dev_f[grid2index(x,y,z,15)] = 1.0/36.0 * v_rho.w;
        dev_f[grid2index(x,y,z,16)] = 1.0/36.0 * v_rho.w;
        dev_f[grid2index(x,y,z,17)] = 1.0/36.0 * v_rho.w;
        dev_f[grid2index(x,y,z,18)] = 1.0/36.0 * v_rho.w;
    }
}

// Swap two arrays pointers
void swapFloatArrays(Float* arr1, Float* arr2)
{
    Float* tmp = arr1;
    arr1 = arr2;
    arr2 = tmp;
}

// Combined streaming and collision step with particle coupling and optional
// periodic boundaries. Derived from A. Monitzer 2013
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
{
    // 3D thread index
    const unsigned int z = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int x = blockDim.z * blockIdx.z + threadIdx.z;

    // Grid dimensions
    const unsigned int nx = devC_grid.num[0];
    const unsigned int ny = devC_grid.num[1];
    const unsigned int nz = devC_grid.num[2];

    // Check that we are not outside the fluid grid
    if (x < nx && y < ny && z < nz) {

        //printf("(x,y,x) = (%d,%d,%d), tidx = %d\n", x, y, z, tidx);

        // Load the fluid distribution into local registers
        __syncthreads();
        Float f_0  = dev_f[grid2index(x,y,z,0)];
        Float f_1  = dev_f[grid2index(x,y,z,1)];
        Float f_2  = dev_f[grid2index(x,y,z,2)];
        Float f_3  = dev_f[grid2index(x,y,z,3)];
        Float f_4  = dev_f[grid2index(x,y,z,4)];
        Float f_5  = dev_f[grid2index(x,y,z,5)];
        Float f_6  = dev_f[grid2index(x,y,z,6)];
        Float f_7  = dev_f[grid2index(x,y,z,7)];
        Float f_8  = dev_f[grid2index(x,y,z,8)];
        Float f_9  = dev_f[grid2index(x,y,z,9)];
        Float f_10 = dev_f[grid2index(x,y,z,10)];
        Float f_11 = dev_f[grid2index(x,y,z,11)];
        Float f_12 = dev_f[grid2index(x,y,z,12)];
        Float f_13 = dev_f[grid2index(x,y,z,13)];
        Float f_14 = dev_f[grid2index(x,y,z,14)];
        Float f_15 = dev_f[grid2index(x,y,z,15)];
        Float f_16 = dev_f[grid2index(x,y,z,16)];
        Float f_17 = dev_f[grid2index(x,y,z,17)];
        Float f_18 = dev_f[grid2index(x,y,z,18)];

        // Fluid constant (Wei et al. 2004), nu: kinematic viscosity [Pa*s]
        const Float tau = 0.5*(1.0 + 6.0*devC_params.nu);

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
        const Float3 e_9  = MAKE_FLOAT3( 1.0,-1.0, 0.0); // edge: -x,+y
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

        // Fluid velocity (v = sum(f_i/e_i)/rho)
        const Float3 v = (f_0/e_0 + f_1/e_1 + f_2/e_2 + f_3/e_3 + f_4/e_4 +
                f_5/e_5 + f_6/e_6 + f_7/e_7 + f_8/e_8 + f_9/e_9 + f_10/e_10 +
                f_11/e_11 + f_12/e_12 + f_13/e_13 + f_14/e_14 + f_15/e_15 +
                f_16/e_16 + f_17/e_17 + f_18/e_18) / rho;

        //// Calculate the force transferred from the particles to the fluid
        Float3 f_particle;
        Float3 f_particles = MAKE_FLOAT3(0.0, 0.0, 0.0);
        Float4 x_particle4; // particle position + radius
        Float  r_particle;  // radius
        Float4 v_particle4; // particle velocity + fixvel
        Float3 v_particle;  // particle velocity

        // 1D thread index
        const unsigned int tidx = x + nx*y + nx*ny*z;

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

        // Gravitational force (F = g * m)
        const Float3 f_gravity = MAKE_FLOAT3(
                devC_params.g[0],
                devC_params.g[1],
                devC_params.g[2])
            * (devC_grid.L[0]/devC_grid.num[0])
            * (devC_grid.L[1]/devC_grid.num[1])
            * (devC_grid.L[2]/devC_grid.num[2]) * rho;

        // The final external force
        const Float3 f_ext = f_particles + f_gravity;

        //// Collide fluid
        // Weights corresponding to each e_i lattice-velocity in D3Q19, sum to 1.0
        f_0  -= bgk(f_0,  tau, v, rho, e_0,  1.0/3.0,  f_ext);
        f_1  -= bgk(f_1,  tau, v, rho, e_1,  1.0/18.0, f_ext);
        f_2  -= bgk(f_2,  tau, v, rho, e_2,  1.0/18.0, f_ext);
        f_3  -= bgk(f_3,  tau, v, rho, e_3,  1.0/18.0, f_ext);
        f_4  -= bgk(f_4,  tau, v, rho, e_4,  1.0/18.0, f_ext);
        f_5  -= bgk(f_5,  tau, v, rho, e_5,  1.0/18.0, f_ext);
        f_6  -= bgk(f_6,  tau, v, rho, e_6,  1.0/18.0, f_ext);
        f_7  -= bgk(f_7,  tau, v, rho, e_7,  1.0/36.0, f_ext);
        f_8  -= bgk(f_8,  tau, v, rho, e_8,  1.0/36.0, f_ext);
        f_9  -= bgk(f_9,  tau, v, rho, e_9,  1.0/36.0, f_ext);
        f_10 -= bgk(f_10, tau, v, rho, e_10, 1.0/36.0, f_ext);
        f_11 -= bgk(f_11, tau, v, rho, e_11, 1.0/36.0, f_ext);
        f_12 -= bgk(f_12, tau, v, rho, e_12, 1.0/36.0, f_ext);
        f_13 -= bgk(f_13, tau, v, rho, e_13, 1.0/36.0, f_ext);
        f_14 -= bgk(f_14, tau, v, rho, e_14, 1.0/36.0, f_ext);
        f_15 -= bgk(f_15, tau, v, rho, e_15, 1.0/36.0, f_ext);
        f_16 -= bgk(f_16, tau, v, rho, e_16, 1.0/36.0, f_ext);
        f_17 -= bgk(f_17, tau, v, rho, e_17, 1.0/36.0, f_ext);
        f_18 -= bgk(f_18, tau, v, rho, e_18, 1.0/36.0, f_ext);


        //// Stream fluid
        // Lower and upper boundaries: bounceback, sides: periodic

        
        // There may be a write conflict due to bounce backs
        __syncthreads();
        
        // Write fluid velocity and density to global memory
        dev_v_rho[tidx] = MAKE_FLOAT4(v.x, v.y, v.z, rho); 

        // Face 0
        dev_f_new[grid2index(x,y,z,0)] = fmax(0.0, f_0);

        // Face 1 (+x): Periodic
        if (x < nx-1) // not at boundary
            dev_f_new[grid2index( x+1,   y,   z,  1)] = fmax(0.0, f_1);
        else        // at boundary
            dev_f_new[grid2index(   0,   y,   z,  1)] = fmax(0.0, f_1);

        // Face 2 (-x): Periodic
        if (x > 0)  // not at boundary
            dev_f_new[grid2index( x-1,   y,   z,  2)] = fmax(0.0, f_2);
        else        // at boundary
            dev_f_new[grid2index(nx-1,   y,   z,  2)] = fmax(0.0, f_2);
        
        // Face 3 (+y): Periodic
        if (y < ny-1) // not at boundary
            dev_f_new[grid2index(   x, y+1,   z,  3)] = fmax(0.0, f_3);
        else        // at boundary
            dev_f_new[grid2index(   x,   0,   z,  3)] = fmax(0.0, f_3);

        // Face 4 (-y): Periodic
        if (y > 0)  // not at boundary
            dev_f_new[grid2index(   x, y-1,   z,  4)] = fmax(0.0, f_4);
        else        // at boundary
            dev_f_new[grid2index(   x,ny-1,   z,  4)] = fmax(0.0, f_4);

        // Face 5 (+z): Bounce back, free slip
        if (z < nz-1) // not at boundary
            dev_f_new[grid2index(   x,   y, z+1,  5)] = fmax(0.0, f_5);
        else        // at boundary
            dev_f_new[grid2index(   x,   y,   z,  6)] = fmax(0.0, f_5);

        // Face 6 (-z): Bounce back, free slip
        if (z > 0)  // not at boundary
            dev_f_new[grid2index(   x,   y, z-1,  6)] = fmax(0.0, f_6);
        else        // at boundary
            dev_f_new[grid2index(   x,   y,   z,  5)] = fmax(0.0, f_6);

        // Edge 7 (+x,+y): Periodic
        if (x < nx-1 && y < ny-1)   // not at boundary
            dev_f_new[grid2index( x+1, y+1,   z,  7)] = fmax(0.0, f_7);
        else if (x < nx-1)  // at +y boundary
            dev_f_new[grid2index( x+1,   0,   z,  7)] = fmax(0.0, f_7);
        else if (y < ny-1)  // at +x boundary
            dev_f_new[grid2index(   0, y+1,   z,  7)] = fmax(0.0, f_7);
        else    // at +x+y boundary
            dev_f_new[grid2index(   0,   0,   z,  7)] = fmax(0.0, f_7);

        // Edge 8 (-x,-y): Periodic
        if (x > 0 && y > 0) // not at boundary
            dev_f_new[grid2index( x-1, y-1,   z,  8)] = fmax(0.0, f_8);
        else if (x > 0) // at -y boundary
            dev_f_new[grid2index( x-1,ny-1,   z,  8)] = fmax(0.0, f_8);
        else if (y > 0) // at -x boundary
            dev_f_new[grid2index(nx-1, y-1,   z,  8)] = fmax(0.0, f_8);
        else    // at -x-y boundary
            dev_f_new[grid2index(nx-1,ny-1,   z,  8)] = fmax(0.0, f_8);

        // Edge 9 (-x,+y): Periodic
        if (x > 0 && y < ny-1)  // not at boundary
            dev_f_new[grid2index( x-1, y+1,   z,  9)] = fmax(0.0, f_9);
        else if (x > 0)     // at +y boundary
            dev_f_new[grid2index( x-1,   0,   z,  9)] = fmax(0.0, f_9);
        else if (y < ny-1)  // at -x boundary
            dev_f_new[grid2index(nx-1, y+1,   z,  9)] = fmax(0.0, f_9);
        else    // at -x+y boundary
            dev_f_new[grid2index(nx-1,   0,   z,  9)] = fmax(0.0, f_9);

        // Edge 10 (+x,-y): Periodic
        if (x < nx-1 && y > 0)  // not at boundary
            dev_f_new[grid2index( x+1, y-1,   z, 10)] = fmax(0.0, f_10);
        else if (x < nx-1)  // at -y boundary
            dev_f_new[grid2index( x+1,ny-1,   z, 10)] = fmax(0.0, f_10);
        else if (y > 0)     // at +x boundary
            dev_f_new[grid2index(   0, y-1,   z, 10)] = fmax(0.0, f_10);
        else    // at +x-y boundary
            dev_f_new[grid2index(   0,ny-1,   z, 10)] = fmax(0.0, f_10);

        // Edge 11 (+x,+z): Periodic & bounce-back (free slip)
        if (x < nx-1 && z < nz-1)   // not at boundary
            dev_f_new[grid2index( x+1,   y, z+1, 11)] = fmax(0.0, f_11);
        else if (x < nx-1)  // at +z boundary
            dev_f_new[grid2index( x+1,   y,   0, 12)] = fmax(0.0, f_11);
        else if (z < nz-1)  // at +x boundary
            dev_f_new[grid2index(   0,   y, z+1, 11)] = fmax(0.0, f_11);
        else    // at +x+z boundary
            dev_f_new[grid2index(   0,   y,   0, 12)] = fmax(0.0, f_11);

        // Edge 12 (-x,-z): Periodic & bounce back (free slip)
        if (x > 0 && z > 0) // not at boundary
            dev_f_new[grid2index( x-1,   y, z-1, 12)] = fmax(0.0, f_12);
        else if (x > 0) // at -z boundary
            dev_f_new[grid2index( x-1,   y,nz-1, 11)] = fmax(0.0, f_12);
        else if (z > 0) // at -x boundary
            dev_f_new[grid2index(nx-1,   y, z-1, 12)] = fmax(0.0, f_12);
        else    // at -x-z boundary
            dev_f_new[grid2index(nx-1,   y,nz-1, 11)] = fmax(0.0, f_12);

        // Edge 13 (+y,+z): Periodic & bounce-back (free slip)
        if (y < ny-1 && z < nz-1)   // not at boundary
            dev_f_new[grid2index(   x, y+1, z+1, 13)] = fmax(0.0, f_13);
        else if (y < ny-1)  // at +z boundary
            dev_f_new[grid2index(   x, y+1,   0, 14)] = fmax(0.0, f_13);
        else if (z < nz-1)  // at +y boundary
            dev_f_new[grid2index(   x,   0, z+1, 13)] = fmax(0.0, f_13);
        else    // at +y+z boundary
            dev_f_new[grid2index(   x,   0,   0, 14)] = fmax(0.0, f_13);

        // Edge 14 (-y,-z): Periodic & bounce-back (free slip)
        if (y > 0 && z > 0) // not at boundary
            dev_f_new[grid2index(   x, y-1, z-1, 14)] = fmax(0.0, f_14);
        else if (y > 0) // at -z boundary
            dev_f_new[grid2index(   x, y-1,nz-1, 13)] = fmax(0.0, f_14);
        else if (z > 0) // at -y boundary
            dev_f_new[grid2index(   x,ny-1, z-1, 14)] = fmax(0.0, f_14);
        else    // at -y-z boundary
            dev_f_new[grid2index(   x,ny-1,nz-1, 13)] = fmax(0.0, f_14);

        // Edge 15 (-x,+z): Periodic & bounce-back (free slip)
        if (x > 0 && z < nz-1)  // not at boundary
            dev_f_new[grid2index( x-1,   y, z+1, 15)] = fmax(0.0, f_15);
        else if (x > 0)     // at +z boundary
            dev_f_new[grid2index( x-1,   y,   0, 16)] = fmax(0.0, f_15);
        else if (z < nz-1)  // at -x boundary
            dev_f_new[grid2index(nx-1,   y, z+1, 15)] = fmax(0.0, f_15);
        else    // at -x+z boundary
            dev_f_new[grid2index(nx-1,   y,   0, 16)] = fmax(0.0, f_15);

        // Edge 16 (+x,-z): Periodic & bounce-back (free slip)
        if (x < nx-1 && z > 0)  // not at boundary
            dev_f_new[grid2index( x+1,   y, z-1, 16)] = fmax(0.0, f_16);
        else if (x < nx-1)  // at -z boundary
            dev_f_new[grid2index( x+1,   y,nz-1, 15)] = fmax(0.0, f_16);
        else if (z > 0)     // at +x boundary
            dev_f_new[grid2index(   0,   y, z-1, 16)] = fmax(0.0, f_16);
        else    // at +x-z boundary
            dev_f_new[grid2index(   0,   y,nz-1, 15)] = fmax(0.0, f_16);

        // Edge 17 (-y,+z): Periodic & bounce-back (free slip)
        if (y > 0 && z < nz-1)  // not at boundary
            dev_f_new[grid2index(   x, y-1, z+1, 17)] = fmax(0.0, f_17);
        else if (y > 0)     // at +z boundary
            dev_f_new[grid2index(   x, y-1,   0, 18)] = fmax(0.0, f_17);
        else if (z < nz-1)  // at -y boundary
            dev_f_new[grid2index(   x,ny-1, z+1, 17)] = fmax(0.0, f_17);
        else    // at -y+z boundary
            dev_f_new[grid2index(   x,ny-1,   0, 18)] = fmax(0.0, f_17);

        // Edge 18 (+y,-z): Periodic & bounce-back (free slip)
        if (y < ny-1 && z > 0)    // not at boundary
            dev_f_new[grid2index(   x, y+1, z-1, 18)] = fmax(0.0, f_18);
        else if (y < ny-1)  // at -z boundary
            dev_f_new[grid2index(   x, y+1,   0, 17)] = fmax(0.0, f_18);
        else if (z > 0)     // at +y boundary
            dev_f_new[grid2index(   x,   0, z-1, 18)] = fmax(0.0, f_18);
        else    // at +y-z boundary
            dev_f_new[grid2index(   x,   0,   0, 17)] = fmax(0.0, f_18);

    }
}

#endif
// vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
