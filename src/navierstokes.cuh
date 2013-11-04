// darcy.cu
// CUDA implementation of Darcy flow

// Enable line below to perform Darcy flow computations on the GPU, disable for
// Enable GPU computation
#define DARCY_GPU

#include <iostream>
#include <cuda.h>
//#include <cutil_math.h>
#include <helper_math.h>

#include "vector_arithmetic.h"	// for arbitrary prec. vectors
#include "sphere.h"
#include "datatypes.h"
#include "utility.cuh"
#include "utility.h"
#include "constants.cuh"
#include "debug.h"

// Relaxation parameter, used in velocity prediction and pressure iteration
#define BETA 0.1

// Initialize memory
void DEM::initNSmemDev(void)
{
    // size of scalar field
    unsigned int memSizeF  = sizeof(Float)*NScells();

    cudaMalloc((void**)&dev_ns_p, memSizeF);     // hydraulic pressure
    //cudaMalloc((void**)&dev_ns_p_new, memSizeF); // new pressure matrix
    cudaMalloc((void**)&dev_ns_dp, memSizeF*3);  // hydraulic pressure gradient
    cudaMalloc((void**)&dev_ns_v, memSizeF*3);   // cell hydraulic velocity
    cudaMalloc((void**)&dev_ns_v_p, memSizeF*3); // predicted cell velocity
    cudaMalloc((void**)&dev_ns_phi, memSizeF);   // cell porosity
    cudaMalloc((void**)&dev_ns_dphi, memSizeF);  // cell porosity change
    cudaMalloc((void**)&dev_ns_div_phi_v_v, memSizeF*3); // div(phi v v)
    cudaMalloc((void**)&dev_ns_epsilon, memSizeF); // pressure difference
    cudaMalloc((void**)&dev_ns_epsilon_new, memSizeF); // new pressure diff.
    cudaMalloc((void**)&dev_ns_norm, memSizeF);  // normalized residual
    cudaMalloc((void**)&dev_ns_f, memSizeF);     // forcing function value
    cudaMalloc((void**)&dev_ns_f1, memSizeF);    // constant addition in forcing
    cudaMalloc((void**)&dev_ns_f2, memSizeF*3);  // constant slope in forcing

    checkForCudaErrors("End of initNSmemDev");
}

// Free memory
void DEM::freeNSmemDev()
{
    cudaFree(dev_ns_p);
    //cudaFree(dev_ns_p_new);
    cudaFree(dev_ns_dp);
    cudaFree(dev_ns_v);
    cudaFree(dev_ns_v_p);
    cudaFree(dev_ns_phi);
    cudaFree(dev_ns_dphi);
    cudaFree(dev_ns_div_phi_v_v);
    cudaFree(dev_ns_epsilon);
    cudaFree(dev_ns_epsilon_new);
    cudaFree(dev_ns_norm);
    cudaFree(dev_ns_f);
    cudaFree(dev_ns_f1);
    cudaFree(dev_ns_f2);
}

// Transfer to device
void DEM::transferNStoGlobalDeviceMemory(int statusmsg)
{
    checkForCudaErrors("Before attempting cudaMemcpy in "
            "transferNStoGlobalDeviceMemory");

    //if (verbose == 1 && statusmsg == 1)
        //std::cout << "  Transfering fluid data to the device:           ";

    // memory size for a scalar field
    unsigned int memSizeF  = sizeof(Float)*NScells();

    cudaMemcpy(dev_ns_p, ns.p, memSizeF, cudaMemcpyHostToDevice);
    checkForCudaErrors("transferNStoGlobalDeviceMemory after first cudaMemcpy");
    //cudaMemcpy(dev_ns_p_new, ns.p_new, memSizeF, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_ns_dp, ns.dp, memSizeF*3, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_ns_v, ns.v, memSizeF*3, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_ns_v_p, ns.v_p, memSizeF*3, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_ns_phi, ns.phi, memSizeF, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_ns_dphi, ns.dphi, memSizeF, cudaMemcpyHostToDevice);

    checkForCudaErrors("End of transferNStoGlobalDeviceMemory");
    //if (verbose == 1 && statusmsg == 1)
        //std::cout << "Done" << std::endl;
}

// Transfer from device
void DEM::transferNSfromGlobalDeviceMemory(int statusmsg)
{
    if (verbose == 1 && statusmsg == 1)
        std::cout << "  Transfering fluid data from the device:         ";

    // memory size for a scalar field
    unsigned int memSizeF  = sizeof(Float)*NScells();

    cudaMemcpy(ns.p, dev_ns_p, memSizeF, cudaMemcpyDeviceToHost);
    //cudaMemcpy(ns.p_new, dev_ns_p_new, memSizeF, cudaMemcpyDeviceToHost);
    cudaMemcpy(ns.dp, dev_ns_dp, memSizeF*3, cudaMemcpyDeviceToHost);
    cudaMemcpy(ns.v, dev_ns_v, memSizeF*3, cudaMemcpyDeviceToHost);
    cudaMemcpy(ns.v_p, dev_ns_v_p, memSizeF*3, cudaMemcpyDeviceToHost);
    cudaMemcpy(ns.phi, dev_ns_phi, memSizeF, cudaMemcpyDeviceToHost);
    cudaMemcpy(ns.dphi, dev_ns_dphi, memSizeF, cudaMemcpyDeviceToHost);
    cudaMemcpy(ns.norm, dev_ns_norm, memSizeF, cudaMemcpyDeviceToHost);

    checkForCudaErrors("End of transferNSfromGlobalDeviceMemory");
    if (verbose == 1 && statusmsg == 1)
        std::cout << "Done" << std::endl;
}

// Transfer the normalized residuals from device to host
void DEM::transferNSnormFromGlobalDeviceMemory()
{
    cudaMemcpy(ns.norm, dev_ns_norm, memSizeF, cudaMemcpyDeviceToHost);
    checkForCudaErrors("End of transferNSfromGlobalDeviceMemory");
}

// Returns the average value of the normalized residuals
double DEM::avgNormResNS()
{
    double res_sum;
    unsigned int N = grid.num[0]*grid.num[1]*grid.num[2];
    for (unsigned int i=0; i<N; ++i) {
        res_sum += static_cast<double>(ns.norm[i]);
    }
    return res/N;
}

// Get linear index from 3D grid position
__inline__ __device__ unsigned int idx(
        const int x, const int y, const int z)
{
    // without ghost nodes
    //return x + dev_grid.num[0]*y + dev_grid.num[0]*dev_grid.num[1]*z;

    // with ghost nodes
    // the ghost nodes are placed at x,y,z = -1 and WIDTH
    return (x+1) + (devC_grid.num[0]+2)*(y+1) +
        (devC_grid.num[0]+2)*(devC_grid.num[1]+2)*(z+1);
}

// Set the initial guess of the values of epsilon
__global__ void setNSepsilon(Float* dev_ns_epsilon)
{
    // 3D thread index
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;

    // check that we are not outside the fluid grid
    if (x < devC_grid.num[0] && y < devC_grid.num[1] && z < devC_grid.num[2]) {
        __syncthreads();
        dev_ns_epsilon[idx(x,y,z)] = 1.0;
    }
}

// Set the initial guess of the values of epsilon
__global__ void setNSdirichlet(Float* dev_ns_epsilon)
{
    // 3D thread index
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;

    // check that we are not outside the fluid grid, and at the top boundary
    if (x < devC_grid.num[0] &&
            y < devC_grid.num[1] &&
            z == devC_grid.num[2]-1) {

        __syncthreads();

        // the new value should be identical to the old value, i.e. the temporal
        // gradient is 0
        dev_ns_epsilon[idx(x,y,z)] = 0.0;
    }
}

__device__ void copyNSvalsDev(
        unsigned int read, unsigned int write,
        Float* dev_ns_p, //Float* dev_ns_p_new,
        Float3* dev_ns_dp, Float3* dev_ns_v, Float3* dev_ns_v_p,
        Float* dev_ns_phi, Float* dev_ns_dphi,
        Float* dev_ns_epsilon)
{
    // Coalesced read
    const Float  p       = dev_ns_p[read];
    //const Float  p_new   = dev_ns_p_new[read];
    const Float3 dp      = dev_ns_dp[read];
    const Float3 v       = dev_ns_v[read];
    const Float3 v_p     = dev_ns_v_p[read];
    const Float  phi     = dev_ns_phi[read];
    const Float  dphi    = dev_ns_dphi[read];
    const Float  epsilon = dev_ns_epsilon[read];

    // Coalesced write
    __syncthreads();
    dev_ns_p[write]       = p;
    //dev_ns_p_new[write]   = p_new;
    dev_ns_dp[write]      = dp;
    dev_ns_v[write]       = v;
    dev_ns_v_p[write]     = v_p;
    dev_ns_phi[write]     = phi;
    dev_ns_dphi[write]    = dphi;
    dev_ns_epsilon[write] = epsilon;
}


// Update ghost nodes from their parent cell values
// The edge (diagonal) cells are not written since they are note read
// Launch this kernel for all cells in the grid
__global__ void setNSghostNodesDev(
        Float* dev_ns_p, //Float* dev_ns_p_new,
        Float3* dev_ns_dp, Float3* dev_ns_v, Float3* dev_ns_v_p,
        Float* dev_ns_phi, Float* dev_ns_dphi,
        Float* dev_ns_epsilon)
{
    // 3D thread index
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;

    // Grid dimensions
    const unsigned int nx = devC_grid.num[0];
    const unsigned int ny = devC_grid.num[1];
    const unsigned int nz = devC_grid.num[2];

    // 1D thread index
    const unsigned int cellidx = idx(x,y,z);

    // 1D position of ghost node
    unsigned int writeidx;

    // check that we are not outside the fluid grid
    if (x < nx && y < ny && z < nz) {

        if (x == 0) {
            writeidx = idx(nx,y,z);
            copyNSvalsDev(cellidx, writeidx,
                    dev_ns_p, //dev_ns_p_new,
                    dev_ns_dp, dev_ns_v, dev_ns_v_p,
                    dev_ns_phi, dev_ns_dphi,
                    dev_ns_epsilon);
        }
        if (x == nx-1) {
            writeidx = idx(-1,y,z);
            copyNSvalsDev(cellidx, writeidx,
                    dev_ns_p, //dev_ns_p_new,
                    dev_ns_dp, dev_ns_v, dev_ns_v_p,
                    dev_ns_phi, dev_ns_dphi,
                    dev_ns_epsilon);
        }

        if (y == 0) {
            writeidx = idx(x,ny,z);
            copyNSvalsDev(cellidx, writeidx,
                    dev_ns_p, //dev_ns_p_new,
                    dev_ns_dp, dev_ns_v, dev_ns_v_p,
                    dev_ns_phi, dev_ns_dphi,
                    dev_ns_epsilon);
        }
        if (y == ny-1) {
            writeidx = idx(x,-1,z);
            copyNSvalsDev(cellidx, writeidx,
                    dev_ns_p, //dev_ns_p_new,
                    dev_ns_dp, dev_ns_v, dev_ns_v_p,
                    dev_ns_phi, dev_ns_dphi,
                    dev_ns_epsilon);
        }

        if (z == 0) {
            writeidx = idx(x,y,nz);
            copyNSvalsDev(cellidx, writeidx,
                    dev_ns_p, //dev_ns_p_new,
                    dev_ns_dp, dev_ns_v, dev_ns_v_p,
                    dev_ns_phi, dev_ns_dphi,
                    dev_ns_epsilon);
        }
        if (z == nz-1) {
            writeidx = idx(x,y,-1);
            copyNSvalsDev(cellidx, writeidx,
                    dev_ns_p, //dev_ns_p_new,
                    dev_ns_dp, dev_ns_v, dev_ns_v_p,
                    dev_ns_phi, dev_ns_dphi,
                    dev_ns_epsilon);
        }
    }
}

// Find the porosity in each cell on the base of a cubic grid, binning particles
// into the cells containing their centers. This approximation causes
// non-continuous porosities through time.
__global__ void findPorositiesCubicDev(
        unsigned int* dev_cellStart,
        unsigned int* dev_cellEnd,
        Float4* dev_x_sorted,
        Float* dev_ns_phi,
        Float* dev_ns_dphi)
{
    // 3D thread index
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;
    
    // Grid dimensions
    const unsigned int nx = devC_grid.num[0];
    const unsigned int ny = devC_grid.num[1];
    const unsigned int nz = devC_grid.num[2];

    // Cell dimensions
    const Float dx = devC_grid.L[0]/nx;
    const Float dy = devC_grid.L[1]/ny;
    const Float dz = devC_grid.L[2]/nz;
    const Float cell_volume = dx*dy*dz;

    Float void_volume = cell_volume;
    Float4 xr;  // particle pos. and radius

    // check that we are not outside the fluid grid
    if (x < nx && y < ny && z < nz) {

        // Calculate linear cell ID
        const unsigned int cellID = x + y*devC_grid.num[0]
            + (devC_grid.num[0] * devC_grid.num[1])*z;

        // Lowest particle index in cell
        const unsigned int startIdx = dev_cellStart[cellID];

        // Read old porosity
        __syncthreads();
        Float phi_0 = dev_ns_phi[idx(x,y,z)];

        Float phi = 0.99;

        // Make sure cell is not empty
        if (startIdx != 0xffffffff) {

            // Highest particle index in cell
            const unsigned int endIdx = dev_cellEnd[cellID];

            // Iterate over cell particles
            for (unsigned int i = startIdx; i<endIdx; ++i) {

                // Read particle position and radius
                __syncthreads();
                xr = dev_x_sorted[i];

                // Subtract particle volume from void volume
                void_volume -= 4.0/3.0*M_PI*xr.w*xr.w*xr.w;
            }

            // Make sure that the porosity is in the interval ]0.0;1.0[
            phi = fmin(0.99, fmax(0.01, void_volume/cell_volume));
        }

        // Save porosity and porosity change
        __syncthreads();
        dev_ns_phi[idx(x,y,z)]  = phi;
        dev_ns_dphi[idx(x,y,z)] = phi - phi_0;
    }
}


// Find the porosity in each cell on the base of a sphere, centered at the cell
// center. This approximation is continuous through time and generally
// preferable to findPorositiesCubicDev, although it's slower.
__global__ void findPorositiesSphericalDev(
        unsigned int* dev_cellStart,
        unsigned int* dev_cellEnd,
        Float4* dev_x_sorted,
        Float* dev_ns_phi,
        Float* dev_ns_dphi,
        unsigned int iteration)
{
    // 3D thread index
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;
    
    // Grid dimensions
    const unsigned int nx = devC_grid.num[0];
    const unsigned int ny = devC_grid.num[1];
    const unsigned int nz = devC_grid.num[2];

    // Cell dimensions
    const Float dx = devC_grid.L[0]/nx;
    const Float dy = devC_grid.L[1]/ny;
    const Float dz = devC_grid.L[2]/nz;

    // Cell sphere radius
    const Float R = fmin(dx, fmin(dy,dz)) * 0.5;
    const Float cell_volume = 4.0/3.0*M_PI*R*R*R;

    Float void_volume = cell_volume;
    Float4 xr;  // particle pos. and radius

    // check that we are not outside the fluid grid
    if (x < nx && y < ny && z < nz) {

        // Cell sphere center position
        const Float3 X = MAKE_FLOAT3(
                x*dx + 0.5*dx,
                y*dy + 0.5*dy,
                z*dz + 0.5*dz);

        Float d, r;
        Float phi = 0.99;

        // Read old porosity
        __syncthreads();
        Float phi_0 = dev_ns_phi[idx(x,y,z)];

        // The cell 3d index
        const int3 gridPos = make_int3((int)x,(int)y,(int)z);

        // The neighbor cell 3d index
        int3 targetCell;

        // The distance modifier for particles across periodic boundaries
        Float3 dist, distmod;

        // Iterate over 27 neighbor cells
        for (int z_dim=-1; z_dim<2; ++z_dim) { // z-axis
            for (int y_dim=-1; y_dim<2; ++y_dim) { // y-axis
                for (int x_dim=-1; x_dim<2; ++x_dim) { // x-axis

                    // Index of neighbor cell this iteration is looking at
                    targetCell = gridPos + make_int3(x_dim, y_dim, z_dim);

                    // Get distance modifier for interparticle
                    // vector, if it crosses a periodic boundary
                    // DISTMOD IS NEVER NOT 0,0,0 !!!!!!
                    distmod = MAKE_FLOAT3(0.0, 0.0, 0.0);
                    if (findDistMod(&targetCell, &distmod) != -1) {

                        // Calculate linear cell ID
                        const unsigned int cellID =
                            targetCell.x + targetCell.y * devC_grid.num[0]
                            + (devC_grid.num[0] * devC_grid.num[1])
                            * targetCell.z; 

                        // Lowest particle index in cell
                        const unsigned int startIdx = dev_cellStart[cellID];

                        // Make sure cell is not empty
                        if (startIdx != 0xffffffff) {

                            // Highest particle index in cell
                            const unsigned int endIdx = dev_cellEnd[cellID];

                            // Iterate over cell particles
                            for (unsigned int i = startIdx; i<endIdx; ++i) {

                                // Read particle position and radius
                                __syncthreads();
                                xr = dev_x_sorted[i];
                                r = xr.w;

                                // Find center distance
                                dist = MAKE_FLOAT3(
                                            X.x - xr.x, 
                                            X.y - xr.y,
                                            X.z - xr.z);
                                dist += distmod;
                                d = length(dist);

                                // Lens shaped intersection
                                if ((R - r) < d && d < (R + r)) {
                                    void_volume -=
                                        1.0/(12.0*d) * (
                                                M_PI*(R + r - d)*(R + r - d)
                                                *(d*d + 2.0*d*r - 3.0*r*r
                                                    + 2.0*d*R + 6.0*r*R
                                                    - 3.0*R*R) );
                                }

                                // Particle fully contained in cell sphere
                                if (d <= R - r) {
                                    void_volume -= 4.0/3.0*M_PI*r*r*r;
                                }
                            }
                        }
                    }
                }
            }
        }

        // Make sure that the porosity is in the interval ]0.0;1.0[
        phi = fmin(0.99, fmax(0.01, void_volume/cell_volume));
        //phi = void_volume/cell_volume;

        Float dphi = phi - phi_0;
        if (iteration == 0) {
            // Do not use the initial porosity estimates
            dphi = 0.0;
        }

        // Save porosity and porosity change
        __syncthreads();
        dev_ns_phi[idx(x,y,z)]  = phi;
        dev_ns_dphi[idx(x,y,z)] = dphi;
    }
}

// Return the discrete Laplacian in a cell in a homogenous, cubic 3D scalar
// field
__device__ Float discreteLaplacian(
        const Float* dev_scalarfield,
        const unsigned int x,
        const unsigned int y,
        const unsigned int z,
        const Float dx2)
{
    // Read neighbor values
    __syncthreads();
    const Float cellval = dev_scalarfield[idx(x,y,z)];
    const Float xn = dev_scalarfield[idx(x-1,y,z)];
    const Float xp = dev_scalarfield[idx(x+1,y,z)];
    const Float yn = dev_scalarfield[idx(x,y-1,z)];
    const Float yp = dev_scalarfield[idx(x,y+1,z)];
    const Float zn = dev_scalarfield[idx(x,y,z-1)];
    const Float zp = dev_scalarfield[idx(x,y,z+1)];

    // Return the discrete Laplacian, obtained by a finite-difference seven
    // point stencil in a three-dimensional regular, cubic grid with cell
    // spacing dx considering the 6 face neighbors
    return (xn + xp + yn + yp + zn + zp - 6.0*cellval)/dx2;
}

// Find the gradient in a cell in a homogeneous, cubic 3D scalar field
__device__ Float3 gradient(
        const Float* dev_scalarfield,
        const unsigned int x,
        const unsigned int y,
        const unsigned int z,
        const Float dx,
        const Float dy,
        const Float dz)
{
    // Read 6 neighbor cells
    __syncthreads();
    const Float xp = dev_scalarfield[idx(x+1,y,z)];
    const Float xn = dev_scalarfield[idx(x-1,y,z)];
    const Float yp = dev_scalarfield[idx(x,y+1,z)];
    const Float yn = dev_scalarfield[idx(x,y-1,z)];
    const Float zp = dev_scalarfield[idx(x,y,z+1)];
    const Float zn = dev_scalarfield[idx(x,y,z-1)];

    // Calculate central-difference gradients
    return MAKE_FLOAT3(
            (xp - xn)/(2.0*dx),
            (yp - yn)/(2.0*dy),
            (zp - zn)/(2.0*dz));
}

// Find the divergence in a cell in a homogeneous, cubic, 3D vector field
__device__ Float divergence(
        const Float3* dev_vectorfield,
        const unsigned int x,
        const unsigned int y,
        const unsigned int z,
        const Float dx,
        const Float dy,
        const Float dz)
{
    // Read 6 neighbor cells
    __syncthreads();
    const Float3 xp = dev_vectorfield[idx(x+1,y,z)];
    const Float3 xn = dev_vectorfield[idx(x-1,y,z)];
    const Float3 yp = dev_vectorfield[idx(x,y+1,z)];
    const Float3 yn = dev_vectorfield[idx(x,y-1,z)];
    const Float3 zp = dev_vectorfield[idx(x,y,z+1)];
    const Float3 zn = dev_vectorfield[idx(x,y,z-1)];

    // Calculate the central-difference gradients and divergence
    return
        (xp.x - xn.x)/(2.0*dx) + 
        (yp.y - yn.y)/(2.0*dy) + 
        (zp.z - zn.z)/(2.0*dz);
}


// Returns the value of Laplacian(epsilon) at a point x,y,z
__device__ Float laplacianEpsilon(
        Float3* dev_ns_v_p,
        Float*  dev_ns_phi,
        Float*  dev_ns_dphi,
        Float*  dev_ns_epsilon,
        const unsigned int x,
        const unsigned int y,
        const unsigned int z,
        const Float dx,
        const Float dy,
        const Float dz)
{
    __syncthreads();
    const Float3 v_p  = dev_ns_v_p[idx(x,y,z)];
    const Float  phi  = dev_ns_phi[idx(x,y,z)];
    const Float  dphi = dev_ns_dphi[idx(x,y,z)];
    const Float  rho  = 1000.0;

    // Calculate derivatives
    const Float  div_v_p      = divergence(dev_ns_v_p, x, y, z, dx, dy, dz);
    const Float3 grad_phi     = gradient(dev_ns_phi, x, y, z, dx, dy, dz);
    const Float3 grad_epsilon = gradient(dev_ns_epsilon, x, y, z, dx, dy, dz);

    return div_v_p*rho/devC_dt
        + dot(grad_phi, v_p)*rho/(devC_dt*phi)
        - dot(grad_phi, grad_epsilon)/phi
        + dphi*rho/(devC_dt*devC_dt*phi);
}


// Find the spatial gradient in e.g. pressures per cell
// using first order central differences
__global__ void findNSgradientsDev(
        Float* dev_scalarfield,     // in
        Float3* dev_vectorfield)    // out
{
    // 3D thread index
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;

    // Grid dimensions
    const unsigned int nx = devC_grid.num[0];
    const unsigned int ny = devC_grid.num[1];
    const unsigned int nz = devC_grid.num[2];

    // Grid sizes
    const Float dx = devC_grid.L[0]/nx;
    const Float dy = devC_grid.L[1]/ny;
    const Float dz = devC_grid.L[2]/nz;

    // 1D thread index
    const unsigned int cellidx = idx(x,y,z);

    // Check that we are not outside the fluid grid
    if (x < nx && y < ny && z < nz) {

        const Float3 grad = gradient(dev_scalarfield, x, y, z, dx, dy, dz);

        // Write gradient
        __syncthreads();
        dev_vectorfield[cellidx] = grad;
    }
}

// Arithmetic mean of two numbers
__inline__ __device__ Float ameanDev(Float a, Float b) {
    return (a+b)*0.5;
}

// Harmonic mean of two numbers
__inline__ __device__ Float hmeanDev(Float a, Float b) {
    return (2.0*a*b)/(a+b);
}




// Perform a time step
__global__ void explNSstepDev(
        Float* dev_ns_p,
        Float* dev_ns_p_new,
        Float* dev_ns_dphi)
{
    // 3D thread index
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;

    // Grid dimensions
    const unsigned int nx = devC_grid.num[0];
    const unsigned int ny = devC_grid.num[1];
    const unsigned int nz = devC_grid.num[2];

    // Cell sizes
    //const Float dx = devC_grid.L[0]/nx;
    //const Float dy = devC_grid.L[1]/ny;
    //const Float dz = devC_grid.L[2]/nz;

    // 1D thread index
    const unsigned int cellidx = idx(x,y,z);

    // Check that we are not outside the fluid grid
    if (x < nx && y < ny && z < nz) {

        // Read cell and the six neighbor cell hydraulic transmissivities
        __syncthreads();

        // Read the cell porosity change
        //const Float dphi = dev_ns_dphi[cellidx];

        // Read the cell and the six neighbor cell hydraulic pressures
        const Float p = dev_ns_p[cellidx];
        /*const Float H_xn = dev_d_H[idx(x-1,y,z)];
        const Float H_xp = dev_d_H[idx(x+1,y,z)];
        const Float H_yn = dev_d_H[idx(x,y-1,z)];
        const Float H_yp = dev_d_H[idx(x,y+1,z)];
        const Float H_zn = dev_d_H[idx(x,y,z-1)];
        const Float H_zp = dev_d_H[idx(x,y,z+1)];*/

        // Neumann (no-flow) boundary condition at +z and -z boundaries
        // enforced by a gradient value of 0.0
        //if (z == 0)
            //TdH_zn = 0.0;
        //if (z == nz-1)
            //TdH_zp = 0.0;

        const Float delta_p = 0.0;

        // The pressure should never be negative
        const Float p_new = fmax(0.0, p + delta_p);

        // Write the new hydraulic pressure in cell
        __syncthreads();
        dev_ns_p_new[cellidx] = p_new;
        //}
    }
}

// Find the divergence of phi v v
__global__ void findNSdivphivv(
        Float3* dev_ns_v,    // in
        Float*  dev_ns_phi,  // in
        Float3* dev_ns_div_phi_v_v) // out
{
    // 3D thread index
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;

    // Grid dimensions
    const unsigned int nx = devC_grid.num[0];
    const unsigned int ny = devC_grid.num[1];
    const unsigned int nz = devC_grid.num[2];

    // Grid sizes
    //const Float dx = devC_grid.L[0]/nx;
    //const Float dy = devC_grid.L[1]/ny;
    //const Float dz = devC_grid.L[2]/nz;

    // 1D thread index
    const unsigned int cellidx = idx(x,y,z);

    // Check that we are not outside the fluid grid
    if (x < nx && y < ny && z < nz) {

        // Read 6 neighbor cells
        __syncthreads();
        const Float3 v      = dev_ns_v[idx(x,y,z)];
        const Float3 v_xp   = dev_ns_v[idx(x+1,y,z)];
        const Float3 v_xn   = dev_ns_v[idx(x-1,y,z)];
        const Float3 v_yp   = dev_ns_v[idx(x,y+1,z)];
        const Float3 v_yn   = dev_ns_v[idx(x,y-1,z)];
        const Float3 v_zp   = dev_ns_v[idx(x,y,z+1)];
        const Float3 v_zn   = dev_ns_v[idx(x,y,z-1)];

        const Float  phi    = dev_ns_phi[idx(x,y,z)];
        const Float  phi_xp = dev_ns_phi[idx(x+1,y,z)];
        const Float  phi_xn = dev_ns_phi[idx(x-1,y,z)];
        const Float  phi_yp = dev_ns_phi[idx(x,y+1,z)];
        const Float  phi_yn = dev_ns_phi[idx(x,y-1,z)];
        const Float  phi_zp = dev_ns_phi[idx(x,y,z+1)];
        const Float  phi_zn = dev_ns_phi[idx(x,y,z-1)];

        // The outer product (v v) looks like:
        // [[ v_x^2    v_x*v_y  v_x*v_z ]
        //  [ v_y*v_x  v_y^2    v_y*v_z ]
        //  [ v_z*v_x  v_z*v_y  v_z^2   ]]

        // Given a tensor T =
        // [[ e_xx  e_xy  e_xz ]
        //  [ e_yx  e_yy  e_yz ]
        //  [ e_zx  e_zy  e_zz ]]
        //  e_ij: i-th row, j-th col

        // div(T) = 
        //  [ de_xx/dx + de_xy/dy + de_xz/dz ,
        //    de_yx/dx + de_yy/dy + de_yz/dz ,
        //    de_zx/dx + de_zy/dy + de_zz/dz ]

        // This function finds the divergence of (phi v v), which is a vector

        // Calculate the divergence
        const Float3 div = MAKE_FLOAT3(0.0,0.0,0.0);
        

        // Write divergence
        __syncthreads();
        dev_ns_div_phi_v_v[cellidx] = div;
    }
}


// Find predicted fluid velocity
__global__ void findPredNSvelocitiesDev(
        Float*  dev_ns_p,
        Float3* dev_ns_v,
        Float*  dev_ns_phi,
        Float3* dev_ns_div_phi_v_v,
        Float3* dev_ns_v_p)
{
    // 3D thread index
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;

    // Grid dimensions
    const unsigned int nx = devC_grid.num[0];
    const unsigned int ny = devC_grid.num[1];
    const unsigned int nz = devC_grid.num[2];

    // Cell sizes
    const Float dx = devC_grid.L[0]/nx;
    const Float dy = devC_grid.L[1]/ny;
    const Float dz = devC_grid.L[2]/nz;

    // 1D thread index
    const unsigned int cellidx = idx(x,y,z);

    // Check that we are not outside the fluid grid
    if (x < nx && y < ny && z < nz) {

        // Values that are needed for calculating the predicted velocity
        __syncthreads();
        const Float3 v           = dev_ns_v[cellidx];
        const Float  phi         = dev_ns_phi[cellidx];
        const Float3 div_phi_v_v = dev_ns_div_phi_v_v[cellidx];

        // Fluid density
        const Float  rho = 1000.0;

        // Particle interaction force
        const Float3 f_i = MAKE_FLOAT3(0.0, 0.0, 0.0);

        // Gravitational drag force on cell fluid mass
        const Float3 f_g = MAKE_FLOAT3(
                devC_params.g[0], devC_params.g[1], devC_params.g[2])
            * rho * dx*dy*dz * phi;

        // Find pressure gradient
        const Float3 grad_p = gradient(dev_ns_p, x, y, z, dx, dy, dz);

        // Calculate the predicted velocity
        const Float3 v_p
            = v - devC_dt*div_phi_v_v
            - devC_dt*BETA/rho*phi*grad_p
            - devC_dt/rho*f_i
            + devC_dt*phi*f_g;

        // Save the predicted velocity
        __syncthreads();
        dev_ns_v_p[cellidx] = v_p;
    }
}

// Perform a single Jacobi iteration
__global__ void jacobiIterationNS(
        Float* dev_ns_epsilon,
        Float* dev_ns_epsilon_new,
        Float* dev_ns_norm,
        Float* dev_ns_f)
{
    // 3D thread index
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;

    // Grid dimensions
    const unsigned int nx = devC_grid.num[0];
    const unsigned int ny = devC_grid.num[1];
    const unsigned int nz = devC_grid.num[2];

    // Cell sizes
    const Float dx = devC_grid.L[0]/nx;
    const Float dy = devC_grid.L[1]/ny;
    const Float dz = devC_grid.L[2]/nz;

    // 1D thread index
    const unsigned int cellidx = idx(x,y,z);

    // Check that we are not outside the fluid grid
    if (x < nx && y < ny && z < nz) {

        //Float e, e_xn, e_xp, e_, e_new, res;

        // Read the epsilon values from the cell and its 6 neighbors
        __syncthreads();
        const Float e    = dev_ns_epsilon[cellidx];
        const Float e_xn = dev_ns_epsilon[idx(x-1,y,z)];
        const Float e_xp = dev_ns_epsilon[idx(x+1,y,z)];
        const Float e_yn = dev_ns_epsilon[idx(x,y-1,z)];
        const Float e_yp = dev_ns_epsilon[idx(x,y+1,z)];
        const Float e_zn = dev_ns_epsilon[idx(x,y,z-1)];
        const Float e_zp = dev_ns_epsilon[idx(x,y,z+1)];

        // Read the value of the forcing function
        const Float f = dev_ns_f[cellidx];

        // Calculate grid coefficients
        const Float div = 2.0*(dx*dx + dy*dy + dz*dz);
        const Float ax = dy*dy*dz*dz/div;
        const Float ay = dz*dz*dx*dx/div;
        const Float az = dx*dx*dy*dy/div;
        const Float af = dx*dx*dy*dy*dz*dz/div;



    }
}


// Computes the new velocity and pressure using the corrector
__global__ void updateNSvelocityPressure(
        Float*  dev_ns_p,
        Float3* dev_ns_v,
        Float3* dev_ns_v_p,
        Float*  dev_ns_epsilon)
{
    // 3D thread index
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;

    // Grid dimensions
    const unsigned int nx = devC_grid.num[0];
    const unsigned int ny = devC_grid.num[1];
    const unsigned int nz = devC_grid.num[2];

    // Cell sizes
    const Float dx = devC_grid.L[0]/nx;
    const Float dy = devC_grid.L[1]/ny;
    const Float dz = devC_grid.L[2]/nz;

    // 1D thread index
    const unsigned int cellidx = idx(x,y,z);

    // Check that we are not outside the fluid grid
    if (x < nx && y < ny && z < nz) {

        // Read values
        __syncthreads();
        const Float  p_old   = dev_ns_p[cellidx];
        const Float  epsilon = dev_ns_epsilon[cellidx];
        const Float3 v_p     = dev_ns_v_p[cellidx];

        // New pressure
        const Float p = BETA*p_old + epsilon;

        // Find corrector gradient
        const Float3 grad_epsilon
            = gradient(dev_ns_epsilon, x, y, z, dx, dy, dz);

        // Fluid density
        const Float rho = 1000.0;

        // Find new velocity
        const Float3 v = v_p - devC_dt/rho*grad_epsilon;

        // Write new values
        __syncthreads();
        dev_ns_p[cellidx] = p;
        dev_ns_v[cellidx] = v;

    }
}

// Print final heads and free memory
void DEM::endNSdev()
{
    freeNSmemDev();
}

// vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
