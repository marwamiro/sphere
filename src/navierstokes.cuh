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

// Initialize memory
void DEM::initDarcyMemDev(void)
{
    // number of cells
    //unsigned int ncells = d.nx*d.ny*d.nz; // without ghost nodes
    unsigned int ncells = (d.nx+2)*(d.ny+2)*(d.nz+2); // with ghost nodes
    unsigned int memSizeF  = sizeof(Float) * ncells;

    cudaMalloc((void**)&dev_d_H, memSizeF);     // hydraulic pressure
    cudaMalloc((void**)&dev_d_H_new, memSizeF); // new pressure matrix
    cudaMalloc((void**)&dev_d_V, memSizeF*3);   // cell hydraulic velocity
    cudaMalloc((void**)&dev_d_dH, memSizeF*3);  // hydraulic pressure gradient
    cudaMalloc((void**)&dev_d_K, memSizeF);     // hydraulic conductivity
    cudaMalloc((void**)&dev_d_T, memSizeF*3);   // hydraulic transmissivity
    cudaMalloc((void**)&dev_d_Ss, memSizeF);    // hydraulic storativi
    cudaMalloc((void**)&dev_d_W, memSizeF);     // hydraulic recharge
    cudaMalloc((void**)&dev_d_phi, memSizeF);   // cell porosity
    cudaMalloc((void**)&dev_d_dphi, memSizeF);  // cell porosity change

    checkForCudaErrors("End of initDarcyMemDev");
}

// Free memory
void DEM::freeDarcyMemDev()
{
    cudaFree(dev_d_H);
    cudaFree(dev_d_H_new);
    cudaFree(dev_d_V);
    cudaFree(dev_d_dH);
    cudaFree(dev_d_K);
    cudaFree(dev_d_T);
    cudaFree(dev_d_Ss);
    cudaFree(dev_d_W);
    cudaFree(dev_d_phi);
    cudaFree(dev_d_dphi);
}

// Transfer to device
void DEM::transferDarcyToGlobalDeviceMemory(int statusmsg)
{
    checkForCudaErrors("Before attempting cudaMemcpy in "
            "transferDarcyToGlobalDeviceMemory");

    //if (verbose == 1 && statusmsg == 1)
        //std::cout << "  Transfering fluid data to the device:           ";

    // number of cells
    //unsigned int ncells = d.nx*d.ny*d.nz; // without ghost nodes
    unsigned int ncells = (d.nx+2)*(d.ny+2)*(d.nz+2); // with ghost nodes
    unsigned int memSizeF  = sizeof(Float) * ncells;

    // Kinematic particle values
    cudaMemcpy(dev_d_H, d.H, memSizeF, cudaMemcpyHostToDevice);
    checkForCudaErrors("transferDarcyToGlobalDeviceMemory after first cudaMemcpy");
    cudaMemcpy(dev_d_H_new, d.H_new, memSizeF, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_d_V, d.V, memSizeF*3, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_d_dH, d.dH, memSizeF*3, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_d_K, d.K, memSizeF, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_d_T, d.T, memSizeF*3, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_d_Ss, d.Ss, memSizeF, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_d_W, d.W, memSizeF, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_d_phi, d.phi, memSizeF, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_d_dphi, d.dphi, memSizeF, cudaMemcpyHostToDevice);

    checkForCudaErrors("End of transferDarcyToGlobalDeviceMemory");
    //if (verbose == 1 && statusmsg == 1)
        //std::cout << "Done" << std::endl;
}

// Transfer from device
void DEM::transferDarcyFromGlobalDeviceMemory(int statusmsg)
{
    if (verbose == 1 && statusmsg == 1)
        std::cout << "  Transfering darcy data from the device:         ";

    // number of cells
    //unsigned int ncells = d.nx*d.ny*d.nz; // without ghost nodes
    unsigned int ncells = (d.nx+2)*(d.ny+2)*(d.nz+2); // with ghost nodes
    unsigned int memSizeF  = sizeof(Float) * ncells;

    // Kinematic particle values
    cudaMemcpy(d.H, dev_d_H, memSizeF, cudaMemcpyDeviceToHost);
    cudaMemcpy(d.H_new, dev_d_H_new, memSizeF, cudaMemcpyDeviceToHost);
    cudaMemcpy(d.V, dev_d_V, memSizeF*3, cudaMemcpyDeviceToHost);
    cudaMemcpy(d.dH, dev_d_dH, memSizeF*3, cudaMemcpyDeviceToHost);
    cudaMemcpy(d.K, dev_d_K, memSizeF, cudaMemcpyDeviceToHost);
    cudaMemcpy(d.T, dev_d_T, memSizeF*3, cudaMemcpyDeviceToHost);
    cudaMemcpy(d.Ss, dev_d_Ss, memSizeF, cudaMemcpyDeviceToHost);
    cudaMemcpy(d.W, dev_d_W, memSizeF, cudaMemcpyDeviceToHost);
    cudaMemcpy(d.phi, dev_d_phi, memSizeF, cudaMemcpyDeviceToHost);
    cudaMemcpy(d.dphi, dev_d_dphi, memSizeF, cudaMemcpyDeviceToHost);

    checkForCudaErrors("End of transferDarcyFromGlobalDeviceMemory");
    if (verbose == 1 && statusmsg == 1)
        std::cout << "Done" << std::endl;
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

__device__ void copyDarcyValsDev(
        unsigned int read, unsigned int write,
        Float* dev_d_H, Float* dev_d_H_new,
        Float3* dev_d_V, Float3* dev_d_dH,
        Float* dev_d_K, Float3* dev_d_T,
        Float* dev_d_Ss, Float* dev_d_W,
        Float* dev_d_phi,
        Float* dev_d_dphi)
{
    // Coalesced read
    const Float  H     = dev_d_H[read];
    const Float  H_new = dev_d_H_new[read];
    const Float3 V     = dev_d_V[read];
    const Float3 dH    = dev_d_dH[read];
    const Float  K     = dev_d_K[read];
    const Float3 T     = dev_d_T[read];
    const Float  Ss    = dev_d_Ss[read];
    const Float  W     = dev_d_W[read];
    const Float  phi   = dev_d_phi[read];
    const Float  dphi  = dev_d_dphi[read];

    // Coalesced write
    __syncthreads();
    dev_d_H[write]     = H;
    dev_d_H_new[write] = H_new;
    dev_d_V[write]     = V;
    dev_d_dH[write]    = dH;
    dev_d_K[write]     = K;
    dev_d_T[write]     = T;
    dev_d_Ss[write]    = Ss;
    dev_d_W[write]     = W;
    dev_d_phi[write]   = phi;
    dev_d_dphi[write]  = dphi;
}

// Update ghost nodes from their parent cell values
// The edge (diagonal) cells are not written since they are note read
// Launch this kernel for all cells in the grid
__global__ void setDarcyGhostNodesDev(
        Float* dev_d_H, Float* dev_d_H_new,
        Float3* dev_d_V, Float3* dev_d_dH,
        Float* dev_d_K, Float3* dev_d_T,
        Float* dev_d_Ss, Float* dev_d_W,
        Float* dev_d_phi,
        Float* dev_d_dphi)
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
            copyDarcyValsDev(cellidx, writeidx,
                    dev_d_H, dev_d_H_new,
                    dev_d_V, dev_d_dH,
                    dev_d_K, dev_d_T,
                    dev_d_Ss, dev_d_W,
                    dev_d_phi,
                    dev_d_dphi);
        }
        if (x == nx-1) {
            writeidx = idx(-1,y,z);
            copyDarcyValsDev(cellidx, writeidx,
                    dev_d_H, dev_d_H_new,
                    dev_d_V, dev_d_dH,
                    dev_d_K, dev_d_T,
                    dev_d_Ss, dev_d_W,
                    dev_d_phi,
                    dev_d_dphi);
        }

        if (y == 0) {
            writeidx = idx(x,ny,z);
            copyDarcyValsDev(cellidx, writeidx,
                    dev_d_H, dev_d_H_new,
                    dev_d_V, dev_d_dH,
                    dev_d_K, dev_d_T,
                    dev_d_Ss, dev_d_W,
                    dev_d_phi,
                    dev_d_dphi);
        }
        if (y == ny-1) {
            writeidx = idx(x,-1,z);
            copyDarcyValsDev(cellidx, writeidx,
                    dev_d_H, dev_d_H_new,
                    dev_d_V, dev_d_dH,
                    dev_d_K, dev_d_T,
                    dev_d_Ss, dev_d_W,
                    dev_d_phi,
                    dev_d_dphi);
        }

        if (z == 0) {
            writeidx = idx(x,y,nz);
            copyDarcyValsDev(cellidx, writeidx,
                    dev_d_H, dev_d_H_new,
                    dev_d_V, dev_d_dH,
                    dev_d_K, dev_d_T,
                    dev_d_Ss, dev_d_W,
                    dev_d_phi,
                    dev_d_dphi);
        }
        if (z == nz-1) {
            writeidx = idx(x,y,-1);
            copyDarcyValsDev(cellidx, writeidx,
                    dev_d_H, dev_d_H_new,
                    dev_d_V, dev_d_dH,
                    dev_d_K, dev_d_T,
                    dev_d_Ss, dev_d_W,
                    dev_d_phi,
                    dev_d_dphi);
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
        Float* dev_d_phi,
        Float* dev_d_dphi)
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
        Float phi_0 = dev_d_phi[idx(x,y,z)];

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
        dev_d_phi[idx(x,y,z)]  = phi;
        dev_d_dphi[idx(x,y,z)] = phi - phi_0;
    }
}


// Find the porosity in each cell on the base of a sphere, centered at the cell
// center. This approximation is continuous through time and generally
// preferable to findPorositiesCubicDev, although it's slower.
__global__ void findPorositiesSphericalDev(
        unsigned int* dev_cellStart,
        unsigned int* dev_cellEnd,
        Float4* dev_x_sorted,
        Float* dev_d_phi,
        Float* dev_d_dphi,
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
        Float phi_0 = dev_d_phi[idx(x,y,z)];

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
        phi = fmin(0.9, fmax(0.1, void_volume/cell_volume));
        //phi = void_volume/cell_volume;

        Float dphi = phi - phi_0;
        if (iteration == 0) {
            // Do not use the initial CPU porosity estimates
            dphi = 0.0;
        }

        // Save porosity and porosity change
        __syncthreads();
        dev_d_phi[idx(x,y,z)]  = phi;
        dev_d_dphi[idx(x,y,z)] = dphi;
    }
}


// Find cell transmissivities from hydraulic conductivities and cell dimensions
// Make sure to compute the porosities (d_phi) beforehand
// d_factor: Grain size factor for Kozeny-Carman relationship
__global__ void findDarcyTransmissivitiesDev(
        Float* dev_d_K,
        Float3* dev_d_T,
        Float* dev_d_phi,
        Float d_factor)
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

    // Density of the fluid [kg/m^3]
    const Float rho = 1000.0;

    // Check that we are not outside the fluid grid
    if (x < nx && y < ny && z < nz) {

        // 1D thread index
        const unsigned int cellidx = idx(x,y,z);

        __syncthreads();

        // Read the cell porosity [-]
        const Float phi = dev_d_phi[cellidx];

        // Calculate permeability from the Kozeny-Carman relationship
        // Nelson 1994 eq. 1c
        // Boek 2012 eq. 16
        //k = a*phi*phi*phi/(1.0 - phi*phi);
        // Schwartz and Zhang 2003
        Float k = phi*phi*phi/((1.0-phi)*(1.0-phi)) * d_factor;

        // Save hydraulic conductivity [m/s]
        //const Float K = k*rho*-devC_params.g[2]/devC_params.nu;
        const Float K = k*rho*-devC_params.g[2]/devC_params.nu;
        //const Float K = 0.5; 
        //const Float K = 1.0e-2; 

        // Hydraulic transmissivity [m2/s]
        Float3 T = {K*dx, K*dy, K*dz};

        // Save values. Note! The K values are unused
        __syncthreads();
        dev_d_K[cellidx] = K;
        dev_d_T[cellidx] = T;

    }
}

// Find the spatial gradient in e.g.pressures per cell
// using first order central differences
__global__ void findDarcyGradientsDev(
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
    Float3 gradient;
    Float xp, xn, yp, yn, zp, zn;
    if (x < nx && y < ny && z < nz) {

        // Read 6 neighbor cells
        __syncthreads();
        xp = dev_scalarfield[idx(x+1,y,z)];
        xn = dev_scalarfield[idx(x-1,y,z)];
        yp = dev_scalarfield[idx(x,y+1,z)];
        yn = dev_scalarfield[idx(x,y-1,z)];
        zp = dev_scalarfield[idx(x,y,z+1)];
        zn = dev_scalarfield[idx(x,y,z-1)];

        // Calculate central-difference gradients
        // x
        gradient.x = (xp - xn)/(2.0*dx);

        // y
        gradient.y = (yp - yn)/(2.0*dy);

        // z
        gradient.z = (zp - zn)/(2.0*dz);

        // Write gradient
        __syncthreads();
        dev_vectorfield[cellidx] = gradient;
    }
}

// Arithmetic mean of two numbers
__device__ Float ameanDev(Float a, Float b) {
    return (a+b)*0.5;
}

// Harmonic mean of two numbers
__device__ Float hmeanDev(Float a, Float b) {
    return (2.0*a*b)/(a+b);
}

// Perform an explicit step.
__global__ void explDarcyStepDev(
        Float* dev_d_H,
        Float* dev_d_H_new,
        Float3* dev_d_T,
        Float* dev_d_Ss,
        Float* dev_d_W,
        Float* dev_d_dphi)
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

        // Explicit 3D finite difference scheme
        // new = old + production*timestep + gradient*timestep

        // Enforce Dirichlet BC
        /*if (x == 0 || y == 0 || z == 0 ||
                x == nx-1 || y == ny-1 || z == nz-1) {
            __syncthreads();
            dev_d_H_new[cellidx] = dev_d_H[cellidx];
        } else {*/

        // Read cell and the six neighbor cell hydraulic transmissivities
        __syncthreads();
        const Float3 T = dev_d_T[cellidx];
        const Float3 T_xn = dev_d_T[idx(x-1,y,z)];
        const Float3 T_xp = dev_d_T[idx(x+1,y,z)];
        const Float3 T_yn = dev_d_T[idx(x,y-1,z)];
        const Float3 T_yp = dev_d_T[idx(x,y+1,z)];
        const Float3 T_zn = dev_d_T[idx(x,y,z-1)];
        const Float3 T_zp = dev_d_T[idx(x,y,z+1)];

        // Read the cell hydraulic specific storativity
        const Float Ss = dev_d_Ss[cellidx];

        // Read the cell hydraulic recharge
        const Float Q = dev_d_W[cellidx];

        // Read the cell porosity change
        const Float dphi = dev_d_dphi[cellidx];

        // Read the cell and the six neighbor cell hydraulic pressures
        const Float H = dev_d_H[cellidx];
        const Float H_xn = dev_d_H[idx(x-1,y,z)];
        const Float H_xp = dev_d_H[idx(x+1,y,z)];
        const Float H_yn = dev_d_H[idx(x,y-1,z)];
        const Float H_yp = dev_d_H[idx(x,y+1,z)];
        const Float H_zn = dev_d_H[idx(x,y,z-1)];
        const Float H_zp = dev_d_H[idx(x,y,z+1)];

        // Calculate the gradients in the pressure between
        // the cell and it's six neighbors
        const Float TdH_xn = hmeanDev(T.x, T_xn.x) * (H_xn - H)/(dx*dx);
        const Float TdH_xp = hmeanDev(T.x, T_xp.x) * (H_xp - H)/(dx*dx);
        const Float TdH_yn = hmeanDev(T.y, T_yn.y) * (H_yn - H)/(dy*dy);
        const Float TdH_yp = hmeanDev(T.y, T_yp.y) * (H_yp - H)/(dy*dy);
              Float TdH_zn = hmeanDev(T.z, T_zn.z) * (H_zn - H)/(dz*dz);
              Float TdH_zp = hmeanDev(T.z, T_zp.z) * (H_zp - H)/(dz*dz);

        // Mean cell dimension
        const Float dx_bar = (dx+dy+dz)/3.0;

        // Neumann (no-flow) boundary condition at +z and -z boundaries
        // enforced by a gradient value of 0.0
        if (z == 0)
            TdH_zn = 0.0;
        if (z == nz-1)
            TdH_zp = 0.0;

        // Determine the Laplacian operator
        const Float laplacianH = 
              TdH_xn + TdH_xp 
            + TdH_yn + TdH_yp
            + TdH_zn + TdH_zp;

        const Float porevol = -dx_bar*dphi/devC_dt;

        // The change in hydraulic pressure
        const Float deltaH = devC_dt/(Ss*dx*dy*dz)*(laplacianH + Q + porevol);

        /*printf("%d,%d,%d: H = %f, deltaH = %f,\tlaplacianH = %f,"
                "dphi = %f,\tpore = %f, "
                "TdH_xn = %f, TdH_xp = %f, "
                "TdH_yn = %f, TdH_yp = %f, "
                "TdH_zn = %f, TdH_zp = %f\n"
                "\tT_xn = %f,\tT_xp = %f,\t"
                "T_yn = %f,\tT_yp = %f,\t"
                "T_zn = %f,\tT_zp = %f\n",
                x,y,z, H, deltaH, laplacianH, dphi, porevol,
                TdH_xn, TdH_xp,
                TdH_yn, TdH_yp,
                TdH_zn, TdH_zp,
                T_xn.x, T_xp.x,
                T_yn.y, T_yp.y,
                T_zn.z, T_zp.z);*/

        // The pressure should never be negative
        const Float H_new = fmax(0.0, H + deltaH);

        // Write the new hydraulic pressure in cell
        __syncthreads();
        dev_d_H_new[cellidx] = H_new;
        //}
    }
}

// Find cell velocity
__global__ void findDarcyVelocitiesDev(
        Float* dev_d_H,
        Float3* dev_d_dH,
        Float3* dev_d_V,
        Float* dev_d_phi,
        Float* dev_d_K)
{
    // 3D thread index
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;

    // 1D thread index
    const unsigned int cellidx = idx(x,y,z);

    // Check that we are not outside the fluid grid
    if (x < devC_grid.num[0] && y < devC_grid.num[1] && z < devC_grid.num[2]) {

        // Flux [m/s]: q = -k/nu * dH
        // Pore velocity [m/s]: v = q/n

        // Dynamic viscosity
        Float nu = devC_params.nu;

        __syncthreads();
        const Float3 dH = dev_d_dH[cellidx];
        const Float K = dev_d_K[cellidx];
        const Float phi = dev_d_phi[cellidx];

        // Calculate flux
        // The sign might need to be reversed, depending on the
        // grid orientation
        Float3 q = MAKE_FLOAT3(
                -K/nu * dH.x,
                -K/nu * dH.y,
                -K/nu * dH.z);

        // Calculate velocity
        Float3 v = MAKE_FLOAT3(
                v.x = q.x/phi,
                v.y = q.y/phi,
                v.z = q.z/phi);

        // Save velocity
        __syncthreads();
        dev_d_V[cellidx] = v;
    }
}

// Print final heads and free memory
void DEM::endDarcyDev()
{
    freeDarcyMemDev();
}

// vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
