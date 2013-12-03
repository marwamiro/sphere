// navierstokes.cuh
// CUDA implementation of porous flow

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
#define BETA 1.0

// Initialize memory
void DEM::initNSmemDev(void)
{
    // size of scalar field
    unsigned int memSizeF  = sizeof(Float)*NScells();

    cudaMalloc((void**)&dev_ns_p, memSizeF);     // hydraulic pressure
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
    cudaMalloc((void**)&dev_ns_v_prod, memSizeF*6); // outer product of velocity

    checkForCudaErrors("End of initNSmemDev");
}

// Free memory
void DEM::freeNSmemDev()
{
    cudaFree(dev_ns_p);
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
    cudaFree(dev_ns_v_prod);
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

    //writeNSarray(ns.p, "ns.p.txt");

    cudaMemcpy(dev_ns_p, ns.p, memSizeF, cudaMemcpyHostToDevice);
    checkForCudaErrors("transferNStoGlobalDeviceMemory after first cudaMemcpy");
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
    cudaMemcpy(ns.norm, dev_ns_norm, sizeof(Float)*NScells(),
            cudaMemcpyDeviceToHost);
    checkForCudaErrors("End of transferNSnormFromGlobalDeviceMemory");
}

// Transfer the pressure change from device to host
void DEM::transferNSepsilonFromGlobalDeviceMemory()
{
    cudaMemcpy(ns.epsilon, dev_ns_epsilon, sizeof(Float)*NScells(),
            cudaMemcpyDeviceToHost);
    checkForCudaErrors("End of transferNSepsilonFromGlobalDeviceMemory");
}

// Transfer the pressure change from device to host
void DEM::transferNSepsilonNewFromGlobalDeviceMemory()
{
    cudaMemcpy(ns.epsilon_new, dev_ns_epsilon_new, sizeof(Float)*NScells(),
            cudaMemcpyDeviceToHost);
    checkForCudaErrors("End of transferNSepsilonFromGlobalDeviceMemory");
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

// Set the initial guess of the values of epsilon.
// The normalized residuals are given an initial value of 0, since the values at
// the Dirichlet boundaries aren't written during the iterations.
__global__ void setNSepsilon(Float* dev_ns_epsilon, Float* dev_ns_norm)
{
    // 3D thread index
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;

    // check that we are not outside the fluid grid
    if (x < devC_grid.num[0] && y < devC_grid.num[1] && z < devC_grid.num[2]) {
        __syncthreads();
        const unsigned int cellidx = idx(x,y,z);
        dev_ns_epsilon[cellidx] = 0.0;
        dev_ns_norm[cellidx]    = 0.0;
    }
}

// Set the initial guess of the values of epsilon. Since the Dirichlet boundary
// values aren't transfered during array swapping, the values also need to be
// written to the new array of epsilons.
__global__ void setNSdirichlet(
        Float* dev_ns_epsilon,
        Float* dev_ns_epsilon_new)
{
    // 3D thread index
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;

    // check that we are not outside the fluid grid, and at the z boundaries
    if (x < devC_grid.num[0] && y < devC_grid.num[1] &&
            (z == devC_grid.num[2]-1 || z == 0)) {

        /*Float val;

        if (z == 0)
            val = 0.0;
        if (z == devC_grid.num[2]-1)
            val = devC_grid.num[2]-1;
            //val = 0.0;*/

        // the new value should be identical to the old value, i.e. the temporal
        // gradient is 0
        __syncthreads();
        const unsigned int cellidx = idx(x,y,z);
        dev_ns_epsilon[cellidx]     = 0.0;
        dev_ns_epsilon_new[cellidx] = 0.0;
        //dev_ns_epsilon[idx(x,y,z)]     = val;
        //dev_ns_epsilon_new[idx(x,y,z)] = val;
    }
}

__device__ void copyNSvalsDev(
        unsigned int read, unsigned int write,
        Float* dev_ns_p, Float3* dev_ns_dp,
        Float3* dev_ns_v, Float3* dev_ns_v_p,
        Float* dev_ns_phi, Float* dev_ns_dphi,
        Float* dev_ns_epsilon)
{
    // Coalesced read
    const Float  p       = dev_ns_p[read];
    const Float3 dp      = dev_ns_dp[read];
    const Float3 v       = dev_ns_v[read];
    const Float3 v_p     = dev_ns_v_p[read];
    const Float  phi     = dev_ns_phi[read];
    const Float  dphi    = dev_ns_dphi[read];
    const Float  epsilon = dev_ns_epsilon[read];

    // Coalesced write
    __syncthreads();
    dev_ns_p[write]       = p;
    dev_ns_dp[write]      = dp;
    dev_ns_v[write]       = v;
    dev_ns_v_p[write]     = v_p;
    dev_ns_phi[write]     = phi;
    dev_ns_dphi[write]    = dphi;
    dev_ns_epsilon[write] = epsilon;
}


// Update ghost nodes from their parent cell values. The edge (diagonal) cells
// are not written since they are not read. Launch this kernel for all cells in
// the grid
__global__ void setNSghostNodesDev(
        Float* dev_ns_p, Float3* dev_ns_dp,
        Float3* dev_ns_v, Float3* dev_ns_v_p,
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
                    dev_ns_p, dev_ns_dp,
                    dev_ns_v, dev_ns_v_p,
                    dev_ns_phi, dev_ns_dphi,
                    dev_ns_epsilon);
        }
        if (x == nx-1) {
            writeidx = idx(-1,y,z);
            copyNSvalsDev(cellidx, writeidx,
                    dev_ns_p, dev_ns_dp,
                    dev_ns_v, dev_ns_v_p,
                    dev_ns_phi, dev_ns_dphi,
                    dev_ns_epsilon);
        }

        if (y == 0) {
            writeidx = idx(x,ny,z);
            copyNSvalsDev(cellidx, writeidx,
                    dev_ns_p, dev_ns_dp,
                    dev_ns_v, dev_ns_v_p,
                    dev_ns_phi, dev_ns_dphi,
                    dev_ns_epsilon);
        }
        if (y == ny-1) {
            writeidx = idx(x,-1,z);
            copyNSvalsDev(cellidx, writeidx,
                    dev_ns_p, dev_ns_dp,
                    dev_ns_v, dev_ns_v_p,
                    dev_ns_phi, dev_ns_dphi,
                    dev_ns_epsilon);
        }

        // Z boundaries fixed
        if (z == 0) {
            writeidx = idx(x,y,-1);
            copyNSvalsDev(cellidx, writeidx,
                    dev_ns_p, dev_ns_dp,
                    dev_ns_v, dev_ns_v_p,
                    dev_ns_phi, dev_ns_dphi,
                    dev_ns_epsilon);
        }
        if (z == nz-1) {
            writeidx = idx(x,y,nz);
            copyNSvalsDev(cellidx, writeidx,
                    dev_ns_p, dev_ns_dp,
                    dev_ns_v, dev_ns_v_p,
                    dev_ns_phi, dev_ns_dphi,
                    dev_ns_epsilon);
        }

        // Z boundaries periodic
        /*if (z == 0) {
            writeidx = idx(x,y,nz);
            copyNSvalsDev(cellidx, writeidx,
                    dev_ns_p, dev_ns_dp,
                    dev_ns_v, dev_ns_v_p,
                    dev_ns_phi, dev_ns_dphi,
                    dev_ns_epsilon);
        }
        if (z == nz-1) {
            writeidx = idx(x,y,-1);
            copyNSvalsDev(cellidx, writeidx,
                    dev_ns_p, dev_ns_dp,
                    dev_ns_v, dev_ns_v_p,
                    dev_ns_phi, dev_ns_dphi,
                    dev_ns_epsilon);
        }*/
    }
}

// Update a field in the ghost nodes from their parent cell values. The edge
// (diagonal) cells are not written since they are not read. Launch this kernel
// for all cells in the grid usind setNSghostNodes<datatype><<<.. , ..>>>( .. );
template<typename T>
__global__ void setNSghostNodes(T* dev_scalarfield)
{
    // 3D thread index
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;

    // Grid dimensions
    const unsigned int nx = devC_grid.num[0];
    const unsigned int ny = devC_grid.num[1];
    const unsigned int nz = devC_grid.num[2];

    // check that we are not outside the fluid grid
    if (x < nx && y < ny && z < nz) {

        const T val = dev_scalarfield[idx(x,y,z)];

        if (x == 0)
            dev_scalarfield[idx(nx,y,z)] = val;
        if (x == nx-1)
            dev_scalarfield[idx(-1,y,z)] = val;

        if (y == 0)
            dev_scalarfield[idx(x,ny,z)] = val;
        if (y == ny-1)
            dev_scalarfield[idx(x,-1,z)] = val;

        if (z == 0)
            dev_scalarfield[idx(x,y,-1)] = val;     // Dirichlet
            //dev_scalarfield[idx(x,y,nz)] = val;    // Periodic -z
        if (z == nz-1)
            dev_scalarfield[idx(x,y,nz)] = val;     // Dirichlet
            //dev_scalarfield[idx(x,y,-1)] = val;    // Periodic +z
    }
}

// Update a scalar field for ghost nodes from their parent cell values. The edge
// (diagonal) cells are not written since they are not read. Launch this kernel
// for all cells in the grid.
__global__ void setNSghostNodes_v_prod(Float* dev_ns_v_prod)
{
    // 3D thread index
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;

    // Grid dimensions
    const unsigned int nx = devC_grid.num[0];
    const unsigned int ny = devC_grid.num[1];
    const unsigned int nz = devC_grid.num[2];

    // check that we are not outside the fluid grid
    if (x < nx && y < ny && z < nz) {

        // Linear index of length-6 vector field entry
        unsigned int cellidx6 = idx(x,y,z)*6;

        // Read parent values
        __syncthreads();
        const Float v0 = dev_ns_v_prod[cellidx6];
        const Float v1 = dev_ns_v_prod[cellidx6+1];
        const Float v2 = dev_ns_v_prod[cellidx6+2];
        const Float v3 = dev_ns_v_prod[cellidx6+3];
        const Float v4 = dev_ns_v_prod[cellidx6+4];
        const Float v5 = dev_ns_v_prod[cellidx6+5];

        if (x == 0) {
            cellidx6 = idx(nx,y,z)*6;
            dev_ns_v_prod[cellidx6] = v0;
            dev_ns_v_prod[cellidx6+1] = v1;
            dev_ns_v_prod[cellidx6+2] = v2;
            dev_ns_v_prod[cellidx6+3] = v3;
            dev_ns_v_prod[cellidx6+4] = v4;
            dev_ns_v_prod[cellidx6+5] = v5;
        }
        if (x == nx-1) {
            cellidx6 = idx(-1,y,z)*6;
            dev_ns_v_prod[cellidx6] = v0;
            dev_ns_v_prod[cellidx6+1] = v1;
            dev_ns_v_prod[cellidx6+2] = v2;
            dev_ns_v_prod[cellidx6+3] = v3;
            dev_ns_v_prod[cellidx6+4] = v4;
            dev_ns_v_prod[cellidx6+5] = v5;
        }

        if (y == 0) {
            cellidx6 = idx(x,ny,z)*6;
            dev_ns_v_prod[cellidx6] = v0;
            dev_ns_v_prod[cellidx6+1] = v1;
            dev_ns_v_prod[cellidx6+2] = v2;
            dev_ns_v_prod[cellidx6+3] = v3;
            dev_ns_v_prod[cellidx6+4] = v4;
            dev_ns_v_prod[cellidx6+5] = v5;
        }
        if (y == ny-1) {
            cellidx6 = idx(x,-1,z)*6;
            dev_ns_v_prod[cellidx6] = v0;
            dev_ns_v_prod[cellidx6+1] = v1;
            dev_ns_v_prod[cellidx6+2] = v2;
            dev_ns_v_prod[cellidx6+3] = v3;
            dev_ns_v_prod[cellidx6+4] = v4;
            dev_ns_v_prod[cellidx6+5] = v5;
        }

        if (z == 0) {
            cellidx6 = idx(x,y,nz)*6;
            dev_ns_v_prod[cellidx6] = v0;
            dev_ns_v_prod[cellidx6+1] = v1;
            dev_ns_v_prod[cellidx6+2] = v2;
            dev_ns_v_prod[cellidx6+3] = v3;
            dev_ns_v_prod[cellidx6+4] = v4;
            dev_ns_v_prod[cellidx6+5] = v5;
        }
        if (z == nz-1) {
            cellidx6 = idx(x,y,-1)*6;
            dev_ns_v_prod[cellidx6] = v0;
            dev_ns_v_prod[cellidx6+1] = v1;
            dev_ns_v_prod[cellidx6+2] = v2;
            dev_ns_v_prod[cellidx6+3] = v3;
            dev_ns_v_prod[cellidx6+4] = v4;
            dev_ns_v_prod[cellidx6+5] = v5;
        }
    }
}

// Update a the forcing values in the ghost nodes from their parent cell values.
// The edge (diagonal) cells are not written since they are not read. Launch
// this kernel for all cells in the grid.
__global__ void setNSghostNodesForcing(
        Float*  dev_ns_f1,
        Float3* dev_ns_f2,
        Float*  dev_ns_f,
        unsigned int nijac)

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
    unsigned int cellidx = idx(x,y,z);

    // check that we are not outside the fluid grid
    if (x < nx && y < ny && z < nz) {

        __syncthreads();
        const Float f  = dev_ns_f[cellidx];
        Float  f1;
        Float3 f2;

        if (nijac == 0) {
            __syncthreads();
            f1 = dev_ns_f1[cellidx];
            f2 = dev_ns_f2[cellidx];
        }

        if (x == 0) {
            cellidx = idx(nx,y,z);
            dev_ns_f[cellidx] = f;
            if (nijac == 0) {
                dev_ns_f1[cellidx] = f1;
                dev_ns_f2[cellidx] = f2;
            }
        }
        if (x == nx-1) {
            cellidx = idx(-1,y,z);
            dev_ns_f[cellidx] = f;
            if (nijac == 0) {
                dev_ns_f1[cellidx] = f1;
                dev_ns_f2[cellidx] = f2;
            }
        }

        if (y == 0) {
            cellidx = idx(x,ny,z);
            dev_ns_f[cellidx] = f;
            if (nijac == 0) {
                dev_ns_f1[cellidx] = f1;
                dev_ns_f2[cellidx] = f2;
            }
        }
        if (y == ny-1) {
            cellidx = idx(x,-1,z);
            dev_ns_f[cellidx] = f;
            if (nijac == 0) {
                dev_ns_f1[cellidx] = f1;
                dev_ns_f2[cellidx] = f2;
            }
        }

        if (z == 0) {
            cellidx = idx(x,y,nz);
            dev_ns_f[cellidx] = f;
            if (nijac == 0) {
                dev_ns_f1[cellidx] = f1;
                dev_ns_f2[cellidx] = f2;
            }
        }
        if (z == nz-1) {
            cellidx = idx(x,y,-1);
            dev_ns_f[cellidx] = f;
            if (nijac == 0) {
                dev_ns_f1[cellidx] = f1;
                dev_ns_f2[cellidx] = f2;
            }
        }
    }
}

// Find the porosity in each cell on the base of a sphere, centered at the cell
// center. 
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
        Float phi = 1.00;

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

        // Make sure that the porosity is in the interval [0.0;1.0]
        phi = fmin(1.00, fmax(0.00, void_volume/cell_volume));
        //phi = void_volume/cell_volume;

        Float dphi = phi - phi_0;
        if (iteration == 0) {
            // Do not use the initial porosity estimates
            dphi = 0.0;
        }

        // Save porosity and porosity change
        __syncthreads();
        phi = 1.0; dphi = 0.0; // disable porosity effects
        dev_ns_phi[idx(x,y,z)]  = phi;
        dev_ns_dphi[idx(x,y,z)] = dphi;
    }
}

// Find the gradient in a cell in a homogeneous, cubic 3D scalar field using
// finite central differences
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
    const Float p  = dev_scalarfield[idx(x,y,z)];
    const Float xp = dev_scalarfield[idx(x+1,y,z)];
    const Float xn = dev_scalarfield[idx(x-1,y,z)];
    const Float yp = dev_scalarfield[idx(x,y+1,z)];
    const Float yn = dev_scalarfield[idx(x,y-1,z)];
    const Float zp = dev_scalarfield[idx(x,y,z+1)];
    const Float zn = dev_scalarfield[idx(x,y,z-1)];

    //__syncthreads();
    //if (p != 0.0)
        //printf("p[%d,%d,%d] =\t%f\n", x,y,z, p);

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

// Find the outer product of v v
__global__ void findvvOuterProdNS(
        Float3* dev_ns_v,       // in
        Float*  dev_ns_v_prod)  // out
{
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;

    // 1D thread index
    const unsigned int cellidx6 = idx(x,y,z)*6;

    // Check that we are not outside the fluid grid
    if (x < devC_grid.num[0] && y < devC_grid.num[1] && z < devC_grid.num[2]) {

        __syncthreads();
        const Float3 v = dev_ns_v[idx(x,y,z)];

        // The outer product (v v) looks like:
        // [[ v_x^2    v_x*v_y  v_x*v_z ]
        //  [ v_y*v_x  v_y^2    v_y*v_z ]
        //  [ v_z*v_x  v_z*v_y  v_z^2   ]]

        // The tensor is symmetrical: value i,j = j,i.
        // Only the upper triangle is saved, with the cells given a linear index
        // enumerated as:
        // [[ 0 1 2 ]
        //  [   3 4 ]
        //  [     5 ]]

        __syncthreads();
        dev_ns_v_prod[cellidx6]   = v.x*v.x;
        dev_ns_v_prod[cellidx6+1] = v.x*v.y;
        dev_ns_v_prod[cellidx6+2] = v.x*v.z;
        dev_ns_v_prod[cellidx6+3] = v.y*v.y;
        dev_ns_v_prod[cellidx6+4] = v.y*v.z;
        dev_ns_v_prod[cellidx6+5] = v.z*v.z;
    }
}

// Find the divergence of phi v v
__global__ void findNSdivphivv(
        Float*  dev_ns_v_prod, // in
        Float*  dev_ns_phi,    // in
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

    // Cell sizes
    const Float dx = devC_grid.L[0]/nx;
    const Float dy = devC_grid.L[1]/ny;
    const Float dz = devC_grid.L[2]/nz;

    // 1D thread index
    const unsigned int cellidx = idx(x,y,z);

    // Check that we are not outside the fluid grid
    if (x < nx && y < ny && z < nz) {

        // Read cell and 6 neighbor cells
        __syncthreads();
        //const Float  phi    = dev_ns_phi[cellidx];
        const Float  phi_xp = dev_ns_phi[idx(x+1,y,z)];
        const Float  phi_xn = dev_ns_phi[idx(x-1,y,z)];
        const Float  phi_yp = dev_ns_phi[idx(x,y+1,z)];
        const Float  phi_yn = dev_ns_phi[idx(x,y-1,z)];
        const Float  phi_zp = dev_ns_phi[idx(x,y,z+1)];
        const Float  phi_zn = dev_ns_phi[idx(x,y,z-1)];

        // The tensor is symmetrical: value i,j = j,i.
        // Only the upper triangle is saved, with the cells given a linear index
        // enumerated as:
        // [[ 0 1 2 ]
        //  [   3 4 ]
        //  [     5 ]]

        // div(T) = 
        //  [ de_xx/dx + de_xy/dy + de_xz/dz ,
        //    de_yx/dx + de_yy/dy + de_yz/dz ,
        //    de_zx/dx + de_zy/dy + de_zz/dz ]

        // This function finds the divergence of (phi v v), which is a vector

        // Calculate the divergence. See
        // https://en.wikipedia.org/wiki/Divergence#Application_in_Cartesian_coordinates
        // The symmetry described in findvvOuterProdNS is used
        __syncthreads();
        const Float3 div = MAKE_FLOAT3(
                ((dev_ns_v_prod[idx(x+1,y,z)*6]*phi_xp
                  - dev_ns_v_prod[idx(x-1,y,z)*6]*phi_xn)/(2.0*dx) +
                 (dev_ns_v_prod[idx(x,y+1,z)*6+1]*phi_yp
                  - dev_ns_v_prod[idx(x,y-1,z)*6+1]*phi_yn)/(2.0*dy) +
                 (dev_ns_v_prod[idx(x,y,z+1)*6+2]*phi_zp
                  - dev_ns_v_prod[idx(x,y,z-1)*6+2]*phi_zn)/(2.0*dz)),
                ((dev_ns_v_prod[idx(x+1,y,z)*6+1]*phi_xp
                  - dev_ns_v_prod[idx(x-1,y,z)*6+1]*phi_xn)/(2.0*dx) +
                 (dev_ns_v_prod[idx(x,y+1,z)*6+3]*phi_yp
                  - dev_ns_v_prod[idx(x,y-1,z)*6+3]*phi_yn)/(2.0*dy) +
                 (dev_ns_v_prod[idx(x,y,z+1)*6+4]*phi_zp
                  - dev_ns_v_prod[idx(x,y,z-1)*6+4]*phi_zn)/(2.0*dz)),
                ((dev_ns_v_prod[idx(x+1,y,z)*6+2]*phi_xp
                  - dev_ns_v_prod[idx(x-1,y,z)*6+2]*phi_xn)/(2.0*dx) +
                 (dev_ns_v_prod[idx(x,y+1,z)*6+4]*phi_yp
                  - dev_ns_v_prod[idx(x,y-1,z)*6+4]*phi_yn)/(2.0*dy) +
                 (dev_ns_v_prod[idx(x,y,z+1)*6+5]*phi_zp
                  - dev_ns_v_prod[idx(x,y,z-1)*6+5]*phi_zn)/(2.0*dz)) );

        //printf("div[%d,%d,%d] = %f\t%f\t%f\n", x, y, z, div.x, div.y, div.z);

        // Write divergence
        __syncthreads();
        dev_ns_div_phi_v_v[cellidx] = div;
    }
}


// Find predicted fluid velocity
__global__ void findPredNSvelocities(
        Float*  dev_ns_p,               // in
        Float3* dev_ns_v,               // in
        Float*  dev_ns_phi,             // in
        Float3* dev_ns_div_phi_v_v,     // in
        Float3* dev_ns_v_p)             // out
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
        //const Float3 f_g
            //= MAKE_FLOAT3(devC_params.g[0], devC_params.g[1], devC_params.g[2])
            //* rho * dx*dy*dz * phi;
        const Float3 f_g = MAKE_FLOAT3(0.0, 0.0, 0.0);

        // Find pressure gradient
        const Float3 grad_p = gradient(dev_ns_p, x, y, z, dx, dy, dz);

        // Calculate the predicted velocity
        const Float3 v_p
            = v - devC_dt*div_phi_v_v
            - devC_dt*BETA/rho*phi*grad_p
            - devC_dt/rho*f_i
            + devC_dt*phi*f_g;

        // Save the predicted velocity
        //printf("[%d,%d,%d] v_p = %f\t%f\t%f,\tgrad_p = %f\t%f\t%f\n",
         //       x, y, z, v_p.x, v_p.y, v_p.z, grad_p.x, grad_p.y, grad_p.z);
        __syncthreads();
        dev_ns_v_p[cellidx] = v_p;
    }
}

// Find the value of the forcing function. Only grad(epsilon) changes during
// the Jacobi iterations. The remaining, constant terms are only calculated
// during the first iteration.
// The forcing function is:
//   f = (div(v_p)*rho)/dt
//     + (grad(phi) dot v_p*rho)/(dt*phi)
//     + (dphi*rho)/(dt*dt*phi)
//     - (grad(phi) dot grad(epsilon))/phi
// The following is calculated in the first Jacobi iteration and saved:
//   f1 = (div(v_p)*rho)/dt
//      + (grad(phi) dot v_p*rho)/(dt*phi)
//      + (dphi*rho)/(dt*dt*phi)
//   f2 = grad(phi)/phi
// At each iteration, the value of the forcing function is found as:
//   f = f1 - f2 dot grad(epsilon)
__global__ void findNSforcing(
        Float*  dev_ns_epsilon,
        Float*  dev_ns_f1,
        Float3* dev_ns_f2,
        Float*  dev_ns_f,
        Float*  dev_ns_phi,
        Float*  dev_ns_dphi,
        Float3* dev_ns_v_p,
        unsigned int nijac)
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

        // Constant forcing function terms
        Float f1;
        Float3 f2;

        // Check if this is the first Jacobi iteration. If it is, find f1 and f2
        if (nijac == 0) {

            // Read needed values
            __syncthreads();
            const Float3 v_p  = dev_ns_v_p[cellidx];
            const Float  phi  = dev_ns_phi[cellidx];
            const Float  dphi = dev_ns_dphi[cellidx];
            const Float  rho  = 1000.0;

            // Calculate derivatives
            const Float  div_v_p
                = divergence(dev_ns_v_p, x, y, z, dx, dy, dz);
            const Float3 grad_phi
                = gradient(dev_ns_phi, x, y, z, dx, dy, dz);

            // Find forcing function coefficients
            f1 = div_v_p*rho/devC_dt
                + dot(grad_phi, v_p)*rho/(devC_dt*phi)
                + dphi*rho/(devC_dt*devC_dt*phi);
            f2 = grad_phi/phi;

            // Save values
            //printf("[%d,%d,%d] dphi = %f\n", x,y,z, dphi);
            //printf("[%d,%d,%d] v_p = %f\tdiv_v_p = %f\n", x,y,z, v_p, div_v_p);
            __syncthreads();
            dev_ns_f1[cellidx] = f1;
            dev_ns_f2[cellidx] = f2;

        } else {

            // Read previously found values
            __syncthreads();
            f1 = dev_ns_f1[cellidx];
            f2 = dev_ns_f2[cellidx];
        }

        // Find the gradient of epsilon, which changes during Jacobi iterations
        const Float3 grad_epsilon
            = gradient(dev_ns_epsilon, x, y, z, dx, dy, dz);

        // Forcing function value
        const Float f = f1 - dot(f2, grad_epsilon);
        //printf("[%d,%d,%d] f1 = %f\tf2 = %f\tf = %f\n", x,y,z, f1, f2, f);

        // Save forcing function value
        __syncthreads();
        dev_ns_f[cellidx] = f;
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
    //if (x < nx && y < ny && z < nz) {

    // internal nodes only
    //if (x > 0 && x < nx-1 && y > 0 && y < ny-1 && z > 0 && z < nz-1) {

    // Perform the epsilon updates for all non-ghost nodes except the Dirichlet
    // boundaries at z=0 and z=nz-1
    if (x < nx && y < ny && z > 0 && z < nz-1) {

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
        //const Float f = 0.0;

        // New value of epsilon in 3D update
        const Float dxdx = dx*dx;
        const Float dydy = dy*dy;
        const Float dzdz = dz*dz;
        const Float e_new
            = (-dxdx*dydy*dzdz*f
                    + dydy*dzdz*(e_xn + e_xp)
                    + dxdx*dzdz*(e_yn + e_yp)
                    + dxdx*dydy*(e_zn + e_zp))
            /(2.0*(dxdx*dydy + dxdx*dzdz + dydy*dzdz));

        // New value of epsilon in 1D update
        //const Float e_new = (e_zp + e_zn - dz*dz*f)/2.0;

        //if (z == 0 || z == nz-1)
            //e_new = 0.0;

        // Find the normalized residual value. A small value is added to the
        // denominator to avoid a divide by zero.
        const Float res_norm = (e_new - e)*(e_new - e)/(e_new*e_new + 1.0e-16);

        // Write results
        __syncthreads();
        dev_ns_epsilon_new[cellidx] = e_new;
        dev_ns_norm[cellidx] = res_norm;
    }
}

///////////////////////////////// UNUSED ///////////////////////////////////////
// Copy all values from one array to the other
template<typename T>
__global__ void copyValues(
        T* dev_read,
        T* dev_write)
{
    // 3D thread index
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;

    // 1D thread index
    const unsigned int cellidx = idx(x,y,z);

    // Check that we are not outside the fluid grid
    if (x < devC_grid.num[0] && y < devC_grid.num[1] && z < devC_grid.num[2]) {

        // Read
        __syncthreads();
        const T val = dev_read[cellidx];

        //printf("[%d,%d,%d] = %f\n", x, y, z, val);

        // Write
        __syncthreads();
        dev_write[cellidx] = val;
    }
}
///////////////////////////////// UNUSED END ///////////////////////////////////

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
        Float p = BETA*p_old + epsilon;

        // Find corrector gradient
        const Float3 grad_epsilon
            = gradient(dev_ns_epsilon, x, y, z, dx, dy, dz);

        // Fluid density
        const Float rho = 1000.0;

        // Find new velocity
        Float3 v = v_p - devC_dt/rho*grad_epsilon;

        /*if (z == 0 || z == nz-1) {
            p = p_old;
            v.z = 0.0;
        }*/

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
