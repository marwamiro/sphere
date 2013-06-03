#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "typedefs.h"
#include "datatypes.h"
#include "constants.h"
#include "sphere.h"

//#include "eigen-nvcc/Eigen/Core"

// Initialize memory
void DEM::initDarcyMem()
{
    unsigned int ncells = d_nx*d_ny*d_nz;
    d_P = new Float[ncells]; // hydraulic pressure matrix
    d_dP = new Float3[ncells]; // Cell spatial gradient in hydraulic pressures
    d_K = new Float[ncells]; // hydraulic conductivity matrix
    d_S = new Float[ncells]; // hydraulic storativity matrix
    d_W = new Float[ncells]; // hydraulic recharge
}

// Free memory
void DEM::freeDarcyMem()
{
    free(d_P);
    free(d_dP);
    free(d_K);
    free(d_S);
    free(d_W);
}

// 3D index to 1D index
unsigned int DEM::idx(
        const unsigned int x,
        const unsigned int y,
        const unsigned int z)
{
    return x + d_nx*y + d_nx*d_ny*z;
}

// Set initial values
void DEM::initDarcyVals()
{
    unsigned int ix, iy, iz;
    for (ix=0; ix<d_nx; ++ix) {
        for (iy=0; iy<d_ny; ++iy) {
            for (iz=0; iz<d_nz; ++iz) {
                d_P[idx(ix,iy,iz)] = 1.0;
                d_K[idx(ix,iy,iz)] = 1.5;
                d_S[idx(ix,iy,iz)] = 7.5e-3;
                d_W[idx(ix,iy,iz)] = 0.0;
            }
        }
    }
}

Float DEM::minVal3dArr(Float* arr)
{
    Float minval = 1e16; // a large val
    Float val;
    unsigned int ix, iy, iz;
    for (ix=0; ix<d_nx; ++ix) {
        for (iy=0; iy<d_ny; ++iy) {
            for (iz=0; iz<d_nz; ++iz) {
                val = arr[idx(ix,iy,iz)];
                if (minval > val)
                    minval = val;
            }
        }
    }
}

// Find the spatial gradient in pressures per cell
void DEM::findDarcyGradients()
{

    std::cout << "dx,dy,dz: "
        << d_dx << ","
        << d_dy << ","
        << d_dz << std::endl;
    const Float dx2 = d_dx*d_dx;
    const Float dy2 = d_dy*d_dy;
    const Float dz2 = d_dz*d_dz;
    std::cout << "dx2,dy2,dz2: "
        << dx2 << ","
        << dy2 << ","
        << dz2 << std::endl;
    Float localP2;

    unsigned int ix, iy, iz;
    for (ix=1; ix<d_nx-1; ++ix) {
        for (iy=1; iy<d_ny-1; ++iy) {
            for (iz=1; iz<d_nz-1; ++iz) {

                localP2 = 2.0*d_P[idx(ix,iy,iz)];

                d_dP[idx(ix,iy,iz)].x
                    = (d_P[idx(ix+1,iy,iz)] - localP2
                            + d_P[idx(ix-1,iy,iz)])/dx2;
                
                d_dP[idx(ix,iy,iz)].y
                    = (d_P[idx(ix,iy+1,iz)] - localP2
                            + d_P[idx(ix,iy-1,iz)])/dx2;

                d_dP[idx(ix,iy,iz)].z
                    = (d_P[idx(ix,iy,iz+1)] - localP2
                            + d_P[idx(ix,iy,iz-1)])/dz2;
            }
        }
    }
}

// Set the gradient to 0.0 in all dimensions at the boundaries
void DEM::setDarcyBCNeumannZero()
{
    Float3 z3 = MAKE_FLOAT3(0.0, 0.0, 0.0);
    unsigned int ix, iy, iz;
    unsigned int nx = d_nx-1;
    unsigned int ny = d_ny-1;
    unsigned int nz = d_nz-1;

    // I don't care that the values at four edges are written twice

    // x-y plane at z=0 and z=d_dz-1
    for (ix=0; ix<d_nx; ++ix) {
        for (iy=0; iy<d_ny; ++iy) {
            d_dP[idx(ix,iy, 0)] = z3;
            d_dP[idx(ix,iy,nz)] = z3;
        }
    }

    // x-z plane at y=0 and y=d_dy-1
    for (ix=0; ix<d_nx; ++ix) {
        for (iz=0; iz<d_nz; ++iz) {
            d_dP[idx(ix, 0,iz)] = z3;
            d_dP[idx(ix,ny,iz)] = z3;
        }
    }

    // y-z plane at x=0 and x=d_dx-1
    for (iy=0; iy<d_ny; ++iy) {
        for (iz=0; iz<d_nz; ++iz) {
            d_dP[idx( 0,iy,iz)] = z3;
            d_dP[idx(nx,iy,iz)] = z3;
        }
    }
}


void DEM::explDarcyStep(const Float dt)
{
    // Find spatial gradients in all cells
    //findDarcyGradients();
    //printDarcyArray3(stdout, d_dP, "d_dP, after findDarcyGradients");

    // Set boundary conditions
    //setDarcyBCNeumannZero();
    //printDarcyArray3(stdout, d_dP, "d_dP, after setDarcyBCNeumannZero");

    // Cell dims squared
    const Float dx2 = d_dx*d_dx;
    const Float dy2 = d_dy*d_dy;
    const Float dz2 = d_dz*d_dz;
    std::cout << "dx2,dy2,dz2: "
        << dx2 << ","
        << dy2 << ","
        << dz2 << std::endl;

    // Explicit 3D finite difference scheme
    // new = old + gradient*timestep
    unsigned int ix, iy, iz, cellidx;
    Float K, P;
    for (ix=1; ix<d_nx-1; ++ix) {
        for (iy=1; iy<d_ny-1; ++iy) {
            for (iz=1; iz<d_nz-1; ++iz) {

                cellidx = idx(ix,iy,iz);

                /*d_P[cellidx]
                    -= (d_W[cellidx]*dt
                    + d_K[cellidx]*d_dP[cellidx].x/d_dx
                    + d_K[cellidx]*d_dP[cellidx].y/d_dy
                    + d_K[cellidx]*d_dP[cellidx].z/d_dz) / d_S[cellidx];*/

                K = d_K[cellidx];   // cell hydraulic conductivity
                P = d_P[cellidx];   // cell hydraulic pressure

                d_P[cellidx]
                    += d_W[cellidx]*dt  // cell recharge
                    + K*dt *            // diffusivity term
                    (
                     (d_P[idx(ix+1,iy,iz)] - 2.0*P + d_P[idx(ix-1,iy,iz)])/dx2 +
                     (d_P[idx(ix,iy+1,iz)] - 2.0*P + d_P[idx(ix,iy-1,iz)])/dy2 +
                     (d_P[idx(ix,iy,iz+1)] - 2.0*P + d_P[idx(ix,iy,iz-1)])/dz2
                    );


            }
        }
    }
}

// Print array values to file stream (stdout, stderr, other file)
void DEM::printDarcyArray(FILE* stream, Float* arr)
{
    unsigned int x, y, z;
    for (z=0; z<d_nz; z++) {
        for (y=0; y<d_ny; y++) {
            for (x=0; x<d_nx; x++) {
                fprintf(stream, "%f\t", arr[idx(x,y,z)]);
            }
            fprintf(stream, "\n");
        }
        fprintf(stream, "\n");
    }
}

// Overload printDarcyArray to add optional description
void DEM::printDarcyArray(FILE* stream, Float* arr, std::string desc)
{
    std::cout << "\n" << desc << ":\n";
    printDarcyArray(stream, arr);
}

// Print array values to file stream (stdout, stderr, other file)
void DEM::printDarcyArray3(FILE* stream, Float3* arr)
{
    unsigned int x, y, z;
    for (z=0; z<d_nz; z++) {
        for (y=0; y<d_ny; y++) {
            for (x=0; x<d_nx; x++) {
                fprintf(stream, "%f,%f,%f\t",
                        arr[idx(x,y,z)].x,
                        arr[idx(x,y,z)].y,
                        arr[idx(x,y,z)].z);
            }
            fprintf(stream, "\n");
        }
        fprintf(stream, "\n");
    }
}

// Overload printDarcyArray to add optional description
void DEM::printDarcyArray3(FILE* stream, Float3* arr, std::string desc)
{
    std::cout << "\n" << desc << ":\n";
    printDarcyArray3(stream, arr);
}


// Find hydraulic conductivities for each cell by finding the particle contents
//

// Solve Darcy flow on a regular, cubic grid
void DEM::startDarcy(
        const Float cellsizemultiplier)
{
    // Number of cells
    d_nx = floor(grid.num[0]*cellsizemultiplier);
    d_ny = floor(grid.num[1]*cellsizemultiplier);
    d_nz = floor(grid.num[2]*cellsizemultiplier);

    // Cell size 
    Float d_dx = grid.L[0]/d_nx;
    Float d_dy = grid.L[1]/d_ny;
    Float d_dz = grid.L[2]/d_nz;

    if (verbose == 1) {
        std::cout << "  - Fluid grid dimensions: "
            << d_nx << " * "
            << d_ny << " * "
            << d_nz << std::endl;
        std::cout << "  - Fluid grid cell size: "
            << d_dx << " * "
            << d_dy << " * "
            << d_dz << std::endl;
    }

    initDarcyMem();
    initDarcyVals();

    // Temporal loop
    //while(time.current <= time.total) {

        explDarcyStep(time.dt);
        time.current += time.dt;
    //}


    printDarcyArray(stdout, d_P, "d_P");
    //printDarcyArray3(stdout, d_dP, "d_dP");
    //printDarcyArray(stdout, d_K, "d_K");
    //printDarcyArray(stdout, d_S, "d_S");
    //printDarcyArray(stdout, d_W, "d_W");

    freeDarcyMem();
}
