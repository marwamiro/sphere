#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "typedefs.h"
#include "datatypes.h"
#include "constants.h"
#include "sphere.h"
#include "utility.h"

// Initialize memory
void DEM::initDarcyMem()
{
    //unsigned int ncells = d_nx*d_ny*d_nz; // without ghost nodes
    unsigned int ncells = (d_nx+2)*(d_ny+2)*(d_nz+2); // with ghost nodes
    d_H     = new Float[ncells];  // hydraulic pressure matrix
    d_H_new = new Float[ncells];  // hydraulic pressure matrix
    d_V     = new Float3[ncells]; // Cell hydraulic velocity
    d_dH    = new Float3[ncells]; // Cell gradient in hydraulic pressures
    d_K     = new Float[ncells];  // hydraulic conductivity matrix
    d_T     = new Float3[ncells]; // hydraulic transmissivity matrix
    d_Ss    = new Float[ncells];  // hydraulic storativity matrix
    d_W     = new Float[ncells];  // hydraulic recharge
    d_phi   = new Float[ncells];  // cell porosity
}

// Free memory
void DEM::freeDarcyMem()
{
    free(d_H);
    free(d_H_new);
    free(d_V);
    free(d_dH);
    free(d_K);
    free(d_T);
    free(d_Ss);
    free(d_W);
    free(d_phi);
}

// 3D index to 1D index
unsigned int DEM::idx(
        const unsigned int x,
        const unsigned int y,
        const unsigned int z)
{
    // without ghost nodes
    //return x + d_nx*y + d_nx*d_ny*z;

    // with ghost nodes
    // the ghost nodes are placed at -1 and WIDTH
    return (x+1) + (d_nx+2)*(y+1) + (d_nx+2)*(d_ny+2)*(z+1);
}

// Set initial values
void DEM::initDarcyVals()
{
    // Hydraulic permeability [m^2]
    const Float k = 1.0e-10;

    // Density of the fluid [kg/m^3]
    const Float rho = 1000.0;

    unsigned int ix, iy, iz, cellidx;
    for (ix=0; ix<d_nx; ++ix) {
        for (iy=0; iy<d_ny; ++iy) {
            for (iz=0; iz<d_nz; ++iz) {

                cellidx = idx(ix,iy,iz);

                // Initial hydraulic head [m]
                //d_H[cellidx] = 1.0;
                // read from input binary

                // Hydraulic permeability [m^2]
                d_K[cellidx] = k*rho*-params.g[2]/params.nu;

                // Hydraulic storativity [-]
                d_Ss[cellidx] = 8.0e-3;
                //d_Ss[cellidx] = 1.0;

                // Hydraulic recharge [Pa/s]
                d_W[cellidx] = 0.0;
            }
        }
    }
}


// Copy values from cell with index 'read' to cell with index 'write'
void DEM::copyDarcyVals(unsigned int read, unsigned int write)
{
    d_H[write]     = d_H[read];
    d_H_new[write] = d_H_new[read];
    d_V[write]     = MAKE_FLOAT3(d_V[read].x, d_V[read].y, d_V[read].z);
    d_dH[write]    = MAKE_FLOAT3(d_dH[read].x, d_dH[read].y, d_dH[read].z);
    d_K[write]     = d_K[read];
    d_T[write]     = MAKE_FLOAT3(d_T[read].x, d_T[read].y, d_T[read].z);
    d_Ss[write]    = d_Ss[read];
    d_W[write]     = d_W[read];
    d_phi[write]   = d_phi[read];
}

// Update ghost nodes from their parent cell values
// The edge (diagonal) cells are not written since they are not read
void DEM::setDarcyGhostNodes()
{
    unsigned int ix, iy, iz;

    // The x-normal plane
    for (iy=0; iy<d_ny; ++iy) {
        for (iz=0; iz<d_nz; ++iz) {

            // Ghost nodes at x=-1
            copyDarcyVals(
                    idx(d_nx-1,iy,iz),  // Read from this cell
                    idx(-1,iy,iz));     // Copy to this cell

            // Ghost nodes at x=d_nx
            copyDarcyVals(
                    idx(0,iy,iz),
                    idx(d_nx,iy,iz));
        }
    }

    // The y-normal plane
    for (ix=0; ix<d_nx; ++ix) {
        for (iz=0; iz<d_nz; ++iz) {

            // Ghost nodes at y=-1
            copyDarcyVals(
                    idx(ix,d_ny-1,iz),
                    idx(ix,-1,iz));

            // Ghost nodes at y=d_ny
            copyDarcyVals(
                    idx(ix,0,iz),
                    idx(ix,d_ny,iz));
        }
    }

    // The z-normal plane
    for (ix=0; ix<d_nx; ++ix) {
        for (iy=0; iy<d_ny; ++iy) {

            // Ghost nodes at z=-1
            copyDarcyVals(
                    idx(ix,iy,d_nz-1),
                    idx(ix,iy,-1));

            // Ghost nodes at z=d_nz
            copyDarcyVals(
                    idx(ix,iy,0),
                    idx(ix,iy,d_nz));
        }
    }
}

// Find cell transmissivities from hydraulic conductivities and cell dimensions
void DEM::findDarcyTransmissivities()
{
    // Find porosities from cell particle content
    findPorosities();

    // Density of the fluid [kg/m^3]
    const Float rho = 1000.0;

    // Kozeny-Carman parameter
    //Float a = 1.0e-8;
    Float a = 1.0;

    unsigned int ix, iy, iz, cellidx;
    Float K, k;
    for (ix=0; ix<d_nx; ++ix) {
        for (iy=0; iy<d_ny; ++iy) {
            for (iz=0; iz<d_nz; ++iz) {

                cellidx = idx(ix,iy,iz);

                // Read cell porosity
                Float phi = d_phi[cellidx];

                // Calculate permeability from the Kozeny-Carman relationship
                // Nelson 1994 eq. 1c
                // Boek 2012 eq. 16
                k = a*phi*phi*phi/(1.0 - phi*phi);

                // Save hydraulic conductivity [m/s]
                //K = d_K[cellidx];
                //K = k*rho*-params.g[2]/params.nu;
                K = 0.5; 
                d_K[cellidx] = K;

                // Hydraulic transmissivity [m2/s]
                Float3 T = {K*d_dx, K*d_dy, K*d_dz};
                d_T[cellidx] = T;
            }
        }
    }
}

// Set the gradient to 0.0 in all dimensions at the boundaries
// Unused
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
            d_dH[idx(ix,iy, 0)] = z3;
            d_dH[idx(ix,iy,nz)] = z3;
        }
    }

    // x-z plane at y=0 and y=d_dy-1
    for (ix=0; ix<d_nx; ++ix) {
        for (iz=0; iz<d_nz; ++iz) {
            d_dH[idx(ix, 0,iz)] = z3;
            d_dH[idx(ix,ny,iz)] = z3;
        }
    }

    // y-z plane at x=0 and x=d_dx-1
    for (iy=0; iy<d_ny; ++iy) {
        for (iz=0; iz<d_nz; ++iz) {
            d_dH[idx( 0,iy,iz)] = z3;
            d_dH[idx(nx,iy,iz)] = z3;
        }
    }
}


// Find the spatial gradient in pressures per cell
void DEM::findDarcyGradients()
{
    // Cell sizes squared
    //const Float dx2 = d_dx*d_dx;
    //const Float dx2 = d_dx*d_dx;
    //const Float dy2 = d_dy*d_dy;
    const Float dx2 = 2.0*d_dx;
    const Float dy2 = 2.0*d_dy;
    const Float dz2 = 2.0*d_dz;

    //Float H;
    unsigned int ix, iy, iz, cellidx;

    // Without ghost-nodes
    /*for (ix=1; ix<d_nx-1; ++ix) {
        for (iy=1; iy<d_ny-1; ++iy) {
            for (iz=1; iz<d_nz-1; ++iz) {*/

    // With ghost-nodes
    for (ix=0; ix<d_nx; ++ix) {
        for (iy=0; iy<d_ny; ++iy) {
            for (iz=1; iz<d_nz-1; ++iz) {

                cellidx = idx(ix,iy,iz);

                //H = d_H[cellidx];   // cell hydraulic pressure

                // First order central differences
                // x-boundary
                d_dH[cellidx].x
                 = (d_H[idx(ix+1,iy,iz)] - d_H[idx(ix-1,iy,iz)])/dx2;
                 //= (d_H[idx(ix+1,iy,iz)] - 2.0*H + d_H[idx(ix-1,iy,iz)])/dx2;

                // y-boundary
                d_dH[cellidx].y
                 = (d_H[idx(ix,iy+1,iz)] - d_H[idx(ix,iy-1,iz)])/dy2;
                 //= (d_H[idx(ix,iy+1,iz)] - 2.0*H + d_H[idx(ix,iy-1,iz)])/dy2;

                // z-boundary
                d_dH[cellidx].z
                 = (d_H[idx(ix,iy,iz+1)] - d_H[idx(ix,iy,iz-1)])/dz2;
                 //= (d_H[idx(ix,iy,iz+1)] - 2.0*H + d_H[idx(ix,iy,iz-1)])/dz2;

                /*
                // Periodic x boundaries
                if (ix == 0) {
                    d_dH[cellidx].x = (d_H[idx(ix+1,iy,iz)]
                                - 2.0*H + d_H[idx(d_nx-1,iy,iz)])/dx2;
                } else if (ix == d_nx-1) {
                    d_dH[cellidx].x = (d_H[idx(0,iy,iz)]
                                - 2.0*H + d_H[idx(ix-1,iy,iz)])/dx2;
                } else {
                    d_dH[cellidx].x = (d_H[idx(ix+1,iy,iz)]
                                - 2.0*H + d_H[idx(ix-1,iy,iz)])/dx2;
                }
                
                // Periodic y boundaries
                if (iy == 0) {
                    d_dH[cellidx].y = (d_H[idx(ix,iy+1,iz)]
                                - 2.0*H + d_H[idx(ix,d_ny-1,iz)])/dy2;
                } else if (iy == d_ny-1) {
                    d_dH[cellidx].y = (d_H[idx(ix,0,iz)]
                                - 2.0*H + d_H[idx(ix,iy-1,iz)])/dy2;
                } else {
                    d_dH[cellidx].y = (d_H[idx(ix,iy+1,iz)]
                            - 2.0*H + d_H[idx(ix,iy-1,iz)])/dy2;
                }*/

            }
        }
    }
}

// Arithmetic mean of two numbers
Float amean(Float a, Float b) {
    return (a+b)*0.5;
}

// Harmonic mean of two numbers
Float hmean(Float a, Float b) {
    return (2.0*a*b)/(a+b);
}

// Perform an explicit step.
// Boundary conditions are fixed values (Dirichlet)
void DEM::explDarcyStep()
{

    // Find transmissivities from cell particle content
    findDarcyTransmissivities();

    // Check the time step length
    checkDarcyTimestep();

    // Cell dims squared
    const Float dx2 = d_dx*d_dx;
    const Float dy2 = d_dy*d_dy;
    const Float dz2 = d_dz*d_dz;

    //setDarcyBCNeumannZero();

    // Update ghost node values from their parent cell values
    setDarcyGhostNodes();

    // Explicit 3D finite difference scheme
    // new = old + production*timestep + gradient*timestep
    unsigned int ix, iy, iz, cellidx;
    Float K, H, deltaH;
    Float Tx, Ty, Tz, S;
    //Float Tx_n, Tx_p, Ty_n, Ty_p, Tz_n, Tz_p;
    Float gradx_n, gradx_p, grady_n, grady_p, gradz_n, gradz_p;
    for (ix=0; ix<d_nx; ++ix) {
        for (iy=0; iy<d_ny; ++iy) {
            for (iz=0; iz<d_nz; ++iz) {

                // Cell linear index
                cellidx = idx(ix,iy,iz);

                // If x,y,z boundaries are fixed values:
                // Enforce Dirichlet BC
                if (ix == 0 || iy == 0 || iz == 0 ||
                        ix == d_nx-1 || iy == d_ny-1 || iz == d_nz-1) {
                    d_H_new[cellidx] = d_H[cellidx];
                // If z boundaries are periodic:
                //if (iz == 0 || iz == d_nz-1) {
                    //d_H_new[cellidx] = d_H[cellidx];
                } else {

                    // Cell hydraulic conductivity
                    K = d_K[cellidx];

                    // Cell hydraulic transmissivities
                    Tx = K*d_dx;
                    Ty = K*d_dy;
                    Tz = K*d_dz;

                    // Cell hydraulic head
                    H = d_H[cellidx];

                    // Harmonic mean of transmissivity
                    // (in neg. and pos. direction along axis from cell)
                    // with periodic x and y boundaries
                    // without ghost nodes
                    /*
                    if (ix == 0)
                        gradx_n = hmean(Tx, d_T[idx(d_nx-1,iy,iz)].x)
                            * (d_H[idx(d_nx-1,iy,iz)] - H)/dx2;
                    else
                        gradx_n = hmean(Tx, d_T[idx(ix-1,iy,iz)].x)
                            * (d_H[idx(ix-1,iy,iz)] - H)/dx2;

                    if (ix == d_nx-1)
                        gradx_p = hmean(Tx, d_T[idx(0,iy,iz)].x)
                            * (d_H[idx(0,iy,iz)] - H)/dx2;
                    else
                        gradx_p = hmean(Tx, d_T[idx(ix+1,iy,iz)].x)
                            * (d_H[idx(ix+1,iy,iz)] - H)/dx2;

                    if (iy == 0)
                        grady_n = hmean(Ty, d_T[idx(ix,d_ny-1,iz)].y)
                            * (d_H[idx(ix,d_ny-1,iz)] - H)/dy2;
                    else
                        grady_n = hmean(Ty, d_T[idx(ix,iy-1,iz)].y)
                            * (d_H[idx(ix,iy-1,iz)] - H)/dy2;

                    if (iy == d_ny-1)
                        grady_p = hmean(Ty, d_T[idx(ix,0,iz)].y)
                            * (d_H[idx(ix,0,iz)] - H)/dy2;
                    else
                        grady_p = hmean(Ty, d_T[idx(ix,iy+1,iz)].y)
                            * (d_H[idx(ix,iy+1,iz)] - H)/dy2;
                            */

                    gradx_n = hmean(Tx, d_T[idx(ix-1,iy,iz)].x)
                        * (d_H[idx(ix-1,iy,iz)] - H)/dx2;
                    gradx_p = hmean(Tx, d_T[idx(ix+1,iy,iz)].x)
                        * (d_H[idx(ix+1,iy,iz)] - H)/dx2;

                    grady_n = hmean(Ty, d_T[idx(ix,iy-1,iz)].y)
                        * (d_H[idx(ix,iy-1,iz)] - H)/dy2;
                    grady_p = hmean(Ty, d_T[idx(ix,iy+1,iz)].y)
                        * (d_H[idx(ix,iy+1,iz)] - H)/dy2;

                    gradz_n = hmean(Tz, d_T[idx(ix,iy,iz-1)].z)
                        * (d_H[idx(ix,iy,iz-1)] - H)/dz2;
                    gradz_p = hmean(Tz, d_T[idx(ix,iy,iz+1)].z)
                        * (d_H[idx(ix,iy,iz+1)] - H)/dz2;

                    /*std::cerr << ix << ',' << iy << ',' << iz << '\t'
                        << H << '\t' << Tx << ',' << Ty << ',' << Tz << '\t'
                        << gradx_n << ',' << gradx_p << '\t'
                        << grady_n << ',' << grady_p << '\t'
                        << gradz_n << ',' << gradz_p << std::endl;*/

                    // Cell hydraulic storativity
                    S = d_Ss[cellidx]*d_dx*d_dy*d_dz;

                    // Laplacian operator
                    deltaH = time.dt/S *
                        (  gradx_n + gradx_p
                         + grady_n + grady_p
                         + gradz_n + gradz_p
                         + d_W[cellidx] );

                    // Calculate new hydraulic pressure in cell
                    d_H_new[cellidx] = H + deltaH;
                }
            }
        }
    }

    // Swap d_H and d_H_new
    Float* tmp = d_H;
    d_H = d_H_new;
    d_H_new = tmp;

    // Find macroscopic cell fluid velocities
    findDarcyVelocities();
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
void DEM::printDarcyArray(FILE* stream, Float3* arr)
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
void DEM::printDarcyArray(FILE* stream, Float3* arr, std::string desc)
{
    std::cout << "\n" << desc << ":\n";
    printDarcyArray(stream, arr);
}

// Find cell velocity
void DEM::findDarcyVelocities()
{
    // Flux [m/s]: q = -k/nu * dH
    // Pore velocity [m/s]: v = q/n
    Float3 q, v, dH;

    // Dynamic viscosity
    Float nu = params.nu;

    // Porosity [-]: n

    // Find cell gradients
    findDarcyGradients();

    unsigned int ix, iy, iz, cellidx;
    for (ix=0; ix<d_nx; ++ix) {
        for (iy=0; iy<d_ny; ++iy) {
            for (iz=0; iz<d_nz; ++iz) {
                
                cellidx = idx(ix,iy,iz);
                dH = d_dH[cellidx];

                // Approximate cell porosity
                Float phi = d_phi[cellidx];

                // Calculate flux
                // The sign might need to be reversed, depending on the
                // grid orientation
                q.x = -d_K[cellidx]/nu * dH.x;
                q.y = -d_K[cellidx]/nu * dH.y;
                q.z = -d_K[cellidx]/nu * dH.z;
                
                // Calculate velocity
                v.x = q.x/phi;
                v.y = q.y/phi;
                v.z = q.z/phi;
                d_V[cellidx] = v;
            }
        }
    }
}

// Return the lower corner coordinates of a cell
Float3 DEM::cellMinBoundaryDarcy(
        const unsigned int x,
        const unsigned int y,
        const unsigned int z)
{
    const Float3 x_min = {x*d_dx, y*d_dy, z*d_dz};
    return x_min;
}

// Return the upper corner coordinates of a cell
Float3 DEM::cellMaxBoundaryDarcy(
        const unsigned int x,
        const unsigned int y,
        const unsigned int z)
{
    const Float3 x_max = {(x+1)*d_dx, (y+1)*d_dy, (z+1)*d_dz};
    return x_max;
}

// Return the volume of a cell
Float DEM::cellVolumeDarcy()
{
    const Float cell_volume = d_dx*d_dy*d_dz;
    return cell_volume;
}

// Find the porosity of a target cell
Float DEM::cellPorosity(
        const unsigned int x,
        const unsigned int y,
        const unsigned int z)
{
    const Float3 x_min = cellMinBoundaryDarcy(x,y,z);
    const Float3 x_max = cellMaxBoundaryDarcy(x,y,z);
    Float cell_volume = cellVolumeDarcy();
    Float void_volume = cell_volume;

    unsigned int i;
    Float4 xr;
    for (i=0; i<np; ++i) {

        // Read the position and radius
        xr = k.x[i];

        if (xr.x >= x_min.x && xr.y >= x_min.y && xr.z >= x_min.z
                && xr.x < x_max.x && xr.y < x_max.y && xr.z < x_max.z) {
            void_volume -= 4.0/3.0*M_PI*xr.w*xr.w*xr.w;
        }
    }

    // Return the porosity, which should always be ]0.0;1.0[
    Float phi = fmin(0.99, fmax(0.01, void_volume/cell_volume));
    phi = 0.5;
    return phi;
}

void DEM::findPorosities()
{
    unsigned int ix, iy, iz, cellidx;
    for (ix=0; ix<d_nx; ++ix) {
        for (iy=0; iy<d_ny; ++iy) {
            for (iz=0; iz<d_nz; ++iz) {
                d_phi[idx(ix,iy,iz)] = cellPorosity(ix,iy,iz);
            }
        }
    }
}

// Find particles with centres inside a spatial interval
// NOTE: This function is untested and unused
std::vector<unsigned int> DEM::particlesInCell(
        const Float3 min, const Float3 max)
{
    // Particles radii inside cell will be stored in this vector
    std::vector<unsigned int> pidx;

    unsigned int i;
    Float4 x;
    for (i=0; i<np; ++i) {

        // Read the position
        x = k.x[i];

        if (x.x >= min.x && x.y >= min.y && x.z >= min.z
                && x.x < max.x && x.y < max.y && x.z < max.z) {
            pidx.push_back(i);
        }
    }
}

// Add fluid drag to the particles inside each cell
void DEM::fluidDragDarcy()
{
    /*unsigned int ix, iy, iz, cellidx;
    for (ix=0; ix<d_nx; ++ix) {
        for (iy=0; iy<d_ny; ++iy) {
            for (iz=0; iz<d_nz; ++iz) {


            }
        }
    }*/
}

// Get maximum value in 3d array with ghost nodes
Float DEM::getTmax()
{
    Float max = -1.0e13; // initialize with a small number
    unsigned int ix,iy,iz;
    Float3 val;
    for (ix=0; ix<d_nx; ++ix) {
        for (iy=0; iy<d_ny; ++iy) {
            for (iz=0; iz<d_nz; ++iz) {
                val = d_T[idx(ix,iy,iz)];
                if (val.x > max)
                    max = val.x;
                if (val.y > max)
                    max = val.y;
                if (val.z > max)
                    max = val.z;
            }
        }
    }
    return max;
}
// Get maximum value in 1d array with ghost nodes
Float DEM::getSsmin()
{
    Float min = 1.0e13; // initialize with a small number
    unsigned int ix,iy,iz;
    Float val;
    for (ix=0; ix<d_nx; ++ix) {
        for (iy=0; iy<d_ny; ++iy) {
            for (iz=0; iz<d_nz; ++iz) {
                val = d_Ss[idx(ix,iy,iz)];
                if (val < min)
                    min = val;
            }
        }
    }
    return min;
}


// Check whether the time step length is sufficient for keeping the explicit
// time stepping method stable
void DEM::checkDarcyTimestep()
{
    Float T_max = getTmax();
    Float S_min = getSsmin()*d_dx*d_dy*d_dz;

    // Use numerical criterion from Sonnenborg & Henriksen 2005
    Float value = T_max/S_min
        * (time.dt/(d_dx*d_dx) + time.dt/(d_dy*d_dy) + time.dt/(d_dz*d_dz));

    if (value > 0.5) {
        std::cerr << "Error! The explicit darcy solution will be unstable.\n"
            << "This happens due to a combination of the following:\n"
            << " - The transmissivity T (i.e. hydraulic conductivity, K) is too large"
            << " (" << T_max << ")\n"
            << " - The storativity S is too small"
            << " (" << S_min << ")\n"
            << " - The time step is too large"
            << " (" << time.dt << ")\n"
            << " - The cell dimensions are too small\n"
            << " Reason: (" << value << " > 0.5)"
            << std::endl;
        exit(1);
    }
}

// Solve Darcy flow on a regular, cubic grid
void DEM::initDarcy(const Float cellsizemultiplier)
{
    if (params.nu <= 0.0) {
        std::cerr << "Error in initDarcy. The dymamic viscosity (params.nu), "
            << "should be larger than 0.0, but is " << params.nu << std::endl;
        exit(1);
    }

    // Number of cells
    d_nx = floor(grid.num[0]*cellsizemultiplier);
    d_ny = floor(grid.num[1]*cellsizemultiplier);
    d_nz = floor(grid.num[2]*cellsizemultiplier);

    // Cell size 
    d_dx = grid.L[0]/d_nx;
    d_dy = grid.L[1]/d_ny;
    d_dz = grid.L[2]/d_nz;

    if (verbose == 1) {
        std::cout << "  - Fluid grid dimensions: "
            << d_nx << "*"
            << d_ny << "*"
            << d_nz << std::endl;
        std::cout << "  - Fluid grid cell size: "
            << d_dx << "*"
            << d_dy << "*"
            << d_dz << std::endl;
    }

    initDarcyMem();
    initDarcyVals();
    findDarcyTransmissivities();

    checkDarcyTimestep();
}

// Print final heads and free memory
void DEM::endDarcy()
{
    FILE* Kfile;
    if ((Kfile = fopen("d_K.txt","w"))) {
        printDarcyArray(Kfile, d_K);
        fclose(Kfile);
    } else {
        fprintf(stderr, "Error, could not open d_K.txt\n");
    }
    printDarcyArray(stdout, d_phi, "d_phi");
    printDarcyArray(stdout, d_K, "d_K");
    //printDarcyArray(stdout, d_H, "d_H");
    //printDarcyArray(stdout, d_V, "d_V");
    freeDarcyMem();
}

// vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
