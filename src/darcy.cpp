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

// Enable line below to make x and y boundaries periodic
#define PERIODIC_XY

// Initialize memory
void DEM::initDarcyMem(const Float cellsizemultiplier)
{
    // Number of cells
    d.nx = floor(grid.num[0]*cellsizemultiplier);
    d.ny = floor(grid.num[1]*cellsizemultiplier);
    d.nz = floor(grid.num[2]*cellsizemultiplier);

    //unsigned int ncells = d.nx*d.ny*d.nz; // without ghost nodes
    unsigned int ncells = (d.nx+2)*(d.ny+2)*(d.nz+2); // with ghost nodes

    d.H     = new Float[ncells];  // hydraulic pressure matrix
    d.H_new = new Float[ncells];  // hydraulic pressure matrix
    d.V     = new Float3[ncells]; // Cell hydraulic velocity
    d.dH    = new Float3[ncells]; // Cell gradient in hydraulic pressures
    d.K     = new Float[ncells];  // hydraulic conductivity matrix
    d.T     = new Float3[ncells]; // hydraulic transmissivity matrix
    d.Ss    = new Float[ncells];  // hydraulic storativity matrix
    d.W     = new Float[ncells];  // hydraulic recharge
    d.phi   = new Float[ncells];  // cell porosity
}

// Free memory
void DEM::freeDarcyMem()
{
    delete[] d.H;
    delete[] d.H_new;
    delete[] d.V;
    delete[] d.dH;
    delete[] d.K;
    delete[] d.T;
    delete[] d.Ss;
    delete[] d.W;
    delete[] d.phi;
}

// 3D index to 1D index
unsigned int DEM::idx(
        const int x,
        const int y,
        const int z)
{
    // without ghost nodes
    //return x + d.nx*y + d.nx*d.ny*z;

    // with ghost nodes
    // the ghost nodes are placed at x,y,z = -1 and WIDTH
    return (x+1) + (d.nx+2)*(y+1) + (d.nx+2)*(d.ny+2)*(z+1);
}

// Set initial values
void DEM::initDarcyVals()
{
    // Hydraulic permeability [m^2]
    const Float k = 1.0e-10;

    // Density of the fluid [kg/m^3]
    const Float rho = 1000.0;

    int ix, iy, iz, cellidx;

    // Set values for all cells, including ghost nodes
    for (ix=-1; ix<=d.nx; ++ix) {
        for (iy=-1; iy<=d.ny; ++iy) {
            for (iz=-1; iz<=d.nz; ++iz) {

                cellidx = idx(ix,iy,iz);

                // Hydraulic storativity [-]
                //d.Ss[cellidx] = 1.0;
                d.Ss[cellidx] = 8.0e-3;

                // Hydraulic recharge [s^-1]
                d.W[cellidx] = 0.0;
            }
        }
    }

    // Extract water from all cells in center
    /*ix = d.nx/2; iy = d.ny/2;
    Float cellvolume = d.dx*d.dy*d.dz;
    for (iz=0; iz<d.nz; ++iz) {
        //d.W[idx(ix,iy,iz)] = -1.0e-4/cellvolume;
        d.W[idx(ix,iy,iz)] = -2.0e-3;
    }*/
}


// Copy values from cell with index 'read' to cell with index 'write'
void DEM::copyDarcyVals(unsigned int read, unsigned int write)
{
    d.H[write]     = d.H[read];
    d.H_new[write] = d.H_new[read];
    d.V[write]     = MAKE_FLOAT3(d.V[read].x, d.V[read].y, d.V[read].z);
    d.dH[write]    = MAKE_FLOAT3(d.dH[read].x, d.dH[read].y, d.dH[read].z);
    d.K[write]     = d.K[read];
    d.T[write]     = MAKE_FLOAT3(d.T[read].x, d.T[read].y, d.T[read].z);
    d.Ss[write]    = d.Ss[read];
    d.W[write]     = d.W[read];
    d.phi[write]   = d.phi[read];
}

// Update ghost nodes from their parent cell values
// The edge (diagonal) cells are not written since they are not read
void DEM::setDarcyGhostNodes()
{
    int ix, iy, iz;

    // The x-normal plane
    for (iy=0; iy<d.ny; ++iy) {
        for (iz=0; iz<d.nz; ++iz) {

            // Ghost nodes at x=-1
            copyDarcyVals(
                    idx(d.nx-1,iy,iz),  // Read from this cell
                    idx(-1,iy,iz));     // Copy to this cell

            // Ghost nodes at x=d.nx
            copyDarcyVals(
                    idx(0,iy,iz),
                    idx(d.nx,iy,iz));
        }
    }

    // The y-normal plane
    for (ix=0; ix<d.nx; ++ix) {
        for (iz=0; iz<d.nz; ++iz) {

            // Ghost nodes at y=-1
            copyDarcyVals(
                    idx(ix,d.ny-1,iz),
                    idx(ix,-1,iz));

            // Ghost nodes at y=d.ny
            copyDarcyVals(
                    idx(ix,0,iz),
                    idx(ix,d.ny,iz));
        }
    }

    // The z-normal plane
    for (ix=0; ix<d.nx; ++ix) {
        for (iy=0; iy<d.ny; ++iy) {

            // Ghost nodes at z=-1
            copyDarcyVals(
                    idx(ix,iy,d.nz-1),
                    idx(ix,iy,-1));

            // Ghost nodes at z=d.nz
            copyDarcyVals(
                    idx(ix,iy,0),
                    idx(ix,iy,d.nz));
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
    //Float a = 1.0;
    
    // Representative grain radius
    Float r_bar2 = meanRadius()*2.0;
    // Grain size factor for Kozeny-Carman relationship
    Float d_factor = r_bar2*r_bar2/180.0;

    int ix, iy, iz, cellidx;
    Float K, k;
    for (ix=0; ix<d.nx; ++ix) {
        for (iy=0; iy<d.ny; ++iy) {
            for (iz=0; iz<d.nz; ++iz) {

                cellidx = idx(ix,iy,iz);

                // Read cell porosity
                Float phi = d.phi[cellidx];

                // Calculate permeability from the Kozeny-Carman relationship
                // Nelson 1994 eq. 1c
                // Boek 2012 eq. 16
                //k = a*phi*phi*phi/(1.0 - phi*phi);
                // Schwartz and Zhang 2003
                k = phi*phi*phi/((1.0-phi)*(1.0-phi)) * d_factor;

                // Save hydraulic conductivity [m/s]
                //K = d.K[cellidx];
                //K = k*rho*-params.g[2]/params.nu;
                K = 0.5; 
                d.K[cellidx] = K;

                // Hydraulic transmissivity [m2/s]
                Float3 T = {K*d.dx, K*d.dy, K*d.dz};
                d.T[cellidx] = T;
            }
        }
    }
}

// Set the gradient to 0.0 in all dimensions at the boundaries
// Unused
void DEM::setDarcyBCNeumannZero()
{
    Float3 z3 = MAKE_FLOAT3(0.0, 0.0, 0.0);
    int ix, iy, iz;
    int nx = d.nx-1;
    int ny = d.ny-1;
    int nz = d.nz-1;

    // I don't care that the values at four edges are written twice

    // x-y plane at z=0 and z=d.dz-1
    for (ix=0; ix<d.nx; ++ix) {
        for (iy=0; iy<d.ny; ++iy) {
            d.dH[idx(ix,iy, 0)] = z3;
            d.dH[idx(ix,iy,nz)] = z3;
        }
    }

    // x-z plane at y=0 and y=d.dy-1
    for (ix=0; ix<d.nx; ++ix) {
        for (iz=0; iz<d.nz; ++iz) {
            d.dH[idx(ix, 0,iz)] = z3;
            d.dH[idx(ix,ny,iz)] = z3;
        }
    }

    // y-z plane at x=0 and x=d.nx-1
    for (iy=0; iy<d.ny; ++iy) {
        for (iz=0; iz<d.nz; ++iz) {
            d.dH[idx( 0,iy,iz)] = z3;
            d.dH[idx(nx,iy,iz)] = z3;
        }
    }
}


// Find the spatial gradient in pressures per cell
void DEM::findDarcyGradients()
{
    // Cell sizes squared
    //const Float dx2 = d.dx*d.dx;
    //const Float dx2 = d.dx*d.dx;
    //const Float dy2 = d.dy*d.dy;
    const Float dx2 = 2.0*d.dx;
    const Float dy2 = 2.0*d.dy;
    const Float dz2 = 2.0*d.dz;

    //Float H;
    int ix, iy, iz, cellidx;

    // Without ghost-nodes
    /*for (ix=1; ix<d.nx-1; ++ix) {
        for (iy=1; iy<d.ny-1; ++iy) {
            for (iz=1; iz<d.nz-1; ++iz) {*/

    // With ghost-nodes
    for (ix=0; ix<d.nx; ++ix) {
        for (iy=0; iy<d.ny; ++iy) {
            //for (iz=1; iz<d.nz-1; ++iz) {
            for (iz=0; iz<d.nz; ++iz) {

                cellidx = idx(ix,iy,iz);

                //H = d.H[cellidx];   // cell hydraulic pressure

                // First order central differences
                // x-boundary
                d.dH[cellidx].x
                 = (d.H[idx(ix+1,iy,iz)] - d.H[idx(ix-1,iy,iz)])/dx2;

                // y-boundary
                d.dH[cellidx].y
                 = (d.H[idx(ix,iy+1,iz)] - d.H[idx(ix,iy-1,iz)])/dy2;

                // z-boundary
                d.dH[cellidx].z
                 = (d.H[idx(ix,iy,iz+1)] - d.H[idx(ix,iy,iz-1)])/dz2;

                /*
                // Periodic x boundaries
                if (ix == 0) {
                    d.dH[cellidx].x = (d.H[idx(ix+1,iy,iz)]
                                - 2.0*H + d.H[idx(d.nx-1,iy,iz)])/dx2;
                } else if (ix == d.nx-1) {
                    d.dH[cellidx].x = (d.H[idx(0,iy,iz)]
                                - 2.0*H + d.H[idx(ix-1,iy,iz)])/dx2;
                } else {
                    d.dH[cellidx].x = (d.H[idx(ix+1,iy,iz)]
                                - 2.0*H + d.H[idx(ix-1,iy,iz)])/dx2;
                }
                
                // Periodic y boundaries
                if (iy == 0) {
                    d.dH[cellidx].y = (d.H[idx(ix,iy+1,iz)]
                                - 2.0*H + d.H[idx(ix,d.ny-1,iz)])/dy2;
                } else if (iy == d.ny-1) {
                    d.dH[cellidx].y = (d.H[idx(ix,0,iz)]
                                - 2.0*H + d.H[idx(ix,iy-1,iz)])/dy2;
                } else {
                    d.dH[cellidx].y = (d.H[idx(ix,iy+1,iz)]
                            - 2.0*H + d.H[idx(ix,iy-1,iz)])/dy2;
                }*/

            }
        }
    }
}

// Arithmetic mean of two numbers
inline Float amean(Float a, Float b) {
    return (a+b)*0.5;
}

// Harmonic mean of two numbers
inline Float hmean(Float a, Float b) {
    return (2.0*a*b)/(a+b);
}

// Perform an explicit step.
void DEM::explDarcyStep()
{

    // Find transmissivities from cell particle content
    findDarcyTransmissivities();

    // Check the time step length
    checkDarcyTimestep();

    // Cell dims squared
    const Float dxdx = d.dx*d.dx;
    const Float dydy = d.dy*d.dy;
    const Float dzdz = d.dz*d.dz;
    const Float dxdydz = d.dx*d.dy*d.dz;

    //setDarcyBCNeumannZero();

    // Update ghost node values from their parent cell values
    setDarcyGhostNodes();

    // Explicit 3D finite difference scheme
    // new = old + production*timestep + gradient*timestep
    int ix, iy, iz, cellidx;
    Float K, H, deltaH;
    Float Tx, Ty, Tz, S;
    //Float Tx_n, Tx_p, Ty_n, Ty_p, Tz_n, Tz_p;
    Float gradx_n, gradx_p, grady_n, grady_p, gradz_n, gradz_p;
    for (ix=0; ix<d.nx; ++ix) {
        for (iy=0; iy<d.ny; ++iy) {
            for (iz=0; iz<d.nz; ++iz) {

                // Cell linear index
                cellidx = idx(ix,iy,iz);

                // If x,y,z boundaries are fixed values: Enforce Dirichlet BC
                /*if (ix == 0 || iy == 0 || iz == 0 ||
                        ix == d.nx-1 || iy == d.ny-1 || iz == d.nz-1) {
                    d.H_new[cellidx] = d.H[cellidx];*/

                // If z boundaries are fixed val, x and y are periodic:
                /*if (iz == 0 || iz == d.nz-1) {
                    d.H_new[cellidx] = d.H[cellidx];

                } else {*/

                // Cell hydraulic transmissivities
                const Float3 T = d.T[cellidx];

                // Cell hydraulic head
                H = d.H[cellidx];

                // Harmonic mean of transmissivity
                // (in neg. and pos. direction along axis from cell)
                // with periodic x and y boundaries
                // without ghost nodes
                /*
                if (ix == 0)
                    gradx_n = hmean(Tx, d.T[idx(d.nx-1,iy,iz)].x)
                        * (d.H[idx(d.nx-1,iy,iz)] - H)/dx2;
                else
                    gradx_n = hmean(Tx, d.T[idx(ix-1,iy,iz)].x)
                        * (d.H[idx(ix-1,iy,iz)] - H)/dx2;

                if (ix == d.nx-1)
                    gradx_p = hmean(Tx, d.T[idx(0,iy,iz)].x)
                        * (d.H[idx(0,iy,iz)] - H)/dx2;
                else
                    gradx_p = hmean(Tx, d.T[idx(ix+1,iy,iz)].x)
                        * (d.H[idx(ix+1,iy,iz)] - H)/dx2;

                if (iy == 0)
                    grady_n = hmean(Ty, d.T[idx(ix,d.ny-1,iz)].y)
                        * (d.H[idx(ix,d.ny-1,iz)] - H)/dy2;
                else
                    grady_n = hmean(Ty, d.T[idx(ix,iy-1,iz)].y)
                        * (d.H[idx(ix,iy-1,iz)] - H)/dy2;

                if (iy == d.ny-1)
                    grady_p = hmean(Ty, d.T[idx(ix,0,iz)].y)
                        * (d.H[idx(ix,0,iz)] - H)/dy2;
                else
                    grady_p = hmean(Ty, d.T[idx(ix,iy+1,iz)].y)
                        * (d.H[idx(ix,iy+1,iz)] - H)/dy2;
                        */

                gradx_n = hmean(T.x, d.T[idx(ix-1,iy,iz)].x)
                    * (d.H[idx(ix-1,iy,iz)] - H)/dxdx;
                gradx_p = hmean(T.x, d.T[idx(ix+1,iy,iz)].x)
                    * (d.H[idx(ix+1,iy,iz)] - H)/dxdx;

                grady_n = hmean(T.y, d.T[idx(ix,iy-1,iz)].y)
                    * (d.H[idx(ix,iy-1,iz)] - H)/dydy;
                grady_p = hmean(T.y, d.T[idx(ix,iy+1,iz)].y)
                    * (d.H[idx(ix,iy+1,iz)] - H)/dydy;

                // Neumann (no-flow) boundary condition at +z and -z boundaries
                // enforced by a gradient value of 0.0
                if (iz == 0)
                    gradz_n = 0.0;
                else
                    gradz_n = hmean(T.z, d.T[idx(ix,iy,iz-1)].z)
                        * (d.H[idx(ix,iy,iz-1)] - H)/dzdz;
                if (iz == d.nz-1)
                    gradz_p = 0.0;
                else
                    gradz_p = hmean(T.z, d.T[idx(ix,iy,iz+1)].z)
                        * (d.H[idx(ix,iy,iz+1)] - H)/dzdz;

                /*std::cerr << ix << ',' << iy << ',' << iz << '\t'
                    << H << '\t' << Tx << ',' << Ty << ',' << Tz << '\t'
                    << gradx_n << ',' << gradx_p << '\t'
                    << grady_n << ',' << grady_p << '\t'
                    << gradz_n << ',' << gradz_p << std::endl;*/

                // Cell hydraulic storativity
                S = d.Ss[cellidx]*dxdydz;

                // Laplacian operator
                deltaH = time.dt/S *
                    (  gradx_n + gradx_p
                     + grady_n + grady_p
                     + gradz_n + gradz_p
                     + d.W[cellidx] );

                // Calculate new hydraulic pressure in cell
                d.H_new[cellidx] = H + deltaH;
                //}
            }
        }
    }

    // Swap d.H and d.H_new
    Float* tmp = d.H;
    d.H = d.H_new;
    d.H_new = tmp;

    // Find macroscopic cell fluid velocities
    findDarcyVelocities();
}

// Print array values to file stream (stdout, stderr, other file)
void DEM::printDarcyArray(FILE* stream, Float* arr)
{
    int x, y, z;
    for (z=0; z<d.nz; z++) {
        for (y=0; y<d.ny; y++) {
            for (x=0; x<d.nx; x++) {
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
    int x, y, z;
    for (z=0; z<d.nz; z++) {
        for (y=0; y<d.ny; y++) {
            for (x=0; x<d.nx; x++) {
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

    int ix, iy, iz, cellidx;
    for (ix=0; ix<d.nx; ++ix) {
        for (iy=0; iy<d.ny; ++iy) {
            for (iz=0; iz<d.nz; ++iz) {
                
                cellidx = idx(ix,iy,iz);
                dH = d.dH[cellidx];

                // Approximate cell porosity
                Float phi = d.phi[cellidx];

                // Calculate flux
                // The sign might need to be reversed, depending on the
                // grid orientation
                q.x = -d.K[cellidx]/nu * dH.x;
                q.y = -d.K[cellidx]/nu * dH.y;
                q.z = -d.K[cellidx]/nu * dH.z;
                
                // Calculate velocity
                v.x = q.x/phi;
                v.y = q.y/phi;
                v.z = q.z/phi;
                d.V[cellidx] = v;
            }
        }
    }
}

// Return the lower corner coordinates of a cell
inline Float3 DEM::cellMinBoundaryDarcy(
        const int x,
        const int y,
        const int z)
{
    const Float3 x_min = {x*d.dx, y*d.dy, z*d.dz};
    return x_min;
}

// Return the upper corner coordinates of a cell
inline Float3 DEM::cellMaxBoundaryDarcy(
        const int x,
        const int y,
        const int z)
{
    const Float3 x_max = {(x+1)*d.dx, (y+1)*d.dy, (z+1)*d.dz};
    return x_max;
}

// Return the volume of a cell
inline Float DEM::cellVolumeDarcy()
{
    const Float cell_volume = d.dx*d.dy*d.dz;
    return cell_volume;
}

// Find the porosity of a target cell
Float DEM::cellPorosity(
        const int x,
        const int y,
        const int z)
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

// Calculate the porosity for each cell
void DEM::findPorosities()
{
    int ix, iy, iz, cellidx;
    for (ix=0; ix<d.nx; ++ix) {
        for (iy=0; iy<d.ny; ++iy) {
            for (iz=0; iz<d.nz; ++iz) {
                d.phi[idx(ix,iy,iz)] = cellPorosity(ix,iy,iz);
            }
        }
    }
}

// Returns the mean particle radius
Float DEM::meanRadius()
{
    unsigned int i;
    Float r_sum;
    for (i=0; i<np; ++i)
        r_sum += k.x[i].w;
    return r_sum/((Float)np);
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
    int ix,iy,iz;
    Float3 val;
    for (ix=0; ix<d.nx; ++ix) {
        for (iy=0; iy<d.ny; ++iy) {
            for (iz=0; iz<d.nz; ++iz) {
                val = d.T[idx(ix,iy,iz)];
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
    int ix,iy,iz;
    Float val;
    for (ix=0; ix<d.nx; ++ix) {
        for (iy=0; iy<d.ny; ++iy) {
            for (iz=0; iz<d.nz; ++iz) {
                val = d.Ss[idx(ix,iy,iz)];
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
    Float S_min = getSsmin()*d.dx*d.dy*d.dz;

    // Use numerical criterion from Sonnenborg & Henriksen 2005
    Float value = T_max/S_min
        * (time.dt/(d.dx*d.dx) + time.dt/(d.dy*d.dy) + time.dt/(d.dz*d.dz));

    if (value > 0.5) {
        std::cerr << "Error! The explicit Darcy solution will be unstable.\n"
            << "This happens due to a combination of the following:\n"
            << " - The transmissivity T (i.e. hydraulic conductivity, K)"
            << " is too large (" << T_max << ")\n"
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

// Initialize darcy arrays, their values, and check the time step length
void DEM::initDarcy()
{
    if (params.nu <= 0.0) {
        std::cerr << "Error in initDarcy. The dymamic viscosity (params.nu), "
            << "should be larger than 0.0, but is " << params.nu << std::endl;
        exit(1);
    }

    // Cell size 
    d.dx = grid.L[0]/d.nx;
    d.dy = grid.L[1]/d.ny;
    d.dz = grid.L[2]/d.nz;

    if (verbose == 1) {
        std::cout << "  - Fluid grid dimensions: "
            << d.nx << "*"
            << d.ny << "*"
            << d.nz << std::endl;
        std::cout << "  - Fluid grid cell size: "
            << d.dx << "*"
            << d.dy << "*"
            << d.dz << std::endl;
    }

    //initDarcyMem(); // done in readbin
    initDarcyVals();
    findDarcyTransmissivities();

    checkDarcyTimestep();
}

// Write values in scalar field to file
void DEM::writeDarcyArray(Float* array, const char* filename)
{
    FILE* file;
    if ((file = fopen(filename,"w"))) {
        printDarcyArray(file, array);
        fclose(file);
    } else {
        fprintf(stderr, "Error, could not open %s.\n", filename);
    }
}

// Write values in vector field to file
void DEM::writeDarcyArray(Float3* array, const char* filename)
{
    FILE* file;
    if ((file = fopen(filename,"w"))) {
        printDarcyArray(file, array);
        fclose(file);
    } else {
        fprintf(stderr, "Error, could not open %s.\n", filename);
    }
}


// Print final heads and free memory
void DEM::endDarcy()
{
    writeDarcyArray(d.phi, "d_phi.txt");
    writeDarcyArray(d.K, "d_K.txt");

    //printDarcyArray(stdout, d.K, "d.K");
    //printDarcyArray(stdout, d.H, "d.H");
    //printDarcyArray(stdout, d.H_new, "d.H_new");
    //printDarcyArray(stdout, d.V, "d.V");
    freeDarcyMem();
}

// vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
