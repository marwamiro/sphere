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
void DEM::initNSmem()
{
    // Number of cells
    ns.nx = floor(grid.num[0]);
    ns.ny = floor(grid.num[1]);
    ns.nz = floor(grid.num[2]);

    unsigned int ncells = NScells();

    ns.p     = new Float[ncells];  // hydraulic pressure
    //ns.p_new = new Float[ncells];  // hydraulic pressure
    ns.dp    = new Float3[ncells]; // hydraulic pressure gradient
    ns.v     = new Float3[ncells]; // hydraulic velocity
    ns.v_p   = new Float3[ncells]; // predicted hydraulic velocity
    ns.phi   = new Float[ncells];  // porosity
    ns.dphi  = new Float[ncells];  // porosity change
    ns.norm  = new Float[ncells];  // normalized residual of epsilon
    ns.epsilon = new Float[ncells];  // normalized residual of epsilon
    ns.epsilon_new = new Float[ncells];  // normalized residual of epsilon
}

unsigned int DEM::NScells()
{
    //return ns.nx*ns.ny*ns.nz; // without ghost nodes
    return (ns.nx+2)*(ns.ny+2)*(ns.nz+2); // with ghost nodes
}

// Free memory
void DEM::freeNSmem()
{
    delete[] ns.p;
    //delete[] ns.p_new;
    delete[] ns.dp;
    delete[] ns.v;
    delete[] ns.v_p;
    delete[] ns.phi;
    delete[] ns.dphi;
    delete[] ns.norm;
    delete[] ns.epsilon;
    delete[] ns.epsilon_new;
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
    return (x+1) + (ns.nx+2)*(y+1) + (ns.nx+2)*(ns.ny+2)*(z+1);
}

// Print array values to file stream (stdout, stderr, other file)
void DEM::printNSarray(FILE* stream, Float* arr)
{
    int x, y, z;

    // show ghost nodes
    for (z=-1; z<=ns.nz; z++) {
        for (y=-1; y<=ns.ny; y++) {
            for (x=-1; x<=ns.nx; x++) {

    // hide ghost nodes
    /*for (z=0; z<ns.nz; z++) {
        for (y=0; y<ns.ny; y++) {
            for (x=0; x<ns.nx; x++) {*/

                fprintf(stream, "%f\t", arr[idx(x,y,z)]);
            }
            fprintf(stream, "\n");
        }
        fprintf(stream, "\n");
    }
}

// Overload printNSarray to add optional description
void DEM::printNSarray(FILE* stream, Float* arr, std::string desc)
{
    std::cout << "\n" << desc << ":\n";
    printNSarray(stream, arr);
}

// Print array values to file stream (stdout, stderr, other file)
void DEM::printNSarray(FILE* stream, Float3* arr)
{
    int x, y, z;
    for (z=0; z<ns.nz; z++) {
        for (y=0; y<ns.ny; y++) {
            for (x=0; x<ns.nx; x++) {
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

// Overload printNSarray to add optional description
void DEM::printNSarray(FILE* stream, Float3* arr, std::string desc)
{
    std::cout << "\n" << desc << ":\n";
    printNSarray(stream, arr);
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

// Returns the average value of the normalized residuals
double DEM::avgNormResNS()
{
    double norm_res_sum, norm_res;

    // do not consider the values of the ghost nodes
    for (int z=0; z<grid.num[2]; ++z) {
        for (int y=0; y<grid.num[1]; ++y) {
            for (int x=0; x<grid.num[0]; ++x) {
                norm_res = static_cast<double>(ns.norm[idx(x,y,z)]);
                if (norm_res != norm_res) {
                    std::cerr << "\nError: normalized residual is NaN ("
                        << norm_res << ") in cell "
                        << x << "," << y << "," << z << std::endl;
                    exit(1);
                }
                norm_res_sum += norm_res;
            }
        }
    }
    return norm_res_sum/(grid.num[0]*grid.num[1]*grid.num[2]);
}


// Returns the average value of the normalized residuals
double DEM::maxNormResNS()
{
    double max_norm_res = -1.0e9; // initialized to a small number
    double norm_res;

    // do not consider the values of the ghost nodes
    for (int z=0; z<grid.num[2]; ++z) {
        for (int y=0; y<grid.num[1]; ++y) {
            for (int x=0; x<grid.num[0]; ++x) {
                norm_res = static_cast<double>(ns.norm[idx(x,y,z)]);
                if (norm_res != norm_res) {
                    std::cerr << "\nError: normalized residual is NaN ("
                        << norm_res << ") in cell "
                        << x << "," << y << "," << z << std::endl;
                    exit(1);
                }
                if (norm_res > max_norm_res)
                    max_norm_res = norm_res;
            }
        }
    }
    return max_norm_res;
}

// Initialize fluid parameters
void DEM::initNS()
{
    if (params.nu <= 0.0) {
        std::cerr << "Error in initNS. The dymamic viscosity (params.nu), "
            << "should be larger than 0.0, but is " << params.nu << std::endl;
        exit(1);
    }

    // Cell size 
    ns.dx = grid.L[0]/ns.nx;
    ns.dy = grid.L[1]/ns.ny;
    ns.dz = grid.L[2]/ns.nz;

    if (verbose == 1) {
        std::cout << "  - Fluid grid dimensions: "
            << ns.nx << "*"
            << ns.ny << "*"
            << ns.nz << std::endl;
        std::cout << "  - Fluid grid cell size: "
            << ns.dx << "*"
            << ns.dy << "*"
            << ns.dz << std::endl;
    }
}

// Write values in scalar field to file
void DEM::writeNSarray(Float* array, const char* filename)
{
    FILE* file;
    if ((file = fopen(filename,"w"))) {
        printNSarray(file, array);
        fclose(file);
    } else {
        fprintf(stderr, "Error, could not open %s.\n", filename);
    }
}

// Write values in vector field to file
void DEM::writeNSarray(Float3* array, const char* filename)
{
    FILE* file;
    if ((file = fopen(filename,"w"))) {
        printNSarray(file, array);
        fclose(file);
    } else {
        fprintf(stderr, "Error, could not open %s.\n", filename);
    }
}


// Print final heads and free memory
void DEM::endNS()
{
    // Write arrays to stdout/text files for debugging
    //writeNSarray(ns.phi, "ns_phi.txt");

    //printNSarray(stdout, ns.K, "ns.K");
    //printNSarray(stdout, ns.H, "ns.H");
    //printNSarray(stdout, ns.H_new, "ns.H_new");
    //printNSarray(stdout, ns.V, "ns.V");

    freeNSmem();
}

// vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
