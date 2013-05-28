#include <iostream>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>

#include "typedefs.h"
#include "datatypes.h"
#include "constants.h"
#include "sphere.h"

// Find hydraulic conductivities for each cell

// Solve Darcy flow through particles
void DEM::startDarcy(
        const Float cellsizemultiplier)
{
    // Number of cells
    int nx = grid.L[0]/grid.num[0];
    int ny = grid.L[1]/grid.num[1];
    int nz = grid.L[2]/grid.num[2];

    // Cell size 
    Float dx = grid.L[0]/nx;
    Float dy = grid.L[1]/nx;
    Float dz = grid.L[2]/nx;

    if (verbose == 1) {
        std::cout << "Fluid grid dimensions: "
            << nx << " * "
            << ny << " * "
            << nz << std::endl;
    }


}

