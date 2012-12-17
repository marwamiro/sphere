#include <iostream>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#include "typedefs.h"
#include "datatypes.h"
#include "constants.h"
#include "sphere.h"

// Constructor: Reads an input binary, and optionally checks
// and reports the values
DEM::DEM(const std::string inputbin, 
        const int verbosity,
        const int checkVals,
        const int dry,
        const int initCuda)
: verbose(verbosity)
{
    using std::cout;
    using std::cerr;

    // Extract sid from input binary filename 
    size_t dotpos = inputbin.rfind('.');
    size_t slashpos = inputbin.rfind('/');
    if (slashpos - dotpos < 1) {
        std::cerr << "Error! Unable to extract simulation id "
            << "from input file name.\n";
    }
    sid = inputbin.substr(slashpos+1, dotpos-slashpos-1);

    // Read target input binary
    readbin(inputbin.c_str());

    // Check numeric values of chosen parameters
    if (checkVals == 1)
        checkValues();

    // Report data values
    if (dry == 1)
        reportValues();

    // If this is a dry run, exit
    if (dry == 1)
        exit(1);

    if (initCuda == 1) {
        // Initialize CUDA
        initializeGPU();

        // Copy constant data to constant device memory
        transferToConstantDeviceMemory();

        // Allocate device memory for particle variables,
        // tied to previously declared pointers in structures
        allocateGlobalDeviceMemory();

        // Transfer data from host to gpu device memory
        transferToGlobalDeviceMemory();
    }

}

// Destructor: Liberates dynamically allocated host memory
DEM::~DEM(void)
{
    delete[] k.x;
    delete[] k.xysum;
    delete[] k.vel;
    delete[] k.force;
    delete[] k.angpos;
    delete[] k.angvel;
    delete[] k.torque;
    delete[] e.es_dot;
    delete[] e.es;
    delete[] e.ev_dot;
    delete[] e.ev;
    delete[] e.p;
    delete[] walls.nx;
    delete[] walls.mvfd;
}


// Check numeric values of selected parameters
void DEM::checkValues(void)
{
    using std::cerr;

    unsigned int i;

    // Check the number of dimensions
    if (nd != ND) {
        cerr << "Error: nd = " << nd << ", ND = " << ND << '\n';
        exit(1);
    }

    // Check the number of possible contacts
    if (NC < 1) {
        cerr << "Error: NC = " << NC << '\n';
        exit(1);
    } else if (NC < 8) {
        cerr << "Warning: NC has a low value (" << NC << "). "
            << "Consider increasing it in 'constants.h'\n";
    }

    // Check that we have a positive number of particles
    if (np < 1) {
        cerr << "Error: np = " << np << '\n';
        exit(1);
    }

    // Check that the current time
    if (time.current > time.total || time.current < 0.0) {
        cerr << "Error: time.current = " << time.current
            << " s, time.total = " << time.total << " s\n";
        exit(1);
    }

    // Check world size
    if (grid.L[0] <= 0.0 || grid.L[1] <= 0.0 || grid.L[2] <= 0.0) {
        cerr << "Error: grid.L[0] = " << grid.L[0] << " m, "
            << "grid.L[1] = " << grid.L[1] << " m, "
            << "grid.L[2] = " << grid.L[2] << " m.\n";
        exit(1);
    }

    // Check grid size
    if (grid.num[0] <= 0 || grid.num[1] <= 0 || grid.num[2] <= 0) {
        cerr << "Error: grid.num[0] = " << grid.num[0] << ", "
            << "grid.num[1] = " << grid.num[1] << ", "
            << "grid.num[2] = " << grid.num[2] << ".\n";
        exit(1);
    }

    // Check grid size again
    if (grid.periodic == 2 && grid.num[0] < 3) {
        cerr << "Error: When 1st dimension boundaries are periodic, "
            << "there must be at least 3 cells in that dimension.";
        exit(1);
    }

    if (grid.periodic == 1 && (grid.num[0] < 3 || grid.num[1] < 3)) {
        cerr << "Error: When 1st and 2nd dimension boundaries are periodic, "
            << "there must be at least 3 cells in each of those dimensions.";
        exit(1);
    }


    // Per-particle checks
    Float4 x;
    for (i = 0; i < np; ++i) {

        // Read value into register
        x = k.x[i];

        // Check that radii are positive values
        if (x.w <= 0.0) {
            cerr << "Error: Particle " << i << " has a radius of "
                << k.x[i].w << " m." << std::endl;
            exit(1);
        }

        // Check that all particles are inside of the grid
        if (x.x < grid.origo[0] ||
                x.y < grid.origo[1] ||
                x.z < grid.origo[2] ||
                x.x > grid.L[0] ||
                x.y > grid.L[1] ||
                x.z > grid.L[2]) {
            cerr << "Error: Particle " << i << " is outside of "
                << "the computational grid\n"
                << "k.x[i] = ["
                << x.x << ", "
                << x.y << ", "
                << x.z << "]\n"
                << "grid.origo = ["
                << grid.origo[0] << ", "
                << grid.origo[1] << ", "
                << grid.origo[2] << "], "
                << "grid.L = ["
                << grid.L[0] << ", "
                << grid.L[1] << ", "
                << grid.L[2] << "]."
                << std::endl;
            exit(1);

        }
    }

    // If present, check that the upper wall is above all particles
    if (walls.nw > 0) {
        Float z_max = 0.0;
        for (i = 0; i < np; ++i) {
            if (k.x[i].z > z_max)
                z_max = k.x[i].z;
        }

        if (walls.nx[0].w < z_max) {
            cerr << "Error: One or more particles have centres above "
                << "the upper, dynamic wall";
            exit(1);
        }
    }



    // Check constant, global parameters
    if (params.k_n <= 0.0) {
        cerr << "Error: k_n = " << params.k_n << " N/m\n";
        exit(1);
    }

    if (params.rho <= 0.0) {
        cerr << "Error: rho = " << params.rho << " kg/m3\n";
        exit(1);
    }
}

// Report key parameter values to stdout
void DEM::reportValues()
{
    using std::cout;
    using std::cerr;
    using std::endl;

    cout << "  - Number of dimensions: nd = " << nd << "\n"
        << "  - Number of particles:  np = " << np << "\n";

    // Check precision choice
    cout << "  - Compiled for ";
    if (sizeof(Float) == sizeof(float)) {
        if (verbose == 1)
            cout << "single";
    } else if (sizeof(Float) == sizeof(double)) {
        if (verbose == 1)
            cout << "double";
    } else {
        cerr << "Error! Chosen precision not available. Check datatypes.h\n";
        exit(1);
    }
    cout << " precision\n";

    cout << "  - Timestep length:      time.dt         = " 
        << time.dt << " s\n"
        << "  - Start at time:        time.current    = " 
        << time.current << " s\n"
        << "  - Total sim. time:      time.total      = " 
        << time.total << " s\n"
        << "  - File output interval: time.file_dt    = " 
        << time.file_dt << " s\n"
        << "  - Start at step count:  time.step_count = " 
        << time.step_count << endl;

    if (params.contactmodel == 1)
        cout << "  - Contact model: Linear-elastic-viscous (n), visco-frictional (t)\n";
    else if (params.contactmodel == 2)
        cout << "  - Contact model: Linear-elastic-visco-frictional\n";
    else if (params.contactmodel == 3)
        cout << "  - Contact model: Nonlinear-elastic-visco-frictional\n";
    else {
        cerr << "Error: Contact model value (" << params.contactmodel << ") not understood.\n";
        exit(1);
    }

    cout << "  - Number of dynamic walls: " << walls.nw << '\n';

    if (grid.periodic == 1)
        cout << "  - 1st and 2nd dim. boundaries: Periodic\n";
    else if (grid.periodic == 2)
        cout << "  - 1st dim. boundaries: Visco-frictional walls\n";
    else
        cout << "  - 1st and 2nd dim. boundaries: Visco-frictional walls\n";

    if (walls.nw > 0) {
        cout << "  - Top BC: ";
        if (walls.wmode[0] == 0)
            cout << "Fixed\n";
        else if (walls.wmode[0] == 1)
            cout << "Deviatoric stress, "
                << walls.mvfd[0].w << " Pa\n";
        else if (walls.wmode[0] == 2)
            cout << "Velocity, "
                << walls.mvfd[0].y << " m/s\n";
        else {
            cerr << "Top BC not recognized!\n";
            exit(1);
        }
    }

    cout << "  - Grid: ";
    if (nd == 1)
        cout << grid.num[0];
    else if (nd == 2)
        cout << grid.num[0] << " * " << grid.num[1];
    else 
        cout << grid.num[0] << " * " 
            << grid.num[1] << " * "
            << grid.num[2];
    cout << " cells\n";
}

// Returns the volume of a spherical cap
Float sphericalCap(const Float h, const Float r)
{
    return M_PI * h * h / 3.0 * (3.0 * r - h);
}

// Calculate the porosity with depth, and write to file in output directory
void DEM::porosity(const int z_slices)
{
    // The porosity value is higher at the boundaries due 
    // to the no-flux BCs.
    
    Float top;
    if (walls.nw > 0)
        top = walls.nx->w;
    else
        top = grid.L[2];

    // Calculate depth slice thickness
    Float h_slice = (top - grid.origo[2]) / (Float)z_slices;

    // Calculate slice volume
    Float V_slice = h_slice * grid.L[0] * grid.L[1];

    // Array of depth values
    Float z_pos[z_slices];

    // Array of porosity values
    Float porosity[z_slices];

    // Loop over vertical slices
#pragma omp parallel for if(np > 100)
    for (int iz = 0; iz<z_slices; ++iz) {

        // The void volume equals the slice volume, with the
        // grain volumes subtracted
        Float V_void = V_slice;

        // Bottom and top position of depth slice
        Float z_slice_low = iz * h_slice;
        Float z_slice_high = z_slice_low + h_slice;

        // Loop over particles to see whether they are inside of the slice
        for (unsigned int i = 0; i<np; ++i) {

            // Read particle values
            Float z_sphere_centre = k.x[i].z;
            Float radius = k.x[i].w;
            
            // Save vertical positions of particle boundaries
            Float z_sphere_low = z_sphere_centre - radius;
            Float z_sphere_high = z_sphere_centre + radius;

            // Sphere volume
            Float V_sphere = 4.0/3.0 * M_PI * radius * radius * radius;

            // If the sphere is inside the slice and not intersecting the
            // boundaries, subtract the entire sphere volume
            if (z_slice_low < z_sphere_low && z_sphere_high < z_slice_high) {
                V_void -= V_sphere;

            } else {

                // If the sphere intersects with the lower boundary,
                // and the centre is below the boundary
                if (z_slice_low > z_sphere_centre && z_slice_low < z_sphere_high) {

                    // Subtract the volume of a spherical cap
                    V_void -= sphericalCap(z_sphere_high - z_slice_low, radius);
                }

                // If the sphere intersects with the lower boundary,
                // and the centre is above the boundary
                else if (z_slice_low < z_sphere_centre && z_slice_low > z_sphere_low) {

                    // Subtract the volume of the sphere, 
                    // then add the volume of the spherical cap below
                    V_void -= V_sphere + sphericalCap(z_slice_low - z_sphere_low, radius);
                }

                // If the sphere intersects with the upper boundary,
                // and the centre is below the boundary
                if (z_slice_high > z_sphere_centre && z_slice_high < z_sphere_high) {

                    // Subtract the volume of the sphere, 
                    // then add the volume of the spherical cap above
                    V_void -= V_sphere + sphericalCap(z_sphere_high - z_slice_high, radius);
                }
                
                // If the sphere intersects with the upper boundary,
                // and the centre is above the boundary
                else if (z_slice_high < z_sphere_centre && z_slice_high > z_sphere_low) {

                    // Subtract the volume of the spherical cap below
                    V_void -= sphericalCap(z_slice_high - z_sphere_low, radius);
                }
                

            }
        }

        // Save the mid z-point
        z_pos[iz] = z_slice_low + 0.5*h_slice;

        // Save the porosity
        porosity[iz] = V_void / V_slice;

        // Report values to stdout
        /*
        std::cout << iz << ": V_void = " << V_void 
            << "\tV_slice = " << V_slice 
            << "\tporosity = " << V_void/V_slice
            << '\n' << std::endl;
            */
    }

    // Save results to text file
    writePorosities(("output/" + sid + "-porosity.txt").c_str(), z_slices, z_pos, porosity);
    //for (int i=z_slices-1; i>=0; --i)
        //std::cout << porosity[i] << std::endl;

}

// vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
