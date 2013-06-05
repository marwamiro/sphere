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

// Constructor: Reads an input binary, and optionally checks
// and reports the values
DEM::DEM(const std::string inputbin, 
        const int verbosity,
        const int checkVals,
        const int dry,
        const int initCuda,
        const int transferConstMem,
        const int darcyflow)
: verbose(verbosity), darcy(darcyflow)
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
        exit(0);

    if (initCuda == 1) {

        // Initialize CUDA
        initializeGPU();

        if (transferConstMem == 1) {
            // Copy constant data to constant device memory
            transferToConstantDeviceMemory();
        }

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
    cout << "  - Domain size: ";
    if (nd == 1)
        cout << grid.L[0];
    else if (nd == 2)
        cout << grid.L[0] << " * " << grid.L[1];
    else 
        cout << grid.L[0] << " * " 
            << grid.L[1] << " * "
            << grid.L[2];
    cout << " m\n";

    cout << "  - No. of particle bonds: " << params.nb0 << endl;
}

// Returns the volume of a spherical cap
Float sphericalCap(const Float h, const Float r)
{
    return M_PI * h * h / 3.0 * (3.0 * r - h);
}

// Returns the max. radius of any particle
Float DEM::r_max()
{
    Float r_max = 0.0;
    Float r;
    for (unsigned int i=0; i<np; ++i) {
        r = k.x[i].w;
        if (r > r_max)
            r_max = r;
    }
    return r;
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

    // Check that the vertical slice height does not exceed the
    // max particle diameter, since this function doesn't work
    // if the sphere intersects more than 1 boundary
    if (h_slice <= r_max()*2.0) {
        std::cerr << "Error! The number of z-slices is too high."
            << std::endl;
        exit(1);
    }

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
            if (z_slice_low <= z_sphere_low && z_sphere_high <= z_slice_high) {
                V_void -= V_sphere;

            } else {

                // If the sphere intersects with the lower boundary,
                // and the centre is below the boundary
                if (z_slice_low >= z_sphere_centre && z_slice_low <= z_sphere_high) {

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
                // and the centre is above the boundary
                if (z_slice_high <= z_sphere_centre && z_slice_high >= z_sphere_low) {

                    // Subtract the volume of the spherical cap below
                    V_void -= sphericalCap(z_slice_high - z_sphere_low, radius);
                }
                // If the sphere intersects with the upper boundary,
                // and the centre is below the boundary
                else if (z_slice_high > z_sphere_centre && z_slice_high < z_sphere_high) {

                    // Subtract the volume of the sphere, 
                    // then add the volume of the spherical cap above
                    V_void -= V_sphere + sphericalCap(z_sphere_high - z_slice_high, radius);
                }
                
                

            }
            
        }

        // Save the mid z-point
        z_pos[iz] = z_slice_low + 0.5*h_slice;

        // Save the porosity
        porosity[iz] = V_void / V_slice;

    }

    // Save results to text file
    //writePorosities(("output/" + sid + "-porosity.txt").c_str(), z_slices, z_pos, porosity);

    // Report values to stdout
    //std::cout << "z-pos" << '\t' << "porosity" << '\n';
    for (int i = 0; i<z_slices; ++i) {
        std::cout << z_pos[i] << '\t' << porosity[i] << '\n'; 
    }

}

// Find the min. spatial positions of the particles, and return these as a vector
Float3 DEM::minPos()
{
    unsigned int i;
    Float3 shared_min = MAKE_FLOAT3(0.0f, 0.0f, 0.0f);

#pragma omp parallel if(np > 100)
    {
        // Max. val per thread
        Float3 min = MAKE_FLOAT3(0.0f, 0.0f, 0.0f);

#pragma omp for nowait
        // Find min val. per thread
        for (i = 0; i<np; ++i) {
            min.x = std::min(min.x, k.x[i].x);
            min.y = std::min(min.y, k.x[i].y);
            min.z = std::min(min.z, k.x[i].z);
        }

        // Find total min, by comparing one thread with the
        // shared result, one at a time
#pragma omp critical
        {
            shared_min.x = std::min(shared_min.x, min.x);
            shared_min.y = std::min(shared_min.y, min.y);
            shared_min.z = std::min(shared_min.z, min.z);
        }
    }

    // Return final result
    return shared_min;
}

// Find the max. spatial positions of the particles, and return these as a vector
Float3 DEM::maxPos()
{
    unsigned int i;
    Float3 shared_max = MAKE_FLOAT3(0.0f, 0.0f, 0.0f);

#pragma omp parallel if(np > 100)
    {
        // Max. val per thread
        Float3 max = MAKE_FLOAT3(0.0f, 0.0f, 0.0f);

#pragma omp for nowait
        // Find max val. per thread
        for (i = 0; i<np; ++i) {
            max.x = std::max(max.x, k.x[i].x);
            max.y = std::max(max.y, k.x[i].y);
            max.z = std::max(max.z, k.x[i].z);
        }

        // Find total max, by comparing one thread with the
        // shared result, one at a time
#pragma omp critical
        {
            shared_max.x = std::max(shared_max.x, max.x);
            shared_max.y = std::max(shared_max.y, max.y);
            shared_max.z = std::max(shared_max.z, max.z);
        }
    }

    // Return final result
    return shared_max;
}


// Finds all overlaps between particles, 
// returns the indexes as a 2-row vector and saves
// the overlap size
void DEM::findOverlaps(
        std::vector< std::vector<unsigned int> > &ij,
        std::vector< Float > &delta_n_ij)
{
    unsigned int i, j;
    Float4 x_i, x_j; // radius is in .w component of struct
    Float3 x_ij;
    Float x_ij_length, delta_n;

    // Loop over particles, find intersections
    for (i=0; i<np; ++i) {

        for (j=0; j<np; ++j) {

            // Only check once par particle pair
            if (i < j) {

                x_i = k.x[i];
                x_j = k.x[j];

                x_ij = MAKE_FLOAT3(
                        x_j.x - x_i.x,
                        x_j.y - x_i.y,
                        x_j.z - x_i.z);

                x_ij_length = sqrt(
                        x_ij.x * x_ij.x +
                        x_ij.y * x_ij.y +
                        x_ij.z * x_ij.z);

                // Distance between spheres
                delta_n = x_ij_length - (x_i.w + x_j.w);

                // Is there overlap?
                if (delta_n < 0.0) {

                    // Store particle indexes and delta_n
                    std::vector<unsigned int> ij_pair(2);
                    ij_pair[0] = i;
                    ij_pair[1] = j;
                    ij.push_back(ij_pair);
                    delta_n_ij.push_back(delta_n);

                }
            }
        }
    }
}

// Calculate force chains and generate visualization script
void DEM::forcechains(const std::string format, const int threedim, 
        const double lower_cutoff,
        const double upper_cutoff)
{
    using std::cout;
    using std::endl;

    // Loop over all particles, find intersections
    std::vector< std::vector<unsigned int> > ij;
    std::vector< Float > delta_n_ij;
    findOverlaps(ij, delta_n_ij);

    // Find minimum position
    Float3 x_min = minPos();
    Float3 x_max = maxPos();

    // Find largest overlap, used for scaling the line thicknesses
    Float delta_n_min = *std::min_element(delta_n_ij.begin(), delta_n_ij.end());
    Float f_n_max = -params.k_n * delta_n_min;

    // Define limits of visualization [0;1]
    //Float lim_low = 0.1;
    //Float lim_low = 0.15;
    //Float lim_high = 0.25;

    if (format == "txt") {
        // Write text header
        cout << "x_1, [m]\t";
        if (threedim == 1)
            cout << "y_1, [m]\t";
        cout << "z_1, [m]\t";
        cout << "x_2, [m]\t";
        if (threedim == 1)
            cout << "y_2, [m]\t";
        cout << "z_2, [m]\t";
        cout << "||f_n||, [N]" << endl;


    } else {

        // Format sid so LaTeX won't encounter problems with the extension
        std::string s = sid;
        std::replace(s.begin(), s.end(), '.', '-');

        // Write Gnuplot header
        cout << "#!/usr/bin/env gnuplot\n" 
            << "# This Gnuplot script is automatically generated using\n"
            << "# the forcechain utility in sphere. For more information,\n"
            << "# see https://github.com/anders-dc/sphere\n"
            << "set size ratio -1\n";
        if (format == "png") {
            cout << "set term pngcairo size 30 cm,20 cm\n";
            cout << "set out '" << s << "-fc.png'\n";
        } else if (format == "epslatex") {
            cout << "set term epslatex size 8.6 cm, 5.6 cm\n";
            //cout << "set out 'plots/" << s << "-fc.tex'\n";
            cout << "set out '" << s << "-fc.tex'\n";
        } else if (format == "epslatex-color") {
            //cout << "set term epslatex color size 12 cm, 8.6 cm\n";
            cout << "set term epslatex color size 8.6 cm, 5.6 cm\n";
            cout << "set out 'plots/" << s << "-fc.tex'\n";
            //cout << "set out '" << s << "-fc.tex'\n";
        }
        cout << "set xlabel '\\sffamily $x_1$, [m]'\n";
        if (threedim == 1) {
            cout << "set ylabel '\\sffamily $x_2$, [m]'\n"
            << "set zlabel '\\sffamily $x_3$, [m]' offset 2\n";
            //<< "set zlabel '\\sffamily $x_3$, [m]' offset 0\n";
        } else
            //cout << "set ylabel '\\sffamily $x_3$, [m]' offset 0\n";
            cout << "set ylabel '\\sffamily $x_3$, [m]' offset 2\n";

        cout << "set cblabel '\\sffamily $||\\boldsymbol{f}_n||$, [N]'\n"
            << "set xyplane at " << x_min.z << '\n'
            << "set pm3d\n"
            << "set view 90.0,0.0\n"
            //<< "set palette defined (0 'gray', 0.5 'blue', 1 'red')\n"
            //<< "set palette defined (0 'white', 0.5 'gray', 1 'red')\n"
            << "set palette defined ( 1 '#000fff', 2 '#0090ff', 3 '#0fffee', 4 '#90ff70', 5 '#ffee00', 6 '#ff7000', 7 '#ee0000', 8 '#7f0000')\n"
            //<< "set cbrange [" << f_n_max*lim_low << ':' << f_n_max*lim_high << "]\n"
            << "set cbrange [" << lower_cutoff << ':' << upper_cutoff << "]\n"
            << endl;
    }

    // Loop over found contacts, report to stdout
    unsigned int n, i, j;
    Float delta_n, f_n, ratio;
    std::string color;
    for (n=0; n<ij.size(); ++n) {

        // Get contact particle indexes
        i = ij[n][0];
        j = ij[n][1];

        // Overlap size
        delta_n = delta_n_ij[n];

        // Normal force on contact
        f_n = -params.k_n * delta_n;

        if (f_n < lower_cutoff)
            continue;   // skip the rest of this iteration

        // Line weight
        ratio = f_n/f_n_max;

        if (format == "txt") {

            // Text output
            cout << k.x[i].x << '\t';
            if (threedim == 1)
                cout << k.x[i].y << '\t';
            cout << k.x[i].z << '\t';
            cout << k.x[j].x << '\t';
            if (threedim == 1)
                cout << k.x[j].y << '\t';
            cout << k.x[j].z << '\t';
            cout << f_n << endl;
        } else {

            // Gnuplot output
            // Save contact pairs if they are above the lower limit
            // and not fixed at their horizontal velocity
            //if (ratio > lim_low && (k.vel[i].w + k.vel[j].w) == 0.0) {
            if (f_n > lower_cutoff && (k.vel[i].w + k.vel[j].w) == 0.0) {

                // Plot contact as arrow without tip
                cout << "set arrow " << n+1 << " from "
                    << k.x[i].x << ',';
                if (threedim == 1)
                    cout << k.x[i].y << ',';
                cout << k.x[i].z;
                cout << " to " << k.x[j].x << ',';
                if (threedim == 1)
                    cout << k.x[j].y, ',';
                cout << k.x[j].z;
                cout << " nohead "
                    << "lw " << ratio * 8.0
                    << " lc palette cb " << f_n 
                    << endl;
            }
        }

    }

    if (format != "txt") {
        // Write Gnuplot footer
        if (threedim == 1)
            cout << "splot ";
        else
            cout << "plot ";

        cout << '[' << x_min.x << ':' << x_max.x << "] ";
        if (threedim == 1)
            cout << '[' << x_min.y << ':' << x_max.y << "] ";
        cout << '[' << x_min.z << ':' << x_max.z << "] " << "NaN notitle" << endl;
    }
}

// vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
