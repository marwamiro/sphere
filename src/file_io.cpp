#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstdlib>

#include "typedefs.h"
#include "datatypes.h"
#include "constants.h"
#include "sphere.h"

// Get the address of the first byte of an object's representation
// See Stroustrup (2008) p. 388
    template<class T>
char* as_bytes(T& i)	// treat a T as a sequence of bytes
{
    // get the address of the first byte of memory used
    // to store the object
    void* addr = &i;

    // treat the object as bytes
    return static_cast<char*>(addr);
}

// Read DEM data from binary file
// Note: Static-size arrays can be bulk-read with e.g.
//   ifs.read(as_bytes(grid.L), sizeof(grid.L))
// while dynamic, and vector arrays (e.g. Float4) must
// be read one value at a time.
void DEM::readbin(const char *target)
{
    using std::cout;  // stdout
    using std::cerr;  // stderr
    using std::endl;  // endline. Implicitly flushes buffer
    unsigned int i;

    // Open input file
    // if target is string: std::ifstream ifs(target.c_str(), std::ios_base::binary);
    std::ifstream ifs(target, std::ios_base::binary);
    if (!ifs) {
        cerr << "Could not read input binary file '"
            << target << "'" << endl;
        exit(1);
    }

    ifs.read(as_bytes(nd), sizeof(nd));
    ifs.read(as_bytes(np), sizeof(np));

    if (nd != ND) {
        cerr << "Dimensionality mismatch between dataset and this SPHERE program.\n"
            << "The dataset is " << nd 
            << "D, this SPHERE binary is " << ND << "D.\n"
            << "This execution is terminating." << endl;
        exit(-1); // Return unsuccessful exit status
    }

    // Check precision choice
    if (sizeof(Float) != sizeof(double) && sizeof(Float) != sizeof(float)) {
        cerr << "Error! Chosen precision not available. Check datatypes.h\n";
        exit(1);
    }

    // Read time parameters
    ifs.read(as_bytes(time.dt), sizeof(time.dt));
    ifs.read(as_bytes(time.current), sizeof(time.current));
    ifs.read(as_bytes(time.total), sizeof(time.total));
    ifs.read(as_bytes(time.file_dt), sizeof(time.file_dt));
    ifs.read(as_bytes(time.step_count), sizeof(time.step_count));

    // For spatial vectors an array of Float4 vectors is chosen for best fit with 
    // GPU memory handling. Vector variable structure: ( x, y, z, <empty>).
    // Indexing starts from 0.

    // Allocate host arrays
    if (verbose == 1)
        cout << "  Allocating host memory:                         ";
    // Allocate more host arrays
    k.x	     = new Float4[np];
    k.xysum  = new Float2[np];
    k.vel    = new Float4[np];
    k.force  = new Float4[np];
    k.angpos = new Float4[np];
    k.angvel = new Float4[np];
    k.torque = new Float4[np];

    e.es_dot = new Float[np];
    e.es     = new Float[np];
    e.ev_dot = new Float[np];
    e.ev     = new Float[np];
    e.p	     = new Float[np];

    if (verbose == 1)
        cout << "Done\n";

    if (verbose == 1)
        cout << "  Reading remaining data from input binary:       ";

    // Read grid parameters
    ifs.read(as_bytes(grid.origo), sizeof(grid.origo));
    ifs.read(as_bytes(grid.L), sizeof(grid.L));
    ifs.read(as_bytes(grid.num), sizeof(grid.num));
    ifs.read(as_bytes(grid.periodic), sizeof(grid.periodic));

    // Read kinematic values
    for (i = 0; i<np; ++i) {
        ifs.read(as_bytes(k.x[i].x), sizeof(Float));
        ifs.read(as_bytes(k.x[i].y), sizeof(Float));
        ifs.read(as_bytes(k.x[i].z), sizeof(Float));
        ifs.read(as_bytes(k.x[i].w), sizeof(Float));
    }
    for (i = 0; i<np; ++i) {
        ifs.read(as_bytes(k.xysum[i].x), sizeof(Float));
        ifs.read(as_bytes(k.xysum[i].y), sizeof(Float));
    }
    for (i = 0; i<np; ++i) {
        ifs.read(as_bytes(k.vel[i].x), sizeof(Float));
        ifs.read(as_bytes(k.vel[i].y), sizeof(Float));
        ifs.read(as_bytes(k.vel[i].z), sizeof(Float));
        ifs.read(as_bytes(k.vel[i].w), sizeof(Float));
    }
    for (i = 0; i<np; ++i) {
        ifs.read(as_bytes(k.force[i].x), sizeof(Float));
        ifs.read(as_bytes(k.force[i].y), sizeof(Float));
        ifs.read(as_bytes(k.force[i].z), sizeof(Float));
        //ifs.read(as_bytes(k.force[i].w), sizeof(Float));
    }
    for (i = 0; i<np; ++i) {
        ifs.read(as_bytes(k.angpos[i].x), sizeof(Float));
        ifs.read(as_bytes(k.angpos[i].y), sizeof(Float));
        ifs.read(as_bytes(k.angpos[i].z), sizeof(Float));
        //ifs.read(as_bytes(k.angpos[i].w), sizeof(Float));
    }
    for (i = 0; i<np; ++i) {
        ifs.read(as_bytes(k.angvel[i].x), sizeof(Float));
        ifs.read(as_bytes(k.angvel[i].y), sizeof(Float));
        ifs.read(as_bytes(k.angvel[i].z), sizeof(Float));
        //ifs.read(as_bytes(k.angvel[i].w), sizeof(Float));
    }
    for (i = 0; i<np; ++i) {
        ifs.read(as_bytes(k.torque[i].x), sizeof(Float));
        ifs.read(as_bytes(k.torque[i].y), sizeof(Float));
        ifs.read(as_bytes(k.torque[i].z), sizeof(Float));
        //ifs.read(as_bytes(k.torque[i].w), sizeof(Float));
    }

    // Read energies
    for (i = 0; i<np; ++i)
        ifs.read(as_bytes(e.es_dot[i]), sizeof(Float));
    for (i = 0; i<np; ++i)
        ifs.read(as_bytes(e.es[i]), sizeof(Float));
    for (i = 0; i<np; ++i)
        ifs.read(as_bytes(e.ev_dot[i]), sizeof(Float));
    for (i = 0; i<np; ++i)
        ifs.read(as_bytes(e.ev[i]), sizeof(Float));
    for (i = 0; i<np; ++i)
        ifs.read(as_bytes(e.p[i]), sizeof(Float));

    // Read constant parameters
    ifs.read(as_bytes(params.g), sizeof(params.g));
    ifs.read(as_bytes(params.k_n), sizeof(params.k_n));
    ifs.read(as_bytes(params.k_t), sizeof(params.k_t));
    ifs.read(as_bytes(params.k_r), sizeof(params.k_r));
    ifs.read(as_bytes(params.gamma_n), sizeof(params.gamma_n));
    ifs.read(as_bytes(params.gamma_t), sizeof(params.gamma_t));
    ifs.read(as_bytes(params.gamma_r), sizeof(params.gamma_r));
    ifs.read(as_bytes(params.mu_s), sizeof(params.mu_s));
    ifs.read(as_bytes(params.mu_d), sizeof(params.mu_d));
    ifs.read(as_bytes(params.mu_r), sizeof(params.mu_r));
    ifs.read(as_bytes(params.gamma_wn), sizeof(params.gamma_wn));
    ifs.read(as_bytes(params.gamma_wt), sizeof(params.gamma_wt));
    ifs.read(as_bytes(params.mu_ws), sizeof(params.mu_s));
    ifs.read(as_bytes(params.mu_wd), sizeof(params.mu_d));
    ifs.read(as_bytes(params.rho), sizeof(params.rho));
    ifs.read(as_bytes(params.contactmodel), sizeof(params.contactmodel));
    ifs.read(as_bytes(params.kappa), sizeof(params.kappa));
    ifs.read(as_bytes(params.db), sizeof(params.db));
    ifs.read(as_bytes(params.V_b), sizeof(params.V_b));

    // Read wall parameters
    ifs.read(as_bytes(walls.nw), sizeof(walls.nw));
    if (walls.nw > MAXWALLS) {
        cerr << "Error; MAXWALLS (" << MAXWALLS << ") in datatypes.h "
            << "is smaller than the number of walls specified in the "
            << "input file (" << walls.nw << ").\n";
        exit(1);
    }

    // Allocate host memory for walls
    // Wall normal (x,y,z), w: wall position on axis parallel to wall normal
    // Wall mass (x), velocity (y), force (z), and deviatoric stress (w)
    walls.nx    = new Float4[walls.nw];
    walls.mvfd  = new Float4[walls.nw];

    for (i = 0; i<walls.nw; ++i)
        ifs.read(as_bytes(walls.wmode[i]), sizeof(walls.wmode[i]));
    for (i = 0; i<walls.nw; ++i) {
        ifs.read(as_bytes(walls.nx[i].x), sizeof(Float));
        ifs.read(as_bytes(walls.nx[i].y), sizeof(Float));
        ifs.read(as_bytes(walls.nx[i].z), sizeof(Float));
        ifs.read(as_bytes(walls.nx[i].w), sizeof(Float));
    }
    for (i = 0; i<walls.nw; ++i) {
        ifs.read(as_bytes(walls.mvfd[i].x), sizeof(Float));
        ifs.read(as_bytes(walls.mvfd[i].y), sizeof(Float));
        ifs.read(as_bytes(walls.mvfd[i].z), sizeof(Float));
        ifs.read(as_bytes(walls.mvfd[i].w), sizeof(Float));
    }
    ifs.read(as_bytes(params.devs_A), sizeof(params.devs_A));
    ifs.read(as_bytes(params.devs_f), sizeof(params.devs_f));


    // Read bond parameters
    ifs.read(as_bytes(params.lambda_bar), sizeof(params.lambda_bar));
    ifs.read(as_bytes(params.nb0), sizeof(params.nb0));
    ifs.read(as_bytes(params.sigma_b), sizeof(params.sigma_b));
    ifs.read(as_bytes(params.tau_b), sizeof(params.tau_b));
    if (params.nb0 > 0) 
        k.bonds = new uint2[params.nb0];
    k.bonds_delta = new Float4[np];
    k.bonds_omega = new Float4[np];
    for (i = 0; i<params.nb0; ++i) {
        ifs.read(as_bytes(k.bonds[i].x), sizeof(unsigned int));
        ifs.read(as_bytes(k.bonds[i].y), sizeof(unsigned int));
    }
    for (i = 0; i<params.nb0; ++i)   // Normal component
        ifs.read(as_bytes(k.bonds_delta[i].w), sizeof(Float));
    for (i = 0; i<params.nb0; ++i) { // Tangential component
        ifs.read(as_bytes(k.bonds_delta[i].x), sizeof(Float));
        ifs.read(as_bytes(k.bonds_delta[i].y), sizeof(Float));
        ifs.read(as_bytes(k.bonds_delta[i].z), sizeof(Float));
    }
    for (i = 0; i<params.nb0; ++i)   // Normal component
        ifs.read(as_bytes(k.bonds_omega[i].w), sizeof(Float));
    for (i = 0; i<params.nb0; ++i) { // Tangential component
        ifs.read(as_bytes(k.bonds_omega[i].x), sizeof(Float));
        ifs.read(as_bytes(k.bonds_omega[i].y), sizeof(Float));
        ifs.read(as_bytes(k.bonds_omega[i].z), sizeof(Float));
    }

    // Read fluid parameters
    ifs.read(as_bytes(params.nu), sizeof(params.nu));
    unsigned int x, y, z;

    if (verbose == 1)
        cout << "Done\n";

    if (params.nu > 0.0 && darcy == 0) {    // Lattice-Boltzmann flow

        if (verbose == 1)
            cout << "  - Reading LBM values:\t\t\t\t  ";

        //f = new Float[grid.num[0]*grid.num[1]*grid.num[2]*19];
        //f_new = new Float[grid.num[0]*grid.num[1]*grid.num[2]*19];
        v_rho = new Float4[grid.num[0]*grid.num[1]*grid.num[2]];

        for (z = 0; z<grid.num[2]; ++z) {
            for (y = 0; y<grid.num[1]; ++y) {
                for (x = 0; x<grid.num[0]; ++x) {
                    i = x + grid.num[0]*y + grid.num[0]*grid.num[1]*z;
                    ifs.read(as_bytes(v_rho[i].x), sizeof(Float));
                    ifs.read(as_bytes(v_rho[i].y), sizeof(Float));
                    ifs.read(as_bytes(v_rho[i].z), sizeof(Float));
                    ifs.read(as_bytes(v_rho[i].w), sizeof(Float));
                }
            }
        }

        if (verbose == 1)
            cout << "Done" << std::endl;

    } else if (params.nu > 0.0 && darcy == 1) {    // Darcy flow

        const Float cellsizemultiplier = 1.0;
        initDarcyMem(cellsizemultiplier);

        if (verbose == 1)
            cout << "  - Reading Darcy values:\t\t\t  ";

        for (z = 0; z<grid.num[2]; ++z) {
            for (y = 0; y<grid.num[1]; ++y) {
                for (x = 0; x<grid.num[0]; ++x) {
                    i = idx(x,y,z);
                    ifs.read(as_bytes(d_V[i].x), sizeof(Float));
                    ifs.read(as_bytes(d_V[i].y), sizeof(Float));
                    ifs.read(as_bytes(d_V[i].z), sizeof(Float));
                    ifs.read(as_bytes(d_H[i]), sizeof(Float));
                }
            }
        }

        if (verbose == 1)
            cout << "Done" << std::endl;
    }

    // Close file if it is still open
    if (ifs.is_open())
        ifs.close();


}

// Write DEM data to binary file
void DEM::writebin(const char *target)
{
    unsigned int i;

    // Open output file
    std::ofstream ofs(target, std::ios_base::binary);
    if (!ofs) {
        std::cerr << "could create output binary file '"
            << target << std::endl;
        exit(1); // Return unsuccessful exit status
    }

    // If double precision: Values can be written directly
    if (sizeof(Float) == sizeof(double)) {

        ofs.write(as_bytes(nd), sizeof(nd));
        ofs.write(as_bytes(np), sizeof(np));

        // Write time parameters
        ofs.write(as_bytes(time.dt), sizeof(time.dt));
        ofs.write(as_bytes(time.current), sizeof(time.current));
        ofs.write(as_bytes(time.total), sizeof(time.total));
        ofs.write(as_bytes(time.file_dt), sizeof(time.file_dt));
        ofs.write(as_bytes(time.step_count), sizeof(time.step_count));

        // Write grid parameters
        ofs.write(as_bytes(grid.origo), sizeof(grid.origo));
        ofs.write(as_bytes(grid.L), sizeof(grid.L));
        ofs.write(as_bytes(grid.num), sizeof(grid.num));
        ofs.write(as_bytes(grid.periodic), sizeof(grid.periodic));

        // Write kinematic values
        for (i = 0; i<np; ++i) {
            ofs.write(as_bytes(k.x[i].x), sizeof(Float));
            ofs.write(as_bytes(k.x[i].y), sizeof(Float));
            ofs.write(as_bytes(k.x[i].z), sizeof(Float));
            ofs.write(as_bytes(k.x[i].w), sizeof(Float));
        }
        for (i = 0; i<np; ++i) {
            ofs.write(as_bytes(k.xysum[i].x), sizeof(Float));
            ofs.write(as_bytes(k.xysum[i].y), sizeof(Float));
        }
        for (i = 0; i<np; ++i) {
            ofs.write(as_bytes(k.vel[i].x), sizeof(Float));
            ofs.write(as_bytes(k.vel[i].y), sizeof(Float));
            ofs.write(as_bytes(k.vel[i].z), sizeof(Float));
            ofs.write(as_bytes(k.vel[i].w), sizeof(Float));
        }
        for (i = 0; i<np; ++i) {
            ofs.write(as_bytes(k.force[i].x), sizeof(Float));
            ofs.write(as_bytes(k.force[i].y), sizeof(Float));
            ofs.write(as_bytes(k.force[i].z), sizeof(Float));
            //ofs.write(as_bytes(k.force[i].w), sizeof(Float));
        }
        for (i = 0; i<np; ++i) {
            ofs.write(as_bytes(k.angpos[i].x), sizeof(Float));
            ofs.write(as_bytes(k.angpos[i].y), sizeof(Float));
            ofs.write(as_bytes(k.angpos[i].z), sizeof(Float));
            //ofs.write(as_bytes(k.angpos[i].w), sizeof(Float));
        }
        for (i = 0; i<np; ++i) {
            ofs.write(as_bytes(k.angvel[i].x), sizeof(Float));
            ofs.write(as_bytes(k.angvel[i].y), sizeof(Float));
            ofs.write(as_bytes(k.angvel[i].z), sizeof(Float));
            //ofs.write(as_bytes(k.angvel[i].w), sizeof(Float));
        }
        for (i = 0; i<np; ++i) {
            ofs.write(as_bytes(k.torque[i].x), sizeof(Float));
            ofs.write(as_bytes(k.torque[i].y), sizeof(Float));
            ofs.write(as_bytes(k.torque[i].z), sizeof(Float));
            //ofs.write(as_bytes(k.torque[i].w), sizeof(Float));
        }

        // Write energies
        for (i = 0; i<np; ++i)
            ofs.write(as_bytes(e.es_dot[i]), sizeof(Float));
        for (i = 0; i<np; ++i)
            ofs.write(as_bytes(e.es[i]), sizeof(Float));
        for (i = 0; i<np; ++i)
            ofs.write(as_bytes(e.ev_dot[i]), sizeof(Float));
        for (i = 0; i<np; ++i)
            ofs.write(as_bytes(e.ev[i]), sizeof(Float));
        for (i = 0; i<np; ++i)
            ofs.write(as_bytes(e.p[i]), sizeof(Float));

        // Write constant parameters
        ofs.write(as_bytes(params.g), sizeof(params.g));
        ofs.write(as_bytes(params.k_n), sizeof(params.k_n));
        ofs.write(as_bytes(params.k_t), sizeof(params.k_t));
        ofs.write(as_bytes(params.k_r), sizeof(params.k_r));
        ofs.write(as_bytes(params.gamma_n), sizeof(params.gamma_n));
        ofs.write(as_bytes(params.gamma_t), sizeof(params.gamma_t));
        ofs.write(as_bytes(params.gamma_r), sizeof(params.gamma_r));
        ofs.write(as_bytes(params.mu_s), sizeof(params.mu_s));
        ofs.write(as_bytes(params.mu_d), sizeof(params.mu_d));
        ofs.write(as_bytes(params.mu_r), sizeof(params.mu_r));
        ofs.write(as_bytes(params.gamma_wn), sizeof(params.gamma_wn));
        ofs.write(as_bytes(params.gamma_wt), sizeof(params.gamma_wt));
        ofs.write(as_bytes(params.mu_ws), sizeof(params.mu_ws));
        ofs.write(as_bytes(params.mu_wd), sizeof(params.mu_wd));
        ofs.write(as_bytes(params.rho), sizeof(params.rho));
        ofs.write(as_bytes(params.contactmodel), sizeof(params.contactmodel));
        ofs.write(as_bytes(params.kappa), sizeof(params.kappa));
        ofs.write(as_bytes(params.db), sizeof(params.db));
        ofs.write(as_bytes(params.V_b), sizeof(params.V_b));

        // Write wall parameters
        ofs.write(as_bytes(walls.nw), sizeof(walls.nw));
        ofs.write(as_bytes(walls.wmode), sizeof(walls.wmode[0])*walls.nw);
        for (i = 0; i<walls.nw; ++i) {
            ofs.write(as_bytes(walls.nx[i].x), sizeof(Float));
            ofs.write(as_bytes(walls.nx[i].y), sizeof(Float));
            ofs.write(as_bytes(walls.nx[i].z), sizeof(Float));
            ofs.write(as_bytes(walls.nx[i].w), sizeof(Float));
        }
        for (i = 0; i<walls.nw; ++i) {
            ofs.write(as_bytes(walls.mvfd[i].x), sizeof(Float));
            ofs.write(as_bytes(walls.mvfd[i].y), sizeof(Float));
            ofs.write(as_bytes(walls.mvfd[i].z), sizeof(Float));
            ofs.write(as_bytes(walls.mvfd[i].w), sizeof(Float));
        }
        ofs.write(as_bytes(params.devs_A), sizeof(params.devs_A));
        ofs.write(as_bytes(params.devs_f), sizeof(params.devs_f));

        // Write bond parameters
        ofs.write(as_bytes(params.lambda_bar), sizeof(params.lambda_bar));
        ofs.write(as_bytes(params.nb0), sizeof(params.nb0));
        ofs.write(as_bytes(params.sigma_b), sizeof(params.sigma_b));
        ofs.write(as_bytes(params.tau_b), sizeof(params.tau_b));
        for (i = 0; i<params.nb0; ++i) {
            ofs.write(as_bytes(k.bonds[i].x), sizeof(unsigned int));
            ofs.write(as_bytes(k.bonds[i].y), sizeof(unsigned int));
        }
        for (i = 0; i<params.nb0; ++i)   // Normal component
            ofs.write(as_bytes(k.bonds_delta[i].w), sizeof(Float));
        for (i = 0; i<params.nb0; ++i) { // Tangential component
            ofs.write(as_bytes(k.bonds_delta[i].x), sizeof(Float));
            ofs.write(as_bytes(k.bonds_delta[i].y), sizeof(Float));
            ofs.write(as_bytes(k.bonds_delta[i].z), sizeof(Float));
        }
        for (i = 0; i<params.nb0; ++i)   // Normal component
            ofs.write(as_bytes(k.bonds_omega[i].w), sizeof(Float));
        for (i = 0; i<params.nb0; ++i) { // Tangential component
            ofs.write(as_bytes(k.bonds_omega[i].x), sizeof(Float));
            ofs.write(as_bytes(k.bonds_omega[i].y), sizeof(Float));
            ofs.write(as_bytes(k.bonds_omega[i].z), sizeof(Float));
        }

        ofs.write(as_bytes(params.nu), sizeof(params.nu));
        int x, y, z;
        if (params.nu > 0.0 && darcy == 0) {    // Lattice Boltzmann flow
            for (z = 0; z<grid.num[2]; ++z) {
                for (y = 0; y<grid.num[1]; ++y) {
                    for (x = 0; x<grid.num[0]; ++x) {
                        i = x + grid.num[0]*y + grid.num[0]*grid.num[1]*z;
                        ofs.write(as_bytes(v_rho[i].x), sizeof(Float));
                        ofs.write(as_bytes(v_rho[i].y), sizeof(Float));
                        ofs.write(as_bytes(v_rho[i].z), sizeof(Float));
                        ofs.write(as_bytes(v_rho[i].w), sizeof(Float));
                    }
                }
            }
        } else if (params.nu > 0.0 && darcy == 1) { // Darcy flow
            for (z=0; z<d_nz; z++) {
                for (y=0; y<d_ny; y++) {
                    for (x=0; x<d_nx; x++) {
                        i = idx(x,y,z);
                        ofs.write(as_bytes(d_V[i].x), sizeof(Float));
                        ofs.write(as_bytes(d_V[i].y), sizeof(Float));
                        ofs.write(as_bytes(d_V[i].z), sizeof(Float));
                        ofs.write(as_bytes(d_H[i]), sizeof(Float));
                        //printf("%d,%d,%d: d_H[%d] = %f\n", x,y,z, i, d_H[i]);
                    }
                }
            }
        }

        // Close file if it is still open
        if (ofs.is_open())
            ofs.close();

    } else {
        std::cerr << "Can't write output when in single precision mode.\n";
        exit(1);
    }
}

// Write image structure to PPM file
void DEM::writePPM(const char *target)
{
    // Open output file
    std::ofstream ofs(target);
    if (!ofs) {
        std::cerr << "Could not create output PPM file '"
            << target << std::endl;
        exit(1); // Return unsuccessful exit status
    }

    if (verbose == 1)
        std::cout << "  Saving image: " << target << std::endl;

    // Write PPM header
    ofs << "P6 " << width << " " << height << " 255\n";

    // Write pixel array to ppm image file
    for (unsigned int i=0; i<height*width; ++i)
        ofs << img[i].r << img[i].g << img[i].b;

    // Close file if it is still open
    if (ofs.is_open())
        ofs.close();
}

// Write write depth vs. porosity values to file
void DEM::writePorosities(
        const char *target,
        const int z_slices,
        const Float *z_pos,
        const Float *porosity)
{
    // Open output file
    std::ofstream ofs(target);
    if (!ofs) {
        std::cerr << "Could not create output porosity file '"
            << target << std::endl;
        exit(1); // Return unsuccessful exit status
    }

    if (verbose == 1)
        std::cout << "  Saving porosities: " << target << std::endl;

    // Write pixel array to ppm image file
    for (int i=0; i<z_slices; ++i)
        ofs << z_pos[i] << '\t' << porosity[i] << '\n';

    // Close file if it is still open
    if (ofs.is_open())
        ofs.close();
}



// vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
