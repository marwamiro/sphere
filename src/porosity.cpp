/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*  SPHERE source code by Anders Damsgaard Christensen, 2010-12,       */
/*  a 3D Discrete Element Method algorithm with CUDA GPU acceleration. */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

// Licence: GNU Public License (GPL) v. 3. See license.txt.
// See doc/sphere-doc.pdf for full documentation.
// Compile with GNU make by typing 'make' in the src/ directory.               
// SPHERE is called from the command line with './sphere_<architecture> projectname' 


// Including library files
#include <iostream>
#include <string>
#include <cstdlib>

// Including user files
#include "constants.h"
#include "datatypes.h"
#include "sphere.h"

//////////////////
// MAIN ROUTINE //
//////////////////
// The main loop returns the value 0 to the shell, if the program terminated
// successfully, and 1 if an error occured which caused the program to crash.
int main(const int argc, const char *argv[]) 
{
    // Default values
    int verbose = 0;
    int nfiles = 0; // number of input files
    int slices = 10; // number of vertical slices

    // Process input parameters
    int i;
    for (i=1; i<argc; ++i) {	// skip argv[0]

        std::string argvi = std::string(argv[i]);

        // Display help if requested
        if (argvi == "-h" || argvi == "--help") {
            std::cout << argv[0] << ": sphere porosity calculator\n"
                << "Usage: " << argv[0] << " [OPTION[S]]... [FILE1 ...]\nOptions:\n"
                << "-h, --help\t\tprint help\n"
                << "-V, --version\t\tprint version information and exit\n"
                << "-v, --verbose\t\tdisplay in-/output file names\n"
                << "-s. --slices\t\tnumber of vertical slices to find porosity within\n"
                << "The porosity values are stored in the output/ folder"
                << std::endl;
            return 0; // Exit with success
        }

        // Display version with fancy ASCII art
        else if (argvi == "-V" || argvi == "--version") {
            std::cout << "Porosity calculator, sphere version " << VERS
                << std::endl;
            return 0;
        }

        else if (argvi == "-v" || argvi == "--verbose")
            verbose = 1;

        else if (argvi == "-s" || argvi == "--slices") {
            slices = atoi(argv[++i]); 
            if (slices < 1) {
                std::cerr << "Error: The number of slices must be a positive, real number (was "
                    << slices << ")" << std::endl;
                return 1;
            }
        }

        // The rest of the values must be input binary files
        else {
            nfiles++;

            if (verbose == 1)
                std::cout << argv[0] << ": processing input file: " << argvi << std::endl;

            // Create DEM class, read data from input binary, check values
            DEM dem(argvi, verbose, 0, 0, 0);

            // Calculate porosity and save as file
            dem.porosity(slices);

        }
    }

    // Check whether there are input files specified
    if (!argv[0] || argc == 1 || nfiles == 0) {
        std::cerr << argv[0] << ": missing input binary file\n"
            << "See `" << argv[0] << " --help` for more information"
            << std::endl;
        return 1; // Return unsuccessful exit status
    }

    return 0; // Return successfull exit status
} 
// END OF FILE
// vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
