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
    int verbose = 1;
    int dry = 0;


    // Process input parameters
    int i;
    for (i=1; i<argc; ++i) {	// skip argv[0]

        std::string argvi = std::string(argv[i]);

        // Display help if requested
        if (argvi == "-h" || argvi == "--help") {
            std::cout << argv[0] << ": particle dynamics simulator\n"
                << "Usage: " << argv[0] << " [OPTION[S]]... [FILE1 ...]\nOptions:\n"
                << "-h, --help\t\tprint help\n"
                << "-n, --dry\t\tshow key experiment parameters and quit\n"
                << std::endl;
            return 0; // Exit with success
        }

        else if (argvi == "-n" || argvi == "--dry")
            dry = 1;
        }


        // The rest of the values must be input binary files
        else {
            if (verbose == 1)
                std::cout << argv[0] << ": processing input file: " << argvi <<
                    std::endl;

            // Create DEM class, read data from input binary,
            // do not check values, do not init cuda, do not transfer const
            // mem
            DEM dem(argvi, verbose, 0, dry, 0, 0);

            // Otherwise, start iterating through time
            else
                dem.startDarcy();


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
