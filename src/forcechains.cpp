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
    int threedim = 1; // 0 if 2d, 1 if 3d
    double lowercutoff = 0.0;
    double uppercutoff = 1.0e9;
    std::string format = "interactive"; // gnuplot terminal type

    // Process input parameters
    int i;
    for (i=1; i<argc; ++i) {	// skip argv[0]

        std::string argvi = std::string(argv[i]);

        // Display help if requested
        if (argvi == "-h" || argvi == "--help") {
            std::cout << argv[0] << ": sphere force chain visualizer\n"
                << "Usage: " << argv[0] << " [OPTION[S]]... [FILE1 ...] > outputfile\nOptions:\n"
                << "-h, --help\t\tPrint help\n"
                << "-V, --version\t\tPrint version information and exit\n"
                << "-v, --verbose\t\tDisplay in-/output file names\n"
                << "-lc <val>, --lower-cutoff <val>\t\tOnly show contacts where the force value is greater\n"
                << "-uc <val>, --upper-cutoff <val>\t\tOnly show contacts where the force value is greater\n"
                << "-f, --format\t\tOutput format to stdout, interactive default. Possible values:\n"
                << "\t\t\tinteractive, png, epslatex, epslatex-color\n"
                << "-2d\t\t\twrite output as 2d coordinates (3d default)\n"
                << "The values below the cutoff are not visualized, the values above are truncated to the upper limit\n"
                << std::endl;
            return 0; // Exit with success
        }

        // Display version with fancy ASCII art
        else if (argvi == "-V" || argvi == "--version") {
            std::cout << "Force chain calculator, sphere version " << VERS
                << std::endl;
            return 0;
        }

        else if (argvi == "-v" || argvi == "--verbose")
            verbose = 1;

        else if (argvi == "-f" || argvi == "--format")
            format = argv[++i];

        else if (argvi == "-lc" || argvi == "--lower-cutoff")
            lowercutoff = atof(argv[++i]);

        else if (argvi == "-uc" || argvi == "--upper-cutoff")
            uppercutoff = atof(argv[++i]);

        else if (argvi == "-2d")
            threedim = 0;

        // The rest of the values must be input binary files
        else {
            nfiles++;

            if (verbose == 1)
                std::cout << argv[0] << ": processing input file: " << argvi << std::endl;

            // Create DEM class, read data from input binary, check values
            DEM dem(argvi, verbose, 0, 0, 0);

            // Calculate porosity and save as file
            dem.forcechains(format, threedim, lowercutoff, uppercutoff);

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
