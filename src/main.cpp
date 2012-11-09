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
  int checkVals = 1;
  int render = 0; // whether to render an image
  int method = 0; // visualization method
  int nfiles = 0; // number of input files
  float max_val = 0.0f;

  // Process input parameters
  int i;
  for (i=1; i<argc; ++i) {	// skip argv[0]

    std::string argvi = std::string(argv[i]);

    // Display help if requested
    if (argvi == "-h" || argvi == "--help") {
      std::cout << argv[0] << ": particle dynamics simulator\n"
	<< "Usage: " << argv[0] << " [OPTION[S]]... [FILE1 ...]\nOptions:\n"
	<< "-h, --help\t\tprint help\n"
	<< "-V, --version\t\tprint version information and exit\n"
	<< "-q, --quiet\t\tsuppress status messages to stdout\n"
	<< "-r, --render\t\trender input files instead of simulating temporal evolution\n"
	<< "-dcv, --dpnt-check-values\t\tdon't check values before running\n" 
	<< "Raytracer (-r) specific options:\n"
	<< "-m <method> <maxval>, --method <method> <maxval>\tcolor visualization method, possible values:\n"
	<< "\t\t\t\tpres, vel, angvel, xdisp, angpos\n"
	<< std::endl;
      return 0; // Exit with success
    }

    // Display version with fancy ASCII art
    else if (argvi == "-V" || argvi == "--version") {
      std::cout << ".-------------------------------------.\n"
	<< "|              _    Compiled for " << ND << "D   |\n" 
	<< "|             | |                     |\n" 
	<< "|    ___ _ __ | |__   ___ _ __ ___    |\n"
	<< "|   / __| '_ \\| '_ \\ / _ \\ '__/ _ \\   |\n"
	<< "|   \\__ \\ |_) | | | |  __/ | |  __/   |\n"
	<< "|   |___/ .__/|_| |_|\\___|_|  \\___|   |\n"
	<< "|       | |                           |\n"
	<< "|       |_|           Version: " << VERS << "   |\n"           
	<< "`-------------------------------------´\n";
      return 0;
    }

    else if (argvi == "-q" || argvi == "--quiet")
      verbose = 0;

    else if (argvi == "-r" || argvi == "--render")
      render = 1;

    else if (argvi == "-dcv" || argvi == "--dont-check-values")
      checkVals = 0;

    else if (argvi == "-m" || argvi == "--method") {

      // Find out which
      if (std::string(argv[i+1]) == "pres")
	method = 1;
      else if (std::string(argv[i+1]) == "vel")
	method = 2;
      else if (std::string(argv[i+1]) == "angvel")
	method = 3;
      else if (std::string(argv[i+1]) == "xdisp")
	method = 4;
      else if (std::string(argv[i+1]) == "angpos")
	method = 5;
      else {
	std::cerr << "Visualization method not understood. See `"
	  << argv[0] << " --help` for more information." << std::endl;
	exit(1);
      }

      // Read max. value of colorbar as next argument
      max_val = atof(argv[i+2]);
      i += 2; // skip ahead
    }


    // The rest of the values must be input binary files
    else {
      nfiles++;

      std::cout << argv[0] << ": processing input file: " << argvi << std::endl;

      // Create DEM class, read data from input binary, check values
      DEM dem(argvi, verbose, checkVals);

      // Render image if requested
      if (render == 1)
	dem.render(method, max_val);

      // Otherwise, start iterating through time
      else
	dem.startTime();
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
