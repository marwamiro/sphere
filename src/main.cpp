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

  // LOCAL VARIABLE DECLARATIONS
  if(!argv[1] || argc != 2) {
    std::cerr << "Error: Specify input binary file, e.g. "
      << argv[0] << " input/test.bin" << std::endl;
    return 1; // Return unsuccessful exit status
  }
  //char *inputbin = argv[1]; // Input binary file read from command line argument
  std::string inputbin = argv[1];

  int verbose = 1;
  int checkVals = 1;

  if (verbose == 1) {
    // Opening cerenomy with fancy ASCII art
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
  }

  std::cout << "Input file: " << inputbin << std::endl;
  
  // Create DEM class, read data from input binary, check values
  DEM dem(inputbin, verbose, checkVals);

  // Start iterating through time
  dem.startTime();

  // Terminate execution
  std::cout << "\nBye!\n";
  return 0; // Return successfull exit status
} 
// END OF FILE
