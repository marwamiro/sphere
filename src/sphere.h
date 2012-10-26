// Make sure the header is only included once
#ifndef SPHERE_H_
#define SPHERE_H_

#include "datatypes.h"

// DEM class
class DEM {

  // Values and functions only accessible from the class internally
  private:

    // Output level
    int verbose;

    // Number of dimensions
    unsigned int nd;

    // Number of particles
    unsigned int np;

    // Structure containing individual particle kinematics
    Kinematics k;
    Kinematics dev_k;

    // Structure containing energy values
    Energies e;
    Energies dev_e;

    // Structure of global parameters
    Params params;
    Params dev_params;

    // Structure containing spatial parameters
    Grid grid;
    Grid dev_grid;

    // Structure of temporal parameters
    Time time;
    Time dev_time;

    // Structure of wall parameters
    Walls walls;
    Walls dev_walls;

    // Wall force arrays
    Float* dev_w_force; 

    // GPU initialization, must be called before startTime()
    void initializeGPU(void);

    // Copy all constant data to constant device memory
    void transferToConstantDeviceMemory(void);

    // Check values stored in constant device memory
    void checkConstantMemory(void);

    // Allocate global device memory to hold data
    void allocateGlobalDeviceMemory(void);

    // Copy non-constant data to global GPU memory
    void transferToGlobalDeviceMemory(void);

    // Copy non-constant data from global GPU memory to host RAM
    void transferFromGlobalDeviceMemory(void);


  // Values and functions accessible from the outside
  public:

    // Constructor, some parameters with default values
    DEM(const char *inputbin, 
	const int verbosity = 1,
	const int checkVals = 1);

    // Read binary input file
    void readbin(const char *target);

    // Write binary output file
    void writebin(const char *target);

    // Check numeric values of selected parameters
    void checkValues(void);

    // Iterate through time, using temporal limits
    // described in "time" struct.
    void startTime(void);

    // Render particles using raytracing
    // render(const char *target,
    //        Float3 lookat,
    //        Float3 eye,
    //        Float  focalLength);

};

#endif
