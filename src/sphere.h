// Make sure the header is only included once
#ifndef SPHERE_H_
#define SPHERE_H_

#include "datatypes.h"

// DEM class
class DEM {

  // Values and functions only accessible from the class internally
  private:

    // Simulation ID
    char *sid;

    // Output level
    int verbose;

    // Number of dimensions
    unsigned int nd;

    // Number of particles
    unsigned int np;

    // Structure containing individual particle kinematics
    Kinematics k;	// host
    Kinematics *dev_k;	// device

    // Structure containing energy values
    Energies e;		// host
    Energies *dev_e;	// device

    // Structure of global parameters
    Params params;	// host

    // Structure containing spatial parameters
    Grid grid;		// host

    // Structure containing sorting arrays
    Sorting *dev_sort;	// device

    // Structure of temporal parameters
    Time time;		// host
    Time *dev_time;	// device

    // Structure of wall parameters
    Walls walls;	// host
    Walls *dev_walls;	// device

    // GPU initialization, must be called before startTime()
    void initializeGPU(void);

    // Copy all constant data to constant device memory
    void transferToConstantDeviceMemory(void);

    // Check values stored in constant device memory
    void checkConstantMemory(void);

    // Allocate global device memory to hold data
    void allocateGlobalDeviceMemory(void);

    // Free dynamically allocated global device memory
    void freeGlobalDeviceMemory(void);

    // Copy non-constant data to global GPU memory
    void transferToGlobalDeviceMemory(void);

    // Copy non-constant data from global GPU memory to host RAM
    void transferFromGlobalDeviceMemory(void);


  // Values and functions accessible from the outside
  public:

    // Constructor, some parameters with default values
    DEM(char *inputbin, 
	const int verbosity = 1,
	const int checkVals = 1);

    // Destructor
    ~DEM(void);

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
