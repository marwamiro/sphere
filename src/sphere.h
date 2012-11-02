// Make sure the header is only included once
#ifndef SPHERE_H_
#define SPHERE_H_

#include "datatypes.h"

// DEM class
class DEM {

  // Values and functions only accessible from the class internally
  private:

    // Input filename (full path)
    std::string inputbin;

    // Simulation ID
    std::string sid;

    // Output level
    int verbose;

    // Number of dimensions
    unsigned int nd;

    // Number of particles
    unsigned int np;

    // HOST STRUCTURES
    // Structure containing individual particle kinematics
    Kinematics k;	// host
    //Kinematics *dev_k;	// device

    // Structure containing energy values
    Energies e;		// host
    //Energies *dev_e;	// device

    // Structure of global parameters
    Params params;	// host

    // Structure containing spatial parameters
    Grid grid;		// host

    // Structure containing sorting arrays
    //Sorting *dev_sort;	// device

    // Structure of temporal parameters
    Time time;		// host
    //Time *dev_time;	// device

    // Structure of wall parameters
    Walls walls;	// host
    //Walls *dev_walls;	// device

    // DEVICE ARRAYS
    Float4 *dev_x;
    Float2 *dev_xysum;
    Float4 *dev_vel;
    Float4 *dev_acc;
    Float4 *dev_force;
    Float4 *dev_angpos;
    Float4 *dev_angvel;
    Float4 *dev_angacc;
    Float4 *dev_torque;
    unsigned int *dev_contacts;
    Float4 *dev_distmod;
    Float4 *dev_delta_t;
    Float *dev_es_dot;
    Float *dev_es;
    Float *dev_ev_dot;
    Float *dev_ev;
    Float *dev_p;
    Float4 *dev_x_sorted;
    Float4 *dev_vel_sorted;
    Float4 *dev_angvel_sorted;
    unsigned int *dev_gridParticleCellID;
    unsigned int *dev_gridParticleIndex;
    unsigned int *dev_cellStart;
    unsigned int *dev_cellEnd;
    int *dev_walls_wmode;
    Float4 *dev_walls_nx; // normal, pos.
    Float4 *dev_walls_mvfd; // Mass, velocity, force, dev. stress
    Float *dev_walls_force_partial; // Pre-sum per wall
    Float *dev_walls_force_pp; // Force per particle per wall

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
    DEM(std::string inputbin, 
	const int verbosity = 1,
	const int checkVals = 1,
	const int render = 0);

    // Destructor
    ~DEM(void);

    // Read binary input file
    void readbin(const char *target);

    // Write binary output file
    void writebin(const char *target);

    // Check numeric values of selected parameters
    void checkValues(void);

    // Report key parameter values to stdout
    void reportValues(void);

    // Iterate through time, using temporal limits
    // described in "time" struct.
    void startTime(void);

    // Render particles using raytracing
    void render(const char *target,
	const Float3 lookat,
	const Float3 eye,
	const Float focalLength = 1.0,
	const int method = 1);

};

#endif
