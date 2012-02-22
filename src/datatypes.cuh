// datatypes.cuh -- Device structure templates and function prototypes

// Avoiding multiple inclusions of header file
#ifndef DATATYPES_CUH_
#define DATATYPES_CUH_

#include "vector_functions.h"

unsigned int iDivUp(unsigned int a, unsigned int b);
void checkForCudaErrors(const char* checkpoint_description);
void checkForCudaErrors(const char* checkpoint_description, const unsigned int iteration);
  
#endif
