// Avoiding multiple inclusions of header file
#ifndef UTILITY_CUH_
#define UTILITY_CUH_

unsigned int iDivUp(unsigned int a, unsigned int b);
void checkForCudaErrors(const char* checkpoint_description);
void checkForCudaErrors(const char* checkpoint_description, const unsigned int iteration);

#endif
// vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
