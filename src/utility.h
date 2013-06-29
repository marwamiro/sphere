#ifndef UTILITY_H_
#define UTILITY_H_

// MISC. UTILITY FUNCTIONS


//Round a / b to nearest higher integer value
unsigned int iDivUp(unsigned int a, unsigned int b);

// Swap two arrays pointers
void swapFloatArrays(Float* arr1, Float* arr2);

// Get minimum/maximum value in 1D or 3D array 
Float minVal(Float* arr, int length);
Float minVal(Float3* arr, int length);
Float maxVal(Float* arr, int length);
Float maxVal(Float3* arr, int length);

#endif
// vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
