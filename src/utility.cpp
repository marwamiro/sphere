// MISC. UTILITY FUNCTIONS

#include "datatypes.h"


//Round a / b to nearest higher integer value
unsigned int iDivUp(unsigned int a, unsigned int b)
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

// Swap two arrays pointers
void swapFloatArrays(Float* arr1, Float* arr2)
{
    Float* tmp = arr1;
    arr1 = arr2;
    arr2 = tmp;
}

// Get minimum value in 1D array 
Float minVal(Float* arr, int length)
{
    Float min = 1.0e13; // initialize with a large number
    unsigned int i;
    Float val;
    for (i=0; i<length; ++i) {
        val = arr[i];
        if (val < min) min = val;
    }
    return min;
}

// Get maximum value in 1d array
Float maxVal(Float* arr, int length)
{
    Float max = -1.0e13; // initialize with a small number
    unsigned int i;
    Float val;
    for (i=0; i<length; ++i) {
        val = arr[i];
        if (val > max) max = val;
    }
    return max;
}

// Get minimum value in 3d array
Float minVal(Float3* arr, int length)
{
    Float min = 1.0e13; // initialize with a large number
    unsigned int i;
    Float3 val;
    for (i=0; i<length; ++i) {
        val = arr[i];
        if (val.x < min) min = val.x;
        if (val.y < min) min = val.y;
        if (val.z < min) min = val.z;
    }
    return min;
}

// Get maximum value in 3d array
Float maxVal(Float3* arr, int length)
{
    Float max = -1.0e13; // initialize with a small number
    unsigned int i;
    Float3 val;
    for (i=0; i<length; ++i) {
        val = arr[i];
        if (val.x > max) max = val.x;
        if (val.y > max) max = val.y;
        if (val.z > max) max = val.z;
    }
    return max;
}
