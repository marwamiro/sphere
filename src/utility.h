#ifndef UTILITY_H_
#define UTILITY_H_

// MISC. UTILITY FUNCTIONS


//Round a / b to nearest higher integer value
unsigned int iDivUp(unsigned int a, unsigned int b) {
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

// Swap two arrays pointers
void swapFloatArrays(Float* arr1, Float* arr2)
{
    Float* tmp = arr1;
    arr1 = arr2;
    arr2 = tmp;
}


#endif
