#ifndef TYPEDEFS_H_
#define TYPEDEFS_H_

#include "vector_functions.h"

//////////////////////
// TYPE DEFINITIONS //
//////////////////////


// Uncomment all five lines below for single precision
/*
typedef Float Float;
typedef Float3 Float3;
typedef Float4 Float4;
#define MAKE_FLOAT3(x, y, z) make_Float3(x, y, z)
#define MAKE_FLOAT4(x, y, z, w) make_Float4(x, y, z, w)
*/


/////// BEWARE: single precision is non-functional at the moment,
//	due to the input algorithm.

// Uncomment all five lines below for double precision
///*
typedef double Float;
typedef double2 Float2;
typedef double3 Float3;
typedef double4 Float4;
#define MAKE_FLOAT3(x, y, z) make_double3(x, y, z)
#define MAKE_FLOAT4(x, y, z, w) make_double4(x, y, z, w)
//*/

#endif
