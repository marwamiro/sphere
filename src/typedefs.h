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
#define MAKE_FLOAT2(x, y) make_float2(x, y)
#define MAKE_FLOAT3(x, y, z) make_float3(x, y, z)
#define MAKE_FLOAT4(x, y, z, w) make_float4(x, y, z, w)
*/


/////// BEWARE: single precision is non-functional at the moment,
//	due to the input algorithm.

// Uncomment all five lines below for double precision
///*
typedef double Float;
typedef double2 Float2;
typedef double3 Float3;
typedef double4 Float4;
#define MAKE_FLOAT2(x, y) make_double2(x, y)
#define MAKE_FLOAT3(x, y, z) make_double3(x, y, z)
#define MAKE_FLOAT4(x, y, z, w) make_double4(x, y, z, w)
//*/

#endif
// vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
