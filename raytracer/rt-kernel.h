#ifndef RT_KERNEL_H_
#define RT_KERNEL_H_

#include <vector_functions.h>
#include "../src/datatypes.h"
#include "header.h"

// Constants
__constant__ float4 const_u;
__constant__ float4 const_v;
__constant__ float4 const_w;
__constant__ float4 const_eye;
__constant__ float4 const_imgplane;
__constant__ float  const_d;
__constant__ float4 const_light;
__constant__ unsigned int const_pixels;
__constant__ Inttype const_np;


// Host prototype functions

extern "C"
void cameraInit(float4 eye, float4 lookat, float imgw, float hw_ratio,
    		unsigned int pixels, Inttype np);

extern "C"
void checkForCudaErrors(const char* checkpoint_description);

extern "C"
int rt(float4* p, Inttype np,
       rgb* img, unsigned int width, unsigned int height,
       f3 origo, f3 L, f3 eye, f3 lookat, float imgw,
       int visualize, float max_val,
       float* fixvel, float* xsum, float* pres, float* es_dot, float* es, float* vel);
#endif
