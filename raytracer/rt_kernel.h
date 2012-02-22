#ifndef RT_KERNEL_H_
#define RT_KERNEL_H_

#include <vector_functions.h>

// Constants
__constant__ float4 const_u;
__constant__ float4 const_v;
__constant__ float4 const_w;
__constant__ float4 const_eye;
__constant__ float4 const_imgplane;
__constant__ float  const_d;
__constant__ float4 const_light;


// Host prototype functions

extern "C"
void cameraInit(float4 eye, float4 lookat, float imgw, float hw_ratio);

extern "C"
void checkForCudaErrors(const char* checkpoint_description);

extern "C"
int rt(float4* p, const unsigned int np, 
       rgb* img, const unsigned int width, const unsigned int height,
       f3 origo, f3 L, f3 eye, f3 lookat, float imgw,
       const int visualize, const float max_val, 
       float* pres, float* es_dot, float* es);

#endif
