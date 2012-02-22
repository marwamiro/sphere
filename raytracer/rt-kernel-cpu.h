#ifndef RT_KERNEL_CPU_H_
#define RT_KERNEL_CPU_H_

#include <vector_functions.h>

// Host prototype functions

void cameraInit(float3 eye, float3 lookat, float imgw, float hw_ratio);

int rt_cpu(float4* p, const unsigned int np, 
       rgb* img, const unsigned int width, const unsigned int height,
       f3 origo, f3 L, f3 eye, f3 lookat, float imgw);

#endif
