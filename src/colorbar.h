#ifndef COLORBAR_H_
#define COLORBAR_H_

// Functions that determine red-, green- and blue color components
// in a blue-white-red colormap. Ratio should be between 0.0-1.0.

__inline__ __host__ __device__ float red(float ratio)
{
    return fmin(1.0f, 0.209f*ratio*ratio*ratio - 2.49f*ratio*ratio + 3.0f*ratio + 0.0109f);
};

__inline__ __host__ __device__ float green(float ratio)
{
    return fmin(1.0f, -2.44f*ratio*ratio + 2.15f*ratio + 0.369f);
};

__inline__ __host__ __device__ float blue(float ratio)
{
    return fmin(1.0f, -2.21f*ratio*ratio + 1.61f*ratio + 0.573f);
};

#endif
// vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
