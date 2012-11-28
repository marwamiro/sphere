#ifndef VECTOR_ARITHMETIC_H_
#define VECTOR_ARITHMETIC_H_

#include "cuda_runtime.h"
#include "datatypes.h"

typedef unsigned int uint;
typedef unsigned short ushort;

#ifndef __CUDACC__
#include <math.h>

////////////////////////////////////////////////////////////////////////////////
// host implementations of CUDA functions
////////////////////////////////////////////////////////////////////////////////

inline Float fminf(Float a, Float b)
{
    return a < b ? a : b;
}

inline Float fmaxf(Float a, Float b)
{
    return a > b ? a : b;
}

inline Float rsqrtf(Float x)
{
    return 1.0f / sqrtf(x);
}
#endif

////////////////////////////////////////////////////////////////////////////////
// negate
////////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ Float3 operator-(Float3 &a)
{
    return MAKE_FLOAT3(-a.x, -a.y, -a.z);
}
inline __host__ __device__ Float4 operator-(Float4 &a)
{
    return MAKE_FLOAT4(-a.x, -a.y, -a.z, -a.w);
}

////////////////////////////////////////////////////////////////////////////////
// addition
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ void operator+=(Float2 &a, Float2 b)
{
    a.x += b.x; a.y += b.y;
}
inline __host__ __device__ Float3 operator+(Float3 a, Float3 b)
{
    return MAKE_FLOAT3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __host__ __device__ void operator+=(Float3 &a, Float3 b)
{
    a.x += b.x; a.y += b.y; a.z += b.z;
}
inline __host__ __device__ Float3 operator+(Float3 a, Float b)
{
    return MAKE_FLOAT3(a.x + b, a.y + b, a.z + b);
}
inline __host__ __device__ void operator+=(Float3 &a, Float b)
{
    a.x += b; a.y += b; a.z += b;
}

inline __host__ __device__ Float3 operator+(Float b, Float3 a)
{
    return MAKE_FLOAT3(a.x + b, a.y + b, a.z + b);
}

inline __host__ __device__ Float4 operator+(Float4 a, Float4 b)
{
    return MAKE_FLOAT4(a.x + b.x, a.y + b.y, a.z + b.z,  a.w + b.w);
}
inline __host__ __device__ void operator+=(Float4 &a, Float4 b)
{
    a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
}
inline __host__ __device__ Float4 operator+(Float4 a, Float b)
{
    return MAKE_FLOAT4(a.x + b, a.y + b, a.z + b, a.w + b);
}
inline __host__ __device__ Float4 operator+(Float b, Float4 a)
{
    return MAKE_FLOAT4(a.x + b, a.y + b, a.z + b, a.w + b);
}
inline __host__ __device__ void operator+=(Float4 &a, Float b)
{
    a.x += b; a.y += b; a.z += b; a.w += b;
}

////////////////////////////////////////////////////////////////////////////////
// subtract
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ Float3 operator-(Float3 a, Float3 b)
{
    return MAKE_FLOAT3(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline __host__ __device__ void operator-=(Float3 &a, Float3 b)
{
    a.x -= b.x; a.y -= b.y; a.z -= b.z;
}
inline __host__ __device__ Float3 operator-(Float3 a, Float b)
{
    return MAKE_FLOAT3(a.x - b, a.y - b, a.z - b);
}
inline __host__ __device__ Float3 operator-(Float b, Float3 a)
{
    return MAKE_FLOAT3(b - a.x, b - a.y, b - a.z);
}
inline __host__ __device__ void operator-=(Float3 &a, Float b)
{
    a.x -= b; a.y -= b; a.z -= b;
}

inline __host__ __device__ Float4 operator-(Float4 a, Float4 b)
{
    return MAKE_FLOAT4(a.x - b.x, a.y - b.y, a.z - b.z,  a.w - b.w);
}
inline __host__ __device__ void operator-=(Float4 &a, Float4 b)
{
    a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w;
}
inline __host__ __device__ Float4 operator-(Float4 a, Float b)
{
    return MAKE_FLOAT4(a.x - b, a.y - b, a.z - b,  a.w - b);
}
inline __host__ __device__ void operator-=(Float4 &a, Float b)
{
    a.x -= b; a.y -= b; a.z -= b; a.w -= b;
}

////////////////////////////////////////////////////////////////////////////////
// multiply
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ Float3 operator*(Float3 a, Float3 b)
{
    return MAKE_FLOAT3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline __host__ __device__ void operator*=(Float3 &a, Float3 b)
{
    a.x *= b.x; a.y *= b.y; a.z *= b.z;
}
inline __host__ __device__ Float3 operator*(Float3 a, Float b)
{
    return MAKE_FLOAT3(a.x * b, a.y * b, a.z * b);
}
inline __host__ __device__ Float3 operator*(Float b, Float3 a)
{
    return MAKE_FLOAT3(b * a.x, b * a.y, b * a.z);
}
inline __host__ __device__ void operator*=(Float3 &a, Float b)
{
    a.x *= b; a.y *= b; a.z *= b;
}

inline __host__ __device__ Float4 operator*(Float4 a, Float4 b)
{
    return MAKE_FLOAT4(a.x * b.x, a.y * b.y, a.z * b.z,  a.w * b.w);
}
inline __host__ __device__ void operator*=(Float4 &a, Float4 b)
{
    a.x *= b.x; a.y *= b.y; a.z *= b.z; a.w *= b.w;
}
inline __host__ __device__ Float4 operator*(Float4 a, Float b)
{
    return MAKE_FLOAT4(a.x * b, a.y * b, a.z * b,  a.w * b);
}
inline __host__ __device__ Float4 operator*(Float b, Float4 a)
{
    return MAKE_FLOAT4(b * a.x, b * a.y, b * a.z, b * a.w);
}
inline __host__ __device__ void operator*=(Float4 &a, Float b)
{
    a.x *= b; a.y *= b; a.z *= b; a.w *= b;
}

////////////////////////////////////////////////////////////////////////////////
// divide
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ Float3 operator/(Float3 a, Float3 b)
{
    return MAKE_FLOAT3(a.x / b.x, a.y / b.y, a.z / b.z);
}
inline __host__ __device__ void operator/=(Float3 &a, Float3 b)
{
    a.x /= b.x; a.y /= b.y; a.z /= b.z;
}
inline __host__ __device__ Float3 operator/(Float3 a, Float b)
{
    return MAKE_FLOAT3(a.x / b, a.y / b, a.z / b);
}
inline __host__ __device__ void operator/=(Float3 &a, Float b)
{
    a.x /= b; a.y /= b; a.z /= b;
}
inline __host__ __device__ Float3 operator/(Float b, Float3 a)
{
    return MAKE_FLOAT3(b / a.x, b / a.y, b / a.z);
}

inline __host__ __device__ Float4 operator/(Float4 a, Float4 b)
{
    return MAKE_FLOAT4(a.x / b.x, a.y / b.y, a.z / b.z,  a.w / b.w);
}
inline __host__ __device__ void operator/=(Float4 &a, Float4 b)
{
    a.x /= b.x; a.y /= b.y; a.z /= b.z; a.w /= b.w;
}
inline __host__ __device__ Float4 operator/(Float4 a, Float b)
{
    return MAKE_FLOAT4(a.x / b, a.y / b, a.z / b,  a.w / b);
}
inline __host__ __device__ void operator/=(Float4 &a, Float b)
{
    a.x /= b; a.y /= b; a.z /= b; a.w /= b;
}
inline __host__ __device__ Float4 operator/(Float b, Float4 a){
    return MAKE_FLOAT4(b / a.x, b / a.y, b / a.z, b / a.w);
}

////////////////////////////////////////////////////////////////////////////////
// min
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ Float3 fminf(Float3 a, Float3 b)
{
    return MAKE_FLOAT3(fminf(a.x,b.x), fminf(a.y,b.y), fminf(a.z,b.z));
}
inline  __host__ __device__ Float4 fminf(Float4 a, Float4 b)
{
    return MAKE_FLOAT4(fminf(a.x,b.x), fminf(a.y,b.y), fminf(a.z,b.z), fminf(a.w,b.w));
}

////////////////////////////////////////////////////////////////////////////////
// max
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ Float3 fmaxf(Float3 a, Float3 b)
{
    return MAKE_FLOAT3(fmaxf(a.x,b.x), fmaxf(a.y,b.y), fmaxf(a.z,b.z));
}
inline __host__ __device__ Float4 fmaxf(Float4 a, Float4 b)
{
    return MAKE_FLOAT4(fmaxf(a.x,b.x), fmaxf(a.y,b.y), fmaxf(a.z,b.z), fmaxf(a.w,b.w));
}

////////////////////////////////////////////////////////////////////////////////
// lerp
// - linear interpolation between a and b, based on value t in [0, 1] range
////////////////////////////////////////////////////////////////////////////////

inline __device__ __host__ Float lerp(Float a, Float b, Float t)
{
    return a + t*(b-a);
}
inline __device__ __host__ Float3 lerp(Float3 a, Float3 b, Float t)
{
    return a + t*(b-a);
}
inline __device__ __host__ Float4 lerp(Float4 a, Float4 b, Float t)
{
    return a + t*(b-a);
}

////////////////////////////////////////////////////////////////////////////////
// clamp
// - clamp the value v to be in the range [a, b]
////////////////////////////////////////////////////////////////////////////////

inline __device__ __host__ Float clamp(Float f, Float a, Float b)
{
    return fmaxf(a, fminf(f, b));
}

inline __device__ __host__ Float3 clamp(Float3 v, Float a, Float b)
{
    return MAKE_FLOAT3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}
inline __device__ __host__ Float3 clamp(Float3 v, Float3 a, Float3 b)
{
    return MAKE_FLOAT3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}
inline __device__ __host__ Float4 clamp(Float4 v, Float a, Float b)
{
    return MAKE_FLOAT4(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b), clamp(v.w, a, b));
}
inline __device__ __host__ Float4 clamp(Float4 v, Float4 a, Float4 b)
{
    return MAKE_FLOAT4(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z), clamp(v.w, a.w, b.w));
}

////////////////////////////////////////////////////////////////////////////////
// dot product
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ Float dot(Float3 a, Float3 b)
{ 
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline __host__ __device__ Float dot(Float4 a, Float4 b)
{ 
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

////////////////////////////////////////////////////////////////////////////////
// length
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ Float length(Float3 v)
{
    return sqrtf(dot(v, v));
}
inline __host__ __device__ Float length(Float4 v)
{
    return sqrtf(dot(v, v));
}

////////////////////////////////////////////////////////////////////////////////
// normalize
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ Float3 normalize(Float3 v)
{
    Float invLen = rsqrtf(dot(v, v));
    return v * invLen;
}
inline __host__ __device__ Float4 normalize(Float4 v)
{
    Float invLen = rsqrtf(dot(v, v));
    return v * invLen;
}

////////////////////////////////////////////////////////////////////////////////
// floor
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ Float3 floorf(Float3 v)
{
    return MAKE_FLOAT3(floorf(v.x), floorf(v.y), floorf(v.z));
}
inline __host__ __device__ Float4 floorf(Float4 v)
{
    return MAKE_FLOAT4(floorf(v.x), floorf(v.y), floorf(v.z), floorf(v.w));
}

////////////////////////////////////////////////////////////////////////////////
// frac - returns the fractional portion of a scalar or each vector component
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ Float fracf(Float v)
{
    return v - floorf(v);
}
inline __host__ __device__ Float3 fracf(Float3 v)
{
    return MAKE_FLOAT3(fracf(v.x), fracf(v.y), fracf(v.z));
}
inline __host__ __device__ Float4 fracf(Float4 v)
{
    return MAKE_FLOAT4(fracf(v.x), fracf(v.y), fracf(v.z), fracf(v.w));
}

////////////////////////////////////////////////////////////////////////////////
// fmod
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ Float3 fmodf(Float3 a, Float3 b)
{
    return MAKE_FLOAT3(fmodf(a.x, b.x), fmodf(a.y, b.y), fmodf(a.z, b.z));
}
inline __host__ __device__ Float4 fmodf(Float4 a, Float4 b)
{
    return MAKE_FLOAT4(fmodf(a.x, b.x), fmodf(a.y, b.y), fmodf(a.z, b.z), fmodf(a.w, b.w));
}

////////////////////////////////////////////////////////////////////////////////
// absolute value
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ Float3 fabs(Float3 v)
{
    return MAKE_FLOAT3(fabs(v.x), fabs(v.y), fabs(v.z));
}
inline __host__ __device__ Float4 fabs(Float4 v)
{
    return MAKE_FLOAT4(fabs(v.x), fabs(v.y), fabs(v.z), fabs(v.w));
}

////////////////////////////////////////////////////////////////////////////////
// reflect
// - returns reflection of incident ray I around surface normal N
// - N should be normalized, reflected vector's length is equal to length of I
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ Float3 reflect(Float3 i, Float3 n)
{
    return i - 2.0f * n * dot(n,i);
}

////////////////////////////////////////////////////////////////////////////////
// cross product
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ Float3 cross(Float3 a, Float3 b)
{ 
    return MAKE_FLOAT3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x); 
}

#endif
// vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
