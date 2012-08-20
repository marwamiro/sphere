#include <iostream>
#include <cutil_math.h>
#include "header.h"
#include "rt-kernel.h"
#include "colorbar.h"

unsigned int iDivUp (unsigned int a, unsigned int b) {
  return (a % b != 0) ? (a / b + 1) : (a / b);
}

__inline__ __host__ __device__ float3 f4_to_f3(float4 in)
{
  return make_float3(in.x, in.y, in.z);
}

__inline__ __host__ __device__ float4 f3_to_f4(float3 in)
{
  return make_float4(in.x, in.y, in.z, 0.0f);
}

// Kernel for initializing image data
__global__ void imageInit(unsigned char* _img, unsigned int pixels)
{
  // Compute pixel position from threadIdx/blockIdx
  unsigned int mempos = threadIdx.x + blockIdx.x * blockDim.x;
  if (mempos > pixels)
    return;

  _img[mempos*4]     = 255;	// Red channel
  _img[mempos*4 + 1] = 255;	// Green channel
  _img[mempos*4 + 2] = 255;	// Blue channel
}

// Calculate ray origins and directions
__global__ void rayInitPerspective(float4* _ray_origo, 
    				   float4* _ray_direction, 
				   float4 eye, 
                                   unsigned int width,
				   unsigned int height)
{
  // Compute pixel position from threadIdx/blockIdx
  unsigned int mempos = threadIdx.x + blockIdx.x * blockDim.x;
  if (mempos > width*height) 
    return;

  // Calculate 2D position from linear index
  unsigned int i = mempos % width;
  unsigned int j = (int)floor((float)mempos/width) % width;
 
  // Calculate pixel coordinates in image plane
  float p_u = const_imgplane.x + (const_imgplane.y - const_imgplane.x)
              * (i + 0.5f) / width;
  float p_v = const_imgplane.z + (const_imgplane.w - const_imgplane.z)
              * (j + 0.5f) / height;

  // Write ray origo and direction to global memory
  _ray_origo[mempos]     = const_eye;
  _ray_direction[mempos] = -const_d*const_w + p_u*const_u + p_v*const_v;
}

// Check wether the pixel's viewing ray intersects with the spheres,
// and shade the pixel correspondingly
__global__ void rayIntersectSpheres(float4* _ray_origo, 
                                    float4* _ray_direction,
                                    float4* _p, 
				    unsigned char* _img)
{
  // Compute pixel position from threadIdx/blockIdx
  unsigned int mempos = threadIdx.x + blockIdx.x * blockDim.x;
  if (mempos > const_pixels)
    return;

  // Read ray data from global memory
  float3 e = f4_to_f3(_ray_origo[mempos]);
  float3 d = f4_to_f3(_ray_direction[mempos]);
  //float  step = length(d);

  // Distance, in ray steps, between object and eye initialized with a large value
  float tdist = 1e10f;

  // Surface normal at closest sphere intersection
  float3 n;

  // Intersection point coordinates
  float3 p;

  // Iterate through all particles
  for (Inttype i=0; i<const_np; ++i) {

    // Read sphere coordinate and radius
    float3 c     = f4_to_f3(_p[i]);
    float  R     = _p[i].w;

    // Calculate the discriminant: d = B^2 - 4AC
    float Delta = (2.0f*dot(d,(e-c)))*(2.0f*dot(d,(e-c)))  // B^2
                  - 4.0f*dot(d,d)	// -4*A
	          * (dot((e-c),(e-c)) - R*R);  // C

    // If the determinant is positive, there are two solutions
    // One where the line enters the sphere, and one where it exits
    if (Delta > 0.0f) { 
      
      // Calculate roots, Shirley 2009 p. 77
      float t_minus = ((dot(-d,(e-c)) - sqrt( dot(d,(e-c))*dot(d,(e-c)) - dot(d,d)
	              * (dot((e-c),(e-c)) - R*R) ) ) / dot(d,d));

      // Check wether intersection is closer than previous values
      if (fabs(t_minus) < tdist) {
	p = e + t_minus*d;
	tdist = fabs(t_minus);
	n = normalize(2.0f * (p - c));   // Surface normal
      }

    } // End of solution branch

  } // End of particle loop

  // Write pixel color
  if (tdist < 1e10f) {

    // Lambertian shading parameters
    float dotprod = fmax(0.0f,dot(n, f4_to_f3(const_light)));
    float I_d = 40.0f;  // Light intensity
    float k_d = 5.0f;  // Diffuse coefficient
    
    // Ambient shading
    float k_a = 10.0f;
    float I_a = 5.0f; // 5.0 for black background

    // Write shading model values to pixel color channels
    _img[mempos*4]     = (unsigned char) ((k_d * I_d * dotprod 
                       + k_a * I_a)*0.48f);
    _img[mempos*4 + 1] = (unsigned char) ((k_d * I_d * dotprod
       		       + k_a * I_a)*0.41f);
    _img[mempos*4 + 2] = (unsigned char) ((k_d * I_d * dotprod
      		       + k_a * I_a)*0.27f);

  }
}

// Check wether the pixel's viewing ray intersects with the spheres,
// and shade the pixel correspondingly using a colormap
__global__ void rayIntersectSpheresColormap(float4* _ray_origo, 
                                            float4* _ray_direction,
					    float4* _p, 
					    float*  _fixvel,
					    float*  _linarr,
					    float max_val,
					    unsigned char* _img)
{
  // Compute pixel position from threadIdx/blockIdx
  unsigned int mempos = threadIdx.x + blockIdx.x * blockDim.x;
  if (mempos > const_pixels)
    return;

  // Read ray data from global memory
  float3 e = f4_to_f3(_ray_origo[mempos]);
  float3 d = f4_to_f3(_ray_direction[mempos]);

  // Distance, in ray steps, between object and eye initialized with a large value
  float tdist = 1e10f;

  // Surface normal at closest sphere intersection
  float3 n;

  // Intersection point coordinates
  float3 p;

  unsigned int hitidx;

  // Iterate through all particles
  for (Inttype i=0; i<const_np; ++i) {

    // Read sphere coordinate and radius
    float3 c     = f4_to_f3(_p[i]);
    float  R     = _p[i].w;

    // Calculate the discriminant: d = B^2 - 4AC
    float Delta = (2.0f*dot(d,(e-c)))*(2.0f*dot(d,(e-c)))  // B^2
                  - 4.0f*dot(d,d)	// -4*A
	          * (dot((e-c),(e-c)) - R*R);  // C

    // If the determinant is positive, there are two solutions
    // One where the line enters the sphere, and one where it exits
    if (Delta > 0.0f) {
      
      // Calculate roots, Shirley 2009 p. 77
      float t_minus = ((dot(-d,(e-c)) - sqrt( dot(d,(e-c))*dot(d,(e-c)) - dot(d,d)
	              * (dot((e-c),(e-c)) - R*R) ) ) / dot(d,d));

      // Check wether intersection is closer than previous values
      if (fabs(t_minus) < tdist) {
	p = e + t_minus*d;
	tdist = fabs(t_minus);
	n = normalize(2.0f * (p - c));   // Surface normal
	hitidx = i;
      }

    } // End of solution branch

  } // End of particle loop

  // Write pixel color
  if (tdist < 1e10) {

    // Fetch particle data used for color
    float ratio = _linarr[hitidx] / max_val;
    float fixvel   = _fixvel[hitidx];

    // Make sure the ratio doesn't exceed the 0.0-1.0 interval
    if (ratio < 0.01f)
      ratio = 0.01f;
    if (ratio > 0.99f)
      ratio = 0.99f;

    // Lambertian shading parameters
    float dotprod = fmax(0.0f,dot(n, f4_to_f3(const_light)));
    float I_d = 40.0f;  // Light intensity
    float k_d = 5.0f;  // Diffuse coefficient
    
    // Ambient shading
    float k_a = 10.0f;
    float I_a = 5.0f;

    float redv   = red(ratio);
    float greenv = green(ratio);
    float bluev  = blue(ratio);

    // Make particle dark grey if the horizontal velocity is fixed
    if (fixvel > 0.f) {
      redv = 0.5;
      greenv = 0.5;
      bluev = 0.5;
    }

    // Write shading model values to pixel color channels
    _img[mempos*4]     = (unsigned char) ((k_d * I_d * dotprod 
                       + k_a * I_a)*redv);
    _img[mempos*4 + 1] = (unsigned char) ((k_d * I_d * dotprod
       		       + k_a * I_a)*greenv);
    _img[mempos*4 + 2] = (unsigned char) ((k_d * I_d * dotprod
      		       + k_a * I_a)*bluev);
  }
}

extern "C"
__host__ void cameraInit(float4 eye, float4 lookat, float imgw, float hw_ratio,
    			 unsigned int pixels, Inttype np)
{
  // Image dimensions in world space (l, r, b, t)
  float4 imgplane = make_float4(-0.5f*imgw, 0.5f*imgw, -0.5f*imgw*hw_ratio, 0.5f*imgw*hw_ratio);

  // The view vector
  float4 view = eye - lookat;

  // Construct the camera view orthonormal base
  //float4 up = make_float4(0.0f, 1.0f, 0.0f, 0.0f);  // Pointing upward along +y
  float4 up = make_float4(0.0f, 0.0f, 1.0f, 0.0f);  // Pointing upward along +z
  float4 w = -view/length(view);		   // w: Pointing backwards
  float4 u = make_float4(cross(make_float3(up.x, up.y, up.z),
			       make_float3(w.x, w.y, w.z)), 0.0f)
            / length(cross(make_float3(up.x, up.y, up.z), make_float3(w.x, w.y, w.z)));
  float4 v = make_float4(cross(make_float3(w.x, w.y, w.z), make_float3(u.x, u.y, u.z)), 0.0f);

  // Focal length 20% of eye vector length
  float d = length(view)*0.8f;

  // Light direction (points towards light source)
  float4 light = normalize(-1.0f*eye*make_float4(1.0f, 0.2f, 0.6f, 0.0f));

  std::cout << "  Transfering camera values to constant memory\n";

  cudaMemcpyToSymbol("const_u", &u, sizeof(u));
  cudaMemcpyToSymbol("const_v", &v, sizeof(v));
  cudaMemcpyToSymbol("const_w", &w, sizeof(w));
  cudaMemcpyToSymbol("const_eye", &eye, sizeof(eye));
  cudaMemcpyToSymbol("const_imgplane", &imgplane, sizeof(imgplane));
  cudaMemcpyToSymbol("const_d", &d, sizeof(d));
  cudaMemcpyToSymbol("const_light", &light, sizeof(light));
  cudaMemcpyToSymbol("const_pixels", &pixels, sizeof(pixels));
  cudaMemcpyToSymbol("const_np", &np, sizeof(np));
}

// Check for CUDA errors
extern "C"
__host__ void checkForCudaErrors(const char* checkpoint_description)
{
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cout << "\nCuda error detected, checkpoint: " << checkpoint_description
              << "\nError string: " << cudaGetErrorString(err) << "\n";
    exit(EXIT_FAILURE);
  }
}


// Wrapper for the rt kernel
extern "C"
__host__ int rt(float4* p, Inttype np,
                rgb* img, unsigned int width, unsigned int height,
		f3 origo, f3 L, f3 eye, f3 lookat, float imgw,
		int visualize, float max_val,
		float* fixvel, float* pres, float* es_dot, float* es, float* vel)
{
  using std::cout;

  cout << "Initializing CUDA:\n";

  // Initialize GPU timestamp recorders
  float t1, t2;
  cudaEvent_t t1_go, t2_go, t1_stop, t2_stop;
  cudaEventCreate(&t1_go);
  cudaEventCreate(&t2_go);
  cudaEventCreate(&t2_stop);
  cudaEventCreate(&t1_stop);

  // Start timer 1
  cudaEventRecord(t1_go, 0);

  // Allocate memory
  cout << "  Allocating device memory\n";
  static float4 *_p;			// Particle positions (x,y,z) and radius (w)
  static float  *_fixvel;               // Indicates whether a particle has a fixed horizontal velocity
  static float  *_linarr;	        // Array for linear values to color the particles after
  static unsigned char *_img; 		// RGBw values in image
  static float4 *_ray_origo;		// Ray origo (x,y,z)
  static float4 *_ray_direction; 	// Ray direction (x,y,z)
  cudaMalloc((void**)&_p, np*sizeof(float4));
  cudaMalloc((void**)&_fixvel, np*sizeof(float));
  cudaMalloc((void**)&_linarr, np*sizeof(float)); // 0 size if visualize = 0;
  cudaMalloc((void**)&_img, width*height*4*sizeof(unsigned char));
  cudaMalloc((void**)&_ray_origo, width*height*sizeof(float4));
  cudaMalloc((void**)&_ray_direction, width*height*sizeof(float4));

  // Transfer particle data
  cout << "  Transfering particle data: host -> device\n";
  cudaMemcpy(_p, p, np*sizeof(float4), cudaMemcpyHostToDevice);
  cudaMemcpy(_fixvel, fixvel, np*sizeof(float), cudaMemcpyHostToDevice);
  if (visualize == 1 || visualize == 4)
    cudaMemcpy(_linarr, pres, np*sizeof(float), cudaMemcpyHostToDevice);
  if (visualize == 2)
    cudaMemcpy(_linarr, es_dot, np*sizeof(float), cudaMemcpyHostToDevice);
  if (visualize == 3)
    cudaMemcpy(_linarr, es, np*sizeof(float), cudaMemcpyHostToDevice);
  if (visualize == 4)
    cudaMemcpy(_linarr, vel, np*sizeof(float), cudaMemcpyHostToDevice);

  // Check for errors after memory allocation
  checkForCudaErrors("CUDA error after memory allocation"); 

  // Arrange thread/block structure
  unsigned int pixels = width*height;
  float hw_ratio = (float)height/(float)width;
  const unsigned int threadsPerBlock = 256;
  const unsigned int blocksPerGrid = iDivUp(pixels, threadsPerBlock);

  // Start timer 2
  cudaEventRecord(t2_go, 0);
  
  // Initialize image to background color
  imageInit<<< blocksPerGrid, threadsPerBlock >>>(_img, pixels);

  // Initialize camera
  cameraInit(make_float4(eye.x, eye.y, eye.z, 0.0f), 
      	     make_float4(lookat.x, lookat.y, lookat.z, 0.0f),
	     imgw, hw_ratio, pixels, np);
  checkForCudaErrors("CUDA error after cameraInit");

  // Construct rays for perspective projection
  rayInitPerspective<<< blocksPerGrid, threadsPerBlock >>>(
      _ray_origo, _ray_direction, 
      make_float4(eye.x, eye.y, eye.z, 0.0f), 
      width, height);

  cudaThreadSynchronize();
  
  // Find closest intersection between rays and spheres
  if (visualize == 1) { // Visualize pressure
    cout << "  Pressure color map range: [0, " << max_val << "] Pa\n"; 
    rayIntersectSpheresColormap<<< blocksPerGrid, threadsPerBlock >>>(
        _ray_origo, _ray_direction,
        _p, _fixvel, _linarr, max_val, _img);
  } else if (visualize == 2) { // es_dot visualization
    cout << "  Shear heat production rate color map range: [0, " << max_val << "] W\n";
    rayIntersectSpheresColormap<<< blocksPerGrid, threadsPerBlock >>>(
	_ray_origo, _ray_direction,
	_p, _fixvel, _linarr, max_val, _img);
  } else if (visualize == 3) { // es visualization
    cout << "  Total shear heat color map range: [0, " << max_val << "] J\n";
    rayIntersectSpheresColormap<<< blocksPerGrid, threadsPerBlock >>>(
	_ray_origo, _ray_direction,
	_p, _fixvel, _linarr, max_val, _img);
  } else if (visualize == 4) { // velocity visualization
    cout << "  Velocity color map range: [0, " << max_val << "] m/s\n";
    rayIntersectSpheresColormap<<< blocksPerGrid, threadsPerBlock >>>(
	_ray_origo, _ray_direction,
	_p, _fixvel, _linarr, max_val, _img);
  } else { // Normal visualization
    rayIntersectSpheres<<< blocksPerGrid, threadsPerBlock >>>(
	_ray_origo, _ray_direction,
	_p, _img);
  }

  // Make sure all threads are done before continuing CPU control sequence
  cudaThreadSynchronize();
  
  // Check for errors
  checkForCudaErrors("CUDA error after kernel execution");

  // Stop timer 2
  cudaEventRecord(t2_stop, 0);
  cudaEventSynchronize(t2_stop);

  // Transfer image data from device to host
  cout << "  Transfering image data: device -> host\n";
  cudaMemcpy(img, _img, width*height*4*sizeof(unsigned char), cudaMemcpyDeviceToHost);

  // Free dynamically allocated device memory
  cudaFree(_p);
  cudaFree(_fixvel);
  cudaFree(_linarr);
  cudaFree(_img);
  cudaFree(_ray_origo);
  cudaFree(_ray_direction);

  // Stop timer 1
  cudaEventRecord(t1_stop, 0);
  cudaEventSynchronize(t1_stop);
  
  // Calculate time spent in t1 and t2
  cudaEventElapsedTime(&t1, t1_go, t1_stop);
  cudaEventElapsedTime(&t2, t2_go, t2_stop);

  // Report time spent
  cout << "  Time spent on entire GPU routine: "
       << t1 << " ms\n";
  cout << "  - Kernels: " << t2 << " ms\n"
       << "  - Memory alloc. and transfer: " << t1-t2 << " ms\n";

  // Return successfully
  return 0;
}
