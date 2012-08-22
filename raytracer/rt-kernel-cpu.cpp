#include <iostream>
#include <cstdio>
#include <cmath>
#include <time.h>
#include <cuda.h>
#include <cutil_math.h>
#include <string.h>
#include <stdlib.h>
#include "header.h"
#include "rt-kernel-cpu.h"

// Constants
float3 constc_u;
float3 constc_v;
float3 constc_w;
float3 constc_eye;
float4 constc_imgplane;
float  constc_d;
float3 constc_light;

__inline__ float3 f4_to_f3(float4 in)
{
  return make_float3(in.x, in.y, in.z);
}

__inline__ float4 f3_to_f4(float3 in)
{
  return make_float4(in.x, in.y, in.z, 0.0f);
}

__inline__ float lengthf3(float3 in)
{
  return sqrt(in.x*in.x + in.y*in.y + in.z*in.z);
}

// Kernel for initializing image data
void imageInit_cpu(unsigned char* _img, unsigned int pixels)
{
  for (unsigned int mempos=0; mempos<pixels; ++mempos) {
    _img[mempos*4]     = 255;	// Red channel
    _img[mempos*4 + 1] = 255;	// Green channel
    _img[mempos*4 + 2] = 255;	// Blue channel
  }
}

// Calculate ray origins and directions
void rayInitPerspective_cpu(float3* _ray_origo, 
    			float3* _ray_direction, 
			float3 eye, 
                        unsigned int width,
			unsigned int height)
{
  int i,j;
  unsigned int mempos;
  float p_u, p_v;
  #pragma omp parallel for private(mempos,j,p_u,p_v)
  for (i=0; i<(int)width; ++i) {
    for (j=0; j<(int)height; ++j) {

      mempos = i + j*width;

      // Calculate pixel coordinates in image plane
      p_u = constc_imgplane.x + (constc_imgplane.y - constc_imgplane.x)
	* (i + 0.5f) / width;
      p_v = constc_imgplane.z + (constc_imgplane.w - constc_imgplane.z)
	* (j + 0.5f) / height;

      // Write ray origo and direction to global memory
      _ray_origo[mempos]     = constc_eye;
      _ray_direction[mempos] = -constc_d*constc_w + p_u*constc_u + p_v*constc_v;
    }
  }
}

// Check wether the pixel's viewing ray intersects with the spheres,
// and shade the pixel correspondingly
void rayIntersectSpheres_cpu(float3* _ray_origo, 
                         float3* _ray_direction,
                         float4* _p, 
			 float* _nuance,
			 unsigned char* _img, 
			 unsigned int pixels,
			 unsigned int np)
{
  long int mempos;
  float3 e, d, n, p, c;
  float tdist, R, Delta, t_minus, dotprod, I_d, k_d, k_a, I_a, nuance;
  Inttype i, ifinal;
  #pragma omp parallel for private(e,d,n,p,c,tdist,R,Delta,t_minus,dotprod,I_d,k_d,k_a,I_a,nuance,i,ifinal)
  for (mempos=0; mempos<pixels; ++mempos) {
    
    // Read ray data from global memory
    e = _ray_origo[mempos];
    d = _ray_direction[mempos];

    // Distance, in ray steps, between object and eye initialized with a large value
    tdist = 1e10f;

    // Iterate through all particles
    for (i=0; i<np; ++i) {

      // Read sphere coordinate and radius
      c = f4_to_f3(_p[i]);
      R = _p[i].w;

      // Calculate the discriminant: d = B^2 - 4AC
      Delta = (2.0f*dot(d,(e-c)))*(2.0f*dot(d,(e-c)))  // B^2
	- 4.0f*dot(d,d)	// -4*A
	* (dot((e-c),(e-c)) - R*R);  // C

      // If the determinant is positive, there are two solutions
      // One where the line enters the sphere, and one where it exits
      if (Delta > 0.0f) { 

	// Calculate roots, Shirley 2009 p. 77
	t_minus = ((dot(-d,(e-c)) - sqrt( dot(d,(e-c))*dot(d,(e-c)) - dot(d,d)
		* (dot((e-c),(e-c)) - R*R) ) ) / dot(d,d));

	// Check wether intersection is closer than previous values
	if (fabs(t_minus) < tdist) {
	  p = e + t_minus*d;
	  tdist = fabs(t_minus);
	  n = normalize(2.0f * (p - c));   // Surface normal
	  ifinal = i;
	}

      } // End of solution branch

    } // End of particle loop

    // Write pixel color
    if (tdist < 1e10) {

      // Lambertian shading parameters
      //float dotprod = fabs(dot(n, constc_light));
      dotprod = fmax(0.0f,dot(n, constc_light));
      I_d = 70.0f;  // Light intensity
      k_d = 5.0f;  // Diffuse coefficient

      // Ambient shading
      k_a = 30.0f;
      I_a = 5.0f;

      // Read color nuance of grain
      nuance = _nuance[ifinal];

      // Write shading model values to pixel color channels
      _img[mempos*4]     = (unsigned char) ((k_d * I_d * dotprod 
	    + k_a * I_a)*0.48f*nuance);
      _img[mempos*4 + 1] = (unsigned char) ((k_d * I_d * dotprod
	    + k_a * I_a)*0.41f*nuance);
      _img[mempos*4 + 2] = (unsigned char) ((k_d * I_d * dotprod
	    + k_a * I_a)*0.27f*nuance);
    }
  }
}


void cameraInit_cpu(float3 eye, float3 lookat, float imgw, float hw_ratio)
{
  // Image dimensions in world space (l, r, b, t)
  float4 imgplane = make_float4(-0.5f*imgw, 0.5f*imgw, -0.5f*imgw*hw_ratio, 0.5f*imgw*hw_ratio);

  // The view vector
  float3 view = eye - lookat;

  // Construct the camera view orthonormal base
  //float3 up = make_float3(0.0f, 1.0f, 0.0f);  // Pointing upward along +y
  float3 up = make_float3(0.0f, 0.0f, 1.0f);  // Pointing upward along +z
  float3 w = -view/length(view);		   // w: Pointing backwards
  float3 u = cross(up, w) / length(cross(up, w));
  float3 v = cross(w, u);

  // Focal length 20% of eye vector length
  float d = lengthf3(view)*0.8f;

  // Light direction (points towards light source)
  float3 light = normalize(-1.0f*eye*make_float3(1.0f, 0.2f, 0.6f));

  std::cout << "  Transfering camera values to constant memory\n";

  constc_u = u;
  constc_v = v;
  constc_w = w;
  constc_eye = eye;
  constc_imgplane = imgplane;
  constc_d = d;
  constc_light = light;

  std::cout << "Rendering image...";
}


// Wrapper for the rt algorithm
int rt_cpu(float4* p, unsigned int np,
       rgb* img, unsigned int width, unsigned int height,
       f3 origo, f3 L, f3 eye, f3 lookat, float imgw) {

  using std::cout;

  cout << "Initializing CPU raytracer:\n";

  // Initialize GPU timestamp recorders
  float t1_go, t2_go, t1_stop, t2_stop;

  // Start timer 1
  t1_go = clock();

  // Generate random nuance values for all grains
  static float* _nuance; // Values between 0.5 and 1.0
  _nuance = new float[np];
  srand(0); // Initialize seed at same value at every run
  for (unsigned int i=0; i<np; ++i)
    _nuance[i] = (float)rand()/RAND_MAX * 0.5f + 0.5f;

  // Allocate memory
  cout << "  Allocating memory\n";
  static unsigned char *_img; 		// RGBw values in image
  static float3* _ray_origo;		// Ray origo (x,y,z)
  static float3* _ray_direction;	// Ray direction (x,y,z)
  _img 		 = new unsigned char[width*height*4];
  _ray_origo 	 = new float3[width*height];
  _ray_direction = new float3[width*height];

  // Arrange thread/block structure
  unsigned int pixels = width*height;
  float hw_ratio = (float)height/(float)width;

  // Start timer 2
  t2_go = clock();
  
  // Initialize image to background color
  imageInit_cpu(_img, pixels);

  // Initialize camera
  cameraInit_cpu(make_float3(eye.x, eye.y, eye.z), 
      	         make_float3(lookat.x, lookat.y, lookat.z),
	         imgw, hw_ratio);

  // Construct rays for perspective projection
  rayInitPerspective_cpu(
      _ray_origo, _ray_direction, 
      make_float3(eye.x, eye.y, eye.z), 
      width, height);
  
  // Find closest intersection between rays and spheres
  rayIntersectSpheres_cpu(
      _ray_origo, _ray_direction,
      p, _nuance, _img, pixels, np);

  // Stop timer 2
  t2_stop = clock();

  memcpy(img, _img, sizeof(unsigned char)*pixels*4);

  // Free dynamically allocated device memory
  delete [] _nuance;
  delete [] _img;
  delete [] _ray_origo;
  delete [] _ray_direction;

  // Stop timer 1
  t1_stop = clock();
  
  // Report time spent 
  cout << " done.\n"
       << "  Time spent on entire CPU raytracing routine: "
       << (t1_stop-t1_go)/CLOCKS_PER_SEC*1000.0 << " ms\n";
  cout << "  - Functions: " << (t2_stop-t2_go)/CLOCKS_PER_SEC*1000.0 << " ms\n";

  // Return successfully
  return 0;
}
