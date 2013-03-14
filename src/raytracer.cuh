#ifndef RAYTRACER_CUH_
#define RAYTRACER_CUH_

#include "colorbar.h"

//#include "cuPrintf.cu"

// Template for discarding the last term in four-component vector structs
__device__ __inline__ float3 f4_to_f3(float4 in) {
    return make_float3(in.x, in.y, in.z);
}

// Kernel for initializing image data
__global__ void imageInit(unsigned char* dev_img, unsigned int pixels)
{
    // Compute pixel position from threadIdx/blockIdx
    unsigned int mempos = threadIdx.x + blockIdx.x * blockDim.x;
    if (mempos > pixels)
        return;

    dev_img[mempos*4]     = 255;	// Red channel
    dev_img[mempos*4 + 1] = 255;	// Green channel
    dev_img[mempos*4 + 2] = 255;	// Blue channel
}

// Calculate ray origins and directions
__global__ void rayInitPerspective(float4* dev_ray_origo, 
        float4* dev_ray_direction, 
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
    float p_u = devC_imgplane.x + (devC_imgplane.y - devC_imgplane.x)
        * (i + 0.5f) / width;
    float p_v = devC_imgplane.z + (devC_imgplane.w - devC_imgplane.z)
        * (j + 0.5f) / height;

    // Write ray origo and direction to global memory
    dev_ray_origo[mempos]     = make_float4(devC_eye, 0.0f);
    dev_ray_direction[mempos] = make_float4(-devC_d*devC_w + p_u*devC_u + p_v*devC_v, 0.0f);
}

// Check wether the pixel's viewing ray intersects with the spheres,
// and shade the pixel correspondingly
__global__ void rayIntersectSpheres(float4* dev_ray_origo, 
        float4* dev_ray_direction,
        Float4* dev_x, 
        unsigned char* dev_img)
{
    // Compute pixel position from threadIdx/blockIdx
    unsigned int mempos = threadIdx.x + blockIdx.x * blockDim.x;
    if (mempos > devC_pixels)
        return;

    // Read ray data from global memory
    float3 e = f4_to_f3(dev_ray_origo[mempos]);
    float3 d = f4_to_f3(dev_ray_direction[mempos]);
    //float  step = length(d);

    // Distance, in ray steps, between object and eye initialized with a large value
    float tdist = 1e10f;

    // Surface normal at closest sphere intersection
    float3 n;

    // Intersection point coordinates
    float3 p;

    //cuPrintf("mepos %d\n", mempos);

    // Iterate through all particles
    for (unsigned int i=0; i<devC_np; ++i) {

        // Read sphere coordinate and radius
        Float4 x = dev_x[i];
        float3 c = make_float3(x.x, x.y, x.z);
        float  R = x.w;

        //cuPrintf("particle %d at: %f, %f, %f, radius: %f\n", i, c.x, c.y, c.z, R);

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
        float dotprod = fmax(0.0f,dot(n, devC_light));
        float I_d = 40.0f;  // Light intensity
        float k_d = 5.0f;  // Diffuse coefficient

        // Ambient shading
        float k_a = 10.0f;
        float I_a = 5.0f; // 5.0 for black background

        // Write shading model values to pixel color channels
        dev_img[mempos*4]     = 
            (unsigned char) ((k_d * I_d * dotprod + k_a * I_a)*0.48f);
        dev_img[mempos*4 + 1] = 
            (unsigned char) ((k_d * I_d * dotprod + k_a * I_a)*0.41f);
        dev_img[mempos*4 + 2] = 
            (unsigned char) ((k_d * I_d * dotprod + k_a * I_a)*0.27f);

    }
}

// Check wether the pixel's viewing ray intersects with the spheres,
// and shade the pixel correspondingly using a colormap
__global__ void rayIntersectSpheresColormap(float4* dev_ray_origo, 
        float4* dev_ray_direction,
        Float4* dev_x, 
        Float4* dev_vel,
        Float*  dev_linarr,
        float max_val,
        float lower_cutoff,
        unsigned char* dev_img)
{
    // Compute pixel position from threadIdx/blockIdx
    unsigned int mempos = threadIdx.x + blockIdx.x * blockDim.x;
    if (mempos > devC_pixels)
        return;

    // Read ray data from global memory
    float3 e = f4_to_f3(dev_ray_origo[mempos]);
    float3 d = f4_to_f3(dev_ray_direction[mempos]);

    // Distance, in ray steps, between object and eye initialized with a large value
    float tdist = 1e10f;

    // Surface normal at closest sphere intersection
    float3 n;

    // Intersection point coordinates
    float3 p;

    unsigned int hitidx;

    // Iterate through all particles
    for (unsigned int i=0; i<devC_np; ++i) {

        __syncthreads();

        // Read sphere coordinate and radius
        Float4 x = dev_x[i];
        float3 c = make_float3(x.x, x.y, x.z);
        float  R = x.w;

        // Calculate the discriminant: d = B^2 - 4AC
        float Delta = (2.0f*dot(d,(e-c)))*(2.0f*dot(d,(e-c)))  // B^2
            - 4.0f*dot(d,d)	// -4*A
            * (dot((e-c),(e-c)) - R*R);  // C



        // If the determinant is positive, there are two solutions
        // One where the line enters the sphere, and one where it exits
        if (lower_cutoff > 0.0) {

            // value on colorbar
            float val = dev_linarr[i];

            // particle is fixed if value > 0
            float fixvel = dev_vel[i].w;

            // only render particles which are above the lower cutoff
            // and which are not fixed at a velocity
            if (Delta > 0.0f && val > lower_cutoff && fixvel == 0.f) {

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

        } else {

            // render particle if it intersects with the ray
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
        }

    } // End of particle loop

    // Write pixel color
    if (tdist < 1e10) {

        __syncthreads();
        // Fetch particle data used for color
        float fixvel = dev_vel[hitidx].w;
        float ratio = dev_linarr[hitidx] / max_val;

        // Make sure the ratio doesn't exceed the 0.0-1.0 interval
        if (ratio < 0.01f)
            ratio = 0.01f;
        if (ratio > 0.99f)
            ratio = 0.99f;

        // Lambertian shading parameters
        float dotprod = fmax(0.0f,dot(n, devC_light));
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
        dev_img[mempos*4]     = (unsigned char) ((k_d * I_d * dotprod 
                    + k_a * I_a)*redv);
        dev_img[mempos*4 + 1] = (unsigned char) ((k_d * I_d * dotprod
                    + k_a * I_a)*greenv);
        dev_img[mempos*4 + 2] = (unsigned char) ((k_d * I_d * dotprod
                    + k_a * I_a)*bluev);
    }
}


__host__ void DEM::cameraInit(
        const float3 eye,
        const float3 lookat, 
        const float imgw,
        const float focalLength)
{
    float hw_ratio = height/width;

    // Image dimensions in world space (l, r, b, t)
    float4 imgplane = make_float4(-0.5f*imgw, 0.5f*imgw, -0.5f*imgw*hw_ratio, 0.5f*imgw*hw_ratio);

    // The view vector
    float3 view = eye - lookat;

    // Construct the camera view orthonormal base
    float3 up = make_float3(0.0f, 0.0f, 1.0f);  // Pointing upward along +z
    float3 w = -view/length(view);		   // w: Pointing backwards
    float3 u = cross(up, w) / length(cross(up, w));
    float3 v = cross(w, u);

    unsigned int pixels = width*height;

    // Focal length 20% of eye vector length
    float d = length(view)*0.8f;

    // Light direction (points towards light source)
    float3 light = normalize(-1.0f*eye*make_float3(1.0f, 0.2f, 0.6f));

    if (verbose == 1)
        std::cout << "  Transfering camera values to constant memory: ";

    /* Reference by string removed in cuda 5.0
    cudaMemcpyToSymbol("devC_u", &u, sizeof(u));
    cudaMemcpyToSymbol("devC_v", &v, sizeof(v));
    cudaMemcpyToSymbol("devC_w", &w, sizeof(w));
    cudaMemcpyToSymbol("devC_eye", &eye, sizeof(eye));
    cudaMemcpyToSymbol("devC_imgplane", &imgplane, sizeof(imgplane));
    cudaMemcpyToSymbol("devC_d", &d, sizeof(d));
    cudaMemcpyToSymbol("devC_light", &light, sizeof(light));
    cudaMemcpyToSymbol("devC_pixels", &pixels, sizeof(pixels));*/
    cudaMemcpyToSymbol(devC_u, &u, sizeof(u));
    cudaMemcpyToSymbol(devC_v, &v, sizeof(v));
    cudaMemcpyToSymbol(devC_w, &w, sizeof(w));
    cudaMemcpyToSymbol(devC_eye, &eye, sizeof(eye));
    cudaMemcpyToSymbol(devC_imgplane, &imgplane, sizeof(imgplane));
    cudaMemcpyToSymbol(devC_d, &d, sizeof(d));
    cudaMemcpyToSymbol(devC_light, &light, sizeof(light));
    cudaMemcpyToSymbol(devC_pixels, &pixels, sizeof(pixels));

    if (verbose == 1)
        std::cout << "Done" << std::endl;
    checkForCudaErrors("During cameraInit");
}

// Allocate global device memory
__host__ void DEM::rt_allocateGlobalDeviceMemory(void)
{
    if (verbose == 1)
        std::cout << "  Allocating device memory: ";
    cudaMalloc((void**)&dev_img, width*height*4*sizeof(unsigned char));
    cudaMalloc((void**)&dev_ray_origo, width*height*sizeof(float4));
    cudaMalloc((void**)&dev_ray_direction, width*height*sizeof(float4));
    if (verbose == 1)
        std::cout << "Done" << std::endl;
    checkForCudaErrors("During rt_allocateGlobalDeviceMemory()");
}


// Free dynamically allocated device memory
__host__ void DEM::rt_freeGlobalDeviceMemory(void)
{
    if (verbose == 1)
        std::cout << "  Freeing device memory: ";
    cudaFree(dev_img);
    cudaFree(dev_ray_origo);
    cudaFree(dev_ray_direction);
    if (verbose == 1)
        std::cout << "Done" << std::endl;
    checkForCudaErrors("During rt_freeGlobalDeviceMemory()");
}

// Transfer image data from device to host
__host__ void DEM::rt_transferFromGlobalDeviceMemory(void)
{
    if (verbose == 1)
        std::cout << "  Transfering image data: device -> host: ";
    cudaMemcpy(img, dev_img, width*height*4*sizeof(unsigned char), cudaMemcpyDeviceToHost);
    if (verbose == 1)
        std::cout << "Done" << std::endl;
    checkForCudaErrors("During rt_transferFromGlobalDeviceMemory()");
}

// Wrapper for the rt kernel
__host__ void DEM::render(
        const int method,
        const float maxval,
        const float lower_cutoff,
        const float focalLength,
        const unsigned int img_width,
        const unsigned int img_height)
    /*float4* p, unsigned int np,
      rgba* img, unsigned int width, unsigned int height,
      f3 origo, f3 L, f3 eye, f3 lookat, float imgw,
      int visualize, float max_val,
      float* fixvel,
      float* xsum,
      float* pres,
      float* es_dot,
      float* es,
      float* vel)*/
{
    // Namespace directives
    using std::cout;
    using std::cerr;
    using std::endl;

    //cudaPrintfInit();

    // Save image dimensions in class object
    width = img_width;
    height = img_height;

    // Allocate memory for output image
    img = new rgba[height*width];
    rt_allocateGlobalDeviceMemory();

    // Arrange thread/block structure
    unsigned int pixels = width*height;
    const unsigned int threadsPerBlock = 256;
    const unsigned int blocksPerGrid = iDivUp(pixels, threadsPerBlock);

    // Initialize image to background color
    imageInit<<< blocksPerGrid, threadsPerBlock >>>(dev_img, pixels);

    // Initialize camera values and transfer to constant memory
    float imgw = grid.L[0]*1.35f; // Screen width in world coordinates
    Float3 maxpos = maxPos();
    // Look at the centre of the mean positions
    float3 lookat = make_float3(maxpos.x, maxpos.y, maxpos.z) / 2.0f; 
    float3 eye = make_float3(
            grid.L[0] * 2.3f,
            grid.L[1] * -5.0f,
            grid.L[2] * 1.3f);
    cameraInit(eye, lookat, imgw, focalLength);

    // Construct rays for perspective projection
    rayInitPerspective<<< blocksPerGrid, threadsPerBlock >>>(
            dev_ray_origo, dev_ray_direction, 
            make_float4(eye.x, eye.y, eye.z, 0.0f), width, height);
    cudaThreadSynchronize();

    Float* linarr;     // Linear array to use for color visualization
    Float* dev_linarr; // Device linear array to use for color visualization
    checkForCudaErrors("Error during cudaMalloc of linear array");
    std::string desc;  // Description of parameter visualized
    std::string unit;  // Unit of parameter values visualized
    unsigned int i;
    int transfer = 0;  // If changed to 1, linarr will be copied to dev_linarr

    // Visualize spheres without color scale overlay
    if (method == 0) {
        rayIntersectSpheres<<< blocksPerGrid, threadsPerBlock >>>(
                dev_ray_origo, dev_ray_direction,
                dev_x, dev_img);
    } else {

        if (method == 1) { // Visualize pressure
            dev_linarr = dev_p;
            desc = "Pressure";
            unit = "Pa";

        } else if (method == 2) { // Visualize velocity
            // Find the magnitude of all linear velocities
            linarr = new Float[np];
#pragma omp parallel for if(np>100)
            for (i = 0; i<np; ++i) {
                linarr[i] = sqrt(k.vel[i].x*k.vel[i].x 
                        + k.vel[i].y*k.vel[i].y 
                        + k.vel[i].z*k.vel[i].z);
            }
            transfer = 1;
            desc = "Linear velocity";
            unit = "m/s";

        } else if (method == 3) { // Visualize angular velocity
            // Find the magnitude of all rotational velocities
            linarr = new Float[np];
#pragma omp parallel for if(np>100)
            for (i = 0; i<np; ++i) {
                linarr[i] = sqrt(k.angvel[i].x*k.angvel[i].x
                        + k.angvel[i].y*k.angvel[i].y 
                        + k.angvel[i].z*k.angvel[i].z);
            }
            transfer = 1;
            desc = "Angular velocity";
            unit = "rad/s";

        } else if (method == 4) { // Visualize xdisp
            // Convert xysum to xsum
            linarr = new Float[np];
#pragma omp parallel for if(np>100)
            for (i = 0; i<np; ++i) {
                linarr[i] = k.xysum[i].x;
            }
            transfer = 1;
            desc = "X-axis displacement";
            unit = "m";

        } else if (method == 5) { // Visualize total rotation
            // Find the magnitude of all rotations
            linarr = new Float[np];
#pragma omp parallel for if(np>100)
            for (i = 0; i<np; ++i) {
                linarr[i] = sqrt(k.angpos[i].x*k.angpos[i].x
                        + k.angpos[i].y*k.angpos[i].y 
                        + k.angpos[i].z*k.angpos[i].z);
            }
            transfer = 1;
            desc = "Angular positions";
            unit = "rad";
        }


        // Report color visualization method and color map range
        if (verbose == 1) {
            cout << "  " << desc << " color map range: [0, " 
                << maxval << "] " << unit << endl;
        }

        // Copy linarr to dev_linarr if required
        if (transfer == 1) {
            cudaMalloc((void**)&dev_linarr, np*sizeof(Float));
            checkForCudaErrors("Error during cudaMalloc of linear array");
            cudaMemcpy(dev_linarr, linarr, np*sizeof(Float), cudaMemcpyHostToDevice);
            checkForCudaErrors("Error during cudaMemcpy of linear array");
        }

        // Start raytracing kernel
        rayIntersectSpheresColormap<<< blocksPerGrid, threadsPerBlock >>>(
                dev_ray_origo, dev_ray_direction,
                dev_x, dev_vel,
                dev_linarr, maxval, lower_cutoff,
                dev_img);

    }

    // Make sure all threads are done before continuing CPU control sequence
    cudaThreadSynchronize();

    // Check for errors
    checkForCudaErrors("CUDA error after kernel execution");

    // Copy image data from global device memory to host memory
    rt_transferFromGlobalDeviceMemory();

    // Free dynamically allocated global device memory
    rt_freeGlobalDeviceMemory();
    checkForCudaErrors("after rt_freeGlobalDeviceMemory");
    if (transfer == 1) {
        delete[] linarr;
        cudaFree(dev_linarr);
        checkForCudaErrors("When calling cudaFree(dev_linarr)");
    }

    //cudaPrintfDisplay(stdout, true);

    // Write image to PPM file
    writePPM(("img_out/" + sid + ".ppm").c_str());

}

#endif
// vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
