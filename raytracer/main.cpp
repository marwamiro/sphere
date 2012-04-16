#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector_functions.h>
#include "header.h"
#include "rt_kernel.h"
#include "rt_kernel_cpu.h"

int main(const int argc, const char* argv[])
{
  using std::cout;

  if (argc < 6 || argc > 8 ){
    cout << "Usage: " << argv[0] << " <GPU or CPU> <sphere-binary.bin> <width> <height> <output-image.ppm>\n"
         << "or\n"
         << "Usage: " << argv[0] << " GPU <sphere-binary.bin | colorbar> <width> <height> <output-image.ppm>"
	 << " <pressure | pressure50 | es | es_dot> <max value in color range>\n";
    return 1;
  }

  cout << "\n----------------------------\n"
       << "This is the SPHERE raytracer\n"
       << "----------------------------\n";

  // Allocate memory for image
  unsigned int width = atoi(argv[3]);
  unsigned int height = atoi(argv[4]);
  if (width < 1 || height < 1) {
    cout << "Image dimensions invalid.\n";
    return 1;
  }
  rgb* img;
  img = new rgb [height*width];

  // Render colorbar image
  if (strcmp(argv[2],"colorbar") == 0) {

    for (unsigned int y=0; y<height; y++) {
      for (unsigned int x=0; x<width; x++) {

	// Colormap value is relative to width position
	float ratio = (float) (x+1)/width;

	// Determine Blue-White-Red color components
	float red   = fmin(1.0f, 0.209f*ratio*ratio*ratio - 2.49f*ratio*ratio + 3.0f*ratio + 0.0109f);
	float green = fmin(1.0f, -2.44f*ratio*ratio + 2.15f*ratio + 0.369f);
	float blue  = fmin(1.0f, -2.21f*ratio*ratio + 1.61f*ratio + 0.573f);

	// Write pixel value to image array
	//img[x + y*width].r = red * 250.f;
	img[x + y*width].r = red * 250.f;
	img[x + y*width].g = green * 250.f;
	img[x + y*width].b = blue * 250.f;

      }
    }
  } else { // Render sphere binary

    cout << "Reading binary: " << argv[2] << "\n";

    FILE* fin;
    if ((fin = fopen(argv[2], "rb")) == NULL) {
      cout << "  Error encountered while trying to open binary '"
	<< argv[2] << "'\n";
      return 1;
    }

    // Read the number of dimensions and particles
    unsigned int nd, np;
    (void)fread(&nd, sizeof(nd), 1, fin);
    if (nd != 3) {
      cout << "  Input binary contains " << nd
	<< "D data, this raytracer is made for 3D data\n";
      return 1;
    }

    (void)fread(&np, sizeof(np), 1, fin);
    cout << "  Number of particles: " << np << "\n";

    // Allocate particle structure array
    cout << "  Allocating host memory\n";
    float4* p      = new float4[np];
    float*  fixvel = new float[np];
    float*  es_dot = new float[np];
    float*  es     = new float[np];
    float*  pres   = new float[np];
    for (unsigned int i=0; i<np; i++) {
      fixvel[i] = 0.0f;
      pres[i]   = 0.0f; // Initialize values to zero
      es_dot[i] = 0.0f;
      es[i]     = 0.0f;
    }

    // Read temporal information
    float dt, file_dt;
    double current, total;
    unsigned int step_count;
    (void)fread(&dt, sizeof(dt), 1, fin);
    (void)fread(&current, sizeof(current), 1, fin);
    (void)fread(&total, sizeof(total), 1, fin);
    (void)fread(&file_dt, sizeof(file_dt), 1, fin);
    (void)fread(&step_count, sizeof(step_count), 1, fin);

    // Canonical coordinate system origo
    f3 origo;
    (void)fread(&origo, sizeof(float)*3, 1, fin);

    // Canonical coordinate system dimensions
    f3 L;
    (void)fread(&L, sizeof(float)*3, 1, fin);

    // Skip over irrelevant data
    float blankf;
    //f3 blankf3;
    unsigned int blankui;
    for (int j=0; j<3; j++) 	// Skip over grid.num data
      (void)fread(&blankui, sizeof(blankui), 1, fin);


    // Load data into particle array
    for (unsigned int i=0; i<np; i++) {
      (void)fread(&p[i].x, sizeof(float), 1, fin);
      for (int j=0; j<4; j++)
	(void)fread(&blankf, sizeof(blankf), 1, fin);

      (void)fread(&p[i].y, sizeof(float), 1, fin);
      for (int j=0; j<4; j++)
	(void)fread(&blankf, sizeof(blankf), 1, fin);

      (void)fread(&p[i].z, sizeof(float), 1, fin);
      for (int j=0; j<4; j++)
	(void)fread(&blankf, sizeof(blankf), 1, fin);
    }
    for (unsigned int i=0; i<np; i++) {
      //(void)fread(&blankf, sizeof(blankf), 1, fin); // fixvel
      (void)fread(&fixvel[i], sizeof(float), 1, fin); // fixvel
      (void)fread(&blankf, sizeof(blankf), 1, fin); // xsum
      (void)fread(&p[i].w, sizeof(float), 1, fin);  // radius
      for (int j=0; j<10; j++)
	(void)fread(&blankf, sizeof(blankf), 1, fin);
      (void)fread(&es_dot[i], sizeof(float), 1, fin);
      (void)fread(&es[i], sizeof(float), 1, fin);
      (void)fread(&pres[i], sizeof(float), 1, fin);
    }

    fclose(fin);	// Binary read complete

    cout << "  Spatial dimensions: " 
      << L.x << "*" << L.y << "*" << L.z << " m\n";

    // Eye position and focus point
    //f3 eye    = {0.5f*L.x, -4.2*L.y, 0.5f*L.z};
    f3 eye    = {2.5f*L.x, -4.2*L.y, 2.0f*L.z};
    f3 lookat = {0.45f*L.x, 0.5f*L.y, 0.45f*L.z};
    if (L.z > (L.x + L.y)/1.5f) { // Render only bottom of world (for initsetup's)
      eye.x = 1.1f*L.x;
      eye.y = 15.1*L.y;
      eye.z = 1.1*L.z;
      /*lookat.x = 0.45f*L.x;
	lookat.y = 0.5f*L.y;
	lookat.z = 0.30f*L.z;*/
    }

    // Screen width in world coordinates
    //float imgw = 1.4f*L.x;
    //float imgw = pow(L.x*L.y*L.z, 0.32f);
    //float imgw = sqrt(L.x*L.y)*1.2f; // Adjust last float to capture entire height
    float imgw = L.x*1.35f;

    // Determine visualization mode
    int visualize = 0; // 0: ordinary view
    float max_val = 0;
    if (argc == 8) {
      if(strcmp(argv[6],"pressure") == 0)
	visualize = 1; // 1: pressure view
      if(strcmp(argv[6],"es_dot") == 0)
	visualize = 2; // 2: es_dot view
      if(strcmp(argv[6],"es") == 0)
	visualize = 3; // 3: es view
      if(strcmp(argv[6],"pressure50") == 0)
	visualize = 4; // 4: pressure50 view

      // Read max. value specified in command args.
      max_val = atof(argv[7]);
    }

    // Render colorbar image
    if (strcmp(argv[2],"colorbar") == 0) {

      for (unsigned int x=0; x<width; x++) {
	for (unsigned int y=0; y<height; y++) {

	  // Colormap value is relative to width position
	  float ratio = (float)x/width;

	  // Determine Blue-White-Red color components
	  float red   = fmin(1.0f, 0.209f*ratio*ratio*ratio - 2.49f*ratio*ratio + 3.0f*ratio + 0.0109f);
	  float green = fmin(1.0f, -2.44f*ratio*ratio + 2.15f*ratio + 0.369f);
	  float blue  = fmin(1.0f, -2.21f*ratio*ratio + 1.61f*ratio + 0.573f);

	  // Write pixel value to image array
	  img[y*height + x].r = (unsigned char) red * 255;
	  img[y*height + x].g = (unsigned char) green * 255;
	  img[y*height + x].b = (unsigned char) blue * 255;

	}
      }
    }

    if (strcmp(argv[1],"GPU") == 0) {

      // Call cuda wrapper
      if (rt(p, np, img, width, height, origo, L, eye, lookat, imgw, visualize, max_val, fixvel, pres, es_dot, es) != 0) {
	cout << "\nError in rt()\n";
	return 1;
      }
    } else if (strcmp(argv[1],"CPU") == 0) {

      // Call CPU wrapper
      if (rt_cpu(p, np, img, width, height, origo, L, eye, lookat, imgw) != 0) {
	cout << "\nError in rt_cpu()\n";
	return 1;
      }
    } else {
      cout << "Please specify CPU or GPU for execution\n";
      return 1;
    }

    // Free dynamically allocated memory
    delete [] p;
    delete [] fixvel;
    delete [] pres;
    delete [] es;
    delete [] es_dot;

  }

  // Write final image to PPM file
  image_to_ppm(img, argv[5], width, height);

  delete [] img;

  cout << "Terminating successfully\n\n";
  return 0;
}
