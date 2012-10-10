#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector_functions.h>
#include "header.h"
#include "rt-kernel.h"
#include "rt-kernel-cpu.h"
#include "colorbar.h"
#include "o-ppm.h"

int main(const int argc, const char* argv[])
{
  using std::cout;

  if (argc < 6 || argc > 8 ){
    cout << "Usage: " << argv[0] << " <GPU or CPU> <particle-data.txt> <width> <height> <output-image.ppm>\n"
         << "or\n"
         << "Usage: " << argv[0] << " GPU <particle-data.txt | colorbar> <width> <height> <output-image.ppm> [pressure,es_dot,es,vel max_val]";
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

    for (unsigned int y=0; y<height; ++y) {
      for (unsigned int x=0; x<width; ++x) {

	// Colormap value is relative to width position
	float ratio = (float) (x+1)/width;

	// Write pixel value to image array
	img[x + y*width].r = red(ratio) * 250.f;
	img[x + y*width].g = green(ratio) * 250.f;
	img[x + y*width].b = blue(ratio) * 250.f;

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

    // Single precision arrays used for computations
    float4* p      = new float4[np];
    float*  fixvel = new float[np];
    float*  xsum   = new float[np];
    float*  es_dot = new float[np];
    float*  ev_dot = new float[np];
    float*  es     = new float[np];
    float*  ev     = new float[np];
    float*  pres   = new float[np];
    float*  vel	   = new float[np];

    // Read temporal information
    double dt, file_dt, current, total;
    unsigned int step_count;
    (void)fread(&dt, sizeof(dt), 1, fin);
    (void)fread(&current, sizeof(current), 1, fin);
    (void)fread(&total, sizeof(total), 1, fin);
    (void)fread(&file_dt, sizeof(file_dt), 1, fin);
    (void)fread(&step_count, sizeof(step_count), 1, fin);

    double d; // Double precision temporary value holder

    // Canonical coordinate system origo
    f3 origo;
    (void)fread(&d, sizeof(d), 1, fin);
    origo.x = (float)d;
    (void)fread(&d, sizeof(d), 1, fin);
    origo.y = (float)d;
    (void)fread(&d, sizeof(d), 1, fin);
    origo.z = (float)d;

    // Canonical coordinate system dimensions
    f3 L;
    (void)fread(&d, sizeof(d), 1, fin);
    L.x = (float)d;
    (void)fread(&d, sizeof(d), 1, fin);
    L.y = (float)d;
    (void)fread(&d, sizeof(d), 1, fin);
    L.z = (float)d;


    // Skip over irrelevant data
    double blankd;
    //f3 blankd3;
    unsigned int blankui;
    for (int j=0; j<3; j++) 	// Skip over grid.num data
      (void)fread(&blankui, sizeof(blankui), 1, fin);

    // Velocity vector, later converted to lenght
    float3 v;

    // Load data into particle array
    for (unsigned int i=0; i<np; i++) {
      (void)fread(&d, sizeof(d), 1, fin);
      p[i].x = (float)d;	// Typecast to single precision
      (void)fread(&d, sizeof(d), 1, fin);
      v.x  = (float)d;
      for (int j=0; j<3; j++)
	(void)fread(&blankd, sizeof(blankd), 1, fin);

      (void)fread(&d, sizeof(d), 1, fin);
      p[i].y = (float)d;
      (void)fread(&d, sizeof(d), 1, fin);
      v.y  = (float)d;
      for (int j=0; j<3; j++)
	(void)fread(&blankd, sizeof(blankd), 1, fin);

      (void)fread(&d, sizeof(d), 1, fin);
      p[i].z = (float)d;
      (void)fread(&d, sizeof(d), 1, fin);
      v.z  = (float)d;
      for (int j=0; j<3; j++)
	(void)fread(&blankd, sizeof(blankd), 1, fin);

      // Save velocity vector length
      vel[i] = sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
    }
    for (unsigned int i=0; i<np; i++) {
      (void)fread(&d, sizeof(d), 1, fin); // fixvel
      fixvel[i] = (float)d;
      (void)fread(&d, sizeof(d), 1, fin); // xsum
      xsum[i] = (float)d;
      (void)fread(&d, sizeof(d), 1, fin);  // radius
      p[i].w = (float)d;
      for (int j=0; j<10; j++)
	(void)fread(&blankd, sizeof(blankd), 1, fin);
      (void)fread(&d, sizeof(d), 1, fin);
      es_dot[i] = (float)d;
      (void)fread(&d, sizeof(d), 1, fin);
      ev_dot[i] = (float)d;
      (void)fread(&d, sizeof(d), 1, fin);
      es[i] = (float)d;
      (void)fread(&d, sizeof(d), 1, fin);
      ev[i] = (float)d;
      (void)fread(&d, sizeof(d), 1, fin);
      pres[i] = (float)d;
    }

    fclose(fin);	// Binary read complete

    cout << "  Spatial dimensions: " 
      << L.x << "*" << L.y << "*" << L.z << " m\n";

    // Eye position and focus point
    //f3 eye    = {0.5f*L.x, -4.2f*L.y, 0.5f*L.z};
    //f3 eye    = {2.5f*L.x, -4.2f*L.y, 2.0f*L.z};
    //f3 eye    = {0.5f*L.x, -5.0f*L.y, 0.5f*L.z};
    f3 eye    = {2.5f*L.x, -5.0f*L.y, 1.5f*L.z};
    f3 lookat = {0.5f*L.x, 0.5f*L.y, 0.5f*L.z};
    if (L.z > (L.x + L.y)/1.5f) { // Render only bottom of world (for initsetup's)
      eye.x = 4.1f*L.x;
      eye.y = -15.1*L.y;
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
      if(strcmp(argv[6],"vel") == 0)
	visualize = 4; // 4: velocity view
      if(strcmp(argv[6],"xsum") == 0)
	visualize = 5; // 5: xsum view

      // Read max. value specified in command args.
      max_val = atof(argv[7]);

    }


    if (strcmp(argv[1],"GPU") == 0) {

      // Call cuda wrapper
      if (rt(p, np, img, width, height, origo, L, eye, lookat, imgw, visualize, max_val, fixvel, xsum, pres, es_dot, es, vel) != 0) {
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
    delete [] xsum;
    delete [] pres;
    delete [] es;
    delete [] ev;
    delete [] es_dot;
    delete [] ev_dot;
    delete [] vel;

  }

  // Write final image to PPM file
  image_to_ppm(img, argv[5], width, height);

  delete [] img;

  cout << "Terminating successfully\n\n";
  return 0;
}
