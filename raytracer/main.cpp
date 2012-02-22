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
         << "Usage: " << argv[0] << " GPU <sphere-binary.bin> <width> <height> <output-image.ppm>"
	 << "<pressure | pressure50 | es | es_dot> <max value in color range>\n";
    return 1;
  }

  cout << "\n----------------------------\n"
       << "This is the SPHERE raytracer\n"
       << "----------------------------\n";


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
  float*  es_dot = new float[np];
  float*  es     = new float[np];
  float*  pres   = new float[np];
  for (unsigned int i=0; i<np; i++) {
    pres[i]   = 0.0f; // Initialize values to zero
    es_dot[i] = 0.0f;
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
    (void)fread(&blankf, sizeof(blankf), 1, fin); // fixvel
    (void)fread(&blankf, sizeof(blankf), 1, fin); // xsum
    (void)fread(&p[i].w, sizeof(float), 1, fin);
    for (int j=0; j<12; j++)
      (void)fread(&blankf, sizeof(blankf), 1, fin);
    (void)fread(&es_dot[i], sizeof(float), 1, fin);
    (void)fread(&es[i], sizeof(float), 1, fin);
    (void)fread(&pres[i], sizeof(float), 1, fin);
  }

  fclose(fin);	// Binary read complete

  cout << "  Spatial dimensions: " 
       << L.x << "*" << L.y << "*" << L.z << " m\n";
 
  // Eye position and focus point
  f3 eye    = {2.5f*L.x, -4.2*L.y, 2.0f*L.z};
  f3 lookat = {0.45f*L.x, 0.5f*L.y, 0.4f*L.z};
  if (L.z > (L.x + L.y)/1.5f) { // Render only bottom of world (for initsetup's)
    eye.x = 2.5f*L.x;
    eye.y = -6.2*L.y;
    eye.z = 0.12*L.z;
    lookat.x = 0.45f*L.x;
    lookat.y = 0.5f*L.y;
    lookat.z = 0.012f*L.z;
  }

  // Screen width in world coordinates
  //float imgw = 1.4f*L.x;
  //float imgw = pow(L.x*L.y*L.z, 0.32f);
  //float imgw = sqrt(L.x*L.y)*1.2f; // Adjust last float to capture entire height
  float imgw = L.x*1.35f;

  // Allocate memory for image
  unsigned int width = atoi(argv[3]);
  unsigned int height = atoi(argv[4]);
  if (width < 1 || height < 1) {
    cout << "Image dimensions invalid.\n";
    return 1;
  }
  rgb* img;
  img = new rgb [height*width];

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

  if (strcmp(argv[1],"GPU") == 0) {
    
    // Call cuda wrapper
    if (rt(p, np, img, width, height, origo, L, eye, lookat, imgw, visualize, max_val, pres, es_dot, es) != 0) {
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


  // Write final image to PPM file
  image_to_ppm(img, argv[5], width, height);


  // Free dynamically allocated memory
  delete [] p;
  delete [] pres;
  delete [] es;
  delete [] es_dot;
  delete [] img;

  cout << "Terminating successfully\n\n";
  return 0;
}
