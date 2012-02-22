// io.cpp
// 	Contains functions to read SPHERE binary files,
// 	and write images in the PPM format

#include <iostream>
#include <cstdio>
#include "header.h"


// Write image structure to PPM file
int image_to_ppm(rgb* rgb_img, const char* filename, const unsigned int width, const unsigned int height)
{
  FILE *fout = fopen(filename, "wb"); 
  if (fout == NULL) {
    std::cout << "File error in img_write() while trying to writing " << filename;
    return 1;
  }

  fprintf(fout, "P6 %d %d 255\n", width, height);

  // Write pixel array to ppm image file
  for (unsigned int i=0; i<height*width; i++) {
      putc(rgb_img[i].r, fout);
      putc(rgb_img[i].g, fout);
      putc(rgb_img[i].b, fout);
  }

  fclose(fout);
  return 0;
}

