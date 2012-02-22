// header.h -- Structure templates and function prototypes

// Standard technique for avoiding multiple inclusions of header file
#ifndef HEADER_H_
#define HEADER_H_


//// Structure declarations ////

struct f3 {
  float x;
  float y;
  float z;
};

// Image structure
struct rgb {
  unsigned char r;
  unsigned char g;
  unsigned char b;
  unsigned char alpha;
};



//// Prototype functions ////

// io.cpp
int image_to_ppm(rgb* rgb_img, const char* filename,
    		 const unsigned int width, const unsigned int height);


#endif
