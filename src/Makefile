# Cuda paths
CUDA_INSTALL_PATH=/usr/local/cuda
#CUDA_BIN=$(CUDA_INSTALL_PATH)/bin

# Define the editor and optional arguments
EDITOR=vim -p

# Define compilers and linker
CC=g++
#NVCC=$(CUDA_BIN)/nvcc
NVCC=nvcc
LINKER=$(CC)

# Flags for producing optimized code
CCFLAGS=-Wall -O3
NVCCFLAGS=--use_fast_math -O3 -m64 -gencode=arch=compute_20,code=\"sm_20,compute_20\" -Xcompiler "-O3"

# Save runtime-generated code in cache.
# Must be specified in env. var. CUDA_DEVCODE_PATH before exec.
#NVCCFLAGS+=--export-dir code_repo

# Verbose compile?
#NVCCFLAGS+=--verbose

# Profile code?
#NVCCFLAGS+=-pg

# Debugable code? Beware that enabling this option will 
# considerably slow down the execution.
#CCFLAGS=-g -O0 -Wall
#NVCCFLAGS=-g -G -O0 -m64 -gencode=arch=compute_20,code=\"sm_20,compute_20\" -Xcompiler "-O0 -g"

DATE=`date +'%Y.%m.%d-%H:%M:%S'`
	BACKUPNAME=sphere.$(DATE).tar.gz

CCFILES=main.cpp file_io.cpp sphere.cpp
CUFILES=device.cu utility.cu
CCOBJECTS=$(CCFILES:.cpp=.o)
CUOBJECTS=$(CUFILES:.cu=.o)
OBJECTS=$(CCOBJECTS) $(CUOBJECTS)

# If a header-file changes, update all objects
DEPS=*.h *.cuh

# Detect OS
OSUPPER=$(shell uname -s 2>/dev/null | tr [:lower:] [:upper:])
	DARWIN=$(strip $(findstring DARWIN, $(OSUPPER)))
	ARCHUPPER=$(shell uname -m 2>/dev/null | tr [:lower:] [:upper:])
	ARCH:=$(strip $(findstring X86_64, $(ARCHUPPER)))

ifneq ($(DARWIN),) 	# If OS X
  #GL_LIBS+=-framework glut -framework OpenGL
  SDKPATH=/Developer/GPU\ Computing/C/common
  EXECUTABLE=../sphere_darwin_$(ARCH)
  CCFLAGS+=-arch x86_64
  LDFLAGS=-L$(SDKPATH)/lib/darwin
else			# If Linux
  CCFLAGS+=-DUNIX
  #GL_LIBS+=-L/usr/X11/lib -lglut -lGL -lGLU
  SDKPATH=$(HOME)/NVIDIA_GPU_Computing_SDK/C/common
  EXECUTABLE=../sphere_linux_$(ARCH)
  LDFLAGS=-L$(SDKPATH)/lib/linux -L$(CUDA_INSTALL_PATH)/lib -L$(CUDA_INSTALL_PATH)/lib64
endif

INCLUDES+=-I. -I$(SDKPATH)/shared/inc -I$(SDKPATH)/inc -I$(CUDA_INSTALL_PATH)/include
LDFLAGS+=-L$(CUDA_INSTALL_PATH)/lib
LDFLAGS+=-L$(SDKPATH)/../../shared/lib -L$(SDKPATH)/../lib 
LDFLAGS+=-lcutil_x86_64 -lcuda -lcudart

#all: $(CCFILES) $(CUFILES) $(EXECUTABLE) raytracer
all: $(CCFILES) $(CUFILES) $(EXECUTABLE)

#raytracer: 
#	$(MAKE) -C ../raytracer/

$(EXECUTABLE): $(OBJECTS)
	$(LINKER) $(OBJECTS) $(LDFLAGS) -o $@

utility.o: utility.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

file_io.o: file_io.cpp $(DEPS)
	$(CC) $(CCFLAGS) $(INCLUDES) -c $< -o $@

device.o: device.cu $(DEPS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

main.o: main.cpp $(DEPS)
	$(CC) $(CCFLAGS) $(INCLUDES) -c $< -o $@

sphere.o: sphere.cpp $(DEPS)
	$(CC) $(CCFLAGS) $(INCLUDES) -c $< -o $@

../sphere_status: sphere_status.cpp
	$(CC) $(CCFLAGS) sphere_status.cpp -o ../sphere_status

edit:
	$(EDITOR) Makefile $(CCFILES) $(CUFILES) *.h *.cuh 

backup:
	tar cvfz ../../sphere_backup/$(BACKUPNAME) ../../sphere
	cp -v ../../sphere_backup/sphere.*.tar.gz /Volumes/ice/adc/code/sphere_backup/

clean:
	$(RM) $(OBJECTS)
	$(RM) ../sphere_*
#	$(MAKE) -C ../raytracer clean

clear:
	# Remove all output data and images
	$(RM) ../img_out/*.{ppm,png}
	$(RM) ../output/*.{dat,bin}

