NVCC=nvcc
CC=g++
LINK=g++

CFLAGS=-c -Wall -O3 -fopenmp
LDFLAGS=-fopenmp

CSOURCES=main.cpp io.cpp rt_kernel_cpu.cpp
NVCCSOURCES=rt_kernel.cu

COBJECTS=$(CSOURCES:.cpp=.o)
CPUOBJECTS=$(CPUSOURCES:.cpp=.o)
NVCCOBJECTS=$(NVCCSOURCES:.cu=.o)
OBJECTS=$(COBJECTS) $(NVCCOBJECTS)

EXECUTABLE=rt


# NVCC debug compilation
NVCCFLAGS+=-use_fast_math -gencode arch=compute_20,code=sm_20

INCLUDES = -I. -I../libs -I/usr/local/cuda/include
LIBS = -L/usr/local/cuda/lib64 -L/usr/local/cuda/lib -lcudart -lcuda 
#GL_LIBS = -lGLEW

# detect OS
OSUPPER = $(shell uname -s 2>/dev/null | tr [:lower:] [:upper:])
# 'linux' is output for Linux system, 'darwin' for OS X
DARWIN = $(strip $(findstring DARWIN, $(OSUPPER)))
ifneq ($(DARWIN),)
    #INCLUDES += -I/opt/local/include
    #LIBS += -L/opt/local/lib
    #GL_LIBS += -framework glut -framework OpenGL 
    CUDA_SDK=/Developer/GPU\ Computing/C
else
    INCLUDES += -I/usr/include -I/usr/local/include
    LIBS += -L/usr/lib -L/usr/local/lib
    #GL_LIBS += -L/usr/X11/lib -lglut -lGL -lGLU
    CUDA_SDK=$(HOME)/NVIDIA_GPU_Computing_SDK/C
endif

# detect OS
ARCHUPPER = $(shell uname -m 2>/dev/null | tr [:lower:] [:upper:])
# 'linux' is output for Linux system, 'darwin' for OS X
ARCH := $(strip $(findstring X86_64, $(ARCHUPPER)))
ifneq ($(ARCH),)
    LIB_ARCH = x86_64
    NVCCFLAGS += -m64
    ifneq ($(DARWIN),)
        CXX_ARCH_FLAGS += -arch x86_64
    else
        CXX_ARCH_FLAGS += -m64
    endif
else
    LIB_ARCH = i386
    NVCCFLAGS += -m32
    ifneq ($(DARWIN),)
        CXX_ARCH_FLAGS += -arch i386
    else
        CXX_ARCH_FLAGS += -m32
    endif
endif


INCLUDES += -I$(CUDA_SDK)/common/inc
#LIBS += -L$(CUDA_SDK)/lib #-lcutil_$(LIB_ARCH)
LIBS += -L$(CUDA_SDK)/lib -lcutil_$(LIB_ARCH)


$(EXECUTABLE): $(OBJECTS)
	$(LINK) $(CXX_ARCH_FLAGS) $(LDFLAGS) $(LIBS) $^ -o $(EXECUTABLE)

$(NVCCOBJECTS): $(NVCCSOURCES)
	$(NVCC) $(NVCCFLAGS) -c $(INCLUDES) $< -o $@

.cpp.o: $(COBJECTS) 
	$(CC) $(CFLAGS) $(CXX_ARCH_FLAGS) -c $(INCLUDES) $< -o $@

clean:
	rm -f doc/*.{log,aux,bbl,blg,out,gz}
	rm -f *.o *.ppm *.png
	rm -f $(EXECUTABLE)

edit:
	vim -p Makefile $(CSOURCES) $(NVCCSOURCES) header.h rt_kernel.h rt_kernel_cpu.h 

run:
	./rt GPU np5000.bin 800 800 out.ppm
	convert out.ppm out.png
	#open out.png
