# Create input/output folders
FILE(MAKE_DIRECTORY input)
FILE(MAKE_DIRECTORY output)
FILE(MAKE_DIRECTORY img_out)

# Link with libcutil (expected to be in ${CUDA_SDK_ROOT_DIR}/C/lib)
#LINK_LIBRARIES("-L${CUDA_SDK_ROOT_DIR}/lib -lcutil")        # For 32 bit systems
LINK_LIBRARIES("-L${CUDA_SDK_ROOT_DIR}/lib -lcutil_x86_64") # For 64 bit systems

# Ohter folders to include
INCLUDE_DIRECTORIES("${CUDA_SDK_ROOT_DIR}/common/inc")
INCLUDE_DIRECTORIES("${CMAKE_BINARY_DIR}/src")

# Include FindCUDA script
INCLUDE(FindCUDA)

# Additional NVCC command line arguments
# NOTE: Multiple arguments must be semi-colon selimited
SET(CUDA_NVCC_FLAGS "--use_fast_math;-O3;-gencode=arch=compute_20,code=\"sm_20,compute_20\"")  # without debug flags
#SET(CUDA_NVCC_FLAGS "--use_fast_math;-O3;-gencode=arch=compute_20,code=\"sm_20,compute_20\"-g;-G")   # with debug flags



# Rule to build executable program 
CUDA_ADD_EXECUTABLE(../sphere main.cpp file_io.cpp sphere.cpp device.cu utility.cu)
CUDA_ADD_EXECUTABLE(../porosity porosity.cpp file_io.cpp sphere.cpp device.cu utility.cu)

