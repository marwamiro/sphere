# Link with libcutil (expected to be in ${CUDA_SDK_ROOT_DIR}/C/lib)
#LINK_LIBRARIES("-L${CUDA_SDK_ROOT_DIR}/lib -lcutil")        # For 32 bit systems
#LINK_LIBRARIES("-L${CUDA_SDK_ROOT_DIR}/lib -lcutil_x86_64") # For 64 bit systems

# Ohter folders to include
SET(CUDA_SDK_ROOT_DIR "/usr/local/cuda-5.0/samples")
INCLUDE_DIRECTORIES("${CUDA_SDK_ROOT_DIR}/common/inc")
INCLUDE_DIRECTORIES("${CMAKE_BINARY_DIR}/src")

# Include FindCUDA script
INCLUDE(FindCUDA)

# Additional NVCC command line arguments
# NOTE: Multiple arguments must be semi-colon selimited
IF (GPU_GENERATION EQUAL 1) # Kepler
    SET(CUDA_NVCC_FLAGS
        "--use_fast_math;-O3;-gencode=arch=compute_35,code=\"sm_35,compute_35\";--fmad=false -ccbin gcc-4.6")
ELSE()  # Fermi
    SET(CUDA_NVCC_FLAGS
        "--use_fast_math;-O3;-gencode=arch=compute_20,code=\"sm_20,compute_20\";--fmad=false -ccbin gcc-4.6")
ENDIF (GPU_GENERATION EQUAL 1)

# Rule to build executable program 
CUDA_ADD_EXECUTABLE(../sphere
    main.cpp file_io.cpp sphere.cpp device.cu utility.cu utility.cpp
    navierstokes.cpp)
CUDA_ADD_EXECUTABLE(../porosity
    porosity.cpp file_io.cpp sphere.cpp device.cu utility.cu utility.cpp
    navierstokes.cpp)
CUDA_ADD_EXECUTABLE(../forcechains
    forcechains.cpp file_io.cpp sphere.cpp device.cu utility.cu utility.cpp
    navierstokes.cpp)
CUDA_ADD_EXECUTABLE(../porousflow
    porousflow.cpp navierstokes.cpp file_io.cpp sphere.cpp device.cu utility.cpp utility.cu)

#ADD_EXECUTABLE(unittests boost-unit-tests.cpp sphere.cpp)
#TARGET_LINK_LIBRARIES(unittests
#                      ${Boost_FILESYSTEM_LIBRARY}
#                      ${Boost_SYSTEM_LIBRARY}
#                      ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})
