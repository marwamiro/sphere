# Create input/output folders
file(MAKE_DIRECTORY input)
file(MAKE_DIRECTORY output)
file(MAKE_DIRECTORY img_out)
file(MAKE_DIRECTORY gnuplot/data)

# The name of the project.
project(sphere_CUDA)

# CMake minimum version required
# FindCUDA script is distributed since version 2.8
cmake_minimum_required(VERSION 2.8)

# Find CUDA
find_package(CUDA REQUIRED)

# Set CUDA GPU generation
# 0: Fermi
# 1: Kepler
set(GPU_GENERATION 1)

# Find OpenMP
find_package(OpenMP)

# Find Boost components
#find_package(Boost COMPONENTS system filesystem unit_test_framework REQUIRED)

# Uncomment to enable testing
enable_testing()

# Set build type = Debug
#set(CMAKE_BUILD_TYPE Debug)
#if (CUDA_FOUND)
#    set(CUDA_NVCC_FLAGS -g;-G)
#endif()

# Set build type = Release
set(CMAKE_BUILD_TYPE Release)

# Add source directory to project.
add_subdirectory(src)

# Add tests
add_subdirectory(tests)
