# Create input/output folders
FILE(MAKE_DIRECTORY input)
FILE(MAKE_DIRECTORY output)
FILE(MAKE_DIRECTORY img_out)
FILE(MAKE_DIRECTORY gnuplot/data)

# The name of the project.
PROJECT(sphere_CUDA)

# CMake minimum version required
# FindCUDA script is distributed since version 2.8
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

# Find CUDA
FIND_PACKAGE(CUDA REQUIRED)

# Find OpenMP
FIND_PACKAGE(OpenMP)

# Find Boost components
#find_package(Boost COMPONENTS system filesystem unit_test_framework REQUIRED)

#SET(CMAKE_BUILD_TYPE Debug)
SET(CMAKE_BUILD_TYPE Release)


#Add source directory to project.
ADD_SUBDIRECTORY(src)
