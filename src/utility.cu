#include <iostream>

// MISC. UTILITY FUNCTIONS

// Error handler for CUDA GPU calls. 
//   Returns error number, filename and line number containing the error to the terminal.
//   Please refer to CUDA_Toolkit_Reference_Manual.pdf, section 4.23.3.3 enum cudaError
//   for error discription. Error enumeration starts from 0.
void checkForCudaErrors(const char* checkpoint_description)
{
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cout << "\nCuda error detected, checkpoint: " << checkpoint_description
      << "\nError string: " << cudaGetErrorString(err) << "\n";
    exit(EXIT_FAILURE);
  }
}

void checkForCudaErrors(const char* checkpoint_description, const unsigned int iteration)
{
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cout << "\nCuda error detected, checkpoint: " << checkpoint_description
      << "\nduring iteration " << iteration
      << "\nError string: " << cudaGetErrorString(err) << "\n";
    exit(EXIT_FAILURE);
  }
}

//Round a / b to nearest higher integer value
unsigned int iDivUp(unsigned int a, unsigned int b) {
  return (a % b != 0) ? (a / b + 1) : (a / b);
}

