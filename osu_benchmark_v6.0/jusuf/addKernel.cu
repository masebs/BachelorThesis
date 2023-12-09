// addKernel.cu
// Marc S. Schneider, 2023

#include <cuda_runtime.h>

__global__ void addKernel(double* result, const double* var1, const double* var2, const int numElements) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < numElements) {
        result[tid] = var1[tid] + var2[tid];
    }
}

#ifdef __cplusplus
extern "C" {
#endif
// Wrapper function for the CUDA kernel
void cudaAdd(const int threadsPerBlock, const int blocksPerGrid, double* d_result, const double* d_var1, const double* d_var2, const int numElements) {
    addKernel<<<blocksPerGrid, threadsPerBlock>>>(d_result, d_var1, d_var2, numElements);
}
#ifdef __cplusplus
}
#endif
