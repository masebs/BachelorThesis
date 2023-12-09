// sumKernel.cu
// Marc S. Schneider 2023

#include <cuda_runtime.h>

#ifdef USE_FLOAT
#define TYPE float
#else
#define TYPE double
#endif

__global__ void sumKernel(TYPE* result, const TYPE* var1, const TYPE* var2, const int numElements) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < numElements) {
        result[tid] = var1[tid] + var2[tid];
    }
}

#ifdef __cplusplus
extern "C" {
#endif
// Wrapper function for the CUDA kernel
void cudaSum(const int threadsPerBlock, const int blocksPerGrid, TYPE* d_result, const TYPE* d_var1, const TYPE* d_var2, const int numElements) {
    sumKernel<<<blocksPerGrid, threadsPerBlock>>>(d_result, d_var1, d_var2, numElements);
}
#ifdef __cplusplus
}
#endif
