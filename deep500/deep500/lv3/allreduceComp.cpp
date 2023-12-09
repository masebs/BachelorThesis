// Custom Allreduce implementation based on MPI_Sendrecv, including message compression
// by ndzip
// Marc S. Schneider 2023

#include <mpi.h>
#include <iostream>
#include <cstring>
#include <cstdlib>

#include "cuda_runtime.h"
#include "ndzip_c_connector.h"
#include <cassert>

#ifdef USE_FLOAT
#define TYPE float
#define INTTYPE uint32_t
#define MPI_TYPE MPI_FLOAT
#else
#define TYPE double
#define INTTYPE uint64_t
#define MPI_TYPE MPI_DOUBLE
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Macro for MPI error check (taken from the OSU benchmark, c/util/osu_util_mpi.h)
#define MPI_CHECK(stmt)                                          \
do {                                                             \
    int mpi_errno = (stmt);                                      \
    if (MPI_SUCCESS != mpi_errno) {                              \
        fprintf(stderr, "[%s:%d] MPI call failed with %d \n",    \
        __FILE__, __LINE__,mpi_errno);                           \
        exit(EXIT_FAILURE);                                      \
    }                                                            \
    assert(MPI_SUCCESS == mpi_errno);                            \
} while (0)

// Macro and function for reporting CUDA errors
#define cudaErrorCheck(ans) { cudaAssert((ans), __FILE__, __LINE__, true); }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort) {
    if (code != cudaSuccess) 
    {
        fprintf(stderr, "cudaAssert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// Wrapper functions for the CUDA kernel, defined in addKernel/minKernel/maxKernel.cu
extern void cudaAdd(const int threadsPerBlock, const int blocksPerGrid, TYPE* d_result, const TYPE* d_var1, const TYPE* d_var2, const int numElements);
extern void cudaMin(const int threadsPerBlock, const int blocksPerGrid, TYPE* d_result, const TYPE* d_var1, const TYPE* d_var2, const int numElements);
extern void cudaMax(const int threadsPerBlock, const int blocksPerGrid, TYPE* d_result, const TYPE* d_var1, const TYPE* d_var2, const int numElements);
  
void allreduceComp(void *sendbuf, void *recvbuf, int count, char* op_str, void* comm_ptr) {
    int rank, num_processes;
    
    MPI_Comm comm = *((MPI_Comm*)comm_ptr); // hack to tranfer the communicator from mpi4py via void*
    MPI_Op op;
    if (!strcmp(op_str, "sum")) op = MPI_SUM;
    else if (!strcmp(op_str, "max")) op = MPI_MAX;
    else if (!strcmp(op_str, "min")) op = MPI_MIN;
    else return;
    
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &num_processes);
    
    const int datatype_size = sizeof(TYPE);
    const int message_size = count * datatype_size;
    //fprintf(stdout, "Message size: %d ", message_size);
    NDZIP_API_sayHi(message_size, ""); 
    
    // Allocate memory for compression and data transfer on GPU and CPU
    void *d_compressed_buffer_mem;
    void *d_compReceive_buffer_mem;
    void *d_send_buffer_mem;
    void *d_receive_buffer_mem;

    TYPE *d_compressed_buffer;
    TYPE *d_compReceive_buffer;
    TYPE *d_send_buffer;
    TYPE *d_receive_buffer;
      
    cudaErrorCheck( cudaMalloc(&d_compressed_buffer_mem, message_size) );
    cudaErrorCheck( cudaMalloc(&d_compReceive_buffer_mem, message_size) );
    cudaErrorCheck( cudaMalloc(&d_send_buffer_mem, message_size) );
    cudaErrorCheck( cudaMalloc(&d_receive_buffer_mem, message_size) );
    
    d_compressed_buffer = (TYPE*)d_compressed_buffer_mem;
    d_compReceive_buffer = (TYPE*)d_compReceive_buffer_mem;
    d_send_buffer = (TYPE*)d_send_buffer_mem;
    d_receive_buffer = (TYPE*)d_receive_buffer_mem;  
    
    // initialize memory with zeroes
    cudaErrorCheck( cudaMemcpy(d_send_buffer, sendbuf, message_size, cudaMemcpyHostToDevice) );
    cudaErrorCheck( cudaMemset(d_compressed_buffer, 0, message_size) );
    cudaErrorCheck( cudaMemset(d_compReceive_buffer, 0, message_size) );
    cudaErrorCheck( cudaMemset(d_receive_buffer, 0, message_size) );

    // Perform the butterfly-based Allreduce algorithm
    for (int mask = 1; mask < num_processes; mask <<= 1) {
        int partner = rank ^ mask;
        int compressed_length = 0;
        int compressed_length_partner = 0;
        
        compressed_length = NDZIP_API_compressBuffer(d_send_buffer, count, 
                            (INTTYPE *)d_compressed_buffer);
        fprintf(stdout, "Compressed length, uncomp size: %d, %d\n", compressed_length, count);;
        cudaErrorCheck( cudaDeviceSynchronize() );
        
        // Exchange compressed length with partner
        MPI_CHECK( MPI_Sendrecv(&compressed_length, 1, MPI_INT, partner, 0,
                                &compressed_length_partner, 1, MPI_INT, partner, 0,
                                MPI_COMM_WORLD, MPI_STATUS_IGNORE) );
        
        // Exchange compressed data with partner
        MPI_CHECK( MPI_Sendrecv(d_compressed_buffer, compressed_length, 
                                MPI_TYPE, partner, 0,
                                d_compReceive_buffer, compressed_length_partner,
                                MPI_TYPE, partner, 0, MPI_COMM_WORLD,
                                MPI_STATUS_IGNORE) );
        
        // Decompress received data and reduce it
        TYPE *buffer_ptr = NDZIP_API_decompressBuffer((INTTYPE *)d_compReceive_buffer,
                                    d_receive_buffer, count);  
        cudaErrorCheck( cudaDeviceSynchronize() );
            
        // Reduce our previous values (in d_receive_buffer) and the newly received ones (in d_compReceive_buffer); this is basically d_send_buffer += d_receive_buffer
        const int threadsPerBlock = 32;
        const int blocksPerGrid = (int)((count - 1) / threadsPerBlock + 1);
        if (op == MPI_SUM) {
            cudaAdd(threadsPerBlock, blocksPerGrid, d_send_buffer, d_send_buffer,
                d_receive_buffer, count); // wrapper for kernel in addKernel.cu 
        }
        else if (op == MPI_MAX) {
            cudaMax(threadsPerBlock, blocksPerGrid, d_send_buffer, d_send_buffer,
                d_receive_buffer, count); // wrapper for kernel in maxKernel.cu
        }
        else if (op == MPI_MIN) {
            cudaMin(threadsPerBlock, blocksPerGrid, d_send_buffer, d_send_buffer,
                d_receive_buffer, count); // wrapper for kernel in minKernel.cu 
        }
        else {
            fprintf(stderr, "ERROR in allreduceComp.cpp: MPI operation not implemented!");
        }
        
        cudaErrorCheck( cudaGetLastError() ); 
        cudaErrorCheck( cudaDeviceSynchronize() ); // wait for kernel to finish, before next butterfly stage or end of total timing
        
    }
    // Butterfly algorithm finished; result is now in d_send_buffer!
    cudaErrorCheck( cudaMemcpy(recvbuf, d_send_buffer, message_size, cudaMemcpyDeviceToHost) );
    cudaErrorCheck( cudaDeviceSynchronize() );
    MPI_CHECK( MPI_Barrier(MPI_COMM_WORLD) );
    
    // Cleanup
    cudaFree(d_compressed_buffer_mem);
    cudaFree(d_compReceive_buffer_mem);
    cudaFree(d_send_buffer_mem);
    cudaFree(d_receive_buffer_mem);
  
}

#ifdef __cplusplus
}
#endif
