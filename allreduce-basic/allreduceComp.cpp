// Custom Allreduce implementation based on MPI_Sendrecv, including message compression
// by ndzip
// Marc S. Schneider 2023

#include <mpi.h>
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <time.h>
#include <unistd.h> // only for usleep in main()

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
extern void cudaSum(const int threadsPerBlock, const int blocksPerGrid, TYPE* d_result, const TYPE* d_var1, const TYPE* d_var2, const int numElements);
extern void cudaMin(const int threadsPerBlock, const int blocksPerGrid, TYPE* d_result, const TYPE* d_var1, const TYPE* d_var2, const int numElements);
extern void cudaMax(const int threadsPerBlock, const int blocksPerGrid, TYPE* d_result, const TYPE* d_var1, const TYPE* d_var2, const int numElements);
  
void allreduceComp(void *sendbuf, void *recvbuf, int count, MPI_Op op, MPI_Comm comm) {
    int rank, num_processes;
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
        
        // Compress message
        compressed_length = NDZIP_API_compressBuffer(d_send_buffer, count, 
                            (INTTYPE *)d_compressed_buffer);
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
            cudaSum(threadsPerBlock, blocksPerGrid, d_send_buffer, d_send_buffer,
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

// Test with count == 1 (one single value per process)
void test_single_values(int rank) {
    std::srand(time(NULL) + rank);
    TYPE send_value = rand() % 50;
    TYPE recv_value, recv_valid;

    std::cout << "Process " << rank << ", my value: " << send_value << std::endl;
    
    allreduceComp(&send_value, &recv_value, 1, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&send_value, &recv_valid, 1, MPI_TYPE, MPI_SUM, MPI_COMM_WORLD);
    
    std::cout << "Process " << rank << ": Result = " << recv_value << ", valid result: " << recv_valid << std::endl << std::flush;
}

// Test with an array of array_size elements on each process
void test_array_values(int rank) {
    const int array_size = 5;
    std::srand(time(NULL) + rank);
    TYPE sendbuf[array_size];
    for (int i = 0; i < array_size; i++)
    {
        sendbuf[i] = static_cast<TYPE>(rand() % 20);
    }
    TYPE recvbuf[array_size], recvbufvalid[array_size];

    // Print input
    std::cout << "Process " << rank << ", my values: ";
    for (int i = 0; i < array_size; i++) {
        std::cout << sendbuf[i]  << ", ";
    }
    std::cout << std::endl << std::flush;
    
    // Perform Allreduce to sum all elements in the array
    allreduceComp(sendbuf, recvbuf, array_size, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(sendbuf, recvbufvalid, array_size, MPI_TYPE, MPI_SUM, MPI_COMM_WORLD);
    
    MPI_Barrier(MPI_COMM_WORLD);
    // Print results
    if (rank == 0) {
        usleep(1000);
        std::cout << "Sum of elements after Allreduce: ";
        for (int i = 0; i < array_size; i++) {
            std::cout << recvbuf[i] << " ";
        }
        std::cout <<  std::endl << "Valid result from MPI_Allreduce: ";
        for (int i = 0; i < array_size; i++) {
            std::cout << recvbufvalid[i] << " ";
        }
        std::cout << std::endl << std::flush;
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    test_single_values(rank);
    if(rank == 0) {
        usleep(1000);
        std::cout << "\n\n" << std::flush;
    }
    MPI_Barrier(MPI_COMM_WORLD);

    test_array_values(rank);
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0) {
      usleep(1000);
      std::cout << std::endl << std::flush;
    }

    MPI_Finalize();
    return 0;
}

#ifdef __cplusplus
}
#endif
