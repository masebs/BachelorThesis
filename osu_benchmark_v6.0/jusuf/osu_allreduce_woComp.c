#define BENCHMARK "OSU MPI%s Allreduce Latency Test"
/*
 * Based on the Ohio State University Micro Benchmark Suite v6.0:
 * https://mvapich.cse.ohio-state.edu/download/mvapich/osu-micro-benchmarks-6.0.tar.gz
 * Modified by Marc S. Schneider (FernUniversität in Hagen)
 * 
 * This is a variant of osu_allreduce.c with the custom butterfly algorithm, but without 
 * compression, in order to compare the speed of the custom allreduce implementation with
 * the original MPI_Allreduce.
 * 
 * Copyright notice for the original OSU Benchmark:
 * Copyright (C) 2002-2023 the Network-Based Computing Laboratory
 * (NBCL), The Ohio State University; published under BSD license, cf. file COPYRIGHT_OSU
 * 
 */

#include "cuda_runtime.h"
#include "ndzip_c_connector.h"
#include <osu_util_mpi.h>
#include <stdbool.h>
#include "print_stats.h"

// Loads the binary benchmark data into send_buffer
void fill_buffer(double *send_buffer, int size) {
  FILE *fileptr;
  char *buffer;

  fileptr = fopen("/p/project/<PROJECT>/schneider11/osu_benchmark_v6.0/testdata/msg_sppm.fp64", "rb");
//   fileptr = fopen("/p/project/<PROJECT>/schneider11/osu_benchmark_v6.0/testdata/num_plasma.fp64", "rb");
//   fileptr = fopen("/p/project/<PROJECT>/schneider11/osu_benchmark_v6.0/testdata/msg_sweep3d.fp64", "rb");
//   fileptr = fopen("/p/project/<PROJECT>/schneider11/osu_benchmark_v6.0/testdata/num_brain.fp64", "rb");

  buffer = (char *)malloc(size * sizeof(char));
  fread(buffer, size, sizeof(char), fileptr);
  fclose(fileptr);

  double *data = (double *)buffer;
  memcpy(send_buffer, data, size);
  free(buffer);
}

void fill_buffer_constant(double *send_buffer, int size) {
  double data[size];
  for (int i = 0; i < size; i++) {
      data[i] = 1;
  };
  memcpy(send_buffer, data, size);
}

// Macro and function for reporting CUDA errors
#define cudaErrorCheck(ans) { cudaAssert((ans), __FILE__, __LINE__, false); }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort) {
   if (code != cudaSuccess) 
   {
      fprintf(stderr, "cudaAssert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// Wrapper function for the CUDA kernel, defined in addKernel.cu
extern void cudaAdd(const int threadsPerBlock, const int blocksPerGrid, double* d_result, const double* d_var1, const double* d_var2, const int numElements);

int main(int argc, char *argv[])
{
    // Standard benchmark variables
    int i, j, numprocs, rank, size;
    int po_ret;
    int errors = 0, local_errors = 0;
    size_t bufsize;
    options.bench = COLLECTIVE;
    options.subtype = REDUCE;

    // Variables for timing
    // Timing is done completely via MPI_Wtime(). One could also use cudaEventRecord for the parts involving CUDA (compression, decompression, reduction), but this yields times which don't really add up to the total time measured by MPI_Wtime, and the total time is slightly higher. However, MPI_Wtime yield very (too) small times for decompression at large size!
    double latency = 0.0, t_start_all = 0.0, t_stop_all = 0.0;
    double t_start_comp = 0.0, t_stop_comp = 0.0, t_start_comm = 0.0, t_stop_comm = 0.0;
    double t_start_decomp = 0.0, t_stop_decomp = 0.0, t_start_reduce = 0.0, t_stop_reduce = 0.0;
    double timer_all = 0.0, timer_comp = 0.0, timer_comm = 0.0;
    double timer_decomp = 0.0, timer_reduce = 0.0;
    double avg_time = 0.0, max_time = 0.0, min_time = 0.0;
    double avg_comp_time = 0.0, avg_comm_time = 0.0, avg_decomp_time = 0.0;
    double avg_reduce_time = 0.0;
    int avgCompressedLength = 0;
    
    // Set up some benchmark stuff
    set_header(HEADER);
    set_benchmark_name("osu_allreduce_withoutCompression");
    po_ret = process_options(argc, argv);

    // Setup CUDA; done via the benchmark's inic_accel; this also calls cudaSetDevice(...)
    options.accel = CUDA; // Force CUDA as accelerator, irrespective of the -d cmd line option
    if (PO_OKAY == po_ret && NONE != options.accel) {
        if (init_accel()) {
            fprintf(stderr, "Error initializing device\n");
            exit(EXIT_FAILURE);
        }
    }
    
    MPI_CHECK( MPI_Init(&argc, &argv) );
    MPI_CHECK( MPI_Comm_rank(MPI_COMM_WORLD, &rank) );
    MPI_CHECK( MPI_Comm_size(MPI_COMM_WORLD, &numprocs) );

    switch (po_ret) {
        case PO_BAD_USAGE:
            print_bad_usage_message(rank);
            MPI_CHECK(MPI_Finalize());
            exit(EXIT_FAILURE);
        case PO_HELP_MESSAGE:
            print_help_message(rank);
            MPI_CHECK(MPI_Finalize());
            exit(EXIT_SUCCESS);
        case PO_VERSION_MESSAGE:
            print_version_message(rank);
            MPI_CHECK(MPI_Finalize());
            exit(EXIT_SUCCESS);
        case PO_OKAY:
            break;
    }

    if (numprocs < 2) {
        if (rank == 0) {
            fprintf(stderr, "This test requires at least two processes\n");
        }

        MPI_CHECK(MPI_Finalize());
        exit(EXIT_FAILURE);
    }

    if (options.max_message_size > options.max_mem_limit) {
        if (rank == 0) {
            fprintf(stderr, "Warning! Increase the Max Memory Limit to be able"
                    " to run up to %ld bytes.\n"
                    " Continuing with max message size of %ld bytes\n",
                    options.max_message_size, options.max_mem_limit);
        }
        options.max_message_size = options.max_mem_limit;
    }

    if (options.min_message_size < MIN_MESSAGE_SIZE) {
        options.min_message_size = MIN_MESSAGE_SIZE;
    }

    // Initialize NDZIP API using the maximum message size as buffer size
    NDZIP_API_sayHi(options.max_message_size, "Init hello");

    const int memory_buffer = options.max_message_size;
    const double validation_tolerance = 0.000001;
    
    // Allocate memory for compression and data transfer on GPU and CPU
    void *d_send_buffer_mem;
    void *d_receive_buffer_mem;

    double *d_send_buffer;
    double *d_receive_buffer;
     
    double *send_buffer;
    double *receive_buffer;
    double *validresult;   // only for validation
    double *result_buffer; // only required to preserve the original data in send_buffer
                           // for validation and subsequent iterations; otherwise we could overwrite it

    cudaErrorCheck( cudaMalloc(&d_send_buffer_mem, memory_buffer) );
    cudaErrorCheck( cudaMalloc(&d_receive_buffer_mem, memory_buffer) );

    send_buffer = (double*)malloc(memory_buffer);
    receive_buffer = (double*)malloc(memory_buffer);
    result_buffer = (double*)malloc(memory_buffer);
    if (options.validate) {
        validresult = (double*)malloc(memory_buffer);
    }

    d_send_buffer = (double*)d_send_buffer_mem;
    d_receive_buffer = (double*)d_receive_buffer_mem;
    
    // initialize memory with zeroes
    memset(send_buffer, 0, memory_buffer);
    memset(receive_buffer, 0, memory_buffer);
    cudaErrorCheck( cudaMemset(d_send_buffer, 0, memory_buffer) );
    cudaErrorCheck( cudaMemset(d_receive_buffer, 0, memory_buffer) );

    print_preamble_detailed(rank);

    // For each requested message size
    for (size = options.min_message_size; size <= options.max_message_size; size *= 2) {

        if (size > LARGE_MESSAGE_SIZE) {
            options.skip = options.skip_large;
            options.iterations = options.iterations_large;
        }

        int count = (int)(size / sizeof(double) + 0.0001);
        
        // Load the data into send_buffer (on all ranks)
        fill_buffer(send_buffer, size);
//         fill_buffer_constant(send_buffer, size);

        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

        timer_all = 0.0; timer_comp = 0.0; timer_comm = 0.0;
        timer_decomp = 0.0; timer_reduce = 0.0;

        for (i = 0; i < options.iterations + options.skip; i++) {
            
            // clear receive and result buffer before routine
            memset(receive_buffer, 0, size);
            memset(result_buffer, 0, size);
            cudaErrorCheck( cudaMemcpy(d_send_buffer, send_buffer, size, cudaMemcpyHostToDevice) );
            cudaErrorCheck( cudaDeviceSynchronize() );

            ///// start of modified Allreduce operation for benchmark
            t_start_all = MPI_Wtime();
            // each rank compresses its fraction of data
            
            // Perform the butterfly-based Allreduce algorithm 
            for (int mask = 1; mask < numprocs; mask <<= 1) {
                int partner = rank ^ mask;
                
                // Exchange uncompressed data with partner
                t_start_comm = MPI_Wtime();
                MPI_CHECK( MPI_Sendrecv(d_send_buffer, count, MPI_DOUBLE, partner, 0,
                                        d_receive_buffer, count, MPI_DOUBLE, partner, 0,
                                        MPI_COMM_WORLD, MPI_STATUS_IGNORE) );
                t_stop_comm = MPI_Wtime();
                
                // Reduce our previous values (in d_receive_buffer) and the newly received ones (in d_compReceive_buffer); this is basically d_send_buffer += d_receive_buffer
                t_start_reduce = MPI_Wtime();
                const int threadsPerBlock = 32;
                const int blocksPerGrid = (int)((count - 1) / threadsPerBlock + 1);
                cudaAdd(threadsPerBlock, blocksPerGrid, d_send_buffer, d_send_buffer,
                        d_receive_buffer, count); // wrapper for kernel in addKernel.cu 
                cudaErrorCheck( cudaGetLastError() ); 
                cudaErrorCheck( cudaDeviceSynchronize() ); // wait for kernel to finish, before next butterfly stage or end of total timing
                t_stop_reduce = MPI_Wtime();
                
                if (i >= options.skip) {
                    timer_comm += t_stop_comm - t_start_comm;
                    timer_reduce += t_stop_reduce - t_start_reduce;
                }
            }
                
            t_stop_all = MPI_Wtime();
            cudaErrorCheck( cudaMemcpy(result_buffer, d_send_buffer, size,
                                       cudaMemcpyDeviceToHost) );
            
            cudaErrorCheck( cudaDeviceSynchronize() );
            MPI_CHECK( MPI_Barrier(MPI_COMM_WORLD) );
            ///// end of modified Allreduce operation for benchmark
            
            if (i >= options.skip) {
                timer_all += t_stop_all - t_start_all;
            }
        }
        
        // validation based on comparison with uncompressed, broadcast data
        if (options.validate) {
            memset(validresult, 0, memory_buffer);
            MPI_CHECK( MPI_Allreduce(send_buffer, validresult, count, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD) );
            MPI_CHECK(MPI_Barrier( MPI_COMM_WORLD) );
            local_errors = 0;
            for (int ii = 0; ii < count; ii++) {
                if (fabs(result_buffer[ii] - validresult[ii]) > validation_tolerance) {
                    local_errors += 1;
                }
            }
            if (local_errors > 0)
                errors = 1;
        }
        
        latency = (double)(timer_all * 1e6) / options.iterations;
        timer_comm = (double)(timer_comm * 1e6) / options.iterations;
        timer_reduce = (double)(timer_reduce * 1e6) / options.iterations;

        MPI_CHECK(MPI_Reduce(&latency, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0,
                MPI_COMM_WORLD));
        MPI_CHECK(MPI_Reduce(&latency, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0,
                MPI_COMM_WORLD));
        MPI_CHECK(MPI_Reduce(&latency, &avg_time, 1, MPI_DOUBLE, MPI_SUM, 0,
                MPI_COMM_WORLD));
        latency = avg_time / numprocs;
        MPI_CHECK(MPI_Reduce(&timer_comm, &avg_comm_time, 1, MPI_DOUBLE, MPI_SUM, 0,
                MPI_COMM_WORLD));
        avg_comm_time /= numprocs;
        MPI_CHECK(MPI_Reduce(&timer_reduce, &avg_reduce_time, 1, MPI_DOUBLE, MPI_SUM, 0,
                MPI_COMM_WORLD));
        avg_reduce_time /= numprocs;
        
        if (options.validate) {
            MPI_CHECK(MPI_Allreduce(&local_errors, &errors, 1, MPI_INT, MPI_SUM,
                        MPI_COMM_WORLD));
        }

        if (options.validate) {
            print_stats_validate_detailed(rank, size, 0.0, latency, min_time,
                    max_time, errors, 0.0, avg_comm_time, 0.0,
                    avg_reduce_time);
        } else {
            print_stats_detailed(rank, size, 0.0, latency, min_time,
                    max_time, 0.0, avg_comm_time, 0.0, avg_reduce_time);
        }

        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

        if (0 != errors) {
            break;
        }
    }
    
    // Cleanup
    cudaFree(d_send_buffer_mem);
    cudaFree(d_receive_buffer_mem);
    free(send_buffer);
    free(receive_buffer);
    free(result_buffer);
    if (options.validate) {
        free(validresult);
    }

    MPI_CHECK(MPI_Finalize());

    if (NONE != options.accel) {
        if (cleanup_accel()) {
            fprintf(stderr, "Error cleaning up device\n");
            exit(EXIT_FAILURE);
        }
    }

    if (0 != errors && options.validate && 0 == rank ) {
        fprintf(stdout, "DATA VALIDATION ERROR: %s exited with status %d on"
                " message size %d.\n", argv[0], EXIT_FAILURE, size);
        exit(EXIT_FAILURE);
    }

    return EXIT_SUCCESS;
}
