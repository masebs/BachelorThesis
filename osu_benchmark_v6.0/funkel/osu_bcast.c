#define BENCHMARK "OSU MPI%s Broadcast Latency Test"
/*
 * Based on the Ohio State University Micro Benchmark Suite v6.0:
 * https://mvapich.cse.ohio-state.edu/download/mvapich/osu-micro-benchmarks-6.0.tar.gz
 * Modified by Marc S. Schneider (FernUniversität in Hagen), following the previous work of MV:
 * - Reading data from an input dataset (fill_buffer)
 * - Using compression of the messages by ndzip
 * - Validating the data (directly implemented, without OSU's functions)
 * - Data type fixed to MPI_DOUBLE
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

// Loads the binary benchmark data into send_buffer
void fill_buffer(double *send_buffer, int size) {
  FILE *fileptr;
  char *buffer;

  fileptr = fopen("/home/00/mschneider/ba/ndzip/test/msg_sppm.fp64", "rb");
//   fileptr = fopen("/home/00/mschneider/ba/ndzip/test/num_plasma.fp64", "rb");

  buffer = (char *)malloc(size * sizeof(char));
  fread(buffer, size, sizeof(char), fileptr);
  fclose(fileptr);

  double *data = (double *)buffer;
  memcpy(send_buffer, data, size);
  free(buffer);
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

int main(int argc, char *argv[])
{
    int i = 0, j, rank, size;
    int numprocs;
    double avg_time = 0.0, max_time = 0.0, min_time = 0.0;
    double latency = 0.0, t_start = 0.0, t_stop = 0.0;
    double timer = 0.0;
    int po_ret;
    int errors = 0, local_errors = 0;
    options.bench = COLLECTIVE;
    options.subtype = BCAST;

    set_header(HEADER);
    set_benchmark_name("osu_bcast");
    po_ret = process_options(argc, argv);

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

    // Initialize NDZIP API using the maximum message size as buffer size
    NDZIP_API_sayHi(options.max_message_size, "Init hello");
    
    const int memory_buffer = options.max_message_size;
    const double validation_tolerance = 0.000001;

    // Allocate memory for compression and data transfer on GPU and CPU
    void *d_compressed_buffer_mem;
    void *d_send_buffer_mem;
    void *d_receive_buffer_mem;

    double *d_compressed_buffer;
    double *d_send_buffer;
    double *d_receive_buffer;

    double *send_buffer;
    double *receive_buffer;

    cudaErrorCheck( cudaMalloc(&d_compressed_buffer_mem, memory_buffer) );
    cudaErrorCheck( cudaMalloc(&d_send_buffer_mem, memory_buffer) );
    cudaErrorCheck( cudaMalloc(&d_receive_buffer_mem, memory_buffer) );

    send_buffer = (double*)malloc(memory_buffer);
    receive_buffer = (double*)malloc(memory_buffer);

    d_compressed_buffer = d_compressed_buffer_mem;
    d_send_buffer = d_send_buffer_mem;
    d_receive_buffer = d_receive_buffer_mem;
    
    // initialize memory with zeroes
    memset(send_buffer, 0, memory_buffer);
    memset(receive_buffer, 0, memory_buffer);
    cudaErrorCheck( cudaMemset(d_send_buffer, 0, memory_buffer) );
    cudaErrorCheck( cudaMemset(d_compressed_buffer, 0, memory_buffer) );
    cudaErrorCheck( cudaMemset(d_receive_buffer, 0, memory_buffer) );
    
    print_preamble(rank);
    
    // For each requested message size
    for (size = options.min_message_size; size <= options.max_message_size; size *= 2) {
        
        if (size > LARGE_MESSAGE_SIZE) {
            options.skip = options.skip_large;
            options.iterations = options.iterations_large;
        }
        
        // Load data into send_buffer of rank 0 (it will be broadcast from there to the other ranks)
        if (rank == 0) {
            fill_buffer(send_buffer, size);
            cudaErrorCheck( cudaMemcpy(d_send_buffer, send_buffer, size, cudaMemcpyHostToDevice) );
        }
        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
        timer = 0.0;

        for (i = 0; i < options.iterations + options.skip; i++) {
            int compressed_length = 0;
            memset(receive_buffer, 0, memory_buffer);
            
            ///// start of modified Bcast operation for benchmark
            t_start = MPI_Wtime();
             if (rank == 0) {
                compressed_length = NDZIP_API_compressBuffer(
                    d_send_buffer, size, (long unsigned int *)d_compressed_buffer);
            }
            
            // bcast compressed buffer size (MPI_INT)
            MPI_CHECK( MPI_Bcast(&compressed_length, 1, MPI_INT, 0, MPI_COMM_WORLD) );
            // bcast compressed buffer (MPI_DOUBLE)
            MPI_CHECK( MPI_Bcast(d_compressed_buffer, compressed_length, MPI_DOUBLE, 0, MPI_COMM_WORLD) );

            double *buffer_ptr = 
                NDZIP_API_decompressBuffer((long unsigned int *)d_compressed_buffer,
                                            d_receive_buffer, size);  
            
            cudaErrorCheck( cudaDeviceSynchronize() );
            t_stop = MPI_Wtime();
            
            cudaErrorCheck( cudaMemcpy(receive_buffer, d_receive_buffer, size, cudaMemcpyDeviceToHost) );
            ///// end of modified Bcast operation for benchmark
            
            MPI_CHECK( MPI_Barrier(MPI_COMM_WORLD) );

            if (i >= options.skip) {
                timer += t_stop - t_start;
            } 
        }

        // validation based on comparison with uncompressed, broadcast data
        if (options.validate) {
            int num_elements = (int)(size / sizeof(double));
            MPI_CHECK(MPI_Bcast(send_buffer, num_elements, MPI_DOUBLE, 0, MPI_COMM_WORLD));
            // send_buffer now contains the original data on all rank
            local_errors = 0;
            for (int ii = 0; ii < num_elements; ii++) {
                if (fabs(receive_buffer[ii] - send_buffer[ii]) > validation_tolerance) {
                    local_errors += 1;
                }
            }
            if (local_errors > 0)
                errors = 1;
        }
        
        latency = (timer * 1e6) / options.iterations;

        MPI_CHECK(MPI_Reduce(&latency, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0,
                MPI_COMM_WORLD));
        MPI_CHECK(MPI_Reduce(&latency, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0,
                MPI_COMM_WORLD));
        MPI_CHECK(MPI_Reduce(&latency, &avg_time, 1, MPI_DOUBLE, MPI_SUM, 0,
                MPI_COMM_WORLD));
        avg_time = avg_time/numprocs;
        if (options.validate) {
            MPI_CHECK(MPI_Allreduce(&local_errors, &errors, 1, MPI_INT, MPI_SUM,
                        MPI_COMM_WORLD));
        }
        if (options.validate) {
            print_stats_validate(rank, size, avg_time, min_time, max_time,
                    errors);
        } else {
            print_stats(rank, size, avg_time, min_time, max_time);
        }
        
        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
        
        if (0 != errors) {
            break;
        }
    }

    // Cleanup
    cudaFree(d_compressed_buffer_mem);
    cudaFree(d_send_buffer_mem);
    cudaFree(d_receive_buffer_mem);
    free(send_buffer);
    free(receive_buffer);

    MPI_CHECK(MPI_Finalize());

    if (NONE != options.accel) {
        if (cleanup_accel()) {
            fprintf(stderr, "Error cleaning up device\n");
            exit(EXIT_FAILURE);
        }
    }

    if (0 != errors && options.validate && 0 == rank) {
        fprintf(stdout, "DATA VALIDATION ERROR: %s exited with status %d on"
                " message size %d.\n", argv[0], EXIT_FAILURE, size);
        exit(EXIT_FAILURE);
    }
    
    return EXIT_SUCCESS;
}

/* vi: set sw=4 sts=4 tw=80: */
