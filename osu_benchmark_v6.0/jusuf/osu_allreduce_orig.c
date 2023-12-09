#define BENCHMARK "OSU MPI%s Allreduce Latency Test"
/*
 * Copyright (C) 2002-2022 the Network-Based Computing Laboratory
 * (NBCL), The Ohio State University.
 *
 * Contact: Dr. D. K. Panda (panda@cse.ohio-state.edu)
 *
 * For detailed copyright and licensing information, please refer to the
 * copyright file COPYRIGHT in the top level OMB directory.
 */
/* Extended to use custom data (fill_buffer), and switched from float to double, and forced to use CUDA. Validation not working!
 * Marc S. Schneider 2023
 */

#include "cuda_runtime.h"
#include <osu_util_mpi.h>

// Loads the binary benchmark data into send_buffer
void fill_buffer(double *send_buffer, int size) {
  FILE *fileptr;
  char *buffer;

  fileptr = fopen("/p/project/<PROJECT>/schneider11/osu_benchmark_v6.0/testdata/msg_sppm.fp64", "rb");
//   fileptr = fopen("/p/project/<PROJECT>/schneider11/osu_benchmark_v6.0/testdata/num_plasma.fp64", "rb");

  buffer = (char *)malloc(size * sizeof(char));
  fread(buffer, size, sizeof(char), fileptr);
  fclose(fileptr);

  double *data = (double *)buffer;
  memcpy(send_buffer, data, size);
  free(buffer);
}

int main(int argc, char *argv[])
{
    int i, j, numprocs, rank, size;
    double latency = 0.0, t_start = 0.0, t_stop = 0.0;
    double timer=0.0;
    double avg_time = 0.0, max_time = 0.0, min_time = 0.0;
    double *sendbuf, *recvbuf, *copybuf;
    int po_ret;
    int errors = 0, local_errors = 0;
    size_t bufsize;
    options.bench = COLLECTIVE;
    options.subtype = REDUCE;

    set_header(HEADER);
    set_benchmark_name("osu_allreduce");
    po_ret = process_options(argc, argv);

    options.accel = CUDA; // Force CUDA as accelerator, irrespective of the -d cmd line option
    if (PO_OKAY == po_ret && NONE != options.accel) {
        if (init_accel()) {
            fprintf(stderr, "Error initializing device\n");
            exit(EXIT_FAILURE);
        }
    }

    MPI_CHECK(MPI_Init(&argc, &argv));
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &numprocs));

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

    options.min_message_size /= sizeof(double);
    if (options.min_message_size < MIN_MESSAGE_SIZE) {
        options.min_message_size = MIN_MESSAGE_SIZE;
    }

    bufsize = (options.max_message_size);
    if (allocate_memory_coll((void**)&sendbuf, bufsize, options.accel)) {
        fprintf(stderr, "Could Not Allocate Memory [rank %d]\n", rank);
        MPI_CHECK(MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE));
    }
    set_buffer(sendbuf, options.accel, 0, bufsize);
    copybuf = (double*)malloc(bufsize);

    if (allocate_memory_coll((void**)&recvbuf, bufsize, options.accel)) {
        fprintf(stderr, "Could Not Allocate Memory [rank %d]\n", rank);
        MPI_CHECK(MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE));
    }
    set_buffer(recvbuf, options.accel, 0, bufsize);
    
    print_preamble(rank);

    for (size = options.min_message_size; size * sizeof(double) <=
            options.max_message_size; size *= 2) {

        if (size > LARGE_MESSAGE_SIZE) {
            options.skip = options.skip_large;
            options.iterations = options.iterations_large;
        }

        int count = (int)(size / sizeof(double) + 0.0001);
    
        // Load the data into send_buffer (on all ranks)
        fill_buffer(copybuf, size);
    
        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

        timer = 0.0;

        for (i = 0; i < options.iterations + options.skip; i++) {
            
            cudaMemcpy(sendbuf, copybuf, size, cudaMemcpyHostToDevice);
            
            if (options.validate) {
                set_buffer_validation(sendbuf, recvbuf, size, options.accel, i);
                for (j = 0; j < options.warmup_validation; j++) {
                    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
                    MPI_CHECK(MPI_Allreduce(sendbuf, recvbuf, size, MPI_DOUBLE,
                                MPI_SUM, MPI_COMM_WORLD ));
                }
                MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
            }

            t_start = MPI_Wtime();
            MPI_CHECK(MPI_Allreduce(sendbuf, recvbuf, size, MPI_DOUBLE, MPI_SUM,
                        MPI_COMM_WORLD ));
            t_stop = MPI_Wtime();
            MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

            if (options.validate) {
                local_errors += validate_data(recvbuf, size, numprocs,
                        options.accel, i);
            }

            if (i >= options.skip) {
                timer += t_stop - t_start;
            }
        }
        latency = (double)(timer * 1e6) / options.iterations;

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
            print_stats_validate(rank, size * sizeof(double), avg_time, min_time,
                    max_time, errors);
        } else {
            print_stats(rank, size * sizeof(double), avg_time, min_time,
                    max_time);
        }

        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

        if (0 != errors) {
            break;
        }
    }

    free_buffer(sendbuf, options.accel);
    free_buffer(recvbuf, options.accel);
    free(copybuf);

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
