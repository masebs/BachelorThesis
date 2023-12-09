#define BENCHMARK "OSU MPI%s Broadcast Latency Test"
/*
 * Copyright (C) 2002-2021 the Network-Based Computing Laboratory
 * (NBCL), The Ohio State University.
 *
 * Contact: Dr. D. K. Panda (panda@cse.ohio-state.edu)
 *
 * For detailed copyright and licensing information, please refer to the
 * copyright file COPYRIGHT in the top level OMB directory.
 */
#include "cuda_runtime.h"
#include "ndzip_c_connector.h"
#include <osu_util_mpi.h>

int memory_buffer = 33554432;

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

int checkBuffer(double *device_ptr, double *host_pointer, int size) {

  double *host_buffer = (double *)malloc(size);
  cudaMemcpy(host_buffer, device_ptr, size, cudaMemcpyDeviceToHost);
  int legit = 1;

  for (int i = 0; i < size / 8; i++) {
    if (host_buffer[i] != host_pointer[i]) {
      legit = 0;
    }
  }
  free(host_buffer);

  return legit;
}

int main(int argc, char *argv[]) {

  void *d_compression_buffer_mem;
  void *d_decompression_buffer_mem;
  void *d_send_buffer_mem;
  void *d_receive_buffer_mem;

  double *d_compression_buffer;
  double *d_decompression_buffer;
  double *d_send_buffer;
  double *d_receive_buffer;

  double *send_buffer;
  double *receive_buffer;

  cudaMalloc(&d_compression_buffer_mem, memory_buffer);
  cudaMalloc(&d_decompression_buffer_mem, memory_buffer);
  cudaMalloc(&d_send_buffer_mem, memory_buffer);
  cudaMalloc(&d_receive_buffer_mem, memory_buffer);

  send_buffer = (double *)malloc(memory_buffer);
  receive_buffer = (double *)malloc(memory_buffer);

  d_compression_buffer = d_compression_buffer_mem;
  d_decompression_buffer = d_decompression_buffer_mem;
  d_send_buffer = d_send_buffer_mem;
  d_receive_buffer = d_receive_buffer_mem;

  int i = 0, j, rank, size;
  int numprocs;
  double avg_time = 0.0, max_time = 0.0, min_time = 0.0;
  double latency = 0.0, t_start = 0.0, t_stop = 0.0;
  double timer = 0.0;
  char *buffer = NULL;
  int po_ret;
  int errors = 0, local_errors = 0;
  options.bench = COLLECTIVE;
  options.subtype = BCAST;

  void *h_data_buffer;

  set_header(HEADER);
  set_benchmark_name("osu_bcast");
  po_ret = process_options(argc, argv);

  if (PO_OKAY == po_ret && NONE != options.accel) {
    if (init_accel()) {
      fprintf(stderr, "Error initializing device\n");
      exit(EXIT_FAILURE);
    }
  }

  MPI_CHECK(MPI_Init(&argc, &argv));
  MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &numprocs));
  
  NDZIP_API_sayHi(memory_buffer, "Init hello");

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
      fprintf(stderr,
              "Warning! Increase the Max Memory Limit to be able"
              " to run up to %ld bytes.\n"
              " Continuing with max message size of %ld bytes\n",
              options.max_message_size, options.max_mem_limit);
    }
    options.max_message_size = options.max_mem_limit;
  }

  if (allocate_memory_coll((void **)&buffer, options.max_message_size,
                           options.accel)) {
    fprintf(stderr, "Could Not Allocate Memory [rank %d]\n", rank);
    MPI_CHECK(MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE));
  }
  set_buffer(buffer, options.accel, 1, options.max_message_size);

  print_preamble(rank);

  for (size = options.min_message_size; size <= options.max_message_size;
       size *= 2) {
    if (size > LARGE_MESSAGE_SIZE) {
      options.skip = options.skip_large;
      options.iterations = options.iterations_large;
    }

    fill_buffer(send_buffer, size);
    cudaMemcpy(d_send_buffer, send_buffer, size, cudaMemcpyHostToDevice);

    timer = 0.0;

    for (i = 0; i < options.iterations + options.skip; i++) {
      if (options.validate) {
        set_buffer_validation(buffer, NULL, size, options.accel, i);
        for (j = 0; j < options.warmup_validation; j++) {
          MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
          MPI_CHECK(MPI_Bcast(buffer, size, MPI_CHAR, 0, MPI_COMM_WORLD));
        }
        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
      }

      int compressed_length = 0;

      // always clear receive buffer before routine
      memset(receive_buffer, 0, memory_buffer);
      cudaMemcpy(d_receive_buffer, receive_buffer, size,
                 cudaMemcpyHostToDevice);

      t_start = MPI_Wtime();

      if (rank == 0) {
        // compress buffer done
        // bcast compressed buffer size (MPI_INT) done
        // bcast compressed buffer (MPI_DOUBLE)
        compressed_length = NDZIP_API_compressBuffer(
            d_send_buffer, size, (long unsigned int *)d_compression_buffer);

        MPI_CHECK(MPI_Bcast(&compressed_length, 1, MPI_INT, 0, MPI_COMM_WORLD));

        MPI_CHECK(MPI_Bcast(d_compression_buffer, compressed_length, MPI_DOUBLE,
                            0, MPI_COMM_WORLD));
      } else {

        // receive bcast compressed size (MPI_INT) done
        // receive bcast compressed buffer (MPI_DOUBLE) done
        // decompress buffer

        MPI_CHECK(MPI_Bcast(&compressed_length, 1, MPI_INT, 0, MPI_COMM_WORLD));

        MPI_CHECK(MPI_Bcast(d_receive_buffer, compressed_length, MPI_DOUBLE, 0,
                            MPI_COMM_WORLD));

        double *buffer_ptr =
            NDZIP_API_decompressBuffer((long unsigned int *)d_receive_buffer,
                                       d_decompression_buffer, size);

        // validate buffer
        // if (!checkBuffer(buffer_ptr, send_buffer, size)) {
        //   printf("corrupt buffer");
        // }
      }

      cudaDeviceSynchronize();
      t_stop = MPI_Wtime();
      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

      if (options.validate) {
        local_errors += validate_data(buffer, size, numprocs, options.accel, i);
      }

      if (i >= options.skip) {
        timer += t_stop - t_start;
      }
    }

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    latency = (timer * 1e6) / options.iterations;

    MPI_CHECK(MPI_Reduce(&latency, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0,
                         MPI_COMM_WORLD));
    MPI_CHECK(MPI_Reduce(&latency, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0,
                         MPI_COMM_WORLD));
    MPI_CHECK(MPI_Reduce(&latency, &avg_time, 1, MPI_DOUBLE, MPI_SUM, 0,
                         MPI_COMM_WORLD));
    avg_time = avg_time / numprocs;

    if (options.validate) {
      MPI_CHECK(MPI_Allreduce(&local_errors, &errors, 1, MPI_INT, MPI_SUM,
                              MPI_COMM_WORLD));
    }

    if (options.validate) {
      print_stats_validate(rank, size, avg_time, min_time, max_time, errors);
    } else {
      print_stats(rank, size, avg_time, min_time, max_time);
    }
    if (0 != errors) {
      break;
    }
  }

  free_buffer(buffer, options.accel);

  MPI_CHECK(MPI_Finalize());

  if (NONE != options.accel) {
    if (cleanup_accel()) {
      fprintf(stderr, "Error cleaning up device\n");
      exit(EXIT_FAILURE);
    }
  }

  if (0 != errors && options.validate && 0 == rank) {
    fprintf(stdout,
            "DATA VALIDATION ERROR: %s exited with status %d on"
            " message size %d.\n",
            argv[0], EXIT_FAILURE, size);
    exit(EXIT_FAILURE);
  }

  cudaFree(d_compression_buffer_mem);
  cudaFree(d_decompression_buffer_mem);
  cudaFree(d_send_buffer_mem);
  cudaFree(d_receive_buffer_mem);
  free(send_buffer);
  free(receive_buffer);
  return EXIT_SUCCESS;
}

/* vi: set sw=4 sts=4 tw=80: */
