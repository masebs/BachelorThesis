#include "cuda_runtime.h"
#include "ndzip_c_connector.h"
#include "stdio.h"
#include "stdlib.h"
#include <stdint.h>
#include <time.h>

int64_t millis()
{
  struct timespec now;
  timespec_get(&now, TIME_UTC);
  return ((int64_t)now.tv_sec) * 1000 + ((int64_t)now.tv_nsec) / 1000000;
}

int main()
{
  FILE *fileptr;
  char *buffer;
  size_t filelen = 2 << 15;
  
  NDZIP_API_sayHi(filelen, "Init");

  printf("%d\n", filelen);

  fileptr = fopen("/home/00/mschneider/ba/ndzip/test/num_plasma.fp64",
                  "rb"); // Open the file in binary mode

  buffer = (char *)malloc(filelen * sizeof(char)); // Enough memory for the file
  fread(buffer, filelen, sizeof(char), fileptr);   // Read in the entire file
  fclose(fileptr);

  double *data = (double *)buffer;

  size_t double_count = filelen / sizeof(double);

  double *device_uncompressed_buffer;
  cudaMalloc((void**)&device_uncompressed_buffer, filelen);

  uint64_t *device_compressed_buffer;
  cudaMalloc((void**)&device_compressed_buffer, filelen);

  cudaMemcpy(device_uncompressed_buffer, data, filelen, cudaMemcpyHostToDevice);

  int64_t start = millis();
  int count = 0;
  for(int ii = 0; ii < 1000; ii++) {
      count = NDZIP_API_compressBuffer(device_uncompressed_buffer, filelen,
                                       device_compressed_buffer);
  }
  cudaDeviceSynchronize();
  int64_t ende = millis();
  printf("compression time (ms): %ld length of compressed stream: %d\n",
         (ende - start), count * sizeof(double));

  double *device_decompressed_buffer;
  cudaMalloc((void**)&device_decompressed_buffer, filelen);

  int64_t start2 = millis();
  double *device_ptr = NDZIP_API_decompressBuffer(
      device_compressed_buffer, device_decompressed_buffer, filelen);
  int64_t ende2 = millis();
  cudaDeviceSynchronize();
  printf("decompression time (ms): %ld\n", (ende2 - start2));

  double *host_data = (double *)malloc(filelen);
  cudaMemcpy(host_data, device_ptr, filelen, cudaMemcpyDeviceToHost);

  for (int i = 0; i < 100; i++)
  {
    printf("%f ", host_data[i]);
    if (i % 15 == 0)
    {
      printf("\n");
    }
  }
  printf("\n");

  int buffer_consistent = 1;
  for (int i = 0; i < filelen / 8; i++)
  {
    if (data[i] != host_data[i])
    {
      buffer_consistent = 0;
    }
  }
  if (buffer_consistent)
  {
    printf("buffer before and after compression consistent\n");
  }
  else
  {
    printf("buffer before and after compression NOT consistent\n");
  }

  cudaFree(device_uncompressed_buffer);
  cudaFree(device_compressed_buffer);
  cudaFree(device_decompressed_buffer);
  free(buffer);
  free(host_data);
  return 0;
}
