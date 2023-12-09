#include <boost/program_options.hpp>
#include <cassert>
#include <cstddef>
//#include <cstdint>
#include <cuda_runtime.h>
#include <fstream>
#include <io/io.hh>
#include <iostream>
#include <memory>
#include <ndzip/cuda.hh>
#include "ndzip_api.h"

// msg_sppm.fp64 Bytes: 278995864 doubles: 34874483

// Macro and function for reporting CUDA errors
#define cudaErrorCheck(ans) { cudaAssert((ans), __FILE__, __LINE__, true); }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort) {
   if (code != cudaSuccess) 
   {
      fprintf(stderr, "cudaAssert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

NDZIP_API::NDZIP_API(int buffersize) {
  std::cout << "New Instance" << std::endl;
//   buffer = (TYPE *)malloc(100 * sizeof(TYPE));
  cudaErrorCheck( cudaMalloc(&d_compressed_buffer_size, sizeof(uint32_t)) );
  size = ndzip::extent(static_cast<ndzip::dim_type>(1));
  //size[0] = 34874483; // msg_sppm
  size[0] = buffersize;
  req = ndzip::compressor_requirements(size);
  cuda_compressor = ndzip::make_cuda_compressor<TYPE>(req, nullptr);
  cuda_decompressor = ndzip::make_cuda_decompressor<TYPE>(1, nullptr);
}

NDZIP_API::~NDZIP_API() {
  std::cout << "releasing allocated memory" << std::endl;
//   free(buffer);
  cudaErrorCheck( cudaFree(d_compressed_buffer_size) );
}

void NDZIP_API::sayHi(const char *name) {
  std::cout << "Hi " << name << std::endl;
}

int NDZIP_API::compress_buffer(TYPE *d_buffer, size_t buffer_size,
                               INTTYPE *d_compressed_buffer) {
  size[0] = buffer_size / sizeof(TYPE);
  cuda_compressor->compress(d_buffer, size, d_compressed_buffer,
                            d_compressed_buffer_size);

  uint32_t h_compressed_buffer_size;
  // copy 4-Byte uint32_t for compressed buffer length from device to host
  cudaErrorCheck( cudaMemcpy(&h_compressed_buffer_size, d_compressed_buffer_size,
             sizeof(uint32_t), cudaMemcpyDeviceToHost) );
  return (int)h_compressed_buffer_size;
}

TYPE *NDZIP_API::decompress_buffer(INTTYPE *compressed_device_buffer,
                                     TYPE *uncompressed_device_buffer,
                                     size_t buffer_size) {
  size[0] = buffer_size / sizeof(TYPE);
  cuda_decompressor->decompress(compressed_device_buffer,
                                uncompressed_device_buffer, size);

  return uncompressed_device_buffer;
}
