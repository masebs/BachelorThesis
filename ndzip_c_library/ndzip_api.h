#ifndef NDZIP_API_H
#define NDZIP_API_H

#include <stdint.h>

#ifdef USE_FLOAT
#define TYPE float
#define INTTYPE uint32_t
#else
#define TYPE double
#define INTTYPE uint64_t
#endif

#include <stdlib.h>

class NDZIP_API {
public:
  NDZIP_API(int buffersize);
  ~NDZIP_API();
  void sayHi(const char *name);
  int compress_buffer(TYPE *d_buffer, size_t buffer_size,
                      INTTYPE *d_compressed_buffer);
  TYPE *decompress_buffer(INTTYPE *compressed_device_buffer,
                            TYPE *uncompressed_device_buffer,
                            size_t buffer_size);

private:
//   TYPE *buffer;
  uint32_t *d_compressed_buffer_size; // in bytes
  ndzip::extent size;
  ndzip::compressor_requirements req;
  std::unique_ptr<ndzip::cuda_compressor<TYPE>> cuda_compressor;
  std::unique_ptr<ndzip::cuda_decompressor<TYPE>> cuda_decompressor;
};

#endif
