#include <cstdlib>
#include <ndzip/cuda.hh>

#include "ndzip_api.h"
#include "ndzip_c_connector.h"

#ifdef __cplusplus
extern "C" {
#endif


static NDZIP_API *NDZIP_API_instance = NULL;

void lazyNDZIP_API(int buffersize) {
  if (NDZIP_API_instance == NULL) {
    NDZIP_API_instance = new NDZIP_API(buffersize);
  }
}

void NDZIP_API_sayHi(int buffersize, const char *name) {
  lazyNDZIP_API(buffersize);
  //NDZIP_API_instance->sayHi(name);
}

int NDZIP_API_compressBuffer(TYPE *d_buffer, int buffer_size,
                             INTTYPE *d_compressed_buffer) {
  lazyNDZIP_API(buffer_size);
  return NDZIP_API_instance->compress_buffer(d_buffer, buffer_size,
                                             d_compressed_buffer);
}

TYPE *NDZIP_API_decompressBuffer(INTTYPE *compressed_device_buffer,
                                   TYPE *uncompressed_device_buffer,
                                   int buffer_size) {
  lazyNDZIP_API(buffer_size);
  return NDZIP_API_instance->decompress_buffer(compressed_device_buffer, 
                                               uncompressed_device_buffer, buffer_size);
}

#ifdef __cplusplus
}
#endif
