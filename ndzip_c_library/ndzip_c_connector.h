#ifndef AAA_C_CONNECTOR_H
#define AAA_C_CONNECTOR_H

#include <stdint.h>

#ifdef USE_FLOAT
#define TYPE float
#define INTTYPE uint32_t
#else
#define TYPE double
#define INTTYPE uint64_t 
#endif

#ifdef __cplusplus
extern "C" {
#endif

void NDZIP_API_sayHi(int buffersize, const char *name);
int NDZIP_API_compressBuffer(TYPE *d_buffer, int buffer_size,
                             INTTYPE *d_compressed_buffer);
TYPE *NDZIP_API_decompressBuffer(INTTYPE *compressed_device_buffer,
                                   TYPE *uncompressed_device_buffer,
                                   int buffer_size);

#ifdef __cplusplus
}
#endif

#endif
