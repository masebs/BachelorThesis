// Custom Allreduce implementation based on MPI_Sendrecv, including message compression
// by ndzip
// Marc S. Schneider 2023

#include <mpi.h>
#include <iostream>
#include <cstring>
#include <cstdlib>

#ifdef __cplusplus
extern "C" {
#endif
  
void allreduceComp(void *sendbuf, void *recvbuf, int count, char* op_str, void* comm_ptr) {
  int rank, num_processes;
  
  MPI_Comm comm = *((MPI_Comm*)comm_ptr); // hack to tranfer the communicator from mpi4py via void*
  MPI_Op op;
  if (!strcmp(op_str, "sum")) op = MPI_SUM;
  else if (!strcmp(op_str, "max")) op = MPI_MAX;
  else if (!strcmp(op_str, "min")) op = MPI_MIN;
  else return;
  
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &num_processes);

  int datatype_size = sizeof(float);

  // Perform the butterfly-based Allreduce algorithm
  for (int mask = 1; mask < num_processes; mask <<= 1) {
    int partner = rank ^ mask;

    // Send and receive data
    MPI_Sendrecv(sendbuf, count, MPI_FLOAT, partner, 0,
                recvbuf, count, MPI_FLOAT, partner, 0, comm, MPI_STATUS_IGNORE);

    // Reduce our previous value (sendbuf) and the newly received one (recvbuf)
    // Using MPI_Reduce_local saves us from manually implementing the different reduction 
    // operations like MPI_SUM, MPI_MAX, MPI_MIN; all of those are automatically supported
    for (int i = 0; i < count; i++) {
       MPI_Reduce_local((char*)recvbuf + i * datatype_size, (char*)sendbuf + i * datatype_size, 1, MPI_FLOAT, op);
    }
  }

  // Reduction complete; final result is in sendbuf; copy into recvbuf
  std::memcpy(recvbuf, sendbuf, count * datatype_size);
}

#ifdef __cplusplus
}
#endif
