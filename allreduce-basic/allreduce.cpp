// Custom Allreduce implementation based on MPI_Sendrecv; 
// suitable for implementation of message compression
// Marc S. Schneider 2023

#include <mpi.h>
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <time.h>
#include <unistd.h> // only for usleep in main()

void allreduce(void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) {
  int rank, num_processes;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &num_processes);

  int datatype_size;
  MPI_Type_size(datatype, &datatype_size);
  
  // Create temporary copy of sendbuf (avoid to modify sendbuf)
  void* tempbuf = new char[count * datatype_size];
  std::memcpy(tempbuf, sendbuf, count * datatype_size);

  // Perform the butterfly-based Allreduce algorithm
  for (int mask = 1; mask < num_processes; mask <<= 1) {
    int partner = rank ^ mask;

    // Send and receive data
    MPI_Sendrecv(tempbuf, count, datatype, partner, 0,
                 recvbuf, count, datatype, partner, 0, comm, MPI_STATUS_IGNORE);

    // Reduce our previous value (tempbuf) and the newly received one (recvbuf)
    // Using MPI_Reduce_local saves us from manually implementing the different reduction 
    // operations like MPI_SUM, MPI_MAX, MPI_MIN; all of those are automatically supported
    for (int i = 0; i < count; i++) {
       MPI_Reduce_local((char*)recvbuf + i * datatype_size, (char*)tempbuf + i * datatype_size, 1, datatype, op);
    }
  }

  // Reduction complete; final result is in tempbuf; copy into recvbuf
  std::memcpy(recvbuf, tempbuf, count * datatype_size);
}

// Test with count == 1 (one single value per process)
void test_single_values(int rank) {
  std::srand(time(NULL) + rank);
  double send_value = rand() % 50;
  double recv_value, recv_valid;

  std::cout << "Process " << rank << ", my value: " << send_value << std::endl;
  
  allreduce(&send_value, &recv_value, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&send_value, &recv_valid, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  
  std::cout << "Process " << rank << ": Result = " << recv_value << ", valid result: " << recv_valid << std::endl << std::flush;
}

// Test with an array of array_size elements on each process
void test_array_values(int rank) {

  const int array_size = 5;
  std::srand(time(NULL) + rank);
  double sendbuf[array_size];
  for (int i = 0; i < array_size; i++)
  {
    sendbuf[i] = double(rand() % 20);
  }
  double recvbuf[array_size], recvbufvalid[array_size];

  // Print input
  std::cout << "Process " << rank << ", my values: ";
  for (int i = 0; i < array_size; i++) {
    std::cout << sendbuf[i]  << ", ";
  }
  std::cout << std::endl << std::flush;
  
  // Perform Allreduce to sum all elements in the array
  allreduce(sendbuf, recvbuf, array_size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(sendbuf, recvbufvalid, array_size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  
  MPI_Barrier(MPI_COMM_WORLD);
  // Print results
  if (rank == 0) {
    usleep(1000);
    std::cout << "Sum of elements after Allreduce: ";
    for (int i = 0; i < array_size; i++) {
      std::cout << recvbuf[i] << " ";
    }
    std::cout <<  std::endl << "Valid result from MPI_Allreduce: ";
    for (int i = 0; i < array_size; i++) {
      std::cout << recvbufvalid[i] << " ";
    }
    std::cout << std::endl << std::flush;
  }
  
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  test_single_values(rank);
  if(rank == 0) {
    usleep(1000);
    std::cout << "\n\n" << std::flush;
  }
  MPI_Barrier(MPI_COMM_WORLD);
  
  test_array_values(rank);
  MPI_Barrier(MPI_COMM_WORLD);
  if(rank == 0) {
    usleep(1000);
    std::cout << std::endl << std::flush;
  }

  MPI_Finalize();
  return 0;
}
