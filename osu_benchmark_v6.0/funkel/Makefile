SPACKBIN:=/home/00/mschneider/ba/spackenv/.spack-env/view/bin
NDZIP:=/home/00/mschneider/ba/ndzip
LIB64:=/home/00/mschneider/ba/spackenv/.spack-env/view/lib64
LIB:=/home/00/mschneider/ba/spackenv/.spack-env/view/lib
INCL:=/home/00/mschneider/ba/spackenv/.spack-env/view/include
NDZIP_C_LIBRARY:=/home/00/mschneider/ba/ndzip_c_library
UTIL:=/home/00/mschneider/ba/original-osu-micro-benchmarks-6.0/c/util

all:  osu_bcast osu_bcast_orig osu_allreduce osu_allreduce_orig osu_bcast_MV osu_allreduce_woComp # osu_allreduce_cudaTiming

clean: 
	rm osu_bcast.o osu_bcast osu_bcast_orig.o osu_bcast_orig osu_allreduce.o osu_allreduce osu_allreduce_orig.o osu_allreduce_orig osu_bcast_MV.o osu_bcast_MV addKernel.o print_stats.o osu_allreduce_woComp.o osu_allreduce_woComp # osu_allreduce_cudaTiming.o osu_allreduce_cudaTiming
	
osu_bcast.o: osu_bcast.c
	$(SPACKBIN)/mpicc -I. -I$(NDZIP_C_LIBRARY) -I$(INCL) -I$(UTIL) -g -O2 -c -o osu_bcast.o osu_bcast.c
osu_bcast: osu_bcast.o 
	$(SPACKBIN)/mpic++ -g -O2 -Wl,-rpath=$(NDZIP)/build:$(NDZIP_C_LIBRARY):$(LIB) -o osu_bcast osu_bcast.o $(UTIL)/osu_util.o $(UTIL)/osu_util_mpi.o $(UTIL)/kernel.o $(NDZIP)/build/libndzip.so $(NDZIP)/build/libndzip-cuda.so -L$(LIB64) -L$(LIB) -L$(NDZIP_C_LIBRARY) -lndzip_api_c_connector -lcuda -lcudart

osu_bcast_orig.o: osu_bcast_orig.c
	$(SPACKBIN)/mpicc -I. -I$(NDZIP_C_LIBRARY) -I$(INCL) -I$(UTIL) -g -O2 -c -o osu_bcast_orig.o osu_bcast_orig.c
osu_bcast_orig: osu_bcast_orig.o
	$(SPACKBIN)/mpic++ -g -O2 -Wl,-rpath=$(NDZIP)/build:$(NDZIP_C_LIBRARY):$(LIB) -o osu_bcast_orig osu_bcast_orig.o $(UTIL)/osu_util.o $(UTIL)/osu_util_mpi.o $(UTIL)/kernel.o $(NDZIP)/build/libndzip.so $(NDZIP)/build/libndzip-cuda.so -L$(LIB64) -L$(LIB) -L$(NDZIP_C_LIBRARY) -lndzip_api_c_connector -lcuda -lcudart

osu_bcast_MV.o: osu_bcast_MV.c
	$(SPACKBIN)/mpicc -I. -I$(NDZIP_C_LIBRARY) -I$(INCL) -I$(UTIL) -g -O2 -c -o osu_bcast_MV.o osu_bcast_MV.c
osu_bcast_MV: osu_bcast_MV.o
	$(SPACKBIN)/mpic++ -g -O2 -Wl,-rpath=$(NDZIP)/build:$(NDZIP_C_LIBRARY):$(LIB) -o osu_bcast_MV osu_bcast_MV.o $(UTIL)/osu_util.o $(UTIL)/osu_util_mpi.o $(UTIL)/kernel.o $(NDZIP)/build/libndzip.so $(NDZIP)/build/libndzip-cuda.so -L$(LIB64) -L$(LIB) -L$(NDZIP_C_LIBRARY) -lndzip_api_c_connector -lcuda -lcudart

osu_allreduce.o: osu_allreduce.c
	$(SPACKBIN)/mpicc -I. -I$(NDZIP_C_LIBRARY) -I$(INCL) -I$(UTIL)  -g -O2 -c -o osu_allreduce.o osu_allreduce.c
addKernel.o: addKernel.cu
	$(SPACKBIN)/nvcc -ccbin $(SPACKBIN)/mpic++ -c -o addKernel.o addKernel.cu
print_stats.o: print_stats.c
	$(SPACKBIN)/mpicc -I. -I$(INCL) -I$(UTIL) -g -O2 -c -o print_stats.o print_stats.c
osu_allreduce: osu_allreduce.o addKernel.o print_stats.o
	$(SPACKBIN)/mpic++ -g -O2 -Wl,-rpath=$(NDZIP)/build:$(NDZIP_C_LIBRARY):$(LIB) -o osu_allreduce osu_allreduce.o addKernel.o print_stats.o $(UTIL)/osu_util.o $(UTIL)/osu_util_mpi.o $(UTIL)/kernel.o $(NDZIP)/build/libndzip.so $(NDZIP)/build/libndzip-cuda.so -L$(LIB64) -L$(LIB) -L$(NDZIP_C_LIBRARY) -lndzip_api_c_connector -lcuda -lcudart

osu_allreduce_woComp.o: osu_allreduce_woComp.c
	$(SPACKBIN)/mpicc -I. -I$(NDZIP_C_LIBRARY) -I$(INCL) -I$(UTIL)  -g -O2 -c -o osu_allreduce_woComp.o osu_allreduce_woComp.c
osu_allreduce_woComp: osu_allreduce_woComp.o addKernel.o print_stats.o
	$(SPACKBIN)/mpic++ -g -O2 -Wl,-rpath=$(NDZIP)/build:$(NDZIP_C_LIBRARY):$(LIB) -o osu_allreduce_woComp osu_allreduce_woComp.o addKernel.o print_stats.o $(UTIL)/osu_util.o $(UTIL)/osu_util_mpi.o $(UTIL)/kernel.o $(NDZIP)/build/libndzip.so $(NDZIP)/build/libndzip-cuda.so -L$(LIB64) -L$(LIB) -L$(NDZIP_C_LIBRARY) -lndzip_api_c_connector -lcuda -lcudart

#osu_allreduce_cudaTiming.o: osu_allreduce_cudaTiming.c
#	$(SPACKBIN)/mpicc -I. -I$(NDZIP_C_LIBRARY) -I$(INCL) -I$(UTIL)  -g -O2 -c -o osu_allreduce_cudaTiming.o osu_allreduce_cudaTiming.c
#osu_allreduce_cudaTiming: osu_allreduce_cudaTiming.o addKernel.o print_stats.o
#	$(SPACKBIN)/mpic++ -g -O2 -Wl,-rpath=$(NDZIP)/build:$(NDZIP_C_LIBRARY) -o osu_allreduce_cudaTiming osu_allreduce_cudaTiming.o addKernel.o print_stats.o $(UTIL)/osu_util.o $(UTIL)/osu_util_mpi.o $(UTIL)/kernel.o $(NDZIP)/build/libndzip.so $(NDZIP)/build/libndzip-cuda.so -L$(LIB64) -L$(LIB) -L$(NDZIP_C_LIBRARY) -lndzip_api_c_connector -lcuda -lcudart

osu_allreduce_orig.o: osu_allreduce_orig.c
	$(SPACKBIN)/mpicc -I. -I$(NDZIP_C_LIBRARY) -I$(INCL) -I$(UTIL) -g -O2 -c -o osu_allreduce_orig.o osu_allreduce_orig.c
osu_allreduce_orig: osu_allreduce_orig.o
	$(SPACKBIN)/mpic++ -g -O2 -Wl,-rpath=$(NDZIP)/build:$(NDZIP_C_LIBRARY):$(LIB) -o osu_allreduce_orig osu_allreduce_orig.o $(UTIL)/osu_util.o $(UTIL)/osu_util_mpi.o $(UTIL)/kernel.o $(NDZIP)/build/libndzip.so $(NDZIP)/build/libndzip-cuda.so -L$(LIB64) -L$(LIB) -L$(NDZIP_C_LIBRARY) -lndzip_api_c_connector -lcuda -lcudart

