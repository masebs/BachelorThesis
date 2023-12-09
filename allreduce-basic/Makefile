MPIBIN:=/p/software/jusuf/stages/2020/software/OpenMPI/4.1.1-GCC-10.3.0/bin
NVCCBIN:=/p/software/jusuf/stages/2020/software/CUDA/11.3/bin
NDZIP:=/p/project/<PROJECT>/schneider11/ndzip
LIB64CUDA:=/p/software/jusuf/stages/2020/software/CUDA/11.3/lib64
# LIBCUDA:=/p/software/jusuf/stages/2020/software/CUDA/11.3/lib
LIB64BOOST:=/p/software/jusuf/stages/2020/software/Boost/1.74.0-GCCcore-10.3.0-nompi/lib64
LIBBOOST:=/p/software/jusuf/stages/2020/software/Boost/1.74.0-GCCcore-10.3.0-nompi/lib
INCL:=/p/software/jusuf/stages/2020/software/CUDA/11.3/include
NDZIP_C_LIBRARY:=/p/project/<PROJECT>/schneider11/ndzip_c_library
TYPE:=USE_FLOAT

all: allreduceComp

clean: 
	rm liballreduceComp.so allreduceComp.so sumKernel.so minKernel.so maxKernel.so

allreduceComp.so: allreduceComp.cpp
	$(MPIBIN)/mpicc -D$(TYPE) -I. -I$(NDZIP_C_LIBRARY) -I$(INCL) -g -O2 -c -shared -fPIC -o allreduceComp.so allreduceComp.cpp
sumKernel.so: sumKernel.cu
	$(NVCCBIN)/nvcc -ccbin $(MPIBIN)/mpic++ -c -shared -Xcompiler -fPIC,-D$(TYPE) -o sumKernel.so sumKernel.cu
minKernel.so: minKernel.cu
	$(NVCCBIN)/nvcc -ccbin $(MPIBIN)/mpic++ -c -shared -Xcompiler -fPIC,-D$(TYPE) -o minKernel.so minKernel.cu
maxKernel.so: maxKernel.cu
	$(NVCCBIN)/nvcc -ccbin $(MPIBIN)/mpic++ -c -shared -Xcompiler -fPIC,-D$(TYPE) -o maxKernel.so maxKernel.cu
allreduceComp: allreduceComp.so sumKernel.so minKernel.so maxKernel.so
	$(MPIBIN)/mpicc -g -O2 -Wl,-rpath=$(NDZIP)/build:$(NDZIP_C_LIBRARY):$(LIB64BOOST):$(LIBBOOST) -shared -o liballreduceComp.so allreduceComp.so sumKernel.so minKernel.so maxKernel.so $(NDZIP)/build/libndzip.so $(NDZIP_C_LIBRARY)/libndzip_api.so $(NDZIP)/build/libndzip-cuda.so $(LIB64BOOST)/libboost_thread-mt-x64.so.1.74.0 -L$(LIB64) -L$(LIB) -L$(NDZIP_C_LIBRARY) -lndzip_api_c_connector -lcuda -lcudart
