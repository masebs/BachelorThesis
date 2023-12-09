#!/bin/bash -x
#SBATCH --account=<PROJECT>
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --output=log-distributed.out
#SBATCH --error=log-distributed.err
#SBATCH --time=00:10:00
#SBATCH --partition=gpus

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/p/software/jusuf/stages/2020/software/Boost/1.74.0-GCCcore-10.3.0-nompi/lib64/:/p/software/jusuf/stages/2020/software/Boost/1.74.0-GCCcore-10.3.0-nompi/lib:/p/project/<PROJECT>/schneider11/ndzip_c_library

module --force purge
module load Stages/2020
module load GCC/10.3
module load OpenMPI/4.1.1 
module load CUDA
module load Python 
module load TensorFlow

#python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))" >& tf-gpus-before-env.log

srun python3 $PROJECT/schneider11/deep500/samples/distributed_training.py mavg >& log-distributed.log &

pid1=$!
wait $pid1

