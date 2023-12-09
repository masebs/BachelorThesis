#!/bin/bash -x
#SBATCH --account=<PROJECT>
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --output=log-allreduce-jusuf.out
#SBATCH --error=log-allreduce-jusuf.err
#SBATCH --time=00:03:00
#SBATCH --partition=gpus

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/p/software/jusuf/stages/2020/software/Boost/1.74.0-GCCcore-10.3.0-nompi/lib64/:/p/software/jusuf/stages/2020/software/Boost/1.74.0-GCCcore-10.3.0-nompi/lib:/p/project/<PROJECT>/schneider11/ndzip_c_library

srun $PROJECT/schneider11/osu_benchmark_v6.0/osu_allreduce -m 8:$((2**25)) -i 200 -x 50 > log-allreduce_jusuf.log &
pid1=$!
wait $pid1
srun $PROJECT/schneider11/osu_benchmark_v6.0/osu_allreduce_orig -m 8:$((2**25)) -i 200 -x 50 > log-allreduce_orig_jusuf.log &
pid2=$!
wait $pid2
#srun $PROJECT/schneider11/osu_benchmark_v6.0/osu_allreduce_cudaTiming -m 8:$((2**25)) -i 200 -x 50 > log-allreduce_cudaTiming_jusuf.log &
srun $PROJECT/schneider11/osu_benchmark_v6.0/osu_allreduce_woComp -m 8:$((2**25)) -i 200 -x 50 > log-allreduce_woComp_jusuf.log &
pid3=$!
wait $pid3

