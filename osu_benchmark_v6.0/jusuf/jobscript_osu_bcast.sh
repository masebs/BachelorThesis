#!/bin/bash -x
#SBATCH --account=<PROJECT>
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --output=log-bcast-jusuf.out
#SBATCH --error=log-bcast-jusuf.err
#SBATCH --time=00:03:00
#SBATCH --partition=gpus

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/p/software/jusuf/stages/2020/software/Boost/1.74.0-GCCcore-10.3.0-nompi/lib64/:/p/software/jusuf/stages/2020/software/Boost/1.74.0-GCCcore-10.3.0-nompi/lib:/p/project/<PROJECT>/schneider11/ndzip_c_library

srun /p/project/<PROJECT>/schneider11/osu_benchmark_v6.0/osu_bcast -m 8:$((2**25)) -i 200 -x 50 > log-bcast_jusuf.log &
pid1=$!
wait $pid1
srun /p/project/<PROJECT>/schneider11/osu_benchmark_v6.0/osu_bcast_orig -m 8:$((2**25)) -i 200 -x 50 > log-bcast_orig_jusuf.log &
pid2=$!
wait $pid2
srun /p/project/<PROJECT>/schneider11/osu_benchmark_v6.0/osu_bcast_MV -m 8:$((2**25)) -i 200 -x 50 > log-bcast_MV_jusuf.log &
pid3=$!
wait $pid3
