#!/bin/sh
#PBS -q h-regular
#PBS -l select=1
#PBS -W group_list=gk77
#PBS -l walltime=18:00:00
#PBS -N LISA-DM
#PBS -j oe
#PBS -M christopher@orudo.cc
#PBS -m abe

export PATH=$PBS_O_PATH:$PATH

cd $PBS_O_WORKDIR

module add cuda9/9.0.176-cuDNN7.1.4 singularity/2.5.1 
singularity exec --nv /lustre/gk77/k77015/.Singularity/imgs/LISA-1.9.simg bin/train.sh config/conll05-lisa-dm.conf --save_dir .model-lisa-dm
