#!/bin/sh
#PBS -q h-regular
#PBS -l select=1
#PBS -W group_list=gk77
#PBS -l walltime=18:00:00
#PBS -N LISA-GOLD
#PBS -j oe
#PBS -M christopher@orudo.cc
#PBS -m abe

export PATH=$PBS_O_PATH:$PATH

cd $PBS_O_WORKDIR

module add cuda9/9.0.176-cuDNN7.1.4 singularity/2.5.1 
{
CUDA_VISIBLE_DEVICES=0 singularity exec --nv /lustre/gk77/k77015/.Singularity/imgs/LISA.simg bin/train.sh config/conll05-lisa-gold-parse.conf --save_dir .model/.model-lisa-gold/run-1 --num_gpus 1
} &
{
pause 600
CUDA_VISIBLE_DEVICES=1 singularity exec --nv /lustre/gk77/k77015/.Singularity/imgs/LISA.simg bin/train.sh config/conll05-lisa-gold-parse.conf --save_dir .model/.model-lisa-gold/run-2 --num_gpus 1
} &
wait
