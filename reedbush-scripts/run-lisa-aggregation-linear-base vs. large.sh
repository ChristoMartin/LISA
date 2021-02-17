#!/bin/sh
#PBS -q h-regular
#PBS -l select=1
#PBS -W group_list=gk77
#PBS -l walltime=18:00:00
#PBS -N LISA-AGGREGATION-LINEAR-BASEvsLARGE
#PBS -j oe
#PBS -M christopher@orudo.cc
#PBS -m abe

export PATH=$PBS_O_PATH:$PATH

cd $PBS_O_WORKDIR

module add cuda9/9.0.176-cuDNN7.1.4 singularity/2.5.1 
{
CUDA_VISIBLE_DEVICES=0 singularity exec --nv /lustre/gk77/k77015/.Singularity/imgs/LISA.simg bin/train.sh config/lisa-aggregations/conll05-linear-aggregation-10layers-base.conf --save_dir .model/.model-lisa-aggregation-mean/run-base/run-1 --num_gpus 1 &> .log/conll05-lisa-linear-aggregation-10layers-base 
} & 

{
sleep 20 
CUDA_VISIBLE_DEVICES=1 singularity exec --nv /lustre/gk77/k77015/.Singularity/imgs/LISA.simg bin/train.sh config/lisa-aggregations/conll05-linear-aggregation-10layers-large.conf --save_dir .model/.model-lisa-aggregation-mean/run-large/run-1 --num_gpus 1 &> .log/conll05-lisa-linear-aggregation-10layers-large 
} &
wait
