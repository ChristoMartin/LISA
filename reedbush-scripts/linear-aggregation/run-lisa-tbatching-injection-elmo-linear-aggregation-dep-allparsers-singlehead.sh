#!/bin/sh
#PBS -q h-regular
#PBS -l select=1
#PBS -W group_list=gk77
#PBS -l walltime=24:00:00
#PBS -N LISA-INJECTION-ELMO-DEP-ALLPARSERS-SINGLEHEAD
#PBS -j oe
#PBS -M christopher@orudo.cc
#PBS -m abe

export PATH=$PBS_O_PATH:$PATH

cd $PBS_O_WORKDIR

module add cuda9/9.0.176-cuDNN7.1.4 singularity/2.5.1 
{
CUDA_VISIBLE_DEVICES=0 singularity exec --nv /lustre/gk77/k77015/.Singularity/imgs/LISA.simg bin/train.sh config/lisa-aggregations-injection/conll05-linear-aggregation-dep-10layers-elmo-base-allparsers.conf --save_dir .model/.model-lisa-injection-linear-aggregation/run-dep-singlehead-allparsers-elmo-base/run-1 --num_gpus 1 &> .log/conll05-lisa-10layers-base-tbatching-injection-elmo-linear-aggregation-dep-allparsers-singlehead-run-1.log
} & 

{
sleep 160
CUDA_VISIBLE_DEVICES=1 singularity exec --nv /lustre/gk77/k77015/.Singularity/imgs/LISA.simg bin/train.sh config/lisa-aggregations-injection/conll05-linear-aggregation-dep-10layers-elmo-base-allparsers.conf --save_dir .model/.model-lisa-injection-linear-aggregation/run-dep-singlehead-allparsers-elmo-base/run-2 --num_gpus 1 &> .log/conll05-lisa-10layers-base-tbatching-injection-elmo-linear-aggregation-dep-allparsers-singlehead-run-2.log
} &
wait
