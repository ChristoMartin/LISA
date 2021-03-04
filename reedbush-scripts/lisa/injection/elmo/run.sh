#!/bin/sh
#PBS -q h-regular
#PBS -l select=1
#PBS -W group_list=gk77
#PBS -l walltime=18:00:00
#PBS -N LISA-HI-ELMO
#PBS -j oe
#PBS -M christopher@orudo.cc
#PBS -m abe

export PATH=$PBS_O_PATH:$PATH

cd $PBS_O_WORKDIR
SAVEDIR=.model/.model-lisa-injection/run-elmo-base
CONF=config/lisa-injection/conll05-lisa-elmo.conf
SINGULARITY_IMG=/lustre/gk77/k77015/.Singularity/imgs/LISA.simg

module add cuda9/9.0.176-cuDNN7.1.4 singularity/2.5.1
{
CUDA_VISIBLE_DEVICES=0 singularity exec --nv $SINGULARITY_IMG bin/train.sh $CONF --save_dir $SAVEDIR/run-1 --num_gpus 1  &> $SAVEDIR/run-1/train.log
} &

{
sleep 600
CUDA_VISIBLE_DEVICES=1 singularity exec --nv $SINGULARITY_IMG bin/train.sh $CONF --save_dir $SAVEDIR/run-2 --num_gpus 1  &> $SAVEDIR/run-2/train.log
} &
wait
