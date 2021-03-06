#!/bin/sh
echo $1
#cat ./$1
while [ "$(ps -fA | grep evaluate_exported.py| wc -l)" -gt 2 ]
do
  sleep 1
done

#cat ./$1 | grep -o "singularity exec .*" | sed 's/\/lustre\/gk77\/k77015\/.Singularity\/imgs\/LISA.simg/\/home\/u00222\/singularity\/images\/LISA.simg/' | while read line; do echo "-p p --gres=gpu:1 --mem=24GB" $line ; done | xargs -n 16 -I '{}' echo "srun {} guard"
cat ./$1 | grep -o "CUDA_VISIBLE_DEVICES.*" | sed 's/bin\/train.sh/bin\/evaluate-exported.sh/'  |sed 's/.log/.result/'  | xargs -n 13 -P 4 -L 1  -I '{}' sh -c '{}' #xargs -n 16 -P 4 -L 1 -I '{}' bash -c "{}"
#cat ./$1 | grep -o "singularity exec .*" | sed 's/\/lustre\/gk77\/k77015\/.Singularity\/imgs\/LISA.simg/\/home\/u00222\/singularity\/images\/LISA.simg/' | while read line; do echo "-p p --gres=gpu:1 --mem=24GB" $line ; done | xargs -n 16 -P 4 -L 1 srun

#cat ./$1 | grep -o "singularity exec (.)*"

