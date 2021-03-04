#!/bin/sh
#echo $1
#cat ./$1
#while [ "$(squeue -u u00222| wc -l)" -gt $4 ]
#do
#  sleep 1
#done
#cat ./$1 | grep -o "singularity exec .*" | sed 's/\/lustre\/gk77\/k77015\/.Singularity\/imgs\/LISA.simg/\/home\/u00222\/singularity\/images\/LISA.simg/' | while read line; do echo "-p p --gres=gpu:1 --mem=24GB" $line ; done | xargs -n 16 -I '{}' echo "srun {} guard"
cat ./$1 | grep -o "singularity exec .*" | sed 's/bin\/train.sh/bin\/evaluate-exported.sh/' | sed 's/\/lustre\/gk77\/k77015\/.Singularity\/imgs\/LISA.simg/\/home\/u00222\/singularity\/images\/LISA.simg/'  |sed 's/.log/.result/'|  while read line; do echo "-p $2 --gres=gpu:1 --time=1-06:00:00 --mem=24GB" $line ; done | xargs -n 17 -P 4 -L 1  -I '{}' bash -c "srun {}" #xargs -n 16 -P 4 -L 1 -I '{}' bash -c "{}"
#cat ./$1 | grep -o "singularity exec .*" | sed 's/\/lustre\/gk77\/k77015\/.Singularity\/imgs\/LISA.simg/\/home\/u00222\/singularity\/images\/LISA.simg/' | while read line; do echo "-p p --gres=gpu:1 --mem=24GB" $line ; done | xargs -n 16 -P 4 -L 1 srun

#cat ./$1 | grep -o "singularity exec (.)*"

