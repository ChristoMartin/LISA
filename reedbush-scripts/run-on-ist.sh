#!/bin/sh
echo $1
#cat ./$1
while [ "$(squeue -u u00222| wc -l)" -gt 10 ]
do
  sleep 1
done
cat ./$1 | grep -o "singularity exec .*" | sed 's/\/lustre\/gk77\/k77015\/.Singularity\/imgs\/LISA.simg/\/home\/u00222\/singularity\/images\/LISA.simg/' | sed 's/&>/>/' | while read line; do echo "-p p --gres=gpu:1 --mem=24GB" $line "2>&1 &"; done
cat ./$1 | grep -o "singularity exec .*" | sed 's/\/lustre\/gk77\/k77015\/.Singularity\/imgs\/LISA.simg/\/home\/u00222\/singularity\/images\/LISA.simg/' | sed 's/&>/>/' | while read line; do echo "-p p --gres=gpu:1 --mem=24GB" $line "2>&1 &"; done | xargs -n 18 -P 4 -L 1 srun

#cat ./$1 | grep -o "singularity exec (.)*"

