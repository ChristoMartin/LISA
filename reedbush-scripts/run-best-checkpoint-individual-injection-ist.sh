#!/bin/bash
echo $1
while [ "$(squeue -u u00222| wc -l)" -gt $4 ]
do
  sleep 1
done

#cat ./$1
#dropper() {
#   echo "${@:1:$#-2}";
#}
target_config=$(echo $2 |sed 's/\//\\\//g' )
echo $target_config
config_name=$(basename -- "$2" .conf)
#cat ./$1 | grep -o "singularity exec .*" | sed 's/\/lustre\/gk77\/k77015\/.Singularity\/imgs\/LISA.simg/\/home\/u00222\/singularity\/images\/LISA.simg/' | while read line; do echo "-p p --gres=gpu:1 --mem=24GB" $line ; done | xargs -n 16 -I '{}' echo "srun {} guard"
cat ./$1 | grep -o "singularity exec .*" | sed 's/bin\/train.sh/bin\/evaluate-exported.sh/' | sed -E  's/run-([0-9])/run-\1\/best_checkpoint/'  | sed 's/\/lustre\/gk77\/k77015\/.Singularity\/imgs\/LISA.simg/\/home\/u00222\/singularity\/images\/LISA.simg/'| sed 's/.log/.result/' |sed -E "s/config\/.+conf/$target_config/" | while read line; do echo "$line-$config_name"; done| while read line; do echo "-p $3 --gres=gpu:1 --time=1-06:00:00 --mem=24GB" $line ; done | xargs -n 17 -P1 -I '{}' sh -c "sleep 2; echo \"{}\"" |xargs  -n 17 -P 4 -L 1 -I '{}' bash -c "srun {}" # |# | while read line; do echo "$line-${$2##*/}"; done | xargs -n 13 -P 4 -L 1  -I '{}' echo '{}' #xargs -n 16 -P 4 -L 1 -I '{}' bash -c "{}"
#cat ./$1 | grep -o "singularity exec .*" | sed 's/\/lustre\/gk77\/k77015\/.Singularity\/imgs\/LISA.simg/\/home\/u00222\/singularity\/images\/LISA.simg/' | while read line; do echo "-p p --gres=gpu:1 --mem=24GB" $line ; done | xargs -n 16 -P 4 -L 1 srun

#cat ./$1 | grep -o "singularity exec (.)*"

