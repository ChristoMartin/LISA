#!/usr/bin/env bash

config_file=$1

source ${config_file}

params=${@:2}

echo "Using CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES

transition_stats=$data_dir/transition_probs.tsv

echo "python3 src/train.py --train_files $train_files   --dev_files $dev_files   --transition_stats $transition_stats   --data_config $data_config --attention_configs $attention_configs  --model_configs $model_configs   --task_configs $task_configs   --layer_configs $layer_configs   --best_eval_key $best_eval_key   $params"
echo $attention_configs
if ! [ -z "$attention_configs" ]
then
  additional_params="$additional_params --attention_configs $attention_configs"
  echo $additional_params
fi
if ! [ -z "$discounting" ]
then
  additional_params="$additional_params --hparams special_attention_mode=discounting"
  echo $additional_params
fi
#echo "python3 src/train.py --train_files $train_files   --dev_files $dev_files   --transition_stats $transition_stats   --data_config $data_config --attention_configs $attention_configs  --model_configs $model_configs   --task_configs $task_configs   --layer_configs $layer_configs   --best_eval_key $best_eval_key   $params $additional_params "

python3 src/train.py \
--train_files $train_files \
--dev_files $dev_files \
--transition_stats $transition_stats \
--data_config $data_config \
--model_configs $model_configs \
--task_configs $task_configs \
--layer_configs $layer_configs \
--best_eval_key $best_eval_key \
$params \
$additional_params



#if [ -z "$attention_configs" ]
#then
#  python3 src/train.py \
#  --train_files $train_files \
#  --dev_files $dev_files \
#  --transition_stats $transition_stats \
#  --data_config $data_config \
#  --model_configs $model_configs \
#  --task_configs $task_configs \
#  --layer_configs $layer_configs \
#  --best_eval_key $best_eval_key \
#  $params
#else
#  python3 src/train.py \
#  --train_files $train_files \
#  --dev_files $dev_files \
#  --transition_stats $transition_stats \
#  --data_config $data_config \
#  --model_configs $model_configs \
#  --task_configs $task_configs \
#  --layer_configs $layer_configs \
#  --attention_configs $attention_configs \
#  --best_eval_key $best_eval_key \
#  $params
#fi
#--num_gpus 2\


