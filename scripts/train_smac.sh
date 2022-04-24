#!/bin/sh
env="StarCraft2"
map="3s5z"
algo="happo"
exp="mlp"
running_max=20
kl_threshold=0.06
echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for number in `seq ${running_max}`;
do
    echo "the ${number}-th running:"
    CUDA_VISIBLE_DEVICES=1 python train/train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --map_name ${map} --running_id ${number} --gamma 0.95 --n_training_threads 32 --n_rollout_threads 20 --num_mini_batch 1 --episode_length 160 --num_env_steps 20000000 --ppo_epoch 5 --stacked_frames 1 --kl_threshold ${kl_threshold} --use_value_active_masks --use_eval --add_center_xy --use_state_agent --share_policy
done
