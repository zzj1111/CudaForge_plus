export CUDA_VISIBLE_DEVICES=4,5,6,7
export REWARD_CUDA_VISIBLE_DEVICES=5
export WANDB_API_KEY="b8f38344ec7231ee89baa74ef7209dd5a43df6b2"
export WANDB_ENTITY="mhong-university-of-minnesota"
#export Model_path="/code/hongpaul-sandbox/temp/CudaForge_plus/verl/data/Qwen3_8b"
#export Model_path="/home/zha00175/data/zha00175/Qwen3-30B-A3B"
export Model_path="Qwen/Qwen3-8B"

export RAY_DEDUP_LOGS=0
export HYDRA_FULL_ERROR=1
export RAY_BACKEND_LOG_LEVEL=debug  # 有些版本支持

max_response_length=16384

loss_mode=gspo
loss_agg_mode="seq-mean-token-mean"

project_name=CudaForge_RL
exp_name="d1221r1_DAPO_kernelbenchlevel1_topp080"
source /code/hongpaul-sandbox/cuda/miniconda3/bin/activate
conda activate /code/hongpaul-sandbox/cuda/miniconda3/envs/cuda

CKPTS_DIR=/code/hongpaul-sandbox/temp/CudaForge_plus/${project_name}/${exp_name}


mkdir -p logs


# export VLLM_ATTENTION_BACKEND=XFORMERS

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
 trainer.project_name=CudaForge_RL \
 algorithm.adv_estimator=grpo \
 data.train_files=./dataset/CudaForge/Level1/train.parquet \
 data.val_files=./dataset/CudaForge/Level1/test.parquet \
 data.train_batch_size=16 \
 data.max_prompt_length=8192 \
 data.max_response_length=16384 \
 actor_rollout_ref.model.path=$Model_path \
 actor_rollout_ref.actor.optim.lr=1e-6 \
 actor_rollout_ref.actor.ppo_mini_batch_size=16 \
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
 actor_rollout_ref.rollout.name=vllm \
 actor_rollout_ref.rollout.n=8 \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
 actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
 actor_rollout_ref.actor.fsdp_config.param_offload=False \
 actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
 actor_rollout_ref.rollout.temperature=0.7 \
 actor_rollout_ref.rollout.top_k=20 \
 actor_rollout_ref.rollout.top_p=0.8 \
 actor_rollout_ref.actor.clip_ratio_low=0.2 \
 actor_rollout_ref.actor.clip_ratio_high=0.28 \
 critic.optim.lr=1e-5 \
 critic.model.path=$Model_path \
 critic.ppo_micro_batch_size_per_gpu=4 \
 reward_model.enable=False \
 reward_model.reward_manager=dapo \
 custom_reward_function.path=./verl/utils/reward_score/CudaForge.py \
 algorithm.kl_ctrl.kl_coef=0 \
 trainer.val_before_train=False \
 trainer.n_gpus_per_node=4 \
 trainer.nnodes=1 \
 trainer.default_local_dir="${CKPTS_DIR}" \
 trainer.save_freq=20 \
 trainer.test_freq=100 \
 trainer.logger='["console","wandb"]' \
 trainer.project_name="DAPO" \
 trainer.experiment_name="d1221r1_DAPO_kernelbenchlevel1_topp080" \
 trainer.total_epochs=20 \
