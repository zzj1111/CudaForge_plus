export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export REWARD_CUDA_VISIBLE_DEVICES=6
export WANDB_API_KEY="b8f38344ec7231ee89baa74ef7209dd5a43df6b2"
export WANDB_ENTITY="mhong-university-of-minnesota"
#export Model_path="/code/hongpaul-sandbox/temp/CudaForge_plus/verl/data/Qwen3_8b"
#export Model_path="/home/zha00175/data/zha00175/Qwen3-30B-A3B"
export Model_path="cognition-ai/Kevin-32B"

export RAY_DEDUP_LOGS=0
export HYDRA_FULL_ERROR=1
export RAY_BACKEND_LOG_LEVEL=debug  # 有些版本支持

max_response_length=16384

loss_mode=grpo

project_name=CudaForge_RL
exp_name="d0107r1_GSPO_level123_kevin32B"


CKPTS_DIR=/code/hongpaul-sandbox/temp/CudaForge_plus/${project_name}/${exp_name}


source /code/hongpaul-sandbox/cuda/miniconda3/bin/activate
conda activate /code/hongpaul-sandbox/cuda/miniconda3/envs/cuda

mkdir -p logs


# export VLLM_ATTENTION_BACKEND=XFORMERS

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
 trainer.project_name=CudaForge_RL \
 algorithm.adv_estimator=grpo \
 data.train_files=./dataset/CudaForge/train.parquet \
 data.val_files=./dataset/CudaForge/test.parquet \
 data.train_batch_size=16 \
 data.max_prompt_length=8192 \
 data.max_response_length=16384 \
 actor_rollout_ref.model.lora_rank=128 \
 actor_rollout_ref.model.lora_alpha=32 \
 actor_rollout_ref.model.target_modules=all-linear \
 actor_rollout_ref.model.path=$Model_path \
 actor_rollout_ref.actor.optim.lr=3e-5 \
 actor_rollout_ref.actor.ppo_mini_batch_size=8 \
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
 actor_rollout_ref.rollout.name=vllm \
 actor_rollout_ref.rollout.n=8 \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
 actor_rollout_ref.rollout.tensor_model_parallel_size=8 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
 actor_rollout_ref.actor.fsdp_config.param_offload=True \
 actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
 actor_rollout_ref.rollout.temperature=0.6 \
 actor_rollout_ref.rollout.top_k=20 \
 actor_rollout_ref.rollout.top_p=0.95 \
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
 trainer.n_gpus_per_node=8 \
 trainer.nnodes=1 \
 trainer.default_local_dir="${CKPTS_DIR}" \
 trainer.save_freq=20 \
 trainer.test_freq=100 \
 trainer.logger='["console","wandb"]' \
 trainer.project_name="DAPO" \
 trainer.experiment_name="d0107r1_GRPO_topp080_32B" \
 trainer.total_epochs=10 \
