set -x

# http_on
on
# export WANDB_API_KEY=7b02516e416592120b0f90b703aa84a121d8300a
export WANDB_API_KEY=6978e0382998718ed96fa7ef0522b79570c23c4d

# MODEL_PATH=ByteDance-Seed/UI-TARS-1.5-7B
# MODEL_PATH=/data/gaohongcheng/.cache/huggingface/hub/models--ByteDance-Seed--UI-TARS-1.5-7B/snapshots/683d002dd99d8f95104d31e70391a39348857f4e
# MODEL_PATH=ByteDance-Seed/UI-TARS-2B-SFT
# MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct
MODEL_PATH=/home/dongyinpeng/mnt/models/UI-TARS-1.5-7B

SYSTEM_PROMPT="""You are helpful assistant."""

NUM_GPUS=8
NUM_ENVS=16
ROLLOUT_N=8
HORIZON=6

((ROLLOUT_BSZ = NUM_ENVS/ROLLOUT_N))

#原始的参数
# worker.rollout.limit_images=5 \

# data.train_files=evaluation_examples/test_success_uitars1.5_wo_impossible.json \
# data.val_files=evaluation_examples/test_success_uitars1.5_wo_impossible.json \

# data.train_files=evaluation_examples/test_subset32.json \
# data.val_files=evaluation_examples/test_subset32.json \


export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export CUDA_LAUNCH_BLOCKING=0

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.format_prompt="${SYSTEM_PROMPT}" \
    data.train_files=evaluation_examples/test_success_uitars1.5_wo_impossible.json \
    data.val_files=evaluation_examples/test_success_uitars1.5_wo_impossible.json \
    data.max_prompt_length=64000 \
    data.max_response_length=2048 \
    data.max_pixels=2116800 \
    data.min_pixels=256 \
    data.rollout_batch_size=2 \
    worker.actor.fsdp.torch_dtype=bf16 \
    worker.actor.optim.strategy=adamw_bf16 \
    worker.actor.max_grad_norm=1.0 \
    worker.actor.optim.lr=1e-6 \
    worker.actor.optim.lr_warmup_ratio=0.05 \
    worker.actor.ulysses_sequence_parallel_size=1 \
    worker.actor.padding_free=true \
    worker.actor.ppo_epochs=1 \
    worker.actor.clip_ratio_low=0.2 \
    worker.actor.clip_ratio_high=0.3 \
    worker.actor.global_batch_size=1 \
    worker.actor.micro_batch_size_per_device_for_update=1 \
    worker.actor.micro_batch_size_per_device_for_experience=1 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.offload.offload_optimizer=true \
    worker.rollout.gpu_memory_utilization=0.6 \
    worker.rollout.temperature=1.0 \
    worker.rollout.n=$ROLLOUT_N \
    worker.rollout.limit_images=$HORIZON \
    worker.rollout.tensor_parallel_size=2 \
    worker.rollout.max_num_batched_tokens=128000 \
    algorithm.disable_kl=True \
    algorithm.kl_coef=0 \
    algorithm.enable_replay=True \
    env.num_envs=$NUM_ENVS \
    env.max_steps=$HORIZON \
    env.use_remote_env=true \
    env.remote_env_config.base_url=http://120.255.0.146 \
    env.remote_env_config.manager_port=9001 \
    trainer.project_name=Safety_GUI \
    trainer.experiment_name=safety_osharm_task \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.logger=["console","wandb"] \
    trainer.nnodes=1 \
    trainer.save_freq=16 \
    trainer.save_limit=3 \
    trainer.val_before_train=false \
    trainer.val_freq=16 \
    trainer.total_episodes=15

# worker.actor.offload.offload_optimizer=true \