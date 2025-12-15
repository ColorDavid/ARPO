# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
PPO config
"""

import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Optional, Tuple

from ..workers.config import WorkerConfig


def recursive_post_init(dataclass_obj):
    if hasattr(dataclass_obj, "post_init"):
        dataclass_obj.post_init()

    for attr in fields(dataclass_obj):
        if is_dataclass(getattr(dataclass_obj, attr.name)):
            recursive_post_init(getattr(dataclass_obj, attr.name))


@dataclass
class DataConfig:
    train_files: str = ""
    val_files: str = ""
    prompt_key: str = "prompt"
    answer_key: str = "answer"
    image_key: str = "images"
    max_prompt_length: int = 512
    max_response_length: int = 512
    rollout_batch_size: int = 512
    val_batch_size: int = -1
    format_prompt: Optional[str] = None
    shuffle: bool = True
    seed: int = 1
    max_pixels: int = 4194304
    min_pixels: int = 262144


@dataclass
class AlgorithmConfig:
    gamma: float = 1.0
    lam: float = 1.0
    adv_estimator: str = "grpo"
    disable_kl: bool = False
    use_kl_loss: bool = False
    kl_penalty: str = "kl"
    kl_coef: float = 1e-3
    kl_type: str = "fixed"
    kl_horizon: float = 0.0
    kl_target: float = 0.0
    enable_replay: bool = False


@dataclass
class TrainerConfig:
    total_episodes: int = 10
    max_steps: Optional[int] = None
    project_name: str = "easy_r1"
    experiment_name: str = "demo"
    logger: Tuple[str] = ("console", "wandb")
    nnodes: int = 1
    n_gpus_per_node: int = 8
    critic_warmup: int = 0
    val_freq: int = -1
    val_before_train: bool = True
    val_only: bool = False
    val_generations_to_log: int = 0
    save_freq: int = -1
    save_limit: int = -1
    save_checkpoint_path: Optional[str] = None
    load_checkpoint_path: Optional[str] = None

    def post_init(self):
        if self.save_checkpoint_path is None:
            self.save_checkpoint_path = os.path.join("checkpoints", self.project_name, self.experiment_name)

@dataclass
class EnvConfig:
    num_envs: int = 32
    max_steps: int = 15
    screen_size: Tuple[int, int] = (1920, 1080)
    # Whether to use HTTP-based remote OSWorld env (EnvWorkerRemote) instead of local DesktopEnv (EnvWorker)
    use_remote_env: bool = False
    # Remote environment configuration, used only when use_remote_env=True.
    # Example:
    # remote_env_config = {
    #     "base_url": "http://<remote-ip>",
    #     "manager_port": 8000,  # 或者直接配置 "env_port"
    #     "vm_name": "Ubuntu.qcow2",
    #     "os_type": "Ubuntu",
    # }
    remote_env_config: Optional[dict] = None


@dataclass
class AbsoluteZeroConfig:
    """Configuration for Absolute Zero style task proposal and harm evaluation."""
    # General settings
    enabled: bool = False
    proposal_frequency: int = 5  # Propose new tasks every N steps
    seed_task_ratio: float = 0.7  # Ratio of seed tasks vs proposed tasks
    max_proposed_tasks_cache: int = 100
    
    # Harm task proposal settings
    enable_harm_proposal: bool = True
    harm_task_ratio: float = 0.2  # Ratio of harm tasks in training batch
    harm_temperature: float = 0.8  # Temperature for harm task generation
    harm_types: Optional[Tuple[str, ...]] = None  # Types of harm to generate, None = all
    
    # Regular task proposal settings
    enable_task_proposal: bool = True
    task_temperature: float = 0.7  # Temperature for task generation
    proposal_types: Optional[Tuple[str, ...]] = None  # Types of proposals, None = all
    
    # Learnability reward settings
    enable_learnability_reward: bool = True
    learnability_threshold: float = 0.3  # Threshold for filtering tasks
    
    # Learnability validation settings (using real environment interaction)
    # NOTE: These settings are for internal configuration (unified mode always used)
    # In unified mode, learnability is checked using training rollout results (rollout.n samples)
    enable_learnability_validation: bool = True  # Whether to validate tasks using real env
    learnability_num_samples: int = 5  # Number of times to execute task for validation (n)
                                       # Only used in non-unified mode; in unified mode, rollout.n is used
    learnability_max_attempts: int = 3  # Max attempts to find a learnable task
    
    # Safety score filtering (dual-dimension filtering)
    # A task is learnable if: min_safe_ratio <= safe_ratio <= max_safe_ratio
    # where safe_ratio = count(safety_score > safety_threshold) / n
    learnability_safety_threshold: float = 0.5  # Threshold for considering a response "safe"
    learnability_min_safe_ratio: float = 0.3  # Min ratio of safe responses to be learnable
    learnability_max_safe_ratio: float = 0.7  # Max ratio of safe responses to be learnable
    
    # Completion score filtering (dual-dimension filtering)
    # A task is learnable if: min_incomplete_ratio <= incomplete_ratio <= max_incomplete_ratio
    # where incomplete_ratio = count(completion_score < completion_threshold) / n
    # For harm tasks with evaluator.func="infeasible", NOT completing is the correct behavior
    learnability_completion_threshold: float = 0.5  # Threshold for considering a task "completed"
    learnability_min_incomplete_ratio: float = 0.3  # Min ratio of incomplete responses to be learnable
    learnability_max_incomplete_ratio: float = 0.7  # Max ratio of incomplete responses to be learnable
    
    # Legacy settings (kept for backward compatibility)
    learnability_min_harmful_ratio: float = 0.3  # Deprecated: use learnability_min_safe_ratio
    learnability_max_harmful_ratio: float = 0.7  # Deprecated: use learnability_max_safe_ratio
    
    # Safety evaluation settings
    safety_reward_weight: float = 1.0  # Weight for safety score in reward
    task_completion_weight: float = 0.5  # Weight for task completion in reward
    
    # Completion score evaluator type
    # "llm" - Use LLM evaluator's task_completion_score (semantic understanding)
    # "rule" - Use rule-based evaluator's eval_result (state-based checking)
    completion_evaluator_type: str = "llm"  # "llm" or "rule"
    
    # Unified reward and filtering settings
    # When enabled, the training rollout results are used for both:
    # Maximum number of repropose attempts when entire question group needs repropose
    max_repropose_attempts: int = 3
    
    # Debug settings
    # When enabled, safety_score and completion_score will be randomly generated around 0.5
    # This is useful for debugging without calling LLM evaluator
    debug_random_scores: bool = False  # If True, use random scores instead of LLM evaluation


@dataclass
class PPOConfig:
    data: DataConfig = field(default_factory=DataConfig)
    worker: WorkerConfig = field(default_factory=WorkerConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    absolute_zero: AbsoluteZeroConfig = field(default_factory=AbsoluteZeroConfig)

    def post_init(self):
        self.worker.rollout.prompt_length = self.data.max_prompt_length
        self.worker.rollout.response_length = self.data.max_response_length
        self.worker.actor.disable_kl = self.algorithm.disable_kl
        self.worker.actor.use_kl_loss = self.algorithm.use_kl_loss
        self.worker.actor.kl_penalty = self.algorithm.kl_penalty
        self.worker.actor.kl_coef = self.algorithm.kl_coef
        
        # Validate AbsoluteZero config dependencies
        self._validate_absolute_zero_config()
    
    def _validate_absolute_zero_config(self):
        """Validate AbsoluteZero configuration dependencies."""
        az = self.absolute_zero
        # No validation needed - unified mode is always used

    def deep_post_init(self):
        recursive_post_init(self)

    def to_dict(self):
        return asdict(self)
