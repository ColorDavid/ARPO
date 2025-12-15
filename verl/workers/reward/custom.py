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
Custom reward manager with support for harm evaluation using LLM.

This module provides reward computation for various task types including:
- Math tasks (math_compute_score)
- R1V tasks (r1v_compute_score)
- Harm evaluation tasks (using LLM-based evaluation)

For harm tasks, the reward is computed by:
1. Sending the agent's trajectory to a third-party LLM (e.g., GPT-4)
2. The LLM evaluates safety and task completion
3. Scores are extracted from the LLM's response
"""

from collections import defaultdict
from typing import Callable, Dict, List, Tuple, TypedDict, Any, Optional

import torch
from transformers import PreTrainedTokenizer

from ...protocol import DataProto
from ...utils.reward_score import (
    math_compute_score,
    r1v_compute_score,
    harm_compute_score,
    create_harm_reward_function,
    harm_compute_score_from_trajectory,
)
from ...utils.reward_score.harm import (
    BaseHarmEvaluator,
    HarmEvaluatorConfig,
    OpenAIHarmEvaluator,
    VLLMHarmEvaluator,
    FunctionHarmEvaluator,
    create_harm_evaluator,
)
from .config import RewardConfig


class RewardScore(TypedDict):
    overall: float
    format: float
    accuracy: float


class HarmRewardScore(TypedDict):
    """Score structure for harm evaluation tasks."""
    overall: float
    safety: float
    task_completion: float
    format: float
    is_safe: bool
    reasoning: str


class CustomRewardManager:
    """
    Reward manager that supports multiple reward functions including LLM-based harm evaluation.
    
    For harm tasks, the reward is computed by:
    1. Extracting the agent's trajectory from the response
    2. Sending trajectory + harm_action to an LLM evaluator
    3. Parsing the LLM's safety assessment
    4. Computing weighted scores
    
    The LLM evaluator can be:
    - OpenAI API (GPT-4, etc.)
    - vLLM server
    - Custom function
    """
    
    def __init__(self, tokenizer: PreTrainedTokenizer, config: RewardConfig):
        self.config = config
        self.tokenizer = tokenizer
        
        # Set up the main score function
        if config.score_function == "math":
            self.compute_score: Callable[[str, str], Dict[str, float]] = math_compute_score
        elif config.score_function == "r1v":
            self.compute_score: Callable[[str, str], Dict[str, float]] = r1v_compute_score
        elif config.score_function == "harm":
            # Create harm reward function with configured weights
            self.compute_score = create_harm_reward_function(
                safety_weight=config.harm_config.safety_weight,
                task_completion_weight=config.harm_config.task_completion_weight,
                format_weight=config.harm_config.format_weight,
            )
        else:
            raise NotImplementedError(f"Unknown score function {config.score_function}.")
        
        # Set up harm-specific score function (for parsing pre-computed responses)
        self.harm_compute_score = create_harm_reward_function(
            safety_weight=config.harm_config.safety_weight,
            task_completion_weight=config.harm_config.task_completion_weight,
            format_weight=config.harm_config.format_weight,
        )
        
        # Initialize LLM evaluator for harm tasks
        self.harm_evaluator: Optional[BaseHarmEvaluator] = None
        if config.use_harm_reward_for_harm_tasks and config.harm_config.use_llm_evaluation:
            self._init_harm_evaluator()
    
    def _init_harm_evaluator(self):
        """Initialize the LLM-based harm evaluator."""
        eval_config = self.config.harm_config.evaluator_config
        
        # Create evaluator config
        harm_eval_config = HarmEvaluatorConfig(
            model_name=eval_config.model_name,
            api_key=eval_config.api_key,
            api_base=eval_config.api_base,
            temperature=eval_config.temperature,
            max_tokens=eval_config.max_tokens,
            timeout=eval_config.timeout,
            max_retries=eval_config.max_retries,
            retry_delay=eval_config.retry_delay,
            batch_size=eval_config.batch_size,
            use_async=eval_config.use_async,
            safety_weight=self.config.harm_config.safety_weight,
            task_completion_weight=self.config.harm_config.task_completion_weight,
            format_weight=self.config.harm_config.format_weight,
        )
        
        # Create evaluator based on type
        if eval_config.evaluator_type == "openai":
            self.harm_evaluator = OpenAIHarmEvaluator(harm_eval_config)
        elif eval_config.evaluator_type == "vllm":
            self.harm_evaluator = VLLMHarmEvaluator(harm_eval_config)
        else:
            print(f"Warning: Unknown evaluator type {eval_config.evaluator_type}, "
                  "falling back to response parsing mode")
            self.harm_evaluator = None
    
    def set_harm_evaluator(self, evaluator: BaseHarmEvaluator):
        """
        Set a custom harm evaluator.
        
        This allows injecting a custom evaluator for testing or
        using a different LLM backend.
        
        Args:
            evaluator: The harm evaluator to use
        """
        self.harm_evaluator = evaluator

    def _is_harm_task(self, data_item: Any) -> bool:
        """Check if a data item is a harm evaluation task."""
        non_tensor = getattr(data_item, 'non_tensor_batch', {})
        if isinstance(non_tensor, dict):
            return non_tensor.get("is_harm_task", False)
        return False
    
    def _get_harm_action(self, data_item: Any) -> Optional[str]:
        """Get the harm_action from a data item."""
        non_tensor = getattr(data_item, 'non_tensor_batch', {})
        if isinstance(non_tensor, dict):
            return non_tensor.get("harm_action", None)
        return None
    
    def _get_trajectory(self, data_item: Any) -> Optional[str]:
        """
        Get the trajectory from a data item.
        
        The trajectory is the agent's action history, which may be stored
        in different fields depending on the data format.
        """
        non_tensor = getattr(data_item, 'non_tensor_batch', {})
        if isinstance(non_tensor, dict):
            # Try different possible field names
            trajectory = non_tensor.get("trajectory")
            if trajectory:
                return trajectory
            
            # Try action_history
            action_history = non_tensor.get("action_history")
            if action_history:
                if isinstance(action_history, list):
                    return "\n".join(str(a) for a in action_history)
                return str(action_history)
        
        return None

    def __call__(self, data: DataProto) -> Tuple[torch.Tensor, Dict[str, List[float]]]:
        """
        Compute rewards for a batch of data.
        
        For harm tasks:
        - If LLM evaluator is available, uses it to evaluate trajectories
        - Otherwise, parses pre-computed evaluation responses
        
        For other tasks, uses the configured score function.
        
        Args:
            data: DataProto containing batch data
            
        Returns:
            Tuple of (reward_tensor, reward_metrics)
        """
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_metrics = defaultdict(list)
        
        # Track harm-specific metrics separately
        harm_metrics = defaultdict(list)
        
        # Collect harm tasks for batch evaluation
        harm_task_indices = []
        harm_trajectories = []
        harm_actions = []
        
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem
            response_ids = data_item.batch["responses"]
            response_mask = data_item.batch["response_mask"]
            valid_response_length = response_mask.sum()
            valid_response_ids = response_ids[:valid_response_length]

            response_str = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=self.config.skip_special_tokens
            )
            
            # Check if this is a harm task
            is_harm = self._is_harm_task(data_item)
            
            if is_harm and self.config.use_harm_reward_for_harm_tasks:
                harm_action = self._get_harm_action(data_item)
                if harm_action is None:
                    harm_action = data_item.non_tensor_batch.get("ground_truth", "")
                
                # Get trajectory (use response_str as fallback)
                trajectory = self._get_trajectory(data_item) or response_str
                
                if self.harm_evaluator is not None:
                    # Collect for batch evaluation
                    harm_task_indices.append(i)
                    harm_trajectories.append(trajectory)
                    harm_actions.append(harm_action)
                else:
                    # Use response parsing mode
                    score = self.harm_compute_score(response_str, harm_action)
                    reward_tensor[i, valid_response_length - 1] = score["overall"]
                    
                    # Track harm-specific metrics
                    harm_metrics["harm_overall"].append(score["overall"])
                    harm_metrics["harm_safety"].append(score.get("safety", 0.0))
                    harm_metrics["harm_task_completion"].append(score.get("task_completion", 0.0))
                    harm_metrics["harm_is_safe"].append(1.0 if score.get("is_safe", False) else 0.0)
                    
                    for key, value in score.items():
                        if isinstance(value, (int, float)):
                            reward_metrics[key].append(value)
            else:
                # Use standard reward computation
                ground_truth = data_item.non_tensor_batch["ground_truth"]
                score = self.compute_score(response_str, ground_truth)
                reward_tensor[i, valid_response_length - 1] = score["overall"]
                
                for key, value in score.items():
                    if isinstance(value, (int, float)):
                        reward_metrics[key].append(value)
        
        # Batch evaluate harm tasks using LLM
        if harm_task_indices and self.harm_evaluator is not None:
            harm_scores = self._evaluate_harm_batch(harm_trajectories, harm_actions)
            
            for idx, score in zip(harm_task_indices, harm_scores):
                data_item = data[idx]
                response_mask = data_item.batch["response_mask"]
                valid_response_length = response_mask.sum()
                
                reward_tensor[idx, valid_response_length - 1] = score["overall"]
                
                # Track harm-specific metrics
                harm_metrics["harm_overall"].append(score["overall"])
                harm_metrics["harm_safety"].append(score.get("safety", 0.0))
                harm_metrics["harm_task_completion"].append(score.get("task_completion", 0.0))
                harm_metrics["harm_is_safe"].append(1.0 if score.get("is_safe", False) else 0.0)
                
                for key, value in score.items():
                    if isinstance(value, (int, float)):
                        reward_metrics[key].append(value)
        
        # Merge harm metrics into main metrics
        for key, values in harm_metrics.items():
            reward_metrics[key] = values

        return reward_tensor, reward_metrics
    
    def _evaluate_harm_batch(
        self,
        trajectories: List[str],
        harm_actions: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Evaluate a batch of harm tasks using the LLM evaluator.
        
        Args:
            trajectories: List of agent trajectories
            harm_actions: List of harmful actions
            
        Returns:
            List of evaluation score dictionaries
        """
        if self.harm_evaluator is None:
            raise RuntimeError("Harm evaluator not initialized")
        
        return self.harm_evaluator.evaluate_batch(trajectories, harm_actions)
    
    def compute_harm_reward(
        self,
        trajectory: str,
        harm_action: str,
    ) -> Dict[str, float]:
        """
        Compute harm reward for a single trajectory using LLM evaluation.
        
        This is a convenience method for computing harm rewards outside
        the main __call__ flow.
        
        Args:
            trajectory: The agent's action trajectory
            harm_action: The harmful action to evaluate
            
        Returns:
            Dictionary with reward scores including:
            - overall: Weighted combination of all scores
            - safety: Safety score (1.0 = safe, 0.0 = unsafe)
            - task_completion: Task completion score
            - is_safe: Boolean indicating if behavior was safe
            - reasoning: LLM's reasoning for the evaluation
        """
        if self.harm_evaluator is not None:
            return self.harm_evaluator.evaluate(trajectory, harm_action)
        else:
            # Fallback to response parsing mode
            return self.harm_compute_score(trajectory, harm_action)
    
    def compute_harm_reward_batch(
        self,
        trajectories: List[str],
        harm_actions: List[str],
    ) -> List[Dict[str, float]]:
        """
        Compute harm rewards for multiple trajectories using LLM evaluation.
        
        Args:
            trajectories: List of agent trajectories
            harm_actions: List of harmful actions
            
        Returns:
            List of reward score dictionaries
        """
        if self.harm_evaluator is not None:
            return self.harm_evaluator.evaluate_batch(trajectories, harm_actions)
        else:
            return [
                self.harm_compute_score(traj, harm)
                for traj, harm in zip(trajectories, harm_actions)
            ]
    
    def compute_learnability_bonus(
        self,
        safety_score: float,
        target_difficulty: Optional[float] = None,
    ) -> float:
        """
        Compute learnability bonus for harm tasks.
        
        Following Absolute Zero paradigm, we prefer tasks of medium difficulty.
        Tasks where the agent is ~50% safe are most learnable.
        
        Args:
            safety_score: The safety score from evaluation
            target_difficulty: Target safety score for maximum learnability
                             (defaults to config value)
            
        Returns:
            Learnability bonus value
        """
        if not self.config.harm_config.enable_learnability_bonus:
            return 0.0
        
        if target_difficulty is None:
            target_difficulty = self.config.harm_config.target_difficulty
        
        # Learnability is highest when safety_score is near target_difficulty
        learnability = 1.0 - abs(safety_score - target_difficulty) * 2
        learnability = max(0.0, learnability)
        
        return learnability * self.config.harm_config.learnability_weight
