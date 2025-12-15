"""
Reward config for various reward functions including harm evaluation.

This module provides configuration for:
1. Standard reward functions (math, r1v)
2. Harm evaluation with LLM-based safety assessment
3. Learnability bonuses for Absolute Zero style training
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class HarmEvaluatorConfig:
    """Configuration for the LLM-based harm evaluator."""
    # Evaluator type: "openai", "vllm", "function"
    evaluator_type: str = "openai"
    
    # Model settings
    model_name: str = "gpt-4"
    api_key: Optional[str] = "sk-wIfAFRigBsJx1nBzzpQnVsyjloVy1ZXSfEj4ZzFZsuAldefd"  # Default API key for LLM judge
    api_base: Optional[str] = "http://149.88.91.225:3000/v1"  # Custom API base URL for LLM judge
    
    # Generation settings
    temperature: float = 0.0  # Low temperature for consistent evaluation
    max_tokens: int = 1024
    timeout: float = 60.0
    
    # Retry settings
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # Batch settings
    batch_size: int = 10
    use_async: bool = True


@dataclass
class HarmRewardConfig:
    """Configuration for harm evaluation reward."""
    # Score weights
    safety_weight: float = 1.0  # Weight for safety score
    task_completion_weight: float = 0.5  # Weight for task completion
    format_weight: float = 0.1  # Weight for format compliance
    
    # Thresholds for binary classification
    safety_threshold: float = 0.5  # Score above this is considered safe
    
    # Learnability settings (Absolute Zero style)
    enable_learnability_bonus: bool = True
    learnability_weight: float = 0.1  # Bonus for medium-difficulty tasks
    target_difficulty: float = 0.5  # Target safety score for max learnability
    
    # LLM evaluator configuration
    evaluator_config: HarmEvaluatorConfig = field(default_factory=HarmEvaluatorConfig)
    
    # Whether to use LLM for evaluation (vs parsing pre-computed responses)
    use_llm_evaluation: bool = True


@dataclass
class RewardConfig:
    """Main reward configuration."""
    reward_type: str = "function"
    score_function: str = "math"  # Options: "math", "r1v", "harm"
    skip_special_tokens: bool = True
    
    # Harm-specific configuration
    harm_config: HarmRewardConfig = field(default_factory=HarmRewardConfig)
    
    # Whether to use separate reward for harm tasks
    use_harm_reward_for_harm_tasks: bool = True
