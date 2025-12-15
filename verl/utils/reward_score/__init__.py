"""
Reward score computation module.

This module provides reward computation functions for various task types:
- Math tasks: math_compute_score
- R1V tasks: r1v_compute_score
- Harm evaluation tasks: harm_compute_score, harm_compute_score_from_trajectory

For harm evaluation, the module supports:
1. LLM-based evaluation using OpenAI, vLLM, or custom evaluators
2. Response parsing mode for pre-computed evaluations
"""

from .math import math_compute_score
from .r1v import r1v_compute_score
from .harm import (
    # Core scoring functions
    harm_compute_score,
    harm_compute_score_default,
    harm_compute_score_from_trajectory,
    harm_compute_score_batch,
    harm_safety_reward,
    harm_task_completion_reward,
    harm_format_reward,
    create_harm_reward_function,
    create_harm_evaluation_prompt,
    
    # LLM Evaluator classes
    BaseHarmEvaluator,
    HarmEvaluatorConfig,
    OpenAIHarmEvaluator,
    VLLMHarmEvaluator,
    FunctionHarmEvaluator,
    create_harm_evaluator,
    
    # Global evaluator management
    set_global_harm_evaluator,
    get_global_harm_evaluator,
    evaluate_trajectory_with_global_evaluator,
    
    # Helper functions
    extract_json_from_response,
)


__all__ = [
    # Math/R1V
    "math_compute_score",
    "r1v_compute_score",
    
    # Harm scoring functions
    "harm_compute_score",
    "harm_compute_score_default",
    "harm_compute_score_from_trajectory",
    "harm_compute_score_batch",
    "harm_safety_reward",
    "harm_task_completion_reward",
    "harm_format_reward",
    "create_harm_reward_function",
    "create_harm_evaluation_prompt",
    
    # LLM Evaluator classes
    "BaseHarmEvaluator",
    "HarmEvaluatorConfig",
    "OpenAIHarmEvaluator",
    "VLLMHarmEvaluator",
    "FunctionHarmEvaluator",
    "create_harm_evaluator",
    
    # Global evaluator management
    "set_global_harm_evaluator",
    "get_global_harm_evaluator",
    "evaluate_trajectory_with_global_evaluator",
    
    # Helper functions
    "extract_json_from_response",
]
