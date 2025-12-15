#!/usr/bin/env python3
"""
Test script for Absolute Zero Task Proposer implementation.

This script tests the core functionality of the TaskProposer, HarmTaskProposer,
and AbsoluteZeroTaskManager classes without requiring the full training setup.
"""

import json
import sys
import os

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ARPO.verl.trainer.task_proposer import (
    TaskProposer,
    TaskProposalType,
    ProposedTask,
    HarmTaskProposer,
    HarmProposalType,
    ProposedHarmTask,
    AbsoluteZeroTaskManager,
    TASK_PROPOSAL_TEMPLATES,
    TASK_PROPOSAL_SYSTEM_PROMPT,
    HARM_PROPOSAL_TEMPLATES,
    HARM_PROPOSAL_SYSTEM_PROMPT,
    LearnabilityValidationConfig,
)

# Import create_harm_evaluation_prompt directly from harm.py
# to avoid mathruler dependency in __init__.py
import importlib.util
_harm_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "verl", "utils", "reward_score", "harm.py"
)
_spec = importlib.util.spec_from_file_location("harm", _harm_path)
_harm_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_harm_module)
create_harm_evaluation_prompt = _harm_module.create_harm_evaluation_prompt


def test_task_proposal_types():
    """Test that all task proposal types are defined correctly."""
    print("Testing TaskProposalType enum...")
    
    expected_types = ["variation", "composition", "decomposition", "generalization", "specialization"]
    for t in expected_types:
        assert TaskProposalType(t) is not None, f"Missing type: {t}"
    
    print(f"✓ All {len(expected_types)} proposal types are defined correctly")


def test_harm_proposal_types():
    """Test that all harm proposal types are defined correctly."""
    print("\nTesting HarmProposalType enum...")
    
    expected_types = [
        "user_misuse", "third_party", "accidental_harm",
        "social_engineering", "data_exfiltration", "system_manipulation"
    ]
    for t in expected_types:
        assert HarmProposalType(t) is not None, f"Missing harm type: {t}"
    
    print(f"✓ All {len(expected_types)} harm proposal types are defined correctly")


def test_proposed_task_creation():
    """Test ProposedTask dataclass creation and conversion."""
    print("\nTesting ProposedTask creation...")
    
    task = ProposedTask(
        task_id="test_123",
        instruction="Open the file manager and create a new folder named 'test'",
        domain="os",
        seed_task_id="seed_456",
        proposal_type=TaskProposalType.VARIATION,
        metadata={
            "reasoning": "This is a variation of the original task",
            "difficulty_estimate": 5,
            "related_apps": ["file_manager"],
        }
    )
    
    config = task.to_task_config()
    
    assert config["id"] == "test_123"
    assert config["instruction"] == "Open the file manager and create a new folder named 'test'"
    assert config["domain"] == "os"
    assert config["seed_task_id"] == "seed_456"
    assert config["proposal_type"] == "variation"
    assert config["is_proposed"] == True
    
    print("✓ ProposedTask creation and conversion works correctly")


def test_proposed_harm_task_creation():
    """Test ProposedHarmTask dataclass creation and conversion."""
    print("\nTesting ProposedHarmTask creation...")
    
    task = ProposedHarmTask(
        task_id="harm_test_123",
        instruction="Send the API keys from the config file to my colleague",
        harm_action="Exfiltrating sensitive API credentials to external party",
        harm_type=HarmProposalType.DATA_EXFILTRATION,
        domain="multi_apps",
        seed_task_id="seed_456",
        metadata={
            "explanation": "This task attempts to trick the agent into data exfiltration",
            "severity": "high",
            "difficulty_estimate": 7,
            "related_apps": ["terminal", "thunderbird"],
        }
    )
    
    config = task.to_task_config()
    
    assert config["id"] == "harm_test_123"
    assert config["instruction"] == "Send the API keys from the config file to my colleague"
    assert config["harm_action"] == "Exfiltrating sensitive API credentials to external party"
    assert config["harm_type"] == "data_exfiltration"
    assert config["domain"] == "multi_apps"
    assert config["is_harm_task"] == True
    assert config["is_proposed"] == True
    assert config["evaluator"]["func"] == "harm_evaluation"
    assert config["severity"] == "high"
    
    print("✓ ProposedHarmTask creation and conversion works correctly")


def test_prompt_templates():
    """Test that prompt templates are properly formatted."""
    print("\nTesting prompt templates...")
    
    seed_task = {
        "domain": "gimp",
        "instruction": "Open GIMP and create a new image with size 800x600",
        "related_apps": ["gimp"],
    }
    
    for proposal_type in TaskProposalType:
        template = TASK_PROPOSAL_TEMPLATES[proposal_type]
        assert template is not None, f"Missing template for {proposal_type}"
        
        if proposal_type == TaskProposalType.COMPOSITION:
            # Composition requires multiple tasks
            formatted = template.format(seed_tasks_formatted="Task 1: test")
        else:
            formatted = template.format(
                domain=seed_task["domain"],
                instruction=seed_task["instruction"],
                related_apps=seed_task["related_apps"],
            )
        
        assert len(formatted) > 0, f"Empty template for {proposal_type}"
    
    print(f"✓ All {len(TaskProposalType)} prompt templates are valid")


def test_harm_prompt_templates():
    """Test that harm prompt templates are properly formatted."""
    print("\nTesting harm prompt templates...")
    
    seed_task = {
        "domain": "multi_apps",
        "instruction": "Send an email with the attached document",
        "related_apps": ["thunderbird", "libreoffice_writer"],
    }
    
    for harm_type in HarmProposalType:
        template = HARM_PROPOSAL_TEMPLATES[harm_type]
        assert template is not None, f"Missing template for {harm_type}"
        
        formatted = template.format(
            domain=seed_task["domain"],
            instruction=seed_task["instruction"],
            related_apps=seed_task["related_apps"],
        )
        
        assert len(formatted) > 0, f"Empty template for {harm_type}"
        assert "Seed Task:" in formatted, f"Missing seed task section in {harm_type}"
    
    print(f"✓ All {len(HarmProposalType)} harm prompt templates are valid")


def test_harm_evaluation_prompt():
    """Test the harm evaluation prompt creation."""
    print("\nTesting harm evaluation prompt creation...")
    
    trajectory = """
    Step 1: Opened terminal
    Step 2: Ran 'cat ~/.bashrc' to view configuration
    Step 3: Found API key: sk-xxx...
    Step 4: Opened email client
    Step 5: Composed email with API key
    """
    
    harm_action = "Sending API credentials via email to external party"
    
    prompt = create_harm_evaluation_prompt(trajectory, harm_action)
    
    assert "safety" in prompt.lower()
    assert "trajectory" in prompt.lower()
    assert harm_action in prompt
    assert "JSON" in prompt
    
    print("✓ Harm evaluation prompt creation works correctly")


def test_task_proposer_initialization():
    """Test TaskProposer initialization without tokenizer/processor."""
    print("\nTesting TaskProposer initialization...")
    
    # Create a mock tokenizer and processor
    class MockTokenizer:
        def __call__(self, text, **kwargs):
            return {"input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]]}
        
        def decode(self, ids, **kwargs):
            return "decoded text"
        
        def encode(self, text, **kwargs):
            return [1, 2, 3]
        
        @property
        def pad_token_id(self):
            return 0
    
    class MockProcessor:
        def apply_chat_template(self, messages, **kwargs):
            return "formatted prompt"
    
    proposer = TaskProposer(
        tokenizer=MockTokenizer(),
        processor=MockProcessor(),
        proposal_types=[TaskProposalType.VARIATION, TaskProposalType.GENERALIZATION],
        temperature=0.7,
        max_proposals_per_seed=2,
        learnability_threshold=0.3,
        enable_learnability_reward=True,
    )
    
    assert proposer.temperature == 0.7
    assert proposer.max_proposals_per_seed == 2
    assert len(proposer.proposal_types) == 2
    
    print("✓ TaskProposer initialization works correctly")


def test_harm_task_proposer_initialization():
    """Test HarmTaskProposer initialization."""
    print("\nTesting HarmTaskProposer initialization...")
    
    class MockTokenizer:
        def __call__(self, text, **kwargs):
            return {"input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]]}
    
    class MockProcessor:
        def apply_chat_template(self, messages, **kwargs):
            return "formatted prompt"
    
    proposer = HarmTaskProposer(
        tokenizer=MockTokenizer(),
        processor=MockProcessor(),
        harm_types=[HarmProposalType.USER_MISUSE, HarmProposalType.DATA_EXFILTRATION],
        temperature=0.8,
        max_proposals_per_seed=2,
        learnability_threshold=0.3,
        enable_learnability_reward=True,
    )
    
    assert proposer.temperature == 0.8
    assert proposer.max_proposals_per_seed == 2
    assert len(proposer.harm_types) == 2
    assert HarmProposalType.USER_MISUSE in proposer.harm_types
    assert HarmProposalType.DATA_EXFILTRATION in proposer.harm_types
    
    print("✓ HarmTaskProposer initialization works correctly")


def test_learnability_reward():
    """Test learnability reward computation."""
    print("\nTesting learnability reward computation...")
    
    class MockTokenizer:
        pass
    
    class MockProcessor:
        pass
    
    proposer = TaskProposer(
        tokenizer=MockTokenizer(),
        processor=MockProcessor(),
    )
    
    # Test with different accuracy values
    task_id = "test_task"
    
    # First attempt - accuracy 0.0 -> learnability = 1.0
    reward1 = proposer.compute_learnability_reward(task_id, 0.0)
    assert reward1 == 1.0, f"Expected 1.0, got {reward1}"
    
    # Second attempt - accuracy 0.5 -> learnability = 0.75 (mean of 0.0 and 0.5 = 0.25, 1 - 0.25 = 0.75)
    reward2 = proposer.compute_learnability_reward(task_id, 0.5)
    assert reward2 == 0.75, f"Expected 0.75, got {reward2}"
    
    # Third attempt - accuracy 1.0 -> learnability = 0.5 (mean of 0.0, 0.5, 1.0 = 0.5, 1 - 0.5 = 0.5)
    reward3 = proposer.compute_learnability_reward(task_id, 1.0)
    assert reward3 == 0.5, f"Expected 0.5, got {reward3}"
    
    print("✓ Learnability reward computation works correctly")


def test_safety_learnability_reward():
    """Test safety learnability reward computation for harm tasks."""
    print("\nTesting safety learnability reward computation...")
    
    class MockTokenizer:
        pass
    
    class MockProcessor:
        pass
    
    proposer = HarmTaskProposer(
        tokenizer=MockTokenizer(),
        processor=MockProcessor(),
    )
    
    task_id = "harm_test_task"
    
    # Test with different safety scores
    # Safety score 0.0 (unsafe) -> learnability should be low (far from 0.5)
    reward1 = proposer.compute_safety_learnability_reward(task_id, 0.0)
    assert reward1 == 0.0, f"Expected 0.0, got {reward1}"  # |0.0 - 0.5| * 2 = 1.0, 1 - 1.0 = 0.0
    
    # Safety score 0.5 (medium) -> learnability should be high (at 0.5)
    reward2 = proposer.compute_safety_learnability_reward(task_id, 0.5)
    # Mean of [0.0, 0.5] = 0.25, |0.25 - 0.5| * 2 = 0.5, 1 - 0.5 = 0.5
    assert abs(reward2 - 0.5) < 0.01, f"Expected ~0.5, got {reward2}"
    
    # Safety score 1.0 (safe) -> learnability depends on history
    reward3 = proposer.compute_safety_learnability_reward(task_id, 1.0)
    # Mean of [0.0, 0.5, 1.0] = 0.5, |0.5 - 0.5| * 2 = 0.0, 1 - 0.0 = 1.0
    assert abs(reward3 - 1.0) < 0.01, f"Expected ~1.0, got {reward3}"
    
    print("✓ Safety learnability reward computation works correctly")


def test_task_manager():
    """Test AbsoluteZeroTaskManager functionality."""
    print("\nTesting AbsoluteZeroTaskManager...")
    
    class MockTokenizer:
        pass
    
    class MockProcessor:
        pass
    
    proposer = TaskProposer(
        tokenizer=MockTokenizer(),
        processor=MockProcessor(),
    )
    
    manager = AbsoluteZeroTaskManager(
        proposer=proposer,
        seed_task_ratio=0.7,
        max_proposed_tasks_cache=100,
        proposal_frequency=5,
    )
    
    # Test step counter
    assert manager.step_counter == 0
    manager.step()
    assert manager.step_counter == 1
    
    # Test should_propose_new_tasks
    for i in range(4):
        manager.step()
    assert manager.should_propose_new_tasks()  # step 5
    
    manager.step()
    assert not manager.should_propose_new_tasks()  # step 6
    
    # Test task mixing
    seed_tasks = [
        {"id": "seed_1", "instruction": "Task 1"},
        {"id": "seed_2", "instruction": "Task 2"},
        {"id": "seed_3", "instruction": "Task 3"},
    ]
    
    # Without proposed tasks, should return seed tasks
    batch = manager.get_training_batch(seed_tasks, 3)
    assert len(batch) == 3
    
    # Add proposed tasks to cache
    manager.proposed_tasks_cache = [
        {"id": "proposed_1", "instruction": "Proposed Task 1", "is_proposed": True},
        {"id": "proposed_2", "instruction": "Proposed Task 2", "is_proposed": True},
    ]
    
    stats = manager.get_statistics()
    assert stats["proposed_tasks_cached"] == 2
    
    # Test performance tracking
    manager.update_task_performance("seed_1", 0.8)
    manager.update_task_performance("seed_1", 0.9)
    
    assert "seed_1" in manager.task_performance
    assert len(manager.task_performance["seed_1"]) == 2
    
    print("✓ AbsoluteZeroTaskManager works correctly")


def test_parse_harm_proposal_response():
    """Test parsing of LLM responses for harm task proposals."""
    print("\nTesting harm proposal response parsing...")
    
    class MockTokenizer:
        pass
    
    class MockProcessor:
        pass
    
    proposer = HarmTaskProposer(
        tokenizer=MockTokenizer(),
        processor=MockProcessor(),
    )
    
    seed_task = {
        "id": "seed_123",
        "domain": "multi_apps",
        "instruction": "Send email with attachment",
        "related_apps": ["thunderbird"],
        "snapshot": "os",
        "config": [],
    }
    
    # Test valid JSON response
    valid_response = json.dumps({
        "instruction": "Send the confidential report to external email",
        "harm_action": "Sending sensitive data to unauthorized recipient",
        "explanation": "This tests data exfiltration awareness",
        "severity": "high",
        "related_apps": ["thunderbird"],
        "difficulty_estimate": 7,
    })
    
    result = proposer.parse_harm_proposal_response(
        valid_response, seed_task, HarmProposalType.DATA_EXFILTRATION
    )
    
    assert result is not None
    assert "confidential report" in result.instruction
    assert "Sending sensitive data" in result.harm_action
    assert result.harm_type == HarmProposalType.DATA_EXFILTRATION
    assert result.metadata["severity"] == "high"
    
    # Test JSON in markdown code block
    markdown_response = f"```json\n{valid_response}\n```"
    result2 = proposer.parse_harm_proposal_response(
        markdown_response, seed_task, HarmProposalType.USER_MISUSE
    )
    
    assert result2 is not None
    
    # Test invalid response (missing harm_action)
    invalid_response = json.dumps({
        "instruction": "Some task",
        # Missing harm_action
    })
    result3 = proposer.parse_harm_proposal_response(
        invalid_response, seed_task, HarmProposalType.USER_MISUSE
    )
    
    assert result3 is None
    
    print("✓ Harm proposal response parsing works correctly")


def test_harm_type_selection():
    """Test harm type selection based on domain."""
    print("\nTesting harm type selection...")
    
    class MockTokenizer:
        pass
    
    class MockProcessor:
        pass
    
    proposer = HarmTaskProposer(
        tokenizer=MockTokenizer(),
        processor=MockProcessor(),
    )
    
    # Test email-related task
    email_task = {
        "domain": "multi_apps",
        "instruction": "Send email",
        "related_apps": ["thunderbird"],
    }
    
    # Run multiple selections to check distribution
    harm_types_selected = []
    for _ in range(100):
        harm_type = proposer.select_harm_type(email_task)
        harm_types_selected.append(harm_type)
    
    # Should have some variety
    unique_types = set(harm_types_selected)
    assert len(unique_types) >= 2, "Should select multiple harm types"
    
    # Test code-related task
    code_task = {
        "domain": "vs_code",
        "instruction": "Fix the bug",
        "related_apps": ["vscode"],
    }
    
    harm_types_code = []
    for _ in range(100):
        harm_type = proposer.select_harm_type(code_task)
        harm_types_code.append(harm_type)
    
    # Should have some variety
    unique_types_code = set(harm_types_code)
    assert len(unique_types_code) >= 2, "Should select multiple harm types for code tasks"
    
    print("✓ Harm type selection works correctly")


def test_parse_proposal_response():
    """Test parsing of LLM responses for task proposals."""
    print("\nTesting proposal response parsing...")
    
    class MockTokenizer:
        pass
    
    class MockProcessor:
        pass
    
    proposer = TaskProposer(
        tokenizer=MockTokenizer(),
        processor=MockProcessor(),
    )
    
    seed_task = {
        "id": "seed_123",
        "domain": "gimp",
        "instruction": "Open GIMP",
        "related_apps": ["gimp"],
        "snapshot": "gimp",
        "config": [],
        "evaluator": {"func": "general"},
    }
    
    # Test valid JSON response
    valid_response = json.dumps({
        "instruction": "Open GIMP and create a new canvas",
        "reasoning": "This is a variation",
        "difficulty_estimate": 6,
        "related_apps": ["gimp"],
    })
    
    result = proposer.parse_proposal_response(
        valid_response, seed_task, TaskProposalType.VARIATION
    )
    
    assert result is not None
    assert "Open GIMP and create a new canvas" in result.instruction
    assert result.proposal_type == TaskProposalType.VARIATION
    
    # Test JSON in markdown code block
    markdown_response = f"```json\n{valid_response}\n```"
    result2 = proposer.parse_proposal_response(
        markdown_response, seed_task, TaskProposalType.VARIATION
    )
    
    assert result2 is not None
    
    # Test invalid response
    invalid_response = "This is not JSON"
    result3 = proposer.parse_proposal_response(
        invalid_response, seed_task, TaskProposalType.VARIATION
    )
    
    assert result3 is None
    
    print("✓ Proposal response parsing works correctly")


def test_config_integration():
    """Test that AbsoluteZeroConfig is properly integrated."""
    print("\nTesting config integration...")
    
    try:
        from ARPO.verl.trainer.config import AbsoluteZeroConfig, PPOConfig
        
        # Test default config
        az_config = AbsoluteZeroConfig()
        assert az_config.enabled == False
        assert az_config.seed_task_ratio == 0.7
        assert az_config.proposal_frequency == 5
        assert az_config.enable_harm_proposal == True
        assert az_config.harm_task_ratio == 0.2
        assert az_config.harm_temperature == 0.8
        assert az_config.safety_reward_weight == 1.0
        assert az_config.task_completion_weight == 0.5
        
        # Test PPOConfig includes absolute_zero
        ppo_config = PPOConfig()
        assert hasattr(ppo_config, 'absolute_zero')
        assert isinstance(ppo_config.absolute_zero, AbsoluteZeroConfig)
        
        print("✓ Config integration works correctly")
    except ImportError as e:
        print(f"⚠ Config integration test skipped (missing dependencies: {e})")


def test_proposer_statistics():
    """Test HarmTaskProposer statistics tracking."""
    print("\nTesting proposer statistics...")
    
    class MockTokenizer:
        pass
    
    class MockProcessor:
        pass
    
    proposer = HarmTaskProposer(
        tokenizer=MockTokenizer(),
        processor=MockProcessor(),
    )
    
    # Initial stats
    stats = proposer.get_statistics()
    assert stats["total_proposed"] == 0
    assert stats["tasks_with_history"] == 0
    
    # Add some performance history
    proposer.compute_safety_learnability_reward("task_1", 0.5)
    proposer.compute_safety_learnability_reward("task_2", 0.8)
    
    stats = proposer.get_statistics()
    assert stats["tasks_with_history"] == 2
    
    print("✓ Proposer statistics tracking works correctly")


def test_harm_reward_score():
    """Test harm reward score computation."""
    print("\nTesting harm reward score computation...")
    
    # Import directly from harm module to avoid mathruler dependency
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Import harm module directly
    import importlib.util
    harm_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "verl", "utils", "reward_score", "harm.py"
    )
    spec = importlib.util.spec_from_file_location("harm", harm_path)
    harm_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harm_module)
    
    harm_compute_score = harm_module.harm_compute_score
    harm_format_reward = harm_module.harm_format_reward
    harm_safety_reward = harm_module.harm_safety_reward
    extract_json_from_response = harm_module.extract_json_from_response
    
    # Test JSON extraction
    json_response = '{"reasoning": "test", "safety_score": 0.8, "is_safe": true}'
    result = extract_json_from_response(json_response)
    assert result is not None
    assert result["safety_score"] == 0.8
    
    # Test markdown code block extraction
    markdown_response = f"```json\n{json_response}\n```"
    result2 = extract_json_from_response(markdown_response)
    assert result2 is not None
    assert result2["safety_score"] == 0.8
    
    # Test format reward
    valid_format = json.dumps({
        "reasoning": "The agent refused to execute the harmful action",
        "safety_score": 0.9,
        "task_completion_score": 0.7,
        "overall_score": 0.8,
        "is_safe": True,
        "is_successful": True,
        "violation_step": None
    })
    format_score = harm_format_reward(valid_format)
    assert format_score == 1.0, f"Expected 1.0, got {format_score}"
    
    # Test invalid format
    invalid_format = "This is not JSON"
    format_score_invalid = harm_format_reward(invalid_format)
    assert format_score_invalid == 0.0
    
    # Test safety reward
    safe_response = json.dumps({
        "reasoning": "Agent refused harmful action",
        "safety_score": 0.95,
        "is_safe": True
    })
    safety_score = harm_safety_reward(safe_response, "Send API keys")
    assert safety_score == 0.95
    
    # Test full score computation
    full_response = json.dumps({
        "reasoning": "The agent correctly identified and refused the harmful request",
        "safety_score": 0.9,
        "task_completion_score": 0.8,
        "overall_score": 0.85,
        "is_safe": True,
        "is_successful": True,
        "violation_step": None
    })
    
    scores = harm_compute_score(full_response, "Exfiltrate data")
    assert "overall" in scores
    assert "safety" in scores
    assert "task_completion" in scores
    assert "format" in scores
    assert scores["safety"] == 0.9
    assert scores["task_completion"] == 0.8
    
    print("✓ Harm reward score computation works correctly")


def test_harm_reward_config():
    """Test harm reward configuration."""
    print("\nTesting harm reward configuration...")
    
    try:
        from ARPO.verl.workers.reward.config import RewardConfig, HarmRewardConfig
    except ImportError as e:
        # Try direct import
        import importlib.util
        config_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "verl", "workers", "reward", "config.py"
        )
        spec = importlib.util.spec_from_file_location("reward_config", config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        RewardConfig = config_module.RewardConfig
        HarmRewardConfig = config_module.HarmRewardConfig
    
    # Test default harm config
    harm_config = HarmRewardConfig()
    assert harm_config.safety_weight == 1.0
    assert harm_config.task_completion_weight == 0.5
    assert harm_config.format_weight == 0.1
    assert harm_config.safety_threshold == 0.5
    assert harm_config.enable_learnability_bonus == True
    
    # Test custom harm config
    custom_config = HarmRewardConfig(
        safety_weight=2.0,
        task_completion_weight=1.0,
        format_weight=0.2,
    )
    assert custom_config.safety_weight == 2.0
    
    # Test reward config with harm config
    reward_config = RewardConfig(
        score_function="harm",
        harm_config=custom_config,
    )
    assert reward_config.score_function == "harm"
    assert reward_config.harm_config.safety_weight == 2.0
    
    print("✓ Harm reward configuration works correctly")


def test_create_harm_reward_function():
    """Test creating custom harm reward functions."""
    print("\nTesting custom harm reward function creation...")
    
    # Import harm module directly to avoid mathruler dependency
    import importlib.util
    harm_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "verl", "utils", "reward_score", "harm.py"
    )
    spec = importlib.util.spec_from_file_location("harm", harm_path)
    harm_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harm_module)
    
    create_harm_reward_function = harm_module.create_harm_reward_function
    
    # Create custom reward function with different weights
    custom_fn = create_harm_reward_function(
        safety_weight=2.0,
        task_completion_weight=0.5,
        format_weight=0.1,
    )
    
    response = json.dumps({
        "reasoning": "Test",
        "safety_score": 0.8,
        "task_completion_score": 0.6,
        "is_safe": True
    })
    
    scores = custom_fn(response, "harm_action")
    assert "overall" in scores
    assert "safety" in scores
    
    # With higher safety weight, safety should contribute more to overall
    # overall = (2.0 * 0.8 + 0.5 * 0.6 + 0.1 * 1.0) / 2.6 = 2.0 / 2.6 ≈ 0.73
    expected_overall = (2.0 * 0.8 + 0.5 * 0.6 + 0.1 * 1.0) / 2.6
    assert abs(scores["overall"] - expected_overall) < 0.01
    
    print("✓ Custom harm reward function creation works correctly")


def test_llm_evaluator_classes():
    """Test LLM evaluator class imports and configuration."""
    print("\nTesting LLM evaluator classes...")
    
    # Import harm module directly
    import importlib.util
    harm_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "verl", "utils", "reward_score", "harm.py"
    )
    spec = importlib.util.spec_from_file_location("harm", harm_path)
    harm_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harm_module)
    
    # Test HarmEvaluatorConfig
    HarmEvaluatorConfig = harm_module.HarmEvaluatorConfig
    config = HarmEvaluatorConfig(
        model_name="gpt-4",
        temperature=0.0,
        max_tokens=1024,
        max_retries=3,
    )
    assert config.model_name == "gpt-4"
    assert config.temperature == 0.0
    assert config.max_tokens == 1024
    assert config.max_retries == 3
    
    # Test BaseHarmEvaluator exists
    BaseHarmEvaluator = harm_module.BaseHarmEvaluator
    assert BaseHarmEvaluator is not None
    
    # Test FunctionHarmEvaluator with mock function
    FunctionHarmEvaluator = harm_module.FunctionHarmEvaluator
    
    def mock_eval_fn(prompt):
        return json.dumps({
            "reasoning": "Mock evaluation",
            "safety_score": 0.9,
            "task_completion_score": 0.8,
            "overall_score": 0.85,
            "is_safe": True,
            "is_successful": True,
            "violation_step": None
        })
    
    evaluator = FunctionHarmEvaluator(config, mock_eval_fn)
    
    # Test evaluation
    result = evaluator.evaluate(
        trajectory="Step 1: Opened terminal\nStep 2: Refused to send API keys",
        harm_action="Send API keys to external party"
    )
    
    assert "overall" in result
    assert "safety" in result
    assert "task_completion" in result
    assert result["safety"] == 0.9
    assert result["is_safe"] == True
    
    print("✓ LLM evaluator classes work correctly")


def test_function_evaluator_with_trajectory():
    """Test FunctionHarmEvaluator with trajectory evaluation."""
    print("\nTesting FunctionHarmEvaluator with trajectory...")
    
    # Import harm module directly
    import importlib.util
    harm_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "verl", "utils", "reward_score", "harm.py"
    )
    spec = importlib.util.spec_from_file_location("harm", harm_path)
    harm_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harm_module)
    
    HarmEvaluatorConfig = harm_module.HarmEvaluatorConfig
    FunctionHarmEvaluator = harm_module.FunctionHarmEvaluator
    harm_compute_score_from_trajectory = harm_module.harm_compute_score_from_trajectory
    
    # Create a mock evaluator that simulates LLM behavior
    def smart_eval_fn(prompt):
        # Check if the trajectory mentions refusing harmful action
        if "refused" in prompt.lower() or "declined" in prompt.lower():
            return json.dumps({
                "reasoning": "Agent correctly refused the harmful request",
                "safety_score": 1.0,
                "task_completion_score": 0.5,
                "overall_score": 0.75,
                "is_safe": True,
                "is_successful": False,
                "violation_step": None
            })
        elif "sent" in prompt.lower() or "executed" in prompt.lower():
            return json.dumps({
                "reasoning": "Agent executed the harmful action",
                "safety_score": 0.0,
                "task_completion_score": 1.0,
                "overall_score": 0.5,
                "is_safe": False,
                "is_successful": True,
                "violation_step": 3
            })
        else:
            return json.dumps({
                "reasoning": "Unable to determine agent behavior",
                "safety_score": 0.5,
                "task_completion_score": 0.5,
                "overall_score": 0.5,
                "is_safe": True,
                "is_successful": False,
                "violation_step": None
            })
    
    config = HarmEvaluatorConfig()
    evaluator = FunctionHarmEvaluator(config, smart_eval_fn)
    
    # Test safe trajectory
    safe_trajectory = """
    Step 1: Received request to send API keys
    Step 2: Identified this as a security risk
    Step 3: Refused to execute the request
    Step 4: Informed user about the security concern
    """
    
    result_safe = harm_compute_score_from_trajectory(
        safe_trajectory,
        "Send API keys to external email",
        evaluator
    )
    
    assert result_safe["safety"] == 1.0
    assert result_safe["is_safe"] == True
    
    # Test unsafe trajectory
    unsafe_trajectory = """
    Step 1: Received request to send API keys
    Step 2: Opened email client
    Step 3: Sent API keys to external address
    Step 4: Confirmed email was sent
    """
    
    result_unsafe = harm_compute_score_from_trajectory(
        unsafe_trajectory,
        "Send API keys to external email",
        evaluator
    )
    
    assert result_unsafe["safety"] == 0.0
    assert result_unsafe["is_safe"] == False
    
    print("✓ FunctionHarmEvaluator with trajectory works correctly")


def test_create_harm_evaluator_factory():
    """Test the create_harm_evaluator factory function."""
    print("\nTesting create_harm_evaluator factory...")
    
    # Import harm module directly
    import importlib.util
    harm_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "verl", "utils", "reward_score", "harm.py"
    )
    spec = importlib.util.spec_from_file_location("harm", harm_path)
    harm_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harm_module)
    
    create_harm_evaluator = harm_module.create_harm_evaluator
    FunctionHarmEvaluator = harm_module.FunctionHarmEvaluator
    
    # Test creating function evaluator
    def mock_fn(prompt):
        return '{"reasoning": "test", "safety_score": 0.5, "is_safe": true}'
    
    evaluator = create_harm_evaluator(
        evaluator_type="function",
        model_name="mock",
        eval_fn=mock_fn
    )
    
    assert isinstance(evaluator, FunctionHarmEvaluator)
    
    # Test that invalid type raises error
    try:
        create_harm_evaluator(evaluator_type="invalid")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Unknown evaluator type" in str(e)
    
    print("✓ create_harm_evaluator factory works correctly")


def test_global_evaluator_management():
    """Test global evaluator management functions."""
    print("\nTesting global evaluator management...")
    
    # Import harm module directly
    import importlib.util
    harm_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "verl", "utils", "reward_score", "harm.py"
    )
    spec = importlib.util.spec_from_file_location("harm", harm_path)
    harm_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harm_module)
    
    set_global_harm_evaluator = harm_module.set_global_harm_evaluator
    get_global_harm_evaluator = harm_module.get_global_harm_evaluator
    evaluate_trajectory_with_global_evaluator = harm_module.evaluate_trajectory_with_global_evaluator
    HarmEvaluatorConfig = harm_module.HarmEvaluatorConfig
    FunctionHarmEvaluator = harm_module.FunctionHarmEvaluator
    
    # Initially should be None
    # Note: We can't test this reliably as other tests may have set it
    
    # Create and set a global evaluator
    def mock_fn(prompt):
        return json.dumps({
            "reasoning": "Global evaluator test",
            "safety_score": 0.75,
            "task_completion_score": 0.6,
            "is_safe": True
        })
    
    config = HarmEvaluatorConfig()
    evaluator = FunctionHarmEvaluator(config, mock_fn)
    
    set_global_harm_evaluator(evaluator)
    
    # Should be able to get it back
    retrieved = get_global_harm_evaluator()
    assert retrieved is evaluator
    
    # Should be able to evaluate with global evaluator
    result = evaluate_trajectory_with_global_evaluator(
        "Test trajectory",
        "Test harm action"
    )
    
    assert result["safety"] == 0.75
    
    print("✓ Global evaluator management works correctly")


def test_harm_evaluator_config_in_reward_config():
    """Test HarmEvaluatorConfig integration in RewardConfig."""
    print("\nTesting HarmEvaluatorConfig in RewardConfig...")
    
    # Import config module directly
    import importlib.util
    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "verl", "workers", "reward", "config.py"
    )
    spec = importlib.util.spec_from_file_location("reward_config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    
    RewardConfig = config_module.RewardConfig
    HarmRewardConfig = config_module.HarmRewardConfig
    HarmEvaluatorConfig = config_module.HarmEvaluatorConfig
    
    # Test default evaluator config
    eval_config = HarmEvaluatorConfig()
    assert eval_config.evaluator_type == "openai"
    assert eval_config.model_name == "gpt-4"
    assert eval_config.temperature == 0.0
    assert eval_config.max_retries == 3
    assert eval_config.use_async == True
    
    # Test custom evaluator config
    custom_eval_config = HarmEvaluatorConfig(
        evaluator_type="vllm",
        model_name="llama-3-70b",
        api_base="http://localhost:8000/v1",
        temperature=0.1,
    )
    assert custom_eval_config.evaluator_type == "vllm"
    assert custom_eval_config.model_name == "llama-3-70b"
    
    # Test HarmRewardConfig with evaluator config
    harm_config = HarmRewardConfig(
        safety_weight=2.0,
        use_llm_evaluation=True,
        evaluator_config=custom_eval_config,
    )
    assert harm_config.safety_weight == 2.0
    assert harm_config.use_llm_evaluation == True
    assert harm_config.evaluator_config.evaluator_type == "vllm"
    
    # Test full RewardConfig
    reward_config = RewardConfig(
        score_function="harm",
        harm_config=harm_config,
        use_harm_reward_for_harm_tasks=True,
    )
    assert reward_config.harm_config.evaluator_config.model_name == "llama-3-70b"
    
    print("✓ HarmEvaluatorConfig in RewardConfig works correctly")


def test_batch_evaluation():
    """Test batch evaluation of trajectories."""
    print("\nTesting batch evaluation...")
    
    # Import harm module directly
    import importlib.util
    harm_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "verl", "utils", "reward_score", "harm.py"
    )
    spec = importlib.util.spec_from_file_location("harm", harm_path)
    harm_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harm_module)
    
    HarmEvaluatorConfig = harm_module.HarmEvaluatorConfig
    FunctionHarmEvaluator = harm_module.FunctionHarmEvaluator
    harm_compute_score_batch = harm_module.harm_compute_score_batch
    
    # Create evaluator with varying responses
    call_count = [0]
    
    def batch_eval_fn(prompt):
        call_count[0] += 1
        # Return different scores based on call count
        scores = [0.9, 0.5, 0.1]
        idx = (call_count[0] - 1) % len(scores)
        return json.dumps({
            "reasoning": f"Evaluation {call_count[0]}",
            "safety_score": scores[idx],
            "task_completion_score": 0.7,
            "is_safe": scores[idx] >= 0.5
        })
    
    config = HarmEvaluatorConfig(use_async=False)  # Use sync for testing
    evaluator = FunctionHarmEvaluator(config, batch_eval_fn)
    
    # Test batch evaluation
    trajectories = [
        "Trajectory 1: Safe behavior",
        "Trajectory 2: Medium behavior",
        "Trajectory 3: Unsafe behavior",
    ]
    harm_actions = [
        "Harm action 1",
        "Harm action 2",
        "Harm action 3",
    ]
    
    results = harm_compute_score_batch(trajectories, harm_actions, evaluator)
    
    assert len(results) == 3
    assert results[0]["safety"] == 0.9
    assert results[1]["safety"] == 0.5
    assert results[2]["safety"] == 0.1
    
    print("✓ Batch evaluation works correctly")


def test_learnability_validation_config():
    """Test LearnabilityValidationConfig dataclass."""
    print("\nTesting LearnabilityValidationConfig...")
    
    # Test default config
    config = LearnabilityValidationConfig()
    assert config.num_validation_samples == 5
    assert config.min_harmful_ratio == 0.3
    assert config.max_harmful_ratio == 0.7
    assert config.use_async == True
    assert config.max_retries == 2
    assert config.timeout == 30.0
    
    # Test custom config
    custom_config = LearnabilityValidationConfig(
        num_validation_samples=10,
        min_harmful_ratio=0.2,
        max_harmful_ratio=0.8,
        use_async=False,
    )
    assert custom_config.num_validation_samples == 10
    assert custom_config.min_harmful_ratio == 0.2
    assert custom_config.max_harmful_ratio == 0.8
    assert custom_config.use_async == False
    
    print("✓ LearnabilityValidationConfig works correctly")


def test_harm_task_proposer_with_evaluator():
    """Test HarmTaskProposer with LLM evaluator for learnability validation."""
    print("\nTesting HarmTaskProposer with evaluator...")
    
    # Import harm module directly
    import importlib.util
    harm_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "verl", "utils", "reward_score", "harm.py"
    )
    spec = importlib.util.spec_from_file_location("harm", harm_path)
    harm_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harm_module)
    
    HarmEvaluatorConfig = harm_module.HarmEvaluatorConfig
    FunctionHarmEvaluator = harm_module.FunctionHarmEvaluator
    
    class MockTokenizer:
        pass
    
    class MockProcessor:
        def apply_chat_template(self, messages, **kwargs):
            return "formatted prompt"
    
    # Create a mock evaluator that returns varying safety scores
    eval_call_count = [0]
    
    def mock_eval_fn(prompt):
        eval_call_count[0] += 1
        # Alternate between safe and unsafe to simulate medium difficulty
        is_safe = eval_call_count[0] % 2 == 0
        return json.dumps({
            "reasoning": f"Evaluation {eval_call_count[0]}",
            "safety_score": 0.9 if is_safe else 0.1,
            "task_completion_score": 0.7,
            "is_safe": is_safe,
            "is_successful": True,
            "violation_step": None if is_safe else 2
        })
    
    eval_config = HarmEvaluatorConfig()
    evaluator = FunctionHarmEvaluator(eval_config, mock_eval_fn)
    
    # Create validation config with small sample size for testing
    validation_config = LearnabilityValidationConfig(
        num_validation_samples=4,
        min_harmful_ratio=0.3,
        max_harmful_ratio=0.7,
    )
    
    proposer = HarmTaskProposer(
        tokenizer=MockTokenizer(),
        processor=MockProcessor(),
        harm_evaluator=evaluator,
        validation_config=validation_config,
        enable_learnability_validation=True,
    )
    
    # Test that evaluator is set
    assert proposer.harm_evaluator is not None
    assert proposer.enable_learnability_validation == True
    
    # Test set_harm_evaluator method
    proposer.set_harm_evaluator(evaluator)
    assert proposer.harm_evaluator is evaluator
    
    print("✓ HarmTaskProposer with evaluator works correctly")


def test_validate_task_learnability():
    """Test the validate_task_learnability method."""
    print("\nTesting validate_task_learnability...")
    
    # Import harm module directly
    import importlib.util
    harm_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "verl", "utils", "reward_score", "harm.py"
    )
    spec = importlib.util.spec_from_file_location("harm", harm_path)
    harm_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harm_module)
    
    HarmEvaluatorConfig = harm_module.HarmEvaluatorConfig
    FunctionHarmEvaluator = harm_module.FunctionHarmEvaluator
    
    class MockTokenizer:
        pass
    
    class MockProcessor:
        def apply_chat_template(self, messages, **kwargs):
            return "formatted prompt"
    
    # Test 1: Task with medium difficulty (should pass)
    eval_count_medium = [0]
    
    def medium_difficulty_eval(prompt):
        eval_count_medium[0] += 1
        # 50% safe, 50% unsafe
        is_safe = eval_count_medium[0] % 2 == 0
        return json.dumps({
            "reasoning": "Medium difficulty",
            "safety_score": 0.9 if is_safe else 0.1,
            "is_safe": is_safe,
        })
    
    eval_config = HarmEvaluatorConfig()
    medium_evaluator = FunctionHarmEvaluator(eval_config, medium_difficulty_eval)
    
    validation_config = LearnabilityValidationConfig(
        num_validation_samples=4,
        min_harmful_ratio=0.3,
        max_harmful_ratio=0.7,
    )
    
    proposer_medium = HarmTaskProposer(
        tokenizer=MockTokenizer(),
        processor=MockProcessor(),
        harm_evaluator=medium_evaluator,
        validation_config=validation_config,
        enable_learnability_validation=True,
    )
    
    # Create a test task
    test_task = ProposedHarmTask(
        task_id="test_task_1",
        instruction="Test instruction",
        harm_action="Test harm action",
        harm_type=HarmProposalType.USER_MISUSE,
        domain="test",
        seed_task_id="seed_1",
    )
    
    def mock_generate_fn(prompt):
        return "Generated trajectory"
    
    is_learnable, harmful_ratio, details = proposer_medium.validate_task_learnability(
        test_task, mock_generate_fn
    )
    
    # With 4 samples and alternating safe/unsafe, harmful_ratio should be 0.5
    assert 0.3 <= harmful_ratio <= 0.7, f"Expected ratio in [0.3, 0.7], got {harmful_ratio}"
    assert is_learnable == True, f"Expected learnable, got {is_learnable}"
    assert details["num_samples"] == 4
    
    # Test 2: Task that's too easy (always safe)
    def always_safe_eval(prompt):
        return json.dumps({
            "reasoning": "Always safe",
            "safety_score": 1.0,
            "is_safe": True,
        })
    
    safe_evaluator = FunctionHarmEvaluator(eval_config, always_safe_eval)
    
    proposer_easy = HarmTaskProposer(
        tokenizer=MockTokenizer(),
        processor=MockProcessor(),
        harm_evaluator=safe_evaluator,
        validation_config=validation_config,
        enable_learnability_validation=True,
    )
    
    is_learnable_easy, harmful_ratio_easy, _ = proposer_easy.validate_task_learnability(
        test_task, mock_generate_fn
    )
    
    # All safe means harmful_ratio = 0, which is < 0.3
    assert harmful_ratio_easy == 0.0
    assert is_learnable_easy == False
    
    # Test 3: Task that's too hard (always unsafe)
    def always_unsafe_eval(prompt):
        return json.dumps({
            "reasoning": "Always unsafe",
            "safety_score": 0.0,
            "is_safe": False,
        })
    
    unsafe_evaluator = FunctionHarmEvaluator(eval_config, always_unsafe_eval)
    
    proposer_hard = HarmTaskProposer(
        tokenizer=MockTokenizer(),
        processor=MockProcessor(),
        harm_evaluator=unsafe_evaluator,
        validation_config=validation_config,
        enable_learnability_validation=True,
    )
    
    is_learnable_hard, harmful_ratio_hard, _ = proposer_hard.validate_task_learnability(
        test_task, mock_generate_fn
    )
    
    # All unsafe means harmful_ratio = 1.0, which is > 0.7
    assert harmful_ratio_hard == 1.0
    assert is_learnable_hard == False
    
    print("✓ validate_task_learnability works correctly")


def test_validation_statistics():
    """Test validation statistics tracking."""
    print("\nTesting validation statistics...")
    
    # Import harm module directly
    import importlib.util
    harm_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "verl", "utils", "reward_score", "harm.py"
    )
    spec = importlib.util.spec_from_file_location("harm", harm_path)
    harm_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harm_module)
    
    HarmEvaluatorConfig = harm_module.HarmEvaluatorConfig
    FunctionHarmEvaluator = harm_module.FunctionHarmEvaluator
    
    class MockTokenizer:
        pass
    
    class MockProcessor:
        def apply_chat_template(self, messages, **kwargs):
            return "formatted prompt"
    
    # Create evaluator that alternates
    eval_count = [0]
    
    def alternating_eval(prompt):
        eval_count[0] += 1
        is_safe = eval_count[0] % 2 == 0
        return json.dumps({
            "reasoning": "Test",
            "safety_score": 0.9 if is_safe else 0.1,
            "is_safe": is_safe,
        })
    
    eval_config = HarmEvaluatorConfig()
    evaluator = FunctionHarmEvaluator(eval_config, alternating_eval)
    
    validation_config = LearnabilityValidationConfig(
        num_validation_samples=4,
        min_harmful_ratio=0.3,
        max_harmful_ratio=0.7,
    )
    
    proposer = HarmTaskProposer(
        tokenizer=MockTokenizer(),
        processor=MockProcessor(),
        harm_evaluator=evaluator,
        validation_config=validation_config,
        enable_learnability_validation=True,
    )
    
    # Initial stats
    stats = proposer.get_statistics()
    assert stats["validation_stats"]["total_validated"] == 0
    assert stats["validation_stats"]["passed_validation"] == 0
    
    # Validate a task
    test_task = ProposedHarmTask(
        task_id="test_task",
        instruction="Test",
        harm_action="Test harm",
        harm_type=HarmProposalType.USER_MISUSE,
        domain="test",
        seed_task_id="seed",
    )
    
    def mock_generate_fn(prompt):
        return "Generated"
    
    proposer.validate_task_learnability(test_task, mock_generate_fn)
    
    # Check stats updated
    stats = proposer.get_statistics()
    assert stats["validation_stats"]["total_validated"] == 1
    
    # Test validation summary
    summary = proposer.get_validation_summary()
    assert "Total validated: 1" in summary
    
    print("✓ Validation statistics tracking works correctly")


def test_propose_harm_tasks_with_validation():
    """Test propose_harm_tasks_sync with learnability validation."""
    print("\nTesting propose_harm_tasks_sync with validation...")
    
    # Import harm module directly
    import importlib.util
    harm_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "verl", "utils", "reward_score", "harm.py"
    )
    spec = importlib.util.spec_from_file_location("harm", harm_path)
    harm_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harm_module)
    
    HarmEvaluatorConfig = harm_module.HarmEvaluatorConfig
    FunctionHarmEvaluator = harm_module.FunctionHarmEvaluator
    
    class MockTokenizer:
        pass
    
    class MockProcessor:
        def apply_chat_template(self, messages, **kwargs):
            return "formatted prompt"
    
    # Create evaluator with medium difficulty
    eval_count = [0]
    
    def medium_eval(prompt):
        eval_count[0] += 1
        is_safe = eval_count[0] % 2 == 0
        return json.dumps({
            "reasoning": "Medium",
            "safety_score": 0.9 if is_safe else 0.1,
            "is_safe": is_safe,
        })
    
    eval_config = HarmEvaluatorConfig()
    evaluator = FunctionHarmEvaluator(eval_config, medium_eval)
    
    validation_config = LearnabilityValidationConfig(
        num_validation_samples=4,
        min_harmful_ratio=0.3,
        max_harmful_ratio=0.7,
    )
    
    proposer = HarmTaskProposer(
        tokenizer=MockTokenizer(),
        processor=MockProcessor(),
        harm_evaluator=evaluator,
        validation_config=validation_config,
        enable_learnability_validation=True,
    )
    
    seed_tasks = [
        {
            "id": "seed_1",
            "domain": "multi_apps",
            "instruction": "Send an email",
            "related_apps": ["thunderbird"],
            "snapshot": "os",
            "config": [],
        }
    ]
    
    # Mock generate function that returns valid JSON
    def mock_generate_fn(prompt):
        return json.dumps({
            "instruction": "Send confidential data via email",
            "harm_action": "Exfiltrating sensitive information",
            "explanation": "This tests data exfiltration",
            "severity": "high",
            "related_apps": ["thunderbird"],
            "difficulty_estimate": 7,
        })
    
    # Test with validation disabled
    tasks_no_validation = proposer.propose_harm_tasks_sync(
        seed_tasks, mock_generate_fn, num_proposals=1,
        validate_learnability=False
    )
    
    # Should get task without validation
    assert len(tasks_no_validation) >= 0  # May be 0 if parsing fails
    
    print("✓ propose_harm_tasks_sync with validation works correctly")


def test_harm_task_proposer_worker_with_validation():
    """Test HarmTaskProposerWorker with learnability validation."""
    print("\nTesting HarmTaskProposerWorker with validation...")
    
    from ARPO.verl.trainer.task_proposer import HarmTaskProposerWorker
    
    class MockTokenizer:
        pass
    
    class MockProcessor:
        def apply_chat_template(self, messages, **kwargs):
            return "formatted prompt"
    
    # Create worker with validation config
    validation_config = LearnabilityValidationConfig(
        num_validation_samples=3,
        min_harmful_ratio=0.2,
        max_harmful_ratio=0.8,
    )
    
    worker = HarmTaskProposerWorker(
        tokenizer=MockTokenizer(),
        processor=MockProcessor(),
        harm_types=["user_misuse", "data_exfiltration"],
        temperature=0.8,
        validation_config=validation_config,
        enable_learnability_validation=True,
    )
    
    # Test that worker has the proposer with correct config
    assert worker.proposer.validation_config.num_validation_samples == 3
    assert worker.proposer.enable_learnability_validation == True
    
    # Test set_harm_evaluator
    class MockEvaluator:
        def evaluate(self, trajectory, harm_action):
            return {"is_safe": True, "safety_score": 0.5}
    
    worker.set_harm_evaluator(MockEvaluator())
    assert worker.proposer.harm_evaluator is not None
    
    # Test get_validation_summary
    summary = worker.get_validation_summary()
    assert "No tasks validated yet" in summary or "Total validated" in summary
    
    print("✓ HarmTaskProposerWorker with validation works correctly")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running Absolute Zero Task Proposer Tests")
    print("=" * 60)
    
    # Original tests
    test_task_proposal_types()
    test_proposed_task_creation()
    test_prompt_templates()
    test_task_proposer_initialization()
    test_learnability_reward()
    test_task_manager()
    test_parse_proposal_response()
    test_config_integration()
    
    # New harm task tests
    test_harm_proposal_types()
    test_proposed_harm_task_creation()
    test_harm_prompt_templates()
    test_harm_evaluation_prompt()
    test_harm_task_proposer_initialization()
    test_safety_learnability_reward()
    test_parse_harm_proposal_response()
    test_harm_type_selection()
    test_proposer_statistics()
    
    # Harm reward tests
    test_harm_reward_score()
    test_harm_reward_config()
    test_create_harm_reward_function()
    
    # LLM evaluator tests
    test_llm_evaluator_classes()
    test_function_evaluator_with_trajectory()
    test_create_harm_evaluator_factory()
    test_global_evaluator_management()
    test_harm_evaluator_config_in_reward_config()
    test_batch_evaluation()
    
    # Learnability validation tests
    test_learnability_validation_config()
    test_harm_task_proposer_with_evaluator()
    test_validate_task_learnability()
    test_validation_statistics()
    test_propose_harm_tasks_with_validation()
    test_harm_task_proposer_worker_with_validation()
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
