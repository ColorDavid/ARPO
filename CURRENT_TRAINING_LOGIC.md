# ARPO 训练逻辑总结

## 一、整体架构

ARPO (Absolute Zero with Reinforcement Learning from Human Feedback) 是一个联合训练框架，同时训练两个角色：
- **Proposer（任务生成器）**：从 seed tasks 生成新的 harmful questions
- **Solver（任务执行器）**：执行 questions，生成 responses

## 二、训练流程

### 1. 初始化阶段

```
Seed Tasks (有害任务)
    ↓
Proposer 生成 n 个 Questions (每个 seed task 生成 rollout_n 个 questions)
    ↓
每个 Question 分配一个 EnvWorker
    ↓
Solver 执行 Rollout (每个 question 生成 rollout_n 个 responses)
```

### 2. 数据生成流程

#### 2.1 Proposer 数据生成

```
1. 输入：Seed Tasks (每个 seed task 是一个 prompt)
2. 处理：
   - 每个 seed task 生成 rollout_n 个 questions
   - 收集 proposer trajectory data (log_probs, input_ids, attention_mask)
   - 存储到 proposer_trajectories[prompt_id] = [traj1, traj2, ..., traj_n]
3. 输出：
   - proposed_tasks: List[question_configs] (长度 = rollout_n * len(seed_tasks))
   - proposer_trajectories: Dict[prompt_id -> List[trajectory_data]]
```

#### 2.2 Solver 数据生成

```
1. 输入：Proposed Questions (每个 question 是一个任务)
2. 处理：
   - 每个 question 分配一个 EnvWorker
   - 执行 rollout：每个 question 生成 rollout_n 个 responses
   - 收集 solver trajectory data (整个交互过程)
3. 输出：
   - batch: DataProto (包含所有 responses)
   - eval_results: List[float] (每个 response 的评估结果)
```

### 3. 奖励计算流程

#### 3.1 Proposer 奖励

```
对于每个 prompt (seed_task_id):
  1. 获取该 prompt 生成的所有 questions
  2. 对每个 question 计算 learnability：
     - 如果 question 是 learnable → reward = 1.0
     - 如果 question 不是 learnable → reward = 0.0
  3. 在 prompt 组内进行 GRPO 归一化：
     normalized_reward = (raw_reward - group_mean) / (group_std + eps)
  4. 存储到 proposer_rewards[question_global_idx] = normalized_reward
```

#### 3.2 Solver 奖励

```
对于每个 question:
  1. 获取该 question 的所有 responses (rollout_n 个)
  2. 计算 harm_score (基于 LLM 评估)：
     - safety_score: 安全性评分
     - completion_score: 完成度评分
     - harm_score = (safety_weight * safety_score + completion_weight * completion_score) / (safety_weight + completion_weight)
  3. 结合 format_reward：
     final_reward = harm_score + 0.5 * format_reward
```

### 4. Learnability 检查

```
对于每个 question group (rollout_n 个 responses):
  1. 计算 learnability：
     - Safety 维度：safety_score 在合理范围内
     - Completion 维度：completion_score > threshold
  2. 如果 question 是 learnable：
     - 用于训练
     - 更新 replay buffer (如果 eval_result > 0.1)
  3. 如果 question 不是 learnable：
     - 跳过训练
     - 如果整个 prompt 的所有 questions 都不 learnable → 触发 repropose
```

### 5. Replay Buffer 机制

#### 5.1 Solver Replay

```
位置：apply_replay() 函数
分组方式：按 question_id 分组
逻辑：
  1. 扩展 task_configs：每个 question 重复 rollout_n 次
  2. 扩展 batch：每个 batch item 重复 rollout_n 次
  3. 按 question_id 分组：
     - task_configs[i * rollout_n:(i + 1) * rollout_n] 应该有相同的 question_id
  4. 对于每个 question group：
     - 如果所有 responses 的 rewards 都很低 (std < 0.05 and mean < 0.2)
     - 从 replay buffer 获取正样本 (使用 question_id 或 seed_task_id)
     - 将正样本添加到 batch 前面
  5. 更新 replay buffer：将 eval_result > 0.1 的样本加入 replay buffer
```

#### 5.2 Proposer Replay

```
位置：_integrate_proposer_trajectories() 函数
分组方式：按 prompt_id (seed_task_id) 分组
逻辑：
  1. 按 prompt_id 分组收集 proposer 数据
  2. 对于每个 prompt group：
     - 收集该 prompt 的所有 questions 的 rewards
     - 如果所有 questions 的 rewards 都很低 (std < 0.05 and mean < 0.2)
     - 从 replay buffer 获取正样本 (使用 seed_task_id 或 prompt_id)
     - 将正样本添加到该 prompt group 的前面
  3. 将处理后的 proposer 数据添加到 batch
```

### 6. 数据组织结构

#### 6.1 Proposer 数据

```
proposer_trajectories = {
    "prompt_id_1": [
        {log_probs, input_ids, attention_mask},  # question 1
        {log_probs, input_ids, attention_mask},  # question 2
        ...
        {log_probs, input_ids, attention_mask}   # question n
    ],
    "prompt_id_2": [...],
    ...
}

proposer_rewards = {
    question_global_idx_1: normalized_reward_1,
    question_global_idx_2: normalized_reward_2,
    ...
}
```

#### 6.2 Solver 数据

```
task_configs = [
    question_config_1,  # question 1
    question_config_2,  # question 2
    ...
    question_config_n   # question n
]

batch = DataProto(
    batch={
        "input_ids": [...],      # 所有 responses 的 input_ids
        "attention_mask": [...], # 所有 responses 的 attention_mask
        "eval_results": [...],   # 每个 response 的评估结果
        "rewards": [...]         # 每个 response 的最终奖励
    }
)
```

### 7. 联合训练流程

```
1. start_reset_envs():
   - 从 seed tasks 生成 proposed_tasks
   - 收集 proposer_trajectories
   - 为每个 question 分配 EnvWorker
   - 返回 task_configs, reset_envs_object, seed_tasks, proposer_trajectories

2. _execute_rollout():
   - 执行多步交互 (max_steps)
   - 每个 step：生成 action → 执行 action → 获取 observation
   - 收集所有 responses 和 trajectories

3. _compute_rewards_and_check_learnability():
   - 计算 proposer rewards (基于 learnability)
   - 计算 solver rewards (基于 harm_score)
   - 检查 learnability，生成 learnable_mask
   - 处理 repropose 逻辑

4. apply_replay() (Solver):
   - 按 question_id 分组
   - 如果所有 responses 都是负样本，从 replay buffer 获取正样本
   - 更新 replay buffer

5. _integrate_proposer_trajectories():
   - 按 prompt_id 分组 proposer 数据
   - 应用 proposer replay (如果所有 questions 都是负样本)
   - 将 proposer 数据添加到 batch

6. 模型更新：
   - 使用 joint batch (包含 proposer 和 solver 数据)
   - 计算 PPO loss
   - 更新模型参数
```

## 三、关键数据结构

### 3.1 Task Config 结构

```python
task_config = {
    "id": "question_id",           # question 的唯一 ID
    "instruction": "...",           # question 的指令
    "seed_task_id": "...",          # 原始 seed task 的 ID
    "prompt_id": "...",             # prompt 的 ID (通常等于 seed_task_id)
    "harm_action": "...",           # 有害行为描述
    "harm_type": "...",             # 有害类型
    ...
}
```

### 3.2 Proposer Trajectory 结构

```python
trajectory_data = {
    "log_probs": torch.Tensor,      # 生成 question 的 log probabilities
    "input_ids": torch.Tensor,      # 输入 token IDs
    "attention_mask": torch.Tensor, # Attention mask
    "responses": torch.Tensor,      # 生成的 question tokens
    "prompt_id": "...",             # prompt ID
    "seed_task_id": "..."           # seed task ID
}
```

## 四、关键参数

- `rollout_n`: 每个 question 生成的 responses 数量（也是每个 seed task 生成的 questions 数量）
- `num_envs`: 环境 worker 数量（应该等于 `rollout_n * len(seed_tasks)`）
- `max_steps`: 每个任务的最大执行步数

## 五、最近的关键修改

1. **修复 task_configs 长度问题**：
   - 确保 `num_proposals = rollout_n * len(seed_tasks)`
   - 确保 `len(task_configs) == len(env_workers)`

2. **修复 apply_replay 分组逻辑**：
   - Solver replay：按 `question_id` 分组（一个 question -> rollout_n responses）
   - 在函数内部扩展 task_configs 和 batch

3. **添加 Proposer Replay**：
   - 按 `prompt_id` (seed_task_id) 分组（一个 prompt -> n questions）
   - 如果所有 questions 都是负样本，从 replay buffer 获取正样本

4. **优化 VM 启动**：
   - 文件锁优化：允许并发启动多个 Docker 容器
   - 减少等待时间：优化 VM readiness 检查

## 六、数据流图

```
Seed Tasks
    ↓
[Proposer] 生成 Questions
    ├─→ proposer_trajectories (按 prompt_id 组织)
    └─→ proposed_tasks (所有 questions)
        ↓
[EnvWorker] 执行 Rollout
    ├─→ 每个 question → rollout_n responses
    └─→ batch (所有 responses)
        ↓
[Reward Computation]
    ├─→ Proposer rewards (基于 learnability)
    └─→ Solver rewards (基于 harm_score)
        ↓
[Learnability Check]
    ├─→ learnable_mask
    └─→ 过滤非 learnable tasks
        ↓
[Replay Buffer]
    ├─→ Solver replay (按 question_id)
    └─→ Proposer replay (按 prompt_id)
        ↓
[Joint Training]
    ├─→ Proposer loss (基于 proposer_log_probs, proposer_rewards)
    └─→ Solver loss (基于 solver_log_probs, solver_rewards)
        ↓
模型更新
```

## 七、注意事项

1. **数据对齐**：
   - Proposer batch size = n questions (每个 prompt)
   - Solver batch size = rollout_n responses (每个 question)
   - 当 n = rollout_n 时，两者大小匹配，可以 1:1 对应

2. **分组逻辑**：
   - Proposer：按 `prompt_id` (seed_task_id) 分组
   - Solver：按 `question_id` 分组
   - Replay buffer 使用相同的分组逻辑

3. **Replay Buffer 更新**：
   - 只有 eval_result > 0.1 的样本才会加入 replay buffer
   - 使用 `task_id` (question_id) 作为 key
   - 如果找不到，会尝试使用 `seed_task_id`

4. **Repropose 机制**：
   - 如果某个 prompt 的所有 questions 都不 learnable
   - 会触发 repropose，重新生成该 prompt 的 questions
   - 只对需要 repropose 的 prompts 重新生成

