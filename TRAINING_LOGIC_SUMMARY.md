# ARPO 训练逻辑总结

## 一、整体架构

ARPO (Absolute Zero Relative Policy Optimization) 是一个用于 GUI 安全训练的强化学习框架，主要特点：

1. **单模型联合训练**：同时训练 Proposer（任务生成器）和 Solver（任务求解器）
2. **Absolute Zero 模式**：自动生成有害任务用于安全训练
3. **GRPO 算法**：使用 Group Relative Policy Optimization 计算 advantage
4. **统一 Reward/Filtering**：一次环境交互同时用于 reward 计算和 learnability 验证

## 二、训练流程概览

```
训练入口 (main.py)
    ↓
RayPPOTrainer.fit() - 主训练循环
    ↓
每个 Episode:
    1. 任务生成阶段 (start_reset_envs)
    2. 环境交互阶段 (rollout)
    3. Reward 计算阶段
    4. Advantage 计算阶段
    5. 策略更新阶段
```

## 三、详细训练步骤

### 3.1 初始化阶段

**文件位置**: `verl/trainer/main.py`, `verl/trainer/ray_trainer.py`

1. **创建资源池和 Worker**
   - 使用 Ray 分布式框架
   - 创建 ActorRollout、Critic、RefPolicy 等 Worker
   - 初始化环境 Workers（OSWorld 环境）

2. **创建数据加载器**
   - 从 `data.train_files` 加载 seed tasks
   - 支持随机采样或顺序采样

3. **初始化模型**
   - Actor 模型（用于生成动作）
   - Critic 模型（可选，用于价值估计）
   - Reference Policy（可选，用于 KL 约束）

### 3.2 主训练循环 (fit 方法)

**文件位置**: `verl/trainer/ray_trainer.py:2323`

#### 3.2.1 Episode 循环

```python
for episode in range(total_episodes):
    for batch_dict in train_dataloader:
        # 1. 任务生成和重置
        # 2. Rollout 执行
        # 3. Reward 计算
        # 4. Advantage 计算
        # 5. 策略更新
```

#### 3.2.2 任务生成阶段 (start_reset_envs)

**文件位置**: `verl/trainer/ray_trainer.py:1958`

**功能**：
- 如果启用 Absolute Zero，从 seed tasks 生成新的 harmful tasks
- 收集 Proposer 轨迹数据（prompt → question group）
- 重置环境，准备初始状态

**关键步骤**：
1. **任务提议**（如果启用）：
   ```python
   proposed_tasks, proposer_trajectories = self._propose_training_tasks(
       seed_tasks=seed_tasks,
       batch_size=num_tasks_needed,
   )
   ```
   - 为每个 seed task 生成 `rollout_n` 个不同的 questions
   - 收集 proposer 的 log_probs, input_ids, attention_mask 等

2. **环境重置**：
   ```python
   reset_envs_object = [
       worker.reset.remote(task_config) 
       for worker, task_config in zip(self.env_workers, task_configs)
   ]
   ```

**输出**：
- `task_configs`: 任务配置列表（每个任务重复 `rollout_n` 次）
- `proposer_trajectories`: Dict[prompt_id, List[trajectory_data]]

#### 3.2.3 Rollout 执行阶段 (_execute_rollout)

**文件位置**: `verl/trainer/ray_trainer.py:2049`

**功能**：执行多步环境交互，收集轨迹数据

**关键步骤**：
1. **多步交互循环**：
   ```python
   for step_idx in range(max_steps):
       # 1. 准备模型输入（观察消息）
       vllm_batch, valid_env_idx = self.prepare_vllm_inputs_full(env_outputs)
       
       # 2. 生成动作
       action_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
       
       # 3. 执行动作
       env_outputs = ray.get([
           worker.step.remote(action_text) 
           for worker, action_text in zip(cur_valid_envs, response_texts)
       ])
       
       # 4. 收集 format_reward（动作解析是否正确）
       if single_output['is_done']:
           format_rewards[cur_env_idx] = single_output['format_reward']
           eval_results_objects[cur_env_idx] = self.env_workers[cur_env_idx].evaluate.remote()
   ```

2. **环境评估**：
   ```python
   eval_results = ray.get(eval_results_objects)  # 0.0 或 1.0
   ```

3. **准备训练数据**：
   ```python
   batch = collate_fn_dataproto(process_results)
   batch.batch["eval_results"] = torch.tensor(eval_results)
   batch.batch["format_rewards"] = torch.tensor(format_rewards)
   ```

**输出**：
- `eval_results`: 环境评估结果（0.0 或 1.0）
- `format_rewards`: 格式奖励（0.0 或 -1.0）
- `batch`: DataProto 包含 input_ids, attention_mask, responses, log_probs 等

#### 3.2.4 Reward 计算阶段 (_compute_rewards_and_check_learnability)

**文件位置**: `verl/trainer/ray_trainer.py:2157`

**功能**：计算最终 reward，并检查任务的 learnability

##### 标准模式（非 Absolute Zero）

```python
rewards = eval_results + 0.5 * format_rewards
```

**公式**：
- `final_reward = eval_results + 0.5 * format_rewards`
- `eval_results`: 0.0（失败）或 1.0（成功）
- `format_rewards`: 0.0（格式正确）或 -1.0（格式错误）

##### Absolute Zero 模式

**文件位置**: `verl/trainer/ray_trainer.py:1199` (_compute_unified_reward_and_filter)

**核心流程**：

1. **计算分数**（对每个 rollout）：
   ```python
   scores = self._compute_scores_for_task(task_config, trajectory, eval_result)
   ```
   - `safety_score`: LLM 评估 agent 是否避免了有害行为（0.0-1.0）
   - `completion_score`: 任务完成分数（LLM 或 Rule 评估，0.0-1.0）
   - `combined_reward`: 加权组合
     ```
     combined_reward = (safety_weight * safety_score + 
                       completion_weight * completion_score) / 
                      (safety_weight + completion_weight)
     ```

2. **Learnability 检查**（双维度过滤）：
   ```python
   is_learnable, details = self._check_learnability(scores_list, task_id)
   ```
   
   **Safety 维度**：
   - `safe_ratio = count(safety_score > threshold) / n`
   - 条件：`min_safe_ratio <= safe_ratio <= max_safe_ratio`
   
   **Completion 维度**：
   - `incomplete_ratio = count(completion_score < threshold) / n`
   - 条件：`min_incomplete_ratio <= incomplete_ratio <= max_incomplete_ratio`
   
   **两个条件都必须满足**，任务才被认为是 learnable

3. **采样决策**：
   - 对于每个 prompt 的 question group：
     - 如果所有 questions 都不 learnable → 标记 repropose (sampled_idx = -1)
     - 如果所有 questions 都 learnable → 标记 repropose (sampled_idx = -1)
     - 如果混合状态 → 随机采样一个 learnable question

4. **Repropose 处理**：
   - 如果 `needs_repropose = True`，进入 repropose 循环
   - 重新生成 question group 并执行 rollout
   - 最多尝试 `max_repropose_attempts` 次

**输出**：
- `harm_rewards`: 每个样本的 reward
- `learnable_mask`: 每个任务是否 learnable
- `sampled_indices`: 每个 prompt 采样到的 question 索引

#### 3.2.5 Proposer Reward 计算

**文件位置**: `verl/trainer/ray_trainer.py:876` (_compute_proposer_rewards)

**功能**：为 Proposer 计算 reward

**逻辑**：
1. **Raw Reward**：
   - 如果 question 是 learnable → `raw_reward = 1.0`
   - 否则 → `raw_reward = 0.0`

2. **GRPO 标准化**（组内标准化）：
   ```python
   normalized_reward = (raw_reward - group_mean) / (group_std + eps)
   ```

#### 3.2.6 整合 Proposer 轨迹数据

**文件位置**: `verl/trainer/ray_trainer.py:959` (_integrate_proposer_trajectories)

**功能**：将 Proposer 轨迹数据整合到 batch 中，支持联合训练

**添加的数据**：
- `proposer_log_probs`: Proposer 的 log probabilities
- `proposer_input_ids`: Proposer 的输入序列
- `proposer_attention_mask`: Proposer 的 attention mask
- `proposer_rewards`: Proposer 的 rewards
- `proposer_advantages`: Proposer 的 advantages（用于训练）

#### 3.2.7 Advantage 计算

**文件位置**: `verl/trainer/ray_trainer.py:176` (compute_advantage)

**算法**：GRPO (Group Relative Policy Optimization)

**核心思想**：
1. 根据 `task_id` 将样本分组
2. 计算每组 reward 的均值和标准差
3. 标准化每个样本的 reward：
   ```python
   normalized_score = (score - group_mean) / (group_std + eps)
   ```

**代码位置**: `verl/trainer/core_algos.py:269` (compute_grpo_outcome_advantage)

**优势**：
- 减少不同任务难度差异的影响
- 突出组内相对表现
- 提高训练稳定性

#### 3.2.8 策略更新阶段

**文件位置**: `verl/workers/actor/dp_actor.py:212` (update_policy)

**功能**：使用 PPO 算法更新策略参数

**关键步骤**：

1. **Solver 训练**：
   ```python
   # 计算新的 log_probs
   log_probs = self._forward_micro_batch(model_inputs, temperature)
   
   # 计算 policy loss
   pg_loss, clipfrac_higher, clipfrac_lower, ppo_kl = core_algos.compute_policy_loss(
       old_log_probs=old_log_probs,
       log_probs=log_probs,
       advantages=advantages,
       response_mask=response_mask,
       clip_ratio_low=clip_ratio_low,
       clip_ratio_high=clip_ratio_high,
   )
   
   # 可选：添加 KL loss
   if use_kl_loss:
       kl_loss = VF.masked_mean(kld, response_mask)
       pg_loss = pg_loss + kl_loss * kl_coef
   ```

2. **Proposer 训练**（联合训练）：
   ```python
   if "proposer_log_probs" in model_inputs:
       # 计算 proposer 的 log_probs
       proposer_log_probs_new = self._forward_micro_batch(proposer_model_inputs, temperature)
       
       # 计算 proposer policy loss
       proposer_pg_loss = core_algos.compute_policy_loss(
           old_log_probs=proposer_old_log_probs,
           log_probs=proposer_log_probs_new,
           advantages=proposer_advantages,
           response_mask=proposer_response_mask,
       )
       
       # 合并 loss
       total_loss = pg_loss + proposer_pg_loss
   ```

3. **梯度更新**：
   ```python
   loss.backward()
   grad_norm = self._optimizer_step()
   ```

**关键参数**：
- `ppo_epochs`: PPO 更新轮数
- `clip_ratio_low/high`: PPO clip 范围
- `kl_coef`: KL 散度系数
- `max_grad_norm`: 梯度裁剪阈值

## 四、关键组件

### 4.1 环境交互 (GUI Agent)

**文件位置**: `verl/trainer/gui_agent.py`

**功能**：
- 封装 OSWorld 环境
- 处理多模态输入（文本 + 图像）
- 执行动作并返回结果

**关键方法**：
- `reset(task_config)`: 重置环境
- `step(action_text)`: 执行动作
- `evaluate()`: 评估任务完成情况

### 4.2 任务管理器 (Task Manager)

**文件位置**: `verl/trainer/task_proposer.py`

**功能**：
- 管理 seed tasks 和 proposed tasks
- 生成新的 harmful tasks
- 跟踪任务性能

**关键类**：
- `AbsoluteZeroTaskManager`: 管理任务生命周期
- `HarmTaskProposer`: 生成有害任务

### 4.3 数据格式 (DataProto)

**文件位置**: `verl/protocol.py`

**功能**：
- 统一的数据格式，支持 tensor 和 non-tensor 数据
- 支持 batch 操作（split, concat, reorder 等）

**关键字段**：
- `batch`: Dict[str, torch.Tensor] - 张量数据
- `non_tensor_batch`: Dict[str, Any] - 非张量数据
- `meta_info`: Dict[str, Any] - 元信息

## 五、配置参数

### 5.1 核心训练参数

```yaml
worker:
  rollout:
    n: 5                    # 每个任务的 rollout 数量
    temperature: 1.0        # 生成温度
  
  actor:
    global_batch_size: 128
    micro_batch_size_per_device_for_update: 4
    ppo_epochs: 1
    clip_ratio_low: 0.2
    clip_ratio_high: 0.3

algorithm:
  adv_estimator: grpo      # 使用 GRPO
  use_kl_loss: true
  kl_coef: 1.0e-2

env:
  num_envs: 32             # 环境数量
  max_steps: 6             # 最大步数
```

### 5.2 Absolute Zero 配置

```yaml
absolute_zero:
  enabled: true
  enable_task_proposal: true
  
  # Learnability 参数
  learnability_safety_threshold: 0.5
  learnability_min_safe_ratio: 0.3
  learnability_max_safe_ratio: 0.7
  learnability_completion_threshold: 0.5
  learnability_min_incomplete_ratio: 0.3
  learnability_max_incomplete_ratio: 0.7
  
  # Reward 权重
  safety_reward_weight: 0.7
  task_completion_weight: 0.3
  
  # Completion 评估方式
  completion_evaluator_type: "rule"  # "llm" 或 "rule"
```

## 六、训练数据流

```
Seed Tasks (Dataloader)
    ↓
Task Proposal (如果启用)
    ↓
Proposed Tasks (每个任务 rollout_n 次)
    ↓
Environment Rollout
    ↓
Trajectories + Eval Results
    ↓
Reward Computation
    ├─→ Safety Score (LLM)
    ├─→ Completion Score (LLM/Rule)
    └─→ Combined Reward
    ↓
Learnability Check
    ├─→ Safety 维度过滤
    └─→ Completion 维度过滤
    ↓
Sampling Decision
    ├─→ Learnable → 保留
    └─→ Non-learnable → 跳过/Repropose
    ↓
GRPO Advantage Computation
    ↓
Policy Update (PPO)
    ├─→ Solver Loss
    └─→ Proposer Loss (联合训练)
```

## 六.1、Proposer 和 Solver 的训练数据详解

### Proposer 的训练数据

**数据来源**：任务生成阶段（`start_reset_envs` → `_propose_training_tasks`）

**数据流程**：
```
Seed Task (instruction + harm_action)
    ↓
Mutation Prompt (文本提示，要求生成新的 harmful task)
    ↓
Proposer 生成 Question
    ├─→ 输入：prompt (mutation prompt)
    ├─→ 输出：generated_text (新的 instruction + harm_action)
    └─→ 收集轨迹数据
```

**Proposer 轨迹数据**（在生成 question 时收集）：

```python
trajectory_data = {
    "log_probs": rollout_log_probs,      # [1, question_length] - 生成 question 的 log probabilities
    "input_ids": input_ids,              # [1, seq_len] - 完整序列 (prompt + question)
    "attention_mask": attention_mask,    # [1, seq_len] - attention mask
    "responses": responses,              # [1, question_length] - 生成的 question tokens
    "prompt_id": prompt_id,              # prompt 标识符
    "seed_task_id": seed_task_id,        # seed task 标识符
}
```

**数据规模**：
- 每个 prompt 生成 `n` 个 questions（question group）
- 例如：`n = 5`，则每个 prompt 有 5 个 questions
- Proposer batch size = `n` questions（使用完整的 question group）

**Proposer Reward 计算**：

```python
# 1. Raw Reward（基于 learnability，间接基于 Solver 的表现）
# Learnability 是通过评估 Solver 的 responses 来判断的
# 但 Proposer 训练时不需要直接使用 Solver 的 response 数据
if question is learnable:
    raw_reward = 1.0
else:
    raw_reward = 0.0

# 2. GRPO 标准化（组内标准化）
group_mean = mean([rewards for all questions in same prompt])
group_std = std([rewards for all questions in same prompt])
normalized_reward = (raw_reward - group_mean) / (group_std + eps)

# 3. 转换为 token-level rewards
proposer_token_level_rewards = normalized_reward.unsqueeze(-1).expand(-1, question_length)
proposer_advantages = proposer_token_level_rewards  # 直接作为 advantage
```

**重要说明**：
- Proposer 训练**只需要 question**（生成的文本），不需要 Solver 的 response
- Proposer 的 reward 基于 question 的 learnability，而 learnability 是通过评估 Solver 的 responses 来判断的
- 但训练时，Proposer 只使用自己的 question 数据（input_ids, log_probs 等），不直接使用 Solver 的 response 数据

**训练时的数据格式**：

```python
batch.batch["proposer_log_probs"]        # [n, question_length] - old log probs（生成 question 时的）
batch.batch["proposer_input_ids"]        # [n, seq_len] - 完整输入序列（prompt + question）
batch.batch["proposer_attention_mask"]   # [n, seq_len] - attention mask
batch.batch["proposer_rewards"]          # [n] - 标量 rewards（基于 learnability）
batch.batch["proposer_advantages"]       # [n, question_length] - token-level advantages
batch.batch["proposer_response_mask"]    # [n, question_length] - response mask（question 部分的 mask）

# 注意：proposer_responses 是从 proposer_input_ids 中提取的 question tokens
# proposer_responses = proposer_input_ids[:, -question_length:]  # 提取 question 部分
```

**训练目标**：
- 学习生成 **learnable** 的 questions
- Learnable = 既不太简单（all safe），也不太困难（all unsafe）
- 通过 GRPO 组内对比，鼓励生成中等难度的任务

**关键点**：
- ✅ Proposer 训练**只需要 question**（生成的文本）
- ❌ Proposer 训练**不需要 Solver 的 response** 数据
- ✅ Proposer 的 reward 基于 question 的 learnability（间接基于 Solver 的表现，但不直接使用 response 数据）

---

### Solver 的训练数据

**数据来源**：环境交互阶段（`_execute_rollout`）

**数据流程**：
```
Question (instruction + harm_action)
    ↓
Environment Observation (多模态：文本 + 图像)
    ↓
Solver 生成 Response (动作)
    ├─→ 输入：observation messages (历史对话 + 当前屏幕截图)
    ├─→ 输出：action_text (动作文本)
    └─→ 收集轨迹数据
```

**Solver 轨迹数据**（在生成 response 时收集）：

```python
# 从环境交互中收集
batch.batch["input_ids"]         # [batch_size, seq_len] - 完整序列 (prompt + response)
batch.batch["attention_mask"]    # [batch_size, seq_len] - attention mask
batch.batch["responses"]         # [batch_size, response_length] - 生成的 response tokens
batch.batch["old_log_probs"]     # [batch_size, response_length] - rollout 时的 log probs
batch.batch["response_mask"]     # [batch_size, response_length] - response mask
```

**数据规模**：
- **Rollout 阶段**：每个 question 执行 `rollout_n` 次 rollout
  - 每个 prompt 有 `n` 个 questions
  - 所以总共有 `n * rollout_n` 个 responses（所有 questions 的所有 rollouts）
- **采样后**：只保留采样到的 1 个 question 的 `rollout_n` 个 responses
  - 例如：`n = 5`, `rollout_n = 5`
  - Rollout 阶段：5 × 5 = 25 个 responses
  - 采样后：只保留 1 个 question 的 5 个 responses
- Solver batch size = `rollout_n` responses（采样后的数据）

**Solver Reward 计算**：

```python
# 1. 环境评估
eval_results = env.evaluate()  # 0.0 或 1.0

# 2. 格式奖励
format_rewards = 0.0 (格式正确) 或 -1.0 (格式错误)

# 3. Harm Reward（如果启用 Absolute Zero）
if is_harm_task:
    safety_score = LLM_evaluate_safety(trajectory)      # 0.0-1.0
    completion_score = evaluate_completion(trajectory)  # 0.0-1.0
    combined_reward = (safety_weight * safety_score + 
                      completion_weight * completion_score) / 
                     (safety_weight + completion_weight)
else:
    combined_reward = eval_results

# 4. 最终 Reward
final_reward = combined_reward + 0.5 * format_rewards

# 5. GRPO Advantage（组内标准化）
# 根据 task_id 分组，组内标准化
normalized_reward = (reward - group_mean) / (group_std + eps)
advantage = normalized_reward
```

**训练时的数据格式**：

```python
batch.batch["input_ids"]         # [rollout_n, seq_len]
batch.batch["attention_mask"]    # [rollout_n, seq_len]
batch.batch["responses"]         # [rollout_n, response_length]
batch.batch["old_log_probs"]     # [rollout_n, response_length]
batch.batch["advantages"]        # [rollout_n, response_length] - token-level advantages
batch.batch["response_mask"]     # [rollout_n, response_length]
batch.batch["rewards"]           # [rollout_n] - 标量 rewards
```

**训练目标**：
- 学习执行 **安全且有效** 的动作
- 对于 harmful tasks：拒绝执行有害操作（safety_score 高）
- 对于正常 tasks：正确完成任务（completion_score 高）

---

### 联合训练的数据对齐

**关键点**：Proposer 和 Solver 的数据需要对齐才能联合训练

**数据流程详解**：

1. **Rollout 阶段**（所有数据）：
   ```
   每个 Prompt:
   ├─→ Question 0: [Response 0-0, Response 0-1, ..., Response 0-(rollout_n-1)]  (rollout_n 个)
   ├─→ Question 1: [Response 1-0, Response 1-1, ..., Response 1-(rollout_n-1)]  (rollout_n 个)
   ├─→ ...
   └─→ Question (n-1): [Response (n-1)-0, ..., Response (n-1)-(rollout_n-1)]  (rollout_n 个)
   
   总计：n × rollout_n 个 responses
   ```

2. **采样阶段**（选择 learnable question）：
   ```
   对每个 Prompt:
   ├─→ 检查每个 Question 的 learnability
   ├─→ 采样 1 个 learnable Question（例如 Question 2）
   └─→ 只保留这个 Question 的 rollout_n 个 responses
   
   最终：每个 Prompt 只有 1 个 Question 的 rollout_n 个 responses
   ```

3. **训练数据对齐**：
   ```python
   # Proposer: 使用完整的 n 个 questions（所有 questions）
   proposer_batch_size = n questions          # 例如：5
   
   # Solver: 使用采样后的 1 个 question 的 rollout_n 个 responses
   solver_batch_size = rollout_n responses   # 例如：5
   
   # 当 n = rollout_n 时，数据可以 1:1 匹配
   # 匹配关系（假设采样到 Question 2）：
   # Proposer Question 0 → (不用于训练，但用于计算 proposer reward)
   # Proposer Question 1 → (不用于训练，但用于计算 proposer reward)
   # Proposer Question 2 → Solver Response 2-0, 2-1, 2-2, 2-3, 2-4  (用于训练)
   # Proposer Question 3 → (不用于训练，但用于计算 proposer reward)
   # Proposer Question 4 → (不用于训练，但用于计算 proposer reward)
   ```

**重要说明**：
- Proposer 使用**所有 `n` 个 questions** 计算 reward（GRPO 组内对比）
- Solver 只使用**采样后的 1 个 question** 的 `rollout_n` 个 responses 进行训练
- 当 `n = rollout_n` 时，Proposer 的 `n` 个 questions 与 Solver 的 `rollout_n` 个 responses 数量相同，可以 1:1 匹配训练

**联合训练 Loss**：
```python
total_loss = solver_pg_loss + proposer_pg_loss

# Solver Loss
solver_pg_loss = PPO_loss(
    old_log_probs=solver_old_log_probs,
    log_probs=solver_new_log_probs,
    advantages=solver_advantages,
)

# Proposer Loss
proposer_pg_loss = PPO_loss(
    old_log_probs=proposer_old_log_probs,
    log_probs=proposer_new_log_probs,
    advantages=proposer_advantages,
)
```

---

### 数据收集时机对比

| 阶段 | Proposer 数据 | Solver 数据 |
|------|--------------|-------------|
| **收集时机** | 任务生成阶段（`start_reset_envs`） | 环境交互阶段（`_execute_rollout`） |
| **收集位置** | `_create_generate_fn()` → `generate_sequences()` | `actor_rollout_wg.generate_sequences()` |
| **输入格式** | 文本 prompt（无图像） | 多模态 messages（文本 + 图像） |
| **输出内容** | Question（instruction + harm_action） | Response（动作文本） |
| **数据规模** | `n` questions per prompt | `rollout_n` responses per question |
| **Reward 来源** | Learnability（基于 Solver 的表现） | 环境评估 + Harm 评估 |

---

### 数据流示例

**完整流程示例**（假设 `n = 3`, `rollout_n = 3`）：

```
1. Proposer 阶段：
   Seed Task → Mutation Prompt → Generate Questions
   ├─→ 收集：proposer_log_probs, proposer_input_ids, ...
   └─→ 生成：Question A, Question B, Question C (n=3)

2. Solver Rollout 阶段（所有 questions）：
   Question A → Environment → Generate Responses
   ├─→ Rollout 1: Response A1
   ├─→ Rollout 2: Response A2
   └─→ Rollout 3: Response A3
   
   Question B → Environment → Generate Responses
   ├─→ Rollout 1: Response B1
   ├─→ Rollout 2: Response B2
   └─→ Rollout 3: Response B3
   
   Question C → Environment → Generate Responses
   ├─→ Rollout 1: Response C1
   ├─→ Rollout 2: Response C2
   └─→ Rollout 3: Response C3
   
   总计：3 × 3 = 9 个 responses
   收集：solver_log_probs, solver_input_ids, ... (所有 9 个)

3. Learnability 检查：
   Question A: [Response A1, A2, A3] → is_learnable = True
   Question B: [Response B1, B2, B3] → is_learnable = False
   Question C: [Response C1, C2, C3] → is_learnable = True

4. 采样决策：
   → 从 learnable questions (A, C) 中随机采样
   → 假设采样到 Question A

5. 数据过滤：
   → 只保留 Question A 的 3 个 responses: [A1, A2, A3]
   → 丢弃 Question B 和 Question C 的 responses

6. 联合训练：
   Proposer: [Question A, B, C] (3 个，用于计算 proposer reward)
   Solver: [Response A1, A2, A3] (3 个，用于训练)
   → 当 n = rollout_n 时，可以 1:1 匹配训练
```

这样设计的好处：
- **单模型联合训练**：Proposer 和 Solver 共享参数，同时优化
- **数据对齐**：通过采样机制确保数据匹配
- **端到端优化**：Proposer 学习生成更好的任务，Solver 学习更好地执行任务

## 七、关键算法

### 7.1 GRPO (Group Relative Policy Optimization)

**核心公式**：
```
normalized_score = (score - group_mean) / (group_std + eps)
advantage = normalized_score
```

**优势**：
- 组内标准化，减少任务难度差异
- 提高训练稳定性
- 适合 outcome-based reward

### 7.2 PPO (Proximal Policy Optimization)

**核心公式**：
```
ratio = exp(log_prob - old_log_prob)
clipped_ratio = clip(ratio, 1-clip_ratio, 1+clip_ratio)
pg_loss = -min(ratio * advantage, clipped_ratio * advantage)
```

**特点**：
- 使用 clip 机制防止策略更新过大
- 支持多轮更新（ppo_epochs）
- 可选 KL 散度约束

### 7.3 联合训练 (Joint Training)

**设计**：
- 单模型同时训练 Proposer 和 Solver
- Proposer: prompt → question group
- Solver: question → response group
- 共享模型参数，分别计算 loss

**Loss 组合**：
```
total_loss = solver_pg_loss + proposer_pg_loss
```

## 八、性能优化

1. **并行化**：
   - 使用 Ray 分布式框架
   - 多环境并行交互
   - 多 GPU 数据并行

2. **内存优化**：
   - FSDP (Fully Sharded Data Parallel)
   - CPU Offload
   - Gradient Checkpointing

3. **计算优化**：
   - VLLM 高效推理
   - Padding-free attention
   - Sequence length balancing

## 九、调试和监控

### 9.1 关键日志

- `[Rollout]`: Rollout 过程日志
- `[HarmScore]`: Harm reward 计算日志
- `[Repropose]`: Repropose 过程日志
- `[AbsoluteZero]`: Absolute Zero 模式日志

### 9.2 关键指标

- `reward/mean`: 平均 reward
- `harm/safety_score_mean`: 平均安全分数
- `harm/completion_score_mean`: 平均完成分数
- `unified_filter/skipped_tasks`: 跳过的任务数
- `actor/pg_loss`: Policy gradient loss
- `actor/kl_loss`: KL divergence loss

## 十、总结

ARPO 的训练逻辑是一个复杂的强化学习流程，主要特点：

1. **联合训练**：Proposer 和 Solver 同时训练
2. **自适应任务生成**：Absolute Zero 自动生成训练任务
3. **智能过滤**：Learnability 检查确保任务质量
4. **高效算法**：GRPO + PPO 组合
5. **统一模式**：一次环境交互用于 reward 和验证

整个流程设计合理，既保证了训练效率，又确保了任务质量和模型安全性。

