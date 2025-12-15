# Absolute Zero 联合更新完整流程图

## 核心设计

```
┌─────────────────────────────────────────────────────────────────────────┐
│              单模型联合更新：Proposer + Solver                            │
│  - Proposer: prompt → question group (收集 logp, reward)                 │
│  - Solver: question → response group (收集 logp, reward)                │
│  - 采样后：每个 prompt 保留 1 个 question + 1 个 response group          │
│  - 整合两类轨迹数据到 dataproto，支撑联合更新                            │
└─────────────────────────────────────────────────────────────────────────┘
```

## 完整流程图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          训练批次开始 (fit())                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
        ┌───────────────────────────────────────────┐
        │ 1. start_reset_envs()                     │
        │    - 从 seed tasks 生成 proposed tasks     │
        │    - 收集 proposer 轨迹数据                 │
        └───────────────────────────────────────────┘
                    │
                    ├──────────────────────────────────────────────┐
                    │                                              │
                    ▼                                              ▼
    ┌───────────────────────────────┐      ┌───────────────────────────────┐
    │ 1.1 Proposer 生成             │      │ 1.2 返回数据                  │
    │     (prompt → question group) │      │                               │
    │                               │      │ - task_configs                │
    │ generate_fn(prompt, metadata) │      │ - proposer_trajectories        │
    │   → (question_text,           │      │   (prompt_id → [traj_data])   │
    │      trajectory_data)         │      │                               │
    │                               │      │ trajectory_data 包含:         │
    │ trajectory_data:              │      │ - log_probs                   │
    │ - log_probs                   │      │ - input_ids                   │
    │ - input_ids                   │      │ - attention_mask              │
    │ - attention_mask              │      │ - responses                   │
    │ - responses                   │      │                               │
    │                               │      │ 保存到 task_config.metadata   │
    └───────────────────────────────┘      └───────────────────────────────┘
                    │                                              │
                    └──────────────────┬───────────────────────────┘
                                       │
                                       ▼
        ┌───────────────────────────────────────────┐
        │ 2. 环境重置 (env.reset)                     │
        │    - 为每个 task_config 重置环境           │
        │    - 准备初始状态                          │
        └───────────────────────────────────────────┘
                    │
                    ▼
        ┌───────────────────────────────────────────┐
        │ 3. Solver Rollout 执行                    │
        │    (多步交互循环)                          │
        │                                            │
        │    for step in max_steps:                 │
        │      ├─→ actor.generate() → actions       │
        │      │   (收集 solver log_probs)          │
        │      │                                    │
        │      └─→ env.step(actions) → env_outputs │
        │          (重复直到完成)                    │
        └───────────────────────────────────────────┘
                    │
                    ▼
        ┌───────────────────────────────────────────┐
        │ 4. 环境评估                                │
        │    - env.evaluate() → eval_results        │
        │    - env.get_history_messages() →         │
        │      trajectories                         │
        └───────────────────────────────────────────┘
                    │
                    ▼
        ┌───────────────────────────────────────────┐
        │ 5. 准备 Solver Batch                      │
        │    - 从 env workers 获取训练数据           │
        │    - 包含: input_ids, attention_mask,     │
        │      responses, log_probs (solver)        │
        └───────────────────────────────────────────┘
                    │
                    ▼
        ┌───────────────────────────────────────────┐
        │ 6. _compute_unified_reward_and_filter()   │
        │    (统一 Reward 计算和 Learnability 过滤)   │
        └───────────────────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        │                       │
        ▼                       ▼
┌───────────────┐      ┌───────────────┐
│ 6.1 计算分数   │      │ 6.2 分组      │
│               │      │               │
│ for each      │      │ prompt_to_    │
│ (task_config, │      │ questions =   │
│  trajectory,  │      │ {prompt_id:   │
│  eval_result):│      │  [q1, q2, ...]│
│   scores =    │      │ }             │
│     _compute_ │      │               │
│     scores_   │      │ 每个 prompt → │
│     for_task()│      │ 1 个 question │
│               │      │ group         │
│   - safety_   │      │               │
│     score     │      │               │
│   - completion│      │               │
│     _score    │      │               │
│   - combined_ │      │               │
│     reward    │      │               │
└───────────────┘      └───────────────┘
        │                       │
        └───────────┬───────────┘
                    ▼
        ┌───────────────────────────────────────────┐
        │ 6.3 Learnability 检查                     │
        │                                            │
        │ for each question in group:               │
        │   group_scores = [scores for rollout_n]   │
        │   is_learnable = _check_learnability()    │
        │                                            │
        │   双维度过滤:                              │
        │   ├─ Safety: safe_ratio 在范围内?         │
        │   └─ Completion: incomplete_ratio 在范围内?│
        └───────────────────────────────────────────┘
                    │
                    ▼
        ┌───────────────────────────────────────────┐
        │ 6.4 Repropose 检查                         │
        │                                            │
        │ for each prompt's question group:         │
        │   all_skip = all(not learnable)          │
        │   all_not_skip = all(learnable)          │
        │                                            │
        │   if all_skip OR all_not_skip:           │
        │     → 标记 repropose (sampled_idx = -1)  │
        │   else:                                   │
        │     → 混合状态，可以采样                    │
        └───────────────────────────────────────────┘
                    │
                    ▼
        ┌───────────────────────────────────────────┐
        │ 6.5 采样决策                                 │
        │ (决定采样哪个 question，返回索引)            │
        │                                            │
        │ for each prompt (混合状态):                │
        │   learnable_questions = [q for q in       │
        │                          questions       │
        │                          if learnable]    │
        │                                            │
        │   if len(learnable_questions) > 0:        │
        │     sampled_idx = random.choice(           │
        │       learnable_questions)                 │
        │   else:                                   │
        │     sampled_idx = -1                       │
        │                                            │
        │   返回:                                    │
        │   - harm_rewards (solver rewards)         │
        │   - learnable_mask                        │
        │   - sampled_indices (question indices)    │
        │     (>=0: 采样索引, -1: 需要 repropose)      │
        └───────────────────────────────────────────┘
                    │
                    ▼
        ┌───────────────────────────────────────────┐
        │ 7. 检查是否需要 Repropose                  │
        │                                            │
        │ needs_repropose = any(                    │
        │   idx < 0 for idx in sampled_indices)     │
        └───────────────────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        │                       │
    否  │                       │   是
        │                       │
        ▼                       ▼
┌───────────────┐      ┌───────────────────────────────┐
│ (跳过 repropose)│      │ 9. Repropose 循环              │
│               │      │                                │
│ 直接进入步骤 10│      │ while repropose_attempt <     │
│               │      │   max_repropose_attempts:     │
│               │      │                                │
│               │      │   9.1 重新 propose             │
│               │      │       - 重新生成 question group│
│               │      │       - 收集新的 proposer      │
│               │      │         trajectories           │
│               │      │       - 更新 proposer_         │
│               │      │         trajectories          │
│               │      │                                │
│               │      │   9.2 重新执行 solver rollout  │
│               │      │       - 重新生成 response group│
│               │      │       - 收集 format_rewards    │
│               │      │       - 更新 batch 中的 solver │
│               │      │         trajectories          │
│               │      │         (input_ids, attention_ │
│               │      │          mask, responses)       │
│               │      │                                │
│               │      │   9.3 重新评估                 │
│               │      │       - env.evaluate()         │
│               │      │       - 获取新的 eval_results  │
│               │      │       - 获取新的 trajectories │
│               │      │                                │
│               │      │   9.4 更新 batch 数据          │
│               │      │       - 更新 solver           │
│               │      │         trajectories          │
│               │      │       - 更新 format_rewards    │
│               │      │       - 更新 eval_results     │
│               │      │                                │
│               │      │   9.5 重新计算 learnability    │
│               │      │       - 调用 _compute_unified_ │
│               │      │         reward_and_filter()    │
│               │      │       - 重新检查每个 question │
│               │      │         的 learnability        │
│               │      │       - 更新 learnable_mask   │
│               │      │                                │
│               │      │   9.6 重新计算 rewards         │
│               │      │       - 重新计算 harm_rewards  │
│               │      │       - 重新计算 combined_     │
│               │      │         rewards = harm_rewards │
│               │      │         + 0.5 * format_rewards │
│               │      │       - 更新 batch.batch[      │
│               │      │         "rewards"]             │
│               │      │       - 更新 batch.batch[     │
│               │      │         "harm_scores"]         │
│               │      │                                │
│               │      │   如果仍有 all_skip 或         │
│               │      │   all_not_skip 的 group:      │
│               │      │     → 继续下一次迭代           │
│               │      │   否则:                        │
│               │      │     → 退出循环                 │
│               │      │                                │
│               │      │   更新 proposer_trajectories   │
└───────────────┘      └───────────────────────────────┘
        │                       │
        └───────────┬───────────┘
                    ▼
        ┌───────────────────────────────────────────┐
        │ 10. 应用采样结果并整合 Proposer 轨迹数据    │
        │ (无论是否需要 repropose，总是执行)          │
        │ (根据步骤 6.5 的 sampled_indices 提取数据)  │
        │                                            │
        │ for each prompt:                          │
        │   sampled_question_idx =                  │
        │     sampled_indices[prompt_idx]            │
        │                                            │
        │   if sampled_question_idx >= 0:           │
        │     # 提取采样到的 question 数据           │
        │     start_idx = sampled_question_idx *     │
        │                  rollout_n                 │
        │     end_idx = (sampled_question_idx + 1) * │
        │                rollout_n                   │
        │                                            │
        │     # 保留 solver 数据                    │
        │     final_task_configs.extend(             │
        │       task_configs[start_idx:end_idx])     │
        │     final_harm_rewards.extend(             │
        │       harm_rewards[start_idx:end_idx])     │
        │                                            │
        │     # 收集 proposer 轨迹                    │
        │     sampled_proposer_trajectories[        │
        │       prompt_id] = proposer_trajectories[  │
        │       prompt_id]                           │
        │                                            │
        │ 结果:                                      │
        │ - 每个 prompt: 1 个 question group         │
        │   (proposer: 1 prompt → 1 question)       │
        │ - 每个 question: 1 个 response group       │
        │   (solver: 1 question → rollout_n responses)│
        └───────────────────────────────────────────┘
                    │
                    ▼
        ┌───────────────────────────────────────────┐
        │ 11. 计算 Proposer Rewards                  │
        │                                            │
        │ _compute_proposer_rewards()               │
        │                                            │
        │ for each prompt:                          │
        │   if sampled_question is learnable:       │
        │     proposer_reward = 1.0                 │
        │   else:                                   │
        │     proposer_reward = 0.0                 │
        │                                            │
        │   (需要 repropose → reward = -0.5)         │
        └───────────────────────────────────────────┘
                    │
                    ▼
        ┌───────────────────────────────────────────┐
        │ 12. 整合到 Batch (DataProto)               │
        │                                            │
        │ _integrate_proposer_trajectories()        │
        │                                            │
        │ 添加到 batch:                              │
        │ - proposer_log_probs                      │
        │ - proposer_input_ids                      │
        │ - proposer_attention_mask                 │
        │ - proposer_rewards                        │
        │                                            │
        │ 合并 rewards:                              │
        │ - combined_rewards =                      │
        │   solver_rewards +                        │
        │   0.1 * proposer_rewards                  │
        │                                            │
        │ 确保 group 数量一致:                        │
        │ - Proposer: 1 group (1 prompt → 1 question)│
        │ - Solver: 1 group (1 question →            │
        │   rollout_n responses)                     │
        └───────────────────────────────────────────┘
                    │
                    ▼
        ┌───────────────────────────────────────────┐
        │ 13. 更新 Batch Rewards                     │
        │                                            │
        │ batch.batch["rewards"] =                  │
        │   harm_rewards +                          │
        │   0.5 * format_rewards                    │
        │                                            │
        │ batch.batch["proposer_rewards"] =         │
        │   proposer_rewards                        │
        │                                            │
        │ batch.batch["combined_rewards"] =         │
        │   rewards + 0.1 * proposer_rewards        │
        └───────────────────────────────────────────┘
                    │
                    ▼
        ┌───────────────────────────────────────────┐
        │ 14. 模型更新 (PPO)                         │
        │                                            │
        │ - 计算 old_log_probs (solver + proposer)  │
        │ - 计算 advantage                           │
        │ - 计算 policy loss                         │
        │ - 反向传播和优化                            │
        │                                            │
        │ 联合更新:                                  │
        │ - Proposer: prompt → question             │
        │ - Solver: question → response              │
        └───────────────────────────────────────────┘
```

## 数据结构说明

### Proposer 轨迹数据

```
1 个 Seed Task (Prompt)
    │
    ├─→ Proposer 生成
    │     │
    │     ├─→ Input: prompt (seed task)
    │     │
    │     ├─→ Output: question group (多个 questions)
    │     │
    │     └─→ Trajectory Data:
    │           - log_probs (proposer 生成 questions 的 logp)
    │           - input_ids (prompt + generated questions)
    │           - attention_mask
    │           - responses (generated questions)
    │
    └─→ 保存到: task_config.metadata["proposer_trajectory"]
```

### Solver 轨迹数据

```
1 个 Question (from proposer)
    │
    ├─→ Solver 生成
    │     │
    │     ├─→ Input: question
    │     │
    │     ├─→ Output: response group (rollout_n 个 responses)
    │     │
    │     └─→ Trajectory Data:
    │           - log_probs (solver 生成 responses 的 logp)
    │           - input_ids (question + generated responses)
    │           - attention_mask
    │           - responses
    │
    └─→ 保存到: batch (DataProto)
```

### 采样后的数据结构

```
采样后 (每个 prompt):
    │
    ├─→ Proposer 轨迹:
    │     - 1 个 prompt → 1 个 question (采样得到)
    │     - log_probs, input_ids, attention_mask
    │     - reward (基于 question 的 learnability)
    │
    └─→ Solver 轨迹:
          - 1 个 question → 1 个 response group
          - rollout_n 个 responses
          - log_probs, input_ids, attention_mask
          - reward (基于 harm scores)
```

## 关键步骤详解

### 步骤 1: Proposer 生成 (收集轨迹数据)

```
┌─────────────────────────────────────────────────────────┐
│ _create_generate_fn() → generate_fn()                   │
│                                                          │
│ Input:                                                  │
│   - prompt: seed task instruction                       │
│   - metadata: {seed_task_id, prompt_id}                 │
│                                                          │
│ Process:                                                │
│   1. 准备 batch (OSWorldDataset)                        │
│   2. actor_rollout_wg.generate_sequences()              │
│      → 生成 question_text                              │
│      → 收集 rollout_log_probs                           │
│   3. 提取 trajectory_data                              │
│                                                          │
│ Output:                                                 │
│   - question_text: JSON string with instruction,         │
│                    harm_action, etc.                    │
│   - trajectory_data: {log_probs, input_ids,             │
│                      attention_mask, responses}         │
└─────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│ propose_harm_tasks_sync()                               │
│                                                          │
│ - 调用 generate_fn(prompt, metadata)                    │
│ - 解析 question_text → ProposedHarmTask                │
│ - 保存 trajectory_data 到                               │
│   proposed_task.metadata["proposer_trajectory"]         │
└─────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│ _propose_tasks()                                         │
│                                                          │
│ - task.to_task_config()                                 │
│ - 保存 proposer_trajectory 到 task_config.metadata      │
│ - 添加 prompt_id 到 task_config                         │
└─────────────────────────────────────────────────────────┘
```

### 步骤 6-7: 统一计算和采样

```
┌─────────────────────────────────────────────────────────┐
│ _compute_unified_reward_and_filter()                    │
│                                                          │
│ 输入:                                                    │
│   - task_configs: [question1, question2, ...]          │
│   - trajectories: [traj1, traj2, ...]                  │
│   - eval_results: [eval1, eval2, ...]                   │
│   - rollout_n: 5                                        │
│                                                          │
│ 处理:                                                    │
│   1. 计算所有样本的分数 (safety, completion)            │
│   2. 按 prompt_id 分组 questions                        │
│   3. 对每个 question 检查 learnability                 │
│      - 调用 _check_learnability()                       │
│      - 检查 safe_ratio 和 incomplete_ratio              │
│   4. 检查每个 prompt 的 question group 是否需要 repropose│
│      - all_skip = all(not learnable)                   │
│      - all_not_skip = all(learnable)                  │
│      - 如果 all_skip OR all_not_skip → repropose        │
│   5. 从 learnable questions 中采样 1 个                 │
│                                                          │
│ 输出:                                                    │
│   - harm_rewards: [reward1, reward2, ...]               │
│   - learnable_mask: [True, False, ...]                  │
│   - sampled_indices: [0, -1, 1, ...]                   │
│     (>=0: 采样索引, -1: 需要 repropose)                  │
└─────────────────────────────────────────────────────────┘
```

### 步骤 9: Repropose 循环详解

```
┌─────────────────────────────────────────────────────────┐
│ Repropose 循环完整流程                                   │
│                                                          │
│ 每次迭代都会完整执行以下步骤:                             │
│                                                          │
│ 9.1 重新 Propose                                         │
│   - 调用 start_reset_envs()                             │
│   - 重新生成 question group (整个 group)                 │
│   - 收集新的 proposer trajectories                       │
│   - 更新 proposer_trajectories                          │
│                                                          │
│ 9.2 重新执行 Solver Rollout                             │
│   - 对重新生成的 questions 执行 rollout                  │
│   - 收集新的 solver trajectories                        │
│   - 收集 format_rewards                                 │
│   - 更新 batch 中的 solver 数据                          │
│     (input_ids, attention_mask, responses)               │
│                                                          │
│ 9.3 重新评估                                             │
│   - env.evaluate() → 新的 eval_results                   │
│   - env.get_history_messages() → 新的 trajectories       │
│                                                          │
│ 9.4 更新 Batch 数据                                      │
│   - 更新 batch 中的 solver trajectories                  │
│   - 更新 format_rewards                                 │
│   - 更新 eval_results                                   │
│                                                          │
│ 9.5 重新计算 Learnability                                │
│   - 调用 _compute_unified_reward_and_filter()           │
│   - 重新计算所有 samples 的 scores                       │
│   - 重新检查每个 question 的 learnability               │
│   - 更新 learnable_mask                                 │
│   - 更新 sampled_indices                                │
│                                                          │
│ 9.6 重新计算 Rewards                                     │
│   - 重新计算 harm_rewards                               │
│   - 重新计算 combined_rewards =                         │
│     harm_rewards + 0.5 * format_rewards                  │
│   - 更新 batch.batch["rewards"]                         │
│   - 更新 batch.batch["harm_scores"]                     │
│                                                          │
│ 检查条件:                                                │
│   - 如果仍有 all_skip 或 all_not_skip 的 group:         │
│     → 继续下一次迭代                                     │
│   - 否则:                                                │
│     → 退出循环，进入步骤 10                              │
└─────────────────────────────────────────────────────────┘
```

### 步骤 10-12: 采样和整合

```
┌─────────────────────────────────────────────────────────┐
│ 采样逻辑                                                 │
│                                                          │
│ for each prompt in prompt_to_questions:                 │
│   sampled_question_idx = sampled_indices[prompt_idx]    │
│                                                          │
│   if sampled_question_idx >= 0:                         │
│     # 采样这个 question                                 │
│     start_idx = sampled_question_idx * rollout_n        │
│     end_idx = (sampled_question_idx + 1) * rollout_n    │
│                                                          │
│     # 保留 solver 数据                                  │
│     final_task_configs.extend(                          │
│       task_configs[start_idx:end_idx])                  │
│     final_harm_rewards.extend(                          │
│       harm_rewards[start_idx:end_idx])                  │
│                                                          │
│     # 收集 proposer 轨迹                                │
│     if prompt_id in proposer_trajectories:              │
│       sampled_proposer_trajectories[prompt_id] =        │
│         proposer_trajectories[prompt_id]                │
└─────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│ _compute_proposer_rewards()                              │
│                                                          │
│ for each prompt in sampled_proposer_trajectories:       │
│   sampled_question_idx = sampled_indices[prompt_idx]     │
│   question_global_idx = question_indices[                │
│     sampled_question_idx]                                │
│   is_learnable = learnable_mask[question_global_idx]     │
│                                                          │
│   if is_learnable:                                      │
│     proposer_reward = 1.0                               │
│   else:                                                 │
│     proposer_reward = 0.0                               │
│                                                          │
│   proposer_rewards[prompt_id] = proposer_reward         │
└─────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│ _integrate_proposer_trajectories()                       │
│                                                          │
│ 1. 提取 proposer 轨迹数据                               │
│    for each prompt:                                     │
│      traj_data = proposer_trajectories[prompt_id][0]    │
│      proposer_log_probs.append(traj_data["log_probs"])  │
│                                                          │
│ 2. 扩展到匹配 solver batch size                         │
│    for each prompt (1 question):                        │
│      for rollout_n times:                               │
│        expanded_proposer_log_probs.append(log_probs)     │
│                                                          │
│ 3. 添加到 batch                                         │
│    batch.batch["proposer_log_probs"] = stack(...)      │
│    batch.batch["proposer_rewards"] = tensor(...)        │
│    batch.batch["combined_rewards"] =                    │
│      solver_rewards + 0.1 * proposer_rewards            │
└─────────────────────────────────────────────────────────┘
```

## 数据对齐说明

### Proposer 和 Solver 数据对齐

```
采样后的数据结构:

Prompt 1:
  Proposer:
    - 1 个 prompt → 1 个 question (采样得到)
    - log_probs: [L_p] (1 个)
    - reward: R_p (1 个)
  
  Solver:
    - 1 个 question → rollout_n 个 responses
    - log_probs: [L_s1, L_s2, ..., L_sn] (rollout_n 个)
    - rewards: [R_s1, R_s2, ..., R_sn] (rollout_n 个)

对齐方式:
  - Proposer log_probs 扩展 rollout_n 次: [L_p, L_p, ..., L_p]
  - Proposer rewards 扩展 rollout_n 次: [R_p, R_p, ..., R_p]
  - 与 Solver 数据一一对应
```

### Batch 结构

```
batch.batch:
  # Solver 数据 (来自 env workers)
  - input_ids: [batch_size, seq_len]
  - attention_mask: [batch_size, seq_len]
  - responses: [batch_size, response_len]
  - rollout_log_probs: [batch_size, response_len]  # Solver logp
  - rewards: [batch_size]  # Solver rewards
  
  # Proposer 数据 (整合后)
  - proposer_log_probs: [batch_size, question_len]  # Proposer logp
  - proposer_input_ids: [batch_size, question_seq_len]
  - proposer_attention_mask: [batch_size, question_seq_len]
  - proposer_rewards: [batch_size]  # Proposer rewards
  
  # 合并 rewards
  - combined_rewards: [batch_size]  # Solver + Proposer

batch.non_tensor_batch:
  - prompt_id: [batch_size]  # 用于匹配 proposer 和 solver
  - task_id: [batch_size]    # Question ID
```

## 关键配置

```yaml
absolute_zero:
  enabled: true
  enable_task_proposal: true
  
  # Repropose 配置
  max_repropose_attempts: 3
  
  # Learnability 验证配置
  learnability_safety_threshold: 0.5
  learnability_min_safe_ratio: 0.3
  learnability_max_safe_ratio: 0.7
  learnability_completion_threshold: 0.5
  learnability_min_incomplete_ratio: 0.3
  learnability_max_incomplete_ratio: 0.7
  
  # Reward 计算配置
  safety_reward_weight: 0.7
  task_completion_weight: 0.3
  completion_evaluator_type: "llm"
```

## 输出指标

```
Metrics:
  # Solver 相关
  - unified_filter/num_learnable: 可学习的 question 数量
  - unified_filter/num_total: 总 question 数量
  - unified_filter/learnable_ratio: 可学习比例
  - unified_filter/num_sampled: 采样到的 question 数量
  
  # Proposer 相关
  - proposer/num_trajectories: 收集的 proposer 轨迹数量
  - proposer/num_integrated: 整合到 batch 的轨迹数量
  - proposer/reward_mean: 平均 proposer reward
  
  # 联合更新相关
  - joint/combined_reward_mean: 平均合并 reward
  - joint/proposer_solver_ratio: Proposer/Solver reward 比例
  
  # Repropose 相关
  - repropose/total_attempts: repropose 总尝试次数
  - repropose/final_non_learnable: 最终仍不可学习的数量
```

## 核心特点

1. **联合轨迹收集**
   - Proposer: prompt → question group (包含 logp)
   - Solver: question → response group (包含 logp)

2. **统一采样**
   - 从 learnability=1 的配对中采样
   - 每个 prompt 采样 1 个 question
   - 每个 question 对应 1 个 response group

3. **Repropose 循环**
   - 当整个 question group 的 learnability 状态一致时（all_skip 或 all_not_skip）触发
   - 每次迭代完整执行：propose → solver → evaluate → compute learnability → compute rewards
   - 重新计算 learnability 和 rewards，确保数据一致性
   - 更新 batch 中的 solver trajectories 和 format_rewards

4. **数据整合**
   - Proposer 和 Solver 轨迹数据都整合到 batch
   - 确保 group 数量一致（均为 1）
   - 合并 rewards 用于联合更新

5. **联合更新**
   - 单模型同时更新 proposer 和 solver
   - 使用 combined_rewards 进行训练
   - 保留两类轨迹的 logp 用于 PPO 更新

