# Absolute Zero 当前完整流程（已删除 Propose 阶段验证）

## 核心设计原则

```
┌─────────────────────────────────────────────────────────────┐
│  统一模式（Unified Mode）始终启用                              │
│  - 跳过 propose 阶段验证                                      │
│  - 在 training 阶段使用 rollout 结果同时进行验证和 reward 计算│
│  - 非 learnable 任务自动跳过                                  │
└─────────────────────────────────────────────────────────────┘
```

## 完整流程图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          训练批次开始 (fit())                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │ 1. start_reset_envs()         │
                    │    - 从 seed tasks 生成       │
                    │      proposed tasks           │
                    │    - 每个 seed → 1 个         │
                    │      question group           │
                    │    - 每个 question group      │
                    │      包含多个 questions       │
                    └───────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │ 2. 环境重置 (env.reset)        │
                    │    - 为每个 task_config        │
                    │      重置环境                 │
                    │    - 准备初始状态              │
                    └───────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │ 3. Rollout 执行               │
                    │    (多步交互循环)              │
                    │                                │
                    │    for step in max_steps:     │
                    │      ├─→ actor.generate()     │
                    │      │   → actions            │
                    │      │                        │
                    │      └─→ env.step(actions)   │
                    │          → env_outputs       │
                    │                                │
                    │    直到完成或达到 max_steps    │
                    └───────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │ 4. 环境评估                    │
                    │    - env.evaluate()            │
                    │      → eval_results            │
                    │      (规则评估结果)             │
                    │    - env.get_history_messages()│
                    │      → trajectories            │
                    │      (完整交互历史)             │
                    └───────────────────────────────┘
                                    │
                                    ▼
        ┌───────────────────────────────────────────────────────┐
        │ 5. _compute_unified_reward_and_filter()              │
        │    (统一 Reward 计算和 Learnability 过滤)             │
        └───────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
                    ▼                               ▼
        ┌───────────────────────┐      ┌───────────────────────┐
        │ 5.1 计算所有样本分数    │      │ 5.2 按 prompt 分组     │
        │                        │      │                        │
        │ for each (task_config,  │      │ prompt_to_questions = │
        │          trajectory,   │      │   {prompt_id: [q1,    │
        │          eval_result):  │      │            q2, ...]}   │
        │   scores =              │      │                        │
        │     _compute_scores_   │      │ 每个 prompt → 1 个     │
        │     for_task()         │      │ question group         │
        │                        │      │                        │
        │   - safety_score       │      │                        │
        │     (LLM 评估)          │      │                        │
        │   - completion_score   │      │                        │
        │     (规则/LLM)          │      │                        │
        │   - combined_reward    │      │                        │
        └───────────────────────┘      └───────────────────────┘
                    │                               │
                    └───────────────┬───────────────┘
                                    ▼
                    ┌───────────────────────────────┐
                    │ 5.3 Learnability 检查          │
                    │                                │
                    │ for each question in group:    │
                    │   group_scores = [scores for   │
                    │                 rollout_n]     │
                    │                                │
                    │   is_learnable =               │
                    │     _check_learnability()     │
                    │                                │
                    │   双维度过滤:                  │
                    │   ├─ Safety 维度:              │
                    │   │   safe_ratio = count(      │
                    │   │     safety_score > thresh) │
                    │   │     / rollout_n            │
                    │   │   条件: min_safe_ratio <=  │
                    │   │         safe_ratio <=      │
                    │   │         max_safe_ratio     │
                    │   │                            │
                    │   └─ Completion 维度:         │
                    │       incomplete_ratio =      │
                    │         count(completion_score │
                    │         < thresh) / rollout_n  │
                    │       条件: min_incomplete_    │
                    │              ratio <=          │
                    │              incomplete_ratio <=│
                    │              max_incomplete_    │
                    │              ratio             │
                    │                                │
                    │   两个条件都满足 → learnable   │
                    └───────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │ 5.4 Repropose 检查              │
                    │                                │
                    │ for each prompt's group:       │
                    │   all_skip = all(not learnable)│
                    │   all_not_skip = all(learnable)│
                    │                                │
                    │   if all_skip OR all_not_skip: │
                    │     → 标记 repropose (-1)      │
                    │   else:                        │
                    │     → 混合状态，可以采样        │
                    └───────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │ 5.5 采样                        │
                    │                                │
                    │ for each prompt (混合状态):    │
                    │   learnable_questions =        │
                    │     [q for q in questions      │
                    │      if q.is_learnable]        │
                    │                                │
                    │   if len(learnable_questions)  │
                    │      > 0:                      │
                    │     sampled_idx = random.choice│
                    │     (learnable_questions)      │
                    │   else:                        │
                    │     sampled_idx = -1           │
                    │     (标记 repropose)           │
                    │                                │
                    │   返回:                        │
                    │   - rewards: List[float]       │
                    │   - learnable_mask: List[bool] │
                    │   - sampled_indices: List[int] │
                    │     (>=0: 采样索引, -1: repropose)│
                    └───────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │ 6. 检查是否需要 Repropose      │
                    │                                │
                    │ needs_repropose = any(         │
                    │   idx < 0 for idx in           │
                    │   sampled_indices)             │
                    └───────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
              否    │                               │   是
                    │                               │
                    ▼                               ▼
        ┌───────────────────────┐      ┌───────────────────────────────┐
        │ 7. 直接采样和过滤      │      │ 8. Repropose 循环              │
        │                        │      │                                │
        │ 根据 sampled_indices   │      │ while repropose_attempt <      │
        │ 过滤数据:              │      │   max_repropose_attempts:     │
        │                        │      │                                │
        │ for each prompt:       │      │   8.1 找到需要 repropose 的    │
        │   if sampled_idx >= 0: │      │       question groups          │
        │     保留该 question    │      │       (sampled_idx == -1)      │
        │     group              │      │                                │
        │   else:                │      │   8.2 重新 propose             │
        │     跳过               │      │       start_reset_envs(        │
        │                        │      │         non_learnable_indices)  │
        │ 过滤 batch (DataProto) │      │       → 重新生成整个 group     │
        └───────────────────────┘      │                                │
                    │                  │   8.3 执行新的 rollout          │
                    │                  │       (仅对需要 repropose 的    │
                    │                  │        envs)                    │
                    │                  │                                │
                    │                  │   8.4 重新评估                 │
                    │                  │       env.evaluate()           │
                    │                  │       env.get_history_         │
                    │                  │       messages()               │
                    │                  │                                │
                    │                  │   8.5 更新数据                  │
                    │                  │       替换原 question group     │
                    │                  │       的数据                   │
                    │                  │                                │
                    │                  │   8.6 重新计算 learnability    │
                    │                  │       _compute_unified_        │
                    │                  │       reward_and_filter()      │
                    │                  │                                │
                    │                  │   8.7 检查是否还有需要         │
                    │                  │       repropose 的 groups      │
                    │                  │       - 如果没有，退出循环      │
                    │                  │       - 如果还有，继续尝试      │
                    │                  └───────────────────────────────┘
                    │                               │
                    └───────────────┬───────────────┘
                                    ▼
                    ┌───────────────────────────────┐
                    │ 9. 最终采样和过滤                │
                    │                                │
                    │ 根据最终的 sampled_indices:    │
                    │                                │
                    │ for each prompt:               │
                    │   if sampled_idx >= 0:         │
                    │     保留 1 个 question +        │
                    │     1 个 response group        │
                    │     (rollout_n 个 responses)    │
                    │                                │
                    │ _filter_batch_by_mask()        │
                    │ 只保留采样到的数据               │
                    └───────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │ 10. 计算最终 Rewards            │
                    │                                │
                    │ harm_rewards (来自 unified)     │
                    │ format_rewards (格式奖励)        │
                    │                                │
                    │ combined_reward =              │
                    │   harm_rewards +                │
                    │   0.5 * format_rewards          │
                    │                                │
                    │ batch.batch["rewards"] =       │
                    │   combined_reward              │
                    └───────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │ 11. 模型更新 (PPO)              │
                    │                                │
                    │ - 计算 old_log_probs            │
                    │ - 计算 advantage                │
                    │ - 计算 policy loss              │
                    │ - 反向传播和优化                │
                    └───────────────────────────────┘
```

## 数据结构说明

### Question Group 结构

```
1 个 Seed Task (Prompt)
    │
    ├─→ Proposer 生成 1 个 Question Group
    │     │
    │     ├─→ Question 1
    │     │     ├─→ Response 1 (rollout_n=5)
    │     │     ├─→ Response 2
    │     │     ├─→ Response 3
    │     │     ├─→ Response 4
    │     │     └─→ Response 5
    │     │
    │     ├─→ Question 2
    │     │     └─→ ... (rollout_n 个 responses)
    │     │
    │     └─→ Question 3
    │           └─→ ... (rollout_n 个 responses)
    │
    └─→ 最终采样结果:
          └─→ 1 个 Question (从 learnable 中随机选择)
                └─→ 1 个 Response Group (rollout_n 个 responses)
```

### Learnability 判定流程

```
对于每个 Question (包含 rollout_n 个 responses):

┌─────────────────────────────────────────────────────────┐
│ Step 1: 计算每个 response 的分数                        │
│                                                          │
│ for each response in rollout_n:                        │
│   scores = _compute_scores_for_task(                    │
│     task_config, trajectory, eval_result)              │
│                                                          │
│   - safety_score (LLM 评估)                             │
│   - completion_score (规则/LLM)                          │
└─────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│ Step 2: Safety 维度过滤                                  │
│                                                          │
│ safe_count = count(safety_score > safety_threshold)     │
│ safe_ratio = safe_count / rollout_n                    │
│                                                          │
│ safety_condition =                                      │
│   min_safe_ratio <= safe_ratio <= max_safe_ratio       │
└─────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│ Step 3: Completion 维度过滤                               │
│                                                          │
│ incomplete_count = count(                               │
│   completion_score < completion_threshold)             │
│ incomplete_ratio = incomplete_count / rollout_n         │
│                                                          │
│ completion_condition =                                  │
│   min_incomplete_ratio <= incomplete_ratio <=          │
│   max_incomplete_ratio                                  │
└─────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│ Step 4: 综合判定                                          │
│                                                          │
│ is_learnable =                                          │
│   safety_condition AND completion_condition             │
└─────────────────────────────────────────────────────────┘
```

### Repropose 触发条件

```
对于每个 Question Group (包含多个 Questions):

┌─────────────────────────────────────────────────────────┐
│ 检查所有 Questions 的 Learnability 状态                   │
│                                                          │
│ question_learnability = [                               │
│   q1.is_learnable,                                      │
│   q2.is_learnable,                                      │
│   q3.is_learnable,                                      │
│   ...                                                   │
│ ]                                                       │
│                                                          │
│ all_skip = all(not x for x in question_learnability)  │
│ all_not_skip = all(question_learnability)              │
└─────────────────────────────────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        │                       │
        ▼                       ▼
┌───────────────┐      ┌───────────────┐
│ all_skip = True│      │ all_not_skip =│
│ OR            │      │ True          │
│ all_not_skip =│      │               │
│ True          │      │               │
└───────┬───────┘      └───────┬───────┘
        │                       │
        └───────────┬───────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │ 触发 Repropose         │
        │                        │
        │ sampled_idx = -1       │
        │ (标记需要 repropose)    │
        └───────────────────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │ 重新生成整个 Question   │
        │ Group                  │
        │                        │
        │ 重复步骤 1-5，直到       │
        │ 达到混合状态            │
        └───────────────────────┘
```

### 采样规则

```
对于每个 Prompt (包含 1 个 Question Group):

┌─────────────────────────────────────────────────────────┐
│ 情况 1: Question Group 需要 Repropose                    │
│                                                          │
│ if sampled_idx == -1:                                   │
│   → 不采样，等待 repropose 循环处理                       │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ 情况 2: Question Group 是混合状态                        │
│                                                          │
│ learnable_questions = [                                 │
│   q for q in questions                                  │
│   if q.is_learnable                                     │
│ ]                                                       │
│                                                          │
│ if len(learnable_questions) > 0:                       │
│   sampled_idx = random.choice(                          │
│     learnable_questions                                 │
│   )                                                     │
│   → 采样 1 个 learnable question                         │
│ else:                                                   │
│   sampled_idx = -1                                      │
│   → 标记需要 repropose                                   │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ 最终结果: 每个 Prompt 保留 1 个 Question                  │
│                                                          │
│ sampled_question = questions[sampled_idx]               │
│                                                          │
│ 保留的数据:                                               │
│ - 1 个 Question                                          │
│ - 1 个 Response Group (rollout_n 个 responses)          │
│ - 对应的 rewards, log_probs, etc.                        │
└─────────────────────────────────────────────────────────┘
```

## 关键配置参数

```yaml
absolute_zero:
  enabled: true                    # 启用 Absolute Zero
  enable_task_proposal: true       # 启用任务生成
  
  # Repropose 配置
  max_repropose_attempts: 3        # 最大 repropose 尝试次数
  
  # Learnability 验证配置（用于训练阶段）
  learnability_safety_threshold: 0.5
  learnability_min_safe_ratio: 0.3
  learnability_max_safe_ratio: 0.7
  learnability_completion_threshold: 0.5
  learnability_min_incomplete_ratio: 0.3
  learnability_max_incomplete_ratio: 0.7
  
  # Reward 计算配置
  safety_reward_weight: 0.7
  task_completion_weight: 0.3
  completion_evaluator_type: "llm"  # "llm" or "rule"
```

## 输出指标

```
Metrics:
  - unified_filter/num_learnable: 可学习的 question 数量
  - unified_filter/num_total: 总 question 数量
  - unified_filter/learnable_ratio: 可学习比例
  - unified_filter/num_sampled: 采样到的 question 数量
  - unified_filter/num_repropose: 需要 repropose 的 question group 数量
  - repropose/total_attempts: repropose 总尝试次数
  - repropose/final_non_learnable: 最终仍不可学习的数量
  - harm/safety_score_mean: 平均安全分数
  - harm/completion_score_mean: 平均完成分数
  - harm/combined_reward_mean: 平均组合奖励
```

## 核心特点总结

1. **统一模式始终启用**
   - 跳过 propose 阶段验证
   - 在 training 阶段使用 rollout 结果同时进行验证和 reward 计算
   - 避免重复环境交互

2. **Question Group 级别处理**
   - 以 question group 为单位进行 repropose
   - 只有当整个 group 全为 skip 或全为 not skip 时才触发 repropose

3. **混合状态要求**
   - 只有当 question group 处于混合状态（部分 learnable，部分不 learnable）时才采样
   - 确保训练数据的多样性

4. **单组采样**
   - 每个 prompt 最终只采样 1 个 question + 1 个 response group
   - 保证 proposer 和 solver 的 group 数量一致

5. **自动跳过**
   - 非 learnable 的任务自动跳过，不参与训练
   - 简化了处理逻辑

