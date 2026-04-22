# DatasetEnv 离线 Reward 配置说明

### 1. 原始 chunk-level delta reward

环境侧改 [dataset_env_libero.yaml](/home/wudi/src/RLinf/examples/embodiment/config/env/dataset_env_libero.yaml)：

```yaml
reward_action_mode: delta
reward_granularity: chunk
```

RLinf 侧改 [libero_grpo_openvlaoft_our_offline.yaml](/home/wudi/src/RLinf/examples/embodiment/config/libero_grpo_openvlaoft_our_offline.yaml)：

```yaml
algorithm:
  adv_type: grpo
  reward_type: action_level
  logprob_type: token_level
```

说明：按原始 action 逐个比较计算距离，然后取平均作为chunk的最终reward，整个 chunk 只有一个 reward，所有 action 共用一个 advantage。

### 2. 最终累计动作的 chunk-level reward

环境侧改 [dataset_env_libero.yaml](/home/wudi/src/RLinf/examples/embodiment/config/env/dataset_env_libero.yaml)：

```yaml
reward_action_mode: cumulative
reward_granularity: chunk
```

RLinf 侧改 [libero_grpo_openvlaoft_our_offline.yaml](/home/wudi/src/RLinf/examples/embodiment/config/libero_grpo_openvlaoft_our_offline.yaml)：

```yaml
algorithm:
  adv_type: grpo
  reward_type: action_level
  logprob_type: token_level
```

说明：先累计整段 action，只用最后一个累计动作算 reward；整个 chunk 仍然共用一个 advantage。

### 3. per-action delta reward + per-action advantage

环境侧改 [dataset_env_libero.yaml](/home/wudi/src/RLinf/examples/embodiment/config/env/dataset_env_libero.yaml)：

```yaml
reward_action_mode: delta
reward_granularity: per_action
```

RLinf 侧改 [libero_grpo_openvlaoft_our_offline.yaml](/home/wudi/src/RLinf/examples/embodiment/config/libero_grpo_openvlaoft_our_offline.yaml)：

```yaml
algorithm:
  adv_type: grpo_per_action
  reward_type: action_level
  logprob_type: token_level
```

说明：每个 action 单独根据距离算 reward；GRPO 也会在每个 action 位置单独算 advantage。

### 4. per-action cumulative reward + per-action advantage

环境侧改 [dataset_env_libero.yaml](/home/wudi/src/RLinf/examples/embodiment/config/env/dataset_env_libero.yaml)：

```yaml
reward_action_mode: cumulative
reward_granularity: per_action
```

RLinf 侧改 [libero_grpo_openvlaoft_our_offline.yaml](/home/wudi/src/RLinf/examples/embodiment/config/libero_grpo_openvlaoft_our_offline.yaml)：

```yaml
algorithm:
  adv_type: grpo_per_action
  reward_type: action_level
  logprob_type: token_level
```

说明：先累计到当前步，再给每个 action 一个 reward；GRPO 也对每个 action 单独算 advantage。
