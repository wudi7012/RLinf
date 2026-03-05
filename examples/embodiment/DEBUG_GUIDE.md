# LIBERO RL 任务 Debug 指南

本指南说明如何通过 Ray Distributed Debugger 和 `breakpoint()` 分阶段调试 `libero_130_grpo_openvlaoft` 任务。

## 运行方式

```bash
# 确保已安装 debugpy
uv pip install debugpy

# 使用 Ray Distributed Debugger 运行（需先在 VSCode/Cursor 中配置 Ray 扩展）
bash examples/embodiment/run_embodiment.sh libero_130_grpo_openvlaoft
```

当程序打印 `use 'ray debug' to connect ...` 时，在 VSCode/Cursor 中通过 Ray 扩展连接，或另开终端执行 `ray debug`。

---

## 断点分布总览

| 阶段 | 文件 | 位置标记 | 进程类型 | 触发频率 | 建议用途 |
|------|------|----------|----------|----------|----------|
| Driver | train_embodied_agent.py | `[DEBUG Driver]` | 主进程 | 1 次 | 检查配置 |  √
| Runner | embodied_runner.py | `[DEBUG Runner]` | 主进程 | 仅 step 0 | 检查主循环 | √
| 1A | env_worker.py | `[DEBUG Phase1A]` | EnvWorker | 1 次/rank | 环境类选择 | √
| 1B | huggingface_worker.py | `[DEBUG Phase1B]` | RolloutWorker | 1 次/rank | 模型加载 |
| 1C | libero_env.py | `[DEBUG Phase1C]` | 子进程 | 1 次/stage | LIBERO 初始化 | √
| 2A-pre | env_worker.py | `[DEBUG Phase2A-pre]` | EnvWorker | 仅首次 | 原始动作 | √
| 2A-post | env_worker.py | `[DEBUG Phase2A-post]` | EnvWorker | 仅首次 | 处理后动作+奖励 |√
| 2B | huggingface_worker.py | `[DEBUG Phase2B]` | RolloutWorker | 仅首次 | 模型推理输出 |
| 2C | huggingface_worker.py | `[DEBUG Phase2C]` | RolloutWorker | 仅首次 | 单步轨迹 |
| 3A | fsdp_actor_worker.py | `[DEBUG Phase3A]` | ActorWorker | 1 次/step | 收到的 rollout batch |
| 3B | fsdp_actor_worker.py | `[DEBUG Phase3B]` | ActorWorker | 1 次/step | 优势与回报 |
| 3C | fsdp_actor_worker.py | `[DEBUG Phase3C]` | ActorWorker | 1 次/step | 训练开始 |
| 4A | libero_env.py | `[DEBUG Phase4A]` | 子进程 | 每次 reset | LIBERO 观测 | √
| 4B | libero_env.py | `[DEBUG Phase4B]` | 子进程 | 有 success 时 | LIBERO 奖励 |√

**注意**：Phase 2 的断点已改为仅首次触发，方便快速查看数据流而不至于卡住。

---

## 分阶段调试建议

### 阶段 1：理解初始化（推荐首先运行）

保留 **1A、1B、1C**，注释其余断点。运行后依次查看：

- **1A**：`train_env_cls`（应为 LiberoEnv）、`self.train_num_envs_per_stage`、`self.cfg.env.train.env_type`
- **1B**：`type(self.hf_model)`（OpenVLAOFTForRLActionPrediction）、`rollout_model_config.model_path`
- **1C**：`self.task_suite.n_tasks`（130）、`self.task_ids`、`self.task_descriptions` 前几条

### 阶段 2：理解 Rollout 数据流

保留 **2A、2B、2C**（均已设为仅首次触发）：

- **2A-pre**：`chunk_actions.shape` 应为 `[num_envs, num_action_chunks, 7]`
- **2A-post**：对比 `prepare_actions` 前后夹爪维度（最后一维）的变化
- **2B**：`actions`、`result["prev_logprobs"]`、`result["prev_values"]`
- **2C**：`dones`、`rewards`、`chunk_step_result`

### 阶段 3：理解 GRPO 训练

保留 **3A、3B、3C**：

- **3A**：`self.rollout_batch.keys()`、`rewards` 分布、`loss_mask`
- **3B**：`advantages_and_returns` 中的 `advantages`、`returns` 形状和数值
- **3C**：`rollout_size`、`shuffle_id`、训练 batch 结构

### 阶段 4：理解 LIBERO 特有逻辑

保留 **4A、4B**：

- **4A**：`obs["main_images"].shape`、`obs["wrist_images"].shape`、`task_descriptions`
- **4B**：仅在任务成功时断点，查看 `terminations`、`step_reward`、`self.returns`

---

## 快速禁用所有断点

若暂时不需要调试，可全局搜索 `breakpoint()` 并注释或删除。所有断点旁均有 `[DEBUG Phase...]` 标记，便于批量查找。

---

## 常见问题

**Q: Ray debug 连接后没有命中断点？**  
A: 确认断点所在代码在 Worker 进程中执行。Driver 端（如 `train_embodied_agent.py`、`embodied_runner.run()`）用普通 VSCode/Cursor 断点即可。

**Q: 多个 Worker 同时断点导致死锁？**  
A: 不同 Worker 的断点可能互相等待。尽量每次只启用一个 Worker 类型的断点，或只在 rank 0 / stage 0 上断点。

**Q: 希望 Phase 2 断点每步都触发？**  
A: 移除 `if not getattr(self, "_debug_phase2x_done", False):` 条件，直接调用 `breakpoint()`。
