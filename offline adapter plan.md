下面给出一份面向实现的设计文档：

**Frozen VLA + Residual Chunk Adapter + Offline Reward + GRPO**

该方案的目标是：**避免直接对 flow-based VLA 本体做 RL，而是在冻结的 VLA 外训练一个小型 residual adapter，并沿用你已经实现的 Dataset-as-Env / offline reward / GRPO 框架。** 这样既保留 VLA 的表征与动作先验，又把优化对象变成一个普通、可稳定训练的小高斯策略。PLD 明确指出，直接优化 expressive foundation policy，尤其是 flow action head，是困难的；相反，轻量 residual Gaussian policy 更适合用现成 RL 方法训练。RLT 则表明，冻结 VLA、让小 policy 条件于 VLA 表征与参考动作，是一条有效路线。

---

# 1. 设计目标

## 1.1 问题定义

你当前已经有一套 offline reward 训练范式：

* 将 SFT 数据集封装为虚拟环境 DatasetEnv；
* 每个 episode 只有 1 个 chunk step；
* 模型看到 observation 后输出 action chunk；
* reward 由 `predicted action` 与 `ground-truth action` 的比较得到；
* 同一 observation 在 group 内复制，通过采样温度产生多个候选动作，用 GRPO 做组内相对优化。

这套范式本身没有问题。问题在于：若被优化对象是 **flow-based VLA 本体**，则会落入你已经识别出的困难区域——flow policy 的概率建模、对数概率、以及标准 policy-gradient 形式并不自然。PLD 也明确把这一点作为采用 residual Gaussian policy 的动机之一。

## 1.2 方案目标

本方案要实现以下三点：

1. **冻结 base VLA**，不更新 flow head、不更新大模型主体；
2. **在 base VLA 外套一个小型 residual chunk adapter**，只训练 adapter；
3. **将你已有的 offline reward + DatasetEnv + GRPO 框架几乎原样复用**，只替换“被训练的策略模块”。

---

# 2. 核心方法概述

设：

* (o)：视觉观测
* (\ell)：语言指令
* (s^p)：低维 proprio 状态
* (\pi_{\text{vla}})：冻结的 base VLA
* (a_{\text{base}}\in\mathbb{R}^{C\times d})：base VLA 输出的 action chunk
* (z)：从 frozen VLA 抽取的状态表征
* (\pi_\theta)：待训练的小 adapter policy
* (\Delta a\in\mathbb{R}^{C\times d})：adapter 输出的 residual chunk

则系统定义为：

[
z = f_{\text{vla}}(o,\ell)
]

[
a_{\text{base}} = \pi_{\text{vla}}(o,\ell)
]

[
\Delta a \sim \pi_\theta(\cdot \mid z, s^p, a_{\text{base}})
]

[
a_{\text{final}} = a_{\text{base}} + \xi \tanh(\Delta a)
]

其中 (\xi>0) 是 residual bound，用于限制 adapter 对 base action 的偏移幅度。PLD 明确采用了以 base action 为条件的 residual policy，并强调将 residual 动作限制在有界区间内，以避免早期偏离 base policy 太远。

这个形式本质上融合了两篇工作的优点：

* **RLT**：小 policy 应该条件于 VLA 表征与 VLA reference action，而不是脱离 base VLA 独立学习。
* **PLD**：更合适的动作形式不是直接替换 VLA 动作，而是在 base action 上学习 residual Gaussian policy。

---

# 3. 总体系统架构

整体数据流如下：

1. DatasetEnv `reset()` 返回一个离线样本的 observation；
2. frozen VLA 前向，输出：

   * 中间表征 `z`
   * base action chunk `a_base`
3. adapter 读入 `(z, s^p, a_base)`，采样 residual chunk `Δa`
4. 合成最终动作：
   [
   a_{\text{final}} = a_{\text{base}} + \xi\tanh(\Delta a)
   ]
5. DatasetEnv 用 `a_final` 和 `gt_action` 计算 reward；
6. group 内多个采样候选通过 GRPO 形成相对 advantage；
7. 只更新 adapter 参数 (\theta)，VLA 保持冻结。

---

# 4. 为什么采用 Residual Chunk Adapter

## 4.1 不直接训练 VLA 本体

直接对 flow-based VLA 做 RL 至少有两类风险：

* 概率形式不自然，训练目标难写成标准稳定的 policy optimization；
* 参数量大、训练代价高、容易破坏 base VLA 的泛化先验。

RLT 和 PLD 分别从不同角度都在回避这个问题：前者冻结 VLA，只训练小 actor/critic；后者冻结 base policy，只训练轻量 residual policy。

## 4.2 不直接从零生成全新动作

你的 reward 是“动作接近专家动作”的 offline reward，而不是传统 sparse success。
在这种 setting 下，让 adapter 直接输出完整 chunk 会扩大搜索空间，导致它更像重新训练一个 policy，而不是微调 base VLA。

残差形式更符合 adapter 的职责：

* base VLA 已经给出合理提案；
* adapter 只需要修正局部误差；
* reward 更容易解释为“是否比 base 更好”。

这和 PLD 的 residual 思路完全一致，也和 RLT 的“local refinement around VLA action”精神一致。

---

# 5. 模型设计

## 5.1 输入与输出

设：

* action chunk 长度为 (C)
* 单步动作维度为 (d)

则：

* `a_base.shape = [B, C, d]`
* `Δa.shape = [B, C, d]`
* `a_final.shape = [B, C, d]`

adapter 的输入为：

[
x_{\text{adapter}} = [z,; s^p,; \mathrm{vec}(a_{\text{base}})]
]

输出为：

[
\mu_\theta(x_{\text{adapter}})\in\mathbb{R}^{C\cdot d}
]

若采用高斯策略，则：

[
\Delta a \sim \mathcal N(\mu_\theta(x), \sigma^2 I)
]

再 reshape 成 ([C,d])。

---

## 5.2 VLA 状态表征 (z)

本方案不要求你一开始实现 RLT 那种 encoder-decoder RL token。
第一版建议直接使用 **frozen VLA 最后一层融合特征的 pooled embedding** 作为 `z`。

* 取最后一层 token embeddings；
* 对视觉/语言 token 做 mean pooling 或取固定 special token；
* 得到 `z ∈ R^{D_vla}`。

这样改动最小，最符合“先把 adapter + offline reward 跑稳”的目标。

---

## 5.3 Adapter Policy 结构

### 推荐架构

采用 **3-layer MLP Gaussian policy**。

这部分与 PLD 附录中用于 RL specialist 的实现风格一致：3-layer MLP Gaussian policy，critic 用 clipped double Q，带 LayerNorm。虽然你这里不一定需要 critic，但 actor 结构建议保持同样的轻量风格。

### 具体定义

令：

* `D_z`：VLA 表征维度
* `D_p`：proprio 维度
* `D_a = C * d`

则输入维度：

[
D_{\text{in}} = D_z + D_p + D_a
]

MLP 定义：

* Linear(`D_in`, 512)
* LayerNorm(512)
* ReLU
* Linear(512, 512)
* LayerNorm(512)
* ReLU
* Linear(512, 512)
* LayerNorm(512)
* ReLU

输出头：

* `mean_head: Linear(512, D_a)`
* `log_std`: 推荐先用一个全局可学习参数 `nn.Parameter([D_a])`

若显存受限，可把 512 改为 256。
若任务复杂且 `D_z` 很大，512 更稳妥。

### 输出形式

[
\mu = \text{mean_head}(h)
]
[
\sigma = \exp(\log \sigma)
]
[
\Delta a \sim \mathcal N(\mu,\sigma^2 I)
]

采样后 reshape 为 `[B, C, d]`，再做：

[
a_{\text{final}} = a_{\text{base}} + \xi \tanh(\Delta a)
]

### 为什么要 `tanh`

因为 PLD 明确强调 residual action magnitude 需要限制在有界范围内，否则初期探索容易偏离 base policy 太远。

---

# 6. Reward 设计

这是本方案里最关键的部分之一。

## 6.1 不建议继续只用绝对误差 reward

原始 reward：

[
r = -\mathrm{MAE}(a_{\text{final}}, a_{gt})
]

它可以训练，但对 adapter 而言不够理想，因为它没有明确体现“adapter 的目标是改进 base VLA”。

---

## 6.2 推荐的主 reward：相对改进 reward

定义：

[
e_{\text{base}} = \mathrm{MAE}(a_{\text{base}}, a_{gt})
]

[
e_{\text{final}} = \mathrm{MAE}(a_{\text{final}}, a_{gt})
]

reward 定义为：

[
r_{\text{imp}} = e_{\text{base}} - e_{\text{final}}
]

解释：

* 若 adapter 使最终动作比 base 更接近专家，则 reward 为正；
* 若 adapter 反而变差，则 reward 为负；
* 若 adapter 基本没动，则 reward 接近 0。

这个定义最符合 adapter 的真实职责：
**不是重做行为克隆，而是在 frozen VLA 基础上产生“改进”。**

---

## 6.3 残差正则

为防止 adapter 无限制偏离 base policy，引入残差惩罚：

[
r = r_{\text{imp}} - \lambda |\Delta a|_1
]

也可以用二范数：

[
r = r_{\text{imp}} - \lambda |\Delta a|_2^2
]

推荐先用 L1，因为它更直接鼓励“只在必要维度、必要时刻上修改”。

这一步在思想上对应：

* PLD 的 bounded residual
* RLT 的“stay close to reference action”的正则趋势。



---

# 7. 训练目标

## 7.1 策略分布

因为你现在训练的是小 Gaussian adapter，而非 flow head，所以可以自然地使用标准 logprob：

[
\log \pi_\theta(\Delta a \mid z, s^p, a_{\text{base}})
]

这正是该方案相较“直接对 flow VLA 做 RL”的核心优势。

---

## 7.2 与 GRPO 的兼容方式

你的 DatasetEnv 已经采用了 group-based sampling 机制：

* 同一个 observation 被复制到同一个 group 内；
* 通过采样温度或 policy stochasticity 产生多个不同候选；
* reward 在 group 内归一化，形成 GRPO advantage。

现在只需要把“被采样的对象”改为 residual adapter 的输出即可：

* 同一组共享 `(obs, a_base, z)`
* adapter 采样出多个不同 `Δa`
* 每个候选都有自己的 `a_final`
* 用 reward 比较这些候选谁对 base 改进更大

这一点和你现有 GRPO 设计完全兼容。

---


# 8. 在 RLinf 中的实现改动

你当前已有的 Offline Reward Feature 设计里，以下部分可直接复用：

* `DatasetEnv`
* 环境注册
* `prepare_actions`
* `EmbodiedRunner`
* `EnvWorker`
* `RolloutWorker`
* `EmbodiedFSDPActor`
* 现有 GRPO advantage 计算逻辑。

因此，本方案的实现重点只在于新增一个 **VLA+Adapter policy wrapper**。

---

## 9.1 新增模块建议

### 文件 1

`rlinf/models/vla_adapter/residual_chunk_adapter.py`

实现 `ResidualChunkAdapterPolicy`

核心职责：

* 调用 frozen VLA
* 取出 `z`
* 取出 `a_base`
* 调用 adapter actor
* 生成 `a_final`
* 返回：

  * `a_final`
  * `logprob(Δa)`
  * `a_base`
  * `delta_action`
  * 调试信息

---

### 文件 2

`rlinf/models/vla_adapter/base_feature_extractor.py`

职责：

* 从你当前 VLA / OpenVLA-OFT / pi0.5 模型中抽取 pooled hidden state
* 统一输出 `z`

第一版只需要支持你当前 backbone 即可。

---

### 文件 3

`rlinf/models/vla_adapter/reward_utils.py`

实现：

* `mae_error(a, gt)`
* `improvement_reward(a_base, a_final, gt)`
* `residual_penalty(delta)`
* `compose_reward(...)`

---

## 9.2 DatasetEnv 的改动

原则上 **不需要改变 env 接口**。
但为了便于分析，建议在 `infos` 中多返回：

* `base_mae`
* `final_mae`
* `delta_norm`
* `improvement = base_mae - final_mae`

这不会破坏现有流程，却能极大帮助调试。

---

## 9.3 配置新增项

在你的训练 config 中新增一个 `adapter` 段：

```yaml
adapter:
  enable: true
  type: residual_chunk_adapter
  hidden_dim: 512
  num_layers: 3
  use_layernorm: true
  residual_bound: 0.5
  std_mode: global_learnable
  init_log_std: -2.0

reward:
  type: improvement
  error_fn: mae
  residual_penalty: l1
  residual_coef: 0.01

vla:
  freeze_backbone: true
  freeze_action_head: true
  feature_type: pooled_last_hidden
  use_base_action: true
```

如果是 SimplerEnv 风格任务，可把 `residual_bound` 调到 0.1，PLD 的经验也是如此。

---

# 10. 模型前向伪代码

下面给出建议的核心伪代码。

```python
class ResidualChunkAdapterPolicy(nn.Module):
    def __init__(self, frozen_vla, adapter_actor, residual_bound=0.5):
        super().__init__()
        self.vla = frozen_vla
        self.actor = adapter_actor
        self.residual_bound = residual_bound

        for p in self.vla.parameters():
            p.requires_grad = False
        self.vla.eval()

    def extract_features_and_base_action(self, obs):
        with torch.no_grad():
            # 1) frozen VLA forward
            # should return:
            # - pooled hidden feature z
            # - base action chunk a_base
            z, a_base = self.vla.forward_for_adapter(obs)
        return z, a_base

    def forward(self, obs, deterministic=False):
        z, a_base = self.extract_features_and_base_action(obs)
        proprio = obs["states"]                       # [B, Dp]
        a_base_flat = a_base.flatten(start_dim=1)    # [B, C*d]

        actor_in = torch.cat([z, proprio, a_base_flat], dim=-1)
        mu, log_std = self.actor(actor_in)

        std = log_std.exp()
        dist = torch.distributions.Normal(mu, std)

        if deterministic:
            delta_flat = mu
        else:
            delta_flat = dist.rsample()

        logprob = dist.log_prob(delta_flat).sum(dim=-1)

        delta = delta_flat.view_as(a_base)           # [B, C, d]
        delta = self.residual_bound * torch.tanh(delta)

        a_final = a_base + delta

        return {
            "actions": a_final,
            "logprob": logprob,
            "base_actions": a_base,
            "delta_actions": delta,
            "features": z,
        }
```

---

# 11. Reward 伪代码

```python
def compute_improvement_reward(base_actions, final_actions, gt_actions,
                               residual_actions=None,
                               residual_coef=0.01):
    base_mae = (base_actions - gt_actions).abs().mean(dim=(-1, -2))
    final_mae = (final_actions - gt_actions).abs().mean(dim=(-1, -2))

    improvement = base_mae - final_mae

    if residual_actions is None:
        penalty = 0.0
    else:
        penalty = residual_coef * residual_actions.abs().mean(dim=(-1, -2))

    reward = improvement - penalty
    info = {
        "base_mae": base_mae,
        "final_mae": final_mae,
        "improvement": improvement,
        "delta_l1": residual_actions.abs().mean(dim=(-1, -2)) if residual_actions is not None else None,
    }
    return reward, info
```

---

# 12. Adapter Actor 伪代码

```python
class AdapterActor(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=512, num_layers=3, init_log_std=-2.0):
        super().__init__()
        layers = []
        dim = in_dim
        for _ in range(num_layers):
            layers += [
                nn.Linear(dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            ]
            dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.mean_head = nn.Linear(hidden_dim, out_dim)
        self.log_std = nn.Parameter(torch.ones(out_dim) * init_log_std)

    def forward(self, x):
        h = self.backbone(x)
        mu = self.mean_head(h)
        log_std = self.log_std.unsqueeze(0).expand_as(mu)
        return mu, log_std
```

---

# 13. DatasetEnv 中的接口建议

你当前的 `chunk_step(actions)` 已经适合本方案。

若你愿意做更干净的工程封装，可以让 `actions` 之外再传一个 `policy_outputs` 字典，但这不是必须。最小改法是：

* `RolloutWorker` 仍只把 `a_final` 发给 env；
* `ActorWorker` 这边自己保存 `a_base`、`delta`、`logprob`；
* `DatasetEnv` 只负责比较 `a_final` 和 `gt_action`，算 reward；
* `base_mae` 可在 actor 端本地额外计算。

这样能避免改 env-worker 通信协议。

---

# 14. 推荐训练超参数

以下是第一版建议值：

* `adapter.hidden_dim = 512`
* `adapter.num_layers = 3`
* `adapter.init_log_std = -2.0`
* `adapter.residual_bound = 0.5`
* `reward.error_fn = mae`
* `reward.type = improvement`
* `reward.residual_coef = 0.01`
* `group_size = 8`
* `temperature_train = 1.0 ~ 1.5`

若任务相对容易、或动作尺度较敏感，可把 `residual_bound` 降到 0.1。PLD 在不同 benchmark 上也采用了不同的 residual bound，说明这一超参数是敏感项。

---

# 15. 建议的消融实验

为了验证该方案是否真的有效，建议至少做以下消融：

## 15.1 无 residual，仅直接输出 full chunk

比较：

* `a_final = adapter(z, s^p, a_base)`
* `a_final = a_base + residual`

预期 residual 更稳。

## 15.2 无 base-action 条件输入

比较：

* `adapter(z, s^p)`
* `adapter(z, s^p, a_base)`

RLT 和 PLD 都表明 reference/base action 是重要信息源。

## 15.3 绝对 reward vs 相对改进 reward

比较：

* `-MAE(a_final, gt)`
* `MAE(a_base, gt) - MAE(a_final, gt)`

预期相对改进 reward 更契合 adapter 目标。

## 15.4 不加 residual penalty

验证是否会出现过大偏移、训练不稳。

---

# 16. 风险与注意事项

## 16.1 Base policy 太差时，adapter 难以救

该方案默认 base VLA 提供“还不错”的初始提案。若 base action 已经离专家很远，局部 residual 可能不够。
应对方法：

* 提高 `residual_bound`
* 或增加一个 curriculum：先只训练 base 误差中等的样本，再扩展到更难样本

## 16.2 Base policy 太好时，reward 接近 0

若某些样本 base 已基本匹配专家，则 improvement reward 近乎为 0。
这是正常现象，说明这些样本对 adapter 训练价值有限。可考虑：

* 对 `base_mae` 过小样本降采样
* 或按 `base_mae` 做 sample weighting

## 16.3 过度依赖 base action

后续可考虑引入 RLT 风格的 reference-dropout，在部分 batch 中把 `a_base` 输入置零，防止 adapter 只学会“复制上下文”。但第一版不建议加入。RLT 使用 reference dropout 的动机正是避免 actor 过度依赖 reference。

---

# 17. 最终结论

本设计文档对应的方案可以概括为：

**在冻结的 flow-based VLA 外部增加一个条件于 VLA 表征与 base action chunk 的 residual Gaussian adapter，仅用你现有的 offline reward + DatasetEnv + GRPO 框架训练该 adapter，而不直接优化 VLA 本体。**

形式上：

[
z = f_{\text{vla}}(o,\ell),\quad a_{\text{base}}=\pi_{\text{vla}}(o,\ell)
]

[
\Delta a \sim \pi_\theta(\cdot \mid z,s^p,a_{\text{base}})
]

[
a_{\text{final}} = a_{\text{base}} + \xi\tanh(\Delta a)
]

[
r = \mathrm{MAE}(a_{\text{base}}, a_{gt}) - \mathrm{MAE}(a_{\text{final}}, a_{gt}) - \lambda |\Delta a|_1
]

这一路线同时继承了：

* 你现有 offline reward pipeline 的工程优势；
* RLT 的“frozen VLA + small policy + reference action conditioning”思想；
* PLD 的“residual Gaussian policy 比直接优化 flow VLA 更现实”的设计逻辑。