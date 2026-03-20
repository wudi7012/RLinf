#!/usr/bin/env bash
set -e

# 配置名（不带 .yaml 后缀）
CONFIG_NAME="libero_130_grpo_openvlaoft_eval"
# 对应的 YAML 文件路径
CONFIG_FILE="examples/embodiment/config/${CONFIG_NAME}.yaml"

# 需要评估的 global_step 列表
STEPS=(50 100 150 200 250)

echo "使用配置文件: ${CONFIG_FILE}"
echo "将依次评估 checkpoints 的 global_step: ${STEPS[*]}"
echo

for step in "${STEPS[@]}"; do
  echo "==== 开始评估 global_step_${step} ===="

  # 用 sed 修改 YAML 中的 ckpt_path，只替换 global_step_后面的数字
  # 注意：这里假设 ckpt_path 这一行中一定包含 'global_step_数字'
  sed -E "s#(ckpt_path: .*/global_step_)[0-9]+/#\1${step}/#" "${CONFIG_FILE}" > "${CONFIG_FILE}.tmp"
  mv "${CONFIG_FILE}.tmp" "${CONFIG_FILE}"

  # 运行评估脚本
  bash examples/embodiment/eval_embodiment.sh "${CONFIG_NAME}"

  echo "==== 评估 global_step_${step} 完成 ===="
  echo
done

echo "全部评估完成: ${STEPS[*]}"
# 配置权限：sudo chown -R wudi:wudi /home/wudi/src/RLinf/*/