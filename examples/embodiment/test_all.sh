#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# 这个脚本的目标：
# 1) 连续跑 5 次评估（null, 50, 100, 150, 200）
# 2) 每次跑之前自动改写 config 里的 ckpt_path
# 3) 每次结果单独归档，避免混淆
# 4) 结束后恢复原始 config，避免污染
# ============================================================

# -------------------------
# 基础路径与固定配置
# -------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_PATH="$(dirname "$(dirname "$SCRIPT_DIR")")"

CONFIG_NAME="libero_130_grpo_openvlaoft_adapter_eval"
CONFIG_FILE="${SCRIPT_DIR}/config/${CONFIG_NAME}.yaml"
EVAL_SCRIPT="${SCRIPT_DIR}/eval_embodiment.sh"

# 你要跑的 5 组：第一组用 null，后四组用 global_step_xxx。
RUN_LABELS=("null" "50" "100" "150" "200")

# -------------------------
# 工具函数
# -------------------------
die() {
  echo "[ERROR] $*" >&2
  exit 1
}

validate_inputs() {
  [[ -f "${CONFIG_FILE}" ]] || die "Config file not found: ${CONFIG_FILE}"
  [[ -x "${EVAL_SCRIPT}" ]] || die "Eval script is not executable: ${EVAL_SCRIPT}"
}

# 从当前 YAML 的 ckpt_path 推断 checkpoints 根路径（直到 .../checkpoints）
detect_ckpt_base_prefix() {
  local current_ckpt
  current_ckpt="$(python - "${CONFIG_FILE}" <<'PY'
import re
import sys

config_path = sys.argv[1]
value = ""
with open(config_path, "r", encoding="utf-8") as f:
    for line in f:
        m = re.match(r"^\s*ckpt_path:\s*(.*)$", line.rstrip("\n"))
        if m:
            v = m.group(1).split("#", 1)[0].strip()
            if v != "null":
                value = v
            break
print(value)
PY
)"

  [[ -n "${current_ckpt}" ]] || die "Failed to detect non-null ckpt_path in ${CONFIG_FILE}"

  python - "${current_ckpt}" <<'PY'
import re
import sys

path = sys.argv[1]
m = re.match(r"^(.*?/checkpoints)/global_step_\d+/actor/model_state_dict/full_weights\.pt$", path)
if not m:
    raise SystemExit(1)
print(m.group(1))
PY
}

# 改写 YAML 中的 ckpt_path
set_ckpt_path() {
  local target="$1"
  python - "${CONFIG_FILE}" "${target}" <<'PY'
import re
import sys

config_path = sys.argv[1]
target = sys.argv[2]

with open(config_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

found = False
for i, line in enumerate(lines):
    if re.match(r"^\s*ckpt_path:\s*", line):
        indent = re.match(r"^(\s*)", line).group(1)
        lines[i] = f"{indent}ckpt_path: {target}\n"
        found = True
        break

if not found:
    raise SystemExit("ckpt_path field not found")

with open(config_path, "w", encoding="utf-8") as f:
    f.writelines(lines)
PY
}

# 从 wrapper.log 中提取这次 eval 实际写到哪里（eval_embodiment.sh 里会打印命令）
parse_source_log_dir() {
  local wrapper_log="$1"
  python - "${wrapper_log}" <<'PY'
import re
import sys

wrapper_log = sys.argv[1]
pattern = re.compile(r"runner\.logger\.log_path=([^\s]+)")
found = ""
with open(wrapper_log, "r", encoding="utf-8", errors="replace") as f:
    for line in f:
        m = pattern.search(line)
        if m:
            found = m.group(1)
print(found)
PY
}

# 根据 label 生成本轮要写入 config 的 ckpt_path
build_ckpt_value() {
  local label="$1"
  local base_prefix="$2"
  if [[ "${label}" == "null" ]]; then
    echo "null"
  else
    echo "${base_prefix}/global_step_${label}/actor/model_state_dict/full_weights.pt"
  fi
}

# 跑单轮评估 + 归档
run_one_eval() {
  local run_id="$1"
  local label="$2"
  local ckpt_value="$3"
  local run_root="$4"
  local summary_file="$5"

  local run_dir wrapper_log source_log_dir organized_dir
  run_dir="${run_root}/run_${run_id}_${label}"
  wrapper_log="${run_dir}/wrapper.log"
  organized_dir="${run_dir}/eval_output"

  mkdir -p "${run_dir}"

  echo "===== Run ${run_id} / label=${label} ====="
  echo "Setting ckpt_path to: ${ckpt_value}"
  set_ckpt_path "${ckpt_value}"
  cp "${CONFIG_FILE}" "${run_dir}/config_snapshot.yaml"

  (
    cd "${REPO_PATH}"
    bash "${EVAL_SCRIPT}" "${CONFIG_NAME}"
  ) 2>&1 | tee "${wrapper_log}"

  source_log_dir="$(parse_source_log_dir "${wrapper_log}")"
  if [[ -n "${source_log_dir}" && -d "${source_log_dir}" ]]; then
    mv "${source_log_dir}" "${organized_dir}"
  else
    echo "[WARN] Could not locate source eval log dir. parsed='${source_log_dir}'" >&2
  fi

  {
    echo "run_id=${run_id}"
    echo "label=${label}"
    echo "ckpt_path=${ckpt_value}"
    echo "source_log_dir=${source_log_dir}"
    echo "organized_eval_dir=${organized_dir}"
  } > "${run_dir}/metadata.txt"

  echo -e "${run_id}\t${label}\t${ckpt_value}\t${source_log_dir}\t${organized_dir}" >> "${summary_file}"
}

# -------------------------
# 主流程
# -------------------------
main() {
  validate_inputs

  # 备份原始配置，脚本退出时自动恢复
  local backup_file
  backup_file="$(mktemp)"
  cp "${CONFIG_FILE}" "${backup_file}"
  trap 'cp "${backup_file}" "${CONFIG_FILE}"; rm -f "${backup_file}"' EXIT

  local ckpt_base_prefix
  ckpt_base_prefix="$(detect_ckpt_base_prefix)" || die "ckpt_path format is unexpected in ${CONFIG_FILE}"

  local run_root summary_file
  run_root="${REPO_PATH}/logs/eval_sweeps/${CONFIG_NAME}_$(date +'%Y%m%d-%H%M%S')"
  mkdir -p "${run_root}"
  summary_file="${run_root}/summary.tsv"
  echo -e "run_id\tlabel\tckpt_path\tsource_log_dir\torganized_dir" > "${summary_file}"

  local idx label run_id ckpt_value
  for idx in "${!RUN_LABELS[@]}"; do
    label="${RUN_LABELS[$idx]}"
    run_id="$(printf '%02d' $((idx + 1)))"
    ckpt_value="$(build_ckpt_value "${label}" "${ckpt_base_prefix}")"
    run_one_eval "${run_id}" "${label}" "${ckpt_value}" "${run_root}" "${summary_file}"
  done

  echo "All runs finished."
  echo "Organized outputs under: ${run_root}"
  echo "Summary file: ${summary_file}"
}

main "$@"
