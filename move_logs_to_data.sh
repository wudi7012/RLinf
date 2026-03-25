#!/usr/bin/env bash

set -euo pipefail

SRC_DIR="${SRC_DIR:-/home/wudi/src/RLinf/logs}"
DST_DIR="${DST_DIR:-/home/wudi/wudi_data/Rlinf_logs}"
SLEEP_SECONDS=0

usage() {
  cat <<'EOF'
Usage:
  ./move_logs_to_data.sh
  ./move_logs_to_data.sh --loop 60
  SRC_DIR=/path/to/src DST_DIR=/path/to/dst ./move_logs_to_data.sh

Description:
  Move all entries under the RLinf logs directory into the destination
  directory on the data disk. Hidden files are included.

Options:
  --loop SECONDS   Keep running and move newly created entries every N seconds.
  -h, --help       Show this help message.
EOF
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --loop)
        if [[ $# -lt 2 ]]; then
          echo "Error: --loop requires a positive integer argument." >&2
          exit 1
        fi
        SLEEP_SECONDS="$2"
        shift 2
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        echo "Error: unknown argument: $1" >&2
        usage >&2
        exit 1
        ;;
    esac
  done

  if [[ "$SLEEP_SECONDS" != "0" ]] && ! [[ "$SLEEP_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: --loop expects a positive integer number of seconds." >&2
    exit 1
  fi
}

move_once() {
  local moved=0
  local skipped=0
  local src_real dst_real item base

  if [[ ! -d "$SRC_DIR" ]]; then
    echo "Source directory does not exist: $SRC_DIR" >&2
    return 1
  fi

  mkdir -p "$DST_DIR"

  src_real="$(realpath "$SRC_DIR")"
  dst_real="$(realpath "$DST_DIR")"

  if [[ "$src_real" == "$dst_real" ]]; then
    echo "Source and destination are the same directory: $src_real" >&2
    return 1
  fi

  shopt -s dotglob nullglob
  for item in "$SRC_DIR"/*; do
    base="$(basename "$item")"
    if [[ "$base" == "." || "$base" == ".." ]]; then
      continue
    fi

    if [[ -e "$DST_DIR/$base" ]]; then
      echo "Skip existing target: $DST_DIR/$base"
      skipped=$((skipped + 1))
      continue
    fi

    echo "Moving: $item -> $DST_DIR/"
    mv -- "$item" "$DST_DIR/"
    moved=$((moved + 1))
  done
  shopt -u dotglob nullglob

  echo "Done. moved=$moved skipped=$skipped"
}

main() {
  parse_args "$@"

  if [[ "$SLEEP_SECONDS" == "0" ]]; then
    move_once
    return
  fi

  echo "Watching $SRC_DIR and moving new entries to $DST_DIR every $SLEEP_SECONDS seconds."
  while true; do
    move_once
    sleep "$SLEEP_SECONDS"
  done
}

main "$@"
