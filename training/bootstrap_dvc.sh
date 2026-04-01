#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_NAME="${1:-localstorage}"
REMOTE_PATH="${2:-$REPO_ROOT/.dvc/local-remote}"

if ! command -v dvc >/dev/null 2>&1; then
  echo "dvc 未安裝，請先執行: pip install dvc"
  exit 1
fi

cd "$REPO_ROOT"

if [ ! -d ".dvc" ]; then
  echo "[DVC] 初始化專案..."
  dvc init
else
  echo "[DVC] 已初始化，略過 dvc init"
fi

mkdir -p "$REMOTE_PATH"
dvc remote add -d "$REMOTE_NAME" "$REMOTE_PATH" --force

echo "[DVC] default remote 設定完成: $REMOTE_NAME -> $REMOTE_PATH"
echo "[DVC] 下一步建議:"
echo "  1) dvc add training/datasets/yolo_harmony_v2_phase8_final"
echo "  2) git add .dvc .gitignore training/datasets/yolo_harmony_v2_phase8_final.dvc"
echo "  3) git commit -m 'Track phase8 dataset with DVC'"
echo "  4) dvc push"
