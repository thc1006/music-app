#!/bin/bash
# 自動執行 Phase 4 -> Phase 5 -> Phase 6
# 全部按順序執行

LOG_DIR="logs"
mkdir -p $LOG_DIR

echo "========================================"
echo "全自動訓練: Phase 4 -> 5 -> 6"
echo "========================================"
echo "開始時間: $(date)"
echo ""

cd /home/thc1006/dev/music-app/training
source venv_yolo12/bin/activate

# ========================================
# Phase 4: 弱類別微調
# ========================================
PHASE4_PID=$(pgrep -of "exp4_focal_loss.py" 2>/dev/null | head -1)

if [ -n "$PHASE4_PID" ]; then
    echo "[Phase 4] 運行中 (PID: $PHASE4_PID)"
    echo "[Phase 4] 等待完成..."

    while kill -0 $PHASE4_PID 2>/dev/null; do
        PROGRESS=$(tail -1 $LOG_DIR/exp4_finetune.log 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' | grep -oE '[0-9]+/30' | tail -1)
        echo "[$(date '+%H:%M:%S')] Phase 4: Epoch $PROGRESS"
        sleep 120
    done

    echo ""
    echo "[Phase 4] 完成! $(date)"
else
    echo "[Phase 4] 未運行，跳過"
fi

# 清理 GPU
python -c "import torch; torch.cuda.empty_cache(); print('GPU 記憶體已清理')"
sleep 5

# ========================================
# Phase 5: DINOv3 蒸餾
# ========================================
echo ""
echo "========================================"
echo "[Phase 5] DINOv3 蒸餾"
echo "========================================"
echo "開始時間: $(date)"

if [ -f "exp5_dinov3_distill.py" ]; then
    python exp5_dinov3_distill.py 2>&1 | tee $LOG_DIR/exp5_dinov3.log
    echo "[Phase 5] 完成! $(date)"
else
    echo "[Phase 5] 腳本不存在，跳過"
fi

# 清理 GPU
python -c "import torch; torch.cuda.empty_cache(); print('GPU 記憶體已清理')"
sleep 5

# ========================================
# Phase 6: YOLO11m 升級
# ========================================
echo ""
echo "========================================"
echo "[Phase 6] YOLO11m 升級訓練"
echo "========================================"
echo "開始時間: $(date)"

if [ -f "exp6_yolo11m.py" ]; then
    python exp6_yolo11m.py 2>&1 | tee $LOG_DIR/exp6_yolo11m.log
    echo "[Phase 6] 完成! $(date)"
else
    echo "[Phase 6] 腳本不存在，跳過"
fi

# ========================================
# 總結
# ========================================
echo ""
echo "========================================"
echo "全部訓練完成!"
echo "========================================"
echo "結束時間: $(date)"
echo ""
echo "結果位置:"
echo "  Phase 4: harmony_omr_v2_experiments/exp4_finetune/"
echo "  Phase 5: harmony_omr_v2_experiments/exp5_dinov3/"
echo "  Phase 6: harmony_omr_v2_experiments/exp6_yolo11m/"
