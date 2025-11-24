#!/bin/bash
# ============================================================
# Phase 3: 突破 mAP50=0.509 瓶頸
# ============================================================
#
# 基於大規模調研的最佳實踐
# 三管齊下策略：LilyPond合成 + Copy-Paste增強 + DeepScores重映射
#
# 使用方式:
#   chmod +x PHASE3_EXECUTE.sh
#   ./PHASE3_EXECUTE.sh
#
# ============================================================

set -e  # 遇到錯誤時停止

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "Phase 3: 突破 mAP50=0.509 瓶頸"
echo "============================================================"
echo ""
echo "當前模型: harmony_omr_v2_phase2/balanced_training/weights/best.pt"
echo "當前 mAP50: 0.509"
echo "目標 mAP50: 0.65-0.70"
echo ""

# 啟用虛擬環境
source venv_yolo12/bin/activate

# ============================================================
# Step 0: 檢查依賴
# ============================================================
echo "[Step 0] 檢查依賴..."

# 檢查 LilyPond
if command -v lilypond &> /dev/null; then
    echo "✅ LilyPond: $(lilypond --version | head -1)"
    HAS_LILYPOND=true
else
    echo "⚠️ LilyPond 未安裝"
    echo "   建議安裝: sudo apt install lilypond"
    echo "   (將使用 Copy-Paste 方法作為替代)"
    HAS_LILYPOND=false
fi

# 檢查 OpenCV
python -c "import cv2; print(f'✅ OpenCV: {cv2.__version__}')" 2>/dev/null || {
    echo "⚠️ OpenCV 未安裝"
    pip install opencv-python
}

# 檢查 Pillow
python -c "import PIL; print(f'✅ Pillow: {PIL.__version__}')" 2>/dev/null || {
    echo "⚠️ Pillow 未安裝"
    pip install Pillow
}

echo ""

# ============================================================
# Step 1: Copy-Paste 增強（不需要 LilyPond）
# ============================================================
echo "[Step 1] Copy-Paste 增強..."
echo "  從現有訓練數據提取稀有符號並增強"

# 提取符號
python phase3_copy_paste_augmentation.py --extract-symbols \
    --source-dataset datasets/yolo_harmony_v2_phase2 \
    --output-dir datasets/yolo_harmony_v2_phase3_copypaste

# 生成增強數據
python phase3_copy_paste_augmentation.py --augment-all \
    --source-dataset datasets/yolo_harmony_v2_phase2 \
    --output-dir datasets/yolo_harmony_v2_phase3_copypaste

echo "✅ Copy-Paste 增強完成"
echo ""

# ============================================================
# Step 2: LilyPond 精確合成（如果可用）
# ============================================================
if [ "$HAS_LILYPOND" = true ]; then
    echo "[Step 2] LilyPond 精確合成..."
    echo "  生成 double_flat, double_sharp, barline_double"

    python phase3_lilypond_precise_synthesis.py --generate-rare \
        --output datasets/yolo_harmony_v2_phase3_synthetic

    echo "✅ LilyPond 合成完成"
else
    echo "[Step 2] 跳過 LilyPond 合成（未安裝）"
fi
echo ""

# ============================================================
# Step 3: 合併數據集
# ============================================================
echo "[Step 3] 準備混合數據集..."

python phase3_copy_paste_augmentation.py --prepare-dataset \
    --source-dataset datasets/yolo_harmony_v2_phase2 \
    --output-dir datasets/yolo_harmony_v2_phase3_copypaste

echo "✅ 數據集準備完成"
echo ""

# ============================================================
# Step 4: 開始訓練
# ============================================================
echo "[Step 4] 開始 Phase 3 訓練..."
echo ""
echo "============================================================"
echo "訓練參數："
echo "  - 數據集: datasets/yolo_harmony_v2_phase3/harmony_phase3.yaml"
echo "  - 基礎模型: harmony_omr_v2_phase2/balanced_training/weights/best.pt"
echo "  - Epochs: 200"
echo "  - Batch: 16"
echo "============================================================"
echo ""

# 確認是否開始訓練
read -p "是否開始訓練？(y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python yolo12_train_phase3.py
    echo ""
    echo "============================================================"
    echo "Phase 3 訓練完成！"
    echo "============================================================"
    echo "最佳模型: harmony_omr_v2_phase3/copypaste_enhanced/weights/best.pt"
else
    echo "訓練已取消"
    echo ""
    echo "手動訓練指令："
    echo "  source venv_yolo12/bin/activate"
    echo "  python yolo12_train_phase3.py"
fi
