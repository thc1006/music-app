#!/bin/bash
# Barline 數據集完整處理流程
# 此腳本會依序執行: 下載 -> 轉換 -> 合併

set -e  # 遇到錯誤立即退出

SCRIPT_DIR="/home/thc1006/dev/music-app/training/scripts"
DATASET_DIR="/home/thc1006/dev/music-app/training/datasets/external_barlines"

echo "========================================"
echo "Barline 數據集處理流程"
echo "========================================"
echo "腳本目錄: $SCRIPT_DIR"
echo "數據集目錄: $DATASET_DIR"
echo ""

# 檢查 Python 環境
if ! command -v python3 &> /dev/null; then
    echo "錯誤: 未找到 python3"
    exit 1
fi

# 檢查依賴
echo "檢查依賴套件..."
python3 -c "import cv2, tqdm, yaml, requests" 2>/dev/null || {
    echo "安裝依賴套件..."
    pip install opencv-python tqdm pyyaml requests
}

# 步驟 1: 下載數據集
echo ""
echo "========================================"
echo "步驟 1: 下載數據集"
echo "========================================"
read -p "是否下載所有數據集? (y/n) [y]: " download_choice
download_choice=${download_choice:-y}

if [[ $download_choice == "y" || $download_choice == "Y" ]]; then
    cd "$SCRIPT_DIR"
    python3 download_all_datasets.py --datasets all
else
    echo "跳過下載步驟"
    echo "如需手動下載，請參考: $DATASET_DIR/README.md"
fi

# 步驟 2: 轉換數據集
echo ""
echo "========================================"
echo "步驟 2: 轉換數據集"
echo "========================================"

# 2.1 轉換 OMR Layout
if [ -d "$DATASET_DIR/omr_layout/datasets-release" ] || [ -d "$DATASET_DIR/omr_layout" ]; then
    echo ""
    echo "轉換 OMR Layout Analysis..."
    cd "$SCRIPT_DIR"
    python3 convert_omr_layout.py || {
        echo "警告: OMR Layout 轉換失敗"
    }
else
    echo "跳過 OMR Layout (未找到數據)"
fi

# 2.2 轉換 AudioLabs
if [ -d "$DATASET_DIR/audiolabs/2019_MeasureDetection_ISMIR2019" ] || [ -d "$DATASET_DIR/audiolabs" ]; then
    echo ""
    echo "轉換 AudioLabs v2..."
    cd "$SCRIPT_DIR"
    python3 convert_audiolabs.py || {
        echo "警告: AudioLabs 轉換失敗"
    }
else
    echo "跳過 AudioLabs (未找到數據)"
fi

# 2.3 轉換 DoReMi
if [ -d "$DATASET_DIR/doremi/DoReMi_1.0" ] || [ -d "$DATASET_DIR/doremi" ]; then
    echo ""
    echo "轉換 DoReMi..."
    cd "$SCRIPT_DIR"
    python3 convert_doremi.py || {
        echo "警告: DoReMi 轉換失敗"
    }
else
    echo "跳過 DoReMi (未找到數據)"
fi

# 步驟 3: 合併數據集
echo ""
echo "========================================"
echo "步驟 3: 合併數據集"
echo "========================================"

# 檢查是否有任何轉換後的數據集
converted_count=0
[ -d "$DATASET_DIR/omr_layout/converted" ] && ((converted_count++))
[ -d "$DATASET_DIR/audiolabs/converted" ] && ((converted_count++))
[ -d "$DATASET_DIR/doremi/converted" ] && ((converted_count++))

if [ $converted_count -gt 0 ]; then
    echo "找到 $converted_count 個已轉換的數據集"
    echo "開始合併..."
    cd "$SCRIPT_DIR"
    python3 merge_barline_datasets.py
else
    echo "錯誤: 未找到任何已轉換的數據集"
    exit 1
fi

# 完成
echo ""
echo "========================================"
echo "處理完成!"
echo "========================================"
echo ""
echo "數據集位置: $DATASET_DIR/merged"
echo "配置文件: $DATASET_DIR/merged/data.yaml"
echo "統計信息: $DATASET_DIR/merged/merge_stats.json"
echo ""
echo "下一步 - 訓練模型:"
echo "  cd /home/thc1006/dev/music-app/training"
echo "  yolo detect train data=$DATASET_DIR/merged/data.yaml model=yolov8n.pt epochs=100"
echo ""
