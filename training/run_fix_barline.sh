#!/bin/bash
# Barline 標註修復執行腳本
# 使用方法：./run_fix_barline.sh

set -e

echo "=================================="
echo "🔧 Barline 標註修復腳本"
echo "=================================="

# 檢查虛擬環境
if [ ! -d "venv_yolo12" ]; then
    echo "❌ 錯誤：找不到 venv_yolo12 虛擬環境"
    exit 1
fi

# 啟動虛擬環境
echo "📦 啟動虛擬環境..."
source venv_yolo12/bin/activate

# 檢查依賴
echo "📦 檢查依賴套件..."
pip install -q matplotlib Pillow tqdm numpy

# 運行測試（可選）
read -p "是否先運行測試驗證邏輯？(y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🧪 運行測試..."
    python test_fix_barline.py
    if [ $? -ne 0 ]; then
        echo "❌ 測試失敗，中止修復"
        exit 1
    fi
fi

# 確認執行
echo ""
echo "準備開始修復 Barline 標註："
echo "  輸入: datasets/yolo_harmony_v2_phase5/"
echo "  輸出: datasets/yolo_harmony_v2_phase6_fixed/"
echo ""
read -p "確定要繼續嗎？(y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ 已取消"
    exit 0
fi

# 執行修復
echo ""
echo "🚀 開始修復..."
python fix_barline_annotations.py

# 檢查結果
if [ $? -eq 0 ]; then
    echo ""
    echo "=================================="
    echo "✅ 修復完成！"
    echo "=================================="
    echo ""
    echo "輸出目錄: datasets/yolo_harmony_v2_phase6_fixed/"
    echo ""
    echo "生成的文件："
    ls -lh datasets/yolo_harmony_v2_phase6_fixed/*.txt datasets/yolo_harmony_v2_phase6_fixed/*.png 2>/dev/null || true
    echo ""
    echo "下一步："
    echo "  1. 檢查修復報告: cat datasets/yolo_harmony_v2_phase6_fixed/fix_report.txt"
    echo "  2. 查看可視化對比: datasets/yolo_harmony_v2_phase6_fixed/*.png"
    echo "  3. 開始 Phase 6 訓練: python yolo12_train_phase6.py"
else
    echo ""
    echo "❌ 修復過程中發生錯誤"
    exit 1
fi
