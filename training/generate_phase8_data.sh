#!/bin/bash
# Phase 8 合成數據生成腳本
# 為稀有類別 (double_flat, dynamic_loud) 生成 5000+ 訓練樣本

set -e  # 遇到錯誤立即退出

# 配置
OUTPUT_DIR="datasets/yolo_synthetic_phase8"
COUNT_PER_CLASS=5000
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$SCRIPT_DIR"

echo "=================================================="
echo "🎼 Phase 8 合成數據生成 - LilyPond"
echo "=================================================="
echo "輸出目錄: $OUTPUT_DIR"
echo "每類別數量: $COUNT_PER_CLASS"
echo ""

# 檢查 LilyPond
echo "🔍 檢查 LilyPond..."
if ! command -v lilypond &> /dev/null; then
    echo "❌ LilyPond 未安裝"
    echo "安裝: sudo apt install lilypond"
    exit 1
fi
lilypond_version=$(lilypond --version | head -1)
echo "✅ $lilypond_version"
echo ""

# 檢查 Python 依賴
echo "🔍 檢查 Python 依賴..."
python -c "import PIL, numpy, scipy" 2>/dev/null || {
    echo "❌ Python 依賴缺失"
    echo "安裝: pip install -r requirements-synthetic.txt"
    exit 1
}
echo "✅ Python 依賴已安裝"
echo ""

# 清理舊數據（如果存在）
if [ -d "$OUTPUT_DIR" ]; then
    echo "⚠️  檢測到舊數據: $OUTPUT_DIR"
    read -p "是否刪除並重新生成? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$OUTPUT_DIR"
        echo "✅ 舊數據已清理"
    else
        echo "❌ 取消操作"
        exit 1
    fi
fi
echo ""

# 開始計時
start_time=$(date +%s)

echo "=================================================="
echo "📊 第一階段: 生成 Class 17 (double_flat)"
echo "=================================================="
python lilypond_synthetic_generator.py \
    --class 17 \
    --count $COUNT_PER_CLASS \
    --output "$OUTPUT_DIR"

if [ $? -ne 0 ]; then
    echo "❌ Class 17 生成失敗"
    exit 1
fi
echo ""

echo "=================================================="
echo "📊 第二階段: 生成 Class 31 (dynamic_loud)"
echo "=================================================="
python lilypond_synthetic_generator.py \
    --class 31 \
    --count $COUNT_PER_CLASS \
    --output "$OUTPUT_DIR"

if [ $? -ne 0 ]; then
    echo "❌ Class 31 生成失敗"
    exit 1
fi
echo ""

# 結束計時
end_time=$(date +%s)
duration=$((end_time - start_time))
minutes=$((duration / 60))
seconds=$((duration % 60))

echo "=================================================="
echo "✅ Phase 8 數據生成完成！"
echo "=================================================="
echo "總耗時: ${minutes}m ${seconds}s"
echo ""
echo "📁 輸出位置:"
echo "  圖片: $OUTPUT_DIR/images/"
echo "  標註: $OUTPUT_DIR/labels/"
echo "  統計: $OUTPUT_DIR/generation_stats.json"
echo ""

# 顯示統計
if [ -f "$OUTPUT_DIR/generation_stats.json" ]; then
    echo "📊 生成統計:"
    cat "$OUTPUT_DIR/generation_stats.json" | python -m json.tool
    echo ""
fi

# 統計文件數量
img_count=$(ls "$OUTPUT_DIR/images/" | wc -l)
label_count=$(ls "$OUTPUT_DIR/labels/" | wc -l)
echo "📈 文件統計:"
echo "  圖片數: $img_count"
echo "  標註數: $label_count"
echo ""

# 下一步提示
echo "🚀 下一步操作:"
echo "  1. 驗證數據質量:"
echo "     python validate_synthetic_data.py $OUTPUT_DIR"
echo ""
echo "  2. 合併到 Phase 4 數據集:"
echo "     python merge_datasets_phase8.py \\"
echo "       --phase4 datasets/yolo_harmony_v2_phase4 \\"
echo "       --synthetic $OUTPUT_DIR \\"
echo "       --output datasets/yolo_harmony_v2_phase8"
echo ""
echo "  3. 開始 Phase 8 訓練:"
echo "     python yolo12_train_phase8.py"
echo ""
