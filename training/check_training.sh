#!/bin/bash
echo "========================================="
echo "YOLO12 訓練狀態檢查"
echo "========================================="

# 檢查進程
if ps aux | grep "yolo12_train_ULTIMATE" | grep -v grep > /dev/null; then
    echo "✅ 訓練進程：運行中"
    ps aux | grep "yolo12_train_ULTIMATE" | grep -v grep | awk '{print "   PID: "$2", 運行時間: "$10", CPU: "$3"%"}'
else
    echo "❌ 訓練進程：已停止"
fi

# 檢查 GPU
echo ""
echo "🎮 GPU 狀態："
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu --format=csv,noheader | \
    awk -F', ' '{printf "   VRAM: %s / %s, 使用率: %s, 功耗: %s, 溫度: %s\n", $1, $2, $3, $4, $5}'

# 當前進度
echo ""
echo "📊 訓練進度："
tail -1 training_ultimate_35classes_batch24.log 2>/dev/null | grep -oP '\d+/600' | head -1 | xargs -I {} echo "   Epoch: {}"

# 最新權重
echo ""
echo "💾 最新權重："
ls -lht harmony_omr_v2_ultimate/train2/weights/ 2>/dev/null | head -3 | tail -2 | \
    awk '{print "   "$9" - "$5" ("$6" "$7" "$8")"}'

echo "========================================="
