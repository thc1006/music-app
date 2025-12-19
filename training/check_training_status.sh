#!/bin/bash
# 快速檢查訓練狀態腳本

echo "========================================="
echo "🔍 Phase 9 修正訓練狀態檢查"
echo "========================================="
echo ""

# 檢查訓練進程
if ps aux | grep -q "[2]425593"; then
    echo "✅ 訓練進程運行中 (PID: 2425593)"
else
    echo "⏸️  訓練進程已結束"
fi
echo ""

# GPU 狀態
echo "📊 GPU 狀態:"
nvidia-smi --query-gpu=name,utilization.gpu,temperature.gpu,memory.used,memory.total --format=csv,noheader
echo ""

# 訓練進度
if [ -f "harmony_omr_v2_phase9_fixed/phase9_with_phase8_config2/results.csv" ]; then
    echo "📈 訓練進度（最近 5 個 epochs）:"
    tail -5 harmony_omr_v2_phase9_fixed/phase9_with_phase8_config2/results.csv | awk -F',' '{printf "Epoch %3d: mAP50=%.4f, mAP50-95=%.4f\n", $1+1, $11, $12}'
    echo ""
    
    # 最佳結果
    BEST_MAP=$(tail -n +2 harmony_omr_v2_phase9_fixed/phase9_with_phase8_config2/results.csv | awk -F',' 'BEGIN{max=0} {if($11>max) {max=$11; epoch=$1+1}} END{printf "Epoch %d: %.4f\n", epoch, max}')
    echo "🏆 當前最佳 mAP50: $BEST_MAP"
else
    echo "⏳ results.csv 尚未生成（第一個 epoch 尚未完成）"
fi
echo ""

# 最佳模型
if [ -f "harmony_omr_v2_phase9_fixed/phase9_with_phase8_config2/weights/best.pt" ]; then
    BEST_SIZE=$(ls -lh harmony_omr_v2_phase9_fixed/phase9_with_phase8_config2/weights/best.pt | awk '{print $5}')
    echo "💾 最佳模型: best.pt ($BEST_SIZE)"
else
    echo "⏳ 最佳模型尚未保存"
fi
echo ""

echo "========================================="
echo "📝 詳細日誌:"
echo "  - 訓練日誌: harmony_omr_v2_phase9_fixed/phase9_with_phase8_config/training.log"
echo "  - 監控日誌: logs/ultimate_pipeline/pipeline_20251209_210437.log"
echo "========================================="
