#!/bin/bash
#
# YOLO12 Ultimate v5 Stable 訓練啟動腳本
# =======================================
#
# 穩定版本：batch=4 確保 0% OOM
#
# 使用方式:
#   chmod +x start_ultimate_v5_stable.sh
#   ./start_ultimate_v5_stable.sh
#

cd /home/thc1006/dev/music-app/training

# 創建日誌目錄
mkdir -p logs

# 激活虛擬環境
source venv_yolo12/bin/activate

# 設置環境變數
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

# 記錄開始時間
echo "=========================================="
echo "YOLO12 Ultimate v5 Stable 穩定訓練"
echo "=========================================="
echo "開始時間: $(date)"
echo "配置: batch=4, imgsz=1280 (0% OOM)"
echo "日誌: logs/ultimate_v5_stable.log"
echo ""

# 使用 nohup 在後台運行
nohup python3 -u yolo12_ultimate_v5_stable.py > logs/ultimate_v5_stable.log 2>&1 &

# 獲取進程 ID
PID=$!
echo $PID > logs/ultimate_v5_stable.pid

echo "✅ 訓練已在後台啟動"
echo "   PID: $PID"
echo ""
echo "監控命令:"
echo "   tail -f logs/ultimate_v5_stable.log"
echo "   nvidia-smi -l 5"
echo ""
echo "查看狀態:"
echo "   ./monitor_ultimate_v5_stable.sh status"
echo ""
