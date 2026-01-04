#!/bin/bash
#
# YOLO12 Ultimate v5 高解析度訓練啟動腳本
# ==========================================
#
# 使用 nohup 在後台運行，即使終端斷開也會繼續執行
#
# 使用方式:
#   chmod +x start_ultimate_v5_training.sh
#   ./start_ultimate_v5_training.sh
#
# 監控命令:
#   tail -f logs/ultimate_v5_training.log
#   nvidia-smi -l 5
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
echo "YOLO12 Ultimate v5 高解析度訓練"
echo "=========================================="
echo "開始時間: $(date)"
echo "日誌文件: logs/ultimate_v5_training.log"
echo ""

# 使用 nohup 在後台運行
nohup python3 -u yolo12_ultimate_v5_highres.py > logs/ultimate_v5_training.log 2>&1 &

# 獲取進程 ID
PID=$!
echo $PID > logs/ultimate_v5_training.pid

echo "✅ 訓練已在後台啟動"
echo "   PID: $PID"
echo ""
echo "監控命令:"
echo "   tail -f logs/ultimate_v5_training.log"
echo "   nvidia-smi -l 5"
echo ""
echo "停止訓練:"
echo "   kill $PID"
echo ""
