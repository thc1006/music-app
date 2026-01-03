#!/bin/bash
# 備份 DINOv3 蒸餾訓練最佳模型
# 用法: ./backup_best_model.sh [版本號]

VERSION=${1:-"v2"}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/home/thc1006/dev/music-app/training/model_backups"

# 創建備份目錄
mkdir -p "$BACKUP_DIR"

# 來源路徑
SOURCE_DIR="/home/thc1006/dev/music-app/training/harmony_omr_v2_dinov3_distill_${VERSION}/dinov3_enhanced"

if [ -f "$SOURCE_DIR/weights/best.pt" ]; then
    # 備份最佳模型
    BACKUP_NAME="dinov3_distill_${VERSION}_best_${TIMESTAMP}.pt"
    cp "$SOURCE_DIR/weights/best.pt" "$BACKUP_DIR/$BACKUP_NAME"

    # 複製訓練結果
    cp "$SOURCE_DIR/results.csv" "$BACKUP_DIR/dinov3_distill_${VERSION}_results_${TIMESTAMP}.csv" 2>/dev/null
    cp "$SOURCE_DIR/args.yaml" "$BACKUP_DIR/dinov3_distill_${VERSION}_args_${TIMESTAMP}.yaml" 2>/dev/null

    echo "✅ 備份完成!"
    echo "   模型: $BACKUP_DIR/$BACKUP_NAME"
    echo "   大小: $(du -h "$BACKUP_DIR/$BACKUP_NAME" | cut -f1)"

    # 顯示備份目錄內容
    echo ""
    echo "📁 備份目錄內容:"
    ls -lh "$BACKUP_DIR"
else
    echo "❌ 找不到模型文件: $SOURCE_DIR/weights/best.pt"
    echo "   訓練可能尚未完成"
fi
