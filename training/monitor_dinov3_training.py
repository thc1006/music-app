#!/usr/bin/env python3
"""
DINOv3 蒸餾訓練進度監控腳本
監控 yolo12_dinov3_distillation_v2.py 的訓練進度
"""

import re
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta

LOG_FILE = Path('/home/thc1006/dev/music-app/training/logs/dinov3_distillation_v2.log')
WEIGHTS_DIR = Path('/home/thc1006/dev/music-app/training/harmony_omr_v2_dinov3_distill_v2/dinov3_enhanced/weights')

def parse_log():
    """解析訓練日誌"""
    if not LOG_FILE.exists():
        print("❌ 日誌文件不存在")
        return None

    with open(LOG_FILE, 'r') as f:
        content = f.read()

    # 找到所有 epoch 進度
    epoch_pattern = r'(\d+)/100\s+[\d.]+G\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)'
    epochs = re.findall(epoch_pattern, content)

    # 找到所有驗證結果
    val_pattern = r'all\s+\d+\s+\d+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)'
    validations = re.findall(val_pattern, content)

    # 找到最新的 mAP 結果
    map_pattern = r'mAP50:\s+([\d.]+)'
    map_results = re.findall(map_pattern, content)

    return {
        'current_epoch': int(epochs[-1][0]) if epochs else 0,
        'latest_losses': {
            'box_loss': float(epochs[-1][1]) if epochs else 0,
            'cls_loss': float(epochs[-1][2]) if epochs else 0,
            'dfl_loss': float(epochs[-1][3]) if epochs else 0,
        } if epochs else {},
        'validations': validations,
        'map_results': map_results,
    }

def check_weights():
    """檢查權重文件"""
    if not WEIGHTS_DIR.exists():
        return None

    best_pt = WEIGHTS_DIR / 'best.pt'
    last_pt = WEIGHTS_DIR / 'last.pt'

    return {
        'best_pt': {
            'exists': best_pt.exists(),
            'size_mb': best_pt.stat().st_size / (1024*1024) if best_pt.exists() else 0,
            'modified': datetime.fromtimestamp(best_pt.stat().st_mtime) if best_pt.exists() else None,
        },
        'last_pt': {
            'exists': last_pt.exists(),
            'size_mb': last_pt.stat().st_size / (1024*1024) if last_pt.exists() else 0,
            'modified': datetime.fromtimestamp(last_pt.stat().st_mtime) if last_pt.exists() else None,
        },
    }

def main():
    print("=" * 60)
    print("DINOv3 蒸餾訓練進度監控")
    print(f"時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 解析日誌
    log_data = parse_log()
    if log_data:
        current_epoch = log_data['current_epoch']
        total_epochs = 100
        progress = current_epoch / total_epochs * 100

        print(f"\n📊 訓練進度: Epoch {current_epoch}/{total_epochs} ({progress:.1f}%)")
        print(f"   [{'█' * int(progress/2)}{' ' * (50 - int(progress/2))}]")

        if log_data['latest_losses']:
            losses = log_data['latest_losses']
            print(f"\n📉 當前損失:")
            print(f"   box_loss: {losses['box_loss']:.4f}")
            print(f"   cls_loss: {losses['cls_loss']:.4f}")
            print(f"   dfl_loss: {losses['dfl_loss']:.4f}")

        if log_data['validations']:
            last_val = log_data['validations'][-1]
            print(f"\n🎯 最新驗證結果:")
            print(f"   Precision: {last_val[0]}")
            print(f"   Recall:    {last_val[1]}")
            print(f"   mAP50:     {last_val[2]}")
            print(f"   mAP50-95:  {last_val[3]}")

        # 估算完成時間
        if current_epoch > 0:
            # 假設每個 epoch 約 4 分鐘
            remaining_epochs = total_epochs - current_epoch
            eta_minutes = remaining_epochs * 4
            eta = datetime.now() + timedelta(minutes=eta_minutes)
            print(f"\n⏱️ 預計完成時間: {eta.strftime('%Y-%m-%d %H:%M')} (約 {eta_minutes/60:.1f} 小時)")

    # 檢查權重文件
    weights = check_weights()
    if weights:
        print(f"\n💾 模型文件:")
        if weights['best_pt']['exists']:
            print(f"   best.pt: {weights['best_pt']['size_mb']:.1f} MB")
            print(f"            更新於: {weights['best_pt']['modified'].strftime('%H:%M:%S')}")
        if weights['last_pt']['exists']:
            print(f"   last.pt: {weights['last_pt']['size_mb']:.1f} MB")
            print(f"            更新於: {weights['last_pt']['modified'].strftime('%H:%M:%S')}")

    # Phase 8 基準對比
    print(f"\n📌 Phase 8 基準:")
    print(f"   mAP50:    0.6444")
    print(f"   mAP50-95: 0.5809")

    print("\n" + "=" * 60)
    print("使用 Ctrl+C 退出")
    print("=" * 60)

if __name__ == "__main__":
    main()
