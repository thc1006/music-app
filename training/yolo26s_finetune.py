#!/usr/bin/env python3
"""
YOLO26s 微調訓練腳本
目標: 利用 YOLO26 的 NMS-Free + STAL 特性改善 OMR 小物件檢測

YOLO26 關鍵特性:
- NMS-Free 端到端推論
- STAL (小目標感知標籤分配)
- ProgLoss (漸進式損失平衡)
- MuSGD 優化器
- 移除 DFL (利於量化導出)

預期改善:
- barline_double 等小物件檢測
- TFLite INT8 導出更穩定
- 不再需要調整 NMS IoU 閾值
"""

import os
import sys
from datetime import datetime
from pathlib import Path

# 確保在正確目錄
os.chdir('/home/thc1006/dev/music-app/training')

def main():
    print("=" * 70)
    print("YOLO26s 微調訓練")
    print(f"開始時間: {datetime.now()}")
    print("=" * 70)

    from ultralytics import YOLO
    import torch

    # 確認 GPU
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"PyTorch: {torch.__version__}")

    # 配置
    MODEL = 'yolo26s.pt'  # 預訓練權重
    DATA_YAML = 'datasets/yolo_harmony_v2_phase8_final/harmony_phase8_final.yaml'
    PROJECT = 'harmony_omr_v2_yolo26'
    NAME = 'yolo26s_finetune'

    # 訓練參數
    config = {
        'data': DATA_YAML,
        'epochs': 100,
        'imgsz': 1280,
        'batch': 8,  # RTX 5090 32GB
        'patience': 30,  # Early stopping
        'save': True,
        'save_period': 10,  # 每 10 epoch 保存
        'cache': False,  # 避免記憶體問題
        'device': 0,
        'workers': 16,
        'project': PROJECT,
        'name': NAME,
        'exist_ok': True,
        'pretrained': True,
        'optimizer': 'AdamW',  # YOLO26 也支援 MuSGD
        'lr0': 0.001,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'cos_lr': True,
        'amp': True,  # YOLO26 支援混合精度
        'plots': True,
        'val': True,
        'verbose': True,
    }

    print(f"\n模型: {MODEL}")
    print(f"數據集: {DATA_YAML}")
    print(f"輸出目錄: {PROJECT}/{NAME}")
    print(f"\n訓練配置:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # 載入模型
    print(f"\n載入 YOLO26s 預訓練權重...")
    model = YOLO(MODEL)

    # 顯示模型資訊
    print("\n模型架構:")
    model.info()

    # 開始訓練
    print("\n" + "=" * 70)
    print("開始訓練...")
    print("=" * 70)

    results = model.train(**config)

    # 訓練完成
    print("\n" + "=" * 70)
    print("訓練完成！")
    print(f"結束時間: {datetime.now()}")
    print("=" * 70)

    # 驗證最佳模型
    print("\n驗證最佳模型...")
    best_model_path = f'{PROJECT}/{NAME}/weights/best.pt'
    if os.path.exists(best_model_path):
        best_model = YOLO(best_model_path)
        val_results = best_model.val(data=DATA_YAML, imgsz=1280)

        print(f"\n最終結果:")
        print(f"  mAP50:     {val_results.box.map50:.4f}")
        print(f"  mAP50-95:  {val_results.box.map:.4f}")
        print(f"  Precision: {val_results.box.mp:.4f}")
        print(f"  Recall:    {val_results.box.mr:.4f}")

        # 與 Ultimate v5 比較
        print(f"\n與 Ultimate v5 (mAP50=0.6979) 比較:")
        diff = (val_results.box.map50 - 0.6979) * 100
        print(f"  差異: {diff:+.2f}%")

        if val_results.box.map50 > 0.70:
            print("\n🎉 SUCCESS: mAP50 > 0.70!")

        # 與 NMS IoU=0.55 調整後比較
        print(f"\n與 NMS IoU=0.55 調整後 (mAP50=0.7059) 比較:")
        diff2 = (val_results.box.map50 - 0.7059) * 100
        print(f"  差異: {diff2:+.2f}%")

    print(f"\n模型保存位置: {PROJECT}/{NAME}/weights/")
    print("  - best.pt: 最佳權重")
    print("  - last.pt: 最後權重")

    return results

if __name__ == '__main__':
    main()
