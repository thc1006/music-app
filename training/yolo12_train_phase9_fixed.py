#!/usr/bin/env python3
"""
Phase 9 修正配置訓練腳本
使用 Phase 8 的成功配置重新訓練 Phase 9 數據集

關鍵修正:
- epochs: 100 → 150
- lr0: 0.0005 → 0.001
- cls: 0.8 → 0.5
- erasing: 0.4 → 0.0
- warmup_epochs: 2 → 3

預期: mAP50 0.65-0.70
"""

from ultralytics import YOLO
from pathlib import Path

def main():
    print("="*80)
    print("Phase 9 修正配置訓練 - 使用 Phase 8 最佳配置")
    print("="*80)

    # 載入 Phase 8 最佳模型
    model = YOLO('harmony_omr_v2_phase8/phase8_training/weights/best.pt')

    # 訓練配置（完全使用 Phase 8 的成功配置）
    results = model.train(
        # 數據集
        data='datasets/yolo_harmony_v2_phase9_merged/harmony_phase9_merged.yaml',

        # 基礎配置
        epochs=150,              # ← Phase 8 配置
        patience=50,
        batch=24,
        imgsz=640,
        device=0,

        # 優化器（Phase 8 成功配置）
        optimizer='AdamW',
        lr0=0.001,               # ← Phase 8 配置（Phase 9 用 0.0005）
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,         # ← Phase 8 配置（Phase 9 用 2）
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,

        # 損失權重（Phase 8 配置）
        cls=0.5,                 # ← Phase 8 配置（Phase 9 用 0.8）
        box=7.5,
        dfl=1.5,

        # 數據增強（Phase 8 配置）
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=2.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0001,
        flipud=0.0,
        fliplr=0.0,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
        erasing=0.0,             # ← Phase 8 配置（Phase 9 用 0.4）

        # 訓練策略
        amp=True,
        close_mosaic=10,
        pretrained=True,

        # 輸出
        project='harmony_omr_v2_phase9_fixed',
        name='phase9_with_phase8_config',
        exist_ok=False,
        save=True,
        save_period=10,
        plots=True,
        val=True,
    )

    print("\n" + "="*80)
    print("訓練完成！")
    print("="*80)

    # 顯示最終結果
    final_metrics = results.results_dict
    print(f"\n最終指標:")
    print(f"  mAP50: {final_metrics.get('metrics/mAP50(B)', 0):.4f}")
    print(f"  mAP50-95: {final_metrics.get('metrics/mAP50-95(B)', 0):.4f}")
    print(f"  Precision: {final_metrics.get('metrics/precision(B)', 0):.4f}")
    print(f"  Recall: {final_metrics.get('metrics/recall(B)', 0):.4f}")

if __name__ == '__main__':
    main()
