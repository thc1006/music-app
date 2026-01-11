#!/usr/bin/env python3
"""
實驗 6: 分階段微調 - 凍結 backbone，只訓練 head
目標: 改善 tie/barline_final 分類混淆問題

預計時間: ~3 小時
"""

import os
import sys
from datetime import datetime
from ultralytics import YOLO

def main():
    # 配置
    BASE_MODEL = 'harmony_omr_v2_ultimate_v5_stable/stable_1280_resumed/weights/best.pt'
    DATA_YAML = 'datasets/yolo_harmony_v2_phase8_final/harmony_phase8_final.yaml'
    PROJECT = 'harmony_omr_v2_experiments'
    NAME = 'exp6_finetune_head'

    print("=" * 70)
    print("實驗 6: 分階段微調 - 凍結 Backbone")
    print("=" * 70)
    print(f"開始時間: {datetime.now()}")
    print(f"基礎模型: {BASE_MODEL}")
    print()

    # 載入最佳模型
    model = YOLO(BASE_MODEL)

    # 獲取模型結構信息
    print("模型層數信息:")
    print(f"  總層數: {len(list(model.model.modules()))}")

    # 訓練配置
    train_args = {
        'data': DATA_YAML,
        'epochs': 30,
        'imgsz': 1280,
        'batch': 8,
        'freeze': 10,           # 凍結前 10 層 (backbone)
        'lr0': 0.0001,          # 小學習率
        'lrf': 0.01,            # 最終學習率比例
        'optimizer': 'AdamW',
        'cos_lr': True,         # 餘弦學習率
        'amp': False,           # 禁用 AMP (避免 nan)
        'patience': 10,         # 早停
        'project': PROJECT,
        'name': NAME,
        'exist_ok': True,
        'verbose': True,
        'val': True,
    }

    print("\n訓練配置:")
    for k, v in train_args.items():
        print(f"  {k}: {v}")
    print()

    # 開始訓練
    print("開始訓練...")
    results = model.train(**train_args)

    # 輸出結果
    print("\n" + "=" * 70)
    print("訓練完成！結果摘要")
    print("=" * 70)

    # 驗證最佳模型
    best_model_path = f'{PROJECT}/{NAME}/weights/best.pt'
    if os.path.exists(best_model_path):
        print(f"\n驗證最佳模型: {best_model_path}")
        best_model = YOLO(best_model_path)
        val_results = best_model.val(
            data=DATA_YAML,
            imgsz=1280,
            batch=8,
            verbose=True
        )

        print(f"\n最終指標:")
        print(f"  mAP50:     {val_results.box.map50:.4f}")
        print(f"  mAP50-95:  {val_results.box.map:.4f}")
        print(f"  Precision: {val_results.box.mp:.4f}")
        print(f"  Recall:    {val_results.box.mr:.4f}")

        # 檢查弱類別改善情況
        weak_classes = {'tie': 8, 'barline_final': 25, 'barline_double': 24}
        print(f"\n弱類別改善:")
        for name, idx in weak_classes.items():
            ap50 = val_results.box.ap50[idx]
            print(f"  {name}: mAP50={ap50:.4f}")

    print(f"\n完成時間: {datetime.now()}")

if __name__ == '__main__':
    main()
