#!/usr/bin/env python3
"""
實驗 2: 分類損失權重調整
目標: 增加分類損失權重，讓模型更注重區分相似類別 (tie/barline_final)

預計時間: ~8 小時
"""

import os
from datetime import datetime
from ultralytics import YOLO

def main():
    # 配置
    BASE_MODEL = 'harmony_omr_v2_ultimate_v5_stable/stable_1280_resumed/weights/best.pt'
    DATA_YAML = 'datasets/yolo_harmony_v2_phase8_final/harmony_phase8_final.yaml'
    PROJECT = 'harmony_omr_v2_experiments'
    NAME = 'exp2_cls_weight_1.0'

    print("=" * 70)
    print("實驗 2: 分類損失權重調整 (cls=1.0)")
    print("=" * 70)
    print(f"開始時間: {datetime.now()}")
    print(f"基礎模型: {BASE_MODEL}")
    print()

    # 載入最佳模型
    model = YOLO(BASE_MODEL)

    # 訓練配置
    train_args = {
        'data': DATA_YAML,
        'epochs': 100,
        'imgsz': 1280,
        'batch': 4,
        # 注意: 新版 Ultralytics 沒有 accumulate 參數
        'cls': 1.0,             # 分類損失權重 (預設 0.5, 增加到 1.0)
        'box': 7.5,             # 定位損失權重 (保持預設)
        'dfl': 1.5,             # DFL 損失權重 (保持預設)
        'lr0': 0.001,           # 初始學習率
        'lrf': 0.01,
        'optimizer': 'SGD',
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'cos_lr': True,
        'amp': False,           # 禁用 AMP
        'patience': 20,
        'project': PROJECT,
        'name': NAME,
        'exist_ok': True,
        'verbose': True,
        'val': True,
        'plots': True,
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
    print("訓練完成！")
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
        weak_classes = {'tie': 8, 'barline_final': 25, 'barline_double': 24, 'ledger_line': 32}
        print(f"\n弱類別改善:")
        for name, idx in weak_classes.items():
            ap50 = val_results.box.ap50[idx]
            print(f"  {name}: mAP50={ap50:.4f}")

    print(f"\n完成時間: {datetime.now()}")

if __name__ == '__main__':
    main()
