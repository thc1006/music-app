#!/usr/bin/env python3
"""
Phase 9 訓練腳本 (修正版)
基於合併後的完整數據集進行訓練

修正說明:
- 原計劃: 清理 tiny bbox → 訓練
- 修正後: 合併未使用數據 → 訓練 (不做 tiny bbox 清理)

原因:
- Tiny bbox 與 mAP 相關係數僅 -0.143 (非常弱)
- barline_double 只有 0.2% tiny bbox 但 mAP 最差 (0.231)
- 發現 8,726 張未使用圖片，包含 +3,724 barline_double 標註

執行方式:
    python yolo12_train_phase9.py
"""

from ultralytics import YOLO
from pathlib import Path
import torch
import json
from datetime import datetime

# 配置
CONFIG = {
    # 數據集 (修正: 使用合併數據集而非清理數據集)
    'data': '/home/thc1006/dev/music-app/training/datasets/yolo_harmony_v2_phase9_merged/harmony_phase9_merged.yaml',

    # 預訓練權重（從 Phase 8 繼續）
    'pretrained': '/home/thc1006/dev/music-app/training/harmony_omr_v2_phase8/phase8_training/weights/best.pt',

    # 輸出目錄
    'project': '/home/thc1006/dev/music-app/training/harmony_omr_v2_phase9',
    'name': 'merged_data_training',

    # 訓練參數
    'epochs': 100,
    'batch': 24,
    'imgsz': 640,
    'patience': 30,

    # 優化器配置
    'optimizer': 'AdamW',
    'lr0': 0.0005,  # 較低的學習率（fine-tuning）
    'lrf': 0.01,
    'weight_decay': 0.0005,
    'warmup_epochs': 2,

    # 損失權重（針對瓶頸類別）
    'box': 7.5,
    'cls': 0.8,  # 提升分類損失權重
    'dfl': 1.5,

    # 數據增強
    'mosaic': 0.5,
    'mixup': 0.1,
    'copy_paste': 0.0,
    'hsv_h': 0.015,
    'hsv_s': 0.5,
    'hsv_v': 0.3,
    'degrees': 5.0,
    'translate': 0.1,
    'scale': 0.3,
    'shear': 2.0,
    'perspective': 0.0001,
    'flipud': 0.0,
    'fliplr': 0.0,
    'erasing': 0.4,

    # 其他設置
    'workers': 16,
    'cache': 'ram',
    'device': '0',
    'amp': True,
    'close_mosaic': 15,
    'save_period': 10,
    'plots': True,
    'verbose': True,
    'deterministic': True,
    'seed': 42,
}


def check_prerequisites():
    """檢查前置條件"""
    print("檢查前置條件...")

    # 檢查數據集
    data_path = Path(CONFIG['data'])
    if not data_path.exists():
        print(f"❌ 數據集配置不存在: {data_path}")
        print("   請先執行: python scripts/merge_phase9_datasets.py")
        return False
    print(f"✅ 數據集配置: {data_path}")

    # 檢查預訓練權重
    weights_path = Path(CONFIG['pretrained'])
    if not weights_path.exists():
        print(f"❌ 預訓練權重不存在: {weights_path}")
        return False
    print(f"✅ 預訓練權重: {weights_path}")

    # 檢查 GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✅ GPU: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        print("⚠️ GPU 不可用，將使用 CPU 訓練（非常慢）")

    return True


def train():
    """執行訓練"""
    print()
    print("=" * 60)
    print("Phase 9 訓練")
    print("=" * 60)
    print(f"開始時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # 載入模型
    print("載入預訓練模型...")
    model = YOLO(CONFIG['pretrained'])

    # 開始訓練
    print("開始訓練...")
    results = model.train(
        data=CONFIG['data'],
        epochs=CONFIG['epochs'],
        batch=CONFIG['batch'],
        imgsz=CONFIG['imgsz'],
        patience=CONFIG['patience'],

        # 輸出設置
        project=CONFIG['project'],
        name=CONFIG['name'],
        exist_ok=True,

        # 優化器
        optimizer=CONFIG['optimizer'],
        lr0=CONFIG['lr0'],
        lrf=CONFIG['lrf'],
        weight_decay=CONFIG['weight_decay'],
        warmup_epochs=CONFIG['warmup_epochs'],

        # 損失權重
        box=CONFIG['box'],
        cls=CONFIG['cls'],
        dfl=CONFIG['dfl'],

        # 數據增強
        mosaic=CONFIG['mosaic'],
        mixup=CONFIG['mixup'],
        copy_paste=CONFIG['copy_paste'],
        hsv_h=CONFIG['hsv_h'],
        hsv_s=CONFIG['hsv_s'],
        hsv_v=CONFIG['hsv_v'],
        degrees=CONFIG['degrees'],
        translate=CONFIG['translate'],
        scale=CONFIG['scale'],
        shear=CONFIG['shear'],
        perspective=CONFIG['perspective'],
        flipud=CONFIG['flipud'],
        fliplr=CONFIG['fliplr'],
        erasing=CONFIG['erasing'],

        # 其他設置
        workers=CONFIG['workers'],
        cache=CONFIG['cache'],
        device=CONFIG['device'],
        amp=CONFIG['amp'],
        close_mosaic=CONFIG['close_mosaic'],
        save_period=CONFIG['save_period'],
        plots=CONFIG['plots'],
        verbose=CONFIG['verbose'],
        deterministic=CONFIG['deterministic'],
        seed=CONFIG['seed'],
    )

    print()
    print("=" * 60)
    print("訓練完成！")
    print("=" * 60)
    print(f"結束時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # 保存訓練報告
    report = {
        'timestamp': datetime.now().isoformat(),
        'config': CONFIG,
        'results': {
            'best_mAP50': float(results.box.map50) if hasattr(results, 'box') else None,
            'best_mAP50_95': float(results.box.map) if hasattr(results, 'box') else None,
        }
    }

    report_path = Path(CONFIG['project']) / CONFIG['name'] / 'training_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"訓練報告已保存: {report_path}")

    # 顯示結果
    if hasattr(results, 'box'):
        print()
        print("最終指標:")
        print(f"  mAP50:    {results.box.map50:.4f}")
        print(f"  mAP50-95: {results.box.map:.4f}")

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Phase 9 OMR 模型訓練')
    parser.add_argument('--auto', '-y', action='store_true',
                        help='自動執行，不需要確認')
    args = parser.parse_args()

    print()
    print("=" * 60)
    print("Phase 9 OMR 模型訓練")
    print("=" * 60)
    print()

    # 檢查前置條件
    if not check_prerequisites():
        print()
        print("前置條件檢查失敗，請解決上述問題後重試。")
        return

    # 確認開始訓練
    print()
    print("配置摘要:")
    print(f"  - 數據集: {CONFIG['data']}")
    print(f"  - 預訓練: {CONFIG['pretrained']}")
    print(f"  - Epochs: {CONFIG['epochs']}")
    print(f"  - Batch:  {CONFIG['batch']}")
    print(f"  - ImgSz:  {CONFIG['imgsz']}")
    print()

    if not args.auto:
        response = input("是否開始訓練？(y/n): ").strip().lower()
        if response != 'y':
            print("訓練已取消。")
            return

    # 執行訓練
    train()


if __name__ == "__main__":
    main()
