#!/usr/bin/env python3
"""
Phase 10.1 訓練 - RTX 5090 優化版本 v2
優化: batch=28（避免 OOM），確保穩定後台運行

變更記錄:
- batch: 39 → 28 (降低 28%，避免 CUDA OOM)
- 其他配置與 Phase 8 完全一致
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

import torch
import torch.backends.cudnn as cudnn
from ultralytics import YOLO
import json
from datetime import datetime
from pathlib import Path

# ===== CUDA 深度優化配置 =====
print("🚀 啟用 GPU CUDA 深度優化...")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
cudnn.benchmark = True
cudnn.deterministic = False
torch.cuda.empty_cache()

device = torch.device('cuda:0')
gpu_name = torch.cuda.get_device_name(0)
gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
compute_cap = torch.cuda.get_device_capability(0)

print(f"\n📊 硬體資訊:")
print(f"   GPU: {gpu_name}")
print(f"   記憶體: {gpu_memory:.1f} GB")
print(f"   Compute Capability: {compute_cap[0]}.{compute_cap[1]}")
print(f"   PyTorch: {torch.__version__}")
print(f"   CUDA: {torch.version.cuda}")
print(f"   cuDNN: {cudnn.version()}")

# ===== 訓練配置（優化版）=====
CONFIG = {
    # 基礎配置
    'model': '/home/thc1006/dev/music-app/training/harmony_omr_v2_phase8/phase8_training/weights/best.pt',
    'data': '/home/thc1006/dev/music-app/training/datasets/yolo_harmony_v2_phase10_1/phase10_1.yaml',

    # 訓練參數（Phase 8 成功配置）
    'epochs': 150,
    'batch': 28,  # 從 39 降到 28，避免 OOM
    'imgsz': 640,

    # 優化器（Phase 8 成功配置）
    'optimizer': 'AdamW',
    'lr0': 0.001,
    'lrf': 0.01,
    'momentum': 0.937,
    'weight_decay': 0.0005,

    # 學習率調度
    'warmup_epochs': 3,
    'warmup_momentum': 0.8,
    'warmup_bias_lr': 0.1,
    'cos_lr': True,

    # 損失函數權重（Phase 8 成功配置）
    'cls': 0.5,
    'box': 7.5,
    'dfl': 1.5,

    # 數據增強（Phase 8 成功配置）
    'hsv_h': 0.015,
    'hsv_s': 0.7,
    'hsv_v': 0.4,
    'degrees': 0.0,
    'translate': 0.1,
    'scale': 0.5,
    'shear': 0.0,
    'perspective': 0.0,
    'flipud': 0.0,
    'fliplr': 0.5,
    'mosaic': 0.5,
    'mixup': 0.1,
    'copy_paste': 0.0,
    'erasing': 0.0,

    # GPU/CUDA 優化配置
    'device': 0,
    'workers': 20,
    'amp': True,
    'half': False,
    'cache': False,
    'rect': False,
    'resume': False,
    'exist_ok': True,
    'pretrained': True,
    'verbose': True,
    'seed': 42,
    'deterministic': False,
    'single_cls': False,
    'image_weights': False,
    'multi_scale': False,

    # 保存配置
    'save': True,
    'save_period': 10,
    'patience': 50,
    'plots': True,
    'val': True,

    # 專案路徑
    'project': '/home/thc1006/dev/music-app/training/harmony_omr_v2_phase10_1',
    'name': 'phase10_1_optimized_v2',
}

def main():
    print("\n" + "="*60)
    print("  Phase 10.1 訓練開始（優化版 v2 - batch=28）")
    print("="*60)

    print(f"\n📋 訓練配置:")
    print(f"   數據集: Phase 8 (32,555) + DeepScores (855) = 33,410 張")
    print(f"   Epochs: {CONFIG['epochs']}")
    print(f"   Batch Size: {CONFIG['batch']} ← 優化（從 39 降低）")
    print(f"   Workers: {CONFIG['workers']}")
    print(f"   學習率: {CONFIG['lr0']} → {CONFIG['lr0'] * CONFIG['lrf']}")
    print(f"   優化器: {CONFIG['optimizer']}")
    print(f"   混合精度: BF16 (AMP)")
    print(f"   TF32: 啟用")
    print(f"   預期 OOM 率: < 2% ← 大幅降低")

    # 載入模型
    print(f"\n📦 載入 Phase 8 最佳模型...")
    model = YOLO(CONFIG['model'])

    # 開始訓練
    print(f"\n🚀 開始訓練...")
    print(f"   預計時間: 6-8 小時（優化後）")
    print(f"   GPU 使用率: 95-100%（穩定）")
    print(f"\n" + "="*60)

    results = model.train(
        data=CONFIG['data'],
        epochs=CONFIG['epochs'],
        batch=CONFIG['batch'],
        imgsz=CONFIG['imgsz'],

        # 優化器
        optimizer=CONFIG['optimizer'],
        lr0=CONFIG['lr0'],
        lrf=CONFIG['lrf'],
        momentum=CONFIG['momentum'],
        weight_decay=CONFIG['weight_decay'],

        # 學習率調度
        warmup_epochs=CONFIG['warmup_epochs'],
        warmup_momentum=CONFIG['warmup_momentum'],
        warmup_bias_lr=CONFIG['warmup_bias_lr'],
        cos_lr=CONFIG['cos_lr'],

        # 損失函數
        cls=CONFIG['cls'],
        box=CONFIG['box'],
        dfl=CONFIG['dfl'],

        # 數據增強
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
        mosaic=CONFIG['mosaic'],
        mixup=CONFIG['mixup'],
        copy_paste=CONFIG['copy_paste'],
        erasing=CONFIG['erasing'],

        # GPU 優化
        device=CONFIG['device'],
        workers=CONFIG['workers'],
        amp=CONFIG['amp'],
        cache=CONFIG['cache'],

        # 其他
        patience=CONFIG['patience'],
        save=CONFIG['save'],
        save_period=CONFIG['save_period'],
        plots=CONFIG['plots'],
        val=CONFIG['val'],

        # 專案設置
        project=CONFIG['project'],
        name=CONFIG['name'],
        exist_ok=CONFIG['exist_ok'],
        pretrained=CONFIG['pretrained'],
        verbose=CONFIG['verbose'],
        seed=CONFIG['seed'],
        deterministic=CONFIG['deterministic'],
    )

    # 訓練完成報告
    print(f"\n" + "="*60)
    print(f"  ✅ Phase 10.1 訓練完成！")
    print(f"="*60)

    # 驗證最佳模型
    best_model_path = Path(CONFIG['project']) / CONFIG['name'] / 'weights' / 'best.pt'

    if best_model_path.exists():
        print(f"\n📊 驗證最佳模型...")
        best_model = YOLO(best_model_path)
        metrics = best_model.val()

        # 保存訓練報告
        training_report = {
            'phase': 'Phase 10.1 Optimized v2',
            'completion_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'hardware': {
                'gpu': gpu_name,
                'gpu_memory_gb': gpu_memory,
                'compute_capability': f"{compute_cap[0]}.{compute_cap[1]}",
                'cuda_version': torch.version.cuda,
                'pytorch_version': torch.__version__,
            },
            'dataset': {
                'train_images': 33410,
                'val_images': 3617,
                'sources': ['Phase 8 Final', 'DeepScores Dynamics']
            },
            'config': CONFIG,
            'results': {
                'mAP50-95': float(metrics.box.map),
                'mAP50': float(metrics.box.map50),
                'mAP75': float(metrics.box.map75),
            }
        }

        report_path = Path(CONFIG['project']) / CONFIG['name'] / 'training_report.json'
        with open(report_path, 'w') as f:
            json.dump(training_report, f, indent=2)

        print(f"\n📈 最終結果:")
        print(f"   mAP50-95: {metrics.box.map:.4f}")
        print(f"   mAP50: {metrics.box.map50:.4f}")
        print(f"   mAP75: {metrics.box.map75:.4f}")
        print(f"\n📁 模型位置: {best_model_path}")
        print(f"📄 訓練報告: {report_path}")

        # 與 Phase 8 比較
        phase8_map = 0.5809
        improvement = (metrics.box.map - phase8_map) / phase8_map * 100

        print(f"\n📊 與 Phase 8 比較:")
        print(f"   Phase 8 mAP50-95: {phase8_map:.4f}")
        print(f"   Phase 10.1 mAP50-95: {metrics.box.map:.4f}")
        print(f"   改進: {improvement:+.2f}%")

        if metrics.box.map > phase8_map:
            print(f"\n🎉 成功！Phase 10.1 超越 Phase 8！")
        elif metrics.box.map > phase8_map * 0.98:
            print(f"\n✅ 穩定！維持 Phase 8 水準（-2% 以內）")
        else:
            print(f"\n⚠️  退化！需要分析原因")
    else:
        print(f"\n❌ 找不到最佳模型: {best_model_path}")

if __name__ == "__main__":
    main()
