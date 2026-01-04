#!/usr/bin/env python3
"""
恢復 Ultimate v5 Stable 訓練 - 從 epoch 90 繼續
================================================

訓練在 epoch 97 崩潰 (出現 nan)，從 epoch 90 恢復訓練。

關鍵改進 (避免再次崩潰):
1. 降低學習率 lr0: 0.0005 -> 0.0002
2. 啟用梯度裁剪 (clip=10.0)
3. 禁用 AMP 混合精度 (更穩定的數值計算)
4. 增加 warmup epochs

創建日期: 2026-01-05
"""

import os
import sys
import gc
import torch
from pathlib import Path
from datetime import datetime

# ============================================================
# 路徑配置
# ============================================================

BASE_DIR = Path('/home/thc1006/dev/music-app/training')
DATASET_DIR = BASE_DIR / 'datasets/yolo_harmony_v2_phase8_final'
OUTPUT_DIR = BASE_DIR / 'harmony_omr_v2_ultimate_v5_stable'
RESUME_MODEL = OUTPUT_DIR / 'stable_1280/weights/epoch90.pt'
DATASET_YAML = DATASET_DIR / 'harmony_phase8_final.yaml'
LOG_DIR = BASE_DIR / 'logs'

# ============================================================
# 恢復訓練配置 (針對穩定性優化)
# ============================================================

# 保持高解析度
IMGSZ = 1280

# 批次配置 (保持不變)
BATCH_SIZE = 4
NBS = 64

# 訓練輪數 - 從 epoch 90 繼續到 200
EPOCHS = 200
PATIENCE = 50  # 降低 patience，更快停止

# 學習率 - 大幅降低以增加穩定性
LR0 = 0.0002  # 從 0.0005 降低到 0.0002
LRF = 0.01

# OMR 專用數據增強
AUGMENTATION_CONFIG = {
    'mosaic': 0.5,
    'degrees': 0.0,
    'translate': 0.1,
    'scale': 0.3,
    'shear': 0.0,
    'perspective': 0.0,
    'hsv_h': 0.015,
    'hsv_s': 0.4,
    'hsv_v': 0.3,
    'flipud': 0.0,
    'fliplr': 0.0,
    'mixup': 0.0,
    'copy_paste': 0.0,
    'erasing': 0.0,
    'bgr': 0.0,
}

# ============================================================
# 環境設置
# ============================================================

def setup_environment():
    """設置訓練環境"""
    print("=" * 70)
    print("恢復 Ultimate v5 Stable 訓練 - 從 epoch 90")
    print("=" * 70)
    print(f"開始時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("")

    if not torch.cuda.is_available():
        print("❌ 錯誤: 未檢測到 CUDA GPU")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3

    print(f"✅ GPU: {gpu_name}")
    print(f"✅ VRAM: {gpu_mem:.1f} GB")

    # CUDA 優化
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256,expandable_segments:True'
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    # 清理記憶體
    torch.cuda.empty_cache()
    gc.collect()
    print("✅ GPU 記憶體已清理")
    print("")

    return gpu_mem

# ============================================================
# 恢復訓練
# ============================================================

def resume_training():
    """從 epoch 90 恢復訓練"""

    from ultralytics import YOLO

    print("=" * 70)
    print("恢復訓練配置")
    print("=" * 70)
    print("")
    print("🔧 穩定性優化 (避免 nan 崩潰):")
    print(f"   - 學習率 lr0: 0.0002 (從 0.0005 降低)")
    print(f"   - AMP: 禁用 (更穩定的數值計算)")
    print(f"   - 梯度裁剪: 啟用")
    print(f"   - batch: {BATCH_SIZE}")
    print(f"   - imgsz: {IMGSZ}")
    print("")

    # 驗證文件
    if not RESUME_MODEL.exists():
        print(f"❌ 錯誤: 恢復模型不存在: {RESUME_MODEL}")
        sys.exit(1)

    if not DATASET_YAML.exists():
        print(f"❌ 錯誤: 數據集配置不存在: {DATASET_YAML}")
        sys.exit(1)

    print(f"✅ 恢復模型: {RESUME_MODEL}")
    print(f"✅ 數據集: {DATASET_YAML}")
    print("")

    # 加載模型
    print("正在加載 epoch 90 模型...")
    model = YOLO(str(RESUME_MODEL))
    print("✅ 模型加載完成")
    print("")

    # 開始恢復訓練
    print("開始恢復訓練...")
    print("=" * 70)

    results = model.train(
        # 基本配置
        data=str(DATASET_YAML),
        epochs=EPOCHS,
        patience=PATIENCE,
        batch=BATCH_SIZE,
        imgsz=IMGSZ,

        # 輸出配置 - 使用新目錄避免覆蓋
        project=str(OUTPUT_DIR),
        name='stable_1280_resumed',
        exist_ok=True,
        save=True,
        save_period=5,  # 更頻繁保存

        # 訓練優化
        device=0,
        workers=12,
        cache=False,
        amp=False,  # ⚠️ 禁用 AMP 以增加穩定性

        # 學習率 - 降低
        lr0=LR0,
        lrf=LRF,
        nbs=NBS,
        warmup_epochs=3,  # 添加 warmup
        warmup_momentum=0.8,
        warmup_bias_lr=0.05,

        # 優化器
        optimizer='AdamW',
        momentum=0.937,
        weight_decay=0.0005,

        # 損失權重
        box=7.5,
        cls=0.5,
        dfl=1.5,

        # 數據增強
        mosaic=AUGMENTATION_CONFIG['mosaic'],
        degrees=AUGMENTATION_CONFIG['degrees'],
        translate=AUGMENTATION_CONFIG['translate'],
        scale=AUGMENTATION_CONFIG['scale'],
        shear=AUGMENTATION_CONFIG['shear'],
        perspective=AUGMENTATION_CONFIG['perspective'],
        hsv_h=AUGMENTATION_CONFIG['hsv_h'],
        hsv_s=AUGMENTATION_CONFIG['hsv_s'],
        hsv_v=AUGMENTATION_CONFIG['hsv_v'],
        flipud=AUGMENTATION_CONFIG['flipud'],
        fliplr=AUGMENTATION_CONFIG['fliplr'],
        mixup=AUGMENTATION_CONFIG['mixup'],
        copy_paste=AUGMENTATION_CONFIG['copy_paste'],
        erasing=AUGMENTATION_CONFIG['erasing'],
        bgr=AUGMENTATION_CONFIG['bgr'],

        # Mosaic 關閉點
        close_mosaic=10,

        # 其他
        deterministic=True,
        seed=42,
        verbose=True,
        plots=True,
    )

    return results

def print_results_summary(results):
    """打印結果摘要"""
    print("")
    print("=" * 70)
    print("訓練完成！結果摘要")
    print("=" * 70)

    if hasattr(results, 'results_dict'):
        rd = results.results_dict
        print(f"mAP50:      {rd.get('metrics/mAP50(B)', 'N/A')}")
        print(f"mAP50-95:   {rd.get('metrics/mAP50-95(B)', 'N/A')}")
        print(f"Precision:  {rd.get('metrics/precision(B)', 'N/A')}")
        print(f"Recall:     {rd.get('metrics/recall(B)', 'N/A')}")

    print("")
    print(f"模型保存位置: {OUTPUT_DIR}/stable_1280_resumed/weights/")
    print(f"完成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ============================================================
# 主程序
# ============================================================

def main():
    """主程序入口"""

    setup_environment()

    try:
        results = resume_training()
        print_results_summary(results)

    except KeyboardInterrupt:
        print("\n⚠️ 訓練被用戶中斷")
        sys.exit(0)

    except Exception as e:
        print(f"\n❌ 訓練錯誤: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("")
    print("🎉 恢復訓練完成！")
    print("=" * 70)

if __name__ == '__main__':
    main()
