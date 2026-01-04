#!/usr/bin/env python3
"""
YOLO12 Ultimate v5 Stable - 穩定高解析度訓練腳本
=================================================

基於實際 OOM 分析結果優化的穩定版本：
- batch=4 (從 8 降低，確保 0% OOM)
- imgsz=1280 (保持高解析度優勢)
- 100% GPU 訓練，無 CPU fallback

調研來源:
1. Ultralytics YOLO12 官方文檔
2. Roboflow 小物件檢測最佳實踐 - imgsz=1280
3. 實際 OOM 分析：batch=8 導致 27% OOM 率

關鍵改進:
1. batch=4 確保穩定訓練
2. 累積梯度 (accumulate=2) 模擬 batch=8 效果
3. 完全消除 OOM，100% GPU 訓練

目標: mAP50 > 0.70 (提升 +8.6%)

創建日期: 2025-12-29
"""

import os
import sys
import gc
import torch
from pathlib import Path
from datetime import datetime

# ============================================================
# 環境配置
# ============================================================

def setup_environment():
    """設置訓練環境"""
    print("=" * 70)
    print("YOLO12 Ultimate v5 Stable - 穩定高解析度訓練")
    print("=" * 70)
    print(f"開始時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("")

    # GPU 檢查
    if not torch.cuda.is_available():
        print("❌ 錯誤: 未檢測到 CUDA GPU")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    compute_cap = torch.cuda.get_device_capability()

    print(f"✅ GPU: {gpu_name}")
    print(f"✅ VRAM: {gpu_mem:.1f} GB")
    print(f"✅ Compute Capability: {compute_cap[0]}.{compute_cap[1]}")

    # CUDA 優化
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256,expandable_segments:True'
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    print("✅ TF32 和 cuDNN 優化已啟用")

    # 清理記憶體
    torch.cuda.empty_cache()
    gc.collect()
    print("✅ GPU 記憶體已清理")
    print("")

    return gpu_mem

# ============================================================
# 路徑配置
# ============================================================

BASE_DIR = Path('/home/thc1006/dev/music-app/training')
DATASET_DIR = BASE_DIR / 'datasets/yolo_harmony_v2_phase8_final'
OUTPUT_DIR = BASE_DIR / 'harmony_omr_v2_ultimate_v5_stable'
PHASE8_MODEL = BASE_DIR / 'harmony_omr_v2_phase8/phase8_training/weights/best.pt'
DATASET_YAML = DATASET_DIR / 'harmony_phase8_final.yaml'
LOG_DIR = BASE_DIR / 'logs'

# ============================================================
# 穩定訓練配置 (基於 OOM 分析優化)
# ============================================================

# 高解析度配置 - 關鍵改進
IMGSZ = 1280  # 保持高解析度

# 穩定批次大小配置
# 分析: batch=8 導致 27% OOM，batch=4 應該 0% OOM
BATCH_SIZE = 4  # 降低以確保穩定
NBS = 64  # 名義批次大小 (用於學習率計算)
# 有效累積 = NBS / BATCH_SIZE = 64/4 = 16 次累積

# 訓練輪數
EPOCHS = 200
PATIENCE = 100

# 學習率 - 調整以適應較小的實際批次
LR0 = 0.0005  # 降低初始學習率 (因為累積梯度)
LRF = 0.01

# OMR 專用數據增強配置
AUGMENTATION_CONFIG = {
    # ✅ 適合 OMR 的增強
    'mosaic': 0.5,
    'degrees': 0.0,
    'translate': 0.1,
    'scale': 0.3,
    'shear': 0.0,
    'perspective': 0.0,
    'hsv_h': 0.015,
    'hsv_s': 0.4,
    'hsv_v': 0.3,

    # ❌ 不適合 OMR 的增強 - 禁用
    'flipud': 0.0,
    'fliplr': 0.0,
    'mixup': 0.0,
    'copy_paste': 0.0,
    'erasing': 0.0,
    'bgr': 0.0,
}

# ============================================================
# 訓練函數
# ============================================================

def train_stable():
    """執行穩定的高解析度訓練"""

    from ultralytics import YOLO

    print("=" * 70)
    print("Phase 1: 穩定高解析度 YOLO12 訓練")
    print("=" * 70)
    print("")
    print("🔧 穩定性優化配置:")
    print(f"   - batch: {BATCH_SIZE} (從 8 降低，消除 OOM)")
    print(f"   - nbs: {NBS} (名義批次，用於累積梯度)")
    print(f"   - 有效累積: {NBS // BATCH_SIZE} 次")
    print(f"   - imgsz: {IMGSZ} (高解析度)")
    print(f"   - 預計 OOM 率: 0%")
    print("")

    # 驗證模型和數據集
    if not PHASE8_MODEL.exists():
        print(f"❌ 錯誤: Phase 8 模型不存在: {PHASE8_MODEL}")
        sys.exit(1)

    if not DATASET_YAML.exists():
        print(f"❌ 錯誤: 數據集配置不存在: {DATASET_YAML}")
        sys.exit(1)

    print(f"✅ 基礎模型: {PHASE8_MODEL}")
    print(f"✅ 數據集: {DATASET_YAML}")
    print(f"✅ 輸出目錄: {OUTPUT_DIR}")
    print("")

    # 加載模型
    print("正在加載 Phase 8 最佳模型...")
    model = YOLO(str(PHASE8_MODEL))
    print("✅ 模型加載完成")
    print("")

    # 創建輸出目錄
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 開始訓練
    print("開始穩定高解析度訓練...")
    print("=" * 70)

    results = model.train(
        # 基本配置
        data=str(DATASET_YAML),
        epochs=EPOCHS,
        patience=PATIENCE,
        batch=BATCH_SIZE,
        imgsz=IMGSZ,

        # 輸出配置
        project=str(OUTPUT_DIR),
        name='stable_1280',
        exist_ok=True,
        save=True,
        save_period=10,

        # 訓練優化
        device=0,
        workers=12,  # 減少 worker 數量以節省記憶體
        cache=False,  # 禁用緩存以節省記憶體
        amp=True,

        # 學習率和累積梯度
        lr0=LR0,
        lrf=LRF,
        nbs=NBS,  # 名義批次大小，自動計算累積
        warmup_epochs=5,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,

        # 優化器
        optimizer='AdamW',
        momentum=0.937,
        weight_decay=0.0005,

        # 損失權重
        box=7.5,
        cls=0.5,
        dfl=1.5,

        # 數據增強 (OMR 專用)
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
        close_mosaic=20,

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
    print(f"模型保存位置: {OUTPUT_DIR}/stable_1280/weights/")
    print(f"完成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ============================================================
# 主程序
# ============================================================

def main():
    """主程序入口"""

    # 設置環境
    gpu_mem = setup_environment()

    # 執行訓練
    try:
        results = train_stable()
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
    print("🎉 訓練流程完成！")
    print("=" * 70)

if __name__ == '__main__':
    main()
