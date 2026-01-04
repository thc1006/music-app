#!/usr/bin/env python3
"""
YOLO12 Ultimate v5 - 高解析度訓練腳本
=====================================

基於 2024-2025 最新研究調研結果設計的最佳訓練策略

調研來源:
1. Ultralytics YOLO12 官方文檔 - 注意力機制架構
2. Roboflow 小物件檢測最佳實踐 - imgsz=1280 推薦
3. Nature Scientific Reports 2024 - SOD-YOLO P2層研究
4. SAHI 論文 - 切片輔助推論提升 12-14% AP

關鍵策略:
1. 高解析度訓練 (imgsz=1280) - 小物件檢測提升關鍵
2. 從 Phase 8 best.pt 繼續訓練 (mAP50=0.6444)
3. OMR 專用數據增強 (禁用翻轉、混合等不適用增強)
4. 更長訓練時間 (200 epochs) + 更耐心的早停 (patience=100)
5. 放棄 DINOv3 蒸餾 (已證明無效，反而降低 2.8%)

目標: mAP50 > 0.70 (提升 +8.6%)

創建日期: 2025-12-29
"""

import os
import sys
import gc
import torch
import subprocess
from pathlib import Path
from datetime import datetime

# ============================================================
# 環境配置
# ============================================================

def setup_environment():
    """設置訓練環境"""
    print("=" * 70)
    print("YOLO12 Ultimate v5 - 高解析度 OMR 訓練")
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
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,expandable_segments:True'
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
OUTPUT_DIR = BASE_DIR / 'harmony_omr_v2_ultimate_v5'
PHASE8_MODEL = BASE_DIR / 'harmony_omr_v2_phase8/phase8_training/weights/best.pt'
DATASET_YAML = DATASET_DIR / 'harmony_phase8_final.yaml'
LOG_DIR = BASE_DIR / 'logs'

# ============================================================
# 訓練配置 (基於調研結果)
# ============================================================

# 高解析度配置 - 關鍵改進
IMGSZ = 1280  # 從 640 提升到 1280，小物件檢測關鍵

# 批次大小 - 根據 1280 解析度調整
# RTX 5090 32GB: imgsz=1280 約需 batch=8-12
BATCH_SIZE = 8

# 訓練輪數 - 更長訓練
EPOCHS = 200
PATIENCE = 100  # 更耐心的早停

# 學習率 - 使用 Phase 8 成功配置
LR0 = 0.001
LRF = 0.01  # 最終學習率比例

# OMR 專用數據增強配置 (基於樂譜特性)
AUGMENTATION_CONFIG = {
    # ✅ 適合 OMR 的增強
    'mosaic': 0.5,        # 馬賽克增強，增加多樣性
    'degrees': 0.0,       # 禁用旋轉（樂譜方向固定）
    'translate': 0.1,     # 輕微平移
    'scale': 0.3,         # 縮放
    'shear': 0.0,         # 禁用剪切
    'perspective': 0.0,   # 禁用透視變換
    'hsv_h': 0.015,       # 輕微色調變化
    'hsv_s': 0.4,         # 飽和度變化
    'hsv_v': 0.3,         # 亮度變化

    # ❌ 不適合 OMR 的增強 - 禁用
    'flipud': 0.0,        # 禁用上下翻轉（樂譜不應翻轉）
    'fliplr': 0.0,        # 禁用左右翻轉（樂譜閱讀順序固定）
    'mixup': 0.0,         # 禁用混合（樂譜符號不應混合）
    'copy_paste': 0.0,    # 禁用複製粘貼（符號位置關係重要）
    'erasing': 0.0,       # 禁用隨機擦除
    'bgr': 0.0,           # 禁用 BGR 轉換
}

# ============================================================
# 訓練函數
# ============================================================

def train_high_resolution():
    """執行高解析度訓練"""

    from ultralytics import YOLO

    print("=" * 70)
    print("Phase 1: 高解析度 YOLO12 訓練 (imgsz=1280)")
    print("=" * 70)

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

    # 訓練配置
    print("訓練配置:")
    print(f"  - imgsz: {IMGSZ} (高解析度)")
    print(f"  - batch: {BATCH_SIZE}")
    print(f"  - epochs: {EPOCHS}")
    print(f"  - patience: {PATIENCE}")
    print(f"  - lr0: {LR0}")
    print("")

    # 創建輸出目錄
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 開始訓練
    print("開始高解析度訓練...")
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
        name='highres_1280',
        exist_ok=True,
        save=True,
        save_period=10,  # 每 10 epochs 保存一次

        # 訓練優化
        device=0,
        workers=16,  # 多核心數據加載
        cache='disk',  # 使用磁盤緩存（節省 RAM）
        amp=True,  # 混合精度訓練

        # 學習率
        lr0=LR0,
        lrf=LRF,
        warmup_epochs=5,  # 更長的預熱
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,

        # 優化器
        optimizer='AdamW',
        momentum=0.937,
        weight_decay=0.0005,

        # 損失權重 (Phase 8 成功配置)
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
        close_mosaic=20,  # 最後 20 epochs 關閉 mosaic

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
    print(f"模型保存位置: {OUTPUT_DIR}/highres_1280/weights/")
    print(f"完成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ============================================================
# 主程序
# ============================================================

def main():
    """主程序入口"""

    # 設置環境
    gpu_mem = setup_environment()

    # 驗證 GPU 記憶體足夠
    if gpu_mem < 20:
        print(f"⚠️ 警告: GPU 記憶體 ({gpu_mem:.1f}GB) 可能不足")
        print("   建議: 減少 batch 大小或 imgsz")

    # 執行訓練
    try:
        results = train_high_resolution()
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
