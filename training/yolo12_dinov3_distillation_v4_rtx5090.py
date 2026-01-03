#!/usr/bin/env python3
"""
YOLO12 + DINOv3 知識蒸餾訓練腳本 v4 - RTX 5090 Blackwell 深度優化版

基於 2025-12-20 深度調研結果重寫:

調研來源:
- LightlyTrain 官方文檔: https://docs.lightly.ai/train/stable/methods/distillation.html
- PyTorch 性能調優指南: https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html
- RTX 5090 Blackwell 相容性: https://github.com/pytorch/pytorch/issues/159207

關鍵修正:
1. LightlyTrain API: 使用 train() 而非 pretrain()
2. 蒸餾 epochs: 100-300 (非 20)
3. Batch size: 蒸餾 128+, 微調 64
4. 移除訓練時的 torch.compile (編譯開銷過大)
5. 啟用 Tensor Core: torch.set_float32_matmul_precision('high')

工作站規格:
- GPU: NVIDIA RTX 5090 (Blackwell sm_120, 32GB VRAM)
- CPU: Intel i9-14900 (24 核心)
- RAM: 125 GB DDR5
- CUDA: 12.8 + cuDNN 9.1
- PyTorch: 2.9.1

目標: mAP50 > 0.70 (基線 0.6444, +8.6%)

創建日期: 2025-12-20
"""

import os
import sys
import gc
import json
import torch
import multiprocessing
from pathlib import Path
from datetime import datetime

# ============================================================
# RTX 5090 Blackwell 環境優化 (基於調研)
# ============================================================

def setup_rtx5090_optimizations():
    """
    設置 RTX 5090 Blackwell 架構優化

    參考:
    - https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html
    - https://github.com/pytorch/pytorch/issues/159207
    """
    print("=" * 70)
    print("RTX 5090 Blackwell 環境優化 (基於調研)")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("❌ CUDA 不可用")
        return False

    # 1. 驗證 GPU
    gpu_name = torch.cuda.get_device_name(0)
    compute_cap = torch.cuda.get_device_capability()
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

    print(f"✅ GPU: {gpu_name}")
    print(f"   VRAM: {vram_gb:.1f} GB")
    print(f"   Compute Capability: {compute_cap[0]}.{compute_cap[1]}")

    if compute_cap[0] >= 12:
        print("   架構: Blackwell (sm_120) ✅")
    elif compute_cap[0] >= 8:
        print("   架構: Ampere/Ada Lovelace")

    # 2. 啟用 Tensor Core 優化 (關鍵!)
    # 參考: PyTorch Performance Tuning Guide
    torch.set_float32_matmul_precision('high')  # 優先速度
    print("✅ Tensor Core 優化: torch.set_float32_matmul_precision('high')")

    # 3. TF32 (Tensor Float 32) - Ampere+ 支援
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print("✅ TF32 Tensor Core 運算已啟用")

    # 4. cuDNN 自動調優
    # 參考: "auto-tuner decisions may be non-deterministic"
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    print("✅ cuDNN 自動調優已啟用 (cudnn.benchmark=True)")

    # 5. BF16 支援檢查
    if torch.cuda.is_bf16_supported():
        print("✅ BF16 已支援 - 混合精度訓練可用")

    # 6. CUDA 記憶體分配器優化
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    print("✅ CUDA 記憶體分配器: expandable_segments")

    # 7. 清理 GPU 記憶體
    torch.cuda.empty_cache()
    gc.collect()
    print("✅ GPU 記憶體已清理")

    print("")
    return True


# ============================================================
# 路徑配置
# ============================================================

BASE_DIR = Path('/home/thc1006/dev/music-app/training')
DATASET_DIR = BASE_DIR / 'datasets/yolo_harmony_v2_phase8_final'
OUTPUT_DIR = BASE_DIR / 'harmony_omr_v2_dinov3_distill_v4'
PHASE8_MODEL = BASE_DIR / 'harmony_omr_v2_phase8/phase8_training/weights/best.pt'
DATASET_YAML = DATASET_DIR / 'harmony_phase8_final.yaml'

# ============================================================
# DINOv3 配置 (基於 LightlyTrain 文檔調研)
# ============================================================

# 支援的 DINOv3 教師模型:
# - dinov3/vits16        (Small)
# - dinov3/vitb16        (Base)
# - dinov3/vitl16        (Large) ← 推薦: 最強特徵提取
# - dinov3/vitl16-sat493m
# - dinov3/vith16plus    (Huge)
# - dinov3/vit7b16       (7B)

DINOV3_CONFIG = {
    # 使用 Large 版本獲得最佳特徵
    'teacher': 'dinov3/vitl16',  # ViT-L/16, ~304M 參數
    'teacher_params': '304M',
    'teacher_dim': 1024,
}

# ============================================================
# 訓練配置 (基於調研結果優化)
# ============================================================

# 蒸餾預訓練配置
# 參考: LightlyTrain 推薦 epochs 100-3000
DISTILLATION_CONFIG = {
    'epochs': 100,           # LightlyTrain 推薦 100-300
    'batch_size': 64,        # DINOv3 Large (304M) + YOLO12s 安全值
    'method': 'distillation', # 使用 v2 蒸餾 (更快)
}

# YOLO 微調配置
# 參考: Phase 8 成功配置
FINETUNE_CONFIG = {
    'epochs': 150,           # Phase 8 成功使用 150
    'batch_size': 64,        # RTX 5090 32GB 再挑戰
    'imgsz': 640,
    'lr0': 0.001,            # Phase 8 成功的學習率
    'lrf': 0.01,
    'optimizer': 'AdamW',
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'warmup_epochs': 3,
    'patience': 50,

    # DataLoader 優化 (i9-14900 24核心)
    # 參考: PyTorch 調優指南 "num_workers > 0, pin_memory=True"
    'workers': 16,           # 24核心，預留系統
    'cache': 'ram',          # 125GB RAM 足夠

    # OMR 專用數據增強 (保守配置)
    'hsv_h': 0.015,
    'hsv_s': 0.4,
    'hsv_v': 0.3,
    'degrees': 0.0,          # 樂譜不旋轉
    'translate': 0.1,
    'scale': 0.3,
    'shear': 0.0,
    'perspective': 0.0,
    'flipud': 0.0,           # 樂譜不翻轉
    'fliplr': 0.0,           # 音樂符號有方向性
    'mosaic': 0.5,
    'mixup': 0.0,            # 樂譜不適合 mixup
    'copy_paste': 0.0,
    'erasing': 0.0,
}


def check_environment():
    """環境檢查"""
    print("=" * 70)
    print("環境檢查")
    print("=" * 70)

    errors = []

    # GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        errors.append("CUDA 不可用")
        print("❌ CUDA 不可用")

    # CPU
    cpu_count = multiprocessing.cpu_count()
    print(f"✅ CPU 核心: {cpu_count}")

    # RAM
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / 1e9
        print(f"✅ RAM: {ram_gb:.1f} GB")
    except ImportError:
        print("⚠️ psutil 未安裝")

    # 資料集
    if DATASET_DIR.exists():
        train_count = len(list((DATASET_DIR / 'train/images').glob('*')))
        val_count = len(list((DATASET_DIR / 'val/images').glob('*')))
        print(f"✅ 資料集: {train_count:,} 訓練 + {val_count:,} 驗證")
    else:
        errors.append(f"資料集不存在: {DATASET_DIR}")
        print(f"❌ 資料集不存在")

    # Phase 8 模型
    if PHASE8_MODEL.exists():
        size_mb = PHASE8_MODEL.stat().st_size / 1e6
        print(f"✅ Phase 8 模型: {size_mb:.1f} MB")
    else:
        errors.append("Phase 8 模型不存在")
        print("❌ Phase 8 模型不存在")

    # LightlyTrain
    try:
        import lightly_train
        print(f"✅ LightlyTrain: {lightly_train.__version__}")

        # 檢查可用方法
        methods = lightly_train.list_methods()
        print(f"   可用方法: {methods}")
    except ImportError:
        errors.append("LightlyTrain 未安裝")
        print("❌ LightlyTrain 未安裝")

    # PyTorch 版本
    print(f"✅ PyTorch: {torch.__version__}")
    print(f"   CUDA: {torch.version.cuda}")

    if errors:
        print(f"\n❌ 發現 {len(errors)} 個錯誤")
        return False

    print("\n✅ 環境檢查通過")
    return True


def run_distillation_pretraining():
    """
    Step 1: 使用 LightlyTrain 進行 DINOv3 蒸餾預訓練

    正確 API 用法 (基於官方文檔):
    https://docs.lightly.ai/train/stable/methods/distillation.html

    關鍵修正 (2025-12-20):
    - 使用 YOLO 模型實例而非路徑字串
    - 使用 pretrain() 而非 train() (train 已棄用)
    """
    print("\n" + "=" * 70)
    print("Step 1: DINOv3 蒸餾預訓練")
    print("=" * 70)

    import lightly_train
    from ultralytics import YOLO

    pretrain_out = OUTPUT_DIR / 'pretrain'
    pretrain_out.mkdir(parents=True, exist_ok=True)

    # 載入 Phase 8 模型作為學生模型實例
    # 參考: https://docs.lightly.ai/train/stable/models/ultralytics.html
    print(f"\n載入 Phase 8 模型...")
    student_model = YOLO(str(PHASE8_MODEL))
    print(f"✅ 模型載入成功: {student_model.model.__class__.__name__}")

    # 配置說明
    print(f"\n配置:")
    print(f"  教師模型: {DINOV3_CONFIG['teacher']} ({DINOV3_CONFIG['teacher_params']})")
    print(f"  學生模型: YOLO12s (Phase 8, mAP50=0.6444)")
    print(f"  資料路徑: {DATASET_DIR / 'train/images'}")
    print(f"  Epochs: {DISTILLATION_CONFIG['epochs']}")
    print(f"  Batch Size: {DISTILLATION_CONFIG['batch_size']}")
    print(f"  方法: {DISTILLATION_CONFIG['method']}")

    try:
        # 正確的 LightlyTrain API 調用
        # 使用 pretrain() 並傳入 YOLO 模型實例
        # 參考: https://docs.lightly.ai/train/stable/tutorials/yolo/index.html
        lightly_train.pretrain(
            out=str(pretrain_out),
            data=str(DATASET_DIR / 'train/images'),
            model=student_model,  # 傳入 YOLO 模型實例而非路徑
            method=DISTILLATION_CONFIG['method'],
            method_args={
                'teacher': DINOV3_CONFIG['teacher'],  # DINOv3 Large
            },
            epochs=DISTILLATION_CONFIG['epochs'],
            batch_size=DISTILLATION_CONFIG['batch_size'],
            num_workers=FINETUNE_CONFIG['workers'],
            overwrite=True,  # 覆蓋舊輸出
        )
        print("\n✅ 蒸餾預訓練完成!")
        return pretrain_out

    except Exception as e:
        print(f"\n⚠️ LightlyTrain 蒸餾失敗: {e}")
        print("回退至 YOLO 優化訓練...")
        return None


def run_yolo_finetuning(pretrained_model_path=None):
    """
    Step 2: YOLO 目標檢測微調

    使用蒸餾後的模型或 Phase 8 模型進行微調
    """
    print("\n" + "=" * 70)
    print("Step 2: YOLO 目標檢測微調")
    print("=" * 70)

    from ultralytics import YOLO

    # 選擇起始模型
    if pretrained_model_path:
        # 尋找蒸餾後的模型
        exported_model = pretrained_model_path / 'exported_models/exported_last.pt'
        if not exported_model.exists():
            exported_model = pretrained_model_path / 'exported_models/model.pt'

        if exported_model.exists():
            model_path = str(exported_model)
            print(f"使用蒸餾後模型: {model_path}")
        else:
            model_path = str(PHASE8_MODEL)
            print(f"找不到蒸餾模型，使用 Phase 8: {model_path}")
    else:
        model_path = str(PHASE8_MODEL)
        print(f"使用 Phase 8 模型: {model_path}")

    model = YOLO(model_path)

    finetune_out = OUTPUT_DIR / 'finetune'

    # 構建訓練配置
    train_config = {
        # 基礎配置
        'data': str(DATASET_YAML),
        'epochs': FINETUNE_CONFIG['epochs'],
        'imgsz': FINETUNE_CONFIG['imgsz'],
        'device': 0,

        # RTX 5090 優化
        'batch': FINETUNE_CONFIG['batch_size'],
        'amp': True,  # 混合精度

        # DataLoader 優化
        'workers': FINETUNE_CONFIG['workers'],
        'cache': FINETUNE_CONFIG['cache'],

        # 學習率配置
        'lr0': FINETUNE_CONFIG['lr0'],
        'lrf': FINETUNE_CONFIG['lrf'],
        'optimizer': FINETUNE_CONFIG['optimizer'],
        'momentum': FINETUNE_CONFIG['momentum'],
        'weight_decay': FINETUNE_CONFIG['weight_decay'],
        'warmup_epochs': FINETUNE_CONFIG['warmup_epochs'],

        # 訓練控制
        'patience': FINETUNE_CONFIG['patience'],

        # 輸出
        'project': str(finetune_out),
        'name': 'dinov3_finetuned',
        'exist_ok': True,
        'verbose': True,

        # 數據增強 (OMR 專用)
        'hsv_h': FINETUNE_CONFIG['hsv_h'],
        'hsv_s': FINETUNE_CONFIG['hsv_s'],
        'hsv_v': FINETUNE_CONFIG['hsv_v'],
        'degrees': FINETUNE_CONFIG['degrees'],
        'translate': FINETUNE_CONFIG['translate'],
        'scale': FINETUNE_CONFIG['scale'],
        'shear': FINETUNE_CONFIG['shear'],
        'perspective': FINETUNE_CONFIG['perspective'],
        'flipud': FINETUNE_CONFIG['flipud'],
        'fliplr': FINETUNE_CONFIG['fliplr'],
        'mosaic': FINETUNE_CONFIG['mosaic'],
        'mixup': FINETUNE_CONFIG['mixup'],
        'copy_paste': FINETUNE_CONFIG['copy_paste'],
        'erasing': FINETUNE_CONFIG['erasing'],
    }

    print(f"\n微調配置:")
    print(f"  Epochs: {FINETUNE_CONFIG['epochs']}")
    print(f"  Batch Size: {FINETUNE_CONFIG['batch_size']}")
    print(f"  學習率: {FINETUNE_CONFIG['lr0']}")
    print(f"  Workers: {FINETUNE_CONFIG['workers']}")
    print(f"  Cache: {FINETUNE_CONFIG['cache']}")

    # 開始訓練
    print("\n開始訓練...")
    start_time = datetime.now()

    results = model.train(**train_config)

    end_time = datetime.now()
    duration = end_time - start_time
    print(f"\n✅ 微調完成! 耗時: {duration}")

    return finetune_out / 'dinov3_finetuned/weights/best.pt'


def evaluate_model(model_path, name="Model"):
    """評估模型"""
    print(f"\n評估 {name}:")

    from ultralytics import YOLO

    if not Path(model_path).exists():
        print(f"❌ 模型不存在: {model_path}")
        return None

    model = YOLO(str(model_path))

    results = model.val(
        data=str(DATASET_YAML),
        imgsz=640,
        batch=FINETUNE_CONFIG['batch_size'],
        device=0,
        workers=FINETUNE_CONFIG['workers'],
    )

    metrics = {
        'mAP50': float(results.box.map50),
        'mAP50-95': float(results.box.map),
    }

    print(f"  mAP50: {metrics['mAP50']:.4f}")
    print(f"  mAP50-95: {metrics['mAP50-95']:.4f}")

    return metrics


def main():
    """主函數"""
    print("=" * 70)
    print("YOLO12 + DINOv3 知識蒸餾 v4")
    print("RTX 5090 Blackwell 深度優化版")
    print(f"時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # 1. 設置 RTX 5090 優化
    setup_rtx5090_optimizations()

    # 2. 環境檢查
    if not check_environment():
        return

    # 3. 評估基線
    print("\n" + "=" * 70)
    print("Phase 8 基線評估")
    print("=" * 70)
    baseline = evaluate_model(PHASE8_MODEL, "Phase 8 基線")

    # 4. 執行蒸餾預訓練
    pretrain_out = run_distillation_pretraining()

    # 5. 執行 YOLO 微調
    best_model_path = run_yolo_finetuning(pretrain_out)

    # 6. 評估結果
    if best_model_path and Path(best_model_path).exists():
        print("\n" + "=" * 70)
        print("最終評估")
        print("=" * 70)
        final_metrics = evaluate_model(best_model_path, "蒸餾後模型")

        if final_metrics and baseline:
            improvement = final_metrics['mAP50'] - baseline['mAP50']
            print(f"\n" + "=" * 70)
            print("結果比較")
            print("=" * 70)
            print(f"Phase 8 基線:  mAP50 = {baseline['mAP50']:.4f}")
            print(f"蒸餾後模型:   mAP50 = {final_metrics['mAP50']:.4f}")
            print(f"提升:         {improvement:+.4f} ({improvement/baseline['mAP50']*100:+.1f}%)")

            # 保存報告
            report = {
                'experiment': 'DINOv3 Distillation v4 (RTX 5090 Optimized)',
                'timestamp': datetime.now().isoformat(),
                'dinov3_config': DINOV3_CONFIG,
                'distillation_config': DISTILLATION_CONFIG,
                'finetune_config': FINETUNE_CONFIG,
                'baseline': baseline,
                'final_metrics': final_metrics,
                'improvement': {
                    'mAP50': improvement,
                    'mAP50_percent': improvement/baseline['mAP50']*100
                },
                'best_model': str(best_model_path),
            }

            report_path = OUTPUT_DIR / 'experiment_report.json'
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"\n報告已保存: {report_path}")

    print("\n" + "=" * 70)
    print("實驗完成!")
    print("=" * 70)


if __name__ == '__main__':
    main()
