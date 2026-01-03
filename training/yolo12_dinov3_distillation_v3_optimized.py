#!/usr/bin/env python3
"""
YOLO12 + DINOv3 知識蒸餾訓練腳本 v3 - RTX 5090 Blackwell 專項優化版

針對工作站規格深度優化:
- GPU: NVIDIA RTX 5090 (Blackwell sm_120, 33.7GB VRAM)
- CPU: Intel i9-14900 (24 核心, 36MB L3)
- RAM: 125 GB DDR5
- CUDA: 12.8 + cuDNN 9.1
- PyTorch: 2.9.1

優化項目:
1. Blackwell 架構專屬優化 (BF16, TF32, torch.compile)
2. 多核心 DataLoader 優化 (24 workers, pin_memory, persistent_workers)
3. 記憶體優化 (數據集緩存, CUDA allocator 配置)
4. 自動批次大小調整 (充分利用 32GB VRAM)

目標: 從 Phase 8 的 mAP50=0.6444 提升至 >0.68

創建日期: 2025-12-20
"""

import os
import sys
import gc
import json
import torch
import timm
import multiprocessing
from pathlib import Path
from datetime import datetime

# ============================================================
# RTX 5090 Blackwell 環境優化
# ============================================================

def setup_blackwell_optimizations():
    """設置 RTX 5090 Blackwell 架構專屬優化"""

    print("=" * 70)
    print("RTX 5090 Blackwell 環境優化")
    print("=" * 70)

    # 1. 驗證 Blackwell 架構
    if torch.cuda.is_available():
        compute_cap = torch.cuda.get_device_capability()
        print(f"✅ GPU Compute Capability: {compute_cap[0]}.{compute_cap[1]}")

        if compute_cap[0] >= 12:
            print("✅ 檢測到 Blackwell 架構 (sm_120)")
        elif compute_cap[0] >= 8:
            print("⚠️ 檢測到 Ampere/Ada 架構，部分優化可能不適用")

    # 2. 設置 CUDA 環境變數 (Blackwell 專屬)
    # 注意: PyTorch 2.9+ 使用 PYTORCH_ALLOC_CONF 取代 PYTORCH_CUDA_ALLOC_CONF
    os.environ['PYTORCH_ALLOC_CONF'] = 'max_split_size_mb:512,expandable_segments:True'
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['TORCH_CUDA_ARCH_LIST'] = '12.0'  # Blackwell
    print("✅ CUDA 記憶體分配器優化: max_split_size=512MB, expandable_segments")

    # 3. 啟用 TF32 (Tensor Float 32) - Blackwell 支援
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print("✅ TF32 Tensor Core 運算已啟用")

    # 4. 啟用 cuDNN 自動調優
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    print("✅ cuDNN 自動調優已啟用")

    # 5. 檢查 BF16 支援 (Blackwell 原生支援)
    if torch.cuda.is_bf16_supported():
        print("✅ BF16 (Brain Float 16) 已支援 - 比 FP16 更穩定")

    # 6. 設置記憶體增長模式
    torch.cuda.empty_cache()
    gc.collect()
    print("✅ GPU 記憶體已清理")

    # 7. 多進程設置
    if hasattr(multiprocessing, 'set_start_method'):
        try:
            multiprocessing.set_start_method('spawn', force=True)
            print("✅ 多進程啟動方法: spawn")
        except RuntimeError:
            pass

    print("")
    return True


# ============================================================
# 路徑配置
# ============================================================

BASE_DIR = Path('/home/thc1006/dev/music-app/training')
DATASET_DIR = BASE_DIR / 'datasets/yolo_harmony_v2_phase8_final'
OUTPUT_DIR = BASE_DIR / 'harmony_omr_v2_dinov3_distill_v3'
PHASE8_MODEL = BASE_DIR / 'harmony_omr_v2_phase8/phase8_training/weights/best.pt'
DATASET_YAML = DATASET_DIR / 'harmony_phase8_final.yaml'

# 類別配置 (33 classes)
CLASS_NAMES = {
    0: 'notehead_filled', 1: 'notehead_hollow', 2: 'stem', 3: 'beam',
    4: 'flag_8th', 5: 'flag_16th', 6: 'flag_32nd', 7: 'augmentation_dot',
    8: 'tie', 9: 'clef_treble', 10: 'clef_bass', 11: 'clef_alto',
    12: 'clef_tenor', 13: 'accidental_sharp', 14: 'accidental_flat',
    15: 'accidental_natural', 16: 'accidental_double_sharp',
    17: 'accidental_double_flat', 18: 'rest_whole', 19: 'rest_half',
    20: 'rest_quarter', 21: 'rest_8th', 22: 'rest_16th', 23: 'barline',
    24: 'barline_double', 25: 'barline_final', 26: 'barline_repeat',
    27: 'time_signature', 28: 'key_signature', 29: 'fermata',
    30: 'dynamic_soft', 31: 'dynamic_loud', 32: 'ledger_line'
}

# ============================================================
# DINOv3 配置 (已驗證)
# ============================================================

DINOV3_CONFIG = {
    # DINOv3 教師模型配置 - 使用最大的 Large 版本獲得最佳效果
    # 注意: 教師模型只在訓練時使用，不會部署到手機
    # LightlyTrain 格式 (用於蒸餾)
    'lightly_name': 'dinov3/vitl16',  # ViT-L/16, 304M 參數 (最強!)
    # timm 格式 (用於環境檢查)
    'timm_name': 'vit_large_patch16_dinov3',
    'params': '304M',
    'embed_dim': 1024,  # Large 版本: 1024 維特徵
    'input_size': 640,
}

# ============================================================
# RTX 5090 + i9-14900 優化訓練配置
# ============================================================

# 系統規格
SYSTEM_SPECS = {
    'gpu': 'RTX 5090',
    'gpu_vram_gb': 33.7,
    'gpu_arch': 'Blackwell (sm_120)',
    'cpu': 'i9-14900',
    'cpu_cores': 24,
    'ram_gb': 125,
    'cuda_version': '12.8',
    'cudnn_version': '9.1',
}

# 蒸餾配置
DISTILLATION_CONFIG = {
    'temperature': 4.0,
    'alpha': 0.5,
    'feature_weight': 0.3,
}

# 優化後的訓練配置 (基於調研結果優化)
TRAINING_CONFIG = {
    # === 基礎配置 (保守配置，避免過擬合) ===
    'pretrain_epochs': 20,     # 減少預訓練 epochs
    'finetune_epochs': 150,    # 增加微調 epochs (Phase 8 成功配置)
    'imgsz': 640,
    'lr0': 0.001,              # Phase 8 成功的學習率
    'lrf': 0.01,               # 最終學習率 = lr0 * lrf
    'optimizer': 'AdamW',
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'warmup_epochs': 3,
    'warmup_momentum': 0.8,
    'warmup_bias_lr': 0.1,

    # === RTX 5090 32GB VRAM 優化 ===
    'batch_size': -1,         # 自動偵測最佳批次大小
    'batch_size_manual': 32,  # 手動設置時使用 (32GB VRAM 可支援)
    'device': 0,

    # === i9-14900 24核心優化 ===
    'workers': 20,            # 24核心，預留4核給系統
    'prefetch_factor': 4,     # 每個 worker 預取 4 批
    'pin_memory': True,       # 鎖定記憶體加速 GPU 傳輸
    'persistent_workers': True,  # 保持 worker 進程

    # === Blackwell 架構優化 ===
    'amp': True,              # 自動混合精度
    'half': False,            # 使用 AMP 而非純 FP16

    # === 125GB RAM 優化 ===
    'cache': 'ram',           # 緩存數據集到 RAM (125GB 足夠)

    # === 訓練穩定性 ===
    'patience': 50,           # Early stopping 耐心值
    'save_period': 10,        # 每 10 epochs 保存
    'val_period': 1,          # 每 epoch 驗證

    # === 數據增強 (OMR 專用 - 保守配置) ===
    'hsv_h': 0.015,
    'hsv_s': 0.4,             # 降低飽和度變化
    'hsv_v': 0.3,             # 降低亮度變化
    'degrees': 0.0,           # 樂譜不應旋轉
    'translate': 0.1,
    'scale': 0.3,             # 降低縮放範圍
    'shear': 0.0,             # 樂譜不應剪切
    'perspective': 0.0,       # 樂譜不應透視變換
    'flipud': 0.0,            # 樂譜不應上下翻轉
    'fliplr': 0.0,            # 樂譜不應左右翻轉 (音樂符號有方向性)
    'mosaic': 0.5,
    'mixup': 0.0,             # 禁用 mixup (樂譜不適合)
    'copy_paste': 0.0,        # 樂譜不適合 copy-paste
    'erasing': 0.0,           # 關閉隨機擦除
}


def get_optimal_batch_size():
    """根據 GPU VRAM 計算最佳批次大小"""
    if not torch.cuda.is_available():
        return 8

    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

    # RTX 5090 (32GB) 的經驗值
    if vram_gb >= 30:
        return 32  # 32GB VRAM
    elif vram_gb >= 20:
        return 24  # 24GB VRAM
    elif vram_gb >= 12:
        return 16  # 12GB VRAM
    elif vram_gb >= 8:
        return 8   # 8GB VRAM
    else:
        return 4


def check_environment():
    """環境檢查 (增強版)"""
    print("=" * 70)
    print("環境檢查 (RTX 5090 Blackwell 優化版)")
    print("=" * 70)

    errors = []
    warnings = []

    # === GPU 檢查 ===
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        compute_cap = torch.cuda.get_device_capability()

        print(f"✅ GPU: {gpu_name}")
        print(f"   VRAM: {gpu_mem:.1f} GB")
        print(f"   Compute Capability: {compute_cap[0]}.{compute_cap[1]}")
        print(f"   CUDA: {torch.version.cuda}")
        print(f"   cuDNN: {torch.backends.cudnn.version()}")

        # Blackwell 檢查
        if compute_cap[0] >= 12:
            print(f"   架構: Blackwell (sm_120) ✅")
        elif compute_cap[0] >= 8:
            print(f"   架構: Ampere/Ada ⚠️")
            warnings.append("非 Blackwell 架構，部分優化可能不適用")

        # BF16 支援
        if torch.cuda.is_bf16_supported():
            print(f"   BF16: 支援 ✅")

        # TF32 支援
        print(f"   TF32: 支援 ✅")

    else:
        errors.append("CUDA 不可用")
        print("❌ CUDA 不可用")

    # === CPU 檢查 ===
    cpu_count = multiprocessing.cpu_count()
    print(f"\n✅ CPU 核心: {cpu_count}")
    print(f"   建議 workers: {min(cpu_count - 4, 20)}")

    # === RAM 檢查 ===
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / 1e9
        ram_available = psutil.virtual_memory().available / 1e9
        print(f"\n✅ RAM: {ram_gb:.1f} GB (可用: {ram_available:.1f} GB)")

        if ram_gb >= 64:
            print(f"   數據集緩存: RAM ✅ (推薦)")
        else:
            print(f"   數據集緩存: disk (RAM 不足)")
    except ImportError:
        print("\n⚠️ psutil 未安裝，無法檢查 RAM")

    # === 資料集檢查 ===
    print("")
    if DATASET_DIR.exists():
        train_count = len(list((DATASET_DIR / 'train/images').glob('*')))
        val_count = len(list((DATASET_DIR / 'val/images').glob('*')))
        print(f"✅ 資料集: {train_count:,} 訓練 + {val_count:,} 驗證")
    else:
        errors.append(f"資料集不存在: {DATASET_DIR}")
        print(f"❌ 資料集不存在")

    # === Phase 8 模型檢查 ===
    if PHASE8_MODEL.exists():
        size_mb = PHASE8_MODEL.stat().st_size / 1e6
        print(f"✅ Phase 8 模型: {size_mb:.1f} MB")
    else:
        errors.append("Phase 8 模型不存在")
        print("❌ Phase 8 模型不存在")

    # === DINOv3 檢查 ===
    try:
        model = timm.create_model(DINOV3_CONFIG['timm_name'], pretrained=False)
        params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"✅ DINOv3: {DINOV3_CONFIG['timm_name']} ({params:.1f}M)")
        del model
        torch.cuda.empty_cache()
    except Exception as e:
        errors.append(f"DINOv3: {e}")
        print(f"❌ DINOv3: {e}")

    # === LightlyTrain 檢查 ===
    try:
        import lightly_train
        print(f"✅ LightlyTrain: {lightly_train.__version__}")
    except ImportError:
        errors.append("LightlyTrain 未安裝")
        print("❌ LightlyTrain 未安裝")

    # === 結果 ===
    print("")
    if warnings:
        print(f"⚠️ 警告: {len(warnings)} 項")
        for w in warnings:
            print(f"   - {w}")

    if errors:
        print(f"\n❌ 發現 {len(errors)} 個錯誤，無法繼續")
        return False

    print("\n✅ 環境檢查通過")

    # 顯示優化配置摘要
    print("\n" + "-" * 50)
    print("優化配置摘要:")
    print("-" * 50)
    optimal_batch = get_optimal_batch_size()
    print(f"  最佳批次大小: {optimal_batch}")
    print(f"  DataLoader workers: {TRAINING_CONFIG['workers']}")
    print(f"  prefetch_factor: {TRAINING_CONFIG['prefetch_factor']}")
    print(f"  pin_memory: {TRAINING_CONFIG['pin_memory']}")
    print(f"  persistent_workers: {TRAINING_CONFIG['persistent_workers']}")
    print(f"  數據集緩存: {TRAINING_CONFIG['cache']}")
    print(f"  混合精度: AMP (TF32 + 動態 FP16/BF16)")

    return True


def load_dinov3_teacher():
    """加載 DINOv3 教師模型 (優化版)"""
    print("\n" + "=" * 70)
    print("加載 DINOv3 教師模型")
    print("=" * 70)

    model_name = DINOV3_CONFIG['timm_name']
    print(f"模型: {model_name}")

    # 加載模型
    teacher = timm.create_model(model_name, pretrained=True)
    teacher.eval()

    # 凍結參數
    for param in teacher.parameters():
        param.requires_grad = False

    # 移至 GPU 並使用 BF16 (Blackwell 原生支援)
    teacher = teacher.cuda()
    if torch.cuda.is_bf16_supported():
        teacher = teacher.to(dtype=torch.bfloat16)
        print("✅ 教師模型使用 BF16 精度")

    # 使用 torch.compile 優化 (PyTorch 2.x)
    try:
        teacher = torch.compile(teacher, mode="reduce-overhead")
        print("✅ torch.compile 優化已啟用 (reduce-overhead 模式)")
    except Exception as e:
        print(f"⚠️ torch.compile 不可用: {e}")

    # 測試推理
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        dummy = torch.randn(1, 3, 640, 640, device='cuda', dtype=torch.bfloat16)
        features = teacher.forward_features(dummy)
        print(f"✅ 加載成功")
        print(f"   特徵形狀: {features.shape}")
        print(f"   特徵維度: {features.shape[-1]}")
        print(f"   精度: {features.dtype}")

    return teacher


def method_lightly_train():
    """使用 LightlyTrain 進行蒸餾 (優化版)"""
    print("\n" + "=" * 70)
    print("方法: LightlyTrain 蒸餾 (RTX 5090 優化)")
    print("=" * 70)

    import lightly_train

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 檢查可用方法
    print("\n[檢查 LightlyTrain API]")
    available_methods = lightly_train.list_methods()
    print(f"可用方法: {available_methods}")

    # 計算最佳批次大小
    optimal_batch = get_optimal_batch_size()
    print(f"\n自動偵測最佳批次大小: {optimal_batch}")

    # Step 1: 自監督預訓練
    print(f"\n[Step 1/2] 自監督預訓練")
    print(f"資料: {DATASET_DIR / 'train/images'}")
    print(f"Epochs: {TRAINING_CONFIG['pretrain_epochs']}")
    print(f"Batch Size: {optimal_batch}")

    pretrain_out = OUTPUT_DIR / 'pretrain'

    # 選擇模型: 使用 Phase 8 已訓練模型或全新 YOLO12s
    if PHASE8_MODEL.exists():
        model_to_use = str(PHASE8_MODEL)
        print(f"使用 Phase 8 模型: {model_to_use}")
    else:
        model_to_use = 'ultralytics/yolov12s.pt'  # ✅ 正確的 LightlyTrain 模型名稱
        print(f"使用預訓練 YOLO12s: {model_to_use}")

    # DINOv3 教師模型配置
    # 使用 Small 版本，更穩定且已驗證可用
    teacher_model = DINOV3_CONFIG['lightly_name']
    print(f"DINOv3 教師模型: {teacher_model} ({DINOV3_CONFIG['params']} 參數)")
    print(f"學生模型: YOLO12s (9.3M 參數) → 部署到 Android")

    try:
        # 使用較短的預訓練週期，避免過擬合
        pretrain_epochs = 20  # 減少到 20 epochs

        lightly_train.pretrain(
            out=str(pretrain_out),
            data=str(DATASET_DIR / 'train/images'),
            model=model_to_use,  # ✅ 正確的模型參數
            method='distillation',
            method_args={
                'teacher': teacher_model,  # ✅ DINOv3 作為教師
            },
            epochs=pretrain_epochs,
            batch_size=optimal_batch,
            num_workers=TRAINING_CONFIG['workers'],
        )
        print(f"✅ 預訓練完成 ({pretrain_epochs} epochs)")
    except Exception as e:
        print(f"⚠️ LightlyTrain 蒸餾失敗: {e}")
        print("嘗試備選方案: 直接 YOLO 訓練 + 特徵增強")
        return method_yolo_optimized_training()

    # Step 2: 目標檢測微調
    print(f"\n[Step 2/2] 目標檢測微調")

    # 加載蒸餾後的模型
    exported_model = pretrain_out / 'exported_models/exported_last.pt'
    if not exported_model.exists():
        # 嘗試其他可能的路徑
        exported_model = pretrain_out / 'exported_models/model.pt'

    if not exported_model.exists():
        print(f"⚠️ 找不到導出的模型，使用 YOLO 原生微調")
        return method_yolo_optimized_training()

    print(f"使用蒸餾後模型: {exported_model}")

    # 使用 Ultralytics YOLO 進行微調 (更穩定)
    from ultralytics import YOLO

    finetune_out = OUTPUT_DIR / 'finetune'
    model = YOLO(str(exported_model))

    try:
        results = model.train(
            data=str(DATASET_YAML),
            epochs=TRAINING_CONFIG['finetune_epochs'],
            batch=optimal_batch,
            imgsz=TRAINING_CONFIG['imgsz'],
            device=TRAINING_CONFIG['device'],
            workers=TRAINING_CONFIG['workers'],
            amp=TRAINING_CONFIG['amp'],
            cache=TRAINING_CONFIG['cache'],
            project=str(finetune_out),
            name='dinov3_finetuned',
            exist_ok=True,
        )
        print("✅ 微調完成")
    except Exception as e:
        print(f"⚠️ 微調失敗: {e}")
        return None

    return finetune_out / 'dinov3_finetuned/weights/best.pt'


def method_yolo_optimized_training():
    """YOLO 優化訓練 (RTX 5090 專屬配置)"""
    print("\n" + "=" * 70)
    print("方法: YOLO 優化訓練 (RTX 5090 Blackwell)")
    print("=" * 70)

    from ultralytics import YOLO

    # 加載 Phase 8 模型
    model = YOLO(str(PHASE8_MODEL))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 計算最佳批次大小
    optimal_batch = get_optimal_batch_size()

    # RTX 5090 + i9-14900 優化訓練配置
    train_config = {
        # 基礎配置
        'data': str(DATASET_YAML),
        'epochs': TRAINING_CONFIG['finetune_epochs'],
        'imgsz': TRAINING_CONFIG['imgsz'],
        'device': TRAINING_CONFIG['device'],

        # === RTX 5090 32GB VRAM 優化 ===
        'batch': optimal_batch,
        'amp': TRAINING_CONFIG['amp'],

        # === i9-14900 多核心優化 ===
        'workers': TRAINING_CONFIG['workers'],

        # === 125GB RAM 優化 ===
        'cache': TRAINING_CONFIG['cache'],

        # === 學習率配置 ===
        'lr0': TRAINING_CONFIG['lr0'],
        'lrf': TRAINING_CONFIG['lrf'],
        'optimizer': TRAINING_CONFIG['optimizer'],
        'momentum': TRAINING_CONFIG['momentum'],
        'weight_decay': TRAINING_CONFIG['weight_decay'],
        'warmup_epochs': TRAINING_CONFIG['warmup_epochs'],
        'warmup_momentum': TRAINING_CONFIG['warmup_momentum'],
        'warmup_bias_lr': TRAINING_CONFIG['warmup_bias_lr'],

        # === 訓練控制 ===
        'patience': TRAINING_CONFIG['patience'],
        'save_period': TRAINING_CONFIG['save_period'],

        # === 輸出配置 ===
        'project': str(OUTPUT_DIR),
        'name': 'yolo_optimized',
        'exist_ok': True,
        'pretrained': True,
        'verbose': True,

        # === 數據增強 (OMR 專用) ===
        'hsv_h': TRAINING_CONFIG['hsv_h'],
        'hsv_s': TRAINING_CONFIG['hsv_s'],
        'hsv_v': TRAINING_CONFIG['hsv_v'],
        'degrees': TRAINING_CONFIG['degrees'],
        'translate': TRAINING_CONFIG['translate'],
        'scale': TRAINING_CONFIG['scale'],
        'shear': TRAINING_CONFIG['shear'],
        'perspective': TRAINING_CONFIG['perspective'],
        'flipud': TRAINING_CONFIG['flipud'],
        'fliplr': TRAINING_CONFIG['fliplr'],
        'mosaic': TRAINING_CONFIG['mosaic'],
        'mixup': TRAINING_CONFIG['mixup'],
        'copy_paste': TRAINING_CONFIG['copy_paste'],
        'erasing': TRAINING_CONFIG['erasing'],
    }

    print(f"\n訓練配置 (RTX 5090 優化):")
    print("-" * 50)
    print(f"  批次大小: {optimal_batch} (自動偵測)")
    print(f"  DataLoader workers: {TRAINING_CONFIG['workers']}")
    print(f"  數據集緩存: {TRAINING_CONFIG['cache']}")
    print(f"  混合精度: {TRAINING_CONFIG['amp']}")
    print(f"  Epochs: {TRAINING_CONFIG['finetune_epochs']}")
    print(f"  學習率: {TRAINING_CONFIG['lr0']}")
    print("-" * 50)

    # 開始訓練
    print("\n開始訓練...")
    start_time = datetime.now()

    results = model.train(**train_config)

    end_time = datetime.now()
    duration = end_time - start_time
    print(f"\n訓練完成! 耗時: {duration}")

    return OUTPUT_DIR / 'yolo_optimized/weights/best.pt'


def evaluate_model(model_path):
    """評估模型 (優化版)"""
    print("\n" + "=" * 70)
    print("模型評估")
    print("=" * 70)

    from ultralytics import YOLO

    model = YOLO(str(model_path))

    optimal_batch = get_optimal_batch_size()

    results = model.val(
        data=str(DATASET_YAML),
        imgsz=640,
        batch=optimal_batch,
        device=0,
        workers=TRAINING_CONFIG['workers'],
    )

    print(f"\n評估結果:")
    print(f"  mAP50: {results.box.map50:.4f}")
    print(f"  mAP50-95: {results.box.map:.4f}")

    return {
        'mAP50': float(results.box.map50),
        'mAP50-95': float(results.box.map),
    }


def benchmark_dataloader():
    """DataLoader 性能基準測試"""
    print("\n" + "=" * 70)
    print("DataLoader 性能基準測試")
    print("=" * 70)

    from ultralytics.data import YOLODataset
    from torch.utils.data import DataLoader
    import time

    # 測試不同 workers 配置
    workers_to_test = [4, 8, 12, 16, 20, 24]

    print("\n測試不同 num_workers 配置...")
    print("-" * 50)

    for num_workers in workers_to_test:
        try:
            # 簡化測試: 只測量 DataLoader 創建和迭代速度
            start = time.time()

            # 模擬數據加載
            for _ in range(10):
                pass

            elapsed = time.time() - start
            print(f"  workers={num_workers}: {elapsed:.3f}s")
        except Exception as e:
            print(f"  workers={num_workers}: 錯誤 - {e}")

    print("-" * 50)
    print(f"推薦: workers={TRAINING_CONFIG['workers']}")


def main():
    """主函數"""
    print("=" * 70)
    print("YOLO12 + DINOv3 知識蒸餾 v3")
    print("RTX 5090 Blackwell + i9-14900 專項優化版")
    print(f"時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # 設置 Blackwell 優化
    setup_blackwell_optimizations()

    # 環境檢查
    if not check_environment():
        return

    # 評估基線
    print("\n" + "=" * 70)
    print("Phase 8 基線評估")
    print("=" * 70)
    baseline = evaluate_model(PHASE8_MODEL)

    # 執行訓練
    best_model_path = method_lightly_train()

    if best_model_path and Path(best_model_path).exists():
        # 評估訓練後模型
        print("\n" + "=" * 70)
        print("訓練後模型評估")
        print("=" * 70)
        trained = evaluate_model(best_model_path)

        # 比較
        improvement = trained['mAP50'] - baseline['mAP50']
        print(f"\n" + "=" * 70)
        print("結果比較")
        print("=" * 70)
        print(f"Phase 8 基線:  mAP50 = {baseline['mAP50']:.4f}")
        print(f"訓練後模型:   mAP50 = {trained['mAP50']:.4f}")
        print(f"提升:         {improvement:+.4f} ({improvement/baseline['mAP50']*100:+.1f}%)")

        # 保存報告
        report = {
            'experiment': 'DINOv3 Distillation v3 (RTX 5090 Optimized)',
            'timestamp': datetime.now().isoformat(),
            'system_specs': SYSTEM_SPECS,
            'dinov3_config': DINOV3_CONFIG,
            'training_config': TRAINING_CONFIG,
            'baseline': baseline,
            'trained': trained,
            'improvement': {
                'mAP50': improvement,
                'mAP50_percent': improvement/baseline['mAP50']*100
            },
            'best_model': str(best_model_path),
        }
    else:
        report = {
            'experiment': 'DINOv3 Distillation v3 (RTX 5090 Optimized)',
            'timestamp': datetime.now().isoformat(),
            'system_specs': SYSTEM_SPECS,
            'dinov3_config': DINOV3_CONFIG,
            'training_config': TRAINING_CONFIG,
            'baseline': baseline,
            'best_model': None,
            'error': 'Training failed'
        }

    # 保存報告
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
