#!/usr/bin/env python3
"""
YOLO12 四部和聲 OMR 訓練腳本 - RTX 5090 優化版

作者: thc1006 + Claude
日期: 2025-11-21
硬體: NVIDIA RTX 5090 (32GB VRAM)
預估時間: 600 epochs × 18-24 hours

🔥 RTX 5090 優化特性:
- Batch size: 64 (針對 YOLO12 記憶體需求優化)
- Workers: 16 (i9-14900 24 核心)
- Mixed precision: FP16 + TF32
- Gradient accumulation: 支援
- 根據 2025 年最新研究優化的 YOLO12 參數

⚠️ YOLO12 特性:
- 記憶體消耗比 YOLO11 高 2 倍
- 訓練可能不穩定（attention blocks）
- 需要更多 epochs (600 vs 250)
- Mosaic: 1.0, Mixup: 0.0, Scale: 0.9

使用方式:
    python yolo12_train_rtx5090.py --model yolo12s
    python yolo12_train_rtx5090.py --model yolo12n --batch 96
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import yaml
import os

try:
    from ultralytics import YOLO
    import torch
except ImportError:
    print("❌ 錯誤: 請先安裝依賴套件")
    print("執行: source venv_yolo12/bin/activate && pip install -r requirements-train.txt")
    sys.exit(1)


# ============= RTX 5090 優化配置 =============

RTX5090_CONFIG = {
    # YOLO12 訓練超參數（根據 2025 研究）
    'epochs': 600,  # YOLO12 需要更多 epochs
    'batch': 128,   # ⚡ RTX 5090 32GB → 極致 batch size
    'imgsz': 640,
    'patience': 100,  # YOLO12 訓練不穩定，需要更多耐心

    # 學習率（YOLO12 推薦）
    'lr0': 0.01,
    'lrf': 0.01,
    'optimizer': 'AdamW',
    'weight_decay': 0.0005,
    'momentum': 0.937,
    'warmup_epochs': 3.0,
    'warmup_momentum': 0.8,
    'warmup_bias_lr': 0.1,

    # YOLO12 特定資料增強（根據官方推薦）
    'degrees': 5.0,      # 旋轉
    'translate': 0.1,    # 平移
    'scale': 0.9,        # ⭐ YOLO12s 推薦 0.9（重要！）
    'shear': 2.0,        # 剪切
    'perspective': 0.0001,  # 透視
    'hsv_h': 0.015,      # 色調
    'hsv_s': 0.5,        # 飽和度
    'hsv_v': 0.4,        # 亮度
    'mosaic': 1.0,       # ⭐ YOLO12 推薦 1.0（重要！）
    'mixup': 0.0,        # ⭐ YOLO12 推薦 0.0（重要！）
    'copy_paste': 0.1,   # ⭐ YOLO12 特有
    'flipud': 0.0,       # 不翻轉（樂譜不應上下翻轉）
    'fliplr': 0.0,       # 不翻轉（樂譜不應左右翻轉）

    # RTX 5090 硬體優化
    'device': 0,
    'workers': 24,       # ⚡ i9-14900 全部 24 核心
    'amp': True,         # ⭐ 自動混合精度（FP16）
    'cache': True,       # ⭐ 快取資料集到 RAM（125GB 可用）
    'multi_scale': True, # ⚡ 多尺度訓練

    # 輸出配置
    'project': 'harmony_omr_yolo12',
    'save_period': 20,   # 每 20 epochs 儲存一次
    'plots': True,
    'verbose': True,
    'seed': 42,

    # 進階優化
    'resume': False,
    'exist_ok': False,
    'pretrained': True,
    'rect': False,       # 矩形訓練（可能影響 OMR）
    'cos_lr': True,      # Cosine learning rate schedule
    'close_mosaic': 10,  # 最後 10 epochs 關閉 mosaic
}

# YOLO12n 特定配置（更輕量）
YOLO12N_CONFIG = RTX5090_CONFIG.copy()
YOLO12N_CONFIG.update({
    'batch': 192,        # ⚡ 小模型可用更大 batch
    'epochs': 500,       # 較少 epochs
    'scale': 0.5,        # YOLO12n 推薦 0.5
})


# ============= 輔助函數 =============

def check_environment():
    """檢查訓練環境（RTX 5090 特化）"""
    print("\n" + "=" * 70)
    print(" " * 20 + "🔥 RTX 5090 訓練環境檢查")
    print("=" * 70)

    # Python 版本
    print(f"\n📦 Python 版本: {sys.version.split()[0]}")

    # PyTorch
    print(f"🔧 PyTorch 版本: {torch.__version__}")
    print(f"✅ CUDA 可用: {torch.cuda.is_available()}")
    print(f"🔢 CUDA 版本: {torch.version.cuda}")

    if not torch.cuda.is_available():
        print("\n❌ 錯誤: 未偵測到 CUDA GPU")
        print("請確認:")
        print("  1. NVIDIA 驅動程式已安裝 (需要 570+)")
        print("  2. CUDA toolkit 已安裝 (12.8)")
        print("  3. PyTorch CUDA 版本已安裝")
        sys.exit(1)

    # GPU 資訊
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    compute_capability = torch.cuda.get_device_capability(0)

    print(f"\n🎮 GPU 名稱: {gpu_name}")
    print(f"💾 GPU 記憶體: {gpu_memory:.2f} GB")
    print(f"⚡ Compute Capability: {compute_capability[0]}.{compute_capability[1]}")

    # RTX 5090 驗證
    if "5090" in gpu_name:
        print("\n🔥 偵測到 RTX 5090！已啟用極致效能配置")
        print("   - 32GB VRAM → Batch size 128 (YOLO12s) / 192 (YOLO12n)")
        print("   - GDDR7 1.79 TB/s → 啟用資料快取")
        print("   - Tensor Cores Gen 5 → 啟用 AMP (FP16)")
        print("   - i9-14900 24 核心 → 全核心資料載入")
    else:
        print(f"\n⚠️  警告: 當前 GPU 非 RTX 5090 ({gpu_name})")
        print("   配置可能需要調整")

    if gpu_memory < 30:
        print(f"\n⚠️  警告: GPU 記憶體 ({gpu_memory:.0f}GB) < 30GB")
        print("   建議降低 batch size")

    # Ultralytics 版本
    try:
        import ultralytics
        print(f"\n🚀 Ultralytics 版本: {ultralytics.__version__}")

        # 檢查是否支援 YOLO12
        if not hasattr(YOLO, '__version__') or int(ultralytics.__version__.split('.')[0]) < 8:
            print("⚠️  警告: Ultralytics 版本可能不支援 YOLO12")
            print("   建議: pip install ultralytics>=8.3.0")
    except:
        pass

    print("\n" + "=" * 70 + "\n")


def load_dataset_config(yaml_path: str):
    """載入並驗證資料集配置"""
    print("=" * 70)
    print(" " * 25 + "📂 資料集配置")
    print("=" * 70)

    yaml_file = Path(yaml_path)
    if not yaml_file.exists():
        print(f"\n❌ 錯誤: 找不到資料集配置檔案: {yaml_path}")
        print("\n請先執行:")
        print("  1. python convert_dataset.py")
        print("  2. 確認 omr_harmony.yaml 存在")
        sys.exit(1)

    with open(yaml_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 驗證必要欄位
    required_fields = ['path', 'train', 'val', 'nc', 'names']
    for field in required_fields:
        if field not in config:
            print(f"❌ 錯誤: 資料集配置缺少欄位: {field}")
            sys.exit(1)

    # 檢查資料集目錄
    dataset_root = Path(config['path'])
    train_dir = dataset_root / config['train']
    val_dir = dataset_root / config['val']

    if not dataset_root.exists():
        print(f"\n⚠️  警告: 資料集根目錄不存在: {dataset_root}")
        print("\n請執行:")
        print("  1. cd training/datasets")
        print("  2. git clone https://github.com/OMR-Research/muscima-pp.git")
        print("  3. cd ../..")
        print("  4. python training/convert_dataset.py")
        sys.exit(1)

    # 統計資料集
    train_images = list(train_dir.glob('*.png')) + list(train_dir.glob('*.jpg')) if train_dir.exists() else []
    val_images = list(val_dir.glob('*.png')) + list(val_dir.glob('*.jpg')) if val_dir.exists() else []

    print(f"\n📁 資料集根目錄: {config['path']}")
    print(f"📊 訓練集: {config['train']} ({len(train_images)} 張)")
    print(f"📊 驗證集: {config['val']} ({len(val_images)} 張)")
    print(f"🎯 類別數: {config['nc']}")

    # Handle both dict and list format for names
    if isinstance(config['names'], dict):
        names_list = [config['names'][i] for i in sorted(config['names'].keys())[:5]]
        print(f"🏷️  類別名稱: {', '.join(names_list)}... (共 {len(config['names'])} 類)")
    else:
        print(f"🏷️  類別名稱: {', '.join(config['names'][:5])}... (共 {len(config['names'])} 類)")

    if len(train_images) == 0:
        print("\n❌ 錯誤: 訓練集沒有圖片")
        sys.exit(1)

    print("\n" + "=" * 70 + "\n")
    return config


def create_run_name(model_variant: str) -> str:
    """建立訓練執行名稱"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{model_variant}_rtx5090_{timestamp}"


def print_training_config(args, config):
    """印出訓練配置"""
    print("=" * 70)
    print(" " * 20 + "⚙️  YOLO12 訓練配置（RTX 5090）")
    print("=" * 70)

    print(f"\n🎯 模型變體: {args.model.upper()}")
    print(f"📁 專案名稱: {config['project']}")
    print(f"🏷️  執行名稱: {args.run_name}")

    print(f"\n📊 訓練參數:")
    print(f"   Batch Size: {config['batch']}")
    print(f"   Epochs: {config['epochs']}")
    print(f"   Image Size: {config['imgsz']}")
    print(f"   Patience: {config['patience']}")

    print(f"\n📈 學習率:")
    print(f"   Initial LR: {config['lr0']}")
    print(f"   Final LR: {config['lrf']}")
    print(f"   Optimizer: {config['optimizer']}")
    print(f"   Cosine LR: {config['cos_lr']}")

    print(f"\n🎨 資料增強 (YOLO12 特化):")
    print(f"   Scale: {config['scale']} ⭐")
    print(f"   Mosaic: {config['mosaic']} ⭐")
    print(f"   Mixup: {config['mixup']} ⭐")
    print(f"   Copy-Paste: {config['copy_paste']} ⭐")
    print(f"   Rotation: ±{config['degrees']}°")
    print(f"   Translation: {config['translate']}")

    print(f"\n⚡ 硬體優化:")
    print(f"   GPU Device: {config['device']}")
    print(f"   Workers: {config['workers']}")
    print(f"   AMP (FP16): {config['amp']} ⭐")
    print(f"   Cache: {config['cache']} ⭐")

    print(f"\n💾 儲存:")
    print(f"   Save Period: 每 {config['save_period']} epochs")
    print(f"   Plots: {config['plots']}")

    print("\n" + "=" * 70 + "\n")


def estimate_training_time(epochs: int, model_variant: str) -> str:
    """預估訓練時間（RTX 5090）"""
    # 根據 2025 benchmarks：RTX 5090 比 4090 快 44%
    # YOLO12s 約 0.8s/iteration @ batch 64
    # 假設每 epoch ~1000 iterations

    if 'yolo12s' in model_variant.lower():
        seconds_per_epoch = 800  # ~13 分鐘
    elif 'yolo12n' in model_variant.lower():
        seconds_per_epoch = 400  # ~7 分鐘
    else:
        seconds_per_epoch = 600

    total_seconds = epochs * seconds_per_epoch
    hours = total_seconds / 3600

    return f"{hours:.1f} 小時"


# ============= 主訓練流程 =============

def train_yolo12(args):
    """訓練 YOLO12 模型（RTX 5090 優化）"""

    # 1. 環境檢查
    check_environment()

    # 2. 載入資料集配置
    dataset_config = load_dataset_config(args.data)

    # 3. 建立執行名稱
    args.run_name = create_run_name(args.model)

    # 4. 選擇配置
    if 'yolo12n' in args.model.lower():
        train_config = YOLO12N_CONFIG.copy()
        print("📦 使用 YOLO12n 配置（輕量級）\n")
    else:
        train_config = RTX5090_CONFIG.copy()
        print("📦 使用 YOLO12s 配置（標準）\n")

    # 5. 覆寫命令列參數
    if args.batch:
        train_config['batch'] = args.batch
    if args.epochs:
        train_config['epochs'] = args.epochs
    if args.imgsz:
        train_config['imgsz'] = args.imgsz

    # 6. 印出配置
    print_training_config(args, train_config)

    # 7. 預估時間
    estimated_time = estimate_training_time(train_config['epochs'], args.model)
    print(f"⏱️  預估訓練時間: {estimated_time}")
    print(f"🎯 預期完成: {datetime.now().strftime('%Y-%m-%d')} + {estimated_time}\n")

    # 8. 載入預訓練模型
    print("=" * 70)
    print(" " * 25 + "📥 載入預訓練模型")
    print("=" * 70)

    model_path = f"{args.model}.pt"
    print(f"\n正在載入: {model_path}\n")

    try:
        model = YOLO(model_path)
        print(f"✅ 模型載入成功\n")
    except Exception as e:
        print(f"❌ 錯誤: 無法載入模型 {model_path}")
        print(f"   {e}\n")
        print("請確認:")
        print(f"  1. {model_path} 檔案存在")
        print("  2. Ultralytics 版本 >= 8.3.0")
        print("  3. 網路連線正常（首次下載需要）")
        sys.exit(1)

    # 9. 開始訓練
    print("=" * 70)
    print(" " * 25 + "🚀 開始訓練 YOLO12")
    print("=" * 70)
    print("\n💡 提示:")
    print("   - 按 Ctrl+C 可中斷訓練")
    print("   - 訓練會自動儲存 checkpoint")
    print("   - 可用 tensorboard --logdir harmony_omr_yolo12/ 監控")
    print("   - 使用 nvidia-smi 監控 GPU 使用率")
    print("\n" + "=" * 70 + "\n")

    try:
        results = model.train(
            # 資料配置
            data=args.data,

            # 訓練超參數
            epochs=train_config['epochs'],
            batch=train_config['batch'],
            imgsz=train_config['imgsz'],

            # 學習率
            lr0=train_config['lr0'],
            lrf=train_config['lrf'],
            momentum=train_config['momentum'],
            weight_decay=train_config['weight_decay'],
            warmup_epochs=train_config['warmup_epochs'],
            warmup_momentum=train_config['warmup_momentum'],
            warmup_bias_lr=train_config['warmup_bias_lr'],

            # 優化器
            optimizer=train_config['optimizer'],
            cos_lr=train_config['cos_lr'],

            # Early stopping
            patience=train_config['patience'],

            # 資料增強（YOLO12 特化）
            degrees=train_config['degrees'],
            translate=train_config['translate'],
            scale=train_config['scale'],
            shear=train_config['shear'],
            perspective=train_config['perspective'],
            hsv_h=train_config['hsv_h'],
            hsv_s=train_config['hsv_s'],
            hsv_v=train_config['hsv_v'],
            mosaic=train_config['mosaic'],
            mixup=train_config['mixup'],
            copy_paste=train_config['copy_paste'],
            flipud=train_config['flipud'],
            fliplr=train_config['fliplr'],
            close_mosaic=train_config['close_mosaic'],

            # 硬體配置（RTX 5090 優化）
            device=train_config['device'],
            workers=train_config['workers'],
            amp=train_config['amp'],
            cache=train_config['cache'],

            # 輸出配置
            project=train_config['project'],
            name=args.run_name,
            exist_ok=train_config['exist_ok'],

            # 儲存設定
            save=True,
            save_period=train_config['save_period'],

            # 驗證設定
            val=True,
            plots=train_config['plots'],

            # 其他
            verbose=train_config['verbose'],
            seed=train_config['seed'],
            resume=train_config['resume'],
            pretrained=train_config['pretrained'],
            rect=train_config['rect'],
        )

    except KeyboardInterrupt:
        print("\n\n" + "=" * 70)
        print(" " * 25 + "⚠️  訓練已中斷")
        print("=" * 70)
        checkpoint_path = f"{train_config['project']}/{args.run_name}/weights/last.pt"
        print(f"\n💾 Checkpoint 已儲存於: {checkpoint_path}")
        print("\n可使用以下指令繼續訓練:")
        print(f"   yolo train resume model={checkpoint_path}\n")
        print("=" * 70 + "\n")
        sys.exit(0)

    # 10. 印出訓練結果
    print("\n" + "=" * 70)
    print(" " * 25 + "✅ 訓練完成")
    print("=" * 70)

    best_model_path = f"{train_config['project']}/{args.run_name}/weights/best.pt"
    print(f"\n🏆 最佳模型: {best_model_path}")

    # 取得指標
    try:
        final_metrics = results.results_dict
        map50 = final_metrics.get('metrics/mAP50(B)', 0)
        map50_95 = final_metrics.get('metrics/mAP50-95(B)', 0)

        print(f"\n📊 最終指標:")
        print(f"   mAP@0.5: {map50:.4f}")
        print(f"   mAP@0.5:0.95: {map50_95:.4f}")
    except:
        print("\n⚠️  無法取得最終指標")

    # 11. 驗證最佳模型
    print("\n" + "=" * 70)
    print(" " * 25 + "🔍 驗證最佳模型")
    print("=" * 70 + "\n")

    best_model = YOLO(best_model_path)
    metrics = best_model.val(
        data=args.data,
        imgsz=train_config['imgsz'],
        batch=train_config['batch'],
        device=train_config['device'],
    )

    print(f"\n📊 驗證結果:")
    print(f"   mAP@0.5: {metrics.box.map50:.4f}")
    print(f"   mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"   Precision: {metrics.box.mp:.4f}")
    print(f"   Recall: {metrics.box.mr:.4f}")

    print("\n" + "=" * 70)
    print(" " * 25 + "🎉 全部完成")
    print("=" * 70)
    print("\n下一步:")
    print(f"  1. 檢查訓練曲線: {train_config['project']}/{args.run_name}/")
    print(f"  2. 執行模型匯出: python export_models.py --model {best_model_path}")
    print("  3. 複製到 Android 專案\n")
    print("=" * 70 + "\n")


# ============= 命令列介面 =============

def parse_args():
    """解析命令列參數"""
    parser = argparse.ArgumentParser(
        description='YOLO12 四部和聲 OMR 訓練腳本 - RTX 5090 優化版',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  # 訓練 YOLO12s (推薦 - 標準模型)
  python yolo12_train_rtx5090.py --model yolo12s

  # 訓練 YOLO12n (輕量級備援)
  python yolo12_train_rtx5090.py --model yolo12n

  # 自訂 batch size (記憶體不足時)
  python yolo12_train_rtx5090.py --model yolo12s --batch 48

  # 快速測試 (50 epochs)
  python yolo12_train_rtx5090.py --model yolo12s --epochs 50

⚠️  注意:
  - YOLO12 記憶體消耗高，RTX 5090 極致配置 batch 128
  - 訓練可能不穩定，請監控 loss
  - 完整訓練需要 12-18 小時（24 核心加速）
  - 如果 OOM，降低 batch: --batch 96 或 --batch 64
        """
    )

    parser.add_argument(
        '--model',
        type=str,
        choices=['yolo12s', 'yolo12n', 'yolo12m', 'yolo12l', 'yolo12x'],
        default='yolo12s',
        help='模型變體 (預設: yolo12s)'
    )

    parser.add_argument(
        '--data',
        type=str,
        default='omr_harmony.yaml',
        help='資料集配置檔案 (預設: omr_harmony.yaml)'
    )

    parser.add_argument(
        '--batch',
        type=int,
        help=f'Batch size (預設: 128 for s, 192 for n)'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        help=f'訓練 epochs (預設: 600 for s, 500 for n)'
    )

    parser.add_argument(
        '--imgsz',
        type=int,
        help=f'圖片大小 (預設: 640)'
    )

    return parser.parse_args()


# ============= 主程式 =============

def main():
    """主程式進入點"""
    print("\n" + "=" * 70)
    print(" " * 15 + "🔥 YOLO12 RTX 5090 訓練系統啟動")
    print("=" * 70 + "\n")

    args = parse_args()
    train_yolo12(args)


if __name__ == '__main__':
    main()
