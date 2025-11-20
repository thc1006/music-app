#!/usr/bin/env python3
"""
YOLO12 四部和聲樂譜辨識訓練腳本

作者: thc1006 + Claude
日期: 2025-11-20
硬體需求: RTX 5060 (8GB VRAM)
預估時間: 200-250 epochs × 8-10 hours = 約 2-3 天

訓練流程:
1. 環境檢查（GPU、CUDA、資料集）
2. 載入 YOLO12s 或 YOLO12n 預訓練模型
3. 訓練（含資料增強、Early stopping）
4. 驗證與指標評估
5. 匯出最佳模型

使用方式:
    python yolo12_train.py --model yolo12s
    python yolo12_train.py --model yolo12n --batch 24
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import yaml

try:
    from ultralytics import YOLO
    import torch
except ImportError:
    print("錯誤: 請先安裝依賴套件")
    print("執行: pip install -r requirements-train.txt")
    sys.exit(1)


# ============= 預設配置 =============

DEFAULT_CONFIG = {
    # 訓練超參數
    'epochs': 250,
    'batch': 16,
    'imgsz': 640,
    'patience': 50,

    # 學習率
    'lr0': 0.01,
    'lrf': 0.01,
    'optimizer': 'AdamW',
    'weight_decay': 0.0005,

    # 資料增強
    'degrees': 5.0,
    'translate': 0.1,
    'scale': 0.2,
    'shear': 2.0,
    'perspective': 0.0001,
    'hsv_h': 0.015,
    'hsv_s': 0.5,
    'hsv_v': 0.4,
    'blur': 0.001,
    'mosaic': 0.5,
    'mixup': 0.1,
    'flipud': 0.0,
    'fliplr': 0.0,

    # 硬體
    'device': 0,
    'workers': 8,

    # 輸出
    'project': 'harmony_omr',
    'save_period': 10,
    'verbose': True,
    'seed': 42,
}


# ============= 輔助函數 =============

def check_environment():
    """檢查訓練環境"""
    print("=" * 60)
    print("環境檢查")
    print("=" * 60)

    # Python 版本
    print(f"Python 版本: {sys.version.split()[0]}")

    # PyTorch
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU 名稱: {torch.cuda.get_device_name(0)}")
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU 記憶體: {gpu_memory:.2f} GB")

        if gpu_memory < 7.5:
            print("⚠️  警告: GPU 記憶體不足 8GB，可能需要降低 batch size")
    else:
        print("❌ 錯誤: 未偵測到 CUDA GPU")
        print("請確認:")
        print("  1. NVIDIA 驅動程式已安裝")
        print("  2. CUDA toolkit 已安裝")
        print("  3. PyTorch CUDA 版本已安裝")
        sys.exit(1)

    print()


def load_dataset_config(yaml_path: str):
    """載入並驗證資料集配置"""
    print("=" * 60)
    print("資料集配置")
    print("=" * 60)

    yaml_file = Path(yaml_path)
    if not yaml_file.exists():
        print(f"❌ 錯誤: 找不到資料集配置檔案: {yaml_path}")
        sys.exit(1)

    with open(yaml_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 驗證必要欄位
    required_fields = ['path', 'train', 'val', 'nc', 'names']
    for field in required_fields:
        if field not in config:
            print(f"❌ 錯誤: 資料集配置缺少欄位: {field}")
            sys.exit(1)

    # 檢查資料集目錄是否存在
    dataset_root = Path(config['path'])
    if not dataset_root.exists():
        print(f"⚠️  警告: 資料集根目錄不存在: {dataset_root}")
        print("請確認已執行資料集準備腳本")

    print(f"資料集根目錄: {config['path']}")
    print(f"訓練集: {config['train']}")
    print(f"驗證集: {config['val']}")
    print(f"類別數: {config['nc']}")
    print(f"類別名稱: {config['names'][:5]}... (共 {len(config['names'])} 類)")
    print()

    return config


def create_run_name(model_variant: str) -> str:
    """建立訓練執行名稱"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{model_variant}_{timestamp}"


def print_training_config(args, config):
    """印出訓練配置"""
    print("=" * 60)
    print("訓練配置")
    print("=" * 60)
    print(f"模型變體: {args.model.upper()}")
    print(f"專案名稱: {config['project']}")
    print(f"執行名稱: {args.run_name}")
    print(f"Batch Size: {args.batch}")
    print(f"Epochs: {args.epochs}")
    print(f"圖片大小: {args.imgsz}")
    print(f"初始學習率: {config['lr0']}")
    print(f"優化器: {config['optimizer']}")
    print(f"GPU Device: {config['device']}")
    print(f"Workers: {config['workers']}")
    print()


# ============= 主訓練流程 =============

def train_yolo12(args):
    """訓練 YOLO12 模型"""

    # 1. 環境檢查
    check_environment()

    # 2. 載入資料集配置
    dataset_config = load_dataset_config(args.data)

    # 3. 建立執行名稱
    args.run_name = create_run_name(args.model)

    # 4. 合併配置
    train_config = DEFAULT_CONFIG.copy()
    if args.batch:
        train_config['batch'] = args.batch
    if args.epochs:
        train_config['epochs'] = args.epochs
    if args.imgsz:
        train_config['imgsz'] = args.imgsz

    # 5. 印出配置
    print_training_config(args, train_config)

    # 6. 載入預訓練模型
    print("=" * 60)
    print(f"載入 {args.model.upper()} 預訓練模型")
    print("=" * 60)

    model_path = f"{args.model}.pt"
    try:
        model = YOLO(model_path)
        print(f"✅ 模型載入成功: {model_path}")
    except Exception as e:
        print(f"❌ 錯誤: 無法載入模型 {model_path}")
        print(f"   {e}")
        print("\n請確認:")
        print(f"  1. {model_path} 檔案存在")
        print("  2. Ultralytics 版本 >= 8.3.0")
        sys.exit(1)

    print()

    # 7. 開始訓練
    print("=" * 60)
    print("開始訓練")
    print("=" * 60)
    print("訓練過程中可以按 Ctrl+C 中斷")
    print("中斷後可從最後一個 checkpoint 繼續訓練")
    print()

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

            # 優化器
            optimizer=train_config['optimizer'],
            weight_decay=train_config['weight_decay'],

            # Early stopping
            patience=train_config['patience'],

            # 資料增強
            degrees=train_config['degrees'],
            translate=train_config['translate'],
            scale=train_config['scale'],
            shear=train_config['shear'],
            perspective=train_config['perspective'],
            hsv_h=train_config['hsv_h'],
            hsv_s=train_config['hsv_s'],
            hsv_v=train_config['hsv_v'],
            blur=train_config['blur'],
            mosaic=train_config['mosaic'],
            mixup=train_config['mixup'],
            flipud=train_config['flipud'],
            fliplr=train_config['fliplr'],

            # 硬體配置
            device=train_config['device'],
            workers=train_config['workers'],

            # 輸出配置
            project=train_config['project'],
            name=args.run_name,
            exist_ok=False,

            # 儲存設定
            save=True,
            save_period=train_config['save_period'],

            # 驗證設定
            val=True,

            # 其他
            verbose=train_config['verbose'],
            seed=train_config['seed'],
        )

    except KeyboardInterrupt:
        print("\n\n⚠️  訓練已中斷")
        print(f"checkpoint 已儲存於: {train_config['project']}/{args.run_name}/weights/last.pt")
        print("可使用以下指令繼續訓練:")
        print(f"  yolo train resume model={train_config['project']}/{args.run_name}/weights/last.pt")
        sys.exit(0)

    # 8. 印出訓練結果
    print("\n" + "=" * 60)
    print("訓練完成")
    print("=" * 60)

    best_model_path = f"{train_config['project']}/{args.run_name}/weights/best.pt"
    print(f"最佳模型: {best_model_path}")

    # 取得指標
    try:
        final_metrics = results.results_dict
        map50 = final_metrics.get('metrics/mAP50(B)', 0)
        map50_95 = final_metrics.get('metrics/mAP50-95(B)', 0)

        print(f"最終 mAP@0.5: {map50:.4f}")
        print(f"最終 mAP@0.5:0.95: {map50_95:.4f}")
    except:
        print("（無法取得最終指標）")

    print()

    # 9. 驗證最佳模型
    print("=" * 60)
    print("驗證最佳模型")
    print("=" * 60)

    best_model = YOLO(best_model_path)
    metrics = best_model.val(
        data=args.data,
        imgsz=train_config['imgsz'],
        batch=train_config['batch'],
        device=train_config['device'],
    )

    print(f"驗證 mAP@0.5: {metrics.box.map50:.4f}")
    print(f"驗證 mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"驗證 Precision: {metrics.box.mp:.4f}")
    print(f"驗證 Recall: {metrics.box.mr:.4f}")

    print("\n" + "=" * 60)
    print("全部完成")
    print("=" * 60)
    print("下一步:")
    print(f"  1. 檢查訓練曲線: {train_config['project']}/{args.run_name}/")
    print(f"  2. 執行模型匯出: python export_models.py --model {best_model_path}")
    print()


# ============= 命令列介面 =============

def parse_args():
    """解析命令列參數"""
    parser = argparse.ArgumentParser(
        description='YOLO12 四部和聲 OMR 訓練腳本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  # 訓練 YOLO12s (推薦)
  python yolo12_train.py --model yolo12s

  # 訓練 YOLO12n (輕量級)
  python yolo12_train.py --model yolo12n --batch 24

  # 自訂 epochs
  python yolo12_train.py --model yolo12s --epochs 200
        """
    )

    parser.add_argument(
        '--model',
        type=str,
        choices=['yolo12s', 'yolo12n'],
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
        help=f'Batch size (預設: {DEFAULT_CONFIG["batch"]}，RTX 5060 建議 12-16)'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        help=f'訓練 epochs (預設: {DEFAULT_CONFIG["epochs"]})'
    )

    parser.add_argument(
        '--imgsz',
        type=int,
        help=f'圖片大小 (預設: {DEFAULT_CONFIG["imgsz"]})'
    )

    return parser.parse_args()


# ============= 主程式 =============

def main():
    """主程式進入點"""
    args = parse_args()
    train_yolo12(args)


if __name__ == '__main__':
    main()
