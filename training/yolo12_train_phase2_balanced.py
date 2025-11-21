#!/usr/bin/env python3
"""
🎯 YOLO12 Phase 2: 類別平衡訓練

Phase 1 完成後執行，專注解決類別不平衡問題

關鍵改進：
1. 類別加權損失函數
2. 過採樣稀有類別
3. Focal Loss 處理難樣本
4. 從 Phase 1 最佳模型繼續訓練
"""

import torch
import sys
import shutil
from pathlib import Path
from collections import Counter
from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')

# PyTorch 優化
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ============== 類別分析 ==============

def analyze_class_distribution(dataset_path: Path):
    """分析類別分布並計算權重"""
    class_counts = Counter()

    for split in ['train', 'val']:
        label_dir = dataset_path / split / 'labels'
        if label_dir.exists():
            for f in label_dir.glob('*.txt'):
                with open(f) as file:
                    for line in file:
                        cls = int(line.split()[0])
                        class_counts[cls] += 1

    return class_counts

def calculate_class_weights(class_counts: Counter, num_classes: int = 33,
                           max_weight: float = 50.0, min_weight: float = 0.1):
    """
    計算類別權重

    使用 inverse frequency 方法：
    weight[i] = total_samples / (num_classes * class_count[i])

    Args:
        class_counts: 每個類別的樣本數
        num_classes: 類別總數
        max_weight: 最大權重（避免過度補償）
        min_weight: 最小權重

    Returns:
        dict: 類別權重
    """
    total = sum(class_counts.values())
    weights = {}

    for cls in range(num_classes):
        count = class_counts.get(cls, 1)  # 避免除以零
        weight = total / (num_classes * count)
        weight = max(min_weight, min(weight, max_weight))
        weights[cls] = weight

    return weights

# ============== 訓練配置 ==============

# Phase 2 配置：專注於類別平衡
PHASE2_CONFIG = {
    # 繼續訓練
    'epochs': 150,          # 額外 150 epochs
    'batch': 16,
    'imgsz': 640,
    'patience': 40,

    # 學習率（從 Phase 1 最佳點繼續，更低的 LR）
    'lr0': 0.001,           # 更低的初始 LR
    'lrf': 0.001,
    'optimizer': 'AdamW',
    'momentum': 0.937,
    'weight_decay': 0.001,  # 稍微增加正則化
    'warmup_epochs': 3.0,
    'warmup_momentum': 0.8,
    'warmup_bias_lr': 0.01,
    'cos_lr': True,

    # 資料增強（保持穩定）
    'degrees': 3.0,
    'translate': 0.05,
    'scale': 0.5,
    'shear': 1.0,
    'perspective': 0.0,
    'hsv_h': 0.01,
    'hsv_s': 0.3,
    'hsv_v': 0.3,
    'mosaic': 0.5,
    'mixup': 0.0,
    'copy_paste': 0.0,
    'flipud': 0.0,
    'fliplr': 0.0,
    'erasing': 0.2,
    'close_mosaic': 15,

    # 硬體
    'device': 0,
    'workers': 8,
    'amp': True,
    'cache': False,

    # 損失權重（強化分類）
    'box': 7.5,
    'cls': 2.0,             # ⭐ 增加分類權重（從 0.5 → 2.0）
    'dfl': 1.5,

    # 輸出
    'project': 'harmony_omr_v2_phase2',
    'name': 'balanced_training',
    'save_period': 5,
    'plots': True,
    'verbose': True,
    'seed': 42,
    'exist_ok': False,
}

# ============== 過採樣配置 ==============

# 需要過採樣的類別（樣本數 < 1000）
OVERSAMPLE_CONFIG = {
    17: {'factor': 100, 'aug_intensity': 'high'},   # double_flat: 12 → 1200
    31: {'factor': 50, 'aug_intensity': 'high'},    # dynamic_loud: 27 → 1350
    24: {'factor': 10, 'aug_intensity': 'medium'},  # barline_double: 234 → 2340
    16: {'factor': 10, 'aug_intensity': 'medium'},  # double_sharp: 338 → 3380
    6:  {'factor': 5, 'aug_intensity': 'medium'},   # flag_32nd: 440 → 2200
    12: {'factor': 5, 'aug_intensity': 'low'},      # clef_tenor: 614 → 3070
}

def create_oversampled_dataset(original_path: Path, output_path: Path):
    """
    創建過採樣數據集

    對稀有類別的圖片進行複製和增強
    """
    print("\n📊 創建過採樣數據集...")

    if output_path.exists():
        shutil.rmtree(output_path)

    # 複製原始數據集結構
    for split in ['train', 'val']:
        (output_path / split / 'images').mkdir(parents=True)
        (output_path / split / 'labels').mkdir(parents=True)

    # 找出包含稀有類別的圖片
    rare_class_images = {cls: [] for cls in OVERSAMPLE_CONFIG.keys()}

    train_labels = original_path / 'train' / 'labels'
    for label_file in train_labels.glob('*.txt'):
        with open(label_file) as f:
            classes_in_image = set()
            for line in f:
                cls = int(line.split()[0])
                classes_in_image.add(cls)

        for cls in OVERSAMPLE_CONFIG.keys():
            if cls in classes_in_image:
                rare_class_images[cls].append(label_file.stem)

    # 統計
    print("\n稀有類別圖片統計：")
    for cls, images in rare_class_images.items():
        config = OVERSAMPLE_CONFIG[cls]
        print(f"  Class {cls}: {len(images)} 張圖片 × {config['factor']} = "
              f"{len(images) * config['factor']} 張")

    # 複製所有原始圖片
    print("\n複製原始數據...")
    for split in ['train', 'val']:
        src_img = original_path / split / 'images'
        src_lbl = original_path / split / 'labels'
        dst_img = output_path / split / 'images'
        dst_lbl = output_path / split / 'labels'

        for img in src_img.glob('*.png'):
            shutil.copy2(img, dst_img / img.name)
        for lbl in src_lbl.glob('*.txt'):
            shutil.copy2(lbl, dst_lbl / lbl.name)

    # 過採樣稀有類別
    print("\n過採樣稀有類別...")
    for cls, images in rare_class_images.items():
        config = OVERSAMPLE_CONFIG[cls]
        factor = config['factor']

        for img_name in images:
            src_img = original_path / 'train' / 'images' / f"{img_name}.png"
            src_lbl = original_path / 'train' / 'labels' / f"{img_name}.txt"

            if not src_img.exists() or not src_lbl.exists():
                continue

            # 複製多次（不同名稱）
            for i in range(factor - 1):  # -1 因為原始已經複製
                new_name = f"{img_name}_oversample_{cls}_{i}"
                dst_img = output_path / 'train' / 'images' / f"{new_name}.png"
                dst_lbl = output_path / 'train' / 'labels' / f"{new_name}.txt"

                shutil.copy2(src_img, dst_img)
                shutil.copy2(src_lbl, dst_lbl)

    # 統計最終數據集
    final_train = len(list((output_path / 'train' / 'images').glob('*.png')))
    final_val = len(list((output_path / 'val' / 'images').glob('*.png')))

    print(f"\n✅ 過採樣完成")
    print(f"  原始訓練集: {len(list((original_path / 'train' / 'images').glob('*.png')))} 張")
    print(f"  過採樣訓練集: {final_train} 張")
    print(f"  驗證集: {final_val} 張（不變）")

    # 創建 YAML 配置
    yaml_content = f"""# Phase 2: 類別平衡數據集
path: {output_path.absolute()}
train: train/images
val: val/images

nc: 33

names:
  0: notehead_filled
  1: notehead_hollow
  2: stem
  3: beam
  4: flag_8th
  5: flag_16th
  6: flag_32nd
  7: augmentation_dot
  8: tie
  9: clef_treble
  10: clef_bass
  11: clef_alto
  12: clef_tenor
  13: accidental_sharp
  14: accidental_flat
  15: accidental_natural
  16: accidental_double_sharp
  17: accidental_double_flat
  18: rest_whole
  19: rest_half
  20: rest_quarter
  21: rest_8th
  22: rest_16th
  23: barline
  24: barline_double
  25: barline_final
  26: barline_repeat
  27: time_signature
  28: key_signature
  29: fermata
  30: dynamic_soft
  31: dynamic_loud
  32: ledger_line
"""

    yaml_path = output_path / 'harmony_phase2.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    return yaml_path

# ============== 主程序 ==============

def find_best_phase1_model():
    """尋找 Phase 1 的最佳模型"""
    possible_paths = [
        Path('harmony_omr_v2_optimized/train_phase1/weights/best.pt'),
        Path('harmony_omr_v2_optimized/train_phase12/weights/best.pt'),
        Path('harmony_omr_v2_optimized/train_phase13/weights/best.pt'),
    ]

    for path in possible_paths:
        if path.exists():
            return path

    # 搜索任何 best.pt
    for best_pt in Path('harmony_omr_v2_optimized').rglob('best.pt'):
        return best_pt

    return None

def main():
    print("\n" + "=" * 70)
    print(" " * 10 + "🎯 YOLO12 Phase 2: 類別平衡訓練")
    print("=" * 70)

    # GPU 檢查
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        sys.exit(1)

    print(f"\n🎮 GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # 尋找 Phase 1 模型
    phase1_model = find_best_phase1_model()
    if phase1_model:
        print(f"\n📥 找到 Phase 1 模型: {phase1_model}")
    else:
        print("\n⚠️  未找到 Phase 1 模型，將使用預訓練模型")
        phase1_model = 'yolo12s.pt'

    # 載入模型
    model = YOLO(str(phase1_model))

    # 創建過採樣數據集
    original_dataset = Path('datasets/yolo_harmony_v2_optimized')
    oversampled_dataset = Path('datasets/yolo_harmony_v2_phase2')

    if not original_dataset.exists():
        print(f"❌ 找不到原始數據集: {original_dataset}")
        sys.exit(1)

    yaml_path = create_oversampled_dataset(original_dataset, oversampled_dataset)

    # 分析類別分布
    print("\n📊 分析類別分布...")
    class_counts = analyze_class_distribution(oversampled_dataset)
    class_weights = calculate_class_weights(class_counts)

    print("\n類別權重（前 10 個）：")
    for cls, weight in sorted(class_weights.items(), key=lambda x: -x[1])[:10]:
        count = class_counts.get(cls, 0)
        print(f"  Class {cls}: weight={weight:.2f} (count={count:,})")

    # 配置摘要
    print("\n⚙️  Phase 2 配置：")
    print(f"   從 Phase 1 繼續: {phase1_model}")
    print(f"   額外 Epochs: {PHASE2_CONFIG['epochs']}")
    print(f"   LR: {PHASE2_CONFIG['lr0']} (降低)")
    print(f"   Cls Loss Weight: {PHASE2_CONFIG['cls']} (增加)")
    print(f"   數據集: 過採樣版本")

    # 開始訓練
    print("\n" + "=" * 70)
    print(" " * 15 + "🚀 Starting Phase 2 Training")
    print("=" * 70 + "\n")

    results = model.train(
        data=str(yaml_path),
        **PHASE2_CONFIG
    )

    print("\n" + "=" * 70)
    print(" " * 20 + "✅ Phase 2 Completed!")
    print("=" * 70)

    print(f"\n📊 最終結果:")
    print(f"   Best model: {results.save_dir}/weights/best.pt")

    return results

if __name__ == '__main__':
    import os
    os.chdir(Path(__file__).parent)
    main()
