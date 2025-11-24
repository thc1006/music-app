#!/usr/bin/env python3
"""
🎯 Phase 3: Copy-Paste 增強策略
================================

目標：無需 LilyPond，直接從現有訓練數據增強稀有類別

原理：
1. 從現有訓練數據中提取稀有類別的符號區域 (crop)
2. 將這些符號貼到其他樂譜圖片的適當位置
3. 自動更新標註

優勢：
- 不需要外部工具
- 符號真實（來自真實數據）
- 標註自動正確

使用方式：
    # 1. 提取稀有類別符號
    python phase3_copy_paste_augmentation.py --extract-symbols

    # 2. 生成增強數據
    python phase3_copy_paste_augmentation.py --augment-all

    # 3. 準備訓練數據集
    python phase3_copy_paste_augmentation.py --prepare-dataset
"""

import os
import sys
import json
import random
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import argparse

try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("⚠️ OpenCV 未安裝，請執行: pip install opencv-python")


# ============== 瓶頸類別配置 ==============

BOTTLENECK_CONFIG = {
    # class_id: (name, target_augmented_count, min_crop_size)
    16: ("accidental_double_sharp", 5000, 20),
    17: ("accidental_double_flat", 5000, 20),
    24: ("barline_double", 3000, 15),
    31: ("dynamic_loud", 5000, 25),
    29: ("fermata", 3000, 25),
    25: ("barline_final", 2000, 15),
    5:  ("flag_16th", 4000, 15),
    6:  ("flag_32nd", 4000, 15),
    15: ("accidental_natural", 3000, 20),
    12: ("clef_tenor", 3000, 30),
    30: ("dynamic_soft", 2000, 25),
    8:  ("tie", 2000, 30),
    23: ("barline", 2000, 10),
    7:  ("augmentation_dot", 2000, 8),
}


@dataclass
class SymbolCrop:
    """提取的符號裁剪"""
    class_id: int
    class_name: str
    source_image: str
    bbox_normalized: Tuple[float, float, float, float]  # x_center, y_center, w, h
    crop_path: Optional[Path] = None


class CopyPasteAugmentor:
    """Copy-Paste 增強器"""

    def __init__(
        self,
        source_dataset: Path,
        output_dir: Path,
        crops_dir: Optional[Path] = None
    ):
        self.source_dataset = Path(source_dataset)
        self.output_dir = Path(output_dir)
        self.crops_dir = crops_dir or (output_dir / "symbol_crops")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.crops_dir.mkdir(parents=True, exist_ok=True)

        # 符號庫：class_id -> List[SymbolCrop]
        self.symbol_bank: Dict[int, List[SymbolCrop]] = defaultdict(list)

        # 統計
        self.stats = defaultdict(int)

    def load_yolo_label(self, label_path: Path) -> List[Tuple[int, float, float, float, float]]:
        """載入 YOLO 格式標註"""
        labels = []
        if not label_path.exists():
            return labels

        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    labels.append((class_id, x_center, y_center, width, height))

        return labels

    def save_yolo_label(
        self,
        label_path: Path,
        labels: List[Tuple[int, float, float, float, float]]
    ):
        """保存 YOLO 格式標註"""
        with open(label_path, 'w') as f:
            for class_id, x, y, w, h in labels:
                f.write(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

    def extract_symbol_crops(self, target_classes: Optional[List[int]] = None):
        """
        從訓練數據中提取稀有類別的符號裁剪

        Args:
            target_classes: 目標類別 ID 列表，None 表示所有瓶頸類別
        """
        if not HAS_CV2:
            print("❌ 需要 OpenCV: pip install opencv-python")
            return

        if target_classes is None:
            target_classes = list(BOTTLENECK_CONFIG.keys())

        print("\n" + "="*60)
        print("提取稀有類別符號")
        print("="*60)

        images_dir = self.source_dataset / "train" / "images"
        labels_dir = self.source_dataset / "train" / "labels"

        if not images_dir.exists():
            print(f"❌ 圖片目錄不存在: {images_dir}")
            return

        # 為每個目標類別創建子目錄
        for class_id in target_classes:
            class_name = BOTTLENECK_CONFIG[class_id][0]
            (self.crops_dir / class_name).mkdir(exist_ok=True)

        # 遍歷所有訓練圖片
        image_files = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg"))
        print(f"掃描 {len(image_files)} 張訓練圖片...")

        for img_path in image_files:
            label_path = labels_dir / f"{img_path.stem}.txt"
            labels = self.load_yolo_label(label_path)

            if not labels:
                continue

            # 過濾出目標類別
            target_labels = [(l, idx) for idx, l in enumerate(labels) if l[0] in target_classes]

            if not target_labels:
                continue

            # 讀取圖片
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            h, w = img.shape[:2]

            for (class_id, x_center, y_center, box_w, box_h), idx in target_labels:
                class_name, _, min_size = BOTTLENECK_CONFIG[class_id]

                # 計算像素座標
                x1 = int((x_center - box_w / 2) * w)
                y1 = int((y_center - box_h / 2) * h)
                x2 = int((x_center + box_w / 2) * w)
                y2 = int((y_center + box_h / 2) * h)

                # 確保在圖片範圍內
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)

                # 檢查大小
                crop_w = x2 - x1
                crop_h = y2 - y1

                if crop_w < min_size or crop_h < min_size:
                    continue

                # 裁剪符號
                crop = img[y1:y2, x1:x2]

                if crop.size == 0:
                    continue

                # 保存裁剪
                crop_filename = f"{img_path.stem}_{idx}_{class_id}.png"
                crop_path = self.crops_dir / class_name / crop_filename

                cv2.imwrite(str(crop_path), crop)

                # 記錄到符號庫
                symbol = SymbolCrop(
                    class_id=class_id,
                    class_name=class_name,
                    source_image=str(img_path),
                    bbox_normalized=(x_center, y_center, box_w, box_h),
                    crop_path=crop_path
                )
                self.symbol_bank[class_id].append(symbol)
                self.stats[class_id] += 1

        # 打印統計
        print("\n符號提取統計:")
        print("-" * 50)
        for class_id in sorted(target_classes):
            class_name = BOTTLENECK_CONFIG[class_id][0]
            count = self.stats[class_id]
            print(f"  Class {class_id:2d} ({class_name:25s}): {count:5d} 個符號")

        # 保存符號庫索引
        self.save_symbol_bank_index()

    def save_symbol_bank_index(self):
        """保存符號庫索引"""
        index = {}
        for class_id, symbols in self.symbol_bank.items():
            index[class_id] = [
                {
                    "class_name": s.class_name,
                    "source_image": s.source_image,
                    "bbox_normalized": s.bbox_normalized,
                    "crop_path": str(s.crop_path) if s.crop_path else None
                }
                for s in symbols
            ]

        index_path = self.crops_dir / "symbol_bank_index.json"
        with open(index_path, 'w') as f:
            json.dump(index, f, indent=2)

        print(f"\n✅ 符號庫索引已保存: {index_path}")

    def load_symbol_bank_index(self):
        """載入符號庫索引"""
        index_path = self.crops_dir / "symbol_bank_index.json"

        if not index_path.exists():
            print(f"⚠️ 符號庫索引不存在: {index_path}")
            print("   請先執行: --extract-symbols")
            return False

        with open(index_path, 'r') as f:
            index = json.load(f)

        for class_id_str, symbols in index.items():
            class_id = int(class_id_str)
            for s in symbols:
                crop_path = Path(s["crop_path"]) if s["crop_path"] else None
                if crop_path and crop_path.exists():
                    symbol = SymbolCrop(
                        class_id=class_id,
                        class_name=s["class_name"],
                        source_image=s["source_image"],
                        bbox_normalized=tuple(s["bbox_normalized"]),
                        crop_path=crop_path
                    )
                    self.symbol_bank[class_id].append(symbol)

        print(f"✅ 載入符號庫: {sum(len(v) for v in self.symbol_bank.values())} 個符號")
        return True

    def paste_symbol(
        self,
        base_img: np.ndarray,
        symbol_crop: np.ndarray,
        target_position: Tuple[float, float],
        scale: float = 1.0
    ) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
        """
        將符號貼到圖片上

        Args:
            base_img: 基底圖片
            symbol_crop: 符號裁剪
            target_position: 目標位置 (normalized x, y)
            scale: 縮放比例

        Returns:
            (修改後的圖片, 新的 bbox)
        """
        h, w = base_img.shape[:2]
        crop_h, crop_w = symbol_crop.shape[:2]

        # 縮放符號
        if scale != 1.0:
            new_w = int(crop_w * scale)
            new_h = int(crop_h * scale)
            symbol_crop = cv2.resize(symbol_crop, (new_w, new_h))
            crop_h, crop_w = symbol_crop.shape[:2]

        # 計算貼上位置（像素）
        target_x = int(target_position[0] * w)
        target_y = int(target_position[1] * h)

        # 計算左上角
        x1 = target_x - crop_w // 2
        y1 = target_y - crop_h // 2

        # 確保在圖片範圍內
        if x1 < 0 or y1 < 0 or x1 + crop_w > w or y1 + crop_h > h:
            # 調整位置
            x1 = max(0, min(x1, w - crop_w))
            y1 = max(0, min(y1, h - crop_h))

        # 貼上符號（簡單覆蓋，可改進為 alpha blending）
        result = base_img.copy()

        # 使用更好的混合方式
        if symbol_crop.shape[2] == 4:  # 有 alpha 通道
            alpha = symbol_crop[:, :, 3] / 255.0
            for c in range(3):
                result[y1:y1+crop_h, x1:x1+crop_w, c] = (
                    alpha * symbol_crop[:, :, c] +
                    (1 - alpha) * result[y1:y1+crop_h, x1:x1+crop_w, c]
                )
        else:
            # 簡單覆蓋，略微透明處理白色背景
            mask = cv2.cvtColor(symbol_crop, cv2.COLOR_BGR2GRAY)
            mask = (mask < 250).astype(np.float32)  # 非白色區域
            mask = cv2.GaussianBlur(mask, (3, 3), 0)

            for c in range(3):
                result[y1:y1+crop_h, x1:x1+crop_w, c] = (
                    mask * symbol_crop[:, :, c] +
                    (1 - mask) * result[y1:y1+crop_h, x1:x1+crop_w, c]
                ).astype(np.uint8)

        # 計算新的 bbox (normalized)
        new_x_center = (x1 + crop_w / 2) / w
        new_y_center = (y1 + crop_h / 2) / h
        new_w = crop_w / w
        new_h = crop_h / h

        return result, (new_x_center, new_y_center, new_w, new_h)

    def augment_image(
        self,
        image_path: Path,
        label_path: Path,
        output_image_path: Path,
        output_label_path: Path,
        target_class: int,
        num_pastes: int = 3
    ) -> bool:
        """
        對單張圖片進行 Copy-Paste 增強

        Args:
            image_path: 輸入圖片路徑
            label_path: 輸入標註路徑
            output_image_path: 輸出圖片路徑
            output_label_path: 輸出標註路徑
            target_class: 要增強的目標類別
            num_pastes: 貼上次數

        Returns:
            是否成功
        """
        if not HAS_CV2:
            return False

        # 檢查符號庫
        if target_class not in self.symbol_bank or len(self.symbol_bank[target_class]) == 0:
            return False

        # 讀取圖片
        img = cv2.imread(str(image_path))
        if img is None:
            return False

        h, w = img.shape[:2]

        # 讀取現有標註
        labels = self.load_yolo_label(label_path)

        # 複製圖片
        result_img = img.copy()
        new_labels = list(labels)

        # 貼上符號
        for _ in range(num_pastes):
            # 隨機選擇一個符號
            symbol = random.choice(self.symbol_bank[target_class])

            if symbol.crop_path is None or not symbol.crop_path.exists():
                continue

            # 讀取符號裁剪
            crop = cv2.imread(str(symbol.crop_path))
            if crop is None:
                continue

            # 隨機選擇位置（在五線譜區域）
            target_x = random.uniform(0.1, 0.9)
            target_y = random.uniform(0.2, 0.8)

            # 隨機縮放
            scale = random.uniform(0.8, 1.2)

            # 貼上
            result_img, new_bbox = self.paste_symbol(
                result_img, crop, (target_x, target_y), scale
            )

            # 添加新標註
            new_labels.append((target_class, *new_bbox))

        # 保存結果
        cv2.imwrite(str(output_image_path), result_img)
        self.save_yolo_label(output_label_path, new_labels)

        return True

    def augment_dataset(
        self,
        target_classes: Optional[List[int]] = None,
        samples_per_class: Optional[Dict[int, int]] = None
    ):
        """
        增強整個數據集

        Args:
            target_classes: 目標類別列表
            samples_per_class: 每個類別的目標增強數量
        """
        if not HAS_CV2:
            print("❌ 需要 OpenCV: pip install opencv-python")
            return

        # 載入符號庫
        if not self.symbol_bank:
            if not self.load_symbol_bank_index():
                return

        if target_classes is None:
            target_classes = list(BOTTLENECK_CONFIG.keys())

        if samples_per_class is None:
            samples_per_class = {
                class_id: BOTTLENECK_CONFIG[class_id][1]
                for class_id in target_classes
            }

        print("\n" + "="*60)
        print("Copy-Paste 數據增強")
        print("="*60)

        # 過濾有足夠符號的類別
        available_classes = [
            c for c in target_classes
            if c in self.symbol_bank and len(self.symbol_bank[c]) >= 10
        ]

        if not available_classes:
            print("❌ 沒有足夠的符號進行增強")
            print("   請先執行: --extract-symbols")
            return

        print(f"可用類別: {len(available_classes)}/{len(target_classes)}")

        # 準備輸出目錄
        output_images = self.output_dir / "augmented" / "images"
        output_labels = self.output_dir / "augmented" / "labels"
        output_images.mkdir(parents=True, exist_ok=True)
        output_labels.mkdir(parents=True, exist_ok=True)

        # 獲取基底圖片列表
        source_images = self.source_dataset / "train" / "images"
        base_images = list(source_images.glob("*.png")) + list(source_images.glob("*.jpg"))

        if not base_images:
            print(f"❌ 沒有基底圖片: {source_images}")
            return

        print(f"基底圖片: {len(base_images)} 張")

        # 為每個類別生成增強數據
        for class_id in available_classes:
            class_name = BOTTLENECK_CONFIG[class_id][0]
            target_count = samples_per_class.get(class_id, 1000)
            available_symbols = len(self.symbol_bank[class_id])

            print(f"\n處理 Class {class_id} ({class_name}):")
            print(f"  可用符號: {available_symbols}")
            print(f"  目標數量: {target_count}")

            success_count = 0

            for i in range(target_count):
                # 隨機選擇基底圖片
                base_img_path = random.choice(base_images)
                base_label_path = (
                    self.source_dataset / "train" / "labels" /
                    f"{base_img_path.stem}.txt"
                )

                # 輸出路徑
                output_name = f"aug_{class_name}_{i:05d}"
                out_img = output_images / f"{output_name}.png"
                out_lbl = output_labels / f"{output_name}.txt"

                # 增強
                num_pastes = random.randint(1, 3)
                if self.augment_image(
                    base_img_path, base_label_path,
                    out_img, out_lbl,
                    class_id, num_pastes
                ):
                    success_count += 1

                if (i + 1) % 500 == 0:
                    print(f"  進度: {i+1}/{target_count}")

            self.stats[class_id] = success_count
            print(f"  ✅ 完成: {success_count}/{target_count}")

        # 統計總結
        print("\n" + "="*60)
        print("增強統計")
        print("="*60)
        for class_id in available_classes:
            class_name = BOTTLENECK_CONFIG[class_id][0]
            count = self.stats[class_id]
            print(f"  Class {class_id:2d} ({class_name:25s}): {count:5d}")

        print(f"\n  總計: {sum(self.stats.values()):,} 張增強圖片")
        print(f"  保存於: {self.output_dir / 'augmented'}")


def prepare_phase3_dataset(
    original_dataset: Path,
    augmented_dir: Path,
    output_dir: Path
):
    """
    準備 Phase 3 混合數據集

    合併：原始訓練數據 + Copy-Paste 增強數據
    """
    print("\n" + "="*60)
    print("準備 Phase 3 數據集")
    print("="*60)

    output_dir = Path(output_dir)
    output_train_images = output_dir / "train" / "images"
    output_train_labels = output_dir / "train" / "labels"
    output_val_images = output_dir / "val" / "images"
    output_val_labels = output_dir / "val" / "labels"

    for d in [output_train_images, output_train_labels, output_val_images, output_val_labels]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. 複製原始訓練數據
    print("\n1. 複製原始訓練數據...")
    orig_train_images = original_dataset / "train" / "images"
    orig_train_labels = original_dataset / "train" / "labels"

    if orig_train_images.exists():
        for img in orig_train_images.glob("*"):
            shutil.copy2(img, output_train_images / img.name)
        for lbl in orig_train_labels.glob("*"):
            shutil.copy2(lbl, output_train_labels / lbl.name)
        print(f"   複製了 {len(list(orig_train_images.glob('*')))} 張原始訓練圖片")

    # 2. 複製原始驗證數據
    print("\n2. 複製原始驗證數據...")
    orig_val_images = original_dataset / "val" / "images"
    orig_val_labels = original_dataset / "val" / "labels"

    if orig_val_images.exists():
        for img in orig_val_images.glob("*"):
            shutil.copy2(img, output_val_images / img.name)
        for lbl in orig_val_labels.glob("*"):
            shutil.copy2(lbl, output_val_labels / lbl.name)
        print(f"   複製了 {len(list(orig_val_images.glob('*')))} 張原始驗證圖片")

    # 3. 添加增強數據到訓練集
    print("\n3. 添加增強數據...")
    aug_images = augmented_dir / "images"
    aug_labels = augmented_dir / "labels"

    if aug_images.exists():
        aug_count = 0
        for img in aug_images.glob("*"):
            shutil.copy2(img, output_train_images / img.name)
            aug_count += 1
        for lbl in aug_labels.glob("*"):
            shutil.copy2(lbl, output_train_labels / lbl.name)
        print(f"   添加了 {aug_count} 張增強圖片")

    # 4. 創建數據集配置
    print("\n4. 創建數據集配置...")

    config = f"""# Phase 3: Copy-Paste 增強數據集
path: {output_dir.absolute()}
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

    yaml_path = output_dir / "harmony_phase3.yaml"
    with open(yaml_path, 'w') as f:
        f.write(config)

    print(f"\n✅ 數據集配置: {yaml_path}")

    # 統計
    final_train = len(list(output_train_images.glob("*")))
    final_val = len(list(output_val_images.glob("*")))
    print(f"\n最終數據集統計:")
    print(f"  訓練集: {final_train:,} 張")
    print(f"  驗證集: {final_val:,} 張")

    return yaml_path


def create_phase3_train_script(output_dir: Path):
    """創建 Phase 3 訓練腳本"""

    script = '''#!/usr/bin/env python3
"""
Phase 3 訓練腳本 - Copy-Paste 增強版
=====================================

從 Phase 2 best.pt 繼續訓練，使用增強後的數據集
"""
from ultralytics import YOLO
from pathlib import Path
import os

def main():
    # 切換到訓練目錄
    os.chdir(Path(__file__).parent)

    print("="*60)
    print("Phase 3: Copy-Paste 增強訓練")
    print("="*60)

    # 載入 Phase 2 最佳模型
    model = YOLO('harmony_omr_v2_phase2/balanced_training/weights/best.pt')

    # 開始訓練
    results = model.train(
        # 數據集
        data='datasets/yolo_harmony_v2_phase3/harmony_phase3.yaml',

        # 訓練參數
        epochs=200,
        batch=16,
        imgsz=640,
        patience=50,  # 更長的耐心

        # 學習率（微調模式）
        lr0=0.0005,
        lrf=0.0001,

        # 數據增強（更溫和，因為已有 Copy-Paste）
        mosaic=0.3,
        mixup=0.1,
        copy_paste=0.0,  # 關閉，因為我們已經手動做了

        hsv_h=0.01,
        hsv_s=0.3,
        hsv_v=0.3,
        degrees=3.0,
        translate=0.05,
        scale=0.3,

        # 輸出
        project='harmony_omr_v2_phase3',
        name='copypaste_enhanced',
        exist_ok=True,

        # 其他
        cos_lr=True,
        amp=True,
        verbose=True,
        save_period=10,
    )

    print("\\n" + "="*60)
    print("Phase 3 訓練完成！")
    print("="*60)
    print(f"最佳模型: harmony_omr_v2_phase3/copypaste_enhanced/weights/best.pt")

    return results


if __name__ == '__main__':
    main()
'''

    script_path = output_dir / "yolo12_train_phase3.py"
    with open(script_path, 'w') as f:
        f.write(script)

    print(f"✅ 訓練腳本: {script_path}")
    return script_path


# ============== 主程序 ==============

def main():
    parser = argparse.ArgumentParser(description='Phase 3: Copy-Paste 增強')

    parser.add_argument('--extract-symbols', action='store_true',
                       help='從訓練數據提取稀有類別符號')
    parser.add_argument('--augment-all', action='store_true',
                       help='生成所有瓶頸類別的增強數據')
    parser.add_argument('--augment-class', type=int,
                       help='生成特定類別的增強數據')
    parser.add_argument('--count', type=int, default=1000,
                       help='生成數量')
    parser.add_argument('--prepare-dataset', action='store_true',
                       help='準備 Phase 3 混合數據集')
    parser.add_argument('--create-script', action='store_true',
                       help='創建訓練腳本')
    parser.add_argument('--full-pipeline', action='store_true',
                       help='執行完整流程：提取→增強→準備數據集')

    parser.add_argument('--source-dataset', type=str,
                       default='datasets/yolo_harmony_v2_phase2',
                       help='原始數據集路徑')
    parser.add_argument('--output-dir', type=str,
                       default='datasets/yolo_harmony_v2_phase3_copypaste',
                       help='輸出目錄')
    parser.add_argument('--status', action='store_true',
                       help='顯示瓶頸類別狀態')

    args = parser.parse_args()

    # 切換到訓練目錄
    training_dir = Path(__file__).parent
    os.chdir(training_dir)

    source_dataset = Path(args.source_dataset)
    output_dir = Path(args.output_dir)

    if args.status:
        print("\n" + "="*70)
        print("瓶頸類別配置")
        print("="*70)
        print(f"{'ID':>3} {'名稱':25s} {'目標增強數':>12} {'最小裁剪':>10}")
        print("-"*55)

        for class_id, (name, target, min_size) in sorted(BOTTLENECK_CONFIG.items()):
            print(f"{class_id:>3} {name:25s} {target:>12,} {min_size:>10}")

        return

    if args.extract_symbols:
        augmentor = CopyPasteAugmentor(source_dataset, output_dir)
        augmentor.extract_symbol_crops()
        return

    if args.augment_all:
        augmentor = CopyPasteAugmentor(source_dataset, output_dir)
        augmentor.augment_dataset()
        return

    if args.augment_class is not None:
        augmentor = CopyPasteAugmentor(source_dataset, output_dir)
        augmentor.augment_dataset(
            target_classes=[args.augment_class],
            samples_per_class={args.augment_class: args.count}
        )
        return

    if args.prepare_dataset:
        augmented_dir = output_dir / "augmented"
        final_dataset = Path("datasets/yolo_harmony_v2_phase3")
        prepare_phase3_dataset(source_dataset, augmented_dir, final_dataset)
        create_phase3_train_script(training_dir)
        return

    if args.create_script:
        create_phase3_train_script(training_dir)
        return

    if args.full_pipeline:
        print("\n" + "="*70)
        print("Phase 3 完整流程")
        print("="*70)

        # Step 1: 提取符號
        print("\n[Step 1/3] 提取稀有類別符號...")
        augmentor = CopyPasteAugmentor(source_dataset, output_dir)
        augmentor.extract_symbol_crops()

        # Step 2: 生成增強數據
        print("\n[Step 2/3] 生成增強數據...")
        augmentor.augment_dataset()

        # Step 3: 準備數據集
        print("\n[Step 3/3] 準備混合數據集...")
        augmented_dir = output_dir / "augmented"
        final_dataset = Path("datasets/yolo_harmony_v2_phase3")
        prepare_phase3_dataset(source_dataset, augmented_dir, final_dataset)

        # 創建訓練腳本
        create_phase3_train_script(training_dir)

        print("\n" + "="*70)
        print("Phase 3 準備完成！")
        print("="*70)
        print("""
下一步：開始訓練
    source venv_yolo12/bin/activate
    python yolo12_train_phase3.py
""")
        return

    # 默認：顯示幫助
    parser.print_help()
    print("\n" + "="*60)
    print("Phase 3 Copy-Paste 增強工作流程")
    print("="*60)
    print("""
完整流程（推薦）：
    python phase3_copy_paste_augmentation.py --full-pipeline

或分步執行：
    1. 提取稀有類別符號:
       python phase3_copy_paste_augmentation.py --extract-symbols

    2. 生成增強數據:
       python phase3_copy_paste_augmentation.py --augment-all

    3. 準備數據集和訓練腳本:
       python phase3_copy_paste_augmentation.py --prepare-dataset

    4. 開始訓練:
       source venv_yolo12/bin/activate
       python yolo12_train_phase3.py
""")


if __name__ == '__main__':
    main()
