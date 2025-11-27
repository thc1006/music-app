#!/usr/bin/env python3
"""
OMR Layout Analysis 數據集轉換腳本
注意: 此數據集主要包含 staff, grand staff, system, measure 等標註
並不直接包含 barline 標註，但可從 measure 邊界推導 barline 位置
"""
import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import yaml
from tqdm import tqdm
import cv2
import argparse


class OMRLayoutConverter:
    """OMR Layout Analysis 數據集轉換器"""

    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 創建輸出目錄結構
        (self.output_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)

        # 類別映射（我們的標準）
        self.class_mapping = {
            'barline': 23,
            'barline_double': 24,
            'barline_final': 25,
            'barline_repeat': 26
        }

        self.stats = {
            'total_images': 0,
            'train_images': 0,
            'val_images': 0,
            'barlines_extracted': 0,
            'skipped': 0
        }

    def find_dataset_files(self) -> Dict[str, List[Path]]:
        """查找數據集文件"""
        print("搜索數據集文件...")

        # OMR Layout Analysis 使用 YOLO 格式
        # 通常在 datasets-release 目錄下
        dataset_paths = {
            'images': [],
            'labels': []
        }

        # 查找所有圖片
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            dataset_paths['images'].extend(self.input_dir.rglob(ext))

        # 查找所有標註文件
        dataset_paths['labels'].extend(self.input_dir.rglob('*.txt'))

        print(f"找到 {len(dataset_paths['images'])} 張圖片")
        print(f"找到 {len(dataset_paths['labels'])} 個標註文件")

        return dataset_paths

    def extract_barlines_from_measures(
        self,
        annotations: List[str],
        img_width: int,
        img_height: int
    ) -> List[str]:
        """
        從 measure 標註提取 barline

        策略：
        1. system_measure (class 3) 和 stave_measure (class 4) 的左右邊界即為 barline
        2. 提取邊界位置並創建細長的 barline bbox
        """
        barlines = []

        for ann in annotations:
            parts = ann.strip().split()
            if len(parts) != 5:
                continue

            class_id, x_center, y_center, width, height = map(float, parts)

            # 只處理 measure 相關的類別
            # 根據 OMR Layout Analysis 的類別定義:
            # 0: staves, 1: grand_staves, 2: systems, 3: system_measures, 4: stave_measures
            if class_id not in [3, 4]:
                continue

            # 計算 measure 的左右邊界
            left_x = x_center - width / 2
            right_x = x_center + width / 2

            # 創建 barline (細長的垂直線)
            barline_width = 0.005  # 相對寬度，約占圖片的 0.5%

            # 左側 barline (起始線)
            left_barline = f"{self.class_mapping['barline']} {left_x:.6f} {y_center:.6f} {barline_width:.6f} {height:.6f}"
            barlines.append(left_barline)

            # 右側 barline (結束線)
            # 如果是最後一個 measure，可能是 final barline
            # 這裡簡化處理，都當作普通 barline
            right_barline = f"{self.class_mapping['barline']} {right_x:.6f} {y_center:.6f} {barline_width:.6f} {height:.6f}"
            barlines.append(right_barline)

        return barlines

    def parse_yolo_format(self, label_path: Path) -> List[str]:
        """解析 YOLO 格式標註"""
        try:
            with open(label_path, 'r') as f:
                return f.readlines()
        except Exception as e:
            print(f"讀取標註失敗 {label_path}: {e}")
            return []

    def convert_dataset(self, val_split: float = 0.15):
        """轉換整個數據集"""
        print("\n開始轉換 OMR Layout Analysis 數據集")
        print("="*60)

        # 查找數據集文件
        dataset_files = self.find_dataset_files()

        if not dataset_files['images']:
            print("✗ 未找到圖片文件")
            return False

        # 建立圖片到標註的映射
        image_label_pairs = []
        for img_path in dataset_files['images']:
            # 查找對應的標註文件
            label_path = img_path.with_suffix('.txt')
            if label_path.exists():
                image_label_pairs.append((img_path, label_path))

        print(f"找到 {len(image_label_pairs)} 對圖片-標註配對")

        if not image_label_pairs:
            print("✗ 未找到有效的圖片-標註配對")
            return False

        # 分割訓練集和驗證集
        import random
        random.shuffle(image_label_pairs)
        split_idx = int(len(image_label_pairs) * (1 - val_split))
        train_pairs = image_label_pairs[:split_idx]
        val_pairs = image_label_pairs[split_idx:]

        print(f"訓練集: {len(train_pairs)} 張")
        print(f"驗證集: {len(val_pairs)} 張")

        # 處理訓練集
        self._process_split(train_pairs, 'train')

        # 處理驗證集
        self._process_split(val_pairs, 'val')

        # 保存統計信息
        self._save_stats()

        # 創建 data.yaml
        self._create_data_yaml()

        return True

    def _process_split(self, pairs: List[Tuple[Path, Path]], split: str):
        """處理單個數據分割"""
        print(f"\n處理 {split} 集...")

        barlines_count = 0

        for img_path, label_path in tqdm(pairs, desc=f"Converting {split}"):
            # 讀取圖片以獲取尺寸
            img = cv2.imread(str(img_path))
            if img is None:
                self.stats['skipped'] += 1
                continue

            img_height, img_width = img.shape[:2]

            # 解析原始標註
            annotations = self.parse_yolo_format(label_path)

            # 提取 barlines
            barlines = self.extract_barlines_from_measures(
                annotations, img_width, img_height
            )

            if not barlines:
                self.stats['skipped'] += 1
                continue

            # 保存圖片
            dest_img_path = self.output_dir / "images" / split / img_path.name
            shutil.copy2(img_path, dest_img_path)

            # 保存標註
            dest_label_path = self.output_dir / "labels" / split / label_path.name
            with open(dest_label_path, 'w') as f:
                f.write('\n'.join(barlines))

            barlines_count += len(barlines)
            self.stats['total_images'] += 1

            if split == 'train':
                self.stats['train_images'] += 1
            else:
                self.stats['val_images'] += 1

        self.stats['barlines_extracted'] += barlines_count
        print(f"{split} 集提取了 {barlines_count} 個 barlines")

    def _save_stats(self):
        """保存統計信息"""
        stats_path = self.output_dir / "conversion_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
        print(f"\n統計信息已保存: {stats_path}")

    def _create_data_yaml(self):
        """創建 YOLO 數據集配置文件"""
        data_yaml = {
            'path': str(self.output_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'nc': 4,  # barline, barline_double, barline_final, barline_repeat
            'names': {
                0: 'barline',
                1: 'barline_double',
                2: 'barline_final',
                3: 'barline_repeat'
            },
            'original_classes': self.class_mapping,
            'source': 'OMR Layout Analysis (derived from measures)',
            'note': 'Barlines extracted from measure boundaries'
        }

        yaml_path = self.output_dir / "data.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)

        print(f"YOLO 配置已創建: {yaml_path}")

    def print_summary(self):
        """打印轉換總結"""
        print("\n" + "="*60)
        print("轉換總結 - OMR Layout Analysis")
        print("="*60)
        print(f"總圖片數: {self.stats['total_images']}")
        print(f"訓練集: {self.stats['train_images']}")
        print(f"驗證集: {self.stats['val_images']}")
        print(f"提取的 barlines: {self.stats['barlines_extracted']}")
        print(f"跳過: {self.stats['skipped']}")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description='轉換 OMR Layout Analysis 數據集')
    parser.add_argument(
        '--input',
        type=str,
        default='/home/thc1006/dev/music-app/training/datasets/external_barlines/omr_layout/datasets-release',
        help='輸入目錄（解壓後的數據集）'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='/home/thc1006/dev/music-app/training/datasets/external_barlines/omr_layout/converted',
        help='輸出目錄'
    )
    parser.add_argument(
        '--val-split',
        type=float,
        default=0.15,
        help='驗證集比例 (預設: 0.15)'
    )

    args = parser.parse_args()

    print("OMR Layout Analysis 轉換器")
    print(f"輸入: {args.input}")
    print(f"輸出: {args.output}")

    converter = OMRLayoutConverter(args.input, args.output)
    success = converter.convert_dataset(args.val_split)

    if success:
        converter.print_summary()
        print("\n✓ 轉換完成！")
        return 0
    else:
        print("\n✗ 轉換失敗")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
