#!/usr/bin/env python3
"""
合併所有 barline 數據集
將 OMR Layout, AudioLabs, DoReMi 三個數據集合併為統一的 YOLO 格式
"""
import os
import json
import shutil
from pathlib import Path
from typing import Dict, List
import yaml
from tqdm import tqdm
import argparse
from collections import defaultdict


class BarlineDatasetMerger:
    """Barline 數據集合併器"""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 創建輸出目錄結構
        (self.output_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)

        # 類別映射（統一標準）
        self.class_mapping = {
            'barline': 0,  # 合併後重新編號為 0-3
            'barline_double': 1,
            'barline_final': 2,
            'barline_repeat': 3
        }

        # 原始類別映射（23-26）到新映射（0-3）
        self.original_to_new = {
            23: 0,  # barline
            24: 1,  # barline_double
            25: 2,  # barline_final
            26: 3   # barline_repeat
        }

        self.stats = {
            'total_images': 0,
            'train_images': 0,
            'val_images': 0,
            'total_barlines': 0,
            'datasets': {},
            'class_distribution': defaultdict(int)
        }

    def merge_datasets(
        self,
        omr_layout_dir: str = None,
        audiolabs_dir: str = None,
        doremi_dir: str = None
    ):
        """合併所有數據集"""
        print("\n開始合併 Barline 數據集")
        print("="*60)

        datasets = {}

        if omr_layout_dir:
            omr_path = Path(omr_layout_dir)
            if omr_path.exists():
                datasets['omr_layout'] = omr_path
                print(f"✓ 找到 OMR Layout: {omr_path}")
            else:
                print(f"⚠ OMR Layout 不存在: {omr_path}")

        if audiolabs_dir:
            audiolabs_path = Path(audiolabs_dir)
            if audiolabs_path.exists():
                datasets['audiolabs'] = audiolabs_path
                print(f"✓ 找到 AudioLabs: {audiolabs_path}")
            else:
                print(f"⚠ AudioLabs 不存在: {audiolabs_path}")

        if doremi_dir:
            doremi_path = Path(doremi_dir)
            if doremi_path.exists():
                datasets['doremi'] = doremi_path
                print(f"✓ 找到 DoReMi: {doremi_path}")
            else:
                print(f"⚠ DoReMi 不存在: {doremi_path}")

        if not datasets:
            print("✗ 未找到任何數據集")
            return False

        # 處理每個數據集
        for dataset_name, dataset_path in datasets.items():
            print(f"\n處理 {dataset_name}...")
            self._merge_single_dataset(dataset_name, dataset_path)

        # 保存統計信息
        self._save_stats()

        # 創建 data.yaml
        self._create_data_yaml()

        return True

    def _merge_single_dataset(self, dataset_name: str, dataset_path: Path):
        """合併單個數據集"""
        stats = {
            'train_images': 0,
            'val_images': 0,
            'barlines': 0
        }

        # 處理訓練集和驗證集
        for split in ['train', 'val']:
            image_dir = dataset_path / "images" / split
            label_dir = dataset_path / "labels" / split

            if not image_dir.exists() or not label_dir.exists():
                print(f"  ⚠ {split} 目錄不存在，跳過")
                continue

            # 獲取所有圖片
            images = list(image_dir.glob('*.png')) + \
                     list(image_dir.glob('*.jpg')) + \
                     list(image_dir.glob('*.jpeg'))

            print(f"  處理 {split} 集: {len(images)} 張圖片")

            for img_path in tqdm(images, desc=f"  {dataset_name} {split}"):
                # 查找對應的標註
                label_path = label_dir / (img_path.stem + '.txt')

                if not label_path.exists():
                    continue

                # 讀取並轉換標註
                new_labels = self._convert_labels(label_path)

                if not new_labels:
                    continue

                # 生成唯一文件名（避免衝突）
                unique_name = f"{dataset_name}_{img_path.stem}{img_path.suffix}"
                unique_label = f"{dataset_name}_{img_path.stem}.txt"

                # 複製圖片
                dest_img = self.output_dir / "images" / split / unique_name
                shutil.copy2(img_path, dest_img)

                # 保存標註
                dest_label = self.output_dir / "labels" / split / unique_label
                with open(dest_label, 'w') as f:
                    f.write('\n'.join(new_labels))

                # 更新統計
                stats[f'{split}_images'] += 1
                stats['barlines'] += len(new_labels)

                # 統計類別分佈
                for label in new_labels:
                    class_id = int(label.split()[0])
                    self.stats['class_distribution'][class_id] += 1

        # 保存數據集統計
        self.stats['datasets'][dataset_name] = stats
        self.stats['train_images'] += stats['train_images']
        self.stats['val_images'] += stats['val_images']
        self.stats['total_images'] += stats['train_images'] + stats['val_images']
        self.stats['total_barlines'] += stats['barlines']

        print(f"  完成: {stats['train_images']} train, {stats['val_images']} val, {stats['barlines']} barlines")

    def _convert_labels(self, label_path: Path) -> List[str]:
        """轉換標註（重新映射類別 ID）"""
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()

            new_labels = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue

                old_class_id = int(parts[0])

                # 轉換類別 ID
                if old_class_id in self.original_to_new:
                    new_class_id = self.original_to_new[old_class_id]
                else:
                    # 如果已經是 0-3，保持不變
                    new_class_id = old_class_id if old_class_id < 4 else 0

                new_label = f"{new_class_id} {' '.join(parts[1:])}"
                new_labels.append(new_label)

            return new_labels

        except Exception as e:
            print(f"轉換標註失敗 {label_path}: {e}")
            return []

    def _save_stats(self):
        """保存統計信息"""
        stats_path = self.output_dir / "merge_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2, default=int)
        print(f"\n統計信息已保存: {stats_path}")

    def _create_data_yaml(self):
        """創建 YOLO 數據集配置文件"""
        data_yaml = {
            'path': str(self.output_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'nc': 4,
            'names': {
                0: 'barline',
                1: 'barline_double',
                2: 'barline_final',
                3: 'barline_repeat'
            },
            'source': 'Merged dataset from OMR Layout, AudioLabs v2, and DoReMi',
            'datasets': list(self.stats['datasets'].keys()),
            'total_images': self.stats['total_images'],
            'total_barlines': self.stats['total_barlines']
        }

        yaml_path = self.output_dir / "data.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)

        print(f"YOLO 配置已創建: {yaml_path}")

    def print_summary(self):
        """打印合併總結"""
        print("\n" + "="*60)
        print("合併總結 - Barline 數據集")
        print("="*60)
        print(f"總圖片數: {self.stats['total_images']}")
        print(f"  訓練集: {self.stats['train_images']}")
        print(f"  驗證集: {self.stats['val_images']}")
        print(f"總 Barlines: {self.stats['total_barlines']}")

        print("\n數據集來源:")
        for dataset_name, stats in self.stats['datasets'].items():
            print(f"  {dataset_name}:")
            print(f"    圖片: {stats['train_images']} train + {stats['val_images']} val")
            print(f"    Barlines: {stats['barlines']}")

        print("\n類別分佈:")
        class_names = ['barline', 'barline_double', 'barline_final', 'barline_repeat']
        for class_id, class_name in enumerate(class_names):
            count = self.stats['class_distribution'].get(class_id, 0)
            percentage = (count / self.stats['total_barlines'] * 100) if self.stats['total_barlines'] > 0 else 0
            print(f"  {class_id} ({class_name}): {count} ({percentage:.1f}%)")

        print("="*60)


def main():
    parser = argparse.ArgumentParser(description='合併 barline 數據集')
    parser.add_argument(
        '--omr-layout',
        type=str,
        default='/home/thc1006/dev/music-app/training/datasets/external_barlines/omr_layout/converted',
        help='OMR Layout 轉換後的目錄'
    )
    parser.add_argument(
        '--audiolabs',
        type=str,
        default='/home/thc1006/dev/music-app/training/datasets/external_barlines/audiolabs/converted',
        help='AudioLabs 轉換後的目錄'
    )
    parser.add_argument(
        '--doremi',
        type=str,
        default='/home/thc1006/dev/music-app/training/datasets/external_barlines/doremi/converted',
        help='DoReMi 轉換後的目錄'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='/home/thc1006/dev/music-app/training/datasets/external_barlines/merged',
        help='輸出目錄'
    )

    args = parser.parse_args()

    print("Barline 數據集合併器")
    print(f"輸出目錄: {args.output}")

    merger = BarlineDatasetMerger(args.output)
    success = merger.merge_datasets(
        omr_layout_dir=args.omr_layout,
        audiolabs_dir=args.audiolabs,
        doremi_dir=args.doremi
    )

    if success:
        merger.print_summary()
        print("\n✓ 合併完成！")
        print(f"\n數據集位置: {args.output}")
        print(f"配置文件: {args.output}/data.yaml")
        return 0
    else:
        print("\n✗ 合併失敗")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
