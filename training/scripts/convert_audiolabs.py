#!/usr/bin/env python3
"""
AudioLabs v2 數據集轉換腳本
從 measure bounding boxes 提取 barline 位置
"""
import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import yaml
from tqdm import tqdm
import cv2
import numpy as np
import argparse


class AudioLabsConverter:
    """AudioLabs v2 數據集轉換器"""

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
            'measures_processed': 0,
            'skipped': 0
        }

    def find_annotation_files(self) -> List[Path]:
        """查找標註文件"""
        print("搜索 AudioLabs 標註文件...")

        # AudioLabs 使用 JSON 格式存儲 measure annotations
        json_files = list(self.input_dir.rglob('*.json'))

        print(f"找到 {len(json_files)} 個標註文件")
        return json_files

    def parse_audiolabs_json(self, json_path: Path) -> Optional[Dict]:
        """解析 AudioLabs JSON 格式"""
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            return data
        except Exception as e:
            print(f"解析 JSON 失敗 {json_path}: {e}")
            return None

    def extract_barlines_from_measures(
        self,
        measures: List[Dict],
        img_width: int,
        img_height: int
    ) -> List[str]:
        """
        從 measure annotations 提取 barlines

        AudioLabs 格式示例:
        {
            "measure_id": 1,
            "bbox": [x, y, width, height],  # 像素坐標
            ...
        }
        """
        barlines = []

        # 按 x 坐標排序 measures
        sorted_measures = sorted(measures, key=lambda m: m.get('bbox', [0])[0])

        for i, measure in enumerate(sorted_measures):
            bbox = measure.get('bbox')
            if not bbox or len(bbox) != 4:
                continue

            x, y, w, h = bbox

            # 轉換為 YOLO 格式 (normalized)
            x_norm = x / img_width
            y_norm = y / img_height
            w_norm = w / img_width
            h_norm = h / img_height

            # 計算 measure 的中心和邊界
            left_x = x_norm
            right_x = x_norm + w_norm
            center_y = y_norm + h_norm / 2

            # Barline 寬度（相對於圖片）
            barline_width = 0.005

            # 左側 barline (measure 起始)
            left_barline = f"{self.class_mapping['barline']} {left_x:.6f} {center_y:.6f} {barline_width:.6f} {h_norm:.6f}"
            barlines.append(left_barline)

            # 判斷是否為最後一個 measure (final barline)
            is_last = (i == len(sorted_measures) - 1)

            if is_last:
                # 最後的 measure 使用 final barline
                right_barline = f"{self.class_mapping['barline_final']} {right_x:.6f} {center_y:.6f} {barline_width:.6f} {h_norm:.6f}"
            else:
                # 檢查下一個 measure 是否在同一行（y 坐標接近）
                next_measure = sorted_measures[i + 1]
                next_y = next_measure.get('bbox', [0, 0])[1]

                # 如果 y 坐標差異大於 measure 高度的一半，說明換行了
                if abs(next_y - y) > h / 2:
                    # 行尾使用 final barline
                    right_barline = f"{self.class_mapping['barline_final']} {right_x:.6f} {center_y:.6f} {barline_width:.6f} {h_norm:.6f}"
                else:
                    # 普通 barline
                    right_barline = f"{self.class_mapping['barline']} {right_x:.6f} {center_y:.6f} {barline_width:.6f} {h_norm:.6f}"

            barlines.append(right_barline)

            self.stats['measures_processed'] += 1

        return barlines

    def convert_dataset(self, val_split: float = 0.15):
        """轉換整個數據集"""
        print("\n開始轉換 AudioLabs v2 數據集")
        print("="*60)

        # 查找標註文件
        annotation_files = self.find_annotation_files()

        if not annotation_files:
            print("✗ 未找到標註文件")
            return False

        # 處理每個標註文件
        all_data = []
        for json_path in tqdm(annotation_files, desc="Loading annotations"):
            data = self.parse_audiolabs_json(json_path)
            if data:
                all_data.append((json_path, data))

        if not all_data:
            print("✗ 未能解析任何標註文件")
            return False

        # 分割訓練集和驗證集
        import random
        random.shuffle(all_data)
        split_idx = int(len(all_data) * (1 - val_split))
        train_data = all_data[:split_idx]
        val_data = all_data[split_idx:]

        print(f"訓練集: {len(train_data)} 個文件")
        print(f"驗證集: {len(val_data)} 個文件")

        # 處理訓練集
        self._process_split(train_data, 'train')

        # 處理驗證集
        self._process_split(val_data, 'val')

        # 保存統計信息
        self._save_stats()

        # 創建 data.yaml
        self._create_data_yaml()

        return True

    def _process_split(self, data_list: List[Tuple[Path, Dict]], split: str):
        """處理單個數據分割"""
        print(f"\n處理 {split} 集...")

        barlines_count = 0

        for json_path, data in tqdm(data_list, desc=f"Converting {split}"):
            # 獲取對應的圖片路徑
            # AudioLabs 通常將圖片和標註放在不同目錄
            img_path = self._find_image_for_annotation(json_path, data)

            if not img_path or not img_path.exists():
                self.stats['skipped'] += 1
                continue

            # 讀取圖片獲取尺寸
            img = cv2.imread(str(img_path))
            if img is None:
                self.stats['skipped'] += 1
                continue

            img_height, img_width = img.shape[:2]

            # 提取 measures
            measures = data.get('measures', [])
            if not measures:
                # 嘗試其他可能的鍵名
                measures = data.get('annotations', [])

            if not measures:
                self.stats['skipped'] += 1
                continue

            # 提取 barlines
            barlines = self.extract_barlines_from_measures(
                measures, img_width, img_height
            )

            if not barlines:
                self.stats['skipped'] += 1
                continue

            # 保存圖片
            dest_img_path = self.output_dir / "images" / split / img_path.name
            shutil.copy2(img_path, dest_img_path)

            # 保存標註
            label_name = img_path.stem + '.txt'
            dest_label_path = self.output_dir / "labels" / split / label_name
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

    def _find_image_for_annotation(self, json_path: Path, data: Dict) -> Optional[Path]:
        """查找對應的圖片文件"""
        # 嘗試從 JSON 中獲取圖片路徑
        if 'image_path' in data:
            img_path = Path(data['image_path'])
            if img_path.is_absolute() and img_path.exists():
                return img_path

        if 'image_filename' in data:
            img_filename = data['image_filename']
        else:
            # 使用 JSON 文件名推斷圖片名
            img_filename = json_path.stem

        # 在常見位置搜索圖片
        search_dirs = [
            json_path.parent,
            json_path.parent.parent / "images",
            json_path.parent.parent / "imgs",
            self.input_dir / "images",
            self.input_dir / "imgs",
        ]

        for search_dir in search_dirs:
            if not search_dir.exists():
                continue

            for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
                img_path = search_dir / (img_filename + ext)
                if img_path.exists():
                    return img_path

        return None

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
            'nc': 4,
            'names': {
                0: 'barline',
                1: 'barline_double',
                2: 'barline_final',
                3: 'barline_repeat'
            },
            'original_classes': self.class_mapping,
            'source': 'AudioLabs v2 (derived from measure boundaries)',
            'note': 'Barlines extracted from measure bounding boxes'
        }

        yaml_path = self.output_dir / "data.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)

        print(f"YOLO 配置已創建: {yaml_path}")

    def print_summary(self):
        """打印轉換總結"""
        print("\n" + "="*60)
        print("轉換總結 - AudioLabs v2")
        print("="*60)
        print(f"總圖片數: {self.stats['total_images']}")
        print(f"訓練集: {self.stats['train_images']}")
        print(f"驗證集: {self.stats['val_images']}")
        print(f"處理的 measures: {self.stats['measures_processed']}")
        print(f"提取的 barlines: {self.stats['barlines_extracted']}")
        print(f"跳過: {self.stats['skipped']}")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description='轉換 AudioLabs v2 數據集')
    parser.add_argument(
        '--input',
        type=str,
        default='/home/thc1006/dev/music-app/training/datasets/external_barlines/audiolabs/2019_MeasureDetection_ISMIR2019',
        help='輸入目錄（解壓後的數據集）'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='/home/thc1006/dev/music-app/training/datasets/external_barlines/audiolabs/converted',
        help='輸出目錄'
    )
    parser.add_argument(
        '--val-split',
        type=float,
        default=0.15,
        help='驗證集比例 (預設: 0.15)'
    )

    args = parser.parse_args()

    print("AudioLabs v2 轉換器")
    print(f"輸入: {args.input}")
    print(f"輸出: {args.output}")

    converter = AudioLabsConverter(args.input, args.output)
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
