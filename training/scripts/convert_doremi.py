#!/usr/bin/env python3
"""
DoReMi 數據集轉換腳本
從 OMR XML 標註提取 barline 類別
"""
import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import yaml
from tqdm import tqdm
import cv2
import xml.etree.ElementTree as ET
import argparse


class DoReMiConverter:
    """DoReMi 數據集轉換器"""

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

        # DoReMi 到我們標準的映射
        self.doremi_to_standard = {
            'barline': 'barline',
            'barLine': 'barline',
            'bar_line': 'barline',
            'double_barline': 'barline_double',
            'doubleBarline': 'barline_double',
            'final_barline': 'barline_final',
            'finalBarline': 'barline_final',
            'repeat_barline': 'barline_repeat',
            'repeatBarline': 'barline_repeat',
            'barline_start_repeat': 'barline_repeat',
            'barline_end_repeat': 'barline_repeat',
        }

        self.stats = {
            'total_images': 0,
            'train_images': 0,
            'val_images': 0,
            'barlines_extracted': 0,
            'barline_types': {},
            'skipped': 0
        }

    def find_omr_xml_files(self) -> List[Path]:
        """查找 OMR XML 標註文件"""
        print("搜索 DoReMi OMR XML 文件...")

        # DoReMi 的 OMR XML 文件通常在 parsed_pages 或類似目錄
        xml_files = []

        # 搜索模式
        patterns = [
            "**/*_omr.xml",
            "**/parsed_pages/*.xml",
            "**/*.xml"
        ]

        for pattern in patterns:
            found = list(self.input_dir.glob(pattern))
            xml_files.extend(found)

        # 去重
        xml_files = list(set(xml_files))

        # 過濾掉 MusicXML 文件（只保留 OMR XML）
        xml_files = [f for f in xml_files if self._is_omr_xml(f)]

        print(f"找到 {len(xml_files)} 個 OMR XML 文件")
        return xml_files

    def _is_omr_xml(self, xml_path: Path) -> bool:
        """檢查是否為 OMR XML 文件（而非 MusicXML）"""
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # OMR XML 通常包含 bounding box 信息
            # MusicXML 使用不同的根元素
            if root.tag in ['score-partwise', 'score-timewise']:
                return False  # MusicXML

            # 檢查是否包含 bbox 或類似的 OMR 標註
            for elem in root.iter():
                if any(attr in elem.attrib for attr in ['left', 'top', 'width', 'height', 'bbox']):
                    return True

            return False

        except Exception:
            return False

    def parse_omr_xml(self, xml_path: Path) -> Optional[List[Dict]]:
        """解析 DoReMi OMR XML 文件"""
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            objects = []

            # DoReMi OMR XML 結構示例:
            # <Page>
            #   <Node id="..." class="barline" left="..." top="..." width="..." height="...">
            #   </Node>
            # </Page>

            for node in root.iter('Node'):
                obj_class = node.get('class', '').lower()

                # 只處理 barline 相關類別
                if 'barline' not in obj_class and 'bar' not in obj_class:
                    continue

                # 提取 bounding box
                try:
                    left = float(node.get('left', 0))
                    top = float(node.get('top', 0))
                    width = float(node.get('width', 0))
                    height = float(node.get('height', 0))
                except (ValueError, TypeError):
                    continue

                if width <= 0 or height <= 0:
                    continue

                objects.append({
                    'class': obj_class,
                    'bbox': [left, top, width, height]
                })

            return objects if objects else None

        except Exception as e:
            print(f"解析 XML 失敗 {xml_path}: {e}")
            return None

    def convert_to_yolo_format(
        self,
        objects: List[Dict],
        img_width: int,
        img_height: int
    ) -> List[str]:
        """轉換為 YOLO 格式"""
        yolo_labels = []

        for obj in objects:
            # 映射類別
            obj_class = obj['class']
            standard_class = None

            for doremi_name, std_name in self.doremi_to_standard.items():
                if doremi_name in obj_class:
                    standard_class = std_name
                    break

            if not standard_class:
                # 預設為普通 barline
                standard_class = 'barline'

            class_id = self.class_mapping[standard_class]

            # 提取 bbox
            left, top, width, height = obj['bbox']

            # 轉換為 YOLO 格式 (normalized center coordinates)
            x_center = (left + width / 2) / img_width
            y_center = (top + height / 2) / img_height
            w_norm = width / img_width
            h_norm = height / img_height

            # 確保值在 [0, 1] 範圍內
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            w_norm = max(0, min(1, w_norm))
            h_norm = max(0, min(1, h_norm))

            yolo_label = f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
            yolo_labels.append(yolo_label)

            # 統計
            if standard_class not in self.stats['barline_types']:
                self.stats['barline_types'][standard_class] = 0
            self.stats['barline_types'][standard_class] += 1

        return yolo_labels

    def convert_dataset(self, val_split: float = 0.15):
        """轉換整個數據集"""
        print("\n開始轉換 DoReMi 數據集")
        print("="*60)

        # 查找 OMR XML 文件
        xml_files = self.find_omr_xml_files()

        if not xml_files:
            print("✗ 未找到 OMR XML 文件")
            return False

        # 處理每個 XML 文件
        all_data = []
        for xml_path in tqdm(xml_files, desc="Loading annotations"):
            objects = self.parse_omr_xml(xml_path)
            if objects:
                all_data.append((xml_path, objects))

        if not all_data:
            print("✗ 未能解析任何 barline 標註")
            return False

        print(f"成功解析 {len(all_data)} 個文件包含 barline 標註")

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

    def _process_split(self, data_list: List[Tuple[Path, List[Dict]]], split: str):
        """處理單個數據分割"""
        print(f"\n處理 {split} 集...")

        barlines_count = 0

        for xml_path, objects in tqdm(data_list, desc=f"Converting {split}"):
            # 查找對應的圖片
            img_path = self._find_image_for_xml(xml_path)

            if not img_path or not img_path.exists():
                self.stats['skipped'] += 1
                continue

            # 讀取圖片獲取尺寸
            img = cv2.imread(str(img_path))
            if img is None:
                self.stats['skipped'] += 1
                continue

            img_height, img_width = img.shape[:2]

            # 轉換為 YOLO 格式
            yolo_labels = self.convert_to_yolo_format(
                objects, img_width, img_height
            )

            if not yolo_labels:
                self.stats['skipped'] += 1
                continue

            # 保存圖片
            dest_img_path = self.output_dir / "images" / split / img_path.name
            shutil.copy2(img_path, dest_img_path)

            # 保存標註
            label_name = img_path.stem + '.txt'
            dest_label_path = self.output_dir / "labels" / split / label_name
            with open(dest_label_path, 'w') as f:
                f.write('\n'.join(yolo_labels))

            barlines_count += len(yolo_labels)
            self.stats['total_images'] += 1

            if split == 'train':
                self.stats['train_images'] += 1
            else:
                self.stats['val_images'] += 1

        self.stats['barlines_extracted'] += barlines_count
        print(f"{split} 集提取了 {barlines_count} 個 barlines")

    def _find_image_for_xml(self, xml_path: Path) -> Optional[Path]:
        """查找對應的圖片文件"""
        # DoReMi 的圖片通常與 XML 文件名對應
        # 例如: page_001_omr.xml -> page_001.png

        base_name = xml_path.stem
        # 移除 '_omr' 後綴
        if base_name.endswith('_omr'):
            base_name = base_name[:-4]

        # 搜索目錄
        search_dirs = [
            xml_path.parent,
            xml_path.parent.parent / "images",
            xml_path.parent.parent / "imgs",
            self.input_dir / "images",
            self.input_dir / "imgs",
        ]

        for search_dir in search_dirs:
            if not search_dir.exists():
                continue

            for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
                img_path = search_dir / (base_name + ext)
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
            'source': 'DoReMi (OMR XML annotations)',
            'note': 'Barlines extracted from DoReMi OMR dataset'
        }

        yaml_path = self.output_dir / "data.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)

        print(f"YOLO 配置已創建: {yaml_path}")

    def print_summary(self):
        """打印轉換總結"""
        print("\n" + "="*60)
        print("轉換總結 - DoReMi")
        print("="*60)
        print(f"總圖片數: {self.stats['total_images']}")
        print(f"訓練集: {self.stats['train_images']}")
        print(f"驗證集: {self.stats['val_images']}")
        print(f"提取的 barlines: {self.stats['barlines_extracted']}")
        print(f"\nBarline 類別分佈:")
        for barline_type, count in sorted(self.stats['barline_types'].items()):
            print(f"  {barline_type}: {count}")
        print(f"\n跳過: {self.stats['skipped']}")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description='轉換 DoReMi 數據集')
    parser.add_argument(
        '--input',
        type=str,
        default='/home/thc1006/dev/music-app/training/datasets/external_barlines/doremi/DoReMi_1.0',
        help='輸入目錄（解壓後的數據集）'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='/home/thc1006/dev/music-app/training/datasets/external_barlines/doremi/converted',
        help='輸出目錄'
    )
    parser.add_argument(
        '--val-split',
        type=float,
        default=0.15,
        help='驗證集比例 (預設: 0.15)'
    )

    args = parser.parse_args()

    print("DoReMi 轉換器")
    print(f"輸入: {args.input}")
    print(f"輸出: {args.output}")

    converter = DoReMiConverter(args.input, args.output)
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
