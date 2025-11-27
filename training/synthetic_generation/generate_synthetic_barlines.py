#!/usr/bin/env python3
"""
Verovio 合成 Barline 數據生成主腳本

使用方法:
    python generate_synthetic_barlines.py --num-images 1000 --output-dir output
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from typing import Dict
import random
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import cv2
import numpy as np
from PIL import Image

# 添加 src 到路徑
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from mei_generator import MEIGenerator
from verovio_renderer import VerovioRenderer
from bbox_extractor import BboxExtractor
from domain_randomization import DomainRandomizer
from yolo_exporter import YOLOExporter


class SyntheticBarlineGenerator:
    """合成 Barline 數據生成器"""

    def __init__(self, config_path: str, output_dir: str):
        """
        初始化生成器

        Args:
            config_path: 配置文件路徑
            output_dir: 輸出目錄
        """
        # 載入配置
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.output_dir = output_dir

        # 初始化各個模組
        self.mei_generator = MEIGenerator(self.config)
        self.renderer = VerovioRenderer(self.config)
        self.bbox_extractor = BboxExtractor()
        self.randomizer = DomainRandomizer(self.config)
        self.exporter = YOLOExporter(self.config)

        # 創建輸出目錄
        self.images_dir = os.path.join(output_dir, 'images')
        self.labels_dir = os.path.join(output_dir, 'labels')
        self.validation_dir = os.path.join(output_dir, 'validation')

        for d in [self.images_dir, self.labels_dir, self.validation_dir]:
            os.makedirs(d, exist_ok=True)

        # 獲取 barline 類型分佈
        self.barline_types = list(self.config['barline_types'].keys())
        self.barline_weights = [
            self.config['barline_types'][t] for t in self.barline_types
        ]

    def generate_single_image(self, index: int) -> Dict:
        """
        生成單張圖像及其標註

        Args:
            index: 圖像索引

        Returns:
            生成結果統計
        """
        try:
            # 隨機選擇 barline 類型
            barline_type = random.choices(
                self.barline_types,
                weights=self.barline_weights,
                k=1
            )[0]

            # 生成 MEI
            mei_content = self.mei_generator.generate_mei(barline_type)

            # 渲染為 SVG 和 PNG
            svg = self.renderer.render_mei_to_svg(mei_content)
            image = self.renderer.render_mei_to_png(mei_content)

            # 獲取圖像尺寸
            img_width, img_height = image.size

            # 提取 bounding boxes
            barlines = self.bbox_extractor.extract_barlines(
                svg, img_width, img_height
            )

            # 應用 domain randomization
            augmented_image = self.randomizer.apply(image)

            # 調整圖像大小到配置指定的尺寸
            target_width = self.config['image']['width']
            target_height = self.config['image']['height']

            if (img_width, img_height) != (target_width, target_height):
                augmented_image = augmented_image.resize(
                    (target_width, target_height),
                    resample=Image.LANCZOS
                )

                # 調整 bbox 座標（已歸一化，無需調整）

            # 保存圖像
            image_filename = f"barline_{index:06d}.png"
            image_path = os.path.join(self.images_dir, image_filename)
            augmented_image.save(image_path)

            # 保存標註
            label_filename = f"barline_{index:06d}.txt"
            label_path = os.path.join(self.labels_dir, label_filename)
            self.exporter.export_annotation(barlines, label_path)

            return {
                'success': True,
                'index': index,
                'barline_type': barline_type,
                'num_barlines': len(barlines)
            }

        except Exception as e:
            return {
                'success': False,
                'index': index,
                'error': str(e)
            }

    def generate_batch(self, num_images: int, workers: int = 4):
        """
        批量生成圖像

        Args:
            num_images: 圖像數量
            workers: 並行工作進程數
        """
        print(f"開始生成 {num_images} 張合成圖像...")
        print(f"使用 {workers} 個並行進程")

        # 如果 workers = 1，使用單進程（方便調試）
        if workers == 1:
            results = []
            for i in tqdm(range(num_images), desc="生成進度"):
                result = self.generate_single_image(i)
                results.append(result)
        else:
            # 多進程生成
            with Pool(processes=workers) as pool:
                results = list(tqdm(
                    pool.imap(self.generate_single_image, range(num_images)),
                    total=num_images,
                    desc="生成進度"
                ))

        # 統計結果
        successful = sum(1 for r in results if r['success'])
        failed = num_images - successful

        print(f"\n生成完成!")
        print(f"成功: {successful}")
        print(f"失敗: {failed}")

        # 失敗詳情
        if failed > 0:
            print("\n失敗樣本:")
            for r in results:
                if not r['success']:
                    print(f"  圖像 {r['index']}: {r['error']}")

        # Barline 類型統計
        print("\nBarline 類型分佈:")
        type_counts = {}
        for r in results:
            if r['success']:
                bt = r['barline_type']
                type_counts[bt] = type_counts.get(bt, 0) + 1

        for bt, count in sorted(type_counts.items()):
            print(f"  {bt}: {count}")

        return results

    def create_validation_samples(self, num_samples: int = 50):
        """
        創建可視化驗證樣本

        Args:
            num_samples: 樣本數量
        """
        print(f"\n創建 {num_samples} 個可視化驗證樣本...")

        image_files = list(Path(self.images_dir).glob('*.png'))
        if len(image_files) == 0:
            print("錯誤: 沒有找到圖像文件")
            return

        # 隨機選擇樣本
        sample_files = random.sample(
            image_files,
            min(num_samples, len(image_files))
        )

        for img_path in tqdm(sample_files, desc="可視化進度"):
            self._create_visualization(img_path)

        print("可視化樣本已保存到:", self.validation_dir)

    def _create_visualization(self, image_path: Path):
        """
        創建單個可視化樣本（繪製 bounding boxes）

        Args:
            image_path: 圖像路徑
        """
        # 讀取圖像
        img = cv2.imread(str(image_path))
        if img is None:
            return

        # 讀取標註
        label_path = image_path.with_suffix('.txt')
        label_path = Path(self.labels_dir) / label_path.name.replace('.png', '.txt')

        if not label_path.exists():
            return

        # 解析標註
        with open(label_path, 'r') as f:
            lines = f.readlines()

        h, w = img.shape[:2]

        # 顏色映射
        colors = {
            0: (0, 255, 0),      # single - 綠色
            1: (255, 0, 0),      # double - 藍色
            2: (0, 0, 255),      # final - 紅色
            3: (255, 255, 0),    # repeat_left - 青色
            4: (255, 0, 255),    # repeat_right - 品紅
            5: (0, 255, 255),    # repeat_both - 黃色
        }

        # 類別名稱
        class_names = {
            0: 'single', 1: 'double', 2: 'final',
            3: 'rpt_left', 4: 'rpt_right', 5: 'rpt_both'
        }

        # 繪製 bounding boxes
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            class_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:])

            # 轉換為像素座標
            x1 = int((x_center - width / 2) * w)
            y1 = int((y_center - height / 2) * h)
            x2 = int((x_center + width / 2) * w)
            y2 = int((y_center + height / 2) * h)

            # 繪製矩形
            color = colors.get(class_id, (128, 128, 128))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # 添加標籤
            label = class_names.get(class_id, f'class_{class_id}')
            cv2.putText(img, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 保存可視化結果
        output_path = os.path.join(self.validation_dir, image_path.name)
        cv2.imwrite(output_path, img)

    def organize_dataset(self):
        """組織數據集為訓練/驗證集"""
        print("\n組織數據集...")

        split_ratio = self.config.get('output', {}).get('split_ratio', {
            'train': 0.9,
            'val': 0.1
        })

        stats = self.exporter.organize_dataset(
            self.images_dir,
            self.labels_dir,
            self.output_dir,
            split_ratio
        )

        print(f"訓練集: {stats['train']} 張")
        print(f"驗證集: {stats['val']} 張")
        print(f"總計: {stats['total']} 張")

        # 創建 dataset.yaml
        yaml_path = self.exporter.create_dataset_yaml(
            self.output_dir,
            'synthetic_barlines'
        )
        print(f"\n數據集配置已保存: {yaml_path}")

        return stats


def main():
    parser = argparse.ArgumentParser(
        description="Verovio 合成 Barline 數據生成器"
    )
    parser.add_argument(
        '--num-images', type=int, default=1000,
        help='生成圖像數量 (默認: 1000)'
    )
    parser.add_argument(
        '--config', type=str,
        default='configs/generation_config.yaml',
        help='配置文件路徑'
    )
    parser.add_argument(
        '--output-dir', type=str, default='output',
        help='輸出目錄'
    )
    parser.add_argument(
        '--workers', type=int, default=cpu_count(),
        help=f'並行工作進程數 (默認: {cpu_count()})'
    )
    parser.add_argument(
        '--validation-samples', type=int, default=50,
        help='可視化驗證樣本數量 (默認: 50)'
    )
    parser.add_argument(
        '--skip-organization', action='store_true',
        help='跳過數據集組織步驟'
    )

    args = parser.parse_args()

    # 檢查配置文件
    config_path = os.path.join(
        os.path.dirname(__file__),
        args.config
    )

    if not os.path.exists(config_path):
        print(f"錯誤: 配置文件不存在: {config_path}")
        sys.exit(1)

    # 創建生成器
    generator = SyntheticBarlineGenerator(config_path, args.output_dir)

    # 生成數據
    results = generator.generate_batch(args.num_images, args.workers)

    # 創建可視化樣本
    if args.validation_samples > 0:
        generator.create_validation_samples(args.validation_samples)

    # 組織數據集
    if not args.skip_organization:
        generator.organize_dataset()

    print("\n完成！")


if __name__ == '__main__':
    main()
