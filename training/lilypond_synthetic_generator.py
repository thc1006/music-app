#!/usr/bin/env python3
"""
🎼 LilyPond 精確合成數據生成器 - Phase 8 稀有類別專用

專門針對：
- Class 17: accidental_double_flat (當前 741 樣本 → 目標 5000+)
- Class 31: dynamic_loud (當前 27 樣本 → 目標 5000+)

技術特點：
1. 使用 LilyPond SVG 輸出提取精確 bounding boxes
2. 領域隨機化（字體大小、間距、旋轉等）
3. 自動生成 YOLO 格式標註
4. 批量處理與進度追蹤

使用方法：
    # 生成 double_flat 數據（5000 樣本）
    python lilypond_synthetic_generator.py --class 17 --count 5000

    # 生成 dynamic_loud 數據（5000 樣本）
    python lilypond_synthetic_generator.py --class 31 --count 5000

    # 一次生成兩個類別
    python lilypond_synthetic_generator.py --both --count 5000
"""

import os
import subprocess
import random
import json
import argparse
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import numpy as np

# ============== 配置 ==============

@dataclass
class BoundingBox:
    """Bounding box 資料結構"""
    class_id: int
    x_center: float  # 歸一化 0-1
    y_center: float  # 歸一化 0-1
    width: float     # 歸一化 0-1
    height: float    # 歸一化 0-1

    def to_yolo_format(self) -> str:
        """轉換為 YOLO 格式字串"""
        return f"{self.class_id} {self.x_center:.6f} {self.y_center:.6f} {self.width:.6f} {self.height:.6f}"

# 類別配置
CLASS_CONFIG = {
    17: {  # accidental_double_flat
        'name': 'accidental_double_flat',
        'symbol': '𝄫',
        'current_count': 741,
        'target_count': 5000,
        'description': '重降記號 (double flat)',
    },
    31: {  # dynamic_loud
        'name': 'dynamic_loud',
        'symbol': 'f/ff/fff/sf/sfz',
        'current_count': 27,
        'target_count': 5000,
        'description': '強記號 (forte dynamics)',
    }
}

# ============== LilyPond 模板 ==============

LILYPOND_HEADER_TEMPLATE = r'''
\version "2.24.0"
\header {{
  tagline = ##f
}}

\paper {{
  indent = 0
  paper-width = {width}\mm
  paper-height = {height}\mm
  top-margin = {margin}\mm
  bottom-margin = {margin}\mm
  left-margin = {margin}\mm
  right-margin = {margin}\mm
  system-system-spacing.basic-distance = #12
}}

#(set-global-staff-size {staff_size})

\layout {{
  \context {{
    \Score
    \override SpacingSpanner.base-shortest-duration = #(ly:make-moment 1/{spacing})
  }}
}}
'''

# ============== 樂譜生成器 ==============

class DoubleFlatGenerator:
    """生成包含重降記號的樂譜"""

    # 所有可用的重降音符
    NOTES = ["ceses", "deses", "eeses", "feses", "geses", "aeses", "beses"]
    OCTAVES = ["", "'", "''", ","]
    DURATIONS = ["1", "2", "4", "8", "16"]

    @staticmethod
    def generate_score(variation: Dict) -> str:
        """生成一個樂譜變體"""
        num_measures = random.randint(4, 8)
        beats_per_measure = random.choice([4, 3, 6])

        measures = []
        for _ in range(num_measures):
            measure_notes = []
            remaining_beats = beats_per_measure

            while remaining_beats > 0:
                # 選擇音符
                note = random.choice(DoubleFlatGenerator.NOTES)
                octave = random.choice(DoubleFlatGenerator.OCTAVES)

                # 選擇時值（確保不超過剩餘拍數）
                available_durations = [d for d in DoubleFlatGenerator.DURATIONS
                                      if 4 / int(d) <= remaining_beats]
                if not available_durations:
                    available_durations = ["16"]
                duration = random.choice(available_durations)

                measure_notes.append(f"{note}{octave}{duration}")
                remaining_beats -= 4 / int(duration)

            measures.append(" ".join(measure_notes))

        # 隨機添加表情記號
        articulations = ["", "-.", "-!", "->", "--"]
        measures_with_articulations = []
        for measure in measures:
            notes = measure.split()
            for i in range(len(notes)):
                if random.random() < 0.2:  # 20% 概率添加表情
                    notes[i] += random.choice(articulations)
            measures_with_articulations.append(" ".join(notes))

        return " | ".join(measures_with_articulations) + " \\bar \"|.\""

class DynamicLoudGenerator:
    """生成包含強記號的樂譜"""

    # 所有強記號
    DYNAMICS = [
        r"\f",      # forte
        r"\ff",     # fortissimo
        r"\fff",    # fortississimo
        r"\sf",     # sforzando
        r"\sfz",    # sforzato
        r"\fp",     # forte-piano
        r"\fz",     # forzando
    ]

    NOTES = ["c", "d", "e", "f", "g", "a", "b"]
    OCTAVES = ["", "'", "''"]
    DURATIONS = ["2", "4", "8"]

    @staticmethod
    def generate_score(variation: Dict) -> str:
        """生成一個樂譜變體"""
        num_phrases = random.randint(3, 6)
        phrases = []

        for _ in range(num_phrases):
            # 每個樂句包含 4-8 個音符
            phrase_length = random.randint(4, 8)
            phrase_notes = []

            # 在樂句開始或中間添加力度記號
            dynamic_positions = random.sample(range(phrase_length),
                                            k=random.randint(1, 3))

            for i in range(phrase_length):
                note = random.choice(DynamicLoudGenerator.NOTES)
                octave = random.choice(DynamicLoudGenerator.OCTAVES)
                duration = random.choice(DynamicLoudGenerator.DURATIONS)

                note_str = f"{note}{octave}{duration}"

                # 添加力度記號
                if i in dynamic_positions:
                    note_str += random.choice(DynamicLoudGenerator.DYNAMICS)

                phrase_notes.append(note_str)

            phrases.append(" ".join(phrase_notes))

        return " | ".join(phrases) + " \\bar \"|.\""

# ============== SVG 解析與 Bbox 提取 ==============

class SVGBboxExtractor:
    """從 LilyPond SVG 輸出提取 bounding boxes"""

    # SVG 元素與類別的映射
    ELEMENT_CLASS_MAP = {
        17: ["accidentals.flatflat"],  # double flat
        31: ["scripts.sforzato", "f", "ff", "fff", "sf", "sfz", "fp", "fz"],  # dynamics
    }

    @staticmethod
    def extract_bboxes(svg_path: str, target_class: int,
                       img_width: int, img_height: int) -> List[BoundingBox]:
        """
        從 SVG 文件提取 bounding boxes

        Args:
            svg_path: SVG 文件路徑
            target_class: 目標類別 ID
            img_width: 圖片寬度（像素）
            img_height: 圖片高度（像素）

        Returns:
            BoundingBox 列表
        """
        try:
            tree = ET.parse(svg_path)
            root = tree.getroot()

            # SVG 命名空間
            ns = {'svg': 'http://www.w3.org/2000/svg'}

            bboxes = []
            target_elements = SVGBboxExtractor.ELEMENT_CLASS_MAP.get(target_class, [])

            # 查找所有相關元素
            for element in root.iter():
                # 檢查元素 class 或 id 屬性
                elem_class = element.get('class', '')
                elem_id = element.get('id', '')

                # 檢查是否匹配目標符號
                is_target = any(target in elem_class or target in elem_id
                              for target in target_elements)

                if is_target:
                    bbox = SVGBboxExtractor._extract_element_bbox(
                        element, img_width, img_height, target_class)
                    if bbox:
                        bboxes.append(bbox)

            return bboxes

        except Exception as e:
            print(f"⚠️  SVG 解析失敗: {e}")
            return []

    @staticmethod
    def _extract_element_bbox(element, img_width: int, img_height: int,
                            class_id: int) -> Optional[BoundingBox]:
        """從 SVG 元素提取 bbox"""
        try:
            # 獲取元素位置和尺寸
            x = float(element.get('x', 0))
            y = float(element.get('y', 0))
            width = float(element.get('width', 10))
            height = float(element.get('height', 10))

            # 處理 transform 屬性（簡化版）
            transform = element.get('transform', '')
            if 'translate' in transform:
                # 簡化處理：提取 translate(x, y)
                import re
                match = re.search(r'translate\(([-\d.]+)[,\s]+([-\d.]+)\)', transform)
                if match:
                    x += float(match.group(1))
                    y += float(match.group(2))

            # 轉換為歸一化座標
            x_center = (x + width / 2) / img_width
            y_center = (y + height / 2) / img_height
            norm_width = width / img_width
            norm_height = height / img_height

            # 檢查合理性
            if (0 <= x_center <= 1 and 0 <= y_center <= 1 and
                0 < norm_width <= 1 and 0 < norm_height <= 1):
                return BoundingBox(class_id, x_center, y_center,
                                 norm_width, norm_height)

        except Exception as e:
            print(f"⚠️  Bbox 提取失敗: {e}")

        return None

# ============== 圖像增強 ==============

class ImageAugmentor:
    """圖像領域隨機化增強"""

    @staticmethod
    def apply_augmentations(img: Image.Image, variation: Dict) -> Image.Image:
        """應用隨機增強"""
        # 1. 旋轉（小角度）
        if variation.get('rotate', True):
            angle = random.uniform(-2, 2)
            img = img.rotate(angle, fillcolor='white', expand=False)

        # 2. 對比度調整
        if variation.get('contrast', True):
            enhancer = ImageEnhance.Contrast(img)
            factor = random.uniform(0.8, 1.2)
            img = enhancer.enhance(factor)

        # 3. 亮度調整
        if variation.get('brightness', True):
            enhancer = ImageEnhance.Brightness(img)
            factor = random.uniform(0.9, 1.1)
            img = enhancer.enhance(factor)

        # 4. 輕微模糊（模擬打印/掃描）
        if variation.get('blur', False) and random.random() < 0.3:
            img = img.filter(ImageFilter.GaussianBlur(radius=0.5))

        # 5. 添加噪點
        if variation.get('noise', False) and random.random() < 0.2:
            img = ImageAugmentor._add_noise(img)

        return img

    @staticmethod
    def _add_noise(img: Image.Image, intensity: float = 0.02) -> Image.Image:
        """添加椒鹽噪點"""
        img_array = np.array(img)
        noise = np.random.normal(0, intensity * 255, img_array.shape)
        noisy = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy)

# ============== 主生成器 ==============

class LilyPondSyntheticGenerator:
    """LilyPond 合成數據生成器主類"""

    def __init__(self, output_dir: Path, temp_dir: Path = None):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 創建子目錄
        self.images_dir = output_dir / 'images'
        self.labels_dir = output_dir / 'labels'
        self.images_dir.mkdir(exist_ok=True)
        self.labels_dir.mkdir(exist_ok=True)

        # 臨時工作目錄
        self.temp_dir = temp_dir or (output_dir / 'temp')
        self.temp_dir.mkdir(exist_ok=True)

        # 統計
        self.stats = {
            'total_generated': 0,
            'total_bboxes': 0,
            'class_counts': {}
        }

    def generate_for_class(self, class_id: int, count: int,
                          augment: bool = True) -> Dict:
        """
        為特定類別生成合成數據

        Args:
            class_id: 目標類別 ID (17 或 31)
            count: 生成數量
            augment: 是否應用圖像增強

        Returns:
            生成統計字典
        """
        class_config = CLASS_CONFIG.get(class_id)
        if not class_config:
            raise ValueError(f"不支持的類別: {class_id}")

        print(f"\n{'='*60}")
        print(f"生成 Class {class_id} ({class_config['name']}) 數據")
        print(f"{'='*60}")
        print(f"  符號: {class_config['symbol']}")
        print(f"  當前樣本: {class_config['current_count']}")
        print(f"  目標數量: {count}")
        print(f"  圖像增強: {'啟用' if augment else '禁用'}")

        success_count = 0
        bbox_count = 0

        for i in range(count):
            try:
                # 1. 生成變體配置
                variation = self._generate_variation(class_id)
                file_id = f"synthetic_c{class_id}_{i:05d}"

                # 2. 生成並渲染 LilyPond
                ly_path, svg_path, png_path = self._render_lilypond(
                    class_id, file_id, variation)

                if not png_path or not svg_path:
                    continue

                # 3. 提取 bounding boxes
                img = Image.open(png_path)
                img_width, img_height = img.size

                bboxes = SVGBboxExtractor.extract_bboxes(
                    svg_path, class_id, img_width, img_height)

                # 如果 SVG 提取失敗，使用啟發式方法
                if not bboxes:
                    bboxes = self._heuristic_bbox_detection(
                        png_path, class_id, img_width, img_height)

                if not bboxes:
                    print(f"  ⚠️  No bboxes found for {file_id}")
                    continue

                # 4. 圖像增強
                if augment:
                    img = ImageAugmentor.apply_augmentations(img, variation)

                # 5. 保存圖片和標註
                final_img_path = self.images_dir / f"{file_id}.png"
                img.save(final_img_path)

                label_path = self.labels_dir / f"{file_id}.txt"
                with open(label_path, 'w') as f:
                    for bbox in bboxes:
                        f.write(bbox.to_yolo_format() + '\n')

                success_count += 1
                bbox_count += len(bboxes)

                # 6. 進度顯示
                if success_count % 100 == 0:
                    print(f"  進度: {success_count}/{count} "
                          f"(平均 {bbox_count/success_count:.1f} bbox/img)")

            except Exception as e:
                print(f"  ❌ 生成失敗 ({i}): {e}")
                continue

        # 統計
        self.stats['total_generated'] += success_count
        self.stats['total_bboxes'] += bbox_count
        self.stats['class_counts'][class_id] = bbox_count

        print(f"\n✅ 完成: {success_count}/{count}")
        print(f"  總 bboxes: {bbox_count}")
        print(f"  平均: {bbox_count/success_count:.2f} bbox/圖片")

        return {
            'class_id': class_id,
            'success_count': success_count,
            'bbox_count': bbox_count,
            'avg_bbox_per_img': bbox_count / success_count if success_count > 0 else 0
        }

    def _generate_variation(self, class_id: int) -> Dict:
        """生成隨機變體配置"""
        return {
            'staff_size': random.choice([18, 20, 22, 24, 26]),
            'width': random.randint(180, 220),
            'height': random.randint(50, 80),
            'margin': random.randint(3, 7),
            'spacing': random.choice([8, 16, 32]),
            'rotate': random.random() < 0.7,
            'contrast': random.random() < 0.8,
            'brightness': random.random() < 0.8,
            'blur': random.random() < 0.3,
            'noise': random.random() < 0.2,
        }

    def _render_lilypond(self, class_id: int, file_id: str,
                        variation: Dict) -> Tuple[str, str, str]:
        """
        生成並渲染 LilyPond

        Returns:
            (ly_path, svg_path, png_path) 或 (None, None, None)
        """
        # 生成樂譜內容
        if class_id == 17:
            score_content = DoubleFlatGenerator.generate_score(variation)
        elif class_id == 31:
            score_content = DynamicLoudGenerator.generate_score(variation)
        else:
            raise ValueError(f"不支持的類別: {class_id}")

        # 生成 LilyPond 文件
        header = LILYPOND_HEADER_TEMPLATE.format(**variation)
        full_content = f'''{header}

\\relative c' {{
  {score_content}
}}
'''

        ly_path = self.temp_dir / f"{file_id}.ly"
        with open(ly_path, 'w', encoding='utf-8') as f:
            f.write(full_content)

        # 渲染 SVG 和 PNG
        try:
            # 生成 SVG（用於提取 bbox）
            svg_result = subprocess.run(
                ['lilypond', '-dbackend=svg', '-o',
                 str(self.temp_dir / file_id), str(ly_path)],
                capture_output=True,
                timeout=30,
                text=True
            )

            # 生成 PNG（用於訓練）
            png_result = subprocess.run(
                ['lilypond', '-dbackend=eps', '-dno-gs-load-fonts',
                 '-dinclude-eps-fonts', '--png', '-dresolution=300',
                 '-o', str(self.temp_dir / file_id), str(ly_path)],
                capture_output=True,
                timeout=30,
                text=True
            )

            # LilyPond 可能生成 .svg 或 -1.svg（多頁面）
            svg_path = self.temp_dir / f"{file_id}.svg"
            if not svg_path.exists():
                svg_path = self.temp_dir / f"{file_id}-1.svg"

            # LilyPond 可能生成 .png 或 -page1.png（多頁面）
            png_path = self.temp_dir / f"{file_id}.png"
            if not png_path.exists():
                png_path = self.temp_dir / f"{file_id}-page1.png"

            if svg_path.exists() and png_path.exists():
                return str(ly_path), str(svg_path), str(png_path)
            else:
                print(f"  ⚠️  渲染文件不存在: svg={svg_path.exists()}, png={png_path.exists()}")
                if svg_result.returncode != 0:
                    print(f"    SVG error: {svg_result.stderr[:200]}")
                if png_result.returncode != 0:
                    print(f"    PNG error: {png_result.stderr[:200]}")
                return None, None, None

        except subprocess.TimeoutExpired:
            print(f"  ⚠️  LilyPond 超時")
            return None, None, None
        except Exception as e:
            print(f"  ⚠️  渲染失敗: {e}")
            return None, None, None

    def _heuristic_bbox_detection(self, png_path: str, class_id: int,
                                  img_width: int, img_height: int) -> List[BoundingBox]:
        """
        啟發式 bbox 檢測（當 SVG 解析失敗時使用）

        使用簡單的模板匹配或連通組件分析
        """
        try:
            img = Image.open(png_path).convert('L')
            img_array = np.array(img)

            # 二值化
            threshold = 200
            binary = (img_array < threshold).astype(np.uint8)

            # 查找非白色區域
            from scipy import ndimage
            labeled, num_features = ndimage.label(binary)

            bboxes = []
            for i in range(1, num_features + 1):
                # 獲取連通組件的 bbox
                y_indices, x_indices = np.where(labeled == i)
                if len(y_indices) == 0:
                    continue

                x_min, x_max = x_indices.min(), x_indices.max()
                y_min, y_max = y_indices.min(), y_indices.max()

                width = x_max - x_min
                height = y_max - y_min

                # 過濾太小或太大的區域
                if width < 5 or height < 5 or width > img_width * 0.5:
                    continue

                # 轉換為歸一化座標
                x_center = (x_min + x_max) / 2 / img_width
                y_center = (y_min + y_max) / 2 / img_height
                norm_width = width / img_width
                norm_height = height / img_height

                bboxes.append(BoundingBox(
                    class_id, x_center, y_center, norm_width, norm_height))

            return bboxes

        except Exception as e:
            print(f"  ⚠️  啟發式檢測失敗: {e}")
            return []

    def cleanup_temp(self):
        """清理臨時文件"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            print(f"✅ 清理臨時目錄: {self.temp_dir}")

    def save_stats(self, output_path: Path = None):
        """保存生成統計"""
        if output_path is None:
            output_path = self.output_dir / 'generation_stats.json'

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2, ensure_ascii=False)

        print(f"✅ 統計已保存: {output_path}")

# ============== 主程序 ==============

def main():
    parser = argparse.ArgumentParser(
        description='LilyPond 精確合成數據生成器',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 生成 5000 個 double_flat 樣本
  python lilypond_synthetic_generator.py --class 17 --count 5000

  # 生成 5000 個 dynamic_loud 樣本
  python lilypond_synthetic_generator.py --class 31 --count 5000

  # 同時生成兩個類別
  python lilypond_synthetic_generator.py --both --count 5000

  # 禁用圖像增強（更快但多樣性低）
  python lilypond_synthetic_generator.py --class 17 --count 1000 --no-augment
        """)

    parser.add_argument('--class', type=int, dest='target_class',
                       choices=[17, 31], help='目標類別 (17=double_flat, 31=dynamic_loud)')
    parser.add_argument('--count', type=int, default=5000,
                       help='生成數量 (default: 5000)')
    parser.add_argument('--both', action='store_true',
                       help='同時生成兩個類別')
    parser.add_argument('--output', type=str,
                       default='datasets/yolo_synthetic_phase8',
                       help='輸出目錄')
    parser.add_argument('--no-augment', action='store_true',
                       help='禁用圖像增強')
    parser.add_argument('--check-lilypond', action='store_true',
                       help='檢查 LilyPond 是否安裝')
    parser.add_argument('--keep-temp', action='store_true',
                       help='保留臨時文件（用於調試）')

    args = parser.parse_args()

    # 檢查 LilyPond
    if args.check_lilypond:
        try:
            result = subprocess.run(['lilypond', '--version'],
                                  capture_output=True, text=True, timeout=5)
            version = result.stdout.split('\n')[0]
            print(f"✅ {version}")
            print("LilyPond 已正確安裝")
        except FileNotFoundError:
            print("❌ LilyPond 未安裝")
            print("安裝方式: sudo apt install lilypond")
        except Exception as e:
            print(f"❌ 檢查失敗: {e}")
        return

    # 驗證參數
    if not args.both and args.target_class is None:
        parser.print_help()
        print("\n❌ 錯誤: 必須指定 --class 或 --both")
        return

    # 初始化生成器
    output_dir = Path(args.output).resolve()
    generator = LilyPondSyntheticGenerator(output_dir)

    print(f"\n{'='*60}")
    print("🎼 LilyPond 精確合成數據生成器 - Phase 8")
    print(f"{'='*60}")
    print(f"輸出目錄: {output_dir}")
    print(f"圖像增強: {'禁用' if args.no_augment else '啟用'}")

    # 生成數據
    try:
        if args.both:
            # 生成兩個類別
            for class_id in [17, 31]:
                result = generator.generate_for_class(
                    class_id, args.count, augment=not args.no_augment)
        else:
            # 生成單個類別
            result = generator.generate_for_class(
                args.target_class, args.count, augment=not args.no_augment)

        # 保存統計
        generator.save_stats()

        # 顯示最終統計
        print(f"\n{'='*60}")
        print("📊 最終統計")
        print(f"{'='*60}")
        print(f"總生成圖片: {generator.stats['total_generated']}")
        print(f"總 bboxes: {generator.stats['total_bboxes']}")
        for class_id, count in generator.stats['class_counts'].items():
            class_name = CLASS_CONFIG[class_id]['name']
            print(f"  Class {class_id} ({class_name}): {count} bboxes")

        print(f"\n✅ 數據集已生成:")
        print(f"  圖片: {generator.images_dir}")
        print(f"  標註: {generator.labels_dir}")

    finally:
        # 清理臨時文件
        if not args.keep_temp:
            generator.cleanup_temp()
        else:
            print(f"⚠️  臨時文件保留於: {generator.temp_dir}")

if __name__ == '__main__':
    import sys

    # 切換到腳本目錄
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    # 檢查依賴
    try:
        from scipy import ndimage
    except ImportError:
        print("❌ 缺少依賴: scipy")
        print("安裝方式: pip install scipy")
        sys.exit(1)

    try:
        from PIL import Image
    except ImportError:
        print("❌ 缺少依賴: Pillow")
        print("安裝方式: pip install Pillow")
        sys.exit(1)

    main()
