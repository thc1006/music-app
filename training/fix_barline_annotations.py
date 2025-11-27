#!/usr/bin/env python3
"""
Barline Annotation Fix Script - Phase 5 to Phase 6

問題修復：
1. barline (ID 23): 34% 寬度 < 0.005 → 擴大到 ≥ 0.015
2. barline_double (ID 24): 67.8% 面積 > 0.1 → 緊縮到合理大小
3. barline_final (ID 25): 95.9% 面積 > 0.1 → 緊縮到合理大小

輸入：yolo_harmony_v2_phase5/
輸出：yolo_harmony_v2_phase6_fixed/

Author: Claude Code
Date: 2025-11-26
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from tqdm import tqdm

# 類別索引
BARLINE = 23
BARLINE_DOUBLE = 24
BARLINE_FINAL = 25
BARLINE_REPEAT = 26

# 修復參數
MIN_BARLINE_WIDTH = 0.015  # barline 最小寬度
THIN_BARLINE_THRESHOLD = 0.01  # 低於此寬度視為極細線
LARGE_AREA_THRESHOLD = 0.1  # 高於此面積視為過大框
MAX_REASONABLE_AREA = 0.05  # barline_double/final 合理最大面積

# 目錄配置
INPUT_DIR = Path("/home/thc1006/dev/music-app/training/datasets/yolo_harmony_v2_phase5")
OUTPUT_DIR = Path("/home/thc1006/dev/music-app/training/datasets/yolo_harmony_v2_phase6_fixed")


@dataclass
class AnnotationStats:
    """標註統計數據"""
    class_id: int
    total_count: int
    thin_count: int = 0  # 極細線數量（寬度 < threshold）
    large_count: int = 0  # 過大框數量（面積 > threshold）
    widths: List[float] = None
    heights: List[float] = None
    areas: List[float] = None
    aspect_ratios: List[float] = None

    def __post_init__(self):
        if self.widths is None:
            self.widths = []
        if self.heights is None:
            self.heights = []
        if self.areas is None:
            self.areas = []
        if self.aspect_ratios is None:
            self.aspect_ratios = []


@dataclass
class FixResult:
    """單個標註修復結果"""
    original_box: List[float]
    fixed_box: List[float]
    fix_type: str  # "expand_width", "shrink_area", "no_change"
    class_id: int


class BarlineAnnotationFixer:
    """Barline 標註修復器"""

    def __init__(self, input_dir: Path, output_dir: Path):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.stats_before = defaultdict(lambda: AnnotationStats(class_id=0, total_count=0))
        self.stats_after = defaultdict(lambda: AnnotationStats(class_id=0, total_count=0))
        self.fix_log = []

    def analyze_annotations(self, labels_dir: Path, stats_dict: Dict) -> None:
        """分析標註統計數據"""
        print(f"\n📊 分析標註: {labels_dir}")
        label_files = list(labels_dir.glob("*.txt"))

        for label_file in tqdm(label_files, desc="掃描標註檔"):
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue

                    class_id = int(parts[0])
                    if class_id not in [BARLINE, BARLINE_DOUBLE, BARLINE_FINAL]:
                        continue

                    # 解析 YOLO 格式 (x_center, y_center, width, height - 歸一化)
                    x_center, y_center, width, height = map(float, parts[1:5])
                    area = width * height
                    aspect_ratio = height / width if width > 0 else 0

                    # 更新統計
                    stats = stats_dict[class_id]
                    stats.class_id = class_id
                    stats.total_count += 1
                    stats.widths.append(width)
                    stats.heights.append(height)
                    stats.areas.append(area)
                    stats.aspect_ratios.append(aspect_ratio)

                    # 統計問題數量
                    if class_id == BARLINE and width < THIN_BARLINE_THRESHOLD:
                        stats.thin_count += 1
                    elif class_id in [BARLINE_DOUBLE, BARLINE_FINAL] and area > LARGE_AREA_THRESHOLD:
                        stats.large_count += 1

    def fix_barline_width(self, x_center: float, width: float, height: float) -> Tuple[float, float, str]:
        """修復極細線 barline 的寬度"""
        if width >= THIN_BARLINE_THRESHOLD:
            return x_center, width, "no_change"

        # 擴大寬度到最小值，保持中心點不變
        new_width = MIN_BARLINE_WIDTH

        # 確保不超出邊界 [0, 1]
        half_width = new_width / 2
        if x_center - half_width < 0:
            x_center = half_width
        elif x_center + half_width > 1:
            x_center = 1 - half_width

        return x_center, new_width, "expand_width"

    def fix_large_barline_area(self, x_center: float, y_center: float,
                                width: float, height: float,
                                class_id: int) -> Tuple[float, float, float, float, str]:
        """修復過大的 barline_double/final 標註框"""
        area = width * height
        if area <= LARGE_AREA_THRESHOLD:
            return x_center, y_center, width, height, "no_change"

        # 基於寬高比分析，智能緊縮
        aspect_ratio = height / width if width > 0 else 1

        # barline_double/final 通常是垂直的，高度 >> 寬度
        if aspect_ratio > 5:
            # 垂直主導：保留高度，緊縮寬度
            # 合理寬度應該約為高度的 1/10 到 1/20
            new_width = min(width, height / 15)
            new_width = max(new_width, 0.01)  # 最小寬度 0.01
            new_height = height

            # 檢查新面積是否合理
            new_area = new_width * new_height
            if new_area > MAX_REASONABLE_AREA:
                # 進一步緊縮高度
                scale_factor = np.sqrt(MAX_REASONABLE_AREA / new_area)
                new_height *= scale_factor
                new_width *= scale_factor

        elif aspect_ratio < 0.2:
            # 水平主導（異常情況，barline 應該是垂直的）
            # 保留寬度，緊縮高度
            new_height = min(height, width / 15)
            new_height = max(new_height, 0.01)
            new_width = width

            new_area = new_width * new_height
            if new_area > MAX_REASONABLE_AREA:
                scale_factor = np.sqrt(MAX_REASONABLE_AREA / new_area)
                new_height *= scale_factor
                new_width *= scale_factor

        else:
            # 寬高比接近 1:1（罕見，可能是錯誤標註）
            # 保守縮小到合理面積
            scale_factor = np.sqrt(MAX_REASONABLE_AREA / area)
            new_width = width * scale_factor
            new_height = height * scale_factor

        # 確保不超出邊界
        new_width = min(new_width, 1.0)
        new_height = min(new_height, 1.0)

        # 調整中心點以保持在邊界內
        half_w, half_h = new_width / 2, new_height / 2
        x_center = max(half_w, min(1 - half_w, x_center))
        y_center = max(half_h, min(1 - half_h, y_center))

        return x_center, y_center, new_width, new_height, "shrink_area"

    def fix_label_file(self, label_file: Path, output_file: Path) -> List[FixResult]:
        """修復單個標註文件"""
        fixes = []
        fixed_lines = []

        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    fixed_lines.append(line)
                    continue

                class_id = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:5])
                original_box = [x_center, y_center, width, height]

                # 應用修復邏輯
                if class_id == BARLINE:
                    new_x, new_width, fix_type = self.fix_barline_width(x_center, width, height)
                    new_y, new_height = y_center, height

                elif class_id in [BARLINE_DOUBLE, BARLINE_FINAL]:
                    new_x, new_y, new_width, new_height, fix_type = self.fix_large_barline_area(
                        x_center, y_center, width, height, class_id
                    )

                else:
                    # 其他類別不修復
                    new_x, new_y, new_width, new_height = x_center, y_center, width, height
                    fix_type = "no_change"

                # 記錄修復結果
                if fix_type != "no_change":
                    fixes.append(FixResult(
                        original_box=original_box,
                        fixed_box=[new_x, new_y, new_width, new_height],
                        fix_type=fix_type,
                        class_id=class_id
                    ))

                # 寫入修復後的標註
                fixed_line = f"{class_id} {new_x:.6f} {new_y:.6f} {new_width:.6f} {new_height:.6f}\n"
                fixed_lines.append(fixed_line)

        # 寫入輸出文件
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            f.writelines(fixed_lines)

        return fixes

    def process_dataset(self, split: str) -> None:
        """處理數據集的某個分割（train 或 val）"""
        print(f"\n🔧 處理 {split} 分割...")

        input_labels_dir = self.input_dir / split / "labels"
        input_images_dir = self.input_dir / split / "images"
        output_labels_dir = self.output_dir / split / "labels"
        output_images_dir = self.output_dir / split / "images"

        # 創建輸出目錄
        output_labels_dir.mkdir(parents=True, exist_ok=True)
        output_images_dir.mkdir(parents=True, exist_ok=True)

        # 處理標註文件
        label_files = list(input_labels_dir.glob("*.txt"))
        print(f"📝 修復 {len(label_files)} 個標註檔...")

        split_fixes = []
        for label_file in tqdm(label_files, desc=f"修復 {split}"):
            output_file = output_labels_dir / label_file.name
            fixes = self.fix_label_file(label_file, output_file)
            split_fixes.extend(fixes)

            # 複製對應的圖片
            image_name = label_file.stem + ".png"
            input_image = input_images_dir / image_name
            output_image = output_images_dir / image_name

            if input_image.exists():
                shutil.copy2(input_image, output_image)

        print(f"✅ {split} 修復完成：{len(split_fixes)} 個標註被修改")
        self.fix_log.extend(split_fixes)

    def generate_statistics_report(self) -> str:
        """生成統計報告"""
        report = []
        report.append("=" * 80)
        report.append("BARLINE 標註修復報告")
        report.append("=" * 80)
        report.append(f"輸入數據集: {self.input_dir}")
        report.append(f"輸出數據集: {self.output_dir}")
        report.append(f"總修復數量: {len(self.fix_log)}\n")

        # 修復前統計
        report.append("-" * 80)
        report.append("修復前統計")
        report.append("-" * 80)

        for class_id in [BARLINE, BARLINE_DOUBLE, BARLINE_FINAL]:
            stats = self.stats_before[class_id]
            if stats.total_count == 0:
                continue

            class_names = {BARLINE: "barline", BARLINE_DOUBLE: "barline_double", BARLINE_FINAL: "barline_final"}
            report.append(f"\n【{class_names[class_id]} (ID {class_id})】")
            report.append(f"  總數: {stats.total_count}")

            if stats.widths:
                report.append(f"  寬度: min={min(stats.widths):.6f}, max={max(stats.widths):.6f}, "
                            f"avg={np.mean(stats.widths):.6f}, median={np.median(stats.widths):.6f}")
                report.append(f"  高度: min={min(stats.heights):.6f}, max={max(stats.heights):.6f}, "
                            f"avg={np.mean(stats.heights):.6f}, median={np.median(stats.heights):.6f}")
                report.append(f"  面積: min={min(stats.areas):.6f}, max={max(stats.areas):.6f}, "
                            f"avg={np.mean(stats.areas):.6f}, median={np.median(stats.areas):.6f}")

            if class_id == BARLINE:
                report.append(f"  ⚠️ 極細線（寬度 < {THIN_BARLINE_THRESHOLD}）: {stats.thin_count} "
                            f"({stats.thin_count/stats.total_count*100:.1f}%)")
            else:
                report.append(f"  ⚠️ 過大框（面積 > {LARGE_AREA_THRESHOLD}）: {stats.large_count} "
                            f"({stats.large_count/stats.total_count*100:.1f}%)")

        # 修復後統計
        report.append("\n" + "-" * 80)
        report.append("修復後統計")
        report.append("-" * 80)

        for class_id in [BARLINE, BARLINE_DOUBLE, BARLINE_FINAL]:
            stats = self.stats_after[class_id]
            if stats.total_count == 0:
                continue

            class_names = {BARLINE: "barline", BARLINE_DOUBLE: "barline_double", BARLINE_FINAL: "barline_final"}
            report.append(f"\n【{class_names[class_id]} (ID {class_id})】")
            report.append(f"  總數: {stats.total_count}")

            if stats.widths:
                report.append(f"  寬度: min={min(stats.widths):.6f}, max={max(stats.widths):.6f}, "
                            f"avg={np.mean(stats.widths):.6f}, median={np.median(stats.widths):.6f}")
                report.append(f"  高度: min={min(stats.heights):.6f}, max={max(stats.heights):.6f}, "
                            f"avg={np.mean(stats.heights):.6f}, median={np.median(stats.heights):.6f}")
                report.append(f"  面積: min={min(stats.areas):.6f}, max={max(stats.areas):.6f}, "
                            f"avg={np.mean(stats.areas):.6f}, median={np.median(stats.areas):.6f}")

            if class_id == BARLINE:
                report.append(f"  ✅ 極細線（寬度 < {THIN_BARLINE_THRESHOLD}）: {stats.thin_count} "
                            f"({stats.thin_count/stats.total_count*100:.1f}%)")
            else:
                report.append(f"  ✅ 過大框（面積 > {LARGE_AREA_THRESHOLD}）: {stats.large_count} "
                            f"({stats.large_count/stats.total_count*100:.1f}%)")

        # 修復摘要
        report.append("\n" + "-" * 80)
        report.append("修復摘要")
        report.append("-" * 80)

        fix_types = defaultdict(int)
        for fix in self.fix_log:
            fix_types[fix.fix_type] += 1

        report.append(f"  擴大寬度 (expand_width): {fix_types['expand_width']}")
        report.append(f"  緊縮面積 (shrink_area): {fix_types['shrink_area']}")
        report.append(f"  無修改 (no_change): {fix_types['no_change']}")

        report.append("\n" + "=" * 80)

        return "\n".join(report)

    def visualize_comparison(self, sample_size: int = 10) -> None:
        """可視化修復前後對比"""
        print(f"\n📊 生成可視化對比（抽樣 {sample_size} 個修復案例）...")

        # 篩選有代表性的修復案例
        expand_fixes = [f for f in self.fix_log if f.fix_type == "expand_width"]
        shrink_fixes = [f for f in self.fix_log if f.fix_type == "shrink_area"]

        # 隨機抽樣
        sample_expand = np.random.choice(expand_fixes, min(sample_size // 2, len(expand_fixes)), replace=False)
        sample_shrink = np.random.choice(shrink_fixes, min(sample_size // 2, len(shrink_fixes)), replace=False)

        # 創建對比圖
        fig, axes = plt.subplots(2, sample_size // 2, figsize=(20, 8))
        fig.suptitle("Barline 標註修復前後對比", fontsize=16)

        for idx, fix in enumerate(sample_expand):
            ax = axes[0, idx] if sample_size // 2 > 1 else axes[0]
            self._draw_comparison(ax, fix, "擴大寬度")

        for idx, fix in enumerate(sample_shrink):
            ax = axes[1, idx] if sample_size // 2 > 1 else axes[1]
            self._draw_comparison(ax, fix, "緊縮面積")

        plt.tight_layout()
        output_path = self.output_dir / "fix_comparison.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ 可視化保存至: {output_path}")
        plt.close()

        # 生成分佈對比圖
        self._plot_distribution_comparison()

    def _draw_comparison(self, ax, fix: FixResult, title: str) -> None:
        """繪製單個修復對比"""
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_title(title, fontsize=10)

        # 繪製修復前（紅色）
        x1, y1, w1, h1 = fix.original_box
        rect_before = patches.Rectangle(
            (x1 - w1/2, y1 - h1/2), w1, h1,
            linewidth=2, edgecolor='red', facecolor='none', label='修復前'
        )
        ax.add_patch(rect_before)

        # 繪製修復後（綠色）
        x2, y2, w2, h2 = fix.fixed_box
        rect_after = patches.Rectangle(
            (x2 - w2/2, y2 - h2/2), w2, h2,
            linewidth=2, edgecolor='green', facecolor='none', label='修復後'
        )
        ax.add_patch(rect_after)

        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    def _plot_distribution_comparison(self) -> None:
        """繪製分佈對比圖"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle("Barline 類別標註分佈對比（修復前 vs 修復後）", fontsize=16)

        class_names = {BARLINE: "barline", BARLINE_DOUBLE: "barline_double", BARLINE_FINAL: "barline_final"}

        for idx, class_id in enumerate([BARLINE, BARLINE_DOUBLE, BARLINE_FINAL]):
            stats_before = self.stats_before[class_id]
            stats_after = self.stats_after[class_id]

            # 寬度分佈
            ax1 = axes[0, idx]
            if stats_before.widths and stats_after.widths:
                ax1.hist(stats_before.widths, bins=50, alpha=0.5, label='修復前', color='red')
                ax1.hist(stats_after.widths, bins=50, alpha=0.5, label='修復後', color='green')
                ax1.set_xlabel('Width')
                ax1.set_ylabel('Count')
                ax1.set_title(f'{class_names[class_id]} - 寬度分佈')
                ax1.legend()
                ax1.grid(True, alpha=0.3)

            # 面積分佈
            ax2 = axes[1, idx]
            if stats_before.areas and stats_after.areas:
                ax2.hist(stats_before.areas, bins=50, alpha=0.5, label='修復前', color='red')
                ax2.hist(stats_after.areas, bins=50, alpha=0.5, label='修復後', color='green')
                ax2.set_xlabel('Area')
                ax2.set_ylabel('Count')
                ax2.set_title(f'{class_names[class_id]} - 面積分佈')
                ax2.legend()
                ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = self.output_dir / "distribution_comparison.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ 分佈對比圖保存至: {output_path}")
        plt.close()

    def run(self) -> None:
        """執行完整的修復流程"""
        print("=" * 80)
        print("🚀 開始 Barline 標註修復流程")
        print("=" * 80)

        # Step 1: 分析修復前的數據
        print("\n📊 階段 1: 分析修復前的數據")
        for split in ["train", "val"]:
            self.analyze_annotations(self.input_dir / split / "labels", self.stats_before)

        # Step 2: 執行修復
        print("\n🔧 階段 2: 執行修復")
        for split in ["train", "val"]:
            self.process_dataset(split)

        # Step 3: 分析修復後的數據
        print("\n📊 階段 3: 分析修復後的數據")
        for split in ["train", "val"]:
            self.analyze_annotations(self.output_dir / split / "labels", self.stats_after)

        # Step 4: 複製配置文件
        print("\n📋 階段 4: 複製配置文件")
        for config_file in ["harmony_phase5.yaml", "README.md"]:
            src = self.input_dir / config_file
            dst = self.output_dir / config_file.replace("phase5", "phase6_fixed")
            if src.exists():
                shutil.copy2(src, dst)
                print(f"  ✅ 複製: {config_file}")

        # Step 5: 生成報告
        print("\n📄 階段 5: 生成修復報告")
        report = self.generate_statistics_report()
        report_path = self.output_dir / "fix_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"✅ 報告保存至: {report_path}")

        # Step 6: 生成可視化
        print("\n📊 階段 6: 生成可視化對比")
        if len(self.fix_log) > 0:
            self.visualize_comparison(sample_size=10)

        print("\n" + "=" * 80)
        print("✅ Barline 標註修復完成!")
        print("=" * 80)
        print(f"輸出目錄: {self.output_dir}")
        print(f"修復報告: {report_path}")
        print(f"可視化圖: {self.output_dir / 'fix_comparison.png'}")
        print(f"分佈對比: {self.output_dir / 'distribution_comparison.png'}")


def main():
    """主函數"""
    fixer = BarlineAnnotationFixer(INPUT_DIR, OUTPUT_DIR)
    fixer.run()


if __name__ == "__main__":
    main()
