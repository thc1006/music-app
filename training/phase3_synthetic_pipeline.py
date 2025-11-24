#!/usr/bin/env python3
"""
🎼 Phase 3: 合成數據增強管道
============================

目標：突破 mAP50=0.509 的瓶頸，達到 0.70-0.75

核心策略：
1. 針對 mAP=0 的類別生成大量合成數據
2. 針對視覺混淆類別生成對比樣本
3. 修復驗證集分佈問題
4. 使用 Copy-Paste 增強稀有類別

使用方式：
    # 1. 安裝 LilyPond
    sudo apt install lilypond

    # 2. 生成合成數據
    python phase3_synthetic_pipeline.py --generate-all

    # 3. 準備 Phase 3 數據集
    python phase3_synthetic_pipeline.py --prepare-dataset

    # 4. 開始訓練
    python phase3_synthetic_pipeline.py --train
"""

import os
import sys
import json
import random
import shutil
import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import argparse

# ============== 瓶頸類別分析 ==============

@dataclass
class BottleneckClass:
    """瓶頸類別資訊"""
    class_id: int
    name: str
    current_map50: float
    val_samples: int
    train_samples: int
    target_synthetic: int  # 目標合成數量
    lilypond_generator: str  # LilyPond 生成函數名
    confusion_with: List[str] = field(default_factory=list)  # 混淆類別
    priority: str = "HIGH"  # HIGH, MEDIUM, LOW

# 基於 Phase 2 評估結果定義瓶頸類別
BOTTLENECK_CLASSES = {
    # === 完全失敗 (mAP50 = 0) - 最高優先級 ===
    16: BottleneckClass(
        class_id=16, name="accidental_double_sharp",
        current_map50=0.0, val_samples=19, train_samples=3096,
        target_synthetic=5000, lilypond_generator="double_sharp",
        confusion_with=["accidental_sharp"], priority="CRITICAL"
    ),
    17: BottleneckClass(
        class_id=17, name="accidental_double_flat",
        current_map50=0.0, val_samples=1, train_samples=1000,
        target_synthetic=5000, lilypond_generator="double_flat",
        confusion_with=["accidental_flat"], priority="CRITICAL"
    ),
    24: BottleneckClass(
        class_id=24, name="barline_double",
        current_map50=0.0, val_samples=72, train_samples=1620,
        target_synthetic=3000, lilypond_generator="barline_double",
        confusion_with=["barline", "barline_final"], priority="CRITICAL"
    ),

    # === 極低表現 (mAP50 < 0.2) ===
    29: BottleneckClass(
        class_id=29, name="fermata",
        current_map50=0.125, val_samples=808, train_samples=8765,
        target_synthetic=3000, lilypond_generator="fermata",
        priority="HIGH"
    ),
    25: BottleneckClass(
        class_id=25, name="barline_final",
        current_map50=0.136, val_samples=2229, train_samples=56457,
        target_synthetic=2000, lilypond_generator="barline_final",
        confusion_with=["barline", "barline_double"], priority="HIGH"
    ),
    5: BottleneckClass(
        class_id=5, name="flag_16th",
        current_map50=0.147, val_samples=258, train_samples=4664,
        target_synthetic=5000, lilypond_generator="flag_16th",
        confusion_with=["flag_8th", "flag_32nd"], priority="HIGH"
    ),
    15: BottleneckClass(
        class_id=15, name="accidental_natural",
        current_map50=0.187, val_samples=1317, train_samples=44615,
        target_synthetic=3000, lilypond_generator="natural",
        confusion_with=["accidental_sharp"], priority="HIGH"
    ),

    # === 低表現 (0.2 < mAP50 < 0.4) ===
    12: BottleneckClass(
        class_id=12, name="clef_tenor",
        current_map50=0.273, val_samples=125, train_samples=3740,
        target_synthetic=3000, lilypond_generator="clef_tenor",
        confusion_with=["clef_alto"], priority="MEDIUM"
    ),
    30: BottleneckClass(
        class_id=30, name="dynamic_soft",
        current_map50=0.280, val_samples=2631, train_samples=27739,
        target_synthetic=2000, lilypond_generator="dynamic_soft",
        confusion_with=["dynamic_loud"], priority="MEDIUM"
    ),
    31: BottleneckClass(
        class_id=31, name="dynamic_loud",
        current_map50=0.651, val_samples=10, train_samples=858,
        target_synthetic=5000, lilypond_generator="dynamic_loud",
        confusion_with=["dynamic_soft"], priority="HIGH"
    ),
    8: BottleneckClass(
        class_id=8, name="tie",
        current_map50=0.298, val_samples=2329, train_samples=56696,
        target_synthetic=2000, lilypond_generator="tie",
        priority="MEDIUM"
    ),
    23: BottleneckClass(
        class_id=23, name="barline",
        current_map50=0.308, val_samples=498, train_samples=9280,
        target_synthetic=2000, lilypond_generator="barline",
        confusion_with=["barline_final", "barline_double"], priority="MEDIUM"
    ),

    # === 中等表現需加強 ===
    6: BottleneckClass(
        class_id=6, name="flag_32nd",
        current_map50=0.532, val_samples=12, train_samples=2080,
        target_synthetic=4000, lilypond_generator="flag_32nd",
        confusion_with=["flag_16th"], priority="HIGH"
    ),
    7: BottleneckClass(
        class_id=7, name="augmentation_dot",
        current_map50=0.397, val_samples=3352, train_samples=84960,
        target_synthetic=2000, lilypond_generator="augmentation_dot",
        priority="MEDIUM"
    ),
}

# ============== LilyPond 生成器 ==============

class LilyPondGenerator:
    """LilyPond 樂譜生成器"""

    HEADER_TEMPLATE = r'''
\version "2.24.0"
\header {{ tagline = ##f }}

\paper {{
  indent = 0
  paper-width = {width}\mm
  paper-height = {height}\mm
  top-margin = 5\mm
  bottom-margin = 5\mm
  left-margin = 5\mm
  right-margin = 5\mm
}}

#(set-global-staff-size {staff_size})
'''

    # LilyPond 字體選項
    FONTS = [
        '',  # 默認 (emmentaler)
        r'\override Staff.TimeSignature.font-name = "Beethoven"',
        r'\override Staff.Clef.font-name = "Gonville"',
    ]

    @staticmethod
    def double_sharp(variation: dict) -> Tuple[str, List[dict]]:
        """生成重升記號樂譜，返回 (LilyPond 代碼, 預期符號列表)"""
        notes = ["cisis", "disis", "eisis", "fisis", "gisis", "aisis", "bisis"]
        octaves = ["", "'", "''"]
        durations = ["4", "2", "8", "1"]

        content = []
        symbols = []

        # 生成 8-16 個音符
        num_notes = random.randint(8, 16)
        for i in range(num_notes):
            note = random.choice(notes)
            octave = random.choice(octaves)
            duration = random.choice(durations)
            content.append(f"{note}{octave}{duration}")
            symbols.append({
                "class_id": 16,
                "class_name": "accidental_double_sharp",
                "index": i
            })

        score = f"\\relative c' {{ {' '.join(content)} }}"
        return score, symbols

    @staticmethod
    def double_flat(variation: dict) -> Tuple[str, List[dict]]:
        """生成重降記號樂譜"""
        notes = ["ceses", "deses", "eeses", "feses", "geses", "aeses", "beses"]
        octaves = ["", "'", "''"]
        durations = ["4", "2", "8", "1"]

        content = []
        symbols = []

        num_notes = random.randint(8, 16)
        for i in range(num_notes):
            note = random.choice(notes)
            octave = random.choice(octaves)
            duration = random.choice(durations)
            content.append(f"{note}{octave}{duration}")
            symbols.append({
                "class_id": 17,
                "class_name": "accidental_double_flat",
                "index": i
            })

        score = f"\\relative c' {{ {' '.join(content)} }}"
        return score, symbols

    @staticmethod
    def barline_double(variation: dict) -> Tuple[str, List[dict]]:
        """生成雙小節線樂譜"""
        notes = ["c", "d", "e", "f", "g", "a", "b"]

        content = []
        symbols = []

        # 生成多個小節，每個以雙小節線結束
        for i in range(random.randint(3, 6)):
            bar_notes = [f"{random.choice(notes)}'{random.choice(['4', '2'])}"
                        for _ in range(4)]
            content.append(' '.join(bar_notes))
            content.append(r'\bar "||"')
            symbols.append({
                "class_id": 24,
                "class_name": "barline_double",
                "index": i
            })

        score = f"\\relative c' {{ {' '.join(content)} }}"
        return score, symbols

    @staticmethod
    def barline_final(variation: dict) -> Tuple[str, List[dict]]:
        """生成終止線樂譜"""
        notes = ["c", "d", "e", "f", "g", "a", "b"]
        content = []

        # 生成一些音符
        for _ in range(8):
            content.append(f"{random.choice(notes)}'{random.choice(['4', '2', '8'])}")

        content.append(r'\bar "|."')

        score = f"\\relative c' {{ {' '.join(content)} }}"
        return score, [{"class_id": 25, "class_name": "barline_final", "index": 0}]

    @staticmethod
    def flag_16th(variation: dict) -> Tuple[str, List[dict]]:
        """生成十六分音符樂譜（單獨的，非連音）"""
        notes = ["c", "d", "e", "f", "g", "a", "b"]
        octaves = ["'", "''"]

        content = []
        symbols = []

        # 混合十六分音符和其他音符，確保十六分是單獨的
        for i in range(random.randint(12, 20)):
            note = random.choice(notes)
            octave = random.choice(octaves)

            if random.random() < 0.6:  # 60% 十六分音符
                content.append(f"{note}{octave}16")
                symbols.append({
                    "class_id": 5,
                    "class_name": "flag_16th",
                    "index": i
                })
            else:
                content.append(f"{note}{octave}{random.choice(['4', '8'])}")

        score = f"\\relative c' {{ {' '.join(content)} }}"
        return score, symbols

    @staticmethod
    def flag_32nd(variation: dict) -> Tuple[str, List[dict]]:
        """生成三十二分音符樂譜"""
        notes = ["c", "d", "e", "f", "g", "a", "b"]
        octaves = ["'", "''"]

        content = []
        symbols = []

        for i in range(random.randint(16, 24)):
            note = random.choice(notes)
            octave = random.choice(octaves)

            if random.random() < 0.5:  # 50% 三十二分
                content.append(f"{note}{octave}32")
                symbols.append({
                    "class_id": 6,
                    "class_name": "flag_32nd",
                    "index": i
                })
            else:
                content.append(f"{note}{octave}16")

        score = f"\\relative c' {{ {' '.join(content)} }}"
        return score, symbols

    @staticmethod
    def natural(variation: dict) -> Tuple[str, List[dict]]:
        """生成還原記號樂譜 - 在升降調中使用還原"""
        # 使用有升降號的調，然後加還原記號
        keys = [r"\key g \major", r"\key d \major", r"\key f \major", r"\key bes \major"]
        key = random.choice(keys)

        # 還原記號用 ! 表示 (c! = C還原)
        content = [key]
        symbols = []

        notes_with_natural = ["c!", "d!", "e!", "f!", "g!", "a!", "b!"]
        notes_normal = ["c", "d", "e", "f", "g", "a", "b"]

        for i in range(random.randint(10, 16)):
            if random.random() < 0.5:
                note = random.choice(notes_with_natural)
                symbols.append({
                    "class_id": 15,
                    "class_name": "accidental_natural",
                    "index": i
                })
            else:
                note = random.choice(notes_normal)

            content.append(f"{note}'{random.choice(['4', '2', '8'])}")

        score = f"\\relative c' {{ {' '.join(content)} }}"
        return score, symbols

    @staticmethod
    def fermata(variation: dict) -> Tuple[str, List[dict]]:
        """生成延長記號樂譜"""
        notes = ["c", "d", "e", "f", "g"]
        content = []
        symbols = []

        for i in range(random.randint(4, 8)):
            note = f"{random.choice(notes)}'{random.choice(['2', '1'])}"
            if random.random() < 0.6:
                content.append(f"{note}\\fermata")
                symbols.append({
                    "class_id": 29,
                    "class_name": "fermata",
                    "index": i
                })
            else:
                content.append(note)

        score = f"\\relative c' {{ {' '.join(content)} }}"
        return score, symbols

    @staticmethod
    def dynamic_loud(variation: dict) -> Tuple[str, List[dict]]:
        """生成強記號樂譜"""
        dynamics = [r"\f", r"\ff", r"\fff", r"\sf", r"\sfz", r"\fp"]
        notes = ["c", "d", "e", "f", "g"]

        content = []
        symbols = []

        for i in range(random.randint(8, 12)):
            note = f"{random.choice(notes)}'{random.choice(['4', '2'])}"
            dyn = random.choice(dynamics)
            content.append(f"{note}{dyn}")
            symbols.append({
                "class_id": 31,
                "class_name": "dynamic_loud",
                "index": i
            })

        score = f"\\relative c' {{ {' '.join(content)} }}"
        return score, symbols

    @staticmethod
    def dynamic_soft(variation: dict) -> Tuple[str, List[dict]]:
        """生成弱記號樂譜"""
        dynamics = [r"\p", r"\pp", r"\ppp", r"\mp"]
        notes = ["c", "d", "e", "f", "g"]

        content = []
        symbols = []

        for i in range(random.randint(8, 12)):
            note = f"{random.choice(notes)}'{random.choice(['4', '2'])}"
            dyn = random.choice(dynamics)
            content.append(f"{note}{dyn}")
            symbols.append({
                "class_id": 30,
                "class_name": "dynamic_soft",
                "index": i
            })

        score = f"\\relative c' {{ {' '.join(content)} }}"
        return score, symbols

    @staticmethod
    def clef_tenor(variation: dict) -> Tuple[str, List[dict]]:
        """生成次中音譜號樂譜"""
        notes = ["c", "d", "e", "f", "g", "a", "b"]

        content = [r"\clef tenor"]
        symbols = [{"class_id": 12, "class_name": "clef_tenor", "index": 0}]

        # 添加一些音符
        for _ in range(8):
            content.append(f"{random.choice(notes)}{random.choice(['4', '2', '8'])}")

        # 可能換譜號再換回來
        if random.random() < 0.5:
            content.append(r"\clef bass")
            for _ in range(4):
                content.append(f"{random.choice(notes)}{random.choice(['4', '2'])}")
            content.append(r"\clef tenor")
            symbols.append({"class_id": 12, "class_name": "clef_tenor", "index": 1})

        score = f"{{ {' '.join(content)} }}"
        return score, symbols

    @staticmethod
    def tie(variation: dict) -> Tuple[str, List[dict]]:
        """生成連結線樂譜"""
        notes = ["c", "d", "e", "f", "g"]
        content = []
        symbols = []

        for i in range(random.randint(4, 8)):
            note = random.choice(notes)
            octave = random.choice(["'", "''"])
            duration = random.choice(["4", "2"])

            # 連結線: note~ note
            content.append(f"{note}{octave}{duration}~")
            content.append(f"{note}{octave}{duration}")
            symbols.append({
                "class_id": 8,
                "class_name": "tie",
                "index": i
            })

        score = f"\\relative c' {{ {' '.join(content)} }}"
        return score, symbols

    @staticmethod
    def barline(variation: dict) -> Tuple[str, List[dict]]:
        """生成普通小節線樂譜"""
        notes = ["c", "d", "e", "f", "g", "a", "b"]
        content = []
        symbols = []

        for i in range(random.randint(4, 8)):
            # 一個小節的音符
            bar_notes = []
            for _ in range(4):
                bar_notes.append(f"{random.choice(notes)}'{random.choice(['4', '8'])}")
            content.append(' '.join(bar_notes))
            content.append('|')
            symbols.append({
                "class_id": 23,
                "class_name": "barline",
                "index": i
            })

        score = f"\\relative c' {{ {' '.join(content)} }}"
        return score, symbols

    @staticmethod
    def augmentation_dot(variation: dict) -> Tuple[str, List[dict]]:
        """生成附點音符樂譜"""
        notes = ["c", "d", "e", "f", "g", "a", "b"]
        content = []
        symbols = []

        for i in range(random.randint(8, 12)):
            note = random.choice(notes)
            octave = random.choice(["'", "''"])

            if random.random() < 0.7:  # 70% 附點
                duration = random.choice(["4.", "2.", "8."])
                symbols.append({
                    "class_id": 7,
                    "class_name": "augmentation_dot",
                    "index": i
                })
            else:
                duration = random.choice(["4", "2", "8"])

            content.append(f"{note}{octave}{duration}")

        score = f"\\relative c' {{ {' '.join(content)} }}"
        return score, symbols

    # 生成器映射
    GENERATORS = {
        "double_sharp": double_sharp.__func__,
        "double_flat": double_flat.__func__,
        "barline_double": barline_double.__func__,
        "barline_final": barline_final.__func__,
        "flag_16th": flag_16th.__func__,
        "flag_32nd": flag_32nd.__func__,
        "natural": natural.__func__,
        "fermata": fermata.__func__,
        "dynamic_loud": dynamic_loud.__func__,
        "dynamic_soft": dynamic_soft.__func__,
        "clef_tenor": clef_tenor.__func__,
        "tie": tie.__func__,
        "barline": barline.__func__,
        "augmentation_dot": augmentation_dot.__func__,
    }


# ============== 合成數據管道 ==============

class Phase3SyntheticPipeline:
    """Phase 3 合成數據生成管道"""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.images_dir = self.output_dir / "images"
        self.labels_dir = self.output_dir / "labels"
        self.metadata_dir = self.output_dir / "metadata"

        for d in [self.images_dir, self.labels_dir, self.metadata_dir]:
            d.mkdir(exist_ok=True)

        self.generator = LilyPondGenerator()
        self.stats = defaultdict(int)

    def check_lilypond(self) -> bool:
        """檢查 LilyPond 是否安裝"""
        try:
            result = subprocess.run(
                ['lilypond', '--version'],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                version = result.stdout.split('\n')[0]
                print(f"✅ LilyPond 已安裝: {version}")
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        print("❌ LilyPond 未安裝")
        print("   安裝方式: sudo apt install lilypond")
        return False

    def generate_lilypond_file(
        self,
        generator_name: str,
        file_id: str,
        variation: dict
    ) -> Tuple[Optional[Path], List[dict]]:
        """
        生成 LilyPond 文件

        Returns:
            (ly_file_path, expected_symbols)
        """
        if generator_name not in self.generator.GENERATORS:
            print(f"  ⚠️ 未知生成器: {generator_name}")
            return None, []

        # 獲取樂譜內容
        gen_func = self.generator.GENERATORS[generator_name]
        score_content, symbols = gen_func(variation)

        # 組裝 LilyPond 文件
        staff_size = variation.get('staff_size', 20)
        width = variation.get('width', 200)
        height = variation.get('height', 80)

        header = self.generator.HEADER_TEMPLATE.format(
            staff_size=staff_size,
            width=width,
            height=height
        )

        full_content = f"{header}\n{score_content}"

        # 寫入文件
        ly_path = self.output_dir / f"{file_id}.ly"
        with open(ly_path, 'w', encoding='utf-8') as f:
            f.write(full_content)

        return ly_path, symbols

    def render_to_png(self, ly_path: Path, output_name: str) -> Optional[Path]:
        """渲染 LilyPond 為 PNG"""
        try:
            # 使用 LilyPond 渲染
            result = subprocess.run(
                [
                    'lilypond',
                    '-dbackend=eps',
                    '-dno-gs-load-fonts',
                    '-dinclude-eps-fonts',
                    '--png',
                    '-dresolution=150',  # 設定解析度
                    '-o', str(self.output_dir / output_name),
                    str(ly_path)
                ],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=str(self.output_dir)
            )

            if result.returncode != 0:
                print(f"  ⚠️ LilyPond 錯誤: {result.stderr[:200]}")
                return None

            # LilyPond 可能生成多個頁面，取第一頁
            png_path = self.output_dir / f"{output_name}.png"
            if not png_path.exists():
                # 嘗試 page1
                page1_path = self.output_dir / f"{output_name}-page1.png"
                if page1_path.exists():
                    shutil.move(page1_path, png_path)

            if png_path.exists():
                return png_path

            print(f"  ⚠️ PNG 未生成")
            return None

        except subprocess.TimeoutExpired:
            print(f"  ⚠️ 渲染超時")
            return None
        except Exception as e:
            print(f"  ⚠️ 渲染錯誤: {e}")
            return None

    def generate_placeholder_annotations(
        self,
        symbols: List[dict],
        image_path: Path
    ) -> List[str]:
        """
        生成佔位符標註

        注意：這是簡化版本，實際需要：
        1. 使用 LilyPond 的 point-and-click 功能獲取精確座標
        2. 或使用模板匹配定位符號
        3. 或使用預訓練模型輔助標註

        目前使用隨機位置作為示例
        """
        annotations = []

        # 簡化：根據符號類型估算位置
        for i, sym in enumerate(symbols):
            class_id = sym['class_id']

            # 估算位置（實際需要更精確的方法）
            # x 位置基於符號索引
            x_center = 0.1 + (i % 10) * 0.08
            x_center = min(0.95, x_center)

            # y 位置在中間五線譜區域
            y_center = 0.4 + random.uniform(-0.15, 0.15)

            # 估算大小
            if class_id in [16, 17, 13, 14, 15]:  # 臨時記號
                w, h = 0.03, 0.06
            elif class_id in [23, 24, 25]:  # 小節線
                w, h = 0.01, 0.25
            elif class_id in [5, 6]:  # 旗標
                w, h = 0.02, 0.08
            elif class_id in [29]:  # fermata
                w, h = 0.04, 0.04
            elif class_id in [30, 31]:  # dynamics
                w, h = 0.04, 0.03
            elif class_id in [7]:  # dot
                w, h = 0.015, 0.015
            elif class_id in [8]:  # tie
                w, h = 0.08, 0.03
            elif class_id in [12]:  # tenor clef
                w, h = 0.04, 0.12
            else:
                w, h = 0.03, 0.05

            # YOLO 格式: class_id x_center y_center width height
            annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

        return annotations

    def generate_for_class(
        self,
        bottleneck: BottleneckClass,
        count: int,
        variations: Optional[List[dict]] = None
    ) -> int:
        """為特定瓶頸類別生成合成數據"""

        if variations is None:
            variations = [
                {'staff_size': 18, 'width': 180, 'height': 60},
                {'staff_size': 20, 'width': 200, 'height': 70},
                {'staff_size': 22, 'width': 220, 'height': 80},
                {'staff_size': 24, 'width': 240, 'height': 90},
            ]

        print(f"\n{'='*60}")
        print(f"生成 Class {bottleneck.class_id} ({bottleneck.name})")
        print(f"  當前 mAP50: {bottleneck.current_map50:.3f}")
        print(f"  目標數量: {count}")
        print(f"  生成器: {bottleneck.lilypond_generator}")
        print(f"{'='*60}")

        success_count = 0

        for i in range(count):
            variation = random.choice(variations)
            file_id = f"syn_{bottleneck.name}_{i:05d}"

            try:
                # 1. 生成 LilyPond 文件
                ly_path, symbols = self.generate_lilypond_file(
                    bottleneck.lilypond_generator,
                    file_id,
                    variation
                )

                if ly_path is None:
                    continue

                # 2. 渲染為 PNG
                png_path = self.render_to_png(ly_path, file_id)

                if png_path is None:
                    # 清理 .ly 文件
                    ly_path.unlink(missing_ok=True)
                    continue

                # 3. 移動 PNG 到 images 目錄
                final_png = self.images_dir / f"{file_id}.png"
                shutil.move(png_path, final_png)

                # 4. 生成標註
                annotations = self.generate_placeholder_annotations(symbols, final_png)

                label_path = self.labels_dir / f"{file_id}.txt"
                with open(label_path, 'w') as f:
                    f.write('\n'.join(annotations))

                # 5. 保存元數據
                metadata = {
                    "file_id": file_id,
                    "class_id": bottleneck.class_id,
                    "class_name": bottleneck.name,
                    "variation": variation,
                    "symbols": symbols,
                    "annotation_type": "placeholder"  # 標記需要人工驗證
                }

                meta_path = self.metadata_dir / f"{file_id}.json"
                with open(meta_path, 'w') as f:
                    json.dump(metadata, f, indent=2)

                # 6. 清理臨時文件
                ly_path.unlink(missing_ok=True)
                for tmp in self.output_dir.glob(f"{file_id}*"):
                    if tmp.suffix not in ['.png', '.json']:
                        tmp.unlink(missing_ok=True)

                success_count += 1
                self.stats[bottleneck.class_id] += 1

                if success_count % 100 == 0:
                    print(f"  進度: {success_count}/{count}")

            except Exception as e:
                print(f"  ⚠️ 生成失敗 ({file_id}): {e}")
                continue

        print(f"  ✅ 完成: {success_count}/{count}")
        return success_count

    def generate_all_bottleneck_classes(self):
        """生成所有瓶頸類別的合成數據"""

        print("\n" + "="*70)
        print("Phase 3: 合成數據生成")
        print("="*70)

        if not self.check_lilypond():
            print("\n⚠️ 請先安裝 LilyPond:")
            print("   sudo apt install lilypond")
            return

        total_target = sum(b.target_synthetic for b in BOTTLENECK_CLASSES.values())
        print(f"\n目標總數: {total_target:,} 張合成圖片")
        print(f"涵蓋類別: {len(BOTTLENECK_CLASSES)} 個瓶頸類別\n")

        # 按優先級排序
        priority_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        sorted_classes = sorted(
            BOTTLENECK_CLASSES.values(),
            key=lambda x: priority_order.get(x.priority, 99)
        )

        for bottleneck in sorted_classes:
            self.generate_for_class(bottleneck, bottleneck.target_synthetic)

        # 打印統計
        print("\n" + "="*70)
        print("生成統計")
        print("="*70)

        for class_id, count in sorted(self.stats.items()):
            name = BOTTLENECK_CLASSES[class_id].name
            target = BOTTLENECK_CLASSES[class_id].target_synthetic
            print(f"  Class {class_id:2d} ({name:25s}): {count:5d}/{target:5d}")

        print(f"\n  總計: {sum(self.stats.values()):,} 張")


# ============== Phase 3 訓練配置 ==============

def create_phase3_dataset_config(
    original_dataset: Path,
    synthetic_dataset: Path,
    output_path: Path
):
    """創建 Phase 3 混合數據集配置"""

    config = f"""# Phase 3: 合成數據增強數據集
# 混合原始 Phase 2 數據 + 合成瓶頸類別數據

path: {output_path}
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

    yaml_path = output_path / "harmony_phase3.yaml"
    with open(yaml_path, 'w') as f:
        f.write(config)

    print(f"✅ 數據集配置已保存: {yaml_path}")
    return yaml_path


def create_phase3_training_script(base_dir: Path):
    """創建 Phase 3 訓練腳本"""

    script = '''#!/usr/bin/env python3
"""
Phase 3 訓練腳本 - 合成數據增強
"""
from ultralytics import YOLO
from pathlib import Path

def train_phase3():
    # 從 Phase 2 best.pt 繼續訓練
    model = YOLO('harmony_omr_v2_phase2/balanced_training/weights/best.pt')

    results = model.train(
        data='datasets/yolo_harmony_v2_phase3/harmony_phase3.yaml',
        epochs=200,
        batch=16,
        imgsz=640,

        # 針對瓶頸類別的優化
        patience=50,  # 更長的耐心

        # 學習率
        lr0=0.0005,   # 較低初始學習率（微調）
        lrf=0.0001,

        # 增強（更溫和，因為合成數據已有變異）
        mosaic=0.3,
        mixup=0.1,
        copy_paste=0.2,  # 重要：用於稀有類別增強

        # 其他
        project='harmony_omr_v2_phase3',
        name='synthetic_enhanced',
        exist_ok=True,

        # 類別權重（如果支持）
        # cls_pw=class_weights,  # 需要檢查 ultralytics 版本

        cos_lr=True,
        amp=True,
        verbose=True,
    )

    return results

if __name__ == '__main__':
    train_phase3()
'''

    script_path = base_dir / "yolo12_train_phase3.py"
    with open(script_path, 'w') as f:
        f.write(script)

    print(f"✅ 訓練腳本已保存: {script_path}")
    return script_path


# ============== 主程序 ==============

def main():
    parser = argparse.ArgumentParser(description='Phase 3: 合成數據增強管道')
    parser.add_argument('--check-lilypond', action='store_true',
                       help='檢查 LilyPond 安裝')
    parser.add_argument('--generate-all', action='store_true',
                       help='生成所有瓶頸類別的合成數據')
    parser.add_argument('--generate-class', type=int,
                       help='生成特定類別的合成數據')
    parser.add_argument('--count', type=int, default=1000,
                       help='生成數量（用於 --generate-class）')
    parser.add_argument('--prepare-dataset', action='store_true',
                       help='準備 Phase 3 混合數據集')
    parser.add_argument('--create-scripts', action='store_true',
                       help='創建訓練腳本')
    parser.add_argument('--output', type=str,
                       default='datasets/yolo_harmony_v2_phase3_synthetic',
                       help='合成數據輸出目錄')
    parser.add_argument('--status', action='store_true',
                       help='顯示瓶頸類別狀態')

    args = parser.parse_args()

    # 切換到訓練目錄
    training_dir = Path(__file__).parent
    os.chdir(training_dir)

    if args.status:
        print("\n" + "="*80)
        print("Phase 3 瓶頸類別狀態")
        print("="*80)
        print(f"{'ID':>3} {'名稱':25s} {'mAP50':>8} {'驗證':>8} {'訓練':>10} {'目標合成':>10} {'優先級':>10}")
        print("-"*80)

        for class_id, b in sorted(BOTTLENECK_CLASSES.items()):
            print(f"{class_id:>3} {b.name:25s} {b.current_map50:>8.3f} "
                  f"{b.val_samples:>8,} {b.train_samples:>10,} "
                  f"{b.target_synthetic:>10,} {b.priority:>10}")

        total = sum(b.target_synthetic for b in BOTTLENECK_CLASSES.values())
        print("-"*80)
        print(f"{'':>3} {'總計':25s} {'':>8} {'':>8} {'':>10} {total:>10,}")
        return

    if args.check_lilypond:
        pipeline = Phase3SyntheticPipeline(Path(args.output))
        pipeline.check_lilypond()
        return

    if args.generate_all:
        pipeline = Phase3SyntheticPipeline(Path(args.output))
        pipeline.generate_all_bottleneck_classes()
        return

    if args.generate_class is not None:
        if args.generate_class not in BOTTLENECK_CLASSES:
            print(f"❌ 未知類別 ID: {args.generate_class}")
            print(f"   可用類別: {list(BOTTLENECK_CLASSES.keys())}")
            return

        pipeline = Phase3SyntheticPipeline(Path(args.output))
        if not pipeline.check_lilypond():
            return

        bottleneck = BOTTLENECK_CLASSES[args.generate_class]
        pipeline.generate_for_class(bottleneck, args.count)
        return

    if args.create_scripts:
        create_phase3_training_script(training_dir)
        return

    # 默認：顯示幫助
    parser.print_help()
    print("\n" + "="*60)
    print("Phase 3 工作流程")
    print("="*60)
    print("""
1. 檢查 LilyPond 安裝:
   python phase3_synthetic_pipeline.py --check-lilypond

2. 查看瓶頸類別狀態:
   python phase3_synthetic_pipeline.py --status

3. 生成所有瓶頸類別數據:
   python phase3_synthetic_pipeline.py --generate-all

4. 生成特定類別數據:
   python phase3_synthetic_pipeline.py --generate-class 17 --count 1000

5. 創建訓練腳本:
   python phase3_synthetic_pipeline.py --create-scripts

6. 開始 Phase 3 訓練:
   python yolo12_train_phase3.py
""")


if __name__ == '__main__':
    main()
