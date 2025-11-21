#!/usr/bin/env python3
"""
🎼 合成數據生成器 - 解決稀有類別問題的核心工具

用途：生成包含特定音樂符號的訓練數據
目標：讓每個類別都有足夠的訓練樣本

使用方法：
    python synthetic_data_generator.py --target-class 17 --count 1000
    python synthetic_data_generator.py --balance-all --min-samples 5000
"""

import os
import subprocess
import random
import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional
import argparse

# ============== 配置 ==============

@dataclass
class ClassInfo:
    """類別資訊"""
    id: int
    name: str
    lilypond_syntax: str
    description: str
    current_count: int
    target_count: int

# 33 類對應的 LilyPond 語法
CLASS_DEFINITIONS = {
    0: ClassInfo(0, "notehead_filled", "c4", "實心音符頭", 501814, 100000),
    1: ClassInfo(1, "notehead_hollow", "c2", "空心音符頭", 58826, 50000),
    2: ClassInfo(2, "stem", "c4", "符幹（自動生成）", 471390, 100000),
    3: ClassInfo(3, "beam", "c8[ d8]", "符槓", 138032, 50000),
    4: ClassInfo(4, "flag_8th", "c'8", "八分音符旗", 30378, 30000),
    5: ClassInfo(5, "flag_16th", "c'16", "十六分音符旗", 3164, 10000),
    6: ClassInfo(6, "flag_32nd", "c'32", "三十二分音符旗", 440, 10000),
    7: ClassInfo(7, "augmentation_dot", "c4.", "附點", 37446, 30000),
    8: ClassInfo(8, "tie", "c4~ c4", "連結線", 10227, 10000),
    9: ClassInfo(9, "clef_treble", r"\clef treble", "高音譜號", 8332, 10000),
    10: ClassInfo(10, "clef_bass", r"\clef bass", "低音譜號", 10810, 10000),
    11: ClassInfo(11, "clef_alto", r"\clef alto", "中音譜號", 2644, 10000),
    12: ClassInfo(12, "clef_tenor", r"\clef tenor", "次中音譜號", 614, 10000),
    13: ClassInfo(13, "accidental_sharp", "cis", "升記號", 27645, 30000),
    14: ClassInfo(14, "accidental_flat", "ces", "降記號", 19568, 30000),
    15: ClassInfo(15, "accidental_natural", "c!", "還原記號", 12705, 20000),
    16: ClassInfo(16, "accidental_double_sharp", "cisis", "重升記號", 338, 10000),
    17: ClassInfo(17, "accidental_double_flat", "ceses", "重降記號", 12, 10000),
    18: ClassInfo(18, "rest_whole", "r1", "全休止符", 22699, 20000),
    19: ClassInfo(19, "rest_half", "r2", "二分休止符", 5030, 10000),
    20: ClassInfo(20, "rest_quarter", "r4", "四分休止符", 15158, 15000),
    21: ClassInfo(21, "rest_8th", "r8", "八分休止符", 19130, 15000),
    22: ClassInfo(22, "rest_16th", "r16", "十六分休止符", 6941, 10000),
    23: ClassInfo(23, "barline", "|", "小節線", 5572, 10000),
    24: ClassInfo(24, "barline_double", r'\bar "||"', "雙小節線", 234, 5000),
    25: ClassInfo(25, "barline_final", r'\bar "|."', "終止線", 9937, 10000),
    26: ClassInfo(26, "barline_repeat", r'\bar ":|."', "反覆記號", 8788, 10000),
    27: ClassInfo(27, "time_signature", r"\time 4/4", "拍號", 16565, 15000),
    28: ClassInfo(28, "key_signature", r"\key g \major", "調號", 24146, 20000),
    29: ClassInfo(29, "fermata", r"\fermata", "延長記號", 3976, 5000),
    30: ClassInfo(30, "dynamic_soft", r"\p", "弱記號", 12537, 10000),
    31: ClassInfo(31, "dynamic_loud", r"\f", "強記號", 27, 10000),
    32: ClassInfo(32, "ledger_line", "c''4", "加線（高音）", 176632, 50000),
}

# ============== LilyPond 模板 ==============

LILYPOND_HEADER = r'''
\version "2.24.0"
\header {
  tagline = ##f
}

\paper {
  indent = 0
  paper-width = 200\mm
  paper-height = 60\mm
  top-margin = 5\mm
  bottom-margin = 5\mm
  left-margin = 5\mm
  right-margin = 5\mm
}

#(set-global-staff-size {staff_size})
'''

# 生成特定類別的 LilyPond 片段
def generate_double_flat_score():
    """生成包含重降記號的樂譜"""
    notes = ["ceses", "deses", "eeses", "feses", "geses", "aeses", "beses"]
    octaves = ["", "'", "''"]
    durations = ["4", "2", "8"]

    score_content = []
    for _ in range(random.randint(8, 16)):
        note = random.choice(notes)
        octave = random.choice(octaves)
        duration = random.choice(durations)
        score_content.append(f"{note}{octave}{duration}")

    return " ".join(score_content)

def generate_double_sharp_score():
    """生成包含重升記號的樂譜"""
    notes = ["cisis", "disis", "eisis", "fisis", "gisis", "aisis", "bisis"]
    octaves = ["", "'", "''"]
    durations = ["4", "2", "8"]

    score_content = []
    for _ in range(random.randint(8, 16)):
        note = random.choice(notes)
        octave = random.choice(octaves)
        duration = random.choice(durations)
        score_content.append(f"{note}{octave}{duration}")

    return " ".join(score_content)

def generate_tenor_clef_score():
    """生成次中音譜號樂譜"""
    return r'''
\clef tenor
c4 d e f | g a b c' | d' e' f' g' |
'''

def generate_dynamic_loud_score():
    """生成強記號樂譜"""
    dynamics = [r"\f", r"\ff", r"\fff", r"\sfz", r"\sf"]
    notes = []
    for _ in range(8):
        dyn = random.choice(dynamics)
        notes.append(f"c4{dyn}")
    return " ".join(notes)

def generate_flag_32nd_score():
    """生成三十二分音符樂譜"""
    notes = ["c", "d", "e", "f", "g", "a", "b"]
    score = []
    for _ in range(16):
        note = random.choice(notes)
        octave = random.choice(["'", "''"])
        score.append(f"{note}{octave}32")
    return " ".join(score)

def generate_barline_double_score():
    """生成雙小節線樂譜"""
    return r'''
c4 d e f \bar "||" g a b c' \bar "||" d' e' f' g' \bar "||"
'''

# 類別對應的生成器
SCORE_GENERATORS = {
    17: generate_double_flat_score,
    16: generate_double_sharp_score,
    12: generate_tenor_clef_score,
    31: generate_dynamic_loud_score,
    6: generate_flag_32nd_score,
    24: generate_barline_double_score,
}

def generate_lilypond_file(class_id: int, output_path: Path, variation: Dict) -> str:
    """
    生成 LilyPond 源文件

    Args:
        class_id: 目標類別 ID
        output_path: 輸出路徑
        variation: 變體配置 (字體大小等)

    Returns:
        LilyPond 文件內容
    """
    # 獲取樂譜內容
    if class_id in SCORE_GENERATORS:
        score_content = SCORE_GENERATORS[class_id]()
    else:
        # 默認生成器
        score_content = "c4 d e f | g a b c' |"

    # 組裝完整 LilyPond 文件
    staff_size = variation.get('staff_size', 20)
    header = LILYPOND_HEADER.format(staff_size=staff_size)

    full_content = f'''
{header}

\\relative c' {{
  {score_content}
}}
'''

    # 寫入文件
    ly_path = output_path.with_suffix('.ly')
    with open(ly_path, 'w') as f:
        f.write(full_content)

    return str(ly_path)

def render_lilypond(ly_path: str, output_dir: Path) -> Optional[str]:
    """
    渲染 LilyPond 文件為 PNG

    Args:
        ly_path: LilyPond 文件路徑
        output_dir: 輸出目錄

    Returns:
        PNG 文件路徑，失敗返回 None
    """
    try:
        result = subprocess.run(
            ['lilypond', '-dbackend=eps', '-dno-gs-load-fonts',
             '-dinclude-eps-fonts', '--png', '-o', str(output_dir / 'output'),
             ly_path],
            capture_output=True,
            timeout=30
        )
        if result.returncode == 0:
            # LilyPond 生成的文件名
            png_path = output_dir / 'output.png'
            if png_path.exists():
                return str(png_path)
    except Exception as e:
        print(f"渲染失敗: {e}")

    return None

def generate_yolo_annotations(png_path: str, class_id: int) -> List[str]:
    """
    生成 YOLO 格式標註（簡化版，實際需要更精確的定位）

    注意：這是簡化版本，實際使用需要：
    1. 解析 LilyPond 輸出的精確座標
    2. 或使用圖像處理自動檢測符號位置

    Args:
        png_path: PNG 圖片路徑
        class_id: 類別 ID

    Returns:
        YOLO 格式標註列表
    """
    # TODO: 實作精確標註
    # 這裡需要：
    # 1. 讀取 LilyPond 的 EPS 輸出獲取座標
    # 2. 或使用模板匹配定位符號

    # 暫時返回佔位符
    annotations = []

    # 示例：假設我們知道符號的大致位置
    # 實際應用需要更精確的方法
    annotations.append(f"{class_id} 0.5 0.5 0.1 0.1")

    return annotations

# ============== 主程序 ==============

class SyntheticDataGenerator:
    """合成數據生成器主類"""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 創建子目錄
        self.images_dir = output_dir / 'images'
        self.labels_dir = output_dir / 'labels'
        self.images_dir.mkdir(exist_ok=True)
        self.labels_dir.mkdir(exist_ok=True)

    def generate_for_class(self, class_id: int, count: int,
                          variations: List[Dict] = None) -> int:
        """
        為特定類別生成合成數據

        Args:
            class_id: 目標類別 ID
            count: 生成數量
            variations: 變體配置列表

        Returns:
            成功生成的數量
        """
        if variations is None:
            variations = [
                {'staff_size': 18},
                {'staff_size': 20},
                {'staff_size': 22},
                {'staff_size': 24},
            ]

        class_info = CLASS_DEFINITIONS.get(class_id)
        if not class_info:
            print(f"未知類別: {class_id}")
            return 0

        print(f"\n生成 Class {class_id} ({class_info.name}) 數據...")
        print(f"  目標數量: {count}")
        print(f"  當前樣本: {class_info.current_count}")

        success_count = 0
        for i in range(count):
            variation = random.choice(variations)
            file_id = f"synthetic_{class_id}_{i:05d}"

            try:
                # 1. 生成 LilyPond 文件
                ly_path = generate_lilypond_file(
                    class_id,
                    self.output_dir / file_id,
                    variation
                )

                # 2. 渲染為 PNG
                png_path = render_lilypond(ly_path, self.output_dir)

                if png_path:
                    # 3. 移動到正確位置
                    final_png = self.images_dir / f"{file_id}.png"
                    os.rename(png_path, final_png)

                    # 4. 生成標註
                    annotations = generate_yolo_annotations(str(final_png), class_id)
                    label_path = self.labels_dir / f"{file_id}.txt"
                    with open(label_path, 'w') as f:
                        f.write('\n'.join(annotations))

                    success_count += 1

                    if success_count % 100 == 0:
                        print(f"  進度: {success_count}/{count}")

            except Exception as e:
                print(f"  生成失敗 ({i}): {e}")

        print(f"  完成: {success_count}/{count}")
        return success_count

    def balance_all_classes(self, min_samples: int = 5000):
        """
        平衡所有類別

        Args:
            min_samples: 每個類別的最小樣本數
        """
        print("=" * 60)
        print("開始類別平衡")
        print("=" * 60)

        for class_id, info in CLASS_DEFINITIONS.items():
            if info.current_count < min_samples:
                needed = min_samples - info.current_count
                print(f"\nClass {class_id} ({info.name}): 需要 {needed} 個額外樣本")
                self.generate_for_class(class_id, needed)

def main():
    parser = argparse.ArgumentParser(description='合成數據生成器')
    parser.add_argument('--target-class', type=int, help='目標類別 ID')
    parser.add_argument('--count', type=int, default=1000, help='生成數量')
    parser.add_argument('--balance-all', action='store_true', help='平衡所有類別')
    parser.add_argument('--min-samples', type=int, default=5000, help='最小樣本數')
    parser.add_argument('--output', type=str, default='datasets/synthetic',
                       help='輸出目錄')
    parser.add_argument('--check-lilypond', action='store_true',
                       help='檢查 LilyPond 是否安裝')

    args = parser.parse_args()

    # 檢查 LilyPond
    if args.check_lilypond:
        try:
            result = subprocess.run(['lilypond', '--version'],
                                   capture_output=True, text=True)
            print(f"LilyPond 版本: {result.stdout.split()[2]}")
            print("✅ LilyPond 已安裝")
        except FileNotFoundError:
            print("❌ LilyPond 未安裝")
            print("安裝方式: sudo apt install lilypond")
        return

    # 初始化生成器
    generator = SyntheticDataGenerator(Path(args.output))

    if args.balance_all:
        generator.balance_all_classes(args.min_samples)
    elif args.target_class is not None:
        generator.generate_for_class(args.target_class, args.count)
    else:
        # 顯示類別狀態
        print("=" * 70)
        print("類別狀態總覽")
        print("=" * 70)
        print(f"{'ID':>3} {'名稱':25s} {'當前':>10} {'目標':>10} {'需求':>10}")
        print("-" * 70)

        for class_id, info in sorted(CLASS_DEFINITIONS.items()):
            needed = max(0, info.target_count - info.current_count)
            status = "✅" if needed == 0 else "❌"
            print(f"{class_id:>3} {info.name:25s} {info.current_count:>10,} "
                  f"{info.target_count:>10,} {needed:>10,} {status}")

        print("\n使用方式:")
        print("  python synthetic_data_generator.py --target-class 17 --count 1000")
        print("  python synthetic_data_generator.py --balance-all --min-samples 5000")
        print("  python synthetic_data_generator.py --check-lilypond")

if __name__ == '__main__':
    os.chdir(Path(__file__).parent)
    main()
