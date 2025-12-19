#!/usr/bin/env python3
"""
Phase 10.1 數據集合併 - CPU 多核心平行化優化版本
合併 Phase 8 + DeepScores Dynamics (855張)
使用 multiprocessing 和 numpy 向量化加速
"""
import os
import shutil
import json
import numpy as np
from pathlib import Path
from collections import Counter
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import time

# 硬體配置
NUM_WORKERS = min(20, cpu_count() - 4)  # i9-14900 24核心，留4核心給系統
CHUNK_SIZE = 100  # 每個 worker 處理的文件數

print(f"🚀 硬體優化配置:")
print(f"   CPU 核心數: {cpu_count()}")
print(f"   使用 Workers: {NUM_WORKERS}")
print(f"   Chunk Size: {CHUNK_SIZE}")

def copy_file_task(args):
    """單個文件複製任務（用於平行化）"""
    src, dst = args
    try:
        shutil.copy2(src, dst)
        return True, None
    except Exception as e:
        return False, str(e)

def batch_copy_files(file_pairs, desc="複製文件"):
    """批次平行化複製文件"""
    print(f"\n📦 {desc} (平行化，{NUM_WORKERS} workers)")

    with Pool(NUM_WORKERS) as pool:
        results = list(tqdm(
            pool.imap(copy_file_task, file_pairs, chunksize=CHUNK_SIZE),
            total=len(file_pairs),
            desc=desc
        ))

    success_count = sum(1 for success, _ in results if success)
    errors = [err for success, err in results if not success]

    print(f"   ✅ 成功: {success_count}/{len(file_pairs)}")
    if errors:
        print(f"   ❌ 錯誤: {len(errors)}")
        for err in errors[:5]:  # 只顯示前 5 個錯誤
            print(f"      {err}")

    return success_count, errors

def analyze_labels_vectorized(label_files):
    """向量化分析標註文件（加速統計）"""
    print(f"\n📊 分析標註文件 (向量化處理)")

    all_class_ids = []
    annotation_counts = []

    for label_file in tqdm(label_files, desc="讀取標註"):
        with open(label_file, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]

        annotation_counts.append(len(lines))

        for line in lines:
            parts = line.split()
            if len(parts) >= 5:
                all_class_ids.append(int(parts[0]))

    # NumPy 向量化統計
    all_class_ids = np.array(all_class_ids)
    annotation_counts = np.array(annotation_counts)

    # 類別分佈
    unique_classes, counts = np.unique(all_class_ids, return_counts=True)
    class_distribution = dict(zip(unique_classes.tolist(), counts.tolist()))

    # 統計指標
    stats = {
        'total_annotations': int(np.sum(annotation_counts)),
        'mean_annotations': float(np.mean(annotation_counts)),
        'median_annotations': float(np.median(annotation_counts)),
        'std_annotations': float(np.std(annotation_counts)),
        'class_distribution': class_distribution
    }

    return stats

def main():
    start_time = time.time()

    # 路徑設定
    base_dir = Path("/home/thc1006/dev/music-app/training/datasets")

    phase8_dir = base_dir / "yolo_harmony_v2_phase8_final"
    deepscores_dir = base_dir / "yolo_deepscores_dynamics"

    output_dir = base_dir / "yolo_harmony_v2_phase10_1"
    output_dir.mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Phase 10.1 數據集合併（CPU 多核心優化）")
    print(f"{'='*60}")
    print(f"\n輸入數據集:")
    print(f"  1. Phase 8 Final: {phase8_dir}")
    print(f"  2. DeepScores Dynamics: {deepscores_dir}")
    print(f"\n輸出目錄: {output_dir}")

    # 創建輸出目錄結構
    (output_dir / "train" / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "train" / "labels").mkdir(parents=True, exist_ok=True)
    (output_dir / "val" / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "val" / "labels").mkdir(parents=True, exist_ok=True)

    # ===== 合併訓練集 =====
    print(f"\n{'='*60}")
    print(f"  步驟 1: 合併訓練集")
    print(f"{'='*60}")

    # Phase 8 訓練集
    phase8_train_images = list((phase8_dir / "train" / "images").glob("*.[jp][pn]g"))
    phase8_train_labels = list((phase8_dir / "train" / "labels").glob("*.txt"))

    print(f"\n📁 Phase 8 訓練集:")
    print(f"   圖片: {len(phase8_train_images)}")
    print(f"   標註: {len(phase8_train_labels)}")

    # DeepScores 數據集（全部作為訓練集）
    deepscores_images = list((deepscores_dir / "images").rglob("*.[jp][pn]g"))
    deepscores_labels = list((deepscores_dir / "labels").rglob("*.txt"))

    print(f"\n📁 DeepScores:")
    print(f"   圖片: {len(deepscores_images)}")
    print(f"   標註: {len(deepscores_labels)}")

    # 準備複製任務（平行化）
    train_image_tasks = []
    train_label_tasks = []

    # Phase 8
    for img in phase8_train_images:
        dst = output_dir / "train" / "images" / img.name
        train_image_tasks.append((img, dst))

    for lbl in phase8_train_labels:
        dst = output_dir / "train" / "labels" / lbl.name
        train_label_tasks.append((lbl, dst))

    # DeepScores（重命名以避免衝突）
    for i, img in enumerate(deepscores_images):
        new_name = f"deepscores_{i:04d}{img.suffix}"
        dst = output_dir / "train" / "images" / new_name
        train_image_tasks.append((img, dst))

    for i, lbl in enumerate(deepscores_labels):
        new_name = f"deepscores_{i:04d}.txt"
        dst = output_dir / "train" / "labels" / new_name
        train_label_tasks.append((lbl, dst))

    # 平行化複製
    batch_copy_files(train_image_tasks, "複製訓練圖片")
    batch_copy_files(train_label_tasks, "複製訓練標註")

    # ===== 合併驗證集 =====
    print(f"\n{'='*60}")
    print(f"  步驟 2: 合併驗證集")
    print(f"{'='*60}")

    phase8_val_images = list((phase8_dir / "val" / "images").glob("*.[jp][pn]g"))
    phase8_val_labels = list((phase8_dir / "val" / "labels").glob("*.txt"))

    print(f"\n📁 Phase 8 驗證集:")
    print(f"   圖片: {len(phase8_val_images)}")
    print(f"   標註: {len(phase8_val_labels)}")

    val_image_tasks = [(img, output_dir / "val" / "images" / img.name) for img in phase8_val_images]
    val_label_tasks = [(lbl, output_dir / "val" / "labels" / lbl.name) for lbl in phase8_val_labels]

    batch_copy_files(val_image_tasks, "複製驗證圖片")
    batch_copy_files(val_label_tasks, "複製驗證標註")

    # ===== 向量化統計分析 =====
    print(f"\n{'='*60}")
    print(f"  步驟 3: 統計分析（向量化）")
    print(f"{'='*60}")

    all_train_labels = list((output_dir / "train" / "labels").glob("*.txt"))
    train_stats = analyze_labels_vectorized(all_train_labels)

    all_val_labels = list((output_dir / "val" / "labels").glob("*.txt"))
    val_stats = analyze_labels_vectorized(all_val_labels)

    # ===== 創建 YAML 配置 =====
    yaml_content = f"""# Phase 10.1 Dataset
# 合併時間: {time.strftime('%Y-%m-%d %H:%M:%S')}
# 訓練集: {len(all_train_labels)} 圖片
# 驗證集: {len(all_val_labels)} 圖片
# 總標註: {train_stats['total_annotations'] + val_stats['total_annotations']:,}

path: {output_dir}
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

    yaml_path = output_dir / "phase10_1.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    # ===== 儲存合併報告 =====
    merge_report = {
        'merge_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'hardware': {
            'cpu_workers': NUM_WORKERS,
            'chunk_size': CHUNK_SIZE
        },
        'datasets': {
            'phase8': {
                'train_images': len(phase8_train_images),
                'train_labels': len(phase8_train_labels),
                'val_images': len(phase8_val_images),
                'val_labels': len(phase8_val_labels)
            },
            'deepscores': {
                'images': len(deepscores_images),
                'labels': len(deepscores_labels)
            }
        },
        'output': {
            'train_images': len(list((output_dir / "train" / "images").glob("*.[jp][pn]g"))),
            'train_labels': len(all_train_labels),
            'val_images': len(list((output_dir / "val" / "images").glob("*.[jp][pn]g"))),
            'val_labels': len(all_val_labels)
        },
        'statistics': {
            'train': train_stats,
            'val': val_stats
        },
        'processing_time_seconds': time.time() - start_time
    }

    report_path = output_dir / "merge_report.json"
    with open(report_path, 'w') as f:
        json.dump(merge_report, f, indent=2)

    # ===== 最終報告 =====
    elapsed = time.time() - start_time

    print(f"\n{'='*60}")
    print(f"  ✅ Phase 10.1 數據集合併完成")
    print(f"{'='*60}")
    print(f"\n📊 最終統計:")
    print(f"   訓練集圖片: {merge_report['output']['train_images']:,}")
    print(f"   訓練集標註: {merge_report['output']['train_labels']:,}")
    print(f"   驗證集圖片: {merge_report['output']['val_images']:,}")
    print(f"   驗證集標註: {merge_report['output']['val_labels']:,}")
    print(f"   總標註數: {train_stats['total_annotations'] + val_stats['total_annotations']:,}")
    print(f"\n⏱️  處理時間: {elapsed:.2f} 秒")
    print(f"   平均速度: {merge_report['output']['train_images'] / elapsed:.0f} 圖片/秒")
    print(f"\n📁 輸出位置:")
    print(f"   數據集: {output_dir}")
    print(f"   配置: {yaml_path}")
    print(f"   報告: {report_path}")

    # DeepScores 類別分佈
    print(f"\n📈 DeepScores 類別分佈:")
    class_names = {30: 'dynamic_soft', 31: 'dynamic_loud'}
    for class_id, count in sorted(train_stats['class_distribution'].items()):
        if class_id in [30, 31]:
            name = class_names[class_id]
            pct = count / train_stats['total_annotations'] * 100
            print(f"   {name:20s}: {count:6d} ({pct:.1f}%)")

if __name__ == "__main__":
    main()
