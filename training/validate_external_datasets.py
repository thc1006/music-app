#!/usr/bin/env python3
"""
驗證外部數據集質量 - Phase 10.1 交叉驗證
分析 DeepScores 和 OpenScore 的標註質量、類別分佈、異質性
"""
import os
import json
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict

def analyze_dataset(label_dir, dataset_name):
    """分析數據集的標註質量"""
    print(f"\n{'='*60}")
    print(f"  {dataset_name} 數據質量分析")
    print(f"{'='*60}\n")

    label_files = list(Path(label_dir).rglob("*.txt"))
    total_files = len(label_files)

    annotation_counts = []
    class_distribution = Counter()
    empty_files = []
    bbox_sizes = []  # 儲存 bbox 面積

    for label_file in label_files:
        with open(label_file, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]

        if len(lines) == 0:
            empty_files.append(label_file.name)
            annotation_counts.append(0)
            continue

        annotation_counts.append(len(lines))

        for line in lines:
            parts = line.split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                class_distribution[class_id] += 1

                # 計算 bbox 面積 (normalized)
                w, h = float(parts[3]), float(parts[4])
                bbox_sizes.append(w * h)

    # 統計指標
    annotation_counts = np.array(annotation_counts)
    mean_annotations = np.mean(annotation_counts)
    median_annotations = np.median(annotation_counts)
    std_annotations = np.std(annotation_counts)
    cv = std_annotations / mean_annotations if mean_annotations > 0 else 0

    # 異質性分析
    print(f"📊 標註數量統計:")
    print(f"   總文件數: {total_files}")
    print(f"   空標註文件: {len(empty_files)} ({len(empty_files)/total_files*100:.1f}%)")
    print(f"   平均標註數: {mean_annotations:.1f}")
    print(f"   中位數標註數: {median_annotations:.1f}")
    print(f"   標準差: {std_annotations:.1f}")
    print(f"   變異係數 (CV): {cv:.2f}")

    # CV 閾值警告
    if cv < 0.5:
        print(f"   ⚠️  警告: 高度同質 (CV < 0.5)！")
    elif cv > 2.0:
        print(f"   ⚠️  警告: 異質性過高 (CV > 2.0)，可能影響訓練！")
    else:
        print(f"   ✅ 異質性正常")

    # 類別分佈
    total_annotations = sum(class_distribution.values())
    print(f"\n📈 類別分佈 (前 10 名):")

    class_names = {
        0: "notehead_filled", 1: "notehead_hollow", 2: "stem", 3: "beam",
        13: "accidental_sharp", 14: "accidental_flat", 15: "accidental_natural",
        23: "barline", 24: "barline_double", 25: "barline_final",
        29: "fermata", 30: "dynamic_soft", 31: "dynamic_loud"
    }

    top_classes = class_distribution.most_common(10)
    for class_id, count in top_classes:
        class_name = class_names.get(class_id, f"class_{class_id}")
        percentage = count / total_annotations * 100
        print(f"   {class_name:25s}: {count:8d} ({percentage:5.1f}%)")

    # 集中度分析
    top_4_count = sum(count for _, count in class_distribution.most_common(4))
    top_4_percentage = top_4_count / total_annotations * 100

    print(f"\n📌 前 4 類別集中度: {top_4_percentage:.1f}%")
    if top_4_percentage > 95:
        print(f"   ❌ 極度集中！可能只增強特定類別，建議謹慎使用")
    elif top_4_percentage > 85:
        print(f"   ⚠️  集中度偏高")
    else:
        print(f"   ✅ 類別分佈合理")

    # Bbox 大小分析
    if bbox_sizes:
        bbox_sizes = np.array(bbox_sizes)
        mean_bbox = np.mean(bbox_sizes)
        median_bbox = np.median(bbox_sizes)

        # 小物件比例 (面積 < 0.01)
        small_bbox_ratio = np.sum(bbox_sizes < 0.01) / len(bbox_sizes) * 100

        print(f"\n📐 Bbox 尺寸分析:")
        print(f"   平均面積: {mean_bbox:.4f}")
        print(f"   中位數面積: {median_bbox:.4f}")
        print(f"   小物件比例 (<0.01): {small_bbox_ratio:.1f}%")

    # 返回統計數據
    return {
        'total_files': total_files,
        'empty_files': len(empty_files),
        'empty_ratio': len(empty_files) / total_files,
        'mean_annotations': mean_annotations,
        'median_annotations': median_annotations,
        'cv': cv,
        'top_4_percentage': top_4_percentage,
        'total_annotations': total_annotations,
        'class_distribution': dict(class_distribution.most_common(10)),
        'small_bbox_ratio': float(small_bbox_ratio) if len(bbox_sizes) > 0 else 0
    }

def select_high_quality_openscore(label_dir, target_count=1500):
    """從 OpenScore 中選擇高質量的 1,500 張圖片"""
    print(f"\n{'='*60}")
    print(f"  OpenScore 高質量子集選擇 (目標: {target_count} 張)")
    print(f"{'='*60}\n")

    label_files = list(Path(label_dir).rglob("*.txt"))

    file_scores = []

    # 優先類別 (fermata, barline_double, barline)
    priority_classes = {29, 24, 23}  # fermata, barline_double, barline

    for label_file in label_files:
        with open(label_file, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]

        if len(lines) == 0:
            continue  # 跳過空文件

        # 計算分數
        priority_count = 0
        total_count = len(lines)

        for line in lines:
            class_id = int(line.split()[0])
            if class_id in priority_classes:
                priority_count += 1

        # 分數 = 優先類別數量 * 10 + 總標註數量
        score = priority_count * 10 + total_count

        file_scores.append((label_file.stem, score, priority_count, total_count))

    # 排序並選擇前 N 個
    file_scores.sort(key=lambda x: x[1], reverse=True)
    selected_files = file_scores[:target_count]

    print(f"✅ 選擇了 {len(selected_files)} 個高質量文件")
    print(f"\n前 10 名文件:")
    for i, (filename, score, priority, total) in enumerate(selected_files[:10], 1):
        print(f"   {i:2d}. {filename[:50]:50s} (優先類別:{priority:3d}, 總標註:{total:4d}, 分數:{score:5d})")

    # 統計選中文件的類別分佈
    selected_priority_count = sum(p for _, _, p, _ in selected_files)
    selected_total_count = sum(t for _, _, _, t in selected_files)

    print(f"\n📊 選中文件統計:")
    print(f"   優先類別標註總數: {selected_priority_count}")
    print(f"   所有標註總數: {selected_total_count}")
    print(f"   優先類別佔比: {selected_priority_count/selected_total_count*100:.1f}%")

    # 儲存選中的文件名列表
    selected_names = [name for name, _, _, _ in selected_files]

    return selected_names

def main():
    base_dir = Path("/home/thc1006/dev/music-app/training/datasets")

    # 1. 分析 DeepScores
    deepscores_stats = analyze_dataset(
        base_dir / "yolo_deepscores_dynamics/labels",
        "DeepScores Dynamics"
    )

    # 2. 分析 OpenScore Lieder
    openscore_stats = analyze_dataset(
        base_dir / "yolo_openscore_lieder/labels",
        "OpenScore Lieder"
    )

    # 3. 選擇高質量 OpenScore 子集
    selected_openscore = select_high_quality_openscore(
        base_dir / "yolo_openscore_lieder/labels",
        target_count=1500
    )

    # 4. 綜合評估
    print(f"\n{'='*60}")
    print(f"  綜合評估與建議")
    print(f"{'='*60}\n")

    # DeepScores 評估
    print("📋 DeepScores Dynamics:")
    if deepscores_stats['empty_ratio'] > 0.05:
        print(f"   ❌ 空標註文件過多 ({deepscores_stats['empty_ratio']*100:.1f}%)")
    else:
        print(f"   ✅ 空標註文件少 ({deepscores_stats['empty_ratio']*100:.1f}%)")

    if deepscores_stats['cv'] > 2.0:
        print(f"   ⚠️  異質性過高 (CV={deepscores_stats['cv']:.2f})")
    elif deepscores_stats['cv'] < 0.5:
        print(f"   ⚠️  過度同質 (CV={deepscores_stats['cv']:.2f})")
    else:
        print(f"   ✅ 異質性適中 (CV={deepscores_stats['cv']:.2f})")

    # OpenScore 評估
    print("\n📋 OpenScore Lieder (1,500 高質量子集):")
    if openscore_stats['top_4_percentage'] > 95:
        print(f"   ⚠️  類別極度集中 ({openscore_stats['top_4_percentage']:.1f}%)")
    else:
        print(f"   ✅ 類別分佈合理")

    # 最終建議
    print(f"\n🎯 Phase 10.1 訓練建議:")
    print(f"   數據組成:")
    print(f"     - Phase 8 基礎: 32,555 張")
    print(f"     - DeepScores: {deepscores_stats['total_files']} 張")
    print(f"     - OpenScore (精選): 1,500 張")
    print(f"     - 總計: {32555 + deepscores_stats['total_files'] + 1500:,} 張")
    print(f"   增量: {(deepscores_stats['total_files'] + 1500) / 32555 * 100:.1f}%")

    # 風險評估
    risks = []
    if deepscores_stats['empty_ratio'] > 0.05:
        risks.append("DeepScores 空標註文件過多")
    if deepscores_stats['cv'] > 2.0 or openscore_stats['cv'] > 2.0:
        risks.append("異質性過高")
    if openscore_stats['top_4_percentage'] > 95:
        risks.append("OpenScore 類別過度集中")

    if risks:
        print(f"\n⚠️  潛在風險:")
        for risk in risks:
            print(f"     - {risk}")
    else:
        print(f"\n✅ 未發現嚴重風險，可以開始訓練")

    # 儲存結果
    result = {
        'deepscores': deepscores_stats,
        'openscore': openscore_stats,
        'selected_openscore_files': selected_openscore,
        'total_images': 32555 + deepscores_stats['total_files'] + 1500,
        'total_increment_pct': (deepscores_stats['total_files'] + 1500) / 32555 * 100
    }

    output_file = "/home/thc1006/dev/music-app/training/phase10_validation_report.json"
    with open(output_file, 'w') as f:
        # 需要轉換 numpy types
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        json.dump(result, f, indent=2, default=convert_numpy)

    print(f"\n📄 驗證報告已儲存: {output_file}")

if __name__ == "__main__":
    main()
