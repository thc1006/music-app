#!/usr/bin/env python3
"""
Phase 8 vs Phase 9 数据科学深度分析脚本
生成详细的统计报告、可视化图表和数据质量评估
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from collections import defaultdict

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class DatasetAnalyzer:
    def __init__(self, phase8_path, phase9_path):
        self.phase8_path = Path(phase8_path)
        self.phase9_path = Path(phase9_path)
        self.report = {}

    def load_results(self):
        """加载训练结果"""
        print("📊 加载训练结果...")
        self.phase8_results = pd.read_csv(self.phase8_path / 'harmony_omr_v2_phase8/phase8_training/results.csv')
        self.phase9_results = pd.read_csv(self.phase9_path / 'harmony_omr_v2_phase9_fixed/phase9_with_phase8_config2/results.csv')
        print(f"  - Phase 8: {len(self.phase8_results)} epochs")
        print(f"  - Phase 9: {len(self.phase9_results)} epochs")

    def analyze_training_dynamics(self):
        """分析训练动态"""
        print("\n🔍 分析训练动态...")

        # 关键指标对比
        metrics = {
            'Phase 8': {
                'final_mAP50': self.phase8_results['metrics/mAP50(B)'].iloc[-1],
                'best_mAP50': self.phase8_results['metrics/mAP50(B)'].max(),
                'final_cls_loss': self.phase8_results['train/cls_loss'].iloc[-1],
                'initial_cls_loss': self.phase8_results['train/cls_loss'].iloc[0],
                'cls_loss_reduction': (1 - self.phase8_results['train/cls_loss'].iloc[-1] /
                                      self.phase8_results['train/cls_loss'].iloc[0]) * 100,
            },
            'Phase 9': {
                'final_mAP50': self.phase9_results['metrics/mAP50(B)'].iloc[-1],
                'best_mAP50': self.phase9_results['metrics/mAP50(B)'].max(),
                'final_cls_loss': self.phase9_results['train/cls_loss'].iloc[-1],
                'initial_cls_loss': self.phase9_results['train/cls_loss'].iloc[0],
                'cls_loss_reduction': (1 - self.phase9_results['train/cls_loss'].iloc[-1] /
                                      self.phase9_results['train/cls_loss'].iloc[0]) * 100,
            }
        }

        self.report['training_dynamics'] = metrics

        # 打印对比
        print("\n关键指标对比：")
        print(f"{'指标':<25} {'Phase 8':>12} {'Phase 9':>12} {'差异':>12}")
        print("-" * 65)
        print(f"{'Final mAP50':<25} {metrics['Phase 8']['final_mAP50']:>12.4f} "
              f"{metrics['Phase 9']['final_mAP50']:>12.4f} "
              f"{(metrics['Phase 9']['final_mAP50'] - metrics['Phase 8']['final_mAP50'])*100:>11.2f}%")
        print(f"{'Best mAP50':<25} {metrics['Phase 8']['best_mAP50']:>12.4f} "
              f"{metrics['Phase 9']['best_mAP50']:>12.4f} "
              f"{(metrics['Phase 9']['best_mAP50'] - metrics['Phase 8']['best_mAP50'])*100:>11.2f}%")
        print(f"{'cls_loss reduction':<25} {metrics['Phase 8']['cls_loss_reduction']:>11.1f}% "
              f"{metrics['Phase 9']['cls_loss_reduction']:>11.1f}% "
              f"{metrics['Phase 9']['cls_loss_reduction'] - metrics['Phase 8']['cls_loss_reduction']:>11.1f}%")

    def analyze_dataset_distribution(self):
        """分析数据集分布"""
        print("\n📈 分析数据集分布...")

        def get_annotation_stats(label_dir):
            counts = []
            class_dist = defaultdict(int)

            for label_file in Path(label_dir).glob('*.txt'):
                with open(label_file) as f:
                    lines = f.readlines()
                    counts.append(len(lines))
                    for line in lines:
                        try:
                            cls = int(line.split()[0])
                            class_dist[cls] += 1
                        except:
                            pass

            return {
                'counts': counts,
                'mean': np.mean(counts),
                'median': np.median(counts),
                'std': np.std(counts),
                'total_images': len(counts),
                'class_distribution': dict(class_dist)
            }

        # Phase 8
        p8_train = self.phase8_path / 'datasets/yolo_harmony_v2_phase8/labels/train'
        if p8_train.exists():
            p8_stats = get_annotation_stats(p8_train)
            self.report['phase8_distribution'] = p8_stats
            print(f"\nPhase 8:")
            print(f"  - 图片数: {p8_stats['total_images']}")
            print(f"  - 标注密度: mean={p8_stats['mean']:.1f}, median={p8_stats['median']:.1f}, std={p8_stats['std']:.1f}")
            print(f"  - 标注密度变异系数 (CV): {p8_stats['std']/p8_stats['mean']:.2f}")

        # Phase 9
        p9_train = self.phase8_path / 'datasets/yolo_harmony_v2_phase9_merged/train/labels'
        if p9_train.exists():
            p9_stats = get_annotation_stats(p9_train)
            self.report['phase9_distribution'] = p9_stats
            print(f"\nPhase 9:")
            print(f"  - 图片数: {p9_stats['total_images']}")
            print(f"  - 标注密度: mean={p9_stats['mean']:.1f}, median={p9_stats['median']:.1f}, std={p9_stats['std']:.1f}")
            print(f"  - 标注密度变异系数 (CV): {p9_stats['std']/p9_stats['mean']:.2f}")

            # 异质性分析
            heterogeneity = p9_stats['std'] / p8_stats['std']
            print(f"\n⚠️  异质性指标:")
            print(f"  - Phase 9 标准差是 Phase 8 的 {heterogeneity:.1f}x")
            print(f"  - Phase 9 CV 是 Phase 8 的 {(p9_stats['std']/p9_stats['mean'])/(p8_stats['std']/p8_stats['mean']):.1f}x")

    def plot_training_curves(self, output_dir='plots'):
        """绘制训练曲线对比"""
        print("\n📊 生成训练曲线图...")
        Path(output_dir).mkdir(exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # mAP50 曲线
        axes[0, 0].plot(self.phase8_results['epoch'], self.phase8_results['metrics/mAP50(B)'],
                       label='Phase 8', linewidth=2, marker='o', markersize=3, alpha=0.7)
        axes[0, 0].plot(self.phase9_results['epoch'], self.phase9_results['metrics/mAP50(B)'],
                       label='Phase 9', linewidth=2, marker='s', markersize=3, alpha=0.7)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('mAP50')
        axes[0, 0].set_title('mAP50 训练曲线对比')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Classification Loss
        axes[0, 1].plot(self.phase8_results['epoch'], self.phase8_results['train/cls_loss'],
                       label='Phase 8', linewidth=2, alpha=0.7)
        axes[0, 1].plot(self.phase9_results['epoch'], self.phase9_results['train/cls_loss'],
                       label='Phase 9', linewidth=2, alpha=0.7)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Classification Loss')
        axes[0, 1].set_title('分类损失对比 (关键指标)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Box Loss
        axes[1, 0].plot(self.phase8_results['epoch'], self.phase8_results['train/box_loss'],
                       label='Phase 8', linewidth=2, alpha=0.7)
        axes[1, 0].plot(self.phase9_results['epoch'], self.phase9_results['train/box_loss'],
                       label='Phase 9', linewidth=2, alpha=0.7)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Box Loss')
        axes[1, 0].set_title('边界框损失对比')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Precision vs Recall
        axes[1, 1].plot(self.phase8_results['metrics/recall(B)'],
                       self.phase8_results['metrics/precision(B)'],
                       label='Phase 8', linewidth=2, alpha=0.7)
        axes[1, 1].plot(self.phase9_results['metrics/recall(B)'],
                       self.phase9_results['metrics/precision(B)'],
                       label='Phase 9', linewidth=2, alpha=0.7)
        axes[1, 1].set_xlabel('Recall')
        axes[1, 1].set_ylabel('Precision')
        axes[1, 1].set_title('Precision-Recall 轨迹')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = Path(output_dir) / 'training_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"  ✅ 已保存: {plot_path}")

    def plot_distribution_analysis(self, output_dir='plots'):
        """绘制数据分布分析"""
        print("\n📊 生成数据分布图...")

        if 'phase8_distribution' not in self.report or 'phase9_distribution' not in self.report:
            print("  ⚠️  跳过（数据未加载）")
            return

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Phase 8 分布
        p8_counts = self.report['phase8_distribution']['counts']
        axes[0].hist(p8_counts, bins=50, alpha=0.7, edgecolor='black')
        axes[0].axvline(np.mean(p8_counts), color='r', linestyle='--',
                       label=f'Mean: {np.mean(p8_counts):.1f}')
        axes[0].axvline(np.median(p8_counts), color='g', linestyle='--',
                       label=f'Median: {np.median(p8_counts):.1f}')
        axes[0].set_xlabel('Annotations per Image')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Phase 8 标注密度分布 (均匀)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')

        # Phase 9 分布
        p9_counts = self.report['phase9_distribution']['counts']
        axes[1].hist(p9_counts, bins=100, alpha=0.7, edgecolor='black', range=(0, 500))
        axes[1].axvline(np.mean(p9_counts), color='r', linestyle='--',
                       label=f'Mean: {np.mean(p9_counts):.1f}')
        axes[1].axvline(np.median(p9_counts), color='g', linestyle='--',
                       label=f'Median: {np.median(p9_counts):.1f}')
        axes[1].set_xlabel('Annotations per Image')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Phase 9 标注密度分布 (异质)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plot_path = Path(output_dir) / 'distribution_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"  ✅ 已保存: {plot_path}")

    def generate_diagnostic_report(self):
        """生成诊断报告"""
        print("\n🔬 生成诊断报告...")

        # 计算关键诊断指标
        diagnostics = {
            'negative_transfer_detected': False,
            'heterogeneity_severity': 'unknown',
            'learning_stagnation': False,
            'recommendations': []
        }

        # 检测负迁移
        p8_final = self.phase8_results['metrics/mAP50(B)'].iloc[-1]
        p9_final = self.phase9_results['metrics/mAP50(B)'].iloc[-1]
        if p9_final < p8_final * 0.95:  # 下降超过5%
            diagnostics['negative_transfer_detected'] = True
            diagnostics['transfer_loss_percent'] = (1 - p9_final / p8_final) * 100

        # 检测异质性严重度
        if 'phase8_distribution' in self.report and 'phase9_distribution' in self.report:
            cv_ratio = (self.report['phase9_distribution']['std'] / self.report['phase9_distribution']['mean']) / \
                      (self.report['phase8_distribution']['std'] / self.report['phase8_distribution']['mean'])
            if cv_ratio > 3:
                diagnostics['heterogeneity_severity'] = 'severe'
            elif cv_ratio > 2:
                diagnostics['heterogeneity_severity'] = 'moderate'
            else:
                diagnostics['heterogeneity_severity'] = 'mild'

        # 检测学习停滞
        p9_last_20_improvement = (self.phase9_results['metrics/mAP50(B)'].iloc[-1] -
                                 self.phase9_results['metrics/mAP50(B)'].iloc[-20]) * 100
        if p9_last_20_improvement < 0.2:  # 最后20轮提升小于0.2%
            diagnostics['learning_stagnation'] = True

        # 生成建议
        if diagnostics['negative_transfer_detected']:
            diagnostics['recommendations'].append({
                'priority': 'HIGH',
                'action': '立即移除 OpenScore Lieder 数据源',
                'reason': f'检测到负迁移，性能下降 {diagnostics["transfer_loss_percent"]:.1f}%'
            })

        if diagnostics['heterogeneity_severity'] == 'severe':
            diagnostics['recommendations'].append({
                'priority': 'HIGH',
                'action': '实施分阶段训练或加权采样',
                'reason': '数据异质性严重，需要域适应策略'
            })

        if diagnostics['learning_stagnation']:
            diagnostics['recommendations'].append({
                'priority': 'MEDIUM',
                'action': '增加训练 epochs 至 150-200',
                'reason': '模型仍有学习潜力但提前停滞'
            })

        self.report['diagnostics'] = diagnostics

        # 打印诊断结果
        print("\n" + "="*70)
        print("🔬 诊断结果")
        print("="*70)
        print(f"负迁移检测: {'⚠️ 是' if diagnostics['negative_transfer_detected'] else '✅ 否'}")
        if diagnostics['negative_transfer_detected']:
            print(f"  → 性能损失: {diagnostics['transfer_loss_percent']:.2f}%")
        print(f"数据异质性: {diagnostics['heterogeneity_severity'].upper()}")
        print(f"学习停滞: {'⚠️ 是' if diagnostics['learning_stagnation'] else '✅ 否'}")

        print("\n📋 推荐行动:")
        for i, rec in enumerate(diagnostics['recommendations'], 1):
            print(f"{i}. [{rec['priority']}] {rec['action']}")
            print(f"   原因: {rec['reason']}")

    def save_report(self, output_path='analysis_report.json'):
        """保存完整报告"""
        print(f"\n💾 保存报告到 {output_path}...")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.report, f, indent=2, ensure_ascii=False)
        print("  ✅ 完成")

def main():
    print("="*70)
    print("Phase 8 vs Phase 9 深度数据科学分析")
    print("="*70)

    # 初始化分析器
    analyzer = DatasetAnalyzer(
        phase8_path='/home/thc1006/dev/music-app/training',
        phase9_path='/home/thc1006/dev/music-app/training'
    )

    # 执行分析流程
    try:
        analyzer.load_results()
        analyzer.analyze_training_dynamics()
        analyzer.analyze_dataset_distribution()
        analyzer.plot_training_curves()
        analyzer.plot_distribution_analysis()
        analyzer.generate_diagnostic_report()
        analyzer.save_report()

        print("\n" + "="*70)
        print("✅ 分析完成！")
        print("="*70)
        print("📁 生成的文件:")
        print("  - plots/training_comparison.png")
        print("  - plots/distribution_comparison.png")
        print("  - analysis_report.json")
        print("\n📖 完整分析报告: PHASE9_DATA_SCIENCE_ANALYSIS.md")

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
