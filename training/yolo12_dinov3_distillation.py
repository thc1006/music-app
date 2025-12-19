#!/usr/bin/env python3
"""
YOLO12 + DINOv3 知识蒸馏训练脚本
Phase 1: 使用 DINOv3 作为教师模型蒸馏到 YOLO12s

目标：验证 DINOv3 对 OMR 任务的提升效果
策略：保持 YOLO12 架构不变，通过知识蒸馏提升性能
"""

import os
import sys
import json
import torch
from pathlib import Path
from datetime import datetime

# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path('/home/thc1006/dev/music-app/training')
DATASET_DIR = BASE_DIR / 'datasets/yolo_harmony_v2_phase8_final'
OUTPUT_DIR = BASE_DIR / 'harmony_omr_v2_dinov3_distill'
PHASE8_BEST = BASE_DIR / 'harmony_omr_v2_phase8/phase8_training/weights/best.pt'

# DINOv3 配置 (2025-12-20 更新：使用 timm 實際可用的模型名稱)
DINOV3_CONFIG = {
    'model_name': 'vit_small_patch16_dinov3',  # ViT-S/16 (21.6M params) ✅ 已驗證
    'pretrained': True,
    'freeze': True,  # 冻结教师模型
    'input_size': 640,  # 與 YOLO 訓練尺寸一致
}

# 蒸馏配置
DISTILLATION_CONFIG = {
    'temperature': 4.0,      # 蒸馏温度
    'alpha': 0.5,            # 蒸馏损失权重
    'feature_weight': 0.3,   # 特征匹配权重
}

# 训练配置（基于 Phase 8）
TRAINING_CONFIG = {
    'epochs': 100,
    'batch': 16,  # 蒸馏需要更多显存
    'imgsz': 640,
    'lr0': 0.0005,  # 蒸馏用较低学习率
    'optimizer': 'AdamW',
    'patience': 30,
    'device': 0,
    'workers': 16,
    'amp': True,
}


def check_environment():
    """检查环境和依赖"""
    print("=" * 60)
    print("环境检查")
    print("=" * 60)

    # GPU
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用!")

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name}")
    print(f"显存: {gpu_mem:.1f} GB")

    # 检查 LightlyTrain
    try:
        import lightly_train
        print("LightlyTrain: ✅ 已安装")
        return 'lightly'
    except ImportError:
        print("LightlyTrain: ❌ 未安装")

    # 检查 timm (用于直接加载 DINOv3)
    try:
        import timm
        print(f"timm: ✅ 版本 {timm.__version__}")
        return 'timm'
    except ImportError:
        print("timm: ❌ 未安装")

    return None


def load_dinov3_teacher():
    """加载 DINOv3 教师模型 (2025-12-20 更新：使用 timm 已驗證的方法)"""
    print("\n" + "=" * 60)
    print("加载 DINOv3 教师模型")
    print("=" * 60)

    model_name = DINOV3_CONFIG['model_name']
    print(f"模型: {model_name}")

    try:
        # ✅ 推薦方法: 使用 timm (已驗證可用)
        import timm
        print("通过 timm 加载...")
        teacher = timm.create_model(
            model_name,
            pretrained=DINOV3_CONFIG['pretrained']
        )
        print(f"✅ 成功加载 {model_name}")

        # 測試推理
        teacher.eval()
        with torch.no_grad():
            dummy = torch.randn(1, 3, 640, 640)
            features = teacher.forward_features(dummy)
            print(f"✅ 640x640 輸入測試通過")
            print(f"   特徵形狀: {features.shape}")
            print(f"   特徵維度: {features.shape[-1]}")

    except Exception as e:
        print(f"❌ timm 加载失败: {e}")

        # 備選方法: 使用 transformers
        try:
            from transformers import AutoModel
            print("尝试通过 transformers 加载...")
            teacher = AutoModel.from_pretrained(
                'facebook/dinov2-small'  # DINOv2 作為備選
            )
            print("⚠️ 使用 DINOv2 作為備選")

        except Exception as e2:
            print(f"❌ transformers 加载失败: {e2}")
            return None

    # 冻结教师模型
    if DINOV3_CONFIG['freeze']:
        for param in teacher.parameters():
            param.requires_grad = False
        teacher.eval()
        print("✅ 教师模型已冻结")

    # 统计参数
    total_params = sum(p.numel() for p in teacher.parameters()) / 1e6
    print(f"✅ 教师模型参数量: {total_params:.1f}M")

    return teacher


def method_a_lightly_train():
    """
    方法 A: 使用 LightlyTrain 进行知识蒸馏
    这是最简单的方法，LightlyTrain 封装了所有蒸馏逻辑
    """
    print("\n" + "=" * 60)
    print("方法 A: LightlyTrain 知识蒸馏")
    print("=" * 60)

    import lightly_train

    # 准备数据集配置
    data_config = {
        'path': str(DATASET_DIR),
        'train': 'train/images',
        'val': 'val/images',
        'nc': 33,
        'names': {
            0: 'notehead_filled',
            1: 'notehead_hollow',
            2: 'stem',
            # ... (完整的33类)
        }
    }

    # 执行蒸馏预训练
    print("开始 DINOv3 → YOLO12s 蒸馏...")

    lightly_train.train(
        out=str(OUTPUT_DIR / 'lightly_distill'),
        data=str(DATASET_DIR / 'train/images'),
        model='ultralytics/yolov12s',
        method='distillation',
        method_args={
            'teacher': 'dinov3/vits16',
            'temperature': DISTILLATION_CONFIG['temperature'],
        },
        epochs=TRAINING_CONFIG['epochs'],
        batch_size=TRAINING_CONFIG['batch'],
    )

    print("✅ 蒸馏预训练完成")

    # 使用蒸馏后的模型进行微调
    print("\n开始目标检测微调...")

    lightly_train.train_object_detection(
        out=str(OUTPUT_DIR / 'detection_finetune'),
        model=str(OUTPUT_DIR / 'lightly_distill/model.pt'),
        data=data_config,
        epochs=50,
        batch_size=TRAINING_CONFIG['batch'],
    )

    print("✅ 目标检测微调完成")


def method_b_custom_distillation():
    """
    方法 B: 自定义知识蒸馏实现
    更灵活，可以针对 OMR 任务进行优化
    """
    print("\n" + "=" * 60)
    print("方法 B: 自定义知识蒸馏")
    print("=" * 60)

    from ultralytics import YOLO
    import torch.nn as nn
    import torch.nn.functional as F

    # 加载教师和学生模型
    teacher = load_dinov3_teacher()
    if teacher is None:
        print("❌ 无法加载教师模型，跳过此方法")
        return

    teacher = teacher.cuda()

    # 加载学生模型（从 Phase 8 开始）
    print(f"\n加载学生模型: {PHASE8_BEST}")
    student = YOLO(str(PHASE8_BEST))

    # 创建特征对齐层
    # DINOv3 ViT-S 输出 384 维特征
    # 需要对齐到 YOLO 的特征维度
    class FeatureAligner(nn.Module):
        def __init__(self, teacher_dim=384, student_dim=256):
            super().__init__()
            self.proj = nn.Sequential(
                nn.Linear(teacher_dim, student_dim),
                nn.ReLU(),
                nn.Linear(student_dim, student_dim)
            )

        def forward(self, x):
            return self.proj(x)

    aligner = FeatureAligner().cuda()

    # 蒸馏损失函数
    def distillation_loss(student_logits, teacher_logits, temperature):
        """KL 散度蒸馏损失"""
        soft_student = F.log_softmax(student_logits / temperature, dim=-1)
        soft_teacher = F.softmax(teacher_logits / temperature, dim=-1)
        return F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (temperature ** 2)

    def feature_loss(student_feat, teacher_feat):
        """特征匹配损失"""
        return F.mse_loss(student_feat, teacher_feat)

    print("\n自定义蒸馏训练需要更深度的集成...")
    print("建议使用方法 A (LightlyTrain) 作为起点")
    print("如果需要更精细的控制，可以基于此框架扩展")

    return teacher, student, aligner


def method_c_feature_fusion():
    """
    方法 C: 特征融合方案
    保留 YOLO12 backbone，融合 DINOv3 特征
    注意：这会增加推理时间
    """
    print("\n" + "=" * 60)
    print("方法 C: 特征融合")
    print("=" * 60)

    print("特征融合方案需要修改 YOLO 架构")
    print("推荐在验证蒸馏效果后再考虑此方案")
    print("详见 DEIMv2 的 STA 适配器设计")


def run_baseline_comparison():
    """
    运行基线对比实验
    """
    print("\n" + "=" * 60)
    print("基线对比实验")
    print("=" * 60)

    from ultralytics import YOLO

    # 加载 Phase 8 模型
    print(f"加载 Phase 8 模型: {PHASE8_BEST}")
    model = YOLO(str(PHASE8_BEST))

    # 在验证集上评估
    print("\n评估 Phase 8 基线性能...")
    results = model.val(
        data=str(DATASET_DIR / 'harmony_phase8_final.yaml'),
        imgsz=640,
        batch=16,
        device=0,
    )

    print("\n" + "-" * 40)
    print("Phase 8 基线结果:")
    print("-" * 40)
    print(f"mAP50: {results.box.map50:.4f}")
    print(f"mAP50-95: {results.box.map:.4f}")

    return results


def main():
    """主函数"""
    print("=" * 70)
    print("YOLO12 + DINOv3 知识蒸馏实验")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 检查环境
    method = check_environment()

    # 运行基线对比
    baseline_results = run_baseline_comparison()

    # 选择蒸馏方法
    if method == 'lightly':
        print("\n检测到 LightlyTrain，使用方法 A")
        method_a_lightly_train()
    elif method == 'timm':
        print("\n使用 timm 加载 DINOv3，使用方法 B")
        method_b_custom_distillation()
    else:
        print("\n请先安装必要依赖:")
        print("  pip install lightly-train")
        print("  或")
        print("  pip install timm>=1.0.20")
        return

    # 保存实验报告
    report = {
        'experiment': 'YOLO12 + DINOv3 Distillation',
        'timestamp': datetime.now().isoformat(),
        'baseline': {
            'model': str(PHASE8_BEST),
            'mAP50': float(baseline_results.box.map50) if baseline_results else None,
        },
        'config': {
            'dinov3': DINOV3_CONFIG,
            'distillation': DISTILLATION_CONFIG,
            'training': TRAINING_CONFIG,
        }
    }

    report_path = OUTPUT_DIR / 'experiment_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n实验报告已保存: {report_path}")
    print("\n" + "=" * 70)
    print("实验完成!")
    print("=" * 70)


if __name__ == '__main__':
    main()
