# Phase 9 恢复与改进行动计划

## 📋 执行摘要

基于深度数据科学分析，Phase 9 失败主要由**数据域不匹配**（60%）、**标注质量问题**（30%）和**训练配置不足**（10%）导致。本文档提供三条清晰的恢复路径，预计 1-4 周可恢复并超越 Phase 8 性能。

---

## 🎯 三条恢复路径对比

| 路径 | 时间 | 预期 mAP50 | 风险 | 推荐场景 |
|------|------|-----------|------|---------|
| **路径 A: 快速验证** | 1 周 | 0.650-0.655 | 低 | 需要快速确认问题根源 |
| **路径 B: 稳健提升** | 2-3 周 | 0.680-0.700 | 中 | ⭐ **推荐**，平衡速度与质量 |
| **路径 C: 突破瓶颈** | 4-6 周 | 0.720-0.750 | 高 | 研究导向，追求极致性能 |

---

## 🚀 路径 A: 快速验证（1 周）

### 目标
验证"OpenScore Lieder 是主要问题来源"的假设，快速恢复到 Phase 8 水平。

### 执行步骤

#### Day 1: 创建清洁数据集

```bash
cd /home/thc1006/dev/music-app/training

# 1. 创建 Phase 9.1 数据集（Phase 8 + DeepScores only）
python3 << 'EOF'
import shutil
from pathlib import Path
import yaml

# 读取 Phase 8 配置
with open('datasets/yolo_harmony_v2_phase8/harmony_phase8.yaml') as f:
    phase8_config = yaml.safe_load(f)

# 创建新数据集目录
phase91_dir = Path('datasets/yolo_harmony_v2_phase9.1')
phase91_dir.mkdir(exist_ok=True)

# 复制 Phase 8 数据
print("复制 Phase 8 数据...")
shutil.copytree('datasets/yolo_harmony_v2_phase8/images', phase91_dir / 'images', dirs_exist_ok=True)
shutil.copytree('datasets/yolo_harmony_v2_phase8/labels', phase91_dir / 'labels', dirs_exist_ok=True)

# 合并 DeepScores 数据
print("合并 DeepScores 数据...")
deepscores_dir = Path('datasets/yolo_deepscores_dynamics')
if deepscores_dir.exists():
    for split in ['train', 'val']:
        # 复制图片
        for img in (deepscores_dir / 'images' / split).glob('*'):
            shutil.copy(img, phase91_dir / 'images' / split / img.name)
        # 复制标注
        for lbl in (deepscores_dir / 'labels' / split).glob('*'):
            shutil.copy(lbl, phase91_dir / 'labels' / split / lbl.name)

# 生成配置文件
config = {
    'path': str(phase91_dir.absolute()),
    'train': 'images/train',
    'val': 'images/val',
    'nc': 33,
    'names': phase8_config['names']
}

with open(phase91_dir / 'harmony_phase9.1.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)

print(f"✅ Phase 9.1 数据集创建完成: {phase91_dir}")
print(f"   预计图片数: ~33,255 (Phase 8: 32,555 + DeepScores: 700)")
EOF
```

#### Day 2-6: 训练 Phase 9.1 模型

```bash
# 创建训练脚本
cat > yolo12_train_phase9.1.py << 'EOF'
from ultralytics import YOLO
import torch

def main():
    # 确认 GPU 可用
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 加载 Phase 8 best 模型作为起点
    model = YOLO('harmony_omr_v2_phase8/phase8_training/weights/best.pt')

    # 使用 Phase 8 最优配置训练
    results = model.train(
        data='datasets/yolo_harmony_v2_phase9.1/harmony_phase9.1.yaml',
        epochs=150,           # Phase 8 最优
        batch=16,
        imgsz=640,
        lr0=0.001,            # Phase 8 最优
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5,              # Phase 8 最优
        dfl=1.5,
        workers=8,
        device=0,
        project='harmony_omr_v2_phase9.1',
        name='phase9.1_training',
        exist_ok=False,
        pretrained=True,
        optimizer='AdamW',
        verbose=True,
        seed=42,
        deterministic=False,
        amp=True,
        # 数据增强（与 Phase 8 一致）
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.0,
        mosaic=0.5,
        mixup=0.0,
        copy_paste=0.0,
        erasing=0.0,          # 禁用
    )

    print("\n" + "="*70)
    print("✅ Phase 9.1 训练完成！")
    print("="*70)
    print(f"最佳 mAP50: {results.results_dict['metrics/mAP50(B)']:.4f}")
    print(f"模型位置: harmony_omr_v2_phase9.1/phase9.1_training/weights/best.pt")
    print("\n对比 Phase 8:")
    print(f"  Phase 8 mAP50: 0.6444")
    print(f"  Phase 9.1 mAP50: {results.results_dict['metrics/mAP50(B)']:.4f}")
    print(f"  差异: {(results.results_dict['metrics/mAP50(B)'] - 0.6444)*100:+.2f}%")

if __name__ == '__main__':
    main()
EOF

# 启动训练（预计 6-8 小时，RTX 5090）
python3 yolo12_train_phase9.1.py 2>&1 | tee phase9.1_training.log
```

#### Day 7: 结果验证与报告

```bash
# 1. 比较结果
python3 << 'EOF'
import pandas as pd

p8 = pd.read_csv('harmony_omr_v2_phase8/phase8_training/results.csv')
p91 = pd.read_csv('harmony_omr_v2_phase9.1/phase9.1_training/results.csv')

print("Phase 8 vs Phase 9.1 对比:")
print(f"Phase 8 final mAP50: {p8['metrics/mAP50(B)'].iloc[-1]:.4f}")
print(f"Phase 9.1 final mAP50: {p91['metrics/mAP50(B)'].iloc[-1]:.4f}")
print(f"差异: {(p91['metrics/mAP50(B)'].iloc[-1] - p8['metrics/mAP50(B)'].iloc[-1])*100:+.2f}%")
EOF

# 2. 生成对比报告
python3 analyze_phase8_vs_phase9.py
```

### 预期结果

| 指标 | Phase 8 | Phase 9.1 预期 | 说明 |
|------|---------|---------------|------|
| mAP50 | 0.6444 | **0.650-0.655** | +0.6% to +1.7% |
| dynamic_loud | 0.760 | **0.800-0.850** | DeepScores 增强 |
| dynamic_soft | 未知 | **0.750-0.800** | DeepScores 增强 |

### 决策点

**如果 Phase 9.1 mAP50 >= 0.650**：
- ✅ 确认 OpenScore Lieder 是主要问题
- → 进入路径 B，改进 OpenScore 数据质量

**如果 Phase 9.1 mAP50 < 0.650**：
- ⚠️ DeepScores 也存在问题
- → 需要更深入的数据审查（参考附录）

---

## 🏆 路径 B: 稳健提升（2-3 周）⭐ 推荐

### 目标
通过伪标注和数据增强改进 OpenScore Lieder，超越 Phase 8 达到 0.680-0.700 mAP50。

### Week 1: 伪标注 OpenScore Lieder

#### Step 1.1: 生成伪标注

```bash
cd /home/thc1006/dev/music-app/training

# 使用 Phase 8 最佳模型对 OpenScore 进行预测
python3 << 'EOF'
from ultralytics import YOLO
from pathlib import Path
import shutil

model = YOLO('harmony_omr_v2_phase8/phase8_training/weights/best.pt')

# 预测 OpenScore 图片
openscore_dir = Path('datasets/yolo_openscore_lieder')
output_dir = Path('datasets/yolo_openscore_lieder_pseudo')
output_dir.mkdir(exist_ok=True)

print("生成伪标注...")
results = model.predict(
    source=str(openscore_dir / 'images/train'),
    save_txt=True,
    save_conf=True,
    conf=0.25,  # 置信度阈值
    iou=0.7,
    project=str(output_dir),
    name='pseudo_labels',
    exist_ok=True
)

print(f"✅ 伪标注生成完成: {output_dir / 'pseudo_labels/labels'}")
EOF
```

#### Step 1.2: 合并伪标注与原始标注

```bash
python3 << 'EOF'
from pathlib import Path
import shutil

original_labels = Path('datasets/yolo_openscore_lieder/labels/train')
pseudo_labels = Path('datasets/yolo_openscore_lieder_pseudo/pseudo_labels/labels')
merged_dir = Path('datasets/yolo_openscore_lieder_merged/labels/train')
merged_dir.mkdir(parents=True, exist_ok=True)

print("合并标注...")
for pseudo_file in pseudo_labels.glob('*.txt'):
    original_file = original_labels / pseudo_file.name
    merged_file = merged_dir / pseudo_file.name

    # 读取原始标注（barlines, fermatas）
    original_annotations = []
    if original_file.exists():
        with open(original_file) as f:
            original_annotations = [line.strip() for line in f if line.strip()]

    # 读取伪标注（所有类别）
    pseudo_annotations = []
    with open(pseudo_file) as f:
        pseudo_annotations = [line.strip() for line in f if line.strip()]

    # 合并：原始标注优先（置信度视为 1.0），伪标注补充
    original_classes = {int(line.split()[0]) for line in original_annotations}
    pseudo_filtered = [line for line in pseudo_annotations
                      if int(line.split()[0]) not in original_classes]

    merged_annotations = original_annotations + pseudo_filtered

    # 写入合并文件
    with open(merged_file, 'w') as f:
        f.write('\n'.join(merged_annotations))

# 复制图片
print("复制图片...")
shutil.copytree('datasets/yolo_openscore_lieder/images/train',
                'datasets/yolo_openscore_lieder_merged/images/train',
                dirs_exist_ok=True)

print(f"✅ 合并完成: {merged_dir}")
EOF
```

#### Step 1.3: 人工抽检（关键！）

```bash
# 抽样 100 张图片进行人工检查
python3 << 'EOF'
import random
from pathlib import Path
import shutil

merged_labels = list(Path('datasets/yolo_openscore_lieder_merged/labels/train').glob('*.txt'))
sample = random.sample(merged_labels, min(100, len(merged_labels)))

review_dir = Path('openscore_review_sample')
review_dir.mkdir(exist_ok=True)

for label_file in sample:
    img_file = Path(str(label_file).replace('labels', 'images').replace('.txt', '.png'))
    shutil.copy(label_file, review_dir / label_file.name)
    shutil.copy(img_file, review_dir / img_file.name)

print(f"✅ 抽样完成: {review_dir}")
print("请使用 CVAT 或 Label Studio 审查标注质量")
EOF
```

### Week 2: 真实化数据增强

```bash
# 创建增强脚本
cat > augment_openscore.py << 'EOF'
import albumentations as A
from PIL import Image
import numpy as np
from pathlib import Path

# 真实化增强管道
transform = A.Compose([
    # 添加噪声（模拟纸张纹理）
    A.GaussNoise(var_limit=(10, 30), p=0.8),

    # 亮度对比度变化（模拟扫描差异）
    A.RandomBrightnessContrast(
        brightness_limit=0.2,
        contrast_limit=0.2,
        p=0.7
    ),

    # 轻微旋转和倾斜（模拟拍照角度）
    A.Affine(
        rotate=(-2, 2),
        shear=(-3, 3),
        p=0.5
    ),

    # 形态学操作（模拟墨水扩散）
    A.Morphological(
        operation='dilate',
        kernel=np.ones((2, 2), np.uint8),
        p=0.3
    ),

    # JPEG 压缩伪影
    A.ImageCompression(
        quality_lower=70,
        quality_upper=95,
        compression_type=A.ImageCompression.ImageCompressionType.JPEG,
        p=0.4
    ),

    # 模糊（模拟对焦不准）
    A.Blur(blur_limit=3, p=0.2),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

def augment_dataset(input_dir, output_dir, num_variants=1):
    """对每张图片生成 num_variants 个增强变体"""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for img_file in (input_dir / 'images/train').glob('*.png'):
        label_file = input_dir / 'labels/train' / f"{img_file.stem}.txt"

        if not label_file.exists():
            continue

        # 读取图片和标注
        image = np.array(Image.open(img_file))

        with open(label_file) as f:
            lines = [line.strip().split() for line in f if line.strip()]

        if not lines:
            continue

        class_labels = [int(line[0]) for line in lines]
        bboxes = [[float(x) for x in line[1:5]] for line in lines]

        # 生成变体
        for i in range(num_variants):
            try:
                transformed = transform(
                    image=image,
                    bboxes=bboxes,
                    class_labels=class_labels
                )

                # 保存增强后的图片
                aug_img_file = output_dir / 'images/train' / f"{img_file.stem}_aug{i}.png"
                aug_img_file.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(transformed['image']).save(aug_img_file)

                # 保存标注
                aug_label_file = output_dir / 'labels/train' / f"{img_file.stem}_aug{i}.txt"
                aug_label_file.parent.mkdir(parents=True, exist_ok=True)

                with open(aug_label_file, 'w') as f:
                    for cls, bbox in zip(transformed['class_labels'], transformed['bboxes']):
                        f.write(f"{cls} {' '.join(map(str, bbox))}\n")

            except Exception as e:
                print(f"⚠️  增强失败 {img_file.name}: {e}")
                continue

if __name__ == '__main__':
    augment_dataset(
        'datasets/yolo_openscore_lieder_merged',
        'datasets/yolo_openscore_lieder_augmented',
        num_variants=2  # 每张图片生成2个变体
    )
    print("✅ 数据增强完成")
EOF

# 安装依赖
pip install albumentations

# 执行增强
python3 augment_openscore.py
```

### Week 3: 分阶段训练

```bash
cat > yolo12_train_phase9.2_staged.py << 'EOF'
from ultralytics import YOLO
import torch

def stage1_phase8_only():
    """Stage 1: 仅使用 Phase 8 数据训练 100 epochs"""
    print("\n" + "="*70)
    print("Stage 1: Phase 8 only training (100 epochs)")
    print("="*70)

    model = YOLO('yolo12s.pt')
    results = model.train(
        data='datasets/yolo_harmony_v2_phase8/harmony_phase8.yaml',
        epochs=100,
        batch=16,
        imgsz=640,
        lr0=0.001,
        cls=0.5,
        project='harmony_omr_v2_phase9.2',
        name='stage1_phase8_only'
    )
    return 'harmony_omr_v2_phase9.2/stage1_phase8_only/weights/best.pt'

def stage2_add_deepscores(stage1_model):
    """Stage 2: 加入 DeepScores fine-tune 30 epochs"""
    print("\n" + "="*70)
    print("Stage 2: Add DeepScores (30 epochs fine-tuning)")
    print("="*70)

    model = YOLO(stage1_model)
    results = model.train(
        data='datasets/yolo_harmony_v2_phase9.1/harmony_phase9.1.yaml',
        epochs=30,
        batch=16,
        imgsz=640,
        lr0=0.0003,  # 降低学习率
        cls=0.5,
        project='harmony_omr_v2_phase9.2',
        name='stage2_add_deepscores'
    )
    return 'harmony_omr_v2_phase9.2/stage2_add_deepscores/weights/best.pt'

def stage3_add_openscore(stage2_model):
    """Stage 3: 加入增强后的 OpenScore fine-tune 20 epochs"""
    print("\n" + "="*70)
    print("Stage 3: Add OpenScore (20 epochs fine-tuning)")
    print("="*70)

    # 合并所有数据集
    # TODO: 需要先创建 phase9.2_full 数据集

    model = YOLO(stage2_model)
    results = model.train(
        data='datasets/yolo_harmony_v2_phase9.2_full/harmony_phase9.2.yaml',
        epochs=20,
        batch=16,
        imgsz=640,
        lr0=0.0001,  # 进一步降低学习率
        cls=0.5,
        project='harmony_omr_v2_phase9.2',
        name='stage3_add_openscore'
    )
    return 'harmony_omr_v2_phase9.2/stage3_add_openscore/weights/best.pt'

if __name__ == '__main__':
    stage1_model = stage1_phase8_only()
    stage2_model = stage2_add_deepscores(stage1_model)
    stage3_model = stage3_add_openscore(stage2_model)

    print("\n" + "="*70)
    print("✅ 分阶段训练完成！")
    print("="*70)
    print(f"最终模型: {stage3_model}")
EOF
```

### 预期结果

| 指标 | Phase 8 | Phase 9.2 预期 | 提升 |
|------|---------|---------------|------|
| mAP50 | 0.6444 | **0.680-0.700** | +5.5% to +8.6% |
| fermata | 未知 | **0.700+** | OpenScore 增强 |
| barline_double | 未知 | **0.750+** | OpenScore 增强 |

---

## 🔬 路径 C: 突破瓶颈（4-6 周）

### 目标
通过 CycleGAN 域适应、超参数搜索和集成学习，突破 0.720 mAP50 瓶颈。

### Week 1-2: CycleGAN 风格转换

```bash
# 1. 克隆 CycleGAN 仓库
git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.git
cd pytorch-CycleGAN-and-pix2pix

# 2. 准备数据
mkdir datasets/openscore2real
# Domain A: Phase 7 真实乐谱
# Domain B: OpenScore 渲染图片

# 3. 训练 CycleGAN (2-3 天，RTX 5090)
python train.py --dataroot datasets/openscore2real \
                --name openscore2real_cyclegan \
                --model cycle_gan \
                --gpu_ids 0 \
                --batch_size 4 \
                --n_epochs 100

# 4. 转换 OpenScore 图片
python test.py --dataroot datasets/openscore2real/testB \
               --name openscore2real_cyclegan \
               --model cycle_gan \
               --results_dir results/openscore_converted
```

### Week 3-4: 超参数搜索

```bash
pip install optuna

cat > hyperparameter_search.py << 'EOF'
import optuna
from ultralytics import YOLO

def objective(trial):
    # 定义超参数搜索空间
    lr0 = trial.suggest_float('lr0', 0.0005, 0.002, log=True)
    cls = trial.suggest_float('cls', 0.3, 0.7)
    box = trial.suggest_float('box', 5.0, 10.0)
    warmup_epochs = trial.suggest_int('warmup_epochs', 3, 10)
    mosaic = trial.suggest_float('mosaic', 0.3, 0.7)

    model = YOLO('yolo12s.pt')
    results = model.train(
        data='datasets/yolo_harmony_v2_phase9.3_full/harmony_phase9.3.yaml',
        epochs=50,  # 快速评估
        batch=16,
        lr0=lr0,
        cls=cls,
        box=box,
        warmup_epochs=warmup_epochs,
        mosaic=mosaic,
        project='hyperparameter_search',
        name=f'trial_{trial.number}'
    )

    return results.results_dict['metrics/mAP50(B)']

# 运行 50 次试验
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print(f"最佳超参数: {study.best_params}")
print(f"最佳 mAP50: {study.best_value:.4f}")
EOF

python3 hyperparameter_search.py
```

### Week 5-6: 集成学习与最终训练

```bash
cat > train_ensemble.py << 'EOF'
from ultralytics import YOLO

# 训练3个不同的模型
models = []
for i in range(3):
    model = YOLO('yolo12s.pt')
    results = model.train(
        data='datasets/yolo_harmony_v2_phase9.3_full/harmony_phase9.3.yaml',
        epochs=200,
        batch=16,
        seed=42 + i,  # 不同的随机种子
        project='harmony_omr_v2_phase9.3',
        name=f'ensemble_model_{i}'
    )
    models.append(f'harmony_omr_v2_phase9.3/ensemble_model_{i}/weights/best.pt')

print(f"集成模型: {models}")
EOF
```

### 预期结果

| 指标 | Phase 8 | Phase 9.3 预期 | 提升 |
|------|---------|---------------|------|
| mAP50 | 0.6444 | **0.720-0.750** | +11.7% to +16.4% |

---

## 📊 进度追踪检查点

### 路径 A 检查点

- [ ] Day 1: Phase 9.1 数据集创建完成
- [ ] Day 6: Phase 9.1 训练完成
- [ ] Day 7: Phase 9.1 mAP50 >= 0.650
- [ ] Day 7: 确认问题根源，决定下一步

### 路径 B 检查点

- [ ] Week 1 Day 3: OpenScore 伪标注生成完成
- [ ] Week 1 Day 5: 人工审查完成（抽样 100 张）
- [ ] Week 2 Day 2: 真实化增强完成
- [ ] Week 2 Day 7: Stage 1 训练完成
- [ ] Week 3 Day 4: Stage 2 训练完成
- [ ] Week 3 Day 7: Stage 3 训练完成，mAP50 >= 0.680

### 路径 C 检查点

- [ ] Week 2: CycleGAN 训练完成
- [ ] Week 3: OpenScore 图片转换完成
- [ ] Week 4: 超参数搜索完成
- [ ] Week 6: 集成模型训练完成，mAP50 >= 0.720

---

## 🛠️ 故障排除指南

### 问题 1: Phase 9.1 仍然表现不佳 (mAP50 < 0.650)

**可能原因**：
1. DeepScores 数据存在问题
2. Phase 8 模型本身有局限性

**解决方案**：
```bash
# 1. 审查 DeepScores 数据质量
python3 << 'EOF'
from pathlib import Path
from PIL import Image

deepscores_dir = Path('datasets/yolo_deepscores_dynamics')
sample_images = list(deepscores_dir.glob('images/train/*.png'))[:10]

for img_path in sample_images:
    img = Image.open(img_path)
    print(f"{img_path.name}: {img.size}, mode={img.mode}")
    # 检查是否过大
    if img.width > 2000 or img.height > 2000:
        print(f"  ⚠️  尺寸过大，建议 resize")
EOF

# 2. 移除 DeepScores，仅使用 Phase 8
# 如果仍不达标，说明需要更根本的架构改进
```

### 问题 2: 伪标注质量差

**解决方案**：
```bash
# 提高置信度阈值
model.predict(..., conf=0.4)  # 从 0.25 提高到 0.4

# 或使用集成伪标注（多个模型投票）
```

### 问题 3: 训练过程 OOM

**解决方案**：
```python
# 减少 batch size
batch=12  # 从 16 降到 12

# 或降低图像尺寸
imgsz=512  # 从 640 降到 512
```

---

## 📈 成功指标

### 最低可接受标准（路径 A）

- Phase 9.1 mAP50 >= 0.650（超越 Phase 8 +0.6%）
- dynamic_loud/soft mAP50 >= 0.750

### 目标标准（路径 B）

- Phase 9.2 mAP50 >= 0.680（超越 Phase 8 +5.5%）
- fermata mAP50 >= 0.700
- barline_double mAP50 >= 0.750

### 优秀标准（路径 C）

- Phase 9.3 mAP50 >= 0.720（超越 Phase 8 +11.7%）
- 所有瓶颈类别 mAP50 >= 0.600

---

## 🔗 相关资源

### 必读文档

- `PHASE9_DATA_SCIENCE_ANALYSIS.md` - 完整的数据科学分析
- `PHASE8_COMPLETE_ANALYSIS.md` - Phase 8 训练总结
- `OPENSCORE_MSCX_RENDERING_STATUS.md` - OpenScore 数据分析

### 工具脚本

- `analyze_phase8_vs_phase9.py` - 自动化分析脚本
- `yolo12_train_phase9.1.py` - Phase 9.1 训练脚本
- `augment_openscore.py` - 真实化数据增强脚本

### 外部资源

- [YOLO Ultralytics 文档](https://docs.ultralytics.com/)
- [Albumentations 文档](https://albumentations.ai/docs/)
- [CycleGAN 论文](https://arxiv.org/abs/1703.10593)
- [Domain Adaptation 综述](https://arxiv.org/abs/2009.00155)

---

## 💬 联系与支持

如果在执行过程中遇到问题，请：

1. 查看 `training/logs/` 目录下的训练日志
2. 运行 `python3 analyze_phase8_vs_phase9.py` 生成诊断报告
3. 查阅相关文档或 GitHub Issues

---

**文档版本**: v1.0
**最后更新**: 2025-12-10
**作者**: Data Science Team
