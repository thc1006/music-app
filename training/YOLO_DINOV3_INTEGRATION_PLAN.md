# YOLO12 + DINOv3 集成完整计划

## 创建日期: 2025-12-20
## 目标: 评估并实施 DINOv3 对 OMR 模型的提升

---

## 📊 执行摘要

### 当前状态
- **最佳模型**: Phase 8, mAP50 = 0.6447
- **瓶颈**: 小物件检测 (flag, dot, accidentals)
- **目标**: 通过 DINOv3 集成提升至 mAP50 > 0.70

### 关键发现

| 因素 | 评估 | 影响 |
|------|------|------|
| DINOv3 小物件能力 | ⚠️ 有限 | OMR 核心需求受限 |
| 知识蒸馏方案 | ✅ 低风险 | 推荐首选 |
| 端侧部署 | ⚠️ 需验证 | ViT 量化更难 |
| 训练资源 | ✅ 充足 | RTX 5090 32GB |

### 推荐路径
**Phase 1: LightlyTrain 知识蒸馏** → 低风险验证 DINOv3 效果

---

## 📋 分阶段实验计划

### Phase 0: 环境准备 (Day 1)

```bash
# 1. 创建新虚拟环境
cd /home/thc1006/dev/music-app/training
python3 -m venv venv_dinov3
source venv_dinov3/bin/activate

# 2. 安装依赖
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install ultralytics==8.3.229
pip install lightly-train
pip install timm>=1.0.20

# 3. 验证
python3 -c "import lightly_train; print('OK')"
```

**检查点**:
- [ ] 虚拟环境创建成功
- [ ] LightlyTrain 安装成功
- [ ] DINOv3 权重可下载

---

### Phase 1: 知识蒸馏实验 (Day 2-4) ⭐

**目标**: 使用 DINOv3 作为教师模型蒸馏到 YOLO12s

**执行**:
```bash
source venv_dinov3/bin/activate
python3 yolo12_dinov3_distillation.py
```

**预期结果**:
| 指标 | Phase 8 Baseline | 蒸馏后目标 |
|------|------------------|-----------|
| mAP50 | 0.6447 | > 0.67 |
| mAP50-95 | 0.5809 | > 0.60 |
| 推理时间 | ~15ms | ~15ms (不变) |

**决策点 A**:
- 如果 mAP50 提升 > 3% → 继续 Phase 2
- 如果 mAP50 提升 1-3% → 评估是否值得
- 如果 mAP50 无提升或下降 → 放弃 DINOv3 方案

---

### Phase 2: DEIMv2 对比实验 (Day 5-7)

**目标**: 评估是否值得切换到 DEIMv2 架构

**执行**:
```bash
# 克隆 DEIMv2
git clone https://github.com/Intellindust-AI-Lab/DEIMv2.git
cd DEIMv2

# 准备 OMR 数据集配置
# (需要将 YOLO 格式转换为 COCO 格式)

# 训练 DEIMv2-S
python train.py --config configs/deimv2_s.yaml \
    --data /path/to/omr_coco_format \
    --epochs 100
```

**对比指标**:
| 模型 | 参数量 | mAP50 | 推理时间 | 端侧部署 |
|------|--------|-------|---------|---------|
| YOLO12s | 9.3M | 0.6447 | ~15ms | ✅ 成熟 |
| YOLO12s + 蒸馏 | 9.3M | ? | ~15ms | ✅ 相同 |
| DEIMv2-S | 9.7M | ? | ? | ⚠️ 需测试 |

**决策点 B**:
- 如果 DEIMv2 性能显著更好 (>5%) → 考虑切换
- 如果性能相近 → 保持 YOLO + 蒸馏
- 考虑端侧部署复杂度

---

### Phase 3: 端侧部署测试 (Day 8-10)

**目标**: 验证最终方案的 Android 部署可行性

**测试项目**:
1. TFLite 转换
2. INT8 量化
3. 推理速度
4. 内存占用

**检查点**:
- [ ] 模型成功转换为 TFLite
- [ ] 量化后精度损失 < 2%
- [ ] 推理时间 < 100ms (中端手机)
- [ ] 内存占用 < 200MB

---

## 🔧 关键代码文件

| 文件 | 用途 |
|------|------|
| `yolo12_dinov3_distillation.py` | 知识蒸馏主脚本 |
| `YOLO_DINOV3_INTEGRATION_PLAN.md` | 本计划文件 |
| `yolo12_train_phase8.py` | Phase 8 baseline 参考 |

---

## ⚠️ 风险和缓解措施

### 风险 1: 小物件检测无提升
- **可能性**: 高
- **影响**: DINOv3 集成价值降低
- **缓解**: 先进行小规模验证，确认效果再全面投入

### 风险 2: 端侧部署困难
- **可能性**: 中
- **影响**: 无法在手机上使用
- **缓解**: 保持 YOLO 架构不变（知识蒸馏方案）

### 风险 3: 训练时间过长
- **可能性**: 低
- **影响**: 延迟项目进度
- **缓解**: RTX 5090 资源充足，可并行实验

---

## 📚 参考资源

- [DINOv3 GitHub](https://github.com/facebookresearch/dinov3)
- [LightlyTrain GitHub](https://github.com/lightly-ai/lightly-train)
- [DEIMv2 GitHub](https://github.com/Intellindust-AI-Lab/DEIMv2)
- [DEIMv2 论文](https://arxiv.org/abs/2509.20787)
- [Yolo-DinoV2](https://github.com/itsprakhar/Yolo-DinoV2)

---

## 📊 实验记录模板

### Phase 1 实验记录

**日期**: ____
**配置**:
- 教师模型: DINOv3 ViT-S/16
- 学生模型: YOLO12s (Phase 8)
- 蒸馏温度: 4.0
- Epochs: 100

**结果**:
| 指标 | Baseline | 蒸馏后 | 提升 |
|------|----------|--------|-----|
| mAP50 | 0.6447 | | |
| mAP50-95 | 0.5809 | | |
| 小物件 AP | | | |
| 中物件 AP | | | |
| 大物件 AP | | | |

**观察和结论**:


---

## ✅ 最终决策清单

在完成所有实验后，根据以下因素做出最终决策：

- [ ] 蒸馏后 mAP50 提升幅度
- [ ] 小物件检测是否改善
- [ ] 端侧部署是否可行
- [ ] 与其他方案（高分辨率训练、合成数据）的对比
- [ ] 投入产出比评估

---

**文档版本**: 1.0
**更新日期**: 2025-12-20
