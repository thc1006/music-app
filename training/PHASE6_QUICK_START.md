# Phase 6 Quick Start Guide

## 快速啟動 Phase 6 訓練

### 前置檢查

1. **確認 GPU 可用**:
```bash
nvidia-smi
# 應顯示 RTX 5090, 記憶體使用 < 1GB
```

2. **確認 Phase 5 完成**:
```bash
ls -lh harmony_omr_v2_phase5/fermata_barline_enhanced/weights/best.pt
# 應該存在，約 18-19 MB
```

3. **確認數據集就緒**:
```bash
ls datasets/yolo_harmony_v2_phase5/harmony_phase5.yaml
# 應該存在
```

### 一鍵啟動

```bash
cd /home/thc1006/dev/music-app/training
python custom_training/train_phase6.py
```

**這將自動執行**：
1. ✅ Stage 1: 加權損失訓練 (150 epochs, ~4-6 小時)
2. ✅ Hard Example Mining (~30-60 分鐘)
3. ✅ Stage 2: 難例微調 (50 epochs, ~1-2 小時)
4. ✅ 最終評估與報告生成

**總計時間**: 6-9 小時

### 監控訓練

#### Terminal 1: 查看訓練進度
```bash
# Stage 1
tail -f harmony_omr_v2_phase6/stage1_weighted_loss/train.log

# Stage 2 (等 Stage 1 完成後)
tail -f harmony_omr_v2_phase6/stage2_hard_examples/train.log
```

#### Terminal 2: 監控 GPU
```bash
watch -n 1 nvidia-smi
```

### 關鍵指標

#### Stage 1 目標
- **mAP50**: 0.58 → 0.62+
- **barline (23)**: mAP 0.20 → 0.30-0.35
- **barline_double (24)**: mAP 0.14 → 0.20-0.25
- **cls_loss**: 1.25 → 0.8-1.0

#### Stage 2 目標
- **barline (23)**: mAP 0.30-0.35 → **0.50-0.60**
- **barline_double (24)**: mAP 0.20-0.25 → **0.40-0.50**
- **Overall mAP50**: 0.62 → **0.65-0.68**

### 預期輸出

訓練完成後：

```
harmony_omr_v2_phase6/
├── stage1_weighted_loss/
│   └── weights/best.pt          # Stage 1 模型
├── hard_example_mining/
│   ├── hard_examples_barline.txt
│   └── hard_examples_stats.json
├── hard_examples_dataset/
│   └── hard_examples.yaml       # 難例數據集
├── stage2_hard_examples/
│   └── weights/best.pt          # 🎯 最終模型
└── phase6_summary.json          # 訓練摘要
```

### 成功標準

✅ **達標**:
- barline mAP50 ≥ 0.50
- barline_double mAP50 ≥ 0.40
- Overall mAP50 ≥ 0.65

⚠️ **部分達標** (需要 Phase 6.1):
- barline mAP50 0.35-0.49
- barline_double mAP50 0.25-0.39
- Overall mAP50 0.62-0.64

❌ **未達標** (需要重新評估策略):
- barline mAP50 < 0.35
- barline_double mAP50 < 0.25
- Overall mAP50 < 0.62

### 如果訓練中斷

恢復 Stage 1:
```bash
# 編輯 train_phase6.py, 修改 stage1 配置
'resume': True  # 添加這行
```

跳過 Stage 1 (如果已完成):
```python
# 在 train_phase6.py main() 中註解掉
# stage1_results, stage1_weights = trainer.stage1_full_dataset()

# 直接指定 Stage 1 權重
stage1_weights = Path("harmony_omr_v2_phase6/stage1_weighted_loss/weights/best.pt")
```

### 故障排除

#### OOM (記憶體不足)
```bash
# 減少 batch size
# 編輯 custom_training/configs/phase6_config.yaml
stage1:
  batch: 12  # 從 16 改為 12
```

#### Stage 1 改進不明顯
```bash
# 增加權重
# 編輯 configs/phase6_config.yaml
stage1:
  class_weights:
    23: 6.0   # 從 4.0 提高
    24: 10.0  # 從 8.0 提高
```

#### Hard Examples 太少
```bash
# 降低難度閾值
# 編輯 configs/phase6_config.yaml
hem:
  min_difficulty: 1.0  # 從 1.5 降低
```

### 下一步

訓練完成後：

1. **評估結果**:
```bash
python custom_training/evaluate_phase6.py
```

2. **與 Phase 5 比較**:
```bash
python compare_phase5_phase6.py
```

3. **如果達標**: 準備部署
4. **如果未達標**: 檢查分析報告，規劃 Phase 6.1

### 聯繫支援

遇到問題？檢查：
- `custom_training/README.md` - 詳細文檔
- `custom_training/configs/phase6_config.yaml` - 配置說明
- `BARLINE_COMPLETE_ANALYSIS.txt` - 問題分析

---

**最後更新**: 2025-11-26
**估計時間**: 6-9 小時
**GPU 需求**: RTX 5090 或同等算力
