# 🔄 Claude Code 重啟恢復指南

**更新時間**: 2025-12-10 16:13 UTC+8
**使用方法**: 重啟 Claude Code 後，直接貼上「快速恢復提示詞」即可

---

## 📋 快速恢復提示詞（直接複製使用）

```
請閱讀以下文件，快速了解當前訓練狀態：

1. training/PHASE9_FIXED_FINAL_REPORT.md - Phase 9 修正訓練失敗報告
2. training/CLAUDE.md - 專案完整背景與歷史
3. training/STATUS_SUMMARY.md - 當前狀態摘要

重點：
- Phase 9 修正訓練已完成，但結果失敗（mAP50-95 = 0.5124，低於 Phase 8 的 0.5809）
- 當前最佳模型：harmony_omr_v2_phase8/phase8_training/weights/best.pt
- 需要決策：回退 Phase 8 或重新規劃數據整合策略
```

---

## 📊 當前狀態摘要（2025-12-10 16:13）

### 🚨 訓練狀態：Phase 9 修正訓練失敗

| 模型版本 | mAP50-95 | mAP50 | Epochs | 狀態 |
|----------|----------|-------|--------|------|
| **Phase 8** | **0.5809** ✅ | 0.6444 | 150 | **當前最佳** |
| Phase 9 原始 | 0.5213 | 0.5841 | 100 | 未達預期 |
| Phase 9 修正 | 0.5124 ❌ | 0.5702 | 150 | **失敗** |

### 🔴 關鍵問題

**Phase 9 失敗原因**：
1. 新增的 OpenScore Lieder (5,238) 和 DeepScores (847) 數據導致負遷移
2. 數據質量/域差異問題
3. 訓練配置可能不適合更大數據集

**監控錯誤**：
- 之前誤以為 mAP50-95 = 0.7705（實際是 val/dfl_loss）
- 導致錯誤判斷訓練進展

### 📁 重要模型位置

**生產使用（當前最佳）**：
```bash
harmony_omr_v2_phase8/phase8_training/weights/best.pt
# mAP50-95: 0.5809, Size: 18.9 MB
```

**失敗模型（保留分析）**：
```bash
harmony_omr_v2_phase9_fixed/phase9_with_phase8_config2/weights/best.pt
# mAP50-95: 0.5124, Size: 19 MB
```

### 🎯 待決策事項

**選項 A**：回退使用 Phase 8（最安全）
```bash
cp harmony_omr_v2_phase8/phase8_training/weights/best.pt \
   production_models/harmony_omr_best.pt
```

**選項 B**：數據集質量審查
- 單獨測試 OpenScore 數據影響
- 檢查 DeepScores 渲染質量
- 漸進式重新整合

**選項 C**：執行 Phase 10（跳過問題數據）
- AudioLabs v2 轉換（940 圖片）
- 生成 double_sharp 合成數據
- 避免使用 OpenScore 數據

---

## 📂 關鍵文件位置

### 訓練結果與報告
```
training/PHASE9_FIXED_FINAL_REPORT.md        - Phase 9 失敗完整分析
training/PHASE8_COMPLETE_ANALYSIS.md         - Phase 8 成功分析
training/PHASE9_EXECUTION_REPORT.md          - Phase 9 原始執行報告
training/PHASE9_CRITICAL_ANALYSIS.md         - Phase 9 瓶頸分析
```

### 訓練日誌
```
training/harmony_omr_v2_phase9_fixed/phase9_with_phase8_config2/results.csv
training/harmony_omr_v2_phase9_fixed/phase9_with_phase8_config/training.log
training/logs/ultimate_pipeline/pipeline_20251209_210437.log
```

### 數據集配置
```
training/datasets/yolo_harmony_v2_phase9_merged/harmony_phase9_merged.yaml
training/datasets/yolo_harmony_v2_phase8/harmony_phase8.yaml
```

### 訓練腳本
```
training/yolo12_train_phase9_fixed.py        - Phase 9 修正訓練腳本
training/yolo12_train_phase8.py              - Phase 8 訓練腳本（成功）
training/ULTIMATE_AUTO_TRAINING_PIPELINE.py  - 自動化訓練管道
```

---

## 🔢 數據集統計

### Phase 8（成功，32K 圖片）
```
訓練集: 29,300 圖片
驗證集: 3,255 圖片
來源: MUSCIMA++, Rebelo, DoReMi, Fornes, Choi
```

### Phase 9（失敗，41K 圖片）
```
訓練集: 41,281 圖片
驗證集: 4,583 圖片
新增: OpenScore Lieder (5,238), DeepScores (847)
問題: 新增數據導致性能下降
```

---

## 🚀 下一步待辦事項

### ✅ 已完成
1. Phase 8 訓練（mAP50-95 = 0.5809）
2. Phase 9 原始訓練（mAP50-95 = 0.5213）
3. Phase 9 修正訓練（失敗，mAP50-95 = 0.5124）
4. 失敗原因分析與報告

### 🔴 待決策（優先級 1）
1. **決定是否回退 Phase 8**
2. **數據集質量審查策略**
3. **下一階段訓練方向**

### 🟡 Phase 10 準備（優先級 2）
1. 生成 double_sharp 合成數據（LilyPond）
2. 轉換 AudioLabs v2 數據集（940 圖片）
3. 避免使用 OpenScore 數據（已證明有問題）

### 🟢 長期任務（優先級 3）
1. Phase 11: 漸進式整合 DeepScores V2
2. Phase 12: 實現 CBAM 注意力機制
3. Phase 13: 高解析度訓練（640→1024）
4. Phase 14: 知識蒸餾
5. Phase 15: TFLite 生產優化

---

## 💻 常用命令

### 檢查 GPU 狀態
```bash
nvidia-smi
```

### 檢查訓練狀態
```bash
./check_training_status.sh
```

### 查看最新訓練結果
```bash
tail -20 harmony_omr_v2_phase9_fixed/phase9_with_phase8_config2/results.csv
```

### 比較模型性能
```bash
python3 -c "
import pandas as pd
p8 = pd.read_csv('harmony_omr_v2_phase8/phase8_training/results.csv')
p9 = pd.read_csv('harmony_omr_v2_phase9_fixed/phase9_with_phase8_config2/results.csv')
print(f'Phase 8 最佳: {p8[\"metrics/mAP50-95(B)\"].max():.4f}')
print(f'Phase 9 修正: {p9[\"metrics/mAP50-95(B)\"].max():.4f}')
"
```

---

## 🔧 環境資訊

**硬體**：
- GPU: NVIDIA RTX 5090 (32GB)
- 訓練速度: ~6-7 it/s (batch=24)
- 單 epoch 時間: ~6-7 分鐘

**軟體**：
- YOLO: ultralytics (YOLO12)
- Python: venv_yolo12 虛擬環境
- 工作目錄: /home/thc1006/dev/music-app/training

**數據集**：
- 33 個音樂符號類別
- 圖片尺寸: 640x640
- 混合精度: AMP enabled

---

## 📖 歷史訓練記錄

| Phase | 完成時間 | mAP50-95 | mAP50 | 說明 |
|-------|----------|----------|-------|------|
| 1 | 2025-11-23 | - | 0.452 | 基礎訓練 |
| 2 | 2025-11-24 | - | 0.509 | 類別平衡 |
| 3 | 2025-11-25 | - | 0.580 | 外部數據整合 |
| 4-7 | 2025-11-26 | - | - | 多階段實驗 |
| **8** | **2025-11-28** | **0.5809** | **0.6444** | ✅ **最佳** |
| 9 原始 | 2025-11-28 | 0.5213 | 0.5841 | ⚠️ 未達預期 |
| 9 修正 | 2025-12-10 | 0.5124 | 0.5702 | ❌ 失敗 |

---

## 🎯 當前目標

**短期（本週）**：
- 決定是否回退 Phase 8
- 審查數據集質量
- 規劃 Phase 10 執行策略

**中期（2週內）**：
- 完成 Phase 10 訓練
- 超越 Phase 8 的 0.5809 mAP50-95
- 建立穩定的數據增強流程

**長期（1個月）**：
- 達到 mAP50-95 > 0.70
- 完成模型優化與量化
- 準備 Android 部署

---

## 🆘 如果遇到問題

### Q: 想知道當前最佳模型在哪？
A: `harmony_omr_v2_phase8/phase8_training/weights/best.pt` (mAP50-95 = 0.5809)

### Q: Phase 9 為什麼失敗？
A: 閱讀 `training/PHASE9_FIXED_FINAL_REPORT.md` 完整分析

### Q: 下一步該做什麼？
A: 參考本文「下一步待辦事項」章節，優先決策是否回退 Phase 8

### Q: 如何重新開始訓練？
A: 確認 GPU 閒置後，執行：
```bash
cd /home/thc1006/dev/music-app/training
source venv_yolo12/bin/activate
python yolo12_train_phase10.py  # 或其他 phase
```

---

## 📞 聯絡資訊

**專案位置**: `/home/thc1006/dev/music-app/training`
**Git Branch**: `main`
**最後更新**: 2025-12-10 16:13 UTC+8

---

**重要提醒**：
- Phase 8 是當前最佳且唯一可靠的生產模型
- Phase 9 數據集存在質量問題，需要審查
- 不要輕易使用 OpenScore 數據，已證明會降低性能
- 所有重要決策前，請先與使用者確認
