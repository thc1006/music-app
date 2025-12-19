# 🚨 Phase 9 修正訓練最終報告（失敗）

**訓練時間**: 2025-12-09 21:04 → 2025-12-10 12:41 (15.6 小時)
**報告時間**: 2025-12-10 16:13

---

## ❌ 結論：Phase 9 修正訓練失敗

### 📊 最終結果對比

| 模型 | mAP50-95 | mAP50 | Epochs | 狀態 |
|------|----------|-------|--------|------|
| **Phase 8** | **0.5809** | 0.6444 | 150 | ✅ **最佳** |
| **Phase 9 原始** | 0.5213 | 0.5841 | 100 | ⚠️ 未達預期 |
| **Phase 9 修正** | **0.5124** | 0.5702 | 150 | ❌ **最差** |

### 🔴 關鍵發現

1. **Phase 9 修正反而比 Phase 9 原始更差**
   - mAP50-95: 0.5124 vs 0.5213 (**-1.7%**)
   - mAP50: 0.5702 vs 0.5841 (**-2.4%**)

2. **相比 Phase 8 大幅退步**
   - mAP50-95: 0.5124 vs 0.5809 (**-11.8%**)
   - mAP50: 0.5702 vs 0.6444 (**-11.5%**)

3. **訓練收斂曲線**
   ```
   Epoch  10: 0.4738
   Epoch  50: 0.5048
   Epoch 100: 0.5116
   Epoch 130: 0.5124 ← 最佳
   Epoch 150: 0.5112 ← 開始過擬合
   ```

---

## 🔍 失敗原因分析

### 1. 監控錯誤導致誤判

**初始錯誤判斷**：
- 在 Epoch 42 時，我觀察到 `val/dfl_loss = 0.7705`
- **錯誤地認為這是 mAP50-95 = 0.7705**
- 實際上 mAP50-95 = 0.5034（遠低於預期）

**為何會誤判**：
- `val/dfl_loss` 和 `mAP50-95` 列位置接近
- 監控腳本只顯示部分列，導致列對齊錯誤
- 沒有明確標注列名

### 2. 數據集問題（負遷移）

Phase 9 數據集變化：
- **Phase 8**: 32,555 訓練圖片（純淨數據）
- **Phase 9**: 41,281 訓練圖片 (+26.8%)
  - 新增 OpenScore Lieder: 5,238 圖片
  - 新增 DeepScores: 847 圖片

**問題假設**：
1. **OpenScore 渲染質量問題**
   - MuseScore 自動渲染可能存在版面問題
   - 符號位置與真實樂譜差異大
   - 標註準確度不足

2. **DeepScores 域差異**
   - 合成數據風格與真實樂譜差異過大
   - 模型在合成數據上學到錯誤特徵

3. **數據不平衡加劇**
   - 新增數據可能集中在少數類別
   - 導致訓練偏向這些類別，損害整體性能

### 3. 訓練配置可能不適合

Phase 9 修正使用 Phase 8 配置：
```python
epochs=150,
lr0=0.001,
cls=0.5,
erasing=0.0,
warmup_epochs=3,
```

**問題**：
- Phase 8 配置是針對 32K 圖片優化的
- Phase 9 有 41K 圖片（+27%），可能需要不同配置
- 更大數據集可能需要：
  - 更低學習率（0.0005）
  - 更多 warmup epochs
  - 更強正則化（erasing > 0）

---

## 🎯 下一步行動建議

### 🔴 立即行動（Tier 1）

#### 選項 A：回退到 Phase 8 ✅ **推薦**

**原因**：
- Phase 8 是當前最佳模型（mAP50-95 = 0.5809）
- 經過完整 150 epochs 訓練
- 數據質量可靠

**行動**：
```bash
# 使用 Phase 8 作為生產模型
cp harmony_omr_v2_phase8/phase8_training/weights/best.pt \
   production_models/harmony_omr_best.pt
```

#### 選項 B：數據集質量審查 ⚠️ **必要**

**步驟**：
1. **隔離測試 OpenScore 數據**
   - 單獨訓練 Phase 8 + OpenScore
   - 驗證是否導致性能下降

2. **檢查 DeepScores 數據**
   - 可視化檢查渲染質量
   - 比較與真實樂譜的差異

3. **標註準確度驗證**
   - 隨機抽樣 100 張圖片
   - 人工檢查 bbox 準確度

#### 選項 C：漸進式數據整合 🟡 **推薦嘗試**

**策略**：
```
Phase 8 (32K) → Phase 8.1 (+2K OpenScore) → Phase 8.2 (+2K DeepScores)
```

**每步驗證**：
- 如果 mAP 下降 > 2%，停止添加該數據源
- 找出導致性能下降的具體數據

### 🟡 短期改進（Tier 2）

#### 1. 調整訓練超參數

針對 41K 數據集：
```python
epochs=200,           # 更多 epochs
lr0=0.0005,          # 更低學習率
cls=0.6,             # 稍微提高分類損失權重
erasing=0.2,         # 適度數據增強
warmup_epochs=5,     # 更長 warmup
```

#### 2. 使用 Phase 8 作為 warm start

```python
model = YOLO('harmony_omr_v2_phase8/phase8_training/weights/best.pt')
# 使用更保守的參數微調
model.train(
    data='...',
    epochs=50,
    lr0=0.0001,  # 極低學習率
    freeze=10,   # 凍結前 10 層
)
```

### 🟢 中長期改進（Tier 3）

#### 1. 優先執行 Phase 10 任務（不依賴 Phase 9 數據）

**任務列表**：
- ✅ 生成 double_sharp 合成數據（LilyPond）
- ✅ 轉換 AudioLabs v2 數據集（940 圖片）
- ⚠️ 跳過 OpenScore Lieder（已證明有問題）

#### 2. 改進 OpenScore 渲染流程

如果要繼續使用 OpenScore：
1. 使用 Verovio 代替 MuseScore（更精確）
2. 手動驗證 fermata 位置準確度
3. 對比多個渲染器輸出

#### 3. 探索其他數據源

- **IMSLP** 真實掃描樂譜（需要手動標註）
- **MuseScore 社群上傳樂譜**（品質參差）
- **學術數據集**（CVC-MUSCIMA, PrIMuS）

---

## 📋 經驗教訓

### ❌ 監控系統問題

**錯誤**：
- 混淆 validation loss 和 mAP 指標
- 沒有明確標注列名
- 過度依賴單一監控點

**改進**：
```python
# 未來監控腳本應明確標注
print(f"Epoch {epoch}: mAP50-95={map50_95:.4f} (not {val_loss})")
```

### ❌ 數據集整合策略

**錯誤**：
- 一次性添加大量未驗證數據（+8K 圖片）
- 沒有先進行小規模測試
- 信任自動渲染結果

**改進**：
- 漸進式整合（每次 +1K）
- 每次驗證性能變化
- 可視化檢查樣本質量

### ✅ 正確的訓練配置

**Phase 8 證明**：
```python
epochs=150,
lr0=0.001,
cls=0.5,
erasing=0.0,
```
這個配置對 32K 圖片是最優的。

---

## 📁 模型文件位置

### 當前最佳模型（生產使用）
```
harmony_omr_v2_phase8/phase8_training/weights/best.pt
mAP50-95: 0.5809 | Size: 18.9 MB
```

### Phase 9 失敗模型（保留供分析）
```
harmony_omr_v2_phase9_fixed/phase9_with_phase8_config2/weights/best.pt
mAP50-95: 0.5124 | Size: 19 MB
```

---

## 🚀 推薦執行順序

1. **立即**：回退使用 Phase 8 模型
2. **本週**：數據集質量審查（OpenScore + DeepScores）
3. **下週**：執行 Phase 10（AudioLabs v2 + 合成數據）
4. **評估後**：決定是否放棄 OpenScore 數據

**終極目標**：找到可靠的數據增強方法，突破 Phase 8 的 0.5809 mAP50-95。
