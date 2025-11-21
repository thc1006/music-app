# 🏆 「無敵」模型長期戰略 - 訓練進度日誌

> **記錄日期**：2025-11-22
> **目標**：打造生產級、可靠的四部和聲樂譜辨識系統

---

## 一、當前進度

| 指標 | 數值 |
|------|------|
| **Phase 1 進度** | Epoch 162/300 (54%) |
| **mAP50** | 0.41 (穩定) |
| **訓練腳本** | `yolo12_train_optimized.py` |
| **Log 檔案** | `training_optimized_phase1.log` |

---

## 二、已建立的戰略文件與工具

### 📄 文件

| 文件 | 用途 |
|------|------|
| `PERFECT_MODEL_STRATEGY.md` | 完整 6-Phase 路線圖 |
| `DATASET_INTEGRATION_PLAN.md` | 數據集整合評估 |
| `synthetic_data_generator.py` | 合成數據生成器（解決稀有類別） |
| `yolo12_train_phase2_balanced.py` | Phase 2 類別平衡訓練腳本 |
| `optimize_dataset_phase1.py` | Phase 1 數據集優化腳本 |

---

## 三、發現的核心問題

### 3.1 類別極度不平衡

```
致命瓶頸：
- Class 17 (重降記號 double_flat): 僅 12 個樣本
- Class 31 (強記號 dynamic_loud): 僅 27 個樣本
- Class 24 (雙小節線 barline_double): 僅 234 個樣本
- Class 16 (重升記號 double_sharp): 僅 338 個樣本

對比最多的類別：
- Class 0 (實心音符頭 notehead_filled): 501,814 個樣本

不平衡比例：41,818 : 1
```

### 3.2 這意味著什麼

```
即使整體 mAP 不錯，實際使用時：
- 重降/重升記號會完全辨識失敗
- 用戶寫出 B♭♭ 時，系統會判斷錯誤
- 錯誤的音高 → 錯誤的和聲判斷 → 用戶失去信任
```

---

## 四、六階段路線圖

```
                          mAP50 目標        時間
                              │
Phase 1 ─────────────────────┼── 0.50-0.55   Week 1-2  ← 目前進行中
                              │
Phase 2 (類別平衡) ──────────┼── 0.60-0.65   Week 3-4
                              │
Phase 3 (合成數據) ──────────┼── 0.70-0.75   Week 5-8
                              │
Phase 4 (高解析度) ──────────┼── 0.80-0.85   Week 9-12
                              │
Phase 5 (真實數據) ──────────┼── 0.85-0.90   Week 13-16
                              │
Phase 6 (生產優化) ──────────┼── 0.90+       Week 17+
                              │
                           「無敵」
```

### 各階段詳細說明

| Phase | 名稱 | 主要任務 | 預期 mAP50 | 時間 |
|-------|------|----------|-----------|------|
| 1 | 基礎訓練 | 穩定訓練、修復 OOM | 0.50-0.55 | Week 1-2 |
| 2 | 類別平衡 | 過採樣、類別加權 | 0.60-0.65 | Week 3-4 |
| 3 | 合成數據 | LilyPond 生成稀有類別 | 0.70-0.75 | Week 5-8 |
| 4 | 高解析度 | 960/1280px、YOLO12m | 0.80-0.85 | Week 9-12 |
| 5 | 真實數據 | OpenOMR、實際作業 | 0.85-0.90 | Week 13-16 |
| 6 | 生產優化 | TFLite、TTA、部署 | 0.90+ | Week 17+ |

---

## 五、關鍵洞察

### 5.1 為什麼「合成數據」是決勝關鍵

| 問題 | 原因 | 解法 |
|------|------|------|
| 重降記號僅 12 個 | 真實樂譜中本就罕見 | 用 LilyPond 生成 10,000 個 |
| 強記號僅 27 個 | DeepScoresV2 來源限制 | 自己生成包含強記號的樂譜 |

**數據量目標：**
```
當前：1,362 張圖片（DeepScoresV2 全部）
目標：10,000+ 張圖片（合成 + 真實混合）
每個類別：至少 5,000 個標註
```

### 5.2 為什麼不能只靠訓練技巧

```
即使用最強的：
- Focal Loss ✓
- 類別加權 ✓
- 過採樣 ✓

如果 Class 17 只有 12 個樣本，
模型物理上不可能學會辨識它。

解法：必須增加數據量（合成數據）
```

### 5.3 和聲分析真正需要的類別

**Tier 1: 核心類別（必須完美）**
- notehead_filled, notehead_hollow（音高核心）
- stem（聲部區分）
- clef_treble, clef_bass, clef_alto, clef_tenor（音高參考）
- accidental_sharp, flat, natural, double_sharp, double_flat（音高修正）
- key_signature（調號）
- barline（小節邊界）

**Tier 2: 有用但非核心**
- beam, flags（節奏）
- rests（休止）
- ledger_line（加線）
- time_signature（拍號）

**Tier 3: 可選**
- dynamics（力度記號）
- fermata（延長記號）
- barline_double, final, repeat

---

## 六、已完成的優化（2025-11-22）

### 6.1 數據集優化
- ✅ stem_down (Class 3) → 合併到 stem (Class 2)
- ✅ slur (Class 30) → 暫時排除（無數據）
- ✅ 驗證集：205 → 273 張 (+33%)
- ✅ 類別數：35 → 33

### 6.2 訓練配置優化
- ✅ Batch: 24 → 16（避免 OOM）
- ✅ LR: 0.01 → 0.005（減少震盪）
- ✅ Mosaic: 1.0 → 0.5（穩定性）
- ✅ Copy-paste: 關閉（不適合樂譜）
- ✅ Epochs: 600 → 300（效率）

### 6.3 結果
- OOM 錯誤：從 7,853 次 → 幾乎沒有
- mAP 震盪：從 ±0.15 → ±0.03
- 訓練時間：從 14+ 天 → 預計 1-2 天

---

## 七、下一步行動

### 自動執行（無需人工）
1. ✅ Phase 1 繼續訓練至完成（預計 1 天內）
2. ✅ Phase 2 腳本已準備好，可自動銜接

### 需要準備的（可平行進行）
1. **安裝 LilyPond**：`sudo apt install lilypond`
2. **收集真實數據**：拍攝四部和聲作業照片
3. **測試集準備**：準備 50-100 張真實測試圖片

### 執行命令備忘

```bash
# 查看當前訓練進度
tail -f training/training_optimized_phase1.log

# 查看 mAP 變化
grep "all.*mAP" training/training_optimized_phase1.log | tail -20

# Phase 1 完成後執行 Phase 2
cd training
python yolo12_train_phase2_balanced.py

# 檢查 LilyPond 是否安裝
python synthetic_data_generator.py --check-lilypond

# 生成稀有類別合成數據
python synthetic_data_generator.py --target-class 17 --count 1000
```

---

## 八、預期成果時間線

| 時間 | 預期成果 | 可用性 |
|------|----------|--------|
| 1 週後 | mAP50 > 0.55 | Demo/POC |
| 1 個月後 | mAP50 > 0.70 | Beta 可發布 |
| 3 個月後 | mAP50 > 0.85 | 生產可用 |
| 6 個月後 | mAP50 > 0.90 | 「無敵」級別 |

---

## 九、成功指標定義

### 量化指標
```
Milestone 1 (Phase 1 完成): mAP50 > 0.55
Milestone 2 (Phase 2 完成): mAP50 > 0.65, 所有類別 > 0.30
Milestone 3 (Phase 3 完成): mAP50 > 0.75, 核心類別 > 0.70
Milestone 4 (Phase 4 完成): mAP50 > 0.85, 核心類別 > 0.80
Milestone 5 (Phase 5 完成): mAP50 > 0.90, 核心類別 > 0.85
```

### 實際測試標準
```
測試集組成：
├── 50 張印刷四部和聲練習 (各出版社)
├── 50 張掃描樂譜 (不同掃描品質)
├── 50 張手寫四部和聲作業 (真實學生作品)
└── 50 張手機拍攝照片 (各種光線角度)

通過標準：
├── 印刷：95% 符號正確辨識
├── 掃描：90% 符號正確辨識
├── 手寫：80% 符號正確辨識
└── 拍照：85% 符號正確辨識
```

---

## 十、總結

**這個長期策略確保了：**
1. ✅ 每個類別都有足夠訓練數據
2. ✅ 模型能處理各種輸入質量
3. ✅ 持續迭代改進有明確路徑
4. ✅ 最終達到生產級可靠性

**核心理念：**
> 「無敵」模型不是一次訓練能達成的，而是需要正確的優先級、足夠的數據、持續的迭代。

---

*文檔版本：1.0*
*創建日期：2025-11-22*
*最後更新：2025-11-22*
