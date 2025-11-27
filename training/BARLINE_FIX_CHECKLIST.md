# Barline 修復執行檢查清單

## 執行前檢查 ✓

- [ ] 確認在正確目錄：`/home/thc1006/dev/music-app/training`
- [ ] 確認 Phase 5 數據集存在：`datasets/yolo_harmony_v2_phase5/`
- [ ] 確認虛擬環境存在：`venv_yolo12/`
- [ ] 硬碟空間充足：需要約 2.5GB（`df -h` 檢查）
- [ ] GPU 空閒（如果要接著訓練）：`nvidia-smi`

## 執行步驟 ⚙️

### 選項 A: 一鍵執行（推薦）

```bash
cd /home/thc1006/dev/music-app/training
./run_fix_barline.sh
```

- [ ] 腳本詢問是否運行測試 → 選擇 `y`（推薦）
- [ ] 測試全部通過 → ✅
- [ ] 腳本詢問是否繼續修復 → 選擇 `y`
- [ ] 修復完成 → 查看輸出訊息

### 選項 B: 手動執行

```bash
cd /home/thc1006/dev/music-app/training
source venv_yolo12/bin/activate

# 步驟 1: 測試（可選）
python test_fix_barline.py

# 步驟 2: 修復
python fix_barline_annotations.py
```

## 執行後檢查 ✓

### 1. 基本檢查

- [ ] 輸出目錄存在：`ls -ld datasets/yolo_harmony_v2_phase6_fixed/`
- [ ] 檔案數量正確：
  ```bash
  echo "Train: $(ls datasets/yolo_harmony_v2_phase6_fixed/train/images/*.png | wc -l) / 22393"
  echo "Val: $(ls datasets/yolo_harmony_v2_phase6_fixed/val/images/*.png | wc -l) / 2517"
  ```
  應該顯示：`Train: 22393 / 22393` 和 `Val: 2517 / 2517`

- [ ] 報告文件生成：
  ```bash
  ls -lh datasets/yolo_harmony_v2_phase6_fixed/{fix_report.txt,*.png}
  ```

### 2. 修復效果檢查

- [ ] 查看修復摘要：
  ```bash
  tail -20 datasets/yolo_harmony_v2_phase6_fixed/fix_report.txt
  ```

- [ ] 確認極細線已修復（barline）：
  ```bash
  grep "極細線" datasets/yolo_harmony_v2_phase6_fixed/fix_report.txt
  ```
  修復後應該顯示 `0 (0.0%)` 或接近 0

- [ ] 確認過大框已緊縮（barline_double/final）：
  ```bash
  grep "過大框" datasets/yolo_harmony_v2_phase6_fixed/fix_report.txt
  ```
  修復後百分比應顯著降低

### 3. 可視化檢查

- [ ] 下載可視化圖片到本地：
  ```bash
  # 在本地電腦執行（替換 user 和 server）
  scp user@server:/home/thc1006/dev/music-app/training/datasets/yolo_harmony_v2_phase6_fixed/*.png ./
  ```

- [ ] 查看 `fix_comparison.png`：
  - 紅色框（修復前）應該比綠色框（修復後）明顯細/大
  - 綠色框應該在合理範圍內

- [ ] 查看 `distribution_comparison.png`：
  - barline 寬度分佈應該右移（變寬）
  - barline_double/final 面積分佈應該左移（變小）

### 4. 隨機抽樣驗證

- [ ] 隨機抽樣 5 個文件檢查：
  ```bash
  for i in {1..5}; do
    SAMPLE=$(ls datasets/yolo_harmony_v2_phase5/train/labels/*.txt | shuf -n 1 | xargs basename)
    echo "=== 檔案: $SAMPLE ==="
    echo "修復前 barline 樣本:"
    grep "^23 " datasets/yolo_harmony_v2_phase5/train/labels/$SAMPLE | head -2
    echo "修復後 barline 樣本:"
    grep "^23 " datasets/yolo_harmony_v2_phase6_fixed/train/labels/$SAMPLE | head -2
    echo ""
  done
  ```

- [ ] 檢查修復後寬度是否 >= 0.015

## 準備 Phase 6 訓練 🚀

### 1. 更新配置文件

- [ ] 檢查 YAML 配置：
  ```bash
  cat datasets/yolo_harmony_v2_phase6_fixed/harmony_phase6_fixed.yaml
  ```

- [ ] 確認路徑正確：
  - `path:` 應指向 `yolo_harmony_v2_phase6_fixed`
  - `train:` 應為 `train/images`
  - `val:` 應為 `val/images`

### 2. 創建訓練腳本（如果還沒有）

- [ ] 複製 Phase 5 訓練腳本：
  ```bash
  cp yolo12_train_phase5.py yolo12_train_phase6.py
  ```

- [ ] 修改配置：
  - 數據集路徑：`data='datasets/yolo_harmony_v2_phase6_fixed/harmony_phase6_fixed.yaml'`
  - 項目名稱：`project='harmony_omr_v2_phase6'`
  - 訓練名稱：`name='barline_fixed_training'`
  - 預訓練模型：從 Phase 5 最佳權重開始

### 3. 檢查 GPU 可用性

- [ ] GPU 記憶體充足：
  ```bash
  nvidia-smi
  ```
  應該顯示 < 500MB 使用量

- [ ] 沒有其他訓練在運行：
  ```bash
  ps aux | grep python | grep train
  ```

### 4. 啟動訓練（準備好時）

- [ ] 使用 tmux/screen 後台運行：
  ```bash
  tmux new -s phase6_train
  cd /home/thc1006/dev/music-app/training
  source venv_yolo12/bin/activate
  python yolo12_train_phase6.py
  ```

- [ ] Detach tmux：按 `Ctrl+B` 然後 `D`

- [ ] 重新連接：`tmux attach -t phase6_train`

## 預期結果 📊

### 訓練完成後應檢查：

- [ ] barline mAP50 >= 0.50（目標：0.50-0.60）
- [ ] barline 召回率 >= 0.45（目標：0.45-0.55）
- [ ] barline_double mAP50 >= 0.35（目標：0.40-0.50）
- [ ] 整體 mAP50 >= 0.65（目標：0.65-0.68）

### 如果結果不理想：

- [ ] barline mAP50 < 0.40
  → 考慮進一步增大最小寬度（0.020）
  → 或增加 barline 類別權重

- [ ] barline_double mAP50 < 0.30
  → 需要合成數據補充（Abjad 生成）
  → 樣本數仍然太少（1,883）

- [ ] 整體 mAP50 < 0.60
  → 檢查是否有其他類別退化
  → 調整訓練超參數（學習率、batch size）

## 備份與清理 🗄️

### 訓練完成後：

- [ ] 備份 Phase 6 最佳權重：
  ```bash
  cp harmony_omr_v2_phase6/barline_fixed_training/weights/best.pt \
     models/phase6_best_$(date +%Y%m%d).pt
  ```

- [ ] 保存修復報告：
  ```bash
  cp datasets/yolo_harmony_v2_phase6_fixed/fix_report.txt \
     reports/phase6_fix_report_$(date +%Y%m%d).txt
  ```

- [ ] （可選）清理中間訓練文件：
  ```bash
  # 保留 weights/ 和 results.csv，刪除其他
  rm -rf harmony_omr_v2_phase6/barline_fixed_training/train/
  ```

## 問題排查 🔧

### 問題 1: 修復腳本報錯

- [ ] 檢查依賴安裝：
  ```bash
  source venv_yolo12/bin/activate
  pip list | grep -E "matplotlib|Pillow|tqdm|numpy"
  ```

- [ ] 重新安裝：
  ```bash
  pip install --upgrade matplotlib Pillow tqdm numpy
  ```

### 問題 2: 硬碟空間不足

- [ ] 檢查空間：
  ```bash
  df -h /home/thc1006/dev/music-app/training
  ```

- [ ] 清理舊訓練：
  ```bash
  # 檢查可刪除的大文件
  du -sh harmony_omr_v2_phase*/*/
  # 謹慎刪除不需要的訓練副本
  ```

### 問題 3: 修復數量為 0

- [ ] 檢查類別 ID 是否正確：
  ```bash
  # 查看 Phase 5 標註中是否有 barline (ID 23)
  grep "^23 " datasets/yolo_harmony_v2_phase5/train/labels/*.txt | wc -l
  ```

- [ ] 應該顯示約 23,000+

### 問題 4: 訓練性能沒有提升

- [ ] 檢查修復是否生效：
  ```bash
  # 對比修復前後的寬度
  grep "^23 " datasets/yolo_harmony_v2_phase5/train/labels/*.txt | \
    awk '{sum+=$4; count++} END {print "Phase 5 平均寬度:", sum/count}'

  grep "^23 " datasets/yolo_harmony_v2_phase6_fixed/train/labels/*.txt | \
    awk '{sum+=$4; count++} END {print "Phase 6 平均寬度:", sum/count}'
  ```

- [ ] Phase 6 平均寬度應該明顯增加

## 完成標記 ✅

- [ ] 所有執行前檢查通過
- [ ] 修復腳本成功完成
- [ ] 所有執行後檢查通過
- [ ] 可視化圖表已查看並確認合理
- [ ] Phase 6 訓練腳本已準備
- [ ] 已閱讀並理解預期結果

---

**檢查清單最後更新**: 2025-11-26
**適用版本**: fix_barline_annotations.py v1.0
