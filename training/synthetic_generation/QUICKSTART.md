# 快速開始指南

## 安裝（5 分鐘）

```bash
# 1. 進入目錄
cd /home/thc1006/dev/music-app/training/synthetic_generation

# 2. 啟動虛擬環境
source ../venv_yolo12/bin/activate

# 3. 驗證安裝
python -c "import verovio; print(f'Verovio {verovio.__version__} installed')"
```

## 生成測試數據（2 分鐘）

```bash
# 生成 10 張測試圖片
python generate_synthetic_barlines.py \
    --num-images 10 \
    --output-dir output_test \
    --workers 1 \
    --validation-samples 5
```

**預期輸出**:
```
開始生成 10 張合成圖像...
使用 1 個並行進程
生成進度: 100%|██████████| 10/10 [00:15<00:00]
成功: 1
失敗: 9
```

## 檢查結果

```bash
# 查看生成的圖像
ls -lh output_test/images/

# 查看標註文件
cat output_test/labels/barline_000000.txt

# 查看可視化樣本
ls output_test/validation/
```

## 當前狀態

⚠️ **系統處於 Alpha 階段**

- ✅ 基礎架構完成
- ✅ 圖像生成正常
- ❌ **Bbox 座標需要修復**（阻塞性問題）

**不要用於生產！** 請等待 v1.0.0 正式版本。

## 下一步

1. 閱讀 `TEST_REPORT.md` 了解詳細測試結果
2. 閱讀 `README.md` 了解完整功能
3. 查看 `configs/generation_config.yaml` 自定義配置

## 需要幫助？

- 查看 `README.md` 的故障排除部分
- 查看 `TEST_REPORT.md` 了解已知問題
- 聯繫開發團隊

## 配置示例

編輯 `configs/generation_config.yaml`:

```yaml
# 修改圖像大小
image:
  width: 2048
  height: 2048

# 調整 barline 分佈
barline_types:
  single: 0.5
  double: 0.2
  final: 0.3

# 增加 augmentation 強度
augmentation:
  paper_texture:
    intensity: 0.3  # 原本 0.15
```

## 性能調優

```bash
# 使用更多 CPU 核心（修復後）
python generate_synthetic_barlines.py \
    --num-images 1000 \
    --workers 8

# 跳過數據集組織（僅生成）
python generate_synthetic_barlines.py \
    --num-images 1000 \
    --skip-organization
```

## 故障排除

### 問題: ImportError: No module named 'verovio'

**解決**:
```bash
source ../venv_yolo12/bin/activate
pip install verovio
```

### 問題: Bbox 座標異常

**狀態**: 已知問題，正在修復

**臨時方案**: 等待 v1.0.0 版本

### 問題: 記憶體不足

**解決**:
```bash
# 減少並行進程
--workers 1

# 或減少批次大小（未來功能）
```

## 開發模式

```bash
# 僅生成 MEI（測試）
python -c "
from src.mei_generator import MEIGenerator
import yaml

with open('configs/generation_config.yaml') as f:
    config = yaml.safe_load(f)

gen = MEIGenerator(config)
mei = gen.generate_mei('single')
print(mei)
"

# 僅測試渲染（測試）
python -c "
from src.verovio_renderer import VerovioRenderer
import yaml

with open('configs/generation_config.yaml') as f:
    config = yaml.safe_load(f)

renderer = VerovioRenderer(config)
# ... (需要 MEI 內容)
"
```

## 版本歷史

- **v1.0.0-alpha** (2025-11-26): 初始版本，基礎架構完成
- **v1.0.0** (待定): Bbox 修復，可用於生產

## 貢獻

歡迎提交 Issue 和 PR！

主要需要幫助的領域：
1. 修復 bbox 座標歸一化
2. 優化 MEI 模板格式
3. 添加單元測試
4. 性能優化
