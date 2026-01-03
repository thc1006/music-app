# LightlyTrain + YOLO12 + DINOv3 蒸餾完整指南

## 調研日期: 2025-12-20
## 來源: 官方文檔調研

---

## 1. 關鍵發現

### ✅ LightlyTrain 確實支援 YOLO12

| 模型名稱 | 說明 |
|---------|------|
| `ultralytics/yolov12n.yaml` | YOLO12 Nano (無預訓練) |
| `ultralytics/yolov12n.pt` | YOLO12 Nano (有預訓練) |
| `ultralytics/yolov12s.yaml` | YOLO12 Small (無預訓練) |
| `ultralytics/yolov12s.pt` | YOLO12 Small (有預訓練) |
| `ultralytics/yolov12m.yaml` | YOLO12 Medium |
| `ultralytics/yolov12l.yaml` | YOLO12 Large |
| `ultralytics/yolov12x.yaml` | YOLO12 XLarge |

### ✅ DINOv3 作為教師模型

可用的 DINOv3 教師模型:
- `dinov3/vitt16` - Tiny
- `dinov3/vits16` - Small (推薦, 平衡性能)
- `dinov3/vitb16` - Base
- `dinov3/vitl16` - Large
- `dinov3/convnext-small`
- `dinov3/convnext-base`

---

## 2. 正確的蒸餾代碼

### 2.1 基本蒸餾 (DINOv3 → YOLO12)

```python
import lightly_train

if __name__ == "__main__":
    # DINOv3 作為教師，YOLO12s 作為學生
    lightly_train.train(
        out="out/yolo12_dinov3_distill",
        data="path/to/images",
        model="ultralytics/yolov12s.pt",  # ✅ 正確的模型名稱
        method="distillation",
        method_args={
            "teacher": "dinov3/vits16",  # ✅ DINOv3 教師
        },
        epochs=50,
        batch_size=32,
        num_workers=20,
    )
```

### 2.2 使用現有 YOLO 模型實例

```python
from ultralytics import YOLO
import lightly_train

if __name__ == "__main__":
    # 加載現有訓練的模型
    model = YOLO("path/to/best.pt")

    lightly_train.train(
        out="out/distill_from_existing",
        data="path/to/images",
        model=model,  # ✅ 傳入模型實例
        method="distillation",
        method_args={"teacher": "dinov3/vitb16"},
    )
```

### 2.3 蒸餾後微調

```python
from ultralytics import YOLO
from pathlib import Path

# 加載蒸餾後的模型
model = YOLO("out/yolo12_dinov3_distill/exported_models/exported_last.pt")

# 在標註數據上微調
data = Path("datasets/yolo_harmony_v2_phase8_final/harmony_phase8_final.yaml").absolute()
model.train(data=str(data), epochs=100, batch=32)
```

---

## 3. 安裝注意事項

### ⚠️ YOLO12 需要特殊安裝

YOLO12 是 Ultralytics 的自定義分支，需要從原始倉庫安裝：

```bash
# 建議在新虛擬環境中執行
pip install git+https://github.com/sunsmarterjie/yolov12

# 如果遇到 FlashAttention 問題
pip install flash-attn --no-build-isolation
```

**警告**: 這會覆蓋現有的 `ultralytics` 安裝！

### 檢查是否需要安裝

```python
# 檢查當前 ultralytics 是否支援 yolo12
from ultralytics import YOLO
try:
    model = YOLO("yolo12s.yaml")
    print("✅ YOLO12 已支援")
except:
    print("❌ 需要安裝 YOLO12 分支")
```

---

## 4. 我們專案的最佳配置

### 4.1 推薦配置

```python
import lightly_train
from pathlib import Path

BASE_DIR = Path('/home/thc1006/dev/music-app/training')
DATASET_DIR = BASE_DIR / 'datasets/yolo_harmony_v2_phase8_final'
OUTPUT_DIR = BASE_DIR / 'harmony_omr_v2_dinov3_distill_v3'
PHASE8_MODEL = BASE_DIR / 'harmony_omr_v2_phase8/phase8_training/weights/best.pt'

if __name__ == "__main__":
    # Step 1: DINOv3 蒸餾預訓練
    lightly_train.train(
        out=str(OUTPUT_DIR / 'pretrain'),
        data=str(DATASET_DIR / 'train/images'),
        model=str(PHASE8_MODEL),  # 從 Phase 8 開始
        method="distillation",
        method_args={
            "teacher": "dinov3/vits16",  # 21.6M 參數, 384 維
        },
        epochs=50,
        batch_size=32,  # RTX 5090 32GB
        num_workers=20,  # i9-14900 24核
    )
```

### 4.2 RTX 5090 優化參數

| 參數 | 值 | 說明 |
|------|-----|------|
| `batch_size` | 32 | 32GB VRAM 最佳值 |
| `num_workers` | 20 | 24核 CPU 預留4核 |
| `precision` | `bf16-mixed` | Blackwell 原生 BF16 |
| `epochs` | 50 | 蒸餾預訓練 |

---

## 5. 可用的蒸餾方法

LightlyTrain 支援多種蒸餾方法：

| 方法 | 說明 |
|------|------|
| `distillation` | 預設蒸餾 (推薦) |
| `distillationv1` | 第一版蒸餾算法 |
| `distillationv2` | 第二版蒸餾算法 |
| `dino` | DINO 自監督 |
| `dinov2` | DINOv2 自監督 |
| `simclr` | SimCLR 對比學習 |

---

## 6. 參考資源

- [LightlyTrain GitHub](https://github.com/lightly-ai/lightly-train)
- [YOLOv12 文檔](https://docs.lightly.ai/train/stable/models/yolov12.html)
- [Ultralytics 整合](https://docs.lightly.ai/train/stable/models/ultralytics.html)
- [YOLO12 原始倉庫](https://github.com/sunsmarterjie/yolov12)

---

**文檔版本**: 1.0
**調研日期**: 2025-12-20
