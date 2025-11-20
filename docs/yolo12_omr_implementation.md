# YOLO12 端側 OMR 完整實作規劃

**文件版本**: 1.0
**建立日期**: 2025-11-20
**作者**: Claude + thc1006
**目標**: 在 Android 手機上完全端側運行樂譜辨識（支援所有等級裝置）

---

## 📋 目錄

1. [執行摘要](#執行摘要)
2. [技術架構](#技術架構)
3. [資料集準備](#資料集準備)
4. [模型訓練](#模型訓練)
5. [模型量化與匯出](#模型量化與匯出)
6. [Android 整合](#android-整合)
7. [符號組裝邏輯](#符號組裝邏輯)
8. [多裝置適配策略](#多裝置適配策略)
9. [測試與驗證](#測試與驗證)
10. [風險與緩解](#風險與緩解)

---

## 執行摘要

### 核心決策

- **模型選擇**: YOLO12s (主) + YOLO12n (備援)
- **訓練硬體**: RTX 5060 GPU
- **目標裝置**: 2025 年所有等級 Android 手機（最低 Android 8.0, 4GB RAM）
- **部署框架**: TensorFlow Lite INT8 量化
- **完全離線**: 無雲端依賴，所有運算在手機端完成

### 預期效能指標

| 裝置等級 | 處理器 | 模型 | 推論時間 | 總時間 |
|---------|--------|------|----------|--------|
| 低階 | SD 6 Gen 1 | YOLO12n INT8 | 0.6-1.3秒 | 2-3秒 |
| 中階 | SD 7 Gen 3 | YOLO12s INT8 | 0.6-1.2秒 | 1.5-2.5秒 |
| 準旗艦 | SD 7+ Gen 3 | YOLO12s INT8 | 0.4-0.7秒 | 1-2秒 |

**目標準確度**: mAP@0.5 > 85%（音符檢測）

### 開發時程

- **Week 1**: 資料準備 + 模型訓練
- **Week 2**: 模型匯出 + Android 基礎整合
- **Week 3**: 符號組裝 + UI 串接
- **Week 4+**: 多裝置測試 + 優化

---

## 技術架構

### 系統架構圖

```
┌─────────────────────────────────────────────────────────────┐
│                     Android Application                      │
├─────────────────────────────────────────────────────────────┤
│  UI Layer (Jetpack Compose)                                 │
│  ├─ CameraScreen: 拍照/選圖                                  │
│  ├─ ProcessingScreen: 辨識進度                               │
│  └─ ResultScreen: 錯誤標記 + 說明                            │
├─────────────────────────────────────────────────────────────┤
│  ViewModel Layer                                             │
│  └─ OmrViewModel: 協調 OMR + 規則引擎                         │
├─────────────────────────────────────────────────────────────┤
│  Domain Layer                                                │
│  ├─ OmrClient (interface)                                    │
│  │   └─ Yolo12OmrClient (impl) ◄── **本文件重點**            │
│  ├─ SymbolAssembler ◄── **本文件重點**                       │
│  └─ HarmonyRuleEngine (已完成)                               │
├─────────────────────────────────────────────────────────────┤
│  Infrastructure Layer                                        │
│  ├─ TensorFlow Lite Interpreter                             │
│  │   ├─ Model: yolo12s_int8.tflite (10MB)                   │
│  │   ├─ Model: yolo12n_int8.tflite (3MB)                    │
│  │   └─ Delegates: NNAPI, GPU, Hexagon                      │
│  └─ Image Processing (Bitmap → Tensor)                      │
└─────────────────────────────────────────────────────────────┘

訓練端（PC with RTX 5060）
┌─────────────────────────────────────────────────────────────┐
│  Training Pipeline                                           │
│  ├─ MUSCIMA++ Dataset (91K symbols)                         │
│  ├─ DeepScoresV2 Dataset (151K symbols)                     │
│  ├─ YOLO12 Training (PyTorch)                               │
│  ├─ Model Export (ONNX → TFLite)                            │
│  └─ INT8 Quantization (4x compression)                      │
└─────────────────────────────────────────────────────────────┘
```

### 資料流詳解

```
1. Input: 樂譜照片 (Bitmap, ~2-4MP)
   ↓
2. Preprocessing:
   - Resize to 640x640
   - RGB → float32 [0-1]
   - Normalize: (pixel - mean) / std
   ↓
3. YOLO12 Inference:
   - Input: float32[1, 640, 640, 3]
   - Output: float32[1, 8400, 84]
     - 8400 = num_anchors
     - 84 = 4 (bbox) + 80 (classes, 我們只用 20)
   ↓
4. NMS (Non-Maximum Suppression):
   - Confidence threshold: 0.25
   - IOU threshold: 0.45
   - Output: List<Detection> (~50-200 objects)
   ↓
5. Symbol Assembly:
   - Sort by Y (top to bottom) → 五線譜分組
   - Sort by X (left to right) → 時間順序
   - Match noteheads + stems + accidentals
   - Generate ChordSnapshot list
   ↓
6. Harmony Analysis:
   - HarmonyRuleEngine.analyze(chords, keySignature)
   - Output: List<HarmonyIssue>
   ↓
7. UI Render:
   - Overlay bounding boxes on original image
   - Display Chinese error messages
```

---

## 資料集準備

### 推薦資料集

#### 1. MUSCIMA++ (優先使用)

```
來源: https://github.com/OMR-Research/muscima-pp
規模: 140 頁樂譜，91,255 個標註符號
格式: XML annotations + PNG images
授權: CC BY-NC-SA 4.0

符號類別（適合四部和聲）:
- noteheadFull (實心符頭)
- noteheadHalf (空心符頭)
- stem (符幹)
- beam (連音線)
- gClef, fClef (高音/低音譜號)
- accidentalSharp, accidentalFlat, accidentalNatural
- timeSignature-*, keySignature-*
- barline, measureSeparator
```

#### 2. DeepScoresV2 (補充資料)

```
來源: https://zenodo.org/record/4012193
規模: 151,286 個符號標註
格式: COCO JSON format
優勢: 合成資料，類別多樣
```

### 資料轉換流程

#### Step 1: 下載資料集

```bash
# 建立資料目錄
mkdir -p training/datasets

cd training/datasets

# 下載 MUSCIMA++
git clone https://github.com/OMR-Research/muscima-pp.git

# 下載 DeepScoresV2
# (需手動從 Zenodo 下載，約 2GB)
```

#### Step 2: 轉換為 YOLO 格式

建立 `training/convert_dataset.py`:

```python
"""
將 MUSCIMA++ XML 標註轉換為 YOLO txt 格式
"""
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple

# 定義我們需要的 20 個類別（針對四部和聲）
HARMONY_CLASSES = [
    # 0-5: 音符符號
    "notehead_filled",
    "notehead_hollow",
    "stem_up",
    "stem_down",
    "beam",
    "flag",

    # 6-9: 譜號
    "clef_treble",
    "clef_bass",
    "clef_alto",
    "clef_tenor",

    # 10-12: 變音記號
    "accidental_sharp",
    "accidental_flat",
    "accidental_natural",

    # 13-15: 節奏/小節
    "rest_quarter",
    "rest_half",
    "rest_whole",
    "barline",

    # 16-19: 調號拍號
    "time_signature",
    "key_signature",
    "staffline"
]

CLASS_TO_IDX = {c: i for i, c in enumerate(HARMONY_CLASSES)}


def parse_muscima_xml(xml_path: Path) -> List[Tuple[str, List[int]]]:
    """
    解析 MUSCIMA++ XML 檔案
    返回: [(class_name, [x, y, width, height]), ...]
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    annotations = []
    for node in root.findall('.//Node'):
        class_name = node.find('ClassName').text

        # 映射 MUSCIMA 類別到我們的類別
        mapped_class = map_muscima_class(class_name)
        if mapped_class is None:
            continue

        # 取得 bounding box
        top = int(node.find('Top').text)
        left = int(node.find('Left').text)
        width = int(node.find('Width').text)
        height = int(node.find('Height').text)

        annotations.append((mapped_class, [left, top, width, height]))

    return annotations


def map_muscima_class(muscima_class: str) -> str | None:
    """將 MUSCIMA++ 類別映射到我們的類別"""
    mapping = {
        'noteheadFull': 'notehead_filled',
        'noteheadHalf': 'notehead_hollow',
        'noteheadWhole': 'notehead_hollow',
        'stem': 'stem_up',  # 後續再判斷方向
        'beam': 'beam',
        'g-clef': 'clef_treble',
        'f-clef': 'clef_bass',
        'c-clef': 'clef_alto',
        'sharp': 'accidental_sharp',
        'flat': 'accidental_flat',
        'natural': 'accidental_natural',
        'rest-quarter': 'rest_quarter',
        'rest-half': 'rest_half',
        'rest-whole': 'rest_whole',
        'barline': 'barline',
        'timeSignature': 'time_signature',
        'keySignature': 'key_signature',
        'staffLine': 'staffline',
    }

    return mapping.get(muscima_class)


def convert_to_yolo_format(
    annotations: List[Tuple[str, List[int]]],
    img_width: int,
    img_height: int
) -> List[str]:
    """
    轉換為 YOLO 格式
    格式: <class_id> <x_center> <y_center> <width> <height>
    所有值正規化到 [0, 1]
    """
    yolo_lines = []

    for class_name, (x, y, w, h) in annotations:
        class_id = CLASS_TO_IDX[class_name]

        # 轉換為中心點座標並正規化
        x_center = (x + w / 2) / img_width
        y_center = (y + h / 2) / img_height
        norm_width = w / img_width
        norm_height = h / img_height

        yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}"
        yolo_lines.append(yolo_line)

    return yolo_lines


def main():
    """主轉換流程"""
    muscima_dir = Path('datasets/muscima-pp')
    output_dir = Path('datasets/yolo_harmony')

    # 建立輸出目錄
    (output_dir / 'images' / 'train').mkdir(parents=True, exist_ok=True)
    (output_dir / 'images' / 'val').mkdir(parents=True, exist_ok=True)
    (output_dir / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
    (output_dir / 'labels' / 'val').mkdir(parents=True, exist_ok=True)

    # 處理所有樣本...
    # (實作省略，需遍歷所有 XML + 圖片)

    print(f"轉換完成！")
    print(f"訓練集: {len(train_images)} 張")
    print(f"驗證集: {len(val_images)} 張")


if __name__ == '__main__':
    main()
```

### 資料增強策略

```python
# 在 yolo12_train.py 中配置
augmentations = dict(
    # 幾何變換
    degrees=5.0,          # 旋轉 ±5 度（模擬拍照角度）
    translate=0.1,        # 平移 10%
    scale=0.2,            # 縮放 ±20%
    shear=2.0,           # 剪切變換
    perspective=0.0001,  # 透視變換（輕微）

    # 顏色調整
    hsv_h=0.015,         # 色調調整
    hsv_s=0.5,           # 飽和度調整
    hsv_v=0.4,           # 亮度調整

    # 模糊與雜訊
    blur=0.001,          # 輕微模糊（模擬對焦不準）

    # 翻轉（不啟用，樂譜不應該翻轉）
    fliplr=0.0,
    flipud=0.0,

    # Mosaic 增強（YOLO 特色）
    mosaic=0.5,          # 50% 機率使用 mosaic
    mixup=0.1,           # 10% 機率使用 mixup
)
```

---

## 模型訓練

### 訓練環境設置

建立 `training/requirements-train.txt`:

```
# 深度學習框架
torch>=2.1.0
torchvision>=0.16.0

# Ultralytics YOLO12
ultralytics>=8.3.0

# 資料處理
numpy>=1.24.0
opencv-python>=4.8.0
Pillow>=10.0.0
pandas>=2.0.0

# 視覺化
matplotlib>=3.7.0
seaborn>=0.12.0
tensorboard>=2.14.0

# 工具
tqdm>=4.65.0
PyYAML>=6.0
scikit-learn>=1.3.0

# TFLite 轉換
tensorflow>=2.14.0
onnx>=1.15.0
onnxruntime>=1.16.0
```

安裝:

```bash
cd training
pip install -r requirements-train.txt
```

### YOLO12s 訓練腳本

建立 `training/yolo12_train.py`:

```python
"""
YOLO12 四部和聲樂譜辨識訓練腳本
硬體需求: RTX 5060 (8GB VRAM)
預估時間: 200 epochs × 8-10 hours = 約 2 天
"""
from ultralytics import YOLO
from pathlib import Path
import torch
import yaml
from datetime import datetime

# ============= 配置區 =============

# 硬體配置
DEVICE = 0  # GPU 0 (RTX 5060)
WORKERS = 8  # 資料載入線程數

# 訓練超參數
BATCH_SIZE = 16  # RTX 5060 8GB 可用 batch size
IMG_SIZE = 640   # YOLO 標準輸入尺寸
EPOCHS = 250     # YOLO12 需要更長訓練時間
PATIENCE = 50    # Early stopping patience

# 學習率策略
LR0 = 0.01       # 初始學習率
LRF = 0.01       # 最終學習率（線性衰減）

# 模型選擇
MODEL_VARIANT = 'yolo12s'  # 或 'yolo12n' 用於備援

# 路徑配置
DATASET_YAML = 'omr_harmony.yaml'
PROJECT_NAME = 'harmony_omr'
RUN_NAME = f'{MODEL_VARIANT}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'

# ============= 訓練流程 =============

def check_environment():
    """檢查訓練環境"""
    print("=== 環境檢查 ===")
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU 名稱: {torch.cuda.get_device_name(0)}")
        print(f"GPU 記憶體: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print()


def load_dataset_config():
    """載入並驗證資料集配置"""
    with open(DATASET_YAML, 'r') as f:
        config = yaml.safe_load(f)

    print("=== 資料集配置 ===")
    print(f"訓練集: {config['train']}")
    print(f"驗證集: {config['val']}")
    print(f"類別數: {config['nc']}")
    print(f"類別名稱: {config['names'][:5]}... (共 {len(config['names'])} 類)")
    print()

    return config


def train_yolo12():
    """訓練 YOLO12 模型"""
    print(f"=== 開始訓練 {MODEL_VARIANT.upper()} ===")
    print(f"專案: {PROJECT_NAME}")
    print(f"執行: {RUN_NAME}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print()

    # 載入預訓練模型
    model = YOLO(f'{MODEL_VARIANT}.pt')

    # 開始訓練
    results = model.train(
        # 資料配置
        data=DATASET_YAML,

        # 訓練超參數
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,

        # 學習率
        lr0=LR0,
        lrf=LRF,

        # 優化器
        optimizer='AdamW',  # YOLO12 推薦
        weight_decay=0.0005,

        # Early stopping
        patience=PATIENCE,

        # 資料增強（參考前面的配置）
        degrees=5.0,
        translate=0.1,
        scale=0.2,
        shear=2.0,
        perspective=0.0001,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,
        blur=0.001,
        mosaic=0.5,
        mixup=0.1,

        # 硬體配置
        device=DEVICE,
        workers=WORKERS,

        # 輸出配置
        project=PROJECT_NAME,
        name=RUN_NAME,
        exist_ok=False,

        # 儲存設定
        save=True,
        save_period=10,  # 每 10 epochs 儲存一次

        # 驗證設定
        val=True,

        # 其他
        verbose=True,
        seed=42,
    )

    print("\n=== 訓練完成 ===")
    print(f"最佳模型: {PROJECT_NAME}/{RUN_NAME}/weights/best.pt")
    print(f"最終 mAP@0.5: {results.results_dict['metrics/mAP50(B)']:.4f}")
    print(f"最終 mAP@0.5:0.95: {results.results_dict['metrics/mAP50-95(B)']:.4f}")

    return results


def validate_model():
    """驗證最佳模型"""
    print("\n=== 模型驗證 ===")

    best_model_path = f"{PROJECT_NAME}/{RUN_NAME}/weights/best.pt"
    model = YOLO(best_model_path)

    # 在驗證集上評估
    metrics = model.val(
        data=DATASET_YAML,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
    )

    print(f"驗證 mAP@0.5: {metrics.box.map50:.4f}")
    print(f"驗證 mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"驗證 Precision: {metrics.box.mp:.4f}")
    print(f"驗證 Recall: {metrics.box.mr:.4f}")

    return metrics


def main():
    """主訓練流程"""
    check_environment()
    load_dataset_config()

    # 訓練
    results = train_yolo12()

    # 驗證
    metrics = validate_model()

    print("\n=== 全部完成 ===")
    print("下一步: 執行 export_models.py 進行模型匯出與量化")


if __name__ == '__main__':
    main()
```

### 資料集配置檔案

建立 `training/omr_harmony.yaml`:

```yaml
# YOLO12 四部和聲資料集配置

# 路徑（相對於此 yaml 檔案）
path: ./datasets/yolo_harmony
train: images/train
val: images/val
test: images/test  # 可選

# 類別數量
nc: 20

# 類別名稱（索引對應 convert_dataset.py 中的 HARMONY_CLASSES）
names:
  0: notehead_filled
  1: notehead_hollow
  2: stem_up
  3: stem_down
  4: beam
  5: flag
  6: clef_treble
  7: clef_bass
  8: clef_alto
  9: clef_tenor
  10: accidental_sharp
  11: accidental_flat
  12: accidental_natural
  13: rest_quarter
  14: rest_half
  15: rest_whole
  16: barline
  17: time_signature
  18: key_signature
  19: staffline
```

### 並行訓練 YOLO12n（備援）

建立 `training/train_both.sh`:

```bash
#!/bin/bash
# 並行訓練 YOLO12s 和 YOLO12n

echo "開始並行訓練 YOLO12s 和 YOLO12n..."

# 訓練 YOLO12s（主模型）
python yolo12_train.py --model yolo12s &
PID_S=$!

# 等待 1 小時，讓 YOLO12s 先使用完整 GPU
sleep 3600

# 訓練 YOLO12n（備援，batch size 更大）
python yolo12_train.py --model yolo12n --batch 24 &
PID_N=$!

# 等待兩個訓練都完成
wait $PID_S
wait $PID_N

echo "兩個模型訓練完成！"
```

---

## 模型量化與匯出

### TFLite INT8 量化流程

建立 `training/export_models.py`:

```python
"""
YOLO12 模型匯出與 INT8 量化
輸出: yolo12s_int8.tflite, yolo12n_int8.tflite
"""
from ultralytics import YOLO
from pathlib import Path
import tensorflow as tf
import numpy as np
from PIL import Image

# ============= 配置 =============

MODELS_TO_EXPORT = [
    'harmony_omr/yolo12s_20251120_XXXXXX/weights/best.pt',  # 替換為實際路徑
    'harmony_omr/yolo12n_20251120_YYYYYY/weights/best.pt',
]

OUTPUT_DIR = Path('../android-app/app/src/main/assets/models')
IMG_SIZE = 640

# ============= 量化用代表性資料集 =============

def representative_dataset_gen():
    """
    提供代表性資料集用於 INT8 量化
    從驗證集隨機抽取 100 張圖片
    """
    dataset_root = Path('datasets/yolo_harmony/images/val')
    image_files = list(dataset_root.glob('*.png'))[:100]

    for img_path in image_files:
        # 載入並預處理圖片
        img = Image.open(img_path).convert('RGB')
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # (1, 640, 640, 3)

        yield [img_array]


# ============= 匯出流程 =============

def export_to_tflite_int8(model_path: str, output_name: str):
    """匯出為 TFLite INT8 量化模型"""
    print(f"\n=== 匯出 {model_path} ===")

    # 載入訓練好的模型
    model = YOLO(model_path)

    # Step 1: 匯出為 TFLite (FP32)
    print("Step 1: 匯出 FP32 TFLite...")
    model.export(
        format='tflite',
        imgsz=IMG_SIZE,
        int8=False,  # 先不量化
    )

    fp32_path = model_path.replace('.pt', '_saved_model/best_float32.tflite')

    # Step 2: 轉換為 INT8
    print("Step 2: INT8 量化...")
    converter = tf.lite.TFLiteConverter.from_saved_model(
        model_path.replace('.pt', '_saved_model')
    )

    # 啟用 INT8 量化
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8  # 輸入也量化
    converter.inference_output_type = tf.uint8  # 輸出也量化

    tflite_quant_model = converter.convert()

    # 儲存
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{output_name}_int8.tflite"

    with open(output_path, 'wb') as f:
        f.write(tflite_quant_model)

    # 比較檔案大小
    import os
    fp32_size = os.path.getsize(fp32_path) / 1e6
    int8_size = os.path.getsize(output_path) / 1e6
    compression_ratio = fp32_size / int8_size

    print(f"✅ 匯出完成: {output_path}")
    print(f"   FP32 大小: {fp32_size:.2f} MB")
    print(f"   INT8 大小: {int8_size:.2f} MB")
    print(f"   壓縮比: {compression_ratio:.2f}x")

    return output_path


def validate_tflite_model(tflite_path: Path):
    """驗證 TFLite 模型可以正常推論"""
    print(f"\n=== 驗證 {tflite_path.name} ===")

    # 載入模型
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()

    # 取得輸入/輸出詳情
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"輸入 shape: {input_details[0]['shape']}")
    print(f"輸入 dtype: {input_details[0]['dtype']}")
    print(f"輸出 shape: {output_details[0]['shape']}")
    print(f"輸出 dtype: {output_details[0]['dtype']}")

    # 測試推論（隨機輸入）
    test_input = np.random.randint(0, 256, size=input_details[0]['shape'], dtype=np.uint8)
    interpreter.set_tensor(input_details[0]['index'], test_input)

    import time
    start = time.time()
    interpreter.invoke()
    end = time.time()

    output = interpreter.get_tensor(output_details[0]['index'])

    print(f"✅ 推論成功")
    print(f"   推論時間: {(end - start) * 1000:.2f} ms (CPU)")
    print(f"   輸出範圍: [{output.min()}, {output.max()}]")


def main():
    """主匯出流程"""
    print("=== YOLO12 模型匯出與量化 ===\n")

    exported_models = []

    # 匯出 YOLO12s
    if len(MODELS_TO_EXPORT) > 0:
        path = export_to_tflite_int8(MODELS_TO_EXPORT[0], 'yolo12s')
        validate_tflite_model(path)
        exported_models.append(path)

    # 匯出 YOLO12n
    if len(MODELS_TO_EXPORT) > 1:
        path = export_to_tflite_int8(MODELS_TO_EXPORT[1], 'yolo12n')
        validate_tflite_model(path)
        exported_models.append(path)

    print("\n=== 全部完成 ===")
    print(f"已匯出 {len(exported_models)} 個模型:")
    for p in exported_models:
        print(f"  - {p}")
    print("\n下一步: 將 .tflite 檔案複製到 android-app/app/src/main/assets/models/")


if __name__ == '__main__':
    main()
```

---

## Android 整合

### 更新 build.gradle.kts

將 TensorFlow Lite 依賴加入 `android-app/app/build.gradle.kts`:

```kotlin
dependencies {
    // ... 現有依賴 ...

    // ========== TensorFlow Lite (YOLO12 推論) ==========
    implementation("org.tensorflow:tensorflow-lite:2.14.0")
    implementation("org.tensorflow:tensorflow-lite-support:0.4.4")
    implementation("org.tensorflow:tensorflow-lite-gpu:2.14.0")  // GPU 加速
    implementation("org.tensorflow:tensorflow-lite-task-vision:0.4.4")  // 視覺任務工具

    // NNAPI Delegate (NPU 加速)
    implementation("org.tensorflow:tensorflow-lite-select-tf-ops:2.14.0")
}
```

### Yolo12OmrClient.kt 實作

建立 `android-app/app/src/main/java/com/example/harmonychecker/core/omr/Yolo12OmrClient.kt`:

```kotlin
package com.example.harmonychecker.core.omr

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import java.nio.ByteBuffer
import kotlin.math.exp

/**
 * YOLO12 TFLite 推論客戶端
 *
 * 負責：
 * 1. 載入 TFLite 模型（INT8 量化）
 * 2. 圖像預處理
 * 3. 執行推論
 * 4. NMS 後處理
 * 5. 輸出符號檢測結果
 */
class Yolo12OmrClient(
    private val context: Context,
    private val modelVariant: ModelVariant = ModelVariant.YOLO12S,
    private val useGpuDelegate: Boolean = true
) : OmrClient {

    // 模型配置
    private val inputSize = 640
    private val numClasses = 20
    private val confidenceThreshold = 0.25f
    private val iouThreshold = 0.45f

    // TFLite Interpreter
    private var interpreter: Interpreter? = null
    private var gpuDelegate: GpuDelegate? = null

    // 類別名稱
    private val classNames = listOf(
        "notehead_filled", "notehead_hollow",
        "stem_up", "stem_down", "beam", "flag",
        "clef_treble", "clef_bass", "clef_alto", "clef_tenor",
        "accidental_sharp", "accidental_flat", "accidental_natural",
        "rest_quarter", "rest_half", "rest_whole",
        "barline", "time_signature", "key_signature", "staffline"
    )

    init {
        loadModel()
    }

    /**
     * 載入 TFLite 模型
     */
    private fun loadModel() {
        val modelPath = when (modelVariant) {
            ModelVariant.YOLO12S -> "models/yolo12s_int8.tflite"
            ModelVariant.YOLO12N -> "models/yolo12n_int8.tflite"
        }

        val options = Interpreter.Options().apply {
            setNumThreads(4)  // 使用 4 個 CPU 執行緒

            if (useGpuDelegate) {
                try {
                    gpuDelegate = GpuDelegate()
                    addDelegate(gpuDelegate)
                } catch (e: Exception) {
                    android.util.Log.w("Yolo12Client", "GPU delegate 初始化失敗，降級到 CPU", e)
                }
            }
        }

        val modelBuffer = FileUtil.loadMappedFile(context, modelPath)
        interpreter = Interpreter(modelBuffer, options)

        android.util.Log.i("Yolo12Client", "模型載入完成: $modelPath")
    }

    /**
     * 實作 OmrClient 介面
     */
    override suspend fun recognizeScore(image: Bitmap): OmrResult {
        val detections = detect(image)

        // 組裝成樂譜結構（由 SymbolAssembler 處理）
        val assembler = SymbolAssembler()
        val chordSnapshots = assembler.assemble(detections, image.width, image.height)

        return OmrResult(
            chords = chordSnapshots,
            keySignature = assembler.detectedKeySignature,
            timeSignature = assembler.detectedTimeSignature,
            raw = detections
        )
    }

    /**
     * YOLO12 推論核心邏輯
     */
    private fun detect(bitmap: Bitmap): List<Detection> {
        // Step 1: 圖像預處理
        val inputTensor = preprocessImage(bitmap)

        // Step 2: 執行推論
        val outputTensor = runInference(inputTensor)

        // Step 3: 後處理（NMS）
        val detections = postprocess(outputTensor, bitmap.width, bitmap.height)

        return detections
    }

    /**
     * 圖像預處理
     */
    private fun preprocessImage(bitmap: Bitmap): ByteBuffer {
        // Resize to 640x640
        val resized = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)

        // 轉換為 ByteBuffer（INT8 輸入）
        val byteBuffer = ByteBuffer.allocateDirect(1 * inputSize * inputSize * 3)
        byteBuffer.rewind()

        val intValues = IntArray(inputSize * inputSize)
        resized.getPixels(intValues, 0, inputSize, 0, 0, inputSize, inputSize)

        for (pixelValue in intValues) {
            // 提取 RGB 並轉為 uint8 [0-255]
            byteBuffer.put(((pixelValue shr 16) and 0xFF).toByte())  // R
            byteBuffer.put(((pixelValue shr 8) and 0xFF).toByte())   // G
            byteBuffer.put((pixelValue and 0xFF).toByte())           // B
        }

        return byteBuffer
    }

    /**
     * 執行推論
     */
    private fun runInference(input: ByteBuffer): Array<Array<ByteArray>> {
        // YOLO12 輸出: [1, 8400, 84] (INT8)
        // 8400 = num_anchors, 84 = 4 (bbox) + 80 (classes)
        val output = Array(1) { Array(8400) { ByteArray(numClasses + 4) } }

        interpreter?.run(input, output)

        return output
    }

    /**
     * 後處理：NMS + 座標還原
     */
    private fun postprocess(
        output: Array<Array<ByteArray>>,
        originalWidth: Int,
        originalHeight: Int
    ): List<Detection> {
        val detections = mutableListOf<Detection>()

        // 解析 YOLO 輸出
        for (i in 0 until 8400) {
            val row = output[0][i]

            // 解量化（INT8 → FP32）
            val dequantized = dequantize(row)

            // 取得 bbox 和信心度
            val centerX = dequantized[0]
            val centerY = dequantized[1]
            val width = dequantized[2]
            val height = dequantized[3]

            // 取得最高信心度的類別
            val classScores = dequantized.sliceArray(4 until dequantized.size)
            val maxScore = classScores.maxOrNull() ?: 0f
            val classId = classScores.indexOf(maxScore)

            if (maxScore < confidenceThreshold) continue

            // 轉換座標（從 640x640 還原到原始尺寸）
            val scaleX = originalWidth.toFloat() / inputSize
            val scaleY = originalHeight.toFloat() / inputSize

            val bbox = RectF(
                (centerX - width / 2) * scaleX,
                (centerY - height / 2) * scaleY,
                (centerX + width / 2) * scaleX,
                (centerY + height / 2) * scaleY
            )

            detections.add(Detection(
                bbox = bbox,
                classId = classId,
                className = classNames[classId],
                confidence = maxScore
            ))
        }

        // NMS
        return nms(detections, iouThreshold)
    }

    /**
     * 反量化（INT8 → FP32）
     */
    private fun dequantize(quantized: ByteArray): FloatArray {
        // 簡化版：假設 scale = 1/128, zero_point = 128
        return quantized.map { (it.toInt() and 0xFF) / 128f - 1f }.toFloatArray()
    }

    /**
     * Non-Maximum Suppression
     */
    private fun nms(detections: List<Detection>, iouThreshold: Float): List<Detection> {
        val sorted = detections.sortedByDescending { it.confidence }
        val selected = mutableListOf<Detection>()
        val suppressed = BooleanArray(sorted.size) { false }

        for (i in sorted.indices) {
            if (suppressed[i]) continue

            selected.add(sorted[i])

            for (j in i + 1 until sorted.size) {
                if (suppressed[j]) continue
                if (calculateIoU(sorted[i].bbox, sorted[j].bbox) > iouThreshold) {
                    suppressed[j] = true
                }
            }
        }

        return selected
    }

    /**
     * 計算 IoU (Intersection over Union)
     */
    private fun calculateIoU(box1: RectF, box2: RectF): Float {
        val intersection = RectF(box1)
        if (!intersection.intersect(box2)) return 0f

        val intersectionArea = intersection.width() * intersection.height()
        val box1Area = box1.width() * box1.height()
        val box2Area = box2.width() * box2.height()
        val unionArea = box1Area + box2Area - intersectionArea

        return intersectionArea / unionArea
    }

    /**
     * 清理資源
     */
    fun close() {
        interpreter?.close()
        gpuDelegate?.close()
    }

    enum class ModelVariant {
        YOLO12S,  // 高準確度，10MB
        YOLO12N   // 輕量級，3MB
    }
}

/**
 * 單一符號檢測結果
 */
data class Detection(
    val bbox: RectF,
    val classId: Int,
    val className: String,
    val confidence: Float
)

/**
 * OMR 辨識結果
 */
data class OmrResult(
    val chords: List<ChordSnapshot>,
    val keySignature: KeySignature?,
    val timeSignature: TimeSignature?,
    val raw: List<Detection>  // 原始檢測結果（用於 debug）
)
```

---

## 符號組裝邏輯

*（繼續撰寫 Symbol Assembler、多裝置適配、測試驗證等章節...由於字數限制，我先創建這個檔案的第一部分）*

---

**未完待續**：本文檔將持續更新，包含：
- Section 7: 符號組裝邏輯（SymbolAssembler.kt 完整實作）
- Section 8: 多裝置適配策略（動態模型選擇）
- Section 9: 測試與驗證
- Section 10: 風險與緩解

**當前版本**: 1.0-draft
**最後更新**: 2025-11-20
