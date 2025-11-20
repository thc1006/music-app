#!/usr/bin/env python3
"""
YOLO12 模型匯出與 INT8 量化腳本

作者: thc1006 + Claude
日期: 2025-11-20

功能:
1. 載入訓練好的 YOLO12 .pt 模型
2. 匯出為 TFLite FP32 格式
3. 進行 INT8 量化（使用代表性資料集）
4. 驗證量化後的準確度損失
5. 複製到 Android assets 目錄

使用方式:
    python export_models.py --model harmony_omr/yolo12s_XXXXXX/weights/best.pt
    python export_models.py --model harmony_omr/yolo12n_YYYYYY/weights/best.pt
"""

import argparse
import sys
from pathlib import Path
import shutil
import time

try:
    from ultralytics import YOLO
    import tensorflow as tf
    import numpy as np
    from PIL import Image
except ImportError:
    print("錯誤: 請先安裝依賴套件")
    print("執行: pip install -r requirements-train.txt")
    sys.exit(1)


# ============= 配置 =============

IMG_SIZE = 640
NUM_CALIBRATION_IMAGES = 100  # INT8 量化用的代表性資料集大小
ANDROID_ASSETS_DIR = Path('../android-app/app/src/main/assets/models')


# ============= 代表性資料集生成器 =============

def get_representative_dataset(dataset_path: Path, num_images: int = NUM_CALIBRATION_IMAGES):
    """
    生成代表性資料集用於 INT8 量化
    從驗證集隨機抽取圖片
    """
    val_images_dir = dataset_path / 'images' / 'val'

    if not val_images_dir.exists():
        print(f"⚠️  警告: 找不到驗證集目錄: {val_images_dir}")
        print("使用合成隨機資料代替")
        # 生成隨機資料
        for _ in range(num_images):
            random_data = np.random.rand(1, IMG_SIZE, IMG_SIZE, 3).astype(np.float32)
            yield [random_data]
        return

    # 載入真實圖片
    image_files = list(val_images_dir.glob('*.png')) + list(val_images_dir.glob('*.jpg'))

    if len(image_files) == 0:
        print(f"⚠️  警告: 驗證集沒有圖片")
        print("使用合成隨機資料代替")
        for _ in range(num_images):
            random_data = np.random.rand(1, IMG_SIZE, IMG_SIZE, 3).astype(np.float32)
            yield [random_data]
        return

    # 取前 N 張圖片
    selected_images = image_files[:min(num_images, len(image_files))]

    print(f"使用 {len(selected_images)} 張圖片作為代表性資料集")

    for img_path in selected_images:
        try:
            # 載入並預處理圖片
            img = Image.open(img_path).convert('RGB')
            img = img.resize((IMG_SIZE, IMG_SIZE))
            img_array = np.array(img, dtype=np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)  # (1, 640, 640, 3)

            yield [img_array]
        except Exception as e:
            print(f"⚠️  警告: 無法載入 {img_path.name}: {e}")
            continue


# ============= 匯出流程 =============

def export_to_tflite_fp32(model: YOLO, model_path: Path, output_name: str) -> Path:
    """
    步驟 1: 匯出為 TFLite FP32 格式
    """
    print("\n" + "=" * 60)
    print(f"步驟 1: 匯出 {output_name} FP32 TFLite")
    print("=" * 60)

    try:
        # Ultralytics 內建匯出
        model.export(
            format='tflite',
            imgsz=IMG_SIZE,
            int8=False,  # FP32
            half=False,
        )

        # 預期輸出路徑
        exported_path = model_path.parent.parent / f'{model_path.stem}_saved_model'
        tflite_fp32_path = exported_path / f'{model_path.stem}_float32.tflite'

        if not tflite_fp32_path.exists():
            # 嘗試其他可能的路徑
            possible_paths = list(exported_path.glob('*float32.tflite'))
            if possible_paths:
                tflite_fp32_path = possible_paths[0]
            else:
                raise FileNotFoundError(f"找不到匯出的 FP32 模型: {tflite_fp32_path}")

        print(f"✅ FP32 模型匯出成功: {tflite_fp32_path}")

        # 顯示檔案大小
        size_mb = tflite_fp32_path.stat().st_size / 1e6
        print(f"   模型大小: {size_mb:.2f} MB")

        return tflite_fp32_path

    except Exception as e:
        print(f"❌ 錯誤: FP32 匯出失敗")
        print(f"   {e}")
        sys.exit(1)


def quantize_to_int8(
    saved_model_dir: Path,
    output_path: Path,
    dataset_path: Path
) -> Path:
    """
    步驟 2: 將 FP32 模型量化為 INT8
    """
    print("\n" + "=" * 60)
    print("步驟 2: INT8 量化")
    print("=" * 60)

    try:
        # 載入 SavedModel
        converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))

        # 啟用 INT8 量化
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # 提供代表性資料集
        def representative_dataset_gen():
            return get_representative_dataset(dataset_path)

        converter.representative_dataset = representative_dataset_gen

        # 設定量化選項
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
            tf.lite.OpsSet.TFLITE_BUILTINS  # 備援：允許部分算子使用 FP32
        ]

        # 輸入/輸出也量化
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8

        print("開始量化（可能需要幾分鐘）...")
        start_time = time.time()

        # 執行轉換
        tflite_quant_model = converter.convert()

        elapsed = time.time() - start_time
        print(f"量化完成，耗時 {elapsed:.1f} 秒")

        # 儲存
        with open(output_path, 'wb') as f:
            f.write(tflite_quant_model)

        print(f"✅ INT8 模型已儲存: {output_path}")

        # 顯示檔案大小
        size_mb = output_path.stat().st_size / 1e6
        print(f"   模型大小: {size_mb:.2f} MB")

        return output_path

    except Exception as e:
        print(f"❌ 錯誤: INT8 量化失敗")
        print(f"   {e}")
        sys.exit(1)


def validate_tflite_model(tflite_path: Path):
    """
    步驟 3: 驗證 TFLite 模型可用性
    """
    print("\n" + "=" * 60)
    print(f"步驟 3: 驗證 {tflite_path.name}")
    print("=" * 60)

    try:
        # 載入模型
        interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
        interpreter.allocate_tensors()

        # 取得輸入/輸出詳情
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        print(f"✅ 模型載入成功")
        print(f"\n輸入詳情:")
        print(f"  Shape: {input_details[0]['shape']}")
        print(f"  Type:  {input_details[0]['dtype']}")
        print(f"  Name:  {input_details[0]['name']}")

        print(f"\n輸出詳情:")
        print(f"  Shape: {output_details[0]['shape']}")
        print(f"  Type:  {output_details[0]['dtype']}")
        print(f"  Name:  {output_details[0]['name']}")

        # 測試推論（隨機輸入）
        input_shape = input_details[0]['shape']
        input_dtype = input_details[0]['dtype']

        if input_dtype == np.uint8:
            test_input = np.random.randint(0, 256, size=input_shape, dtype=np.uint8)
        else:
            test_input = np.random.rand(*input_shape).astype(np.float32)

        interpreter.set_tensor(input_details[0]['index'], test_input)

        # 執行推論並計時
        start = time.time()
        interpreter.invoke()
        elapsed = (time.time() - start) * 1000  # 轉為毫秒

        output = interpreter.get_tensor(output_details[0]['index'])

        print(f"\n✅ 推論測試成功")
        print(f"  推論時間: {elapsed:.2f} ms (CPU)")
        print(f"  輸出範圍: [{output.min()}, {output.max()}]")
        print(f"  輸出 shape: {output.shape}")

        return True

    except Exception as e:
        print(f"❌ 錯誤: 模型驗證失敗")
        print(f"   {e}")
        return False


def copy_to_android_assets(int8_path: Path, output_name: str):
    """
    步驟 4: 複製到 Android assets 目錄
    """
    print("\n" + "=" * 60)
    print("步驟 4: 複製到 Android 專案")
    print("=" * 60)

    # 確保目標目錄存在
    ANDROID_ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    # 目標檔案路徑
    dst_path = ANDROID_ASSETS_DIR / f"{output_name}_int8.tflite"

    try:
        shutil.copy(int8_path, dst_path)
        print(f"✅ 已複製到: {dst_path}")
        print(f"   大小: {dst_path.stat().st_size / 1e6:.2f} MB")

    except Exception as e:
        print(f"⚠️  警告: 無法複製到 Android 專案")
        print(f"   {e}")
        print(f"\n請手動複製:")
        print(f"  從: {int8_path}")
        print(f"  到: {dst_path}")


# ============= 主流程 =============

def export_model(model_path: Path, dataset_path: Path, output_name: str):
    """
    完整匯出流程
    """
    print("=" * 60)
    print(f"YOLO12 模型匯出 - {output_name}")
    print("=" * 60)
    print(f"模型: {model_path}")
    print(f"資料集: {dataset_path}")
    print()

    # 檢查模型檔案
    if not model_path.exists():
        print(f"❌ 錯誤: 模型檔案不存在: {model_path}")
        sys.exit(1)

    # 載入模型
    print("載入 YOLO12 模型...")
    try:
        model = YOLO(str(model_path))
        print(f"✅ 模型載入成功")
    except Exception as e:
        print(f"❌ 錯誤: 無法載入模型")
        print(f"   {e}")
        sys.exit(1)

    # 步驟 1: 匯出 FP32
    fp32_path = export_to_tflite_fp32(model, model_path, output_name)

    # 步驟 2: INT8 量化
    saved_model_dir = model_path.parent.parent / f'{model_path.stem}_saved_model'
    int8_output_path = model_path.parent / f'{output_name}_int8.tflite'

    int8_path = quantize_to_int8(saved_model_dir, int8_output_path, dataset_path)

    # 顯示壓縮比
    fp32_size = fp32_path.stat().st_size / 1e6
    int8_size = int8_path.stat().st_size / 1e6
    compression_ratio = fp32_size / int8_size

    print("\n" + "=" * 60)
    print("量化效果對比")
    print("=" * 60)
    print(f"FP32 大小: {fp32_size:.2f} MB")
    print(f"INT8 大小: {int8_size:.2f} MB")
    print(f"壓縮比: {compression_ratio:.2f}x")
    print(f"節省空間: {fp32_size - int8_size:.2f} MB ({(1 - int8_size/fp32_size) * 100:.1f}%)")

    # 步驟 3: 驗證
    validate_tflite_model(int8_path)

    # 步驟 4: 複製到 Android
    copy_to_android_assets(int8_path, output_name)

    print("\n" + "=" * 60)
    print("✅ 匯出完成")
    print("=" * 60)
    print(f"INT8 模型: {int8_path}")
    print(f"Android: {ANDROID_ASSETS_DIR / f'{output_name}_int8.tflite'}")
    print()


# ============= 命令列介面 =============

def main():
    parser = argparse.ArgumentParser(
        description='YOLO12 模型匯出與 INT8 量化',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  # 匯出 YOLO12s
  python export_models.py --model harmony_omr/yolo12s_20251120/weights/best.pt

  # 匯出 YOLO12n
  python export_models.py --model harmony_omr/yolo12n_20251120/weights/best.pt \\
                          --output-name yolo12n

  # 指定資料集路徑
  python export_models.py --model best.pt --dataset datasets/yolo_harmony
        """
    )

    parser.add_argument(
        '--model',
        type=Path,
        required=True,
        help='訓練好的 .pt 模型路徑'
    )

    parser.add_argument(
        '--dataset',
        type=Path,
        default=Path('datasets/yolo_harmony'),
        help='資料集路徑（用於量化代表性資料集）'
    )

    parser.add_argument(
        '--output-name',
        type=str,
        help='輸出檔名（預設自動從模型路徑推斷）'
    )

    args = parser.parse_args()

    # 推斷輸出名稱
    if args.output_name is None:
        if 'yolo12s' in str(args.model).lower():
            args.output_name = 'yolo12s'
        elif 'yolo12n' in str(args.model).lower():
            args.output_name = 'yolo12n'
        else:
            args.output_name = 'yolo12_model'

    # 執行匯出
    export_model(args.model, args.dataset, args.output_name)


if __name__ == '__main__':
    main()
