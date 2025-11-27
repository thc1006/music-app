#!/usr/bin/env python3
"""
Phase 6 Ultimate Training Script
合併所有數據並執行終極訓練

數據來源:
1. Phase 5 修復後數據 (yolo_harmony_v2_phase6_fixed) - 24,910 圖片
2. MUSCIMA++ barlines (muscima_barlines_yolo) - 140 圖片
3. 合成 barlines (synthetic_barlines_yolo) - 2,000 圖片
"""

import os
import shutil
import yaml
from pathlib import Path
from datetime import datetime

# 設定路徑
BASE_DIR = Path("/home/thc1006/dev/music-app/training")
DATASETS_DIR = BASE_DIR / "datasets"

# 數據來源
PHASE6_FIXED = DATASETS_DIR / "yolo_harmony_v2_phase6_fixed"
MUSCIMA_BARLINES = DATASETS_DIR / "muscima_barlines_yolo"
SYNTHETIC_BARLINES = DATASETS_DIR / "synthetic_barlines_yolo"

# 輸出目錄
OUTPUT_DIR = DATASETS_DIR / "yolo_harmony_v2_phase6_ultimate"

def merge_datasets():
    """合併所有數據集"""
    print("=" * 70)
    print("🚀 Phase 6 Ultimate - 合併數據集")
    print("=" * 70)

    # 創建輸出目錄
    for split in ['train', 'val']:
        (OUTPUT_DIR / split / 'images').mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / split / 'labels').mkdir(parents=True, exist_ok=True)

    stats = {'train': {'images': 0, 'labels': 0}, 'val': {'images': 0, 'labels': 0}}

    # 1. 符號鏈接 Phase 6 Fixed 數據（主要數據）- 節省空間
    print("\n📦 1. 鏈接 Phase 6 Fixed 數據（使用符號鏈接節省空間）...")
    for split in ['train', 'val']:
        src_img = PHASE6_FIXED / split / 'images'
        src_lbl = PHASE6_FIXED / split / 'labels'

        if src_img.exists():
            for img in src_img.glob('*'):
                dst = OUTPUT_DIR / split / 'images' / img.name
                if not dst.exists():
                    os.symlink(img.resolve(), dst)
                stats[split]['images'] += 1

        if src_lbl.exists():
            for lbl in src_lbl.glob('*.txt'):
                dst = OUTPUT_DIR / split / 'labels' / lbl.name
                if not dst.exists():
                    os.symlink(lbl.resolve(), dst)
                stats[split]['labels'] += 1

    print(f"  ✅ Train: {stats['train']['images']} 圖片, {stats['train']['labels']} 標註")
    print(f"  ✅ Val: {stats['val']['images']} 圖片, {stats['val']['labels']} 標註")

    # 2. 合併 MUSCIMA++ barlines（需要重新映射類別 ID）
    print("\n📦 2. 合併 MUSCIMA++ barlines...")
    muscima_stats = merge_barline_dataset(MUSCIMA_BARLINES, "muscima", stats)

    # 3. 合併合成 barlines
    print("\n📦 3. 合併合成 barlines...")
    synthetic_stats = merge_barline_dataset(SYNTHETIC_BARLINES, "synthetic", stats)

    # 4. 創建 data.yaml
    print("\n📄 4. 創建配置文件...")
    create_data_yaml()

    # 5. 輸出統計
    print("\n" + "=" * 70)
    print("✅ 合併完成!")
    print("=" * 70)
    print(f"\n📊 最終統計:")
    print(f"  訓練集: {stats['train']['images']} 圖片")
    print(f"  驗證集: {stats['val']['images']} 圖片")
    print(f"  總計: {stats['train']['images'] + stats['val']['images']} 圖片")
    print(f"\n📁 輸出目錄: {OUTPUT_DIR}")

    return stats

def merge_barline_dataset(src_dir, prefix, stats):
    """合併 barline 專用數據集，重新映射類別 ID"""

    # barline 數據集的類別 ID (0-3) 需要映射到完整數據集 (23-26)
    class_mapping = {
        '0': '23',  # barline
        '1': '24',  # barline_double
        '2': '25',  # barline_final
        '3': '26',  # barline_repeat
    }

    added = {'train': 0, 'val': 0}

    for split in ['train', 'val']:
        src_img = src_dir / split / 'images'
        src_lbl = src_dir / split / 'labels'

        if not src_img.exists():
            continue

        for img in src_img.glob('*'):
            # 避免文件名衝突
            new_name = f"{prefix}_{img.name}"
            dst = OUTPUT_DIR / split / 'images' / new_name
            if not dst.exists():
                os.symlink(img.resolve(), dst)  # 使用符號鏈接
            stats[split]['images'] += 1
            added[split] += 1

        if src_lbl.exists():
            for lbl in src_lbl.glob('*.txt'):
                new_name = f"{prefix}_{lbl.name}"

                # 讀取並重新映射類別 ID
                with open(lbl, 'r') as f:
                    lines = f.readlines()

                remapped_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        old_class = parts[0]
                        new_class = class_mapping.get(old_class, old_class)
                        parts[0] = new_class
                        remapped_lines.append(' '.join(parts))

                # 標註文件需要寫入（因為類別 ID 變了），但很小
                with open(OUTPUT_DIR / split / 'labels' / new_name, 'w') as f:
                    f.write('\n'.join(remapped_lines))

                stats[split]['labels'] += 1

    print(f"  ✅ 添加: Train {added['train']}, Val {added['val']} 圖片")
    return added

def create_data_yaml():
    """創建 YOLO 訓練配置文件"""

    # 完整的 33 類別
    names = {
        0: 'notehead_filled',
        1: 'notehead_hollow',
        2: 'stem',
        3: 'beam',
        4: 'flag_8th',
        5: 'flag_16th',
        6: 'flag_32nd',
        7: 'augmentation_dot',
        8: 'tie',
        9: 'clef_treble',
        10: 'clef_bass',
        11: 'clef_alto',
        12: 'clef_tenor',
        13: 'accidental_sharp',
        14: 'accidental_flat',
        15: 'accidental_natural',
        16: 'accidental_double_sharp',
        17: 'accidental_double_flat',
        18: 'rest_whole',
        19: 'rest_half',
        20: 'rest_quarter',
        21: 'rest_8th',
        22: 'rest_16th',
        23: 'barline',
        24: 'barline_double',
        25: 'barline_final',
        26: 'barline_repeat',
        27: 'time_signature',
        28: 'key_signature',
        29: 'fermata',
        30: 'dynamic_soft',
        31: 'dynamic_loud',
        32: 'ledger_line',
    }

    config = {
        'path': str(OUTPUT_DIR),
        'train': 'train/images',
        'val': 'val/images',
        'nc': 33,
        'names': names,
    }

    yaml_path = OUTPUT_DIR / 'harmony_phase6_ultimate.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    print(f"  ✅ 配置文件: {yaml_path}")
    return yaml_path

def start_training():
    """啟動 Phase 6 終極訓練"""

    print("\n" + "=" * 70)
    print("🎯 Phase 6 Ultimate Training")
    print("=" * 70)

    # 訓練配置
    config = {
        'data': str(OUTPUT_DIR / 'harmony_phase6_ultimate.yaml'),
        'model': str(BASE_DIR / 'harmony_omr_v2_phase5/fermata_barline_enhanced/weights/best.pt'),
        'epochs': 200,
        'batch': 16,
        'imgsz': 640,
        'device': 0,
        'project': str(BASE_DIR / 'harmony_omr_v2_phase6'),
        'name': 'ultimate_barline_fixed',

        # 優化的損失權重
        'box': 7.5,
        'cls': 2.5,  # 提高分類權重
        'dfl': 1.5,

        # 優化器
        'optimizer': 'AdamW',
        'lr0': 0.001,
        'lrf': 0.01,

        # 數據增強
        'mosaic': 1.0,
        'mixup': 0.1,
        'copy_paste': 0.0,
        'degrees': 0.0,  # OMR 不旋轉
        'flipud': 0.0,
        'fliplr': 0.0,

        # 早停
        'patience': 50,
        'save_period': 10,
    }

    print("\n📋 訓練配置:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # 生成訓練命令
    cmd = f"""
cd {BASE_DIR}
source venv_yolo12/bin/activate

nohup python -c "
from ultralytics import YOLO

model = YOLO('{config['model']}')

results = model.train(
    data='{config['data']}',
    epochs={config['epochs']},
    batch={config['batch']},
    imgsz={config['imgsz']},
    device={config['device']},
    project='{config['project']}',
    name='{config['name']}',
    box={config['box']},
    cls={config['cls']},
    dfl={config['dfl']},
    optimizer='{config['optimizer']}',
    lr0={config['lr0']},
    lrf={config['lrf']},
    mosaic={config['mosaic']},
    mixup={config['mixup']},
    copy_paste={config['copy_paste']},
    degrees={config['degrees']},
    flipud={config['flipud']},
    fliplr={config['fliplr']},
    patience={config['patience']},
    save_period={config['save_period']},
    verbose=True,
)

print('Training completed!')
print(f'Best mAP50: {{results.results_dict.get(\"metrics/mAP50(B)\", \"N/A\")}}')
" > {BASE_DIR}/phase6_training.log 2>&1 &

echo $! > {BASE_DIR}/phase6_training.pid
echo "Training started with PID: $(cat {BASE_DIR}/phase6_training.pid)"
"""

    # 保存訓練腳本
    script_path = BASE_DIR / 'start_phase6_training.sh'
    with open(script_path, 'w') as f:
        f.write(cmd)
    os.chmod(script_path, 0o755)

    print(f"\n📝 訓練腳本已保存: {script_path}")
    print("\n🚀 執行以下命令開始訓練:")
    print(f"   bash {script_path}")

    return script_path

if __name__ == '__main__':
    # 合併數據集
    stats = merge_datasets()

    # 準備訓練腳本
    script_path = start_training()

    print("\n" + "=" * 70)
    print("✅ Phase 6 Ultimate 準備完成!")
    print("=" * 70)
    print(f"\n下一步:")
    print(f"  1. 檢查數據: ls -la {OUTPUT_DIR}")
    print(f"  2. 開始訓練: bash {script_path}")
    print(f"  3. 監控進度: tail -f {BASE_DIR}/phase6_training.log")
