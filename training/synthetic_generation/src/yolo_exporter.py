"""
YOLO 格式輸出模組

將標註轉換為 YOLO 格式並管理輸出文件
"""

import os
from typing import List, Dict
from pathlib import Path


class YOLOExporter:
    """YOLO 格式輸出管理器"""

    def __init__(self, config: Dict):
        """
        初始化輸出器

        Args:
            config: 配置字典
        """
        self.config = config
        self.class_mapping = config.get('yolo_classes', {})

    def export_annotation(self, barlines: List[Dict],
                         output_path: str):
        """
        導出 YOLO 格式標註文件

        Args:
            barlines: barline 數據列表
            output_path: 輸出標註文件路徑
        """
        # 確保輸出目錄存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        lines = []
        for barline in barlines:
            line = self._format_yolo_line(barline)
            if line:
                lines.append(line)

        # 寫入文件
        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))

    def _format_yolo_line(self, barline: Dict) -> str:
        """
        格式化單個 YOLO 標註行

        Args:
            barline: barline 數據

        Returns:
            YOLO 格式字符串
        """
        # 獲取類別 ID
        barline_type = barline['class']
        class_id = self.class_mapping.get(barline_type)

        if class_id is None:
            return None

        # 獲取歸一化座標
        x_center, y_center, width, height = barline['normalized_bbox']

        # YOLO 格式：class_id x_center y_center width height
        return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

    def organize_dataset(self, images_dir: str, labels_dir: str,
                        output_dir: str, split_ratio: Dict[str, float]):
        """
        組織數據集為訓練/驗證集

        Args:
            images_dir: 圖像目錄
            labels_dir: 標註目錄
            output_dir: 輸出根目錄
            split_ratio: {'train': 0.9, 'val': 0.1}
        """
        import shutil
        import random

        # 獲取所有圖像文件
        image_files = list(Path(images_dir).glob('*.png'))
        random.shuffle(image_files)

        # 計算分割點
        train_ratio = split_ratio.get('train', 0.9)
        split_point = int(len(image_files) * train_ratio)

        train_images = image_files[:split_point]
        val_images = image_files[split_point:]

        # 創建目錄結構
        for split in ['train', 'val']:
            os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
            os.makedirs(os.path.join(output_dir, split, 'labels'), exist_ok=True)

        # 複製文件
        self._copy_split(train_images, labels_dir, output_dir, 'train')
        self._copy_split(val_images, labels_dir, output_dir, 'val')

        # 生成統計報告
        stats = {
            'train': len(train_images),
            'val': len(val_images),
            'total': len(image_files)
        }

        return stats

    def _copy_split(self, image_files: List[Path], labels_dir: str,
                   output_dir: str, split: str):
        """
        複製文件到指定分割

        Args:
            image_files: 圖像文件列表
            labels_dir: 標註目錄
            output_dir: 輸出根目錄
            split: 'train' 或 'val'
        """
        import shutil

        for img_path in image_files:
            # 複製圖像
            dest_img = os.path.join(output_dir, split, 'images', img_path.name)
            shutil.copy2(img_path, dest_img)

            # 複製對應標註
            label_name = img_path.stem + '.txt'
            label_path = os.path.join(labels_dir, label_name)

            if os.path.exists(label_path):
                dest_label = os.path.join(output_dir, split, 'labels', label_name)
                shutil.copy2(label_path, dest_label)

    def create_dataset_yaml(self, output_dir: str, dataset_name: str):
        """
        創建 YOLO 數據集配置文件

        Args:
            output_dir: 輸出目錄
            dataset_name: 數據集名稱
        """
        # 反轉類別映射以獲取名稱列表
        num_classes = len(self.class_mapping)
        class_names = [''] * num_classes

        for name, idx in self.class_mapping.items():
            class_names[idx] = name

        yaml_content = f"""# Verovio 合成 Barline 數據集
# 生成時間: {self._get_timestamp()}

path: {os.path.abspath(output_dir)}
train: train/images
val: val/images

# 類別數量
nc: {num_classes}

# 類別名稱
names:
"""
        for idx, name in enumerate(class_names):
            yaml_content += f"  {idx}: {name}\n"

        # 寫入文件
        yaml_path = os.path.join(output_dir, f'{dataset_name}.yaml')
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)

        return yaml_path

    def _get_timestamp(self) -> str:
        """獲取當前時間戳"""
        from datetime import datetime
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')
