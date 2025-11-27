"""
Domain Randomization 模組

模擬真實世界的紙張紋理、掃描噪聲、幾何變換等
"""

import numpy as np
import cv2
from PIL import Image
import random
from typing import Tuple
from scipy.ndimage import gaussian_filter


class DomainRandomizer:
    """領域隨機化增強器"""

    def __init__(self, config: dict):
        """
        初始化增強器

        Args:
            config: 配置字典
        """
        self.config = config.get('augmentation', {})

    def apply(self, image: Image.Image) -> Image.Image:
        """
        應用所有增強

        Args:
            image: 輸入圖像

        Returns:
            增強後的圖像
        """
        if not self.config.get('enabled', True):
            return image

        # 轉換為 numpy array
        img_array = np.array(image)

        # 1. 紙張紋理
        if self._should_apply('paper_texture'):
            img_array = self._add_paper_texture(img_array)

        # 2. 掃描噪聲
        if self._should_apply('scan_noise'):
            img_array = self._add_scan_noise(img_array)

        # 3. 幾何變換
        if self._should_apply('geometric'):
            img_array = self._apply_geometric_transforms(img_array)

        # 4. 亮度/對比度
        if self._should_apply('brightness_contrast'):
            img_array = self._adjust_brightness_contrast(img_array)

        # 轉換回 PIL Image
        result = Image.fromarray(img_array.astype(np.uint8))

        # 5. JPEG 壓縮
        if self._should_apply('jpeg_compression'):
            result = self._apply_jpeg_compression(result)

        return result

    def _should_apply(self, augmentation_name: str) -> bool:
        """檢查是否應該應用某個增強"""
        aug_config = self.config.get(augmentation_name, {})
        if not aug_config.get('enabled', False):
            return False

        probability = aug_config.get('probability', 1.0)
        return random.random() < probability

    def _add_paper_texture(self, img: np.ndarray) -> np.ndarray:
        """
        添加紙張紋理（Perlin noise）

        Args:
            img: 輸入圖像數組

        Returns:
            添加紋理後的圖像
        """
        config = self.config['paper_texture']

        h, w = img.shape[:2]

        # 生成 Perlin noise（簡化版）
        # 使用多個頻率的疊加
        texture = self._generate_perlin_noise(
            (h, w),
            scale=config.get('scale', 100),
            octaves=config.get('octaves', 3),
            persistence=config.get('persistence', 0.5)
        )

        # 歸一化到 0-1
        texture = (texture - texture.min()) / (texture.max() - texture.min())

        # 調整強度
        intensity = config.get('intensity', 0.15)
        texture = (texture * 2 - 1) * intensity * 255  # -intensity*255 到 +intensity*255

        # 擴展維度以匹配圖像通道
        if len(img.shape) == 3:
            texture = texture[:, :, np.newaxis]

        # 添加到圖像
        result = img.astype(np.float32) + texture
        result = np.clip(result, 0, 255)

        return result.astype(np.uint8)

    def _generate_perlin_noise(self, shape: Tuple[int, int],
                               scale: float, octaves: int,
                               persistence: float) -> np.ndarray:
        """
        生成 Perlin noise（簡化版）

        Args:
            shape: 輸出形狀
            scale: 頻率
            octaves: 疊加層數
            persistence: 振幅衰減

        Returns:
            噪聲數組
        """
        noise = np.zeros(shape)
        amplitude = 1.0
        frequency = 1.0

        for _ in range(octaves):
            # 生成隨機噪聲
            octave_noise = np.random.randn(
                int(shape[0] / scale * frequency) + 1,
                int(shape[1] / scale * frequency) + 1
            )

            # 高斯平滑
            octave_noise = gaussian_filter(octave_noise, sigma=1.0)

            # 調整大小
            octave_noise = cv2.resize(
                octave_noise,
                (shape[1], shape[0]),
                interpolation=cv2.INTER_LINEAR
            )

            noise += octave_noise * amplitude

            amplitude *= persistence
            frequency *= 2

        return noise

    def _add_scan_noise(self, img: np.ndarray) -> np.ndarray:
        """
        添加掃描噪聲（線條和斑點）

        Args:
            img: 輸入圖像

        Returns:
            添加噪聲後的圖像
        """
        config = self.config['scan_noise']
        result = img.copy()

        h, w = img.shape[:2]

        # 添加掃描線
        line_prob = config.get('line_probability', 0.05)
        for y in range(h):
            if random.random() < line_prob:
                # 添加水平掃描線
                intensity = random.randint(-20, 20)
                result[y, :] = np.clip(result[y, :].astype(np.int16) + intensity, 0, 255)

        # 添加隨機斑點
        spot_prob = config.get('spot_probability', 0.02)
        spot_size_range = config.get('spot_size', [1, 3])

        num_spots = int(h * w * spot_prob)
        for _ in range(num_spots):
            x = random.randint(0, w - 1)
            y = random.randint(0, h - 1)
            size = random.randint(*spot_size_range)
            intensity = random.choice([0, 255])  # 黑色或白色斑點

            # 繪製斑點
            y1 = max(0, y - size)
            y2 = min(h, y + size)
            x1 = max(0, x - size)
            x2 = min(w, x + size)

            result[y1:y2, x1:x2] = intensity

        return result

    def _apply_geometric_transforms(self, img: np.ndarray) -> np.ndarray:
        """
        應用幾何變換（旋轉、透視）

        Args:
            img: 輸入圖像

        Returns:
            變換後的圖像
        """
        config = self.config['geometric']
        result = img.copy()

        h, w = img.shape[:2]

        # 旋轉
        rotation_config = config.get('rotation', {})
        if rotation_config.get('probability', 0) > random.random():
            angle_range = rotation_config.get('angle_range', [-3, 3])
            angle = random.uniform(*angle_range)

            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            result = cv2.warpAffine(result, M, (w, h),
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=255)

        # 透視變換
        perspective_config = config.get('perspective', {})
        if perspective_config.get('probability', 0) > random.random():
            distortion = perspective_config.get('distortion', 0.02)

            # 生成隨機透視變換
            src_points = np.float32([
                [0, 0], [w, 0], [w, h], [0, h]
            ])

            dst_points = src_points + np.random.randn(4, 2) * distortion * w
            dst_points = dst_points.astype(np.float32)

            M = cv2.getPerspectiveTransform(src_points, dst_points)
            result = cv2.warpPerspective(result, M, (w, h),
                                        borderMode=cv2.BORDER_CONSTANT,
                                        borderValue=255)

        return result

    def _adjust_brightness_contrast(self, img: np.ndarray) -> np.ndarray:
        """
        調整亮度和對比度

        Args:
            img: 輸入圖像

        Returns:
            調整後的圖像
        """
        config = self.config['brightness_contrast']

        # 亮度調整
        brightness_range = config.get('brightness_range', [-30, 30])
        brightness = random.uniform(*brightness_range)

        # 對比度調整
        contrast_range = config.get('contrast_range', [0.8, 1.2])
        contrast = random.uniform(*contrast_range)

        # 應用調整
        result = img.astype(np.float32)
        result = result * contrast + brightness
        result = np.clip(result, 0, 255)

        return result.astype(np.uint8)

    def _apply_jpeg_compression(self, img: Image.Image) -> Image.Image:
        """
        應用 JPEG 壓縮偽影

        Args:
            img: PIL Image

        Returns:
            壓縮後的圖像
        """
        config = self.config['jpeg_compression']

        quality_range = config.get('quality_range', [70, 95])
        quality = random.randint(*quality_range)

        # 保存到內存中的 JPEG 格式
        from io import BytesIO
        buffer = BytesIO()
        img.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)

        # 重新載入
        result = Image.open(buffer)

        return result
