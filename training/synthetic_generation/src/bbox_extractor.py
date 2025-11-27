"""
SVG Bounding Box 提取模組

從 Verovio 生成的 SVG 中提取 barline 元素的精確座標
"""

from typing import List, Tuple, Dict
from lxml import etree
import re


class BboxExtractor:
    """SVG Bounding Box 提取器"""

    def __init__(self):
        """初始化提取器"""
        self.namespaces = {
            'svg': 'http://www.w3.org/2000/svg'
        }

    def extract_barlines(self, svg_content: str,
                        image_width: int,
                        image_height: int) -> List[Dict]:
        """
        從 SVG 提取 barline bounding boxes

        Args:
            svg_content: SVG XML 字符串
            image_width: 圖像寬度
            image_height: 圖像高度

        Returns:
            [{class: str, bbox: (x, y, w, h)}] 列表
        """
        root = etree.fromstring(svg_content.encode('utf-8'))

        barlines = []

        # 查找所有 barline 相關的元素
        # Verovio 使用 class="barLine" 標記
        for element in root.xpath('.//svg:g[contains(@class, "barLine")]',
                                 namespaces=self.namespaces):
            barline_data = self._extract_barline_data(element, image_width, image_height)
            if barline_data:
                barlines.append(barline_data)

        return barlines

    def _extract_barline_data(self, element, img_width: int,
                              img_height: int) -> Dict:
        """
        提取單個 barline 的數據

        Args:
            element: SVG barLine 元素
            img_width: 圖像寬度
            img_height: 圖像高度

        Returns:
            {class: str, bbox: (x, y, w, h), normalized_bbox: (x, y, w, h)}
        """
        # 獲取 barline 類型（從 class 或 data 屬性）
        barline_type = self._get_barline_type(element)

        # 計算 bounding box
        bbox = self._calculate_bbox(element)

        if bbox is None:
            return None

        x, y, w, h = bbox

        # 歸一化座標（YOLO 格式）
        x_center = (x + w / 2) / img_width
        y_center = (y + h / 2) / img_height
        width_norm = w / img_width
        height_norm = h / img_height

        return {
            'class': barline_type,
            'bbox': bbox,
            'normalized_bbox': (x_center, y_center, width_norm, height_norm)
        }

    def _get_barline_type(self, element) -> str:
        """
        識別 barline 類型

        Args:
            element: SVG 元素

        Returns:
            barline 類型字符串
        """
        # 檢查 class 屬性
        class_attr = element.get('class', '')

        # Verovio 的 barline 類型編碼
        if 'barLineRptstart' in class_attr:
            return 'repeat_left'
        elif 'barLineRptend' in class_attr:
            return 'repeat_right'
        elif 'barLineRptboth' in class_attr:
            return 'repeat_both'
        elif 'barLineEnd' in class_attr:
            return 'final'
        elif 'barLineDbl' in class_attr:
            return 'double'
        else:
            return 'single'

    def _calculate_bbox(self, element) -> Tuple[float, float, float, float]:
        """
        計算元素的 bounding box

        Args:
            element: SVG 元素

        Returns:
            (x, y, width, height) 或 None
        """
        # 嘗試使用 getBBox（如果 SVG 有內嵌）
        # 否則手動計算所有子元素的邊界

        # 收集所有路徑和形狀的座標
        points = []

        # 查找所有 path 元素
        for path in element.xpath('.//svg:path', namespaces=self.namespaces):
            d = path.get('d', '')
            path_points = self._parse_path_data(d)
            points.extend(path_points)

        # 查找所有 line 元素
        for line in element.xpath('.//svg:line', namespaces=self.namespaces):
            x1 = float(line.get('x1', 0))
            y1 = float(line.get('y1', 0))
            x2 = float(line.get('x2', 0))
            y2 = float(line.get('y2', 0))
            points.extend([(x1, y1), (x2, y2)])

        # 查找所有 rect 元素
        for rect in element.xpath('.//svg:rect', namespaces=self.namespaces):
            x = float(rect.get('x', 0))
            y = float(rect.get('y', 0))
            w = float(rect.get('width', 0))
            h = float(rect.get('height', 0))
            points.extend([(x, y), (x + w, y + h)])

        if not points:
            return None

        # 計算邊界
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        width = x_max - x_min
        height = y_max - y_min

        return (x_min, y_min, width, height)

    def _parse_path_data(self, path_d: str) -> List[Tuple[float, float]]:
        """
        解析 SVG path d 屬性，提取座標點

        Args:
            path_d: path 的 d 屬性字符串

        Returns:
            [(x, y)] 座標列表
        """
        points = []

        # 簡化版本：提取所有數字對
        # 完整實現需要處理所有 SVG path 命令
        numbers = re.findall(r'-?\d+\.?\d*', path_d)

        # 每兩個數字配對
        for i in range(0, len(numbers) - 1, 2):
            try:
                x = float(numbers[i])
                y = float(numbers[i + 1])
                points.append((x, y))
            except (ValueError, IndexError):
                continue

        return points
