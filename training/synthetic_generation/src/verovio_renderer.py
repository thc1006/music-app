"""
Verovio 渲染核心模組

負責將 MEI 格式轉換為 SVG 和 PNG
"""

import os
from typing import Dict, Tuple, Optional
import verovio
from PIL import Image
import io


class VerovioRenderer:
    """Verovio 渲染引擎"""

    def __init__(self, config: Dict):
        """
        初始化渲染器

        Args:
            config: 配置字典
        """
        self.config = config
        self.tk = verovio.toolkit()
        self._setup_verovio()

    def _setup_verovio(self):
        """配置 Verovio 參數"""
        verovio_config = self.config.get('verovio', {})

        # 設置基本參數
        self.tk.setOptions({
            'scale': verovio_config.get('scale', 100),
            'font': verovio_config.get('font', 'Bravura'),
            'pageWidth': verovio_config.get('page_width', 2100),
            'pageHeight': verovio_config.get('page_height', 2970),
            'pageMarginTop': verovio_config.get('page_margin_top', 100),
            'pageMarginBottom': verovio_config.get('page_margin_bottom', 100),
            'pageMarginLeft': verovio_config.get('page_margin_left', 100),
            'pageMarginRight': verovio_config.get('page_margin_right', 100),
            'spacingStaff': verovio_config.get('spacing_staff', 12),
            'spacingSystem': verovio_config.get('spacing_system', 12),
        })

    def render_mei_to_svg(self, mei_content: str) -> str:
        """
        將 MEI 內容渲染為 SVG

        Args:
            mei_content: MEI XML 字符串

        Returns:
            SVG 字符串
        """
        # 載入 MEI
        self.tk.loadData(mei_content)

        # 渲染為 SVG
        svg = self.tk.renderToSVG(1)  # 渲染第一頁

        return svg

    def render_mei_to_png(self, mei_content: str,
                         output_path: Optional[str] = None) -> Image.Image:
        """
        將 MEI 內容渲染為 PNG

        Args:
            mei_content: MEI XML 字符串
            output_path: 輸出路徑（可選）

        Returns:
            PIL Image 對象
        """
        # 先渲染為 SVG
        svg = self.render_mei_to_svg(mei_content)

        # 使用 cairosvg 轉換為 PNG（如果可用）
        # 否則使用 Pillow 的 SVG 支持
        try:
            import cairosvg
            png_data = cairosvg.svg2png(
                bytestring=svg.encode('utf-8'),
                dpi=self.config.get('image', {}).get('dpi', 300)
            )
            image = Image.open(io.BytesIO(png_data))
        except ImportError:
            # Fallback: 使用 Verovio 的 MIDI 渲染（較低質量）
            # 或者提示用戶安裝 cairosvg
            print("Warning: cairosvg not available, trying alternative method...")
            # 這裡可以使用其他方法，如 svglib + reportlab
            from svglib.svglib import svg2rlg
            from reportlab.graphics import renderPM

            # 將 SVG 字符串寫入臨時文件
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.svg', delete=False) as f:
                f.write(svg)
                temp_svg = f.name

            try:
                drawing = svg2rlg(temp_svg)
                png_data = renderPM.drawToString(drawing, fmt='PNG')
                image = Image.open(io.BytesIO(png_data))
            finally:
                os.unlink(temp_svg)

        # 保存到文件（如果指定）
        if output_path:
            image.save(output_path)

        return image

    def get_svg_dimensions(self, svg: str) -> Tuple[int, int]:
        """
        獲取 SVG 尺寸

        Args:
            svg: SVG 字符串

        Returns:
            (width, height) 元組
        """
        from lxml import etree

        root = etree.fromstring(svg.encode('utf-8'))
        width = int(float(root.get('width').replace('px', '')))
        height = int(float(root.get('height').replace('px', '')))

        return width, height
