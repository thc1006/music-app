"""
MEI 模板生成模組

程式化生成包含各種 barline 類型的 MEI 文件
"""

import random
from typing import List, Dict


class MEIGenerator:
    """MEI 文件生成器"""

    # Barline 類型映射到 MEI form 屬性
    BARLINE_FORMS = {
        'single': 'single',
        'double': 'dbl',
        'final': 'end',
        'repeat_left': 'rptstart',
        'repeat_right': 'rptend',
        'repeat_both': 'rptboth'
    }

    def __init__(self, config: Dict):
        """
        初始化生成器

        Args:
            config: 配置字典
        """
        self.config = config
        self.music_config = config.get('music', {})

    def generate_mei(self, barline_type: str) -> str:
        """
        生成包含指定 barline 類型的 MEI

        Args:
            barline_type: Barline 類型

        Returns:
            MEI XML 字符串
        """
        # 隨機選擇音樂參數
        time_sig = random.choice(self.music_config.get('time_signatures', ['4/4']))
        clef = random.choice(self.music_config.get('clefs', ['treble']))
        key_sig = random.choice(self.music_config.get('key_signatures', [0]))
        measures = random.choice(self.music_config.get('measures_per_image', [4]))

        # 解析拍號
        beats, beat_type = time_sig.split('/')

        # 生成 MEI 結構
        mei = self._build_mei_structure(
            clef=clef,
            key_sig=key_sig,
            time_sig=(beats, beat_type),
            measures=measures,
            barline_type=barline_type
        )

        return mei

    def _build_mei_structure(self, clef: str, key_sig: int,
                            time_sig: tuple, measures: int,
                            barline_type: str) -> str:
        """
        構建完整的 MEI XML 結構

        Args:
            clef: 譜號類型
            key_sig: 調號（升降記號數量）
            time_sig: 拍號 (beats, beat_type)
            measures: 小節數
            barline_type: Barline 類型

        Returns:
            MEI XML 字符串
        """
        beats, beat_type = time_sig

        # 決定在哪些位置放置 barline
        barline_positions = self._decide_barline_positions(measures, barline_type)

        # 生成小節內容
        measures_xml = []
        for i in range(measures):
            measure_xml = self._generate_measure(
                measure_n=i + 1,
                beats=int(beats),
                beat_type=int(beat_type),
                barline_right=barline_positions.get(i, None)
            )
            measures_xml.append(measure_xml)

        # 組裝完整 MEI
        mei = f"""<?xml version="1.0" encoding="UTF-8"?>
<mei xmlns="http://www.music-encoding.org/ns/mei" meiversion="5.0">
  <music>
    <body>
      <mdiv>
        <score>
          <scoreDef>
            <staffGrp>
              <staffDef n="1" lines="5" clef.shape="{self._get_clef_shape(clef)}"
                        clef.line="{self._get_clef_line(clef)}"
                        key.sig="{key_sig}s" meter.count="{beats}" meter.unit="{beat_type}"/>
            </staffGrp>
          </scoreDef>
          <section>
            {''.join(measures_xml)}
          </section>
        </score>
      </mdiv>
    </body>
  </music>
</mei>"""

        return mei

    def _decide_barline_positions(self, measures: int,
                                  barline_type: str) -> Dict[int, str]:
        """
        決定 barline 放置位置

        Args:
            measures: 總小節數
            barline_type: Barline 類型

        Returns:
            {measure_index: barline_form} 字典
        """
        positions = {}

        if barline_type == 'final':
            # 終止線只放在最後
            positions[measures - 1] = 'end'
        elif barline_type == 'double':
            # 雙線放在中間和最後
            if measures > 2:
                positions[measures // 2] = 'dbl'
            positions[measures - 1] = 'dbl'
        elif barline_type in ['repeat_left', 'repeat_right', 'repeat_both']:
            # 反復記號
            form = self.BARLINE_FORMS[barline_type]
            if measures > 2:
                positions[0] = form if 'start' in form or 'both' in form else 'single'
                positions[measures - 1] = form
            else:
                positions[measures - 1] = form
        else:
            # 單線（默認）
            positions[measures - 1] = 'single'

        return positions

    def _generate_measure(self, measure_n: int, beats: int,
                         beat_type: int, barline_right: str = None) -> str:
        """
        生成單個小節的 MEI

        Args:
            measure_n: 小節號
            beats: 拍子數
            beat_type: 拍子單位
            barline_right: 右側 barline 類型

        Returns:
            MEI measure XML
        """
        notes_per_measure = random.choice(
            self.music_config.get('notes_per_measure', [4])
        )

        # 生成音符
        notes = self._generate_notes(beats, beat_type, notes_per_measure)

        # 構建 measure XML
        barline_xml = ''
        if barline_right:
            barline_xml = f'<barLine form="{barline_right}"/>'

        measure = f"""
            <measure n="{measure_n}">
              <staff n="1">
                <layer n="1">
                  {notes}
                </layer>
              </staff>
              {barline_xml}
            </measure>"""

        return measure

    def _generate_notes(self, beats: int, beat_type: int,
                       count: int) -> str:
        """
        生成音符序列

        Args:
            beats: 拍子數
            beat_type: 拍子單位
            count: 音符數量

        Returns:
            MEI notes XML
        """
        notes = []
        durations = ['4', '2', '8', '1']  # 四分、二分、八分、全音符

        for i in range(count):
            pitch = random.choice(['C', 'D', 'E', 'F', 'G', 'A', 'B'])
            octave = random.choice([3, 4, 5])
            dur = random.choice(durations)

            note = f'<note pname="{pitch}" oct="{octave}" dur="{dur}"/>'
            notes.append(note)

        return '\n                  '.join(notes)

    def _get_clef_shape(self, clef: str) -> str:
        """獲取 MEI clef shape"""
        clef_map = {
            'treble': 'G',
            'bass': 'F',
            'alto': 'C',
            'tenor': 'C'
        }
        return clef_map.get(clef, 'G')

    def _get_clef_line(self, clef: str) -> str:
        """獲取 MEI clef line"""
        clef_map = {
            'treble': '2',
            'bass': '4',
            'alto': '3',
            'tenor': '4'
        }
        return clef_map.get(clef, '2')
