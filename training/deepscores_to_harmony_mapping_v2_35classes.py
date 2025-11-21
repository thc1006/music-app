#!/usr/bin/env python3
"""
DeepScoresV2 (208 classes) → Harmony OMR V2 (35 classes) 映射
🏆 終極完整方案 - 追求極致準確度與完整性
"""

# DeepScoresV2 類別映射到 35 類和聲符號系統
DEEPSCORES_TO_HARMONY_V2 = {
    # ==================== Tier 1: 音符與節奏 (10 類: 0-9) ====================

    # 0: notehead_filled - 實心符頭
    25: 0,  # noteheadBlackOnLine
    26: 0,  # noteheadBlackOnLineSmall
    27: 0,  # noteheadBlackInSpace
    28: 0,  # noteheadBlackInSpaceSmall
    157: 0, # noteheadFullSmall (MUSCIMA++)

    # 1: notehead_hollow - 空心符頭
    29: 1,  # noteheadHalfOnLine
    30: 1,  # noteheadHalfOnLineSmall
    31: 1,  # noteheadHalfInSpace
    32: 1,  # noteheadHalfInSpaceSmall
    33: 1,  # noteheadWholeOnLine
    34: 1,  # noteheadWholeOnLineSmall
    35: 1,  # noteheadWholeInSpace
    36: 1,  # noteheadWholeInSpaceSmall
    37: 1,  # noteheadDoubleWholeOnLine
    38: 1,  # noteheadDoubleWholeOnLineSmall
    39: 1,  # noteheadDoubleWholeInSpace
    40: 1,  # noteheadDoubleWholeInSpaceSmall
    158: 1, # noteheadHalfSmall (MUSCIMA++)
    159: 1, # noteheadWhole (MUSCIMA++)

    # 2: stem_up - 向上符桿
    42: 2,  # stem (大部分向上)
    161: 2, # stem (MUSCIMA++)

    # 3: stem_down - 向下符桿 (NEW)
    # Note: DeepScoresV2 不區分方向，由後處理判斷

    # 4: beam - 連音線
    122: 4, # beam
    201: 4, # beam (MUSCIMA++)

    # 5: flag_8th - 八分音符旗 (NEW - 細分)
    48: 5,  # flag8thUp
    49: 5,  # flag8thUpSmall
    54: 5,  # flag8thDown
    55: 5,  # flag8thDownSmall
    163: 5, # flag8thUp (MUSCIMA++)
    167: 5, # flag8thDown (MUSCIMA++)

    # 6: flag_16th - 十六分音符旗 (NEW)
    50: 6,  # flag16thUp
    56: 6,  # flag16thDown
    164: 6, # flag16thUp (MUSCIMA++)
    168: 6, # flag16thDown (MUSCIMA++)

    # 7: flag_32nd - 三十二分音符旗 (NEW)
    51: 7,  # flag32ndUp
    57: 7,  # flag32ndDown
    165: 7, # flag32ndUp (MUSCIMA++)
    169: 7, # flag32ndDown (MUSCIMA++)

    # 8: augmentation_dot - 附點 (NEW - 極重要！)
    41: 8,  # augmentationDot
    160: 8, # augmentationDot (MUSCIMA++)

    # 9: tie - 延音線 (NEW)
    102: 9,  # tie
    200: 9,  # tieStart
    199: 9,  # tieStop

    # ==================== Tier 2: 譜號與調性 (9 類: 10-18) ====================

    # 10: clef_treble - 高音譜號
    6: 10,   # clefG
    142: 10, # clefG (MUSCIMA++)

    # 11: clef_bass - 低音譜號
    9: 11,   # clefF
    144: 11, # clefF (MUSCIMA++)

    # 12: clef_alto - 中音譜號
    7: 12,   # clefCAlto
    143: 12, # clefC (MUSCIMA++)

    # 13: clef_tenor - 次中音譜號
    8: 13,   # clefCTenor

    # 14: accidental_sharp - 升記號
    64: 14, # accidentalSharp
    65: 14, # accidentalSharpSmall
    173: 14, # accidentalSharp (MUSCIMA++)

    # 15: accidental_flat - 降記號
    60: 15, # accidentalFlat
    61: 15, # accidentalFlatSmall
    171: 15, # accidentalFlat (MUSCIMA++)

    # 16: accidental_natural - 還原記號
    62: 16, # accidentalNatural
    63: 16, # accidentalNaturalSmall
    172: 16, # accidentalNatural (MUSCIMA++)

    # 17: accidental_double_sharp - 重升記號 (NEW)
    66: 17, # accidentalDoubleSharp
    174: 17, # accidentalDoubleSharp (MUSCIMA++)

    # 18: accidental_double_flat - 重降記號 (NEW)
    67: 18, # accidentalDoubleFlat
    175: 18, # accidentalDoubleFlat (MUSCIMA++)

    # ==================== Tier 3: 休止符 (5 類: 19-23) ====================

    # 19: rest_whole - 全休止符
    85: 19, # restWhole
    84: 19, # restDoubleWhole
    183: 19, # restWhole (MUSCIMA++)

    # 20: rest_half - 二分休止符
    86: 20, # restHalf
    184: 20, # restHalf (MUSCIMA++)

    # 21: rest_quarter - 四分休止符
    87: 21, # restQuarter
    185: 21, # restQuarter (MUSCIMA++)

    # 22: rest_8th - 八分休止符 (NEW - 細分)
    88: 22, # rest8th
    186: 22, # rest8th (MUSCIMA++)

    # 23: rest_16th - 十六分休止符 (NEW)
    89: 23, # rest16th
    187: 23, # rest16th (MUSCIMA++)
    # 更小的休止符映射到 rest_16th (較少用)
    90: 23, # rest32nd
    91: 23, # rest64th
    92: 23, # rest128th
    188: 23, # rest32nd (MUSCIMA++)
    189: 23, # rest64th (MUSCIMA++)

    # ==================== Tier 4: 樂譜結構 (6 類: 24-29) ====================

    # 24: barline - 單小節線 (NEW)
    3: 24,   # barline
    139: 24, # measureSeparator (MUSCIMA++)

    # 25: barline_double - 雙小節線 (NEW)
    120: 25, # barlineDouble

    # 26: barline_final - 終止線 (NEW)
    121: 26, # barlineFinal

    # 27: barline_repeat - 反覆記號 (NEW)
    4: 27,   # repeatDot
    5: 27,   # repeatLeft
    6: 27,   # repeatRight

    # 28: time_signature - 拍號
    13: 28, # timeSig0
    14: 28, # timeSig1
    15: 28, # timeSig2
    16: 28, # timeSig3
    17: 28, # timeSig4
    18: 28, # timeSig5
    19: 28, # timeSig6
    20: 28, # timeSig7
    21: 28, # timeSig8
    22: 28, # timeSig9
    23: 28, # timeSigCommon
    24: 28, # timeSigCutCommon
    155: 28, # timeSigCommon (MUSCIMA++)
    156: 28, # timeSigCutCommon (MUSCIMA++)
    # 數字（用於拍號）
    146: 28, # numeral0 (MUSCIMA++)
    147: 28, # numeral1
    148: 28, # numeral2
    149: 28, # numeral3
    150: 28, # numeral4
    151: 28, # numeral5
    152: 28, # numeral6
    153: 28, # numeral7
    154: 28, # numeral9

    # 29: key_signature - 調號
    68: 29, # keyFlat
    69: 29, # keyNatural
    70: 29, # keySharp

    # ==================== Tier 5: 樂句與表情 (4 類: 30-33) ====================

    # 30: slur - 圓滑線 (NEW - 樂句劃分)
    101: 30, # slur

    # 31: fermata - 延長記號 (NEW)
    93: 31,  # fermataAbove
    94: 31,  # fermataBelow

    # 32: dynamic_soft - 弱 (p, pp, ppp) (NEW)
    123: 32, # dynamicPiano
    124: 32, # dynamicPP
    125: 32, # dynamicPPP

    # 33: dynamic_loud - 強 (f, ff, fff) (NEW)
    128: 33, # dynamicForte
    129: 33, # dynamicFF
    130: 33, # dynamicFFF

    # ==================== Tier 6: 特殊符號 (1 類: 34) ====================

    # 34: ledger_line - 加線 (NEW - 替代 staffline)
    2: 34,   # ledgerLine
    138: 34, # legerLine (MUSCIMA++)
}

# ❌ 明確排除的類別（主要 OOM 來源）
EXCLUDED_CLASSES = {
    135: "staff",  # staffline - 175K instances, 主要 OOM 來源！
    208: "staff (MUSCIMA++)",
    # 其他過度細緻或無關的符號
    # (dynamics 中等力度: mf, mp, etc.)
    126: "dynamicMP",
    127: "dynamicMF",
    # (articulation: staccato, accent, etc.)
    # (ornaments: trill, mordent, etc.)
    # (lyrics, text)
}

# 反向映射：Harmony V2 class name → class id (35 classes)
HARMONY_V2_CLASS_NAMES = {
    # Tier 1: 音符與節奏
    0: "notehead_filled",
    1: "notehead_hollow",
    2: "stem_up",
    3: "stem_down",  # 由後處理判斷
    4: "beam",
    5: "flag_8th",
    6: "flag_16th",
    7: "flag_32nd",
    8: "augmentation_dot",
    9: "tie",

    # Tier 2: 譜號與調性
    10: "clef_treble",
    11: "clef_bass",
    12: "clef_alto",
    13: "clef_tenor",
    14: "accidental_sharp",
    15: "accidental_flat",
    16: "accidental_natural",
    17: "accidental_double_sharp",
    18: "accidental_double_flat",

    # Tier 3: 休止符
    19: "rest_whole",
    20: "rest_half",
    21: "rest_quarter",
    22: "rest_8th",
    23: "rest_16th",

    # Tier 4: 樂譜結構
    24: "barline",
    25: "barline_double",
    26: "barline_final",
    27: "barline_repeat",
    28: "time_signature",
    29: "key_signature",

    # Tier 5: 樂句與表情
    30: "slur",
    31: "fermata",
    32: "dynamic_soft",
    33: "dynamic_loud",

    # Tier 6: 特殊符號
    34: "ledger_line",
}

def get_harmony_class_id(deepscores_class_id: int) -> int:
    """
    將 DeepScoresV2 類別 ID 映射到 Harmony V2 35 類 ID

    Args:
        deepscores_class_id: DeepScoresV2 的類別 ID (1-208)

    Returns:
        Harmony V2 類別 ID (0-34)，如果無映射則返回 -1
    """
    return DEEPSCORES_TO_HARMONY_V2.get(deepscores_class_id, -1)

def get_mapped_classes_count():
    """獲取已映射的 DeepScoresV2 類別數量"""
    return len(DEEPSCORES_TO_HARMONY_V2)

def get_class_name(harmony_class_id: int) -> str:
    """獲取 Harmony V2 類別名稱"""
    return HARMONY_V2_CLASS_NAMES.get(harmony_class_id, "unknown")

if __name__ == "__main__":
    print(f"🏆 DeepScoresV2 → Harmony V2 (35 Classes) 映射統計:")
    print(f"已映射 DeepScoresV2 類別: {get_mapped_classes_count()} / 208")
    print(f"Harmony V2 目標類別: 35")

    # 統計每個 Harmony 類別有多少個 DeepScores 類別映射到它
    harmony_counts = {}
    for ds_cls, h_cls in DEEPSCORES_TO_HARMONY_V2.items():
        harmony_counts[h_cls] = harmony_counts.get(h_cls, 0) + 1

    print("\n各 Harmony V2 類別的來源數量:")
    for h_id in sorted(harmony_counts.keys()):
        print(f"  {h_id:2d} {get_class_name(h_id):25s}: {harmony_counts[h_id]:3d} 個來源類別")

    print(f"\n❌ 排除的類別數: {len(EXCLUDED_CLASSES)}")
    print("   主要排除: staff/staffline (175K instances - OOM 主因)")
