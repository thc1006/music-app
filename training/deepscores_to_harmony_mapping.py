#!/usr/bin/env python3
"""
DeepScoresV2 (208 classes) → Harmony OMR (20 classes) 映射
針對四部和聲樂譜辨識優化的類別映射
"""

# DeepScoresV2 類別映射到我們的 20 類和聲符號系統
DEEPSCORES_TO_HARMONY = {
    # ==================== 符頭 (Noteheads) ====================
    # 實心符頭 → notehead_filled (class 0)
    25: 0,  # noteheadBlackOnLine
    26: 0,  # noteheadBlackOnLineSmall
    27: 0,  # noteheadBlackInSpace
    28: 0,  # noteheadBlackInSpaceSmall
    157: 0, # noteheadFullSmall (MUSCIMA++)

    # 空心符頭 → notehead_hollow (class 1)
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

    # ==================== 符幹 (Stems) ====================
    # 向上/向下符幹統一為 stem_up (簡化處理，可由位置推斷方向)
    42: 2,  # stem → stem_up (class 2)
    161: 2, # stem (MUSCIMA++)

    # ==================== 連音線 (Beams) ====================
    122: 4, # beam → beam (class 4)
    201: 4, # beam (MUSCIMA++)

    # ==================== 符尾 (Flags) ====================
    48: 5,  # flag8thUp → flag (class 5)
    49: 5,  # flag8thUpSmall
    50: 5,  # flag16thUp
    51: 5,  # flag32ndUp
    52: 5,  # flag64thUp
    53: 5,  # flag128thUp
    54: 5,  # flag8thDown
    55: 5,  # flag8thDownSmall
    56: 5,  # flag16thDown
    57: 5,  # flag32ndDown
    58: 5,  # flag64thDown
    59: 5,  # flag128thDown
    163: 5, # flag8thUp (MUSCIMA++)
    164: 5, # flag16thUp (MUSCIMA++)
    165: 5, # flag32ndUp (MUSCIMA++)
    166: 5, # flag64thUp (MUSCIMA++)
    167: 5, # flag8thDown (MUSCIMA++)
    168: 5, # flag16thDown (MUSCIMA++)
    169: 5, # flag32ndDown (MUSCIMA++)
    170: 5, # flag64thDown (MUSCIMA++)

    # ==================== 譜號 (Clefs) ====================
    6: 6,   # clefG → clef_treble (class 6)
    142: 6, # clefG (MUSCIMA++)

    9: 7,   # clefF → clef_bass (class 7)
    144: 7, # clefF (MUSCIMA++)

    7: 8,   # clefCAlto → clef_alto (class 8)
    143: 8, # clefC (MUSCIMA++)

    8: 9,   # clefCTenor → clef_tenor (class 9)

    # ==================== 升降記號 (Accidentals) ====================
    64: 10, # accidentalSharp → accidental_sharp (class 10)
    65: 10, # accidentalSharpSmall
    66: 10, # accidentalDoubleSharp
    173: 10, # accidentalSharp (MUSCIMA++)
    174: 10, # accidentalDoubleSharp (MUSCIMA++)

    60: 11, # accidentalFlat → accidental_flat (class 11)
    61: 11, # accidentalFlatSmall
    67: 11, # accidentalDoubleFlat
    171: 11, # accidentalFlat (MUSCIMA++)
    175: 11, # accidentalDoubleFlat (MUSCIMA++)

    62: 12, # accidentalNatural → accidental_natural (class 12)
    63: 12, # accidentalNaturalSmall
    172: 12, # accidentalNatural (MUSCIMA++)

    # ==================== 休止符 (Rests) ====================
    87: 13, # restQuarter → rest_quarter (class 13)
    185: 13, # restQuarter (MUSCIMA++)

    86: 14, # restHalf → rest_half (class 14)
    184: 14, # restHalf (MUSCIMA++)

    85: 15, # restWhole → rest_whole (class 15)
    84: 15, # restDoubleWhole
    183: 15, # restWhole (MUSCIMA++)

    # 其他休止符映射到對應類別（如果需要擴展）
    88: 13, # rest8th → 暫時映射到 quarter
    89: 13, # rest16th
    90: 13, # rest32nd
    91: 13, # rest64th
    92: 13, # rest128th
    186: 13, # rest8th (MUSCIMA++)
    187: 13, # rest16th (MUSCIMA++)
    188: 13, # rest32nd (MUSCIMA++)
    189: 13, # rest64th (MUSCIMA++)

    # ==================== 小節線 (Barlines) ====================
    # DeepScoresV2 沒有明確的 barline 類別，需要從其他來源識別
    # 暫時留空，class 16

    # ==================== 拍號 (Time Signatures) ====================
    13: 17, # timeSig0 → time_signature (class 17)
    14: 17, # timeSig1
    15: 17, # timeSig2
    16: 17, # timeSig3
    17: 17, # timeSig4
    18: 17, # timeSig5
    19: 17, # timeSig6
    20: 17, # timeSig7
    21: 17, # timeSig8
    22: 17, # timeSig9
    23: 17, # timeSigCommon
    24: 17, # timeSigCutCommon
    155: 17, # timeSigCommon (MUSCIMA++)
    156: 17, # timeSigCutCommon (MUSCIMA++)

    # 數字（用於拍號）
    146: 17, # numeral0 (MUSCIMA++)
    147: 17, # numeral1
    148: 17, # numeral2
    149: 17, # numeral3
    150: 17, # numeral4
    151: 17, # numeral5
    152: 17, # numeral6
    153: 17, # numeral7
    154: 17, # numeral9

    # ==================== 調號 (Key Signatures) ====================
    68: 18, # keyFlat → key_signature (class 18)
    69: 18, # keyNatural
    70: 18, # keySharp

    # ==================== 五線譜線 (Staff Lines) ====================
    135: 19, # staff → staffline (class 19)
    208: 19, # staff (MUSCIMA++)

    # Ledger lines
    2: 19,   # ledgerLine → 也映射到 staffline
    138: 19, # legerLine (MUSCIMA++)
}

# 反向映射：Harmony class name → class id
HARMONY_CLASS_NAMES = {
    0: "notehead_filled",
    1: "notehead_hollow",
    2: "stem_up",
    3: "stem_down",  # 未使用，保留
    4: "beam",
    5: "flag",
    6: "clef_treble",
    7: "clef_bass",
    8: "clef_alto",
    9: "clef_tenor",
    10: "accidental_sharp",
    11: "accidental_flat",
    12: "accidental_natural",
    13: "rest_quarter",
    14: "rest_half",
    15: "rest_whole",
    16: "barline",  # 未映射，需要其他方法檢測
    17: "time_signature",
    18: "key_signature",
    19: "staffline",
}

def get_harmony_class_id(deepscores_class_id: int) -> int:
    """
    將 DeepScoresV2 類別 ID 映射到 Harmony 20 類 ID

    Args:
        deepscores_class_id: DeepScoresV2 的類別 ID (1-208)

    Returns:
        Harmony 類別 ID (0-19)，如果無映射則返回 -1
    """
    return DEEPSCORES_TO_HARMONY.get(deepscores_class_id, -1)

def get_mapped_classes_count():
    """獲取已映射的 DeepScoresV2 類別數量"""
    return len(DEEPSCORES_TO_HARMONY)

def get_class_name(harmony_class_id: int) -> str:
    """獲取 Harmony 類別名稱"""
    return HARMONY_CLASS_NAMES.get(harmony_class_id, "unknown")

if __name__ == "__main__":
    print(f"DeepScoresV2 → Harmony 映射統計:")
    print(f"已映射 DeepScoresV2 類別: {get_mapped_classes_count()} / 208")
    print(f"Harmony 目標類別: 20")

    # 統計每個 Harmony 類別有多少個 DeepScores 類別映射到它
    harmony_counts = {}
    for ds_cls, h_cls in DEEPSCORES_TO_HARMONY.items():
        harmony_counts[h_cls] = harmony_counts.get(h_cls, 0) + 1

    print("\n各 Harmony 類別的來源數量:")
    for h_id in sorted(harmony_counts.keys()):
        print(f"  {h_id:2d} {get_class_name(h_id):20s}: {harmony_counts[h_id]:3d} 個來源類別")
