# 四部和聲規則檢查核心（Harmony Rule Engine Core）

這個小專案是一個**純規則導向**的四部和聲檢查核心，目標是把你在《和聲學總論》《Harmony 筆記》《Cadence 例題》等資料裡的規則，整理成一份**程式可以呼叫的邏輯庫**。

目前先聚焦在：

- 聲部進行（橫向）：
  - 過大跳進、禁止的旋律音程（七度、增減音程）。
  - 內聲部跳進限制。
  - 聲部交錯（crossing）與超越（overlap）。
- 聲部間關係（縱向）：
  - 平行八度／五度。
  - 隱伏八度／五度與例外情形。
- 和弦結構與重複音：
  - 三和弦的重複音、可省略音、導音的處理。
  - 七級和弦與六四和弦的特殊規則。
- 導音解決與終止式雛形：
  - 導音在外聲部時須向上解決到主音。
  - 利用最後兩個和弦嘗試分類簡單的終止式（PAC/IAC/HC/PC/DC）。

> ⚠️ 注意：這只是「規則引擎原型」。實際專案中，會由 OMR → MusicXML → 轉換為這裡的資料結構，再交給規則引擎檢查。

---

## 專案結構

- `README.md`：本說明文件。
- `harmony_rules.py`：主要的程式庫，提供 `HarmonyAnalyzer` 類別與多個檢查函式。
- `harmony_rules_zh.md`：將目前實作的規則條文化，對應到程式中的 `rule_id`。

未來你可以在此基礎上，慢慢補上更多：

- 聲部音域與間距限制。
- 功能和聲（T/S/D）與進行合理性。
- 非和弦音（passing / neighbor / suspension）的判斷。
- 更完整的終止式偵測。

---

## 安裝與使用方式（草稿）

本模組是純 Python，無額外依賴。

```bash
python -m pip install --upgrade pip
# 目前只是單檔程式庫，直接放進你的專案即可
```

在你的專案中：

```python
from harmony_rules import NoteEvent, ChordSnapshot, KeySignature, HarmonyAnalyzer

# 1. 準備和弦序列（通常會由 MusicXML / OMR 轉換而來）
chords = [
    ChordSnapshot(
        index=0,
        measure=1,
        beat=1.0,
        notes={
            "S": NoteEvent("S", midi=72, measure=1, beat=1.0),  # C5
            "A": NoteEvent("A", midi=67, measure=1, beat=1.0),  # G4
            "T": NoteEvent("T", midi=64, measure=1, beat=1.0),  # E4
            "B": NoteEvent("B", midi=55, measure=1, beat=1.0),  # G3
        },
    ),
    # ... 更多 ChordSnapshot
]

# 2. 指定調性（若要檢查導音解決與終止式，建議提供）
key = KeySignature(tonic_midi=60, mode="major")  # C 大調

# 3. 執行分析
analyzer = HarmonyAnalyzer(chords=chords, key=key)
violations = analyzer.analyze()

for v in violations:
    print(v.rule_id, v.message_zh, v.location)
```

---

## 與 Claude Code 的搭配思路（簡述）

- 在專案根目錄撰寫一份 `CLAUDE.md`（目前此壓縮檔尚未附上，你之後可以補）。
- 在 `CLAUDE.md` 中描述：
  - 如何從 MusicXML 建立 `ChordSnapshot` 陣列。
  - 如何呼叫 `HarmonyAnalyzer.analyze()`。
  - 讓 Claude Code 協助你產生更多規則、測試案例與單元測試。

這樣一來，就可以用 Claude Code 來：

1. 幫你把老師出的和聲作業轉成測試資料。
2. 自動產生針對某條規則的「錯誤範例」和「正確改寫」。
3. 幫忙重構與擴充這個規則引擎。