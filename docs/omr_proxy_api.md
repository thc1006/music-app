# 雲端 OMR Proxy API 規格草案

本檔說明「樂譜影像 → 四部和聲結構化 JSON」的雲端 API 設計。  
目標：Android 端只需上傳圖片與收到 JSON，所有 OMR / 大模型邏輯都在後端完成。

---
## 1. 端點設計

- Method: POST  
- Path: `/api/omr/score`  
- 認證方式：
  - 建議使用 API key / JWT / 其他安全機制，由後端實作。  
  - Android APP 不應直接暴露第三方 LLM 服務的密鑰。

---
## 2. 請求格式（Request）

建議使用 `application/json`，圖片以 Base64 編碼傳送。


### 2.1 Request JSON Schema（概念版）

```jsonc
{
  "type": "object",
  "required": ["image_base64"],
  "properties": {
    "image_base64": {
      "type": "string",
      "description": "樂譜圖片的 Base64 編碼（建議 JPEG / PNG）。"
    },
    "filename": {
      "type": "string",
      "description": "可選，用於記錄檔名或 debug。"
    },
    "options": {
      "type": "object",
      "properties": {
        "staff_layout": {
          "type": "string",
          "enum": ["SATB_GRAND_STAFF", "PIANO_REDUCTION", "UNKNOWN"],
          "description": "樂譜排版提示，例如合唱四部在兩行鋼琴譜或四行獨立。"
        },
        "expected_voices": {
          "type": "array",
          "items": { "type": "string", "enum": ["S", "A", "T", "B"] },
          "description": "預期聲部列表，預設為 ["S","A","T","B"]。"
        },
        "language_hint": {
          "type": "string",
          "enum": ["zh-TW", "en"],
          "description": "說明文字語言提示，主要用於 LLM OMR。"
        }
      }
    }
  }
}
```

---
## 3. 回應格式（Response）

### 3.1 Response JSON Schema（概念版）

```jsonc
{
  "type": "object",
  "required": ["measures"],
  "properties": {
    "measures": {
      "type": "array",
      "description": "每一小節的資料。",
      "items": {
        "type": "object",
        "required": ["index", "chords"],
        "properties": {
          "index": {
            "type": "integer",
            "description": "小節編號，從 1 開始。"
          },
          "time_signature": {
            "type": "string",
            "description": "可選，例如 "4/4", "3/4"。"
          },
          "key_signature": {
            "type": "string",
            "description": "可選，例如 "C major", "a minor"。"
          },
          "chords": {
            "type": "array",
            "description": "此小節中的和聲快照（可視為拍點或和弦變化）。",
            "items": {
              "type": "object",
              "required": ["beat", "notes"],
              "properties": {
                "beat": {
                  "type": "number",
                  "description": "此和弦在小節中的拍位置，例如 1.0, 1.5, 2.0。"
                },
                "notes": {
                  "type": "array",
                  "items": {
                    "type": "object",
                    "required": ["voice", "pitch", "duration"],
                    "properties": {
                      "voice": {
                        "type": "string",
                        "enum": ["S", "A", "T", "B"],
                        "description": "聲部，例如 S, A, T, B。"
                      },
                      "pitch": {
                        "type": "string",
                        "description": "音高，例如 "C4", "F#3"。"
                      },
                      "duration": {
                        "type": "number",
                        "description": "音符時值（以拍為單位，例如四分音符=1.0）。"
                      },
                      "tie": {
                        "type": "string",
                        "enum": ["NONE", "START", "STOP", "CONTINUE"],
                        "description": "可選，連結音狀態。"
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    },
    "raw_model_output": {
      "type": "string",
      "description": "可選，原始 LLM 或 OMR 模型輸出（debug 用）。"
    },
    "warnings": {
      "type": "array",
      "items": { "type": "string" },
      "description": "解析過程中的警告訊息。"
    }
  }
}
```

---
## 4. Android 端對應資料結構（簡述）

Android 端會將上述 JSON 轉換為：

- KeySignature / TimeSignature 等簡單型別或小 class。  
- 一組 List<ChordSnapshot>，其中每個 ChordSnapshot 包含：
  - measureIndex: Int  
  - beat: Double  
  - notes: List<NoteEvent>

這些結構再交給 Kotlin 規則引擎進行檢查。

---
## 5. 錯誤處理建議

- 4xx：請求錯誤（格式錯誤、缺少 image_base64）。  
- 5xx：後端內部錯誤或外部 OMR / LLM 服務失敗。  
- 建議在 Response 中加入簡易 error 欄位供 APP 顯示，例如：

```jsonc
{
  "error": {
    "code": "OMR_TIMEOUT",
    "message": "OMR 服務逾時，請稍後再試。"
  }
}
```

APP 收到 error 時，可提示使用者重試或調整圖片品質。
