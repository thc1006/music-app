# OpenScore Lieder Corpus Analysis Report

**Date**: 2025-11-25
**Analyst**: Claude Code
**Corpus Version**: OpenScore Lieder (GitHub main branch)

---

## 📊 Executive Summary

The OpenScore Lieder corpus contains **1,410 professionally-encoded art song MusicXML files** with exceptional potential for Phase 4/5 training data:

| Metric | Value | Significance |
|--------|-------|--------------|
| **Total Files** | 1,410 | All successfully parsed |
| **Fermata Annotations** | **5,748** | 164x more than MUSCIMA++ (35) |
| **Files with Fermatas** | 896 (63.5%) | High coverage |
| **Barline Annotations** | 8,518 | Comprehensive coverage |
| **Double/Final Barlines** | 99.8% coverage | Nearly universal |
| **Parts per File** | 91.5% are 2-part | Vocal + Piano (harmony context) |

---

## 🎯 Key Findings

### 1. Fermata Coverage (Primary Target)

**OpenScore Lieder provides 164x more fermata annotations than MUSCIMA++**

```
MUSCIMA++:        35 fermatas
OpenScore Lieder: 5,748 fermatas
Multiplier:       164.2x
```

#### Distribution
- **Files with fermatas**: 896 (63.5%)
- **Average fermatas per file (all)**: 4.08
- **Average fermatas per file (with >0)**: 6.42
- **Maximum in single file**: 46 fermatas ("The River", Elgar Op.60)

#### Top Fermata Sources
1. **Elgar, Edward** - "The River" (46 fermatas)
2. **Viardot, Pauline** - "Scène d'Hermione" (45 fermatas)
3. **Parratt, Walter** - "The Triumph of Victoria" (42 fermatas, 6-part)
4. **Guest, Jane Mary** - "The Bonnie Wee Wife" (37 fermatas)

### 2. Barline Coverage

**Comprehensive barline type distribution:**

| Barline Type | Count | Percentage | YOLO Mapping |
|--------------|-------|------------|--------------|
| `light-light` | 3,958 | 46.5% | `barline` |
| `light-heavy` | 3,554 | 41.7% | `barline_double` / `barline_final` |
| `heavy-light` | 463 | 5.4% | `barline_double` |
| `regular` | 320 | 3.8% | `barline` |
| `none` | 136 | 1.6% | (invisible) |
| `dotted` | 71 | 0.8% | `barline` |
| `dashed` | 16 | 0.2% | `barline` |

**Key Insight**: 99.8% of files contain double/final barlines, addressing Phase 3's `barline_double` scarcity (mAP50 = 0).

### 3. Part Distribution

| Part Count | Files | Percentage | Music Type |
|------------|-------|------------|------------|
| 1 part | 5 | 0.4% | Monophonic |
| **2 parts** | **1,290** | **91.5%** | **Vocal + Piano** |
| 3 parts | 55 | 3.9% | Voice + Piano (split) |
| 4 parts | 21 | 1.5% | SATB arrangements |
| 5+ parts | 39 | 2.8% | Choral works |

**Strategic Value**:
- 2-part songs (vocal + piano) provide **harmony context** without full 4-part SATB
- Piano part contains chords and harmonic progressions
- Vocal part demonstrates melodic fermata placement (phrase endings)

---

## 💡 Training Data Potential

### Phase 4 Integration Strategy

#### Current Bottleneck Classes (Phase 3)
| Class | Phase 3 mAP50 | OpenScore Can Help? |
|-------|---------------|---------------------|
| `fermata` | N/A | ✅ **5,748 annotations** |
| `barline_double` | 0.000 | ✅ **4,017 light-heavy/heavy-light** |
| `barline_final` | N/A | ✅ **3,554 light-heavy** |

#### Rendering Pipeline Proposal

**Option 1: MuseScore Batch Rendering**
```bash
# Install MuseScore 2/3
sudo apt-get install musescore

# Batch render to PNG (1200 DPI for symbol clarity)
for file in scores/**/*.mxl; do
    musescore "$file" -o "${file%.mxl}.png" -r 1200
done
```

**Option 2: Verovio (Lightweight, Programmable)**
```bash
# Install via npm/pip
npm install -g verovio
# OR
pip install verovio

# Python batch rendering
python render_openscore_with_verovio.py
```

**Option 3: LilyPond (Highest Quality)**
```bash
# Convert MusicXML → LilyPond → SVG/PNG
musicxml2ly file.mxl -o file.ly
lilypond -fpng -dresolution=300 file.ly
```

#### Expected Output
- **1,410 rendered images** (full scores)
- **~20,000+ image crops** (focusing on fermata/barline regions)
- **Diverse visual styles** (different engraving engines)
- **Augmentation**: rotate, scale, add noise (simulate camera photos)

---

## 📈 Comparison with Existing Datasets

| Dataset | Images | Fermatas | Barlines | License | Status |
|---------|--------|----------|----------|---------|--------|
| **MUSCIMA++** | 140 | **35** | 3,330 | CC-BY-NC-SA | ✅ Phase 4 |
| **OpenScore Lieder** | **1,410** (MusicXML) | **5,748** | 8,518 | CC-0 | 🟢 Ready to Render |
| **Rebelo** | 9,900 | 0 | 0 | CC-BY-SA 4.0 | ✅ Phase 4 |
| **DoReMi** | 5,218 | Unknown | ~5,000+ | Research | ✅ Phase 3 |

**Advantage**:
- OpenScore is **CC-0 (public domain)** → fully commercial-ready
- MusicXML format → can render at any resolution/style
- Art songs → different context than orchestral scores (DoReMi/MUSCIMA++)

---

## 🎼 Musical Context & Diversity

### Composers Represented
- **German Lieder**: Schubert, Schumann, Brahms, Wolf, Strauss
- **French Mélodies**: Fauré, Debussy, Viardot, Grandval
- **English Song**: Elgar, Parry, Vaughan Williams, Quilter
- **Victorian Era**: Cowen, Parratt, Somervell

### Fermata Contexts in Art Songs
1. **Phrase Endings**: Most common (vocal cadence points)
2. **Dramatic Pauses**: Operatic/recitative sections
3. **Piano Interludes**: Between vocal phrases
4. **Final Cadences**: Often multiple fermatas (voice + piano)

**Training Benefit**: Model learns fermata placement in **melodic/harmonic context**, not just isolated symbols.

---

## 🚀 Recommended Next Steps

### Immediate Actions (Week 5-6)

1. **Install Rendering Engine** (Priority Order):
   ```bash
   # Option A: MuseScore (most compatible)
   sudo apt-get install musescore3

   # Option B: Verovio (Python-friendly)
   pip install verovio

   # Option C: LilyPond (highest quality)
   sudo apt-get install lilypond
   ```

2. **Create Rendering Script** (`render_openscore_to_yolo.py`):
   - Extract MusicXML files
   - Render to high-DPI PNG (≥1200 DPI)
   - Detect fermata/barline bounding boxes from MusicXML coordinates
   - Convert to YOLO format
   - Split train/val (80/20)

3. **Targeted Rendering Strategy**:
   - **Phase 4a**: Render 896 files with fermatas (priority)
   - **Phase 4b**: Render files with rare barline types (dotted, dashed)
   - **Phase 5**: Render all 1,410 files for comprehensive coverage

4. **Quality Control**:
   - Verify rendered images contain clear fermata symbols
   - Check bounding box accuracy (MusicXML coordinates may need calibration)
   - Compare rendered outputs across different engines (MuseScore vs Verovio)

### Integration with Phase 4

**Current Phase 4 Dataset**:
```
training/datasets/yolo_harmony_v2_phase4/
├── train: 22,110 images
├── val: 2,456 images
└── total: 24,566 images
```

**After OpenScore Integration (Phase 4.5)**:
```
yolo_harmony_v2_phase4.5/
├── train: ~23,000 images (base + OpenScore fermata-rich files)
├── val: ~2,600 images
└── total: ~25,600 images
```

**Expected Improvements**:
| Class | Phase 3 | Phase 4 (current) | Phase 4.5 (with OpenScore) |
|-------|---------|-------------------|---------------------------|
| `fermata` | 8,440 annot. | 9,710 annot. | **15,458 annot.** (+59%) |
| `barline_double` | 1,228 annot. | 1,734 annot. | **5,751 annot.** (+232%) |
| `barline` | 25,000 annot. | 30,979 annot. | **39,497 annot.** (+28%) |

---

## 🛠️ Technical Implementation Notes

### MusicXML → Bounding Box Conversion Challenges

1. **Coordinate Systems**:
   - MusicXML uses "tenths" (1/10 of staff space)
   - Rendering engines use pixels
   - Need to establish scaling factor (tenths → pixels)

2. **Fermata Position**:
   - MusicXML: `<fermata>` attached to `<notations>` element
   - Visual position: above/below staff, centered on notehead
   - Bounding box: need to calculate from rendered image

3. **Barline Extraction**:
   - MusicXML: `<barline>` with `<bar-style>` element
   - Position: x-coordinate at measure boundary
   - Height: spans entire staff system

### Proposed Rendering Workflow

```python
# Pseudocode for render_openscore_to_yolo.py

for mxl_file in openscore_files:
    # 1. Parse MusicXML
    score = parse_musicxml(mxl_file)

    # 2. Render to PNG
    image = render_with_musescore(mxl_file, dpi=1200)

    # 3. Extract symbol positions from MusicXML
    fermatas = score.find_all('fermata')
    barlines = score.find_all('barline')

    # 4. Map MusicXML coordinates to pixel coordinates
    for fermata in fermatas:
        x, y = map_musicxml_to_pixels(fermata, dpi=1200)
        bbox = create_bbox(x, y, width=50, height=30)  # Estimate
        yolo_annotations.append(bbox)

    # 5. Save YOLO format
    save_yolo_annotation(image_path, yolo_annotations)
```

**Challenge**: MusicXML doesn't provide exact pixel coordinates — need to either:
- **Option A**: Use rendering engine API to get coordinates (Verovio supports this)
- **Option B**: Estimate from staff position + symbol type
- **Option C**: Use computer vision to detect symbols in rendered image, then match with MusicXML

**Recommended**: Use **Verovio** — it provides both rendering AND coordinate mapping via toolkit API.

---

## 📚 References

- **OpenScore Lieder Repository**: https://github.com/OpenScore/Lieder
- **License**: CC-0 (Public Domain Dedication)
- **MusicXML Standard**: https://www.w3.org/2021/06/musicxml40/
- **Verovio Toolkit**: https://www.verovio.org/
- **MuseScore**: https://musescore.org/

---

## ✅ Conclusion

**OpenScore Lieder is a game-changer for Phase 4/5 training:**

1. **Massive Fermata Coverage**: 5,748 annotations (164x MUSCIMA++)
2. **Commercial-Ready**: CC-0 license (unlike MUSCIMA++ NC restriction)
3. **Diverse Visual Context**: Art songs (different from orchestral scores)
4. **High-Quality Encodings**: Professional MusicXML, ready to render
5. **Addresses Key Bottlenecks**: fermata (new class) + barline_double (mAP50=0)

**Recommendation**: Prioritize OpenScore rendering for **Phase 4.5** before moving to Phase 5 high-resolution training.

**Next File to Create**: `render_openscore_to_yolo.py` (Verovio-based rendering + YOLO conversion)
