# OpenScore Lieder - Quick Report 🎵

**Analysis Date**: 2025-11-25 22:45 UTC+8
**Status**: ✅ Download Complete | ✅ Analysis Complete | 🔄 Rendering Pipeline WIP

---

## 🎯 TL;DR - Why This Matters

**OpenScore Lieder solves our fermata bottleneck:**

```
MUSCIMA++:        35 fermata annotations
OpenScore Lieder: 5,748 fermata annotations
                  ↑
                  164x MORE!
```

**Plus CC-0 license = fully commercial-ready** (no MUSCIMA++ NC restrictions)

---

## 📊 Key Numbers

| Metric | Value |
|--------|-------|
| **MusicXML Files** | 1,410 |
| **Fermata Annotations** | 5,748 |
| **Files with Fermatas** | 896 (63.5%) |
| **Barline Annotations** | 8,518 |
| **License** | CC-0 (Public Domain) |
| **Size** | ~200MB |
| **Download Location** | `training/datasets/external/openscore_lieder/` |

---

## 🚀 Expected Impact on Phase 4.5

### Before (Phase 4)
```
training/datasets/yolo_harmony_v2_phase4/
├── 24,566 images
├── 9,710 fermata annotations
└── 1,734 barline_double annotations
```

### After (Phase 4.5 with OpenScore)
```
training/datasets/yolo_harmony_v2_phase4.5/
├── ~25,600 images (+1,000 from OpenScore)
├── 15,458 fermata annotations (+59% ⬆)
└── 5,751 barline_double annotations (+232% ⬆)
```

**Expected mAP50 Improvements**:
- `fermata`: New class → target 0.65+
- `barline_double`: 0 → target 0.40+

---

## 📁 Generated Files

1. **Analysis Report** (Comprehensive)
   - Path: `training/OPENSCORE_LIEDER_ANALYSIS.md`
   - Contents: Full analysis, statistics, recommendations

2. **Analysis Script** (Runnable)
   - Path: `training/analyze_openscore_lieder.py`
   - Usage: `python analyze_openscore_lieder.py --show-examples 20`

3. **Rendering Script** (WIP)
   - Path: `training/render_openscore_to_yolo.py`
   - Status: Skeleton complete, needs pixel coordinate extraction
   - Next: Implement Verovio coordinate mapping

4. **Dataset Info** (Quick Reference)
   - Path: `training/datasets/external/openscore_lieder/DATASET_INFO.md`
   - Contents: Quick stats and usage instructions

---

## 🎼 Top Fermata Sources (for Priority Rendering)

| Rank | Composer | Title | Fermatas |
|------|----------|-------|----------|
| 1 | Elgar | The River, Op.60 | 46 |
| 2 | Viardot | Scène d'Hermione | 45 |
| 3 | Parratt | The Triumph of Victoria | 42 |
| 4 | Guest | The Bonnie Wee Wife | 37 |
| 5 | Elgar | The Torch, Op.60 | 37 |

*(Full list in OPENSCORE_LIEDER_ANALYSIS.md)*

---

## ⚡ Next Steps (Recommended Priority)

### Immediate (This Week)
1. **Install Verovio**
   ```bash
   pip install verovio pillow lxml tqdm
   ```

2. **Test Rendering on Sample Files**
   ```bash
   cd /home/thc1006/dev/music-app/training
   python render_openscore_to_yolo.py --limit 10 --filter fermata
   ```

3. **Complete Pixel Coordinate Extraction**
   - Verovio provides `toolkit.getElementsAtTime()` API
   - Need to map MusicXML element IDs to pixel coordinates
   - Estimated effort: 4-6 hours

### Short-term (Next Week)
4. **Render Fermata-Rich Subset**
   - Render 896 files with fermatas
   - Generate ~5,000 images (cropped + augmented)
   - Estimated time: 2-3 hours (automated)

5. **Merge with Phase 4 Dataset**
   - Run `merge_datasets_phase4.5.py` (to be created)
   - Validate YOLO format
   - Split train/val (80/20)

### Medium-term (Following Week)
6. **Train Phase 4.5 Model**
   - Start from Phase 3 best.pt
   - 150 epochs on RTX 5090
   - Expected training time: ~5 hours
   - Target mAP50: 0.60-0.62

---

## 🛠️ Technical Notes

### Rendering Options

| Tool | Pros | Cons | Recommendation |
|------|------|------|----------------|
| **Verovio** | Python API, SVG coords, fast | No native PNG | ⭐ Best choice |
| **MuseScore** | Direct PNG, widely used | Slow, no API | Backup option |
| **LilyPond** | Highest quality | Complex setup | Phase 5 |

**Chosen**: Verovio + cairosvg (for SVG→PNG conversion)

### Coordinate Extraction Strategy

```python
# Verovio provides:
tk = verovio.toolkit()
tk.loadData(musicxml_string)

# Get element coordinates
element_info = tk.getElementAttr(element_id)
# Returns: x, y, width, height in SVG units

# Convert to YOLO format:
# 1. Scale to image dimensions
# 2. Normalize to [0, 1]
# 3. Convert to center_x, center_y, width, height
```

---

## 📈 Strategic Value

### Advantages over Synthetic Data
1. **Real engravings**: Professional MusicXML from MuseScore editors
2. **Musical context**: Fermatas in actual phrase endings (not random)
3. **Diversity**: 1,410 different composers, styles, periods
4. **Quick turnaround**: Rendering < 1 week vs synthetic generation 2-3 weeks

### Complements Other Datasets
- **MUSCIMA++**: Orchestral scores (different layout)
- **Rebelo**: Isolated symbols (no context)
- **OpenScore**: Song format (vocal + piano, harmony context)

### License Advantage
- **CC-0** = Public Domain Dedication
- No attribution required
- **Fully commercial-ready** (can sell app with model trained on this data)
- Unlike MUSCIMA++ (NC = non-commercial restriction)

---

## 🔗 References

- **OpenScore GitHub**: https://github.com/OpenScore/Lieder
- **Verovio Toolkit**: https://www.verovio.org/
- **MusicXML Standard**: https://www.w3.org/2021/06/musicxml40/

---

## ✅ Status Checklist

- [x] Download OpenScore Lieder corpus
- [x] Analyze fermata/barline content
- [x] Create analysis report
- [x] Create rendering script skeleton
- [x] Update CLAUDE.md with findings
- [ ] Install Verovio dependencies
- [ ] Complete coordinate extraction logic
- [ ] Test rendering on sample files
- [ ] Render full fermata-rich subset
- [ ] Merge with Phase 4 dataset
- [ ] Train Phase 4.5 model

---

**Recommendation**: Prioritize OpenScore rendering for **Phase 4.5** before moving to synthetic data generation (Phase 5). The fermata coverage alone justifies this approach.

**Estimated Total Time**: 1-2 weeks to Phase 4.5 training completion.
