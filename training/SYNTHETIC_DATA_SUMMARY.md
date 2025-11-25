# Synthetic Data Generation Research Summary

**Date**: 2025-11-25
**Status**: Research Complete ✅
**Next Phase**: Implementation (Phase 5)

---

## Executive Summary

I've completed comprehensive research on synthetic music notation generation methods to address the bottleneck symbols (fermata, barline_double, etc.) in our Phase 4 training. The research covered 6 major approaches and 15+ tools/libraries.

### 🎯 Recommended Solution

**Primary Method**: **Abjad + LilyPond** with domain randomization augmentation

**Why**:
- Python-native workflow (no external APIs)
- Publication-quality music engraving
- Full programmatic control over symbol placement
- Active development and documentation
- GPL license (compatible with our project)

**Timeline**: 2-3 weeks to generate 20,000+ high-quality synthetic images

---

## Key Research Findings

### 1. LilyPond-Based Generation ⭐ BEST

#### Abjad (Python API for LilyPond)
- **URL**: https://github.com/Abjad/abjad
- **Status**: Actively maintained, 3.31 stable release
- **Requirements**: Python 3.12+, LilyPond 2.25.26+
- **License**: GPL v3

**Pros**:
- Object-oriented Python API
- Precise control over music notation
- Can generate unlimited variations
- High-quality typesetting
- Supports fermatas, barlines, all accidentals

**Code Example**:
```python
import abjad

# Create notes with fermata
notes = [abjad.Note("c'4"), abjad.Note("d'4")]
fermata = abjad.Fermata()
abjad.attach(fermata, notes[1])

staff = abjad.Staff(notes)
score = abjad.Score([staff])

# Render to PNG + SVG
abjad.persist.as_png(score, "output.png", resolution=400)
abjad.persist.as_svg(score, "output.svg")
```

**Bounding Box Extraction**:
- LilyPond SVG output embeds metadata
- Can use event-listener for precise coordinates
- Parse SVG XML to extract symbol positions
- Convert to YOLO format

**Estimated Effort**: 5-7 days

---

#### python-ly (LilyPond File Manipulation)
- **URL**: https://github.com/frescobaldi/python-ly
- **Status**: Mature, used by Frescobaldi
- **Use Case**: Parse/manipulate existing LilyPond files

**Less suitable** for generation (better for editing existing files).

---

### 2. Verovio (MEI/MusicXML Renderer) ⭐ GOOD ALTERNATIVE

- **URL**: https://github.com/rism-digital/verovio
- **Status**: Very active, v5.6.0
- **Requirements**: Python 3.9+
- **License**: LGPL v3

**Pros**:
- Very fast rendering (JavaScript engine)
- SVG output preserves MEI hierarchy
- Built-in coordinate system
- No external dependencies

**Cons**:
- MEI/MusicXML format is verbose
- Less intuitive than Abjad for generation
- Better for rendering existing files

**Code Example**:
```python
import verovio

tk = verovio.toolkit()
tk.setOptions({"pageHeight": 2970, "pageWidth": 2100, "scale": 100})
tk.loadData(mei_content)
svg = tk.renderToSVG(1)
```

**Estimated Effort**: 4-5 days

---

### 3. Music21 + MuseScore

- **Music21**: https://web.mit.edu/music21/
- **MuseScore CLI**: https://musescore.org/

**Pros**:
- Familiar Python API for musicians
- Rich music theory library
- Good for MusicXML export

**Cons**:
- Requires MuseScore installation (~200MB)
- No direct bounding box extraction
- Two-step process

**Code Example**:
```python
from music21 import note, stream, expressions

s = stream.Score()
n = note.Note("C4", quarterLength=1.0)
n.expressions.append(expressions.Fermata())

s.write('musicxml', fp='output.xml')
# Then use MuseScore CLI to export PNG
```

**Estimated Effort**: 4-5 days

---

### 4. Font-Based Generation (SMuFL + Bravura) ⭐ FAST SUPPLEMENT

- **SMuFL**: https://www.smufl.org/
- **Bravura Font**: https://github.com/steinbergmedia/bravura
- **License**: SIL Open Font License (very permissive)

**Pros**:
- Very fast generation (pure Python + PIL)
- Full control over symbol placement
- Easy bounding box annotation
- Lightweight

**Cons**:
- Manual staff layout required
- Less realistic than engraving engines
- Need to handle complex notation manually

**SMuFL Codepoints**:
```python
SYMBOLS = {
    'fermata': '\uE4C0',
    'barline_double': '\uE031',
    'barline_final': '\uE032',
    'double_sharp': '\uE263',
    'double_flat': '\uE264',
}
```

**Code Example**:
```python
from PIL import Image, ImageDraw, ImageFont

img = Image.new('RGB', (800, 400), 'white')
draw = ImageDraw.Draw(img)
font = ImageFont.truetype('Bravura.otf', size=48)

# Draw fermata
draw.text((100, 100), '\uE4C0', font=font, fill='black')
```

**Estimated Effort**: 3-4 days

---

### 5. Domain Randomization & Data Augmentation ⭐ ESSENTIAL

Research shows domain randomization is critical for OMR model robustness.

**Key Techniques**:
1. **Background Textures**: Paper textures, noise
2. **Geometric Transforms**: Rotation (-5° to +5°), perspective
3. **Lighting**: Shadows, brightness/contrast variation
4. **Staff Variations**: Line spacing, thickness, curvature
5. **Font Variations**: Different music fonts

**Research Paper**: [Real World Music Object Recognition](https://transactions.ismir.net/articles/10.5334/tismir.157)
- Introduces **ScoreAug** technique
- Uses real-world textures from IMSLP
- Achieves 25% mAP improvement with synthetic data

**Code Example**:
```python
import cv2
import numpy as np

def apply_domain_randomization(img):
    # 1. Add paper texture
    texture = cv2.imread("paper_texture.jpg")
    img = cv2.addWeighted(img, 0.7, texture, 0.3, 0)

    # 2. Random rotation
    angle = random.uniform(-5, 5)
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h))

    # 3. Perspective transform
    # ... (see full guide for implementation)

    return img
```

**Augmentation Factor**: 5-10x per base image

---

### 6. Existing OMR Dataset Tools

#### OMR-Datasets (apacha)
- **URL**: https://github.com/apacha/OMR-Datasets
- **Contains**: Links to 30+ OMR datasets
- **Use Case**: Reference for data formats

#### Audiveris omr-dataset-tools
- **URL**: https://github.com/Audiveris/omr-dataset-tools
- **Use Case**: MuseScore → synthetic data pipeline
- **Status**: Documentation-focused

#### DeepScores Generation Method
- **Research**: https://arxiv.org/pdf/1804.00525
- **Method**: MusicXML → LilyPond → 400 DPI images
- **Scale**: 300,000 images generated
- **Key Insight**: All synthetic, but lacks diversity (same font/renderer)

---

## Recommended Implementation Pipeline

### Week 1: Base Generation
**Goal**: Generate 2,000 clean synthetic images

**Tools**: Abjad + LilyPond

**Output**:
- 1,000 images with fermatas (5-10 per image)
- 1,000 images with barline variants

**Script Structure**:
```python
# abjad_generator.py
def generate_fermata_score(num_fermatas=5):
    notes = []
    for i in range(20):
        note = abjad.Note(...)
        if random.random() < 0.25:
            abjad.attach(abjad.Fermata(), note)
        notes.append(note)
    return abjad.Score([abjad.Staff(notes)])

# Batch generation
for i in range(1000):
    score = generate_fermata_score()
    render_to_png_and_svg(score, f"fermata_{i:04d}", dpi=400)
```

**Estimated Time**: 4-6 hours (with parallelization)

---

### Week 2: Bounding Box Extraction
**Goal**: Extract precise YOLO annotations from SVG

**Method**: Parse LilyPond SVG output

**Script Structure**:
```python
# bbox_extractor.py
import xml.etree.ElementTree as ET

def extract_bounding_boxes(svg_path):
    tree = ET.parse(svg_path)
    bboxes = []

    for element in tree.iter():
        if 'fermata' in element.get('id', '').lower():
            x = float(element.get('x'))
            y = float(element.get('y'))
            width = float(element.get('width'))
            height = float(element.get('height'))

            bboxes.append({
                'class': 'fermata',
                'bbox': [x, y, width, height]
            })

    return bboxes

def convert_to_yolo_format(bboxes, img_w, img_h):
    # Normalize to [0, 1] and format as:
    # <class_id> <x_center> <y_center> <width> <height>
    ...
```

**Estimated Time**: 1-2 days

---

### Week 2-3: Domain Randomization
**Goal**: Generate 20,000 augmented images

**Method**: Apply 10 augmentations per base image

**Augmentations**:
1. Paper textures (50% probability)
2. Rotation ±5° (30%)
3. Perspective transform (30%)
4. Gaussian blur (30%)
5. Brightness/contrast (50%)
6. Salt & pepper noise (20%)

**Script Structure**:
```python
# augmentation.py
def augment_batch(input_dir, output_dir, aug_per_image=10):
    for img_path in Path(input_dir).glob("*.png"):
        img = Image.open(img_path)

        for i in range(aug_per_image):
            aug_img = apply_domain_randomization(img)
            aug_img.save(output_dir / f"{img_path.stem}_aug{i:02d}.png")
```

**Estimated Time**: 2-4 hours (batch processing)

---

### Week 3: Font-Based Supplementation (Optional)
**Goal**: Generate 1,000 additional images using Bravura font

**Focus**: Isolated symbol detection

**Script Structure**:
```python
# font_generator.py
from PIL import Image, ImageDraw, ImageFont

def generate_font_based_image(symbol, count=10):
    img = Image.new('RGB', (800, 400), 'white')
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype('Bravura.otf', 48)

    # Draw staff
    draw_staff(img, draw, y=100, width=800)

    # Draw random symbols
    for i in range(count):
        x = random.randint(50, 750)
        y = random.randint(80, 150)
        draw.text((x, y), symbol, font=font, fill='black')
```

**Estimated Time**: 1 day

---

## Expected Results

### Dataset Size
- **Base images**: 2,000
- **Augmented images**: 20,000 (10x augmentation)
- **Font-based images**: 1,000
- **Total**: 23,000 images

### Symbol Distribution
| Symbol | Instances | Current mAP50 | Target mAP50 |
|--------|-----------|--------------|--------------|
| fermata | ~100,000 | 0.286 | 0.65+ |
| barline_double | ~50,000 | 0.356 | 0.70+ |
| double_sharp | ~30,000 | 0.804 | 0.85+ |
| double_flat | ~30,000 | 0.707 | 0.80+ |

### Quality Metrics
- **Resolution**: 400 DPI (same as external datasets)
- **Format**: PNG + YOLO txt annotations
- **Diversity**: 10x augmentation ensures variation
- **Realism**: Domain randomization mimics real-world conditions

---

## Implementation Checklist

### Phase 5.1: Setup (Week 1, Day 1-2)
- [ ] Install Abjad (pip install abjad)
- [ ] Install LilyPond (sudo apt-get install lilypond)
- [ ] Test basic rendering
- [ ] Create directory structure
- [ ] Download Bravura font
- [ ] Collect paper texture library (20-30 images)

### Phase 5.2: Base Generation (Week 1, Day 3-7)
- [ ] Implement `abjad_generator.py`
- [ ] Create fermata generation templates
- [ ] Create barline generation templates
- [ ] Generate 2,000 base images
- [ ] Verify SVG output quality

### Phase 5.3: Bounding Box Extraction (Week 2, Day 1-3)
- [ ] Implement `bbox_extractor.py`
- [ ] Parse LilyPond SVG format
- [ ] Convert to YOLO format
- [ ] Validate annotations (visualize on images)
- [ ] Handle edge cases (overlapping symbols, etc.)

### Phase 5.4: Augmentation (Week 2, Day 4-6)
- [ ] Implement `augmentation.py`
- [ ] Add texture blending
- [ ] Add geometric transforms
- [ ] Add lighting variations
- [ ] Generate 20,000 augmented images
- [ ] Validate augmentation quality

### Phase 5.5: Font-Based Generation (Week 3, Day 1-2)
- [ ] Implement `font_generator.py`
- [ ] Load SMuFL font (Bravura)
- [ ] Render isolated symbols
- [ ] Generate 1,000 supplemental images

### Phase 5.6: Dataset Integration (Week 3, Day 3-5)
- [ ] Merge with Phase 4 dataset
- [ ] Update `data.yaml`
- [ ] Split train/val (90/10)
- [ ] Verify class distribution
- [ ] Run dataset validation script

### Phase 5.7: Training & Validation (Week 3, Day 6-7)
- [ ] Train test model (50 epochs)
- [ ] Evaluate mAP50 improvements
- [ ] Identify remaining weak spots
- [ ] Iterate on generation parameters if needed

---

## Cost-Benefit Analysis

### Effort vs. Alternatives

| Approach | Setup | Generation | Quality | Cost |
|----------|-------|-----------|---------|------|
| Manual annotation | 1h | 1000h+ | ⭐⭐⭐⭐⭐ | High |
| Abjad synthesis | 4h | 10h | ⭐⭐⭐⭐ | Low |
| Font-based only | 1h | 2h | ⭐⭐⭐ | Low |
| Cloud OMR API | 2h | $$$$ | ⭐⭐⭐ | High |

### Return on Investment

**Time Investment**: 2-3 weeks (40-60 hours)

**Expected Gains**:
- fermata mAP50: +0.40 (0.28 → 0.65+)
- barline_double mAP50: +0.35 (0.35 → 0.70+)
- Overall mAP50: +0.08-0.12 (0.58 → 0.66-0.70)

**Long-term Benefits**:
- Reusable generation pipeline
- Can generate more data anytime
- No licensing concerns (GPL compatible)
- Full control over data distribution

---

## Risk Assessment

### Technical Risks
| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Bounding box extraction inaccurate | Medium | High | Use LilyPond event-listener, validate visually |
| Synthetic data too different from real | Low | Medium | Apply domain randomization, test on real images |
| LilyPond rendering too slow | Low | Low | Parallelize generation, use batch mode |
| Overfitting to synthetic style | Medium | Medium | Mix 70% real + 30% synthetic in training |

### Scheduling Risks
| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Bounding box extraction takes longer | Medium | Medium | Start with font-based method (known coords) |
| GPU unavailable for validation | Low | Low | Use CPU for validation (slower but works) |

---

## References & Documentation

### Implementation Guides
- **Comprehensive Guide**: `training/docs/synthetic_data_generation_guide.md`
- **Quick Start**: `training/synthetic_generation/README.md`

### Research Papers
1. [Real World Music Object Recognition](https://transactions.ismir.net/articles/10.5334/tismir.157) - ScoreAug technique
2. [DeepScores Dataset](https://arxiv.org/pdf/1804.00525) - Large-scale synthetic generation
3. [Domain Randomization for Object Detection](https://arxiv.org/abs/2506.07539) - Manufacturing applications
4. [Synthio: Augmenting Small-Scale Audio Classification](https://arxiv.org/pdf/2410.02056) - Synthetic data best practices

### Tools & Libraries
1. [Abjad Documentation](https://abjad.github.io/)
2. [LilyPond Manual](https://lilypond.org/doc/v2.25/Documentation/)
3. [Verovio Reference Book](https://book.verovio.org/)
4. [SMuFL Specification](https://www.smufl.org/)
5. [Bravura Font Repository](https://github.com/steinbergmedia/bravura)

### Dataset Resources
1. [OMR-Datasets Collection](https://github.com/apacha/OMR-Datasets)
2. [Audiveris OMR Dataset Tools](https://github.com/Audiveris/omr-dataset-tools)
3. [MUSCIMA++ Tutorial](https://muscima.readthedocs.io/en/latest/Tutorial.html)

---

## Next Steps

### Immediate Actions
1. Review comprehensive guide: `docs/synthetic_data_generation_guide.md`
2. Set up development environment (Abjad + LilyPond)
3. Run test generation (10 images) to verify pipeline
4. Decide on augmentation strategy based on GPU availability

### Phase 5 Launch
1. Implement base generation (Week 1)
2. Develop bbox extraction (Week 2)
3. Apply augmentation (Week 2)
4. Integrate with Phase 4 dataset (Week 3)
5. Train and evaluate Phase 5 model (Week 3)

### Success Criteria
- [ ] Generate 20,000+ synthetic images
- [ ] Achieve >95% bbox annotation accuracy
- [ ] fermata mAP50 > 0.65
- [ ] barline_double mAP50 > 0.70
- [ ] Overall mAP50 > 0.65

---

**Status**: Research Complete ✅
**Recommendation**: Proceed with Abjad + LilyPond implementation
**Next Milestone**: Phase 5.1 Setup (2-3 days)
