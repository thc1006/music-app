# Synthetic Data Generation Guide for Music Symbol Detection

**Last Updated**: 2025-11-25
**Target**: Generate 10,000+ high-quality synthetic training images with accurate bounding boxes for fermata and barline symbols

---

## Executive Summary

Based on comprehensive research, the **recommended approach** is:

1. **Primary Method**: Abjad + LilyPond (Python-native, full control)
2. **Secondary Method**: Verovio (MEI/MusicXML → SVG with coordinates)
3. **Augmentation**: Domain randomization techniques from OMR research

**Estimated Timeline**: 2-3 weeks for full implementation

---

## Method 1: Abjad + LilyPond (⭐ RECOMMENDED)

### Overview
Abjad is a Python API for building LilyPond files programmatically. It provides object-oriented music notation generation with full control over symbols, layout, and rendering.

### Resources
- **Documentation**: https://abjad.github.io/
- **GitHub**: https://github.com/Abjad/abjad
- **PyPI**: https://pypi.org/project/abjad/

### Requirements
- Python 3.12+
- LilyPond 2.25.26+
- Abjad 3.31

### Installation

```bash
# Install LilyPond (Ubuntu/Debian)
sudo apt-get install lilypond

# Install Abjad
pip install abjad

# Verify installation
python -c "import abjad; print(abjad.__version__)"
lilypond --version
```

### Code Example: Generate Fermata Images

```python
import abjad
import random
from pathlib import Path

def generate_fermata_score(num_fermatas=5):
    """Generate a score with random fermatas."""

    # Create notes with random pitches and durations
    notes = []
    durations = ['4', '8', '2', '1']
    pitches = ['c', 'd', 'e', 'f', 'g', 'a', 'b']

    for i in range(20):
        pitch = random.choice(pitches)
        octave = random.choice(["'", "''"])
        duration = random.choice(durations)
        note = abjad.Note(f"{pitch}{octave}{duration}")

        # Add fermata to random notes
        if random.random() < num_fermatas / 20:
            fermata = abjad.Fermata()
            abjad.attach(fermata, note)

        notes.append(note)

    # Create staff and score
    staff = abjad.Staff(notes)
    score = abjad.Score([staff])

    return score

def generate_barline_score():
    """Generate a score with various barline types."""

    staff = abjad.Staff()

    # Add measures with different barline types
    measure1 = abjad.Container("c'4 d'4 e'4 f'4")
    measure2 = abjad.Container("g'4 a'4 b'4 c''4")
    measure3 = abjad.Container("d''4 e''4 f''4 g''4")

    staff.extend([measure1, measure2, measure3])

    # Add different barline types
    # Single barline (default)
    abjad.attach(abjad.BarLine('|'), measure1[-1])

    # Double barline
    abjad.attach(abjad.BarLine('||'), measure2[-1])

    # Final barline
    abjad.attach(abjad.BarLine('|.'), measure3[-1])

    score = abjad.Score([staff])
    return score

def render_to_png_and_svg(score, output_path, dpi=400):
    """Render score to PNG and SVG with high resolution."""

    # Create LilyPond file
    lilypond_file = abjad.LilyPondFile([score])

    # Set paper size and resolution
    lilypond_file.paper_block.paper_width = 210  # A4 width in mm
    lilypond_file.paper_block.paper_height = 297  # A4 height in mm

    # Render to PNG
    png_path = Path(output_path).with_suffix('.png')
    abjad.persist.as_png(lilypond_file, png_path, resolution=dpi)

    # Render to SVG (for bounding box extraction)
    svg_path = Path(output_path).with_suffix('.svg')
    abjad.persist.as_svg(lilypond_file, svg_path)

    return png_path, svg_path

# Generate batch
for i in range(100):
    score = generate_fermata_score(num_fermatas=5)
    render_to_png_and_svg(score, f"output/fermata_{i:04d}", dpi=400)
```

### Extracting Bounding Boxes from SVG

LilyPond SVG output embeds metadata that can be parsed to extract coordinates:

```python
import xml.etree.ElementTree as ET
from pathlib import Path

def extract_bounding_boxes(svg_path):
    """Extract bounding boxes from LilyPond SVG output."""

    tree = ET.parse(svg_path)
    root = tree.getroot()

    # LilyPond embeds bounding box info in SVG
    # Each symbol has an id and position
    bboxes = []

    for element in root.iter():
        # Look for fermata and barline elements
        if 'fermata' in str(element.get('id', '')).lower():
            x = float(element.get('x', 0))
            y = float(element.get('y', 0))
            width = float(element.get('width', 0))
            height = float(element.get('height', 0))

            bboxes.append({
                'class': 'fermata',
                'x': x,
                'y': y,
                'width': width,
                'height': height
            })

    return bboxes

def convert_to_yolo_format(bboxes, img_width, img_height):
    """Convert bounding boxes to YOLO format."""

    yolo_annotations = []

    for bbox in bboxes:
        # YOLO format: <class_id> <x_center> <y_center> <width> <height>
        # All normalized to [0, 1]
        x_center = (bbox['x'] + bbox['width'] / 2) / img_width
        y_center = (bbox['y'] + bbox['height'] / 2) / img_height
        width = bbox['width'] / img_width
        height = bbox['height'] / img_height

        class_id = 26 if bbox['class'] == 'fermata' else 0  # Adjust class IDs

        yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    return "\n".join(yolo_annotations)
```

### Advanced: Using LilyPond's Event Listener

For more precise bounding box extraction:

```python
# Add to LilyPond file header
lilypond_header = r"""
\include "event-listener.ly"

\paper {
  #(define dump-extents #t)
}

\layout {
  \context {
    \Score
    \consists #listeners
  }
}
"""

# This generates a .dat file with exact symbol positions
```

### Pros
✅ Full control over music notation generation
✅ Python-native workflow, no external API calls
✅ High-quality typesetting (publication standard)
✅ Can generate unlimited variations
✅ Free and open-source (GPL)
✅ Active development and documentation

### Cons
❌ Requires LilyPond installation (~100MB)
❌ Bounding box extraction requires SVG parsing
❌ Rendering can be slow for large batches (2-5 seconds per image)
❌ Learning curve for Abjad API

### Estimated Effort
- **Setup**: 2-4 hours
- **Basic generation**: 1-2 days
- **Bounding box extraction**: 2-3 days
- **Batch generation pipeline**: 1-2 days
- **Total**: ~5-7 days

---

## Method 2: Verovio (MEI/MusicXML → SVG)

### Overview
Verovio is a fast, lightweight library for rendering MEI (Music Encoding Initiative) and MusicXML files to SVG with precise coordinate information.

### Resources
- **Documentation**: https://book.verovio.org/
- **GitHub**: https://github.com/rism-digital/verovio
- **PyPI**: https://pypi.org/project/verovio/

### Requirements
- Python 3.9+
- Verovio 5.6.0+

### Installation

```bash
pip install verovio
```

### Code Example

```python
import verovio
import json
from pathlib import Path

def generate_with_verovio(mei_content, output_path, dpi=400):
    """Render MEI/MusicXML to SVG with Verovio."""

    tk = verovio.toolkit()

    # Set options
    options = {
        "pageHeight": 2970,  # A4 height at 100 dpi
        "pageWidth": 2100,   # A4 width at 100 dpi
        "scale": 100,        # Scale factor
        "breaks": "auto",
        "adjustPageHeight": True,
    }
    tk.setOptions(json.dumps(options))

    # Load MEI/MusicXML
    tk.loadData(mei_content)

    # Render to SVG
    svg = tk.renderToSVG(1)  # Page 1

    # Save SVG
    with open(output_path, 'w') as f:
        f.write(svg)

    # Extract element positions
    elements = tk.getElementsAtTime(1000)  # Get all elements

    return svg, elements

def extract_fermatas_from_verovio(tk):
    """Extract fermata positions from Verovio toolkit."""

    # Query MEI structure for fermata elements
    mei = tk.getMEI(0, False)  # Get MEI representation

    # Parse MEI XML to find fermata elements and their positions
    # Verovio preserves MEI hierarchy in SVG

    return fermatas

# Example usage
mei_file = Path("input/score.mei").read_text()
svg, elements = generate_with_verovio(mei_file, "output/score.svg")
```

### MEI Template for Fermatas

```xml
<?xml version="1.0" encoding="UTF-8"?>
<mei xmlns="http://www.music-encoding.org/ns/mei" meiversion="4.0.0">
  <music>
    <body>
      <mdiv>
        <score>
          <scoreDef>
            <staffGrp>
              <staffDef n="1" lines="5" clef.shape="G" clef.line="2"/>
            </staffGrp>
          </scoreDef>
          <section>
            <measure n="1">
              <staff n="1">
                <layer n="1">
                  <note pname="c" oct="4" dur="4"/>
                  <note pname="d" oct="4" dur="4">
                    <fermata/>
                  </note>
                  <note pname="e" oct="4" dur="4"/>
                  <note pname="f" oct="4" dur="4"/>
                </layer>
              </staff>
            </measure>
          </section>
        </score>
      </mdiv>
    </body>
  </mei>
</mei>
```

### Pros
✅ Very fast rendering (JavaScript engine)
✅ SVG output preserves MEI hierarchy (easy to extract bounding boxes)
✅ No external dependencies (pure Python binding)
✅ Supports MusicXML, Humdrum, MEI
✅ LGPL license (permissive)
✅ Built-in coordinate system

### Cons
❌ MEI/MusicXML format is verbose
❌ Less intuitive than Python API (Abjad)
❌ Limited programmatic generation (mainly for rendering existing files)
❌ Requires learning MEI schema

### Estimated Effort
- **Setup**: 1-2 hours
- **MEI template creation**: 1-2 days
- **Batch generation**: 2-3 days
- **Total**: ~4-5 days

---

## Method 3: Music21 + MuseScore

### Overview
Use music21 to generate MusicXML programmatically, then export via MuseScore command-line.

### Resources
- **Music21 Documentation**: https://web.mit.edu/music21/
- **MuseScore CLI**: https://musescore.org/

### Installation

```bash
pip install music21

# Install MuseScore
sudo apt-get install musescore3  # Ubuntu
# or download from https://musescore.org/
```

### Code Example

```python
from music21 import note, stream, expressions, converter
import subprocess
from pathlib import Path

def generate_fermata_score_music21():
    """Generate score with fermatas using music21."""

    s = stream.Score()
    part = stream.Part()

    # Create measures with fermatas
    for i in range(4):
        m = stream.Measure(number=i+1)

        for j in range(4):
            n = note.Note("C4", quarterLength=1.0)

            # Add fermata to some notes
            if j == 2:
                fermata = expressions.Fermata()
                n.expressions.append(fermata)

            m.append(n)

        part.append(m)

    s.append(part)
    return s

def export_to_png(musicxml_path, png_path, dpi=400):
    """Export MusicXML to PNG using MuseScore CLI."""

    cmd = [
        'mscore3',
        '-o', str(png_path),
        '--trim-image', '0',  # No trimming
        '-r', str(dpi),       # Resolution
        str(musicxml_path)
    ]

    subprocess.run(cmd, check=True)

# Generate and export
score = generate_fermata_score_music21()
musicxml_path = "output/score.musicxml"
score.write('musicxml', fp=musicxml_path)

export_to_png(musicxml_path, "output/score.png", dpi=400)
```

### Pros
✅ Familiar Python API for musicians
✅ Rich music theory library
✅ MusicXML export standard
✅ Integration with notation software

### Cons
❌ Requires MuseScore installation (large, ~200MB)
❌ No direct bounding box extraction
❌ Two-step process (music21 → MuseScore)
❌ MuseScore CLI limitations

### Estimated Effort
- **Setup**: 2-3 hours
- **Implementation**: 3-4 days
- **Total**: ~4-5 days

---

## Method 4: Font-Based Generation (SMuFL + Bravura)

### Overview
Render individual music symbols using SMuFL fonts (Bravura) on staff backgrounds.

### Resources
- **SMuFL Specification**: https://www.smufl.org/
- **Bravura Font**: https://github.com/steinbergmedia/bravura
- **SMuFL Metadata**: https://w3c.github.io/smufl/latest/

### Installation

```bash
# Download Bravura font
wget https://github.com/steinbergmedia/bravura/releases/download/bravura-1.392/bravura-1.392.zip
unzip bravura-1.392.zip

# Install font
sudo cp bravura-1.392/otf/Bravura.otf /usr/share/fonts/truetype/
fc-cache -f -v
```

### Code Example

```python
from PIL import Image, ImageDraw, ImageFont
import random

def draw_staff(img, draw, y_position, width):
    """Draw a 5-line staff."""
    line_spacing = 12  # pixels
    for i in range(5):
        y = y_position + i * line_spacing
        draw.line([(0, y), (width, y)], fill='black', width=2)

def draw_fermata(draw, x, y, font):
    """Draw a fermata symbol using Bravura font."""
    # SMuFL codepoint for fermata: U+E4C0
    fermata_char = '\uE4C0'
    draw.text((x, y), fermata_char, font=font, fill='black')

def generate_synthetic_image(output_path, num_fermatas=5):
    """Generate synthetic score image with fermatas."""

    width, height = 800, 400
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)

    # Load Bravura font
    font = ImageFont.truetype('/usr/share/fonts/truetype/Bravura.otf', size=48)

    # Draw staff
    staff_y = 100
    draw_staff(img, draw, staff_y, width)

    # Draw random fermatas
    bboxes = []
    for i in range(num_fermatas):
        x = random.randint(50, width - 50)
        y = staff_y + random.randint(-20, 60)

        draw_fermata(draw, x, y, font)

        # Record bounding box (approximate)
        bboxes.append({
            'class': 'fermata',
            'x': x,
            'y': y,
            'width': 40,
            'height': 40
        })

    img.save(output_path)
    return bboxes

# Generate batch
for i in range(1000):
    bboxes = generate_synthetic_image(f"output/font_based_{i:04d}.png")
```

### SMuFL Codepoints for Target Symbols

```python
SMUFL_SYMBOLS = {
    'fermata': '\uE4C0',           # Fermata above
    'fermata_below': '\uE4C1',     # Fermata below
    'barline': '\uE030',           # Single barline
    'barline_double': '\uE031',    # Double barline
    'barline_final': '\uE032',     # Final barline
    'double_sharp': '\uE263',      # Double sharp
    'double_flat': '\uE264',       # Double flat
    'natural': '\uE261',           # Natural
}
```

### Pros
✅ Very fast generation (pure Python + PIL)
✅ Full control over symbol placement
✅ No external dependencies (except font)
✅ Easy bounding box annotation
✅ Lightweight

### Cons
❌ Manual staff layout required
❌ Less realistic (no engraving engine)
❌ Need to handle complex music notation manually
❌ Font rendering quality depends on PIL

### Estimated Effort
- **Setup**: 1 day
- **Implementation**: 2-3 days
- **Total**: ~3-4 days

---

## Method 5: Domain Randomization & Data Augmentation

### Overview
Apply domain randomization techniques from OMR research to increase data diversity.

### Key Techniques

1. **Background Textures**: Add paper textures, noise
2. **Distortions**: Rotation, perspective transform, elastic deformation
3. **Lighting**: Shadows, brightness variations
4. **Staff Line Variations**: Spacing, thickness, curvature
5. **Font Variations**: Different music fonts (Bravura, Petaluma, etc.)

### Code Example

```python
import cv2
import numpy as np
from PIL import Image
import random

def apply_domain_randomization(img):
    """Apply domain randomization to synthetic score image."""

    # Convert PIL to OpenCV
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # 1. Add paper texture
    if random.random() < 0.5:
        texture = cv2.imread("textures/paper_01.jpg")
        texture = cv2.resize(texture, (img_cv.shape[1], img_cv.shape[0]))
        img_cv = cv2.addWeighted(img_cv, 0.7, texture, 0.3, 0)

    # 2. Random rotation
    if random.random() < 0.3:
        angle = random.uniform(-5, 5)
        h, w = img_cv.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        img_cv = cv2.warpAffine(img_cv, M, (w, h), borderValue=(255, 255, 255))

    # 3. Perspective transform
    if random.random() < 0.3:
        h, w = img_cv.shape[:2]
        pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        pts2 = pts1 + np.random.uniform(-20, 20, pts1.shape)
        M = cv2.getPerspectiveTransform(pts1, pts2)
        img_cv = cv2.warpPerspective(img_cv, M, (w, h), borderValue=(255, 255, 255))

    # 4. Gaussian blur (camera focus)
    if random.random() < 0.3:
        kernel_size = random.choice([3, 5])
        img_cv = cv2.GaussianBlur(img_cv, (kernel_size, kernel_size), 0)

    # 5. Brightness/contrast
    if random.random() < 0.5:
        alpha = random.uniform(0.8, 1.2)  # Contrast
        beta = random.randint(-20, 20)     # Brightness
        img_cv = cv2.convertScaleAbs(img_cv, alpha=alpha, beta=beta)

    # 6. Salt & pepper noise
    if random.random() < 0.2:
        noise = np.random.rand(*img_cv.shape[:2])
        img_cv[noise < 0.01] = 0
        img_cv[noise > 0.99] = 255

    # Convert back to PIL
    img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    return img_pil

def augment_batch(input_dir, output_dir, augmentations_per_image=5):
    """Apply augmentation to all images in directory."""

    from pathlib import Path

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    for img_file in input_path.glob("*.png"):
        img = Image.open(img_file)

        # Generate multiple augmented versions
        for i in range(augmentations_per_image):
            aug_img = apply_domain_randomization(img)
            aug_name = img_file.stem + f"_aug{i:02d}.png"
            aug_img.save(output_path / aug_name)

# Usage
augment_batch("synthetic_clean/", "synthetic_augmented/", augmentations_per_image=10)
```

### Additional Augmentation Strategies

Based on real-world OMR research:

1. **ScoreAug** (from Real World Music Object Recognition paper):
   - Use diverse real-world paper textures from IMSLP
   - Combine synthetic + real backgrounds
   - Random heuristics for variation

2. **Staff Line Deformations**:
   ```python
   def deform_staff_lines(img, wave_amplitude=2, wave_frequency=0.01):
       """Apply wave deformation to staff lines."""
       h, w = img.shape[:2]

       map_x = np.zeros((h, w), dtype=np.float32)
       map_y = np.zeros((h, w), dtype=np.float32)

       for i in range(h):
           for j in range(w):
               map_x[i, j] = j
               map_y[i, j] = i + wave_amplitude * np.sin(wave_frequency * j)

       return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)
   ```

3. **Shadow Simulation**:
   ```python
   def add_shadow(img, shadow_intensity=0.3):
       """Add random shadow pattern."""
       h, w = img.shape[:2]

       # Create gradient shadow
       shadow = np.zeros((h, w), dtype=np.float32)
       x = random.randint(0, w)
       y = random.randint(0, h)

       for i in range(h):
           for j in range(w):
               dist = np.sqrt((i - y)**2 + (j - x)**2)
               shadow[i, j] = np.exp(-dist / (w * 0.5))

       shadow = 1 - shadow * shadow_intensity

       img_float = img.astype(np.float32)
       img_shadowed = img_float * shadow[:, :, np.newaxis]

       return img_shadowed.astype(np.uint8)
   ```

### Pros
✅ Dramatically increases dataset diversity
✅ Improves model robustness to real-world conditions
✅ Can be applied to any synthetic generation method
✅ Research-proven effectiveness

### Cons
❌ Computationally intensive
❌ Need texture/background libraries
❌ Risk of over-augmentation (unrealistic images)

---

## Recommended Pipeline

### Step 1: Base Generation (Abjad + LilyPond)
Generate 2,000 clean synthetic images with:
- 1,000 images focusing on fermatas (5-10 per image)
- 1,000 images focusing on barlines (all types)

**Estimated time**: 4-6 hours (with parallelization)

### Step 2: Bounding Box Extraction
Parse LilyPond SVG output to extract precise coordinates:
- Use LilyPond's event-listener for symbol-level tracking
- Convert to YOLO format
- Validate annotations

**Estimated time**: 1-2 days

### Step 3: Domain Randomization
Apply 5-10 augmentations per base image:
- 2,000 base images × 10 augmentations = 20,000 images
- Focus on real-world variations (textures, rotations, lighting)

**Estimated time**: 2-4 hours (batch processing)

### Step 4: Font-Based Supplementation
Generate additional 1,000 images using Bravura font:
- Focus on isolated symbol detection
- Vary staff spacing, symbol size, font weight

**Estimated time**: 1 day

### Final Dataset
- **Total**: 23,000 images
- **Fermatas**: ~100,000 instances (5 per image avg)
- **Barlines**: ~100,000 instances (4-5 per image avg)
- **Format**: PNG (400 DPI) + YOLO annotations

---

## Implementation Roadmap

### Week 1: Setup & Base Generation
- [ ] Day 1-2: Install Abjad, LilyPond, test basic rendering
- [ ] Day 3-4: Implement fermata generation script
- [ ] Day 5-6: Implement barline generation script
- [ ] Day 7: Generate 2,000 base images

### Week 2: Bounding Box & Augmentation
- [ ] Day 1-3: Implement SVG parsing for bounding boxes
- [ ] Day 4-5: Implement domain randomization pipeline
- [ ] Day 6: Run augmentation on all base images (20,000 total)
- [ ] Day 7: Validate dataset, convert to YOLO format

### Week 3: Supplementation & Validation
- [ ] Day 1-2: Implement font-based generation (1,000 images)
- [ ] Day 3-4: Merge with Phase 4 dataset
- [ ] Day 5: Train test model to validate quality
- [ ] Day 6-7: Iterate based on validation results

---

## Code Repository Structure

```
training/
├── synthetic_generation/
│   ├── abjad_generator.py       # Main Abjad generation script
│   ├── bbox_extractor.py        # SVG → YOLO bounding box extraction
│   ├── augmentation.py          # Domain randomization pipeline
│   ├── font_generator.py        # SMuFL font-based generation
│   ├── templates/
│   │   ├── fermata_template.py
│   │   └── barline_template.py
│   ├── textures/                # Paper textures for augmentation
│   │   ├── paper_01.jpg
│   │   └── ...
│   └── configs/
│       ├── generation_config.yaml
│       └── augmentation_config.yaml
├── datasets/
│   └── yolo_synthetic_fermata_barline/
│       ├── images/
│       │   ├── train/
│       │   └── val/
│       ├── labels/
│       │   ├── train/
│       │   └── val/
│       └── data.yaml
└── scripts/
    ├── generate_batch.sh        # Batch generation script
    └── merge_with_phase4.py     # Merge with existing dataset
```

---

## Cost-Benefit Analysis

### Method Comparison

| Method | Setup Time | Gen Time (1k imgs) | Quality | Control | Bbox Accuracy |
|--------|-----------|-------------------|---------|---------|---------------|
| Abjad + LilyPond | 4h | 2-4h | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Verovio | 2h | 1-2h | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Music21 + MuseScore | 3h | 3-5h | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| Font-Based | 1h | 0.5h | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

### Recommended Hybrid Approach

**For fermatas**: Abjad + Domain Randomization
- Need realistic musical context
- Fermatas appear in varied positions

**For barlines**: Font-Based + Abjad
- More predictable positioning
- Faster generation for simple cases

---

## References & Resources

### Core Technologies
- [Abjad Documentation](https://abjad.github.io/)
- [LilyPond Manual](https://lilypond.org/doc/v2.25/Documentation/)
- [Verovio Reference Book](https://book.verovio.org/)
- [SMuFL Specification](https://www.smufl.org/)
- [Music21 Documentation](https://web.mit.edu/music21/)

### Research Papers
- [Real World Music Object Recognition](https://transactions.ismir.net/articles/10.5334/tismir.157) - ScoreAug technique
- [DeepScores Dataset](https://arxiv.org/pdf/1804.00525) - Synthetic score generation
- [Domain Randomization for Object Detection](https://arxiv.org/abs/2506.07539)

### Datasets & Tools
- [OMR-Datasets Collection](https://github.com/apacha/OMR-Datasets)
- [Audiveris OMR Dataset Tools](https://github.com/Audiveris/omr-dataset-tools)
- [Bravura Font Repository](https://github.com/steinbergmedia/bravura)

### Additional Resources
- [MUSCIMA++ Tutorial](https://muscima.readthedocs.io/en/latest/Tutorial.html)
- [python-ly Documentation](https://python-ly.readthedocs.io/)
- [MuseScore CLI Guide](https://musescore.org/en/node/342132)

---

## Next Steps

1. **Immediate**: Set up Abjad + LilyPond environment
2. **Week 1**: Implement base generation pipeline
3. **Week 2**: Add bounding box extraction and augmentation
4. **Week 3**: Validate and merge with Phase 4 dataset
5. **Future**: Implement continuous generation pipeline for ongoing model improvement

---

**Questions or need help implementing any of these methods?** Refer to the code examples above or consult the linked documentation resources.
