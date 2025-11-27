#!/usr/bin/env python3
"""
Generate Synthetic Barline Training Data

This script generates synthetic barline images using simple drawing techniques
combined with domain randomization to create diverse training samples.

Generates:
  - Single barlines (class 23)
  - Double barlines (class 24)
  - Final barlines (heavy-light, class 25)
  - Repeat barlines (with dots, class 26)

Output: /home/thc1006/dev/music-app/training/datasets/synthetic_barlines_yolo
"""

import os
import random
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter
from tqdm import tqdm


# Configuration
OUTPUT_DIR = Path("/home/thc1006/dev/music-app/training/datasets/synthetic_barlines_yolo")
NUM_SAMPLES_PER_CLASS = 500  # Generate 500 samples per barline type

# Image parameters
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 640
STAFF_LINE_HEIGHT = 8  # pixels between staff lines
STAFF_LINES = 5

# Barline parameters
THIN_BARLINE_WIDTH = 2
THICK_BARLINE_WIDTH = 6
DOUBLE_BARLINE_GAP = 8
REPEAT_DOT_RADIUS = 4


def create_staff(draw, y_offset, width):
    """Draw a 5-line staff."""
    staff_color = random.randint(0, 40)  # Slight variation in darkness

    for i in range(STAFF_LINES):
        y = y_offset + i * STAFF_LINE_HEIGHT
        # Add slight thickness variation
        thickness = random.randint(1, 2)
        draw.line([(0, y), (width, y)], fill=staff_color, width=thickness)


def add_paper_texture(image):
    """Add realistic paper texture to the image."""
    # Convert to numpy for noise
    img_array = np.array(image, dtype=np.float32)

    # Add Gaussian noise
    noise = np.random.normal(250, 10, img_array.shape)
    img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)

    # Create paper texture pattern
    texture = Image.new('L', image.size, 255)
    texture_array = np.array(texture, dtype=np.float32)
    texture_noise = np.random.normal(0, 5, texture_array.shape)
    texture_array = np.clip(texture_array + texture_noise, 240, 255).astype(np.uint8)

    # Blend
    img_array = np.minimum(img_array, texture_array)

    return Image.fromarray(img_array)


def add_domain_randomization(image):
    """Apply domain randomization effects."""
    effects = []

    # Random blur
    if random.random() < 0.3:
        image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
        effects.append("blur")

    # Random brightness
    if random.random() < 0.5:
        enhancer = random.uniform(0.8, 1.2)
        img_array = np.array(image, dtype=np.float32)
        img_array = np.clip(img_array * enhancer, 0, 255).astype(np.uint8)
        image = Image.fromarray(img_array)
        effects.append("brightness")

    # Small rotation
    if random.random() < 0.3:
        angle = random.uniform(-2, 2)
        image = image.rotate(angle, fillcolor=255, expand=False)
        effects.append(f"rotate {angle:.1f}°")

    return image


def generate_single_barline(image_id):
    """Generate a single thin barline."""
    # Create blank image
    img = Image.new('L', (IMAGE_WIDTH, IMAGE_HEIGHT), 255)
    draw = ImageDraw.Draw(img)

    # Random position for staff
    staff_y = random.randint(100, IMAGE_HEIGHT - 200)

    # Draw staff
    create_staff(draw, staff_y, IMAGE_WIDTH)

    # Draw single barline at random x position
    barline_x = random.randint(150, IMAGE_WIDTH - 150)
    barline_top = staff_y
    barline_bottom = staff_y + (STAFF_LINES - 1) * STAFF_LINE_HEIGHT

    # Slight height variation
    barline_top -= random.randint(0, 3)
    barline_bottom += random.randint(0, 3)

    # Draw barline
    draw.rectangle(
        [barline_x, barline_top, barline_x + THIN_BARLINE_WIDTH, barline_bottom],
        fill=0
    )

    # Add effects
    img = add_paper_texture(img)
    img = add_domain_randomization(img)

    # Calculate YOLO bbox
    x_center = (barline_x + THIN_BARLINE_WIDTH / 2) / IMAGE_WIDTH
    y_center = (barline_top + barline_bottom) / 2 / IMAGE_HEIGHT
    width = (THIN_BARLINE_WIDTH + 4) / IMAGE_WIDTH  # Add small margin
    height = (barline_bottom - barline_top) / IMAGE_HEIGHT

    label = f"23 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

    return img, label


def generate_double_barline(image_id):
    """Generate a double barline (two thin lines)."""
    img = Image.new('L', (IMAGE_WIDTH, IMAGE_HEIGHT), 255)
    draw = ImageDraw.Draw(img)

    staff_y = random.randint(100, IMAGE_HEIGHT - 200)
    create_staff(draw, staff_y, IMAGE_WIDTH)

    barline_x = random.randint(150, IMAGE_WIDTH - 150)
    barline_top = staff_y - random.randint(0, 3)
    barline_bottom = staff_y + (STAFF_LINES - 1) * STAFF_LINE_HEIGHT + random.randint(0, 3)

    # Draw two thin lines
    draw.rectangle(
        [barline_x, barline_top, barline_x + THIN_BARLINE_WIDTH, barline_bottom],
        fill=0
    )
    draw.rectangle(
        [barline_x + DOUBLE_BARLINE_GAP, barline_top,
         barline_x + DOUBLE_BARLINE_GAP + THIN_BARLINE_WIDTH, barline_bottom],
        fill=0
    )

    img = add_paper_texture(img)
    img = add_domain_randomization(img)

    # Calculate bbox (encompassing both lines)
    bbox_left = barline_x
    bbox_right = barline_x + DOUBLE_BARLINE_GAP + THIN_BARLINE_WIDTH

    x_center = (bbox_left + bbox_right) / 2 / IMAGE_WIDTH
    y_center = (barline_top + barline_bottom) / 2 / IMAGE_HEIGHT
    width = (bbox_right - bbox_left + 4) / IMAGE_WIDTH
    height = (barline_bottom - barline_top) / IMAGE_HEIGHT

    label = f"24 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

    return img, label


def generate_final_barline(image_id):
    """Generate a final barline (thin + thick)."""
    img = Image.new('L', (IMAGE_WIDTH, IMAGE_HEIGHT), 255)
    draw = ImageDraw.Draw(img)

    staff_y = random.randint(100, IMAGE_HEIGHT - 200)
    create_staff(draw, staff_y, IMAGE_WIDTH)

    barline_x = random.randint(150, IMAGE_WIDTH - 150)
    barline_top = staff_y - random.randint(0, 3)
    barline_bottom = staff_y + (STAFF_LINES - 1) * STAFF_LINE_HEIGHT + random.randint(0, 3)

    # Draw thin line (left)
    draw.rectangle(
        [barline_x, barline_top, barline_x + THIN_BARLINE_WIDTH, barline_bottom],
        fill=0
    )

    # Draw thick line (right)
    thick_x = barline_x + DOUBLE_BARLINE_GAP
    draw.rectangle(
        [thick_x, barline_top, thick_x + THICK_BARLINE_WIDTH, barline_bottom],
        fill=0
    )

    img = add_paper_texture(img)
    img = add_domain_randomization(img)

    bbox_left = barline_x
    bbox_right = thick_x + THICK_BARLINE_WIDTH

    x_center = (bbox_left + bbox_right) / 2 / IMAGE_WIDTH
    y_center = (barline_top + barline_bottom) / 2 / IMAGE_HEIGHT
    width = (bbox_right - bbox_left + 4) / IMAGE_WIDTH
    height = (barline_bottom - barline_top) / IMAGE_HEIGHT

    label = f"25 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

    return img, label


def generate_repeat_barline(image_id):
    """Generate a repeat barline (with dots)."""
    img = Image.new('L', (IMAGE_WIDTH, IMAGE_HEIGHT), 255)
    draw = ImageDraw.Draw(img)

    staff_y = random.randint(100, IMAGE_HEIGHT - 200)
    create_staff(draw, staff_y, IMAGE_WIDTH)

    barline_x = random.randint(150, IMAGE_WIDTH - 150)
    barline_top = staff_y - random.randint(0, 3)
    barline_bottom = staff_y + (STAFF_LINES - 1) * STAFF_LINE_HEIGHT + random.randint(0, 3)

    # Draw thick + thin barlines
    draw.rectangle(
        [barline_x, barline_top, barline_x + THICK_BARLINE_WIDTH, barline_bottom],
        fill=0
    )

    thin_x = barline_x + THICK_BARLINE_WIDTH + DOUBLE_BARLINE_GAP
    draw.rectangle(
        [thin_x, barline_top, thin_x + THIN_BARLINE_WIDTH, barline_bottom],
        fill=0
    )

    # Draw repeat dots (in spaces 2 and 3 of staff)
    dot_x = thin_x + 10
    dot_y1 = staff_y + STAFF_LINE_HEIGHT * 1.5
    dot_y2 = staff_y + STAFF_LINE_HEIGHT * 2.5

    draw.ellipse(
        [dot_x - REPEAT_DOT_RADIUS, dot_y1 - REPEAT_DOT_RADIUS,
         dot_x + REPEAT_DOT_RADIUS, dot_y1 + REPEAT_DOT_RADIUS],
        fill=0
    )
    draw.ellipse(
        [dot_x - REPEAT_DOT_RADIUS, dot_y2 - REPEAT_DOT_RADIUS,
         dot_x + REPEAT_DOT_RADIUS, dot_y2 + REPEAT_DOT_RADIUS],
        fill=0
    )

    img = add_paper_texture(img)
    img = add_domain_randomization(img)

    bbox_left = barline_x
    bbox_right = dot_x + REPEAT_DOT_RADIUS + 2

    x_center = (bbox_left + bbox_right) / 2 / IMAGE_WIDTH
    y_center = (barline_top + barline_bottom) / 2 / IMAGE_HEIGHT
    width = (bbox_right - bbox_left + 4) / IMAGE_WIDTH
    height = (barline_bottom - barline_top) / IMAGE_HEIGHT

    label = f"26 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

    return img, label


def generate_dataset():
    """Generate complete synthetic barline dataset."""
    print("=" * 70)
    print("Synthetic Barline Generator")
    print("=" * 70)

    # Create output directories
    (OUTPUT_DIR / "train" / "images").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "train" / "labels").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "val" / "images").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "val" / "labels").mkdir(parents=True, exist_ok=True)

    generators = [
        ("barline", generate_single_barline, 23),
        ("barline_double", generate_double_barline, 24),
        ("barline_final", generate_final_barline, 25),
        ("barline_repeat", generate_repeat_barline, 26),
    ]

    stats = {name: 0 for name, _, _ in generators}
    total_generated = 0

    for class_name, generator_func, class_id in generators:
        print(f"\nGenerating {NUM_SAMPLES_PER_CLASS} samples for {class_name}...")

        for i in tqdm(range(NUM_SAMPLES_PER_CLASS), desc=f"  {class_name}"):
            # Generate image and label
            img, label = generator_func(total_generated)

            # Determine split (90/10)
            is_train = (i % 10) != 0
            split = "train" if is_train else "val"

            # Save image
            img_filename = f"synthetic_{class_name}_{i:04d}.png"
            img_path = OUTPUT_DIR / split / "images" / img_filename
            img.save(img_path)

            # Save label
            label_path = OUTPUT_DIR / split / "labels" / f"synthetic_{class_name}_{i:04d}.txt"
            with open(label_path, 'w') as f:
                f.write(label)

            stats[class_name] += 1
            total_generated += 1

    # Final statistics
    print("\n" + "=" * 70)
    print("Generation Complete")
    print("=" * 70)
    print(f"Total images generated: {total_generated}")

    print("\nSamples by class:")
    for class_name, count in stats.items():
        print(f"  {class_name}: {count}")

    # Count final dataset
    train_images = len(list((OUTPUT_DIR / "train" / "images").glob("*.png")))
    val_images = len(list((OUTPUT_DIR / "val" / "images").glob("*.png")))

    print(f"\nDataset split:")
    print(f"  Train: {train_images} images")
    print(f"  Val:   {val_images} images")

    # Create YAML
    create_dataset_yaml(stats)

    print("\n" + "=" * 70)
    print("Next Steps:")
    print("=" * 70)
    print("1. Review generated samples visually")
    print("2. Merge with MUSCIMA barlines:")
    print("   python /home/thc1006/dev/music-app/training/scripts/merge_barline_datasets.py")


def create_dataset_yaml(stats):
    """Create YOLO dataset configuration file."""
    yaml_content = f"""# Synthetic Barlines Dataset (YOLO Format)
# Generated using domain randomization

path: {OUTPUT_DIR}
train: train/images
val: val/images

nc: 4
names:
  0: barline
  1: barline_double
  2: barline_final
  3: barline_repeat

# Class mapping (to merge with harmony_phase5.yaml):
# 23: barline
# 24: barline_double
# 25: barline_final
# 26: barline_repeat

# Statistics:
# barline: {stats.get('barline', 0)}
# barline_double: {stats.get('barline_double', 0)}
# barline_final: {stats.get('barline_final', 0)}
# barline_repeat: {stats.get('barline_repeat', 0)}
"""

    yaml_path = OUTPUT_DIR / "synthetic_barlines.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"\n✓ Created dataset config: {yaml_path}")


if __name__ == "__main__":
    try:
        random.seed(42)  # For reproducibility
        np.random.seed(42)
        generate_dataset()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
