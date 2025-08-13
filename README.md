# Layout Comparison Tool

A minimal MVP tool for comparing design mockups with implementation screenshots to detect layout issues.

## Features

- **P0 Issues (Critical)**: Detect overlapping elements, especially text overlapping with images
- **P1 Issues (Important)**: Detect alignment and spacing inconsistencies  
- **P2 Issues (Minor)**: Detect proportion inconsistencies in images/cards
- **Optional OCR**: Text detection using pytesseract (gracefully degrades if not installed)
- **Configurable thresholds**: Adjustable parameters for different use cases

## Installation

### 1. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Optional: Install Tesseract OCR
For enhanced text detection (optional):
- **macOS**: `brew install tesseract`
- **Ubuntu**: `sudo apt-get install tesseract-ocr`
- **Windows**: Download from https://github.com/UB-Mannheim/tesseract/wiki

## Usage

### Basic Usage
```bash
python compare_layout.py --design samples/design.png --impl samples/page.png --out artifacts
```

### Advanced Usage with Custom Parameters
```bash
python compare_layout.py \
  --design samples/design.png \
  --impl samples/page.png \
  --out artifacts \
  --width 1920 \
  --align_tol 12 \
  --ratio_tol 0.15 \
  --min_box 20
```

## Output Files

### 1. `artifacts/issues.json`
JSON file containing detected issues grouped by priority:

```json
[
  {
    "priority": "P0",
    "type": "overlap", 
    "bbox": [x, y, width, height],
    "hint": "Text and image overlap, suggest increasing container min-height or gap"
  },
  {
    "priority": "P1",
    "type": "misalign",
    "bbox": [x, y, width, height], 
    "hint": "Left edge not aligned, suggest aligning to main grid (tolerance 8px)"
  }
]
```

### 2. `artifacts/suggestion_prompt.txt`
English prompt for frontend AI with prioritized suggestions:

```
【Objective】Focus on readability and human-like layout rather than pixel-perfect accuracy.

【P0 Critical Issues】
• Text and image overlap, suggest increasing container min-height or gap
Recommended actions: Avoid z-index overlays, use grid/flex/gap/min-height/wrapping instead

【P1 Important Issues】  
• Left edge not aligned, suggest aligning to main grid (tolerance 8px)
Recommended actions: Align to unified left/right edges, standardize card/button styles (border-radius, padding), unify heading hierarchy

【Acceptance Criteria】
• No overlapping elements
• Left/right alignment tolerance <= 8px
• No horizontal scrolling on mobile
• Consistent spacing
• Unified image proportions
```

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--width` | 1280 | Target width for image resizing |
| `--align_tol` | 8 | Alignment tolerance in pixels |
| `--ratio_tol` | 0.1 | Ratio tolerance (10%) |
| `--min_box` | 16 | Minimum box size to consider |

## Detection Rules

### P0 (Critical) - Overlaps
- IoU > 0.2 between any two boxes
- IoU > 0.1 if text is involved (requires pytesseract)
- Suggests using grid/flex/gap instead of z-index

### P1 (Important) - Alignment & Spacing  
- Left/center/right edges not aligned within tolerance
- Inconsistent spacing between adjacent elements (>40% variance)
- Suggests unified grid system and consistent spacing

### P2 (Minor) - Proportions
- Aspect ratio deviation >10% from group mean
- Only applies to larger elements (area > 1000px²)
- Suggests unified aspect ratios with object-fit

## Sample Files

Place your images in the `samples/` directory:
- `samples/design.png` - Design mockup
- `samples/page.png` - Implementation screenshot

## Notes

- **OCR Dependency**: If pytesseract is not installed, text detection is skipped but the script continues to work
- **Image Formats**: Supports common formats (PNG, JPG, etc.)
- **Performance**: Processing time depends on image size and complexity
- **Accuracy**: This is a minimal MVP - results may need manual review for complex layouts

## Troubleshooting

### Common Issues

1. **"Cannot read image"**: Check file path and image format
2. **No issues detected**: Try reducing `--min_box` or `--align_tol` values
3. **Too many false positives**: Increase tolerance values
4. **OCR not working**: Install tesseract or ignore OCR-related warnings

### Performance Tips

- Use smaller images for faster processing
- Adjust `--min_box` to filter noise
- Increase tolerances for rough comparisons
