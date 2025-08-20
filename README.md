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
  --samples samples \
  --out artifacts \
  --width 1920 \
  --align_tol 12 \
  --ratio_tol 0.15 \
  --min_box 20 \
  --overlay 1 \
  --save_crops 1 \
  --topk 5 \
  --use_ssim_focus 0
```

## Output Files

### 1. `artifacts/pair_X/issues.json`
JSON file containing detected issues with enhanced metadata:

```json
[
  {
    "id": "1",
    "priority": "P0",
    "type": "overlap", 
    "bbox": [x, y, width, height],
    "section": "header",
    "score": 0.85,
    "hint": "Text element at (x,y) overlaps with element at (other_x,other_y), suggest using grid/flex/gap/min-height/wrap instead of z-index",
    "overlap_ratio": 0.85,
    "iou": 0.75
  }
]
```

### 2. `artifacts/pair_X/issues.csv`
CSV report with tabular format for easy analysis.

### 3. `artifacts/pair_X/overlay.png`
Visual overlay showing detected issues with color-coded bounding boxes:
- **Red (P0)**: Critical overlap issues
- **Orange (P1)**: Important alignment/spacing issues  
- **Blue (P2)**: Minor proportion issues

### 4. `artifacts/pair_X/crops/`
Directory containing cropped images of each detected issue for detailed inspection.

### 5. `artifacts/pair_X/suggestion_prompt.txt`
English prompt for frontend AI with Top-K limiting and section-based organization:

```
【Objective】Focus on readability and human-like layout rather than pixel-perfect accuracy.

【P1 Important Issues - Alignment & Spacing】

HEADER section:
  Spacing issues (5):
    • Element at (x,y) has inconsistent spacing, suggest using unified 8px or 16px spacing system
    • ... and 3 more spacing issues

Recommended actions: Align to unified left/right edges, standardize card/button styles (border-radius, padding), unify heading hierarchy

【Issue Summary】
• P0 (Critical): 0 overlap issues
• P1 (Important): 17 alignment/spacing issues
• P2 (Minor): 0 aspect ratio issues
• Total: 17 issues to address
• Showing top 5 issues per section

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
| `--thin_px` | 4 | Minimum side length for thin elements |
| `--min_area` | 1500 | Minimum area for elements |
| `--overlay` | 1 | Generate overlay visualization (0/1) |
| `--save_crops` | 1 | Save issue crops (0/1) |
| `--crop_size` | 160 | Crop size for issue visualization |
| `--topk` | 5 | Top-K issues to show per section |
| `--use_ssim_focus` | 0 | Use SSIM hotspots for scoring (0/1) |

## Detection Rules

### P0 (Critical) - Overlaps
- **Enhanced overlap detection**: Only reports overlaps that affect readability
- **Text-involved overlaps**: Overlap ratio > 30% when text is involved
- **Large element overlaps**: Overlap ratio > 50% for large elements (area > 3000px²)
- **Robust filtering**: Uses quantized coordinates and NMS to reduce false positives
- **Suggests**: grid/flex/gap/min-height/wrap instead of z-index

### P1 (Important) - Alignment & Spacing  
- **Adaptive row tolerance**: Based on median box height (6-20px range)
- **Content edge estimation**: Automatically detects main content boundaries
- **Class-aware grouping**: Separates text, media, and icon elements
- **IQR-based spacing**: Uses interquartile range for robust variance detection
- **Suggests**: Unified grid system and consistent spacing

### P2 (Minor) - Proportions
- **Media-only analysis**: Only applies to media elements (area ≥ 4000px², ratio 0.5-2.0)
- **Group-based comparison**: Compares within same element class
- **Deviation threshold**: >10% from group mean aspect ratio
- **Suggests**: Unified aspect ratios with object-fit: cover

### Enhanced Features
- **Box preprocessing**: Quantization, deduplication, NMS, filtering
- **Element classification**: text, media, icon, other
- **Section identification**: header, main_content_top, main_content_bottom, footer
- **SSIM hotspots**: Optional structural similarity analysis for focus scoring

## Sample Files

Place your image pairs in the `samples/` directory:
- `samples/1_design.png` + `samples/1_page.png` - First pair
- `samples/2_design.png` + `samples/2_page.png` - Second pair
- `samples/N_design.png` + `samples/N_page.png` - Nth pair

The script will automatically detect and process all pairs.

## Notes

- **OCR Dependency**: If pytesseract is not installed, text detection is skipped but the script continues to work
- **Image Formats**: Supports common formats (PNG, JPG, etc.)
- **Performance**: Processing time depends on image size and complexity
- **Accuracy**: Enhanced with robust preprocessing to reduce false positives
- **Visualization**: Generates overlay images and issue crops for better analysis
- **Batch Processing**: Automatically processes multiple image pairs
- **Top-K Limiting**: Configurable limit on number of issues shown per section

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
