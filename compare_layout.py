# compare_layout.py developed by Bob Tianqi Wei

import cv2
import numpy as np
import json
import argparse
import os
from PIL import Image
from skimage import measure
from typing import List, Dict, Tuple, Optional
import math

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

class LayoutComparator:
    def __init__(self, width: int = 1280, align_tol: int = 8, ratio_tol: float = 0.1, min_box: int = 16):
        self.width = width
        self.align_tol = align_tol
        self.ratio_tol = ratio_tol
        self.min_box = min_box
        
    def identify_sections(self, boxes: List[Tuple[int, int, int, int]], image_height: int) -> Dict[int, str]:
        """Identify page sections based on vertical position"""
        if not boxes:
            return {}
        
        # Sort boxes by y position
        sorted_boxes = sorted(boxes, key=lambda box: box[1])
        
        # Define section boundaries (top 20%, next 30%, next 30%, bottom 20%)
        section_boundaries = [
            (0, int(image_height * 0.2), "header"),
            (int(image_height * 0.2), int(image_height * 0.5), "main_content_top"),
            (int(image_height * 0.5), int(image_height * 0.8), "main_content_bottom"),
            (int(image_height * 0.8), image_height, "footer")
        ]
        
        section_map = {}
        for i, (x, y, w, h) in enumerate(boxes):
            section_name = "unknown"
            for start_y, end_y, name in section_boundaries:
                if start_y <= y < end_y:
                    section_name = name
                    break
            section_map[i] = section_name
        
        return section_map
        
    def resize_image(self, image_path: str) -> np.ndarray:
        """Resize image to target width while maintaining aspect ratio"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        height, width = img.shape[:2]
        scale = self.width / width
        new_height = int(height * scale)
        
        resized = cv2.resize(img, (self.width, new_height))
        return resized
    
    def extract_contours(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Extract bounding boxes from image using contour detection"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Try multiple thresholding methods
        boxes = []
        
        # Method 1: Adaptive threshold
        thresh1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 11, 2)
        
        # Method 2: Otsu threshold
        _, thresh2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Method 3: Canny edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Process each method
        for thresh in [thresh1, thresh2, edges]:
            # Morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter and add bounding boxes
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w >= self.min_box and h >= self.min_box:
                    boxes.append((x, y, w, h))
        
        # Remove duplicates (boxes with high overlap)
        unique_boxes = []
        for box in boxes:
            is_duplicate = False
            for existing_box in unique_boxes:
                iou = self.calculate_iou(box, existing_box)
                if iou > 0.8:  # High overlap threshold for duplicates
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_boxes.append(box)
        
        return unique_boxes
    
    def calculate_iou(self, box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
        """Calculate Intersection over Union between two bounding boxes"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Calculate intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        union = w1 * h1 + w2 * h2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def detect_text_boxes(self, image: np.ndarray, boxes: List[Tuple[int, int, int, int]]) -> List[int]:
        """Detect which boxes contain text using OCR (if available)"""
        if not TESSERACT_AVAILABLE:
            return []
        
        text_boxes = []
        for i, (x, y, w, h) in enumerate(boxes):
            # Extract ROI
            roi = image[y:y+h, x:x+w]
            if roi.size == 0:
                continue
                
            try:
                # OCR the region
                text = pytesseract.image_to_string(roi, config='--psm 6')
                if text.strip():  # If text is found
                    text_boxes.append(i)
            except:
                continue
        
        return text_boxes
    
    def detect_p0_overlaps(self, boxes: List[Tuple[int, int, int, int]], 
                          text_boxes: List[int]) -> List[Dict]:
        """Detect P0 overlap issues"""
        issues = []
        
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                iou = self.calculate_iou(boxes[i], boxes[j])
                
                # Check for overlaps
                if iou > 0.2:
                    # Check if one is text box
                    is_text_involved = (i in text_boxes or j in text_boxes)
                    threshold = 0.1 if is_text_involved else 0.2
                    
                    if iou > threshold:
                        x1, y1, w1, h1 = boxes[i]
                        x2, y2, w2, h2 = boxes[j]
                        
                        # Determine which element is the problem element (usually the smaller one)
                        area1 = w1 * h1
                        area2 = w2 * h2
                        problem_box = boxes[i] if area1 <= area2 else boxes[j]
                        other_box = boxes[j] if area1 <= area2 else boxes[i]
                        
                        x, y, w, h = problem_box
                        other_x, other_y, other_w, other_h = other_box
                        
                        # Create more descriptive hint
                        if is_text_involved:
                            hint = f"Text element at ({x},{y}) overlaps with element at ({other_x},{other_y}), suggest increasing container min-height or gap"
                        else:
                            hint = f"Element at ({x},{y}) overlaps with element at ({other_x},{other_y}), suggest adjusting layout or increasing spacing"
                        
                        issues.append({
                            "priority": "P0",
                            "type": "overlap",
                            "bbox": [x, y, w, h],
                            "hint": hint,
                            "overlap_with": [other_x, other_y, other_w, other_h],
                            "iou": round(iou, 3)
                        })
        
        return issues
    
    def detect_p1_alignment(self, boxes: List[Tuple[int, int, int, int]]) -> List[Dict]:
        """Detect P1 alignment issues"""
        issues = []
        
        if len(boxes) < 2:
            return issues
        
        # Collect alignment points
        left_points = [box[0] for box in boxes]
        center_points = [box[0] + box[2] // 2 for box in boxes]
        right_points = [box[0] + box[2] for box in boxes]
        
        # Check left alignment
        left_buckets = self._bucket_points(left_points, self.align_tol)
        for i, left in enumerate(left_points):
            if not self._is_aligned(left, left_buckets, self.align_tol):
                x, y, w, h = boxes[i]
                # Find the closest aligned position
                closest_aligned = min(left_buckets, key=lambda bucket: abs(bucket - left))
                offset = left - closest_aligned
                
                issues.append({
                    "priority": "P1",
                    "type": "misalign",
                    "bbox": [x, y, w, h],
                    "hint": f"Element at ({x},{y}) left edge not aligned, offset by {offset}px from grid at {closest_aligned}px, suggest aligning to main grid (tolerance {self.align_tol}px)",
                    "current_position": left,
                    "suggested_position": closest_aligned,
                    "offset": offset
                })
        
        # Check spacing consistency
        spacing_issues = self._check_spacing_consistency(boxes)
        issues.extend(spacing_issues)
        
        return issues
    
    def _bucket_points(self, points: List[int], tolerance: int) -> List[int]:
        """Group points into buckets based on tolerance"""
        if not points:
            return []
        
        buckets = []
        sorted_points = sorted(points)
        
        current_bucket = [sorted_points[0]]
        for point in sorted_points[1:]:
            if abs(point - current_bucket[0]) <= tolerance:
                current_bucket.append(point)
            else:
                buckets.append(sum(current_bucket) // len(current_bucket))
                current_bucket = [point]
        
        if current_bucket:
            buckets.append(sum(current_bucket) // len(current_bucket))
        
        return buckets
    
    def _is_aligned(self, point: int, buckets: List[int], tolerance: int) -> bool:
        """Check if point is aligned with any bucket"""
        for bucket_center in buckets:
            if abs(point - bucket_center) <= tolerance:
                return True
        return False
    
    def _check_spacing_consistency(self, boxes: List[Tuple[int, int, int, int]]) -> List[Dict]:
        """Check spacing consistency between adjacent boxes in same horizontal band"""
        issues = []
        
        # Group boxes by horizontal bands
        bands = {}
        for i, (x, y, w, h) in enumerate(boxes):
            band_key = y // 24  # 24px tolerance for same horizontal band
            if band_key not in bands:
                bands[band_key] = []
            bands[band_key].append((i, x, y, w, h))
        
        # Check spacing in each band
        for band_boxes in bands.values():
            if len(band_boxes) < 2:
                continue
            
            # Sort by x position
            band_boxes.sort(key=lambda x: x[1])
            
            # Calculate spacings
            spacings = []
            for j in range(len(band_boxes) - 1):
                current_right = band_boxes[j][1] + band_boxes[j][3]
                next_left = band_boxes[j + 1][1]
                spacing = next_left - current_right
                if spacing > 0:
                    spacings.append(spacing)
            
            if len(spacings) >= 2:
                mean_spacing = sum(spacings) / len(spacings)
                variance = sum((s - mean_spacing) ** 2 for s in spacings) / len(spacings)
                
                if variance > (mean_spacing * 0.4) ** 2:  # 40% tolerance
                    for i, x, y, w, h in band_boxes:
                        issues.append({
                            "priority": "P1",
                            "type": "spacing",
                            "bbox": [x, y, w, h],
                            "hint": f"Element at ({x},{y}) has inconsistent spacing in row at y={y}px, current spacing variance is {variance:.1f}px, suggest using unified 8px or 16px spacing system",
                            "row_y": y,
                            "spacing_variance": round(variance, 1),
                            "mean_spacing": round(mean_spacing, 1)
                        })
        
        return issues
    
    def detect_p2_proportions(self, boxes: List[Tuple[int, int, int, int]]) -> List[Dict]:
        """Detect P2 proportion issues"""
        issues = []
        
        # Filter for potential image/card elements (larger areas)
        min_area = 1000  # Minimum area to consider as image/card
        image_boxes = [(i, x, y, w, h) for i, (x, y, w, h) in enumerate(boxes) 
                      if w * h >= min_area]
        
        if len(image_boxes) < 2:
            return issues
        
        # Calculate aspect ratios
        ratios = []
        for _, x, y, w, h in image_boxes:
            ratio = w / h if h > 0 else 0
            ratios.append(ratio)
        
        # Calculate mean ratio
        mean_ratio = sum(ratios) / len(ratios)
        
        # Check for outliers
        for i, (box_idx, x, y, w, h) in enumerate(image_boxes):
            ratio = ratios[i]
            if abs(ratio - mean_ratio) / mean_ratio > self.ratio_tol:
                issues.append({
                    "priority": "P2",
                    "type": "ratio",
                    "bbox": [x, y, w, h],
                    "hint": f"Element at ({x},{y}) has aspect ratio {ratio:.2f}:1, differs from group mean {mean_ratio:.2f}:1 by {abs(ratio - mean_ratio) / mean_ratio * 100:.1f}%, suggest unifying to {mean_ratio:.2f}:1 + object-fit: cover",
                    "current_ratio": round(ratio, 2),
                    "group_mean_ratio": round(mean_ratio, 2),
                    "deviation_percent": round(abs(ratio - mean_ratio) / mean_ratio * 100, 1)
                })
        
        return issues
    
    def generate_suggestion_prompt(self, issues: List[Dict]) -> str:
        """Generate English suggestion prompt for frontend AI"""
        prompt = "【Objective】Focus on readability and human-like layout rather than pixel-perfect accuracy.\n\n"
        
        # Group issues by priority and section
        p0_issues = [issue for issue in issues if issue["priority"] == "P0"]
        p1_issues = [issue for issue in issues if issue["priority"] == "P1"]
        p2_issues = [issue for issue in issues if issue["priority"] == "P2"]
        
        # Group by section
        def group_by_section(issue_list):
            sections = {}
            for issue in issue_list:
                section = issue.get("section", "unknown")
                if section not in sections:
                    sections[section] = []
                sections[section].append(issue)
            return sections
        
        # P0 issues by section
        if p0_issues:
            prompt += "【P0 Critical Issues - Element Overlaps】\n"
            p0_sections = group_by_section(p0_issues)
            for section, section_issues in p0_sections.items():
                prompt += f"\n{section.upper()} section:\n"
                for issue in section_issues[:5]:  # Limit to 5 issues per section
                    prompt += f"• {issue['hint']}\n"
                if len(section_issues) > 5:
                    prompt += f"• ... and {len(section_issues) - 5} more overlap issues\n"
            prompt += "\nRecommended actions: Avoid z-index overlays, use grid/flex/gap/min-height/wrapping instead\n\n"
        
        # P1 issues by section
        if p1_issues:
            prompt += "【P1 Important Issues - Alignment & Spacing】\n"
            p1_sections = group_by_section(p1_issues)
            for section, section_issues in p1_sections.items():
                prompt += f"\n{section.upper()} section:\n"
                # Group by type
                spacing_issues = [i for i in section_issues if i["type"] == "spacing"]
                align_issues = [i for i in section_issues if i["type"] == "misalign"]
                
                if spacing_issues:
                    prompt += f"  Spacing issues ({len(spacing_issues)}):\n"
                    for issue in spacing_issues[:3]:
                        prompt += f"    • {issue['hint']}\n"
                    if len(spacing_issues) > 3:
                        prompt += f"    • ... and {len(spacing_issues) - 3} more spacing issues\n"
                
                if align_issues:
                    prompt += f"  Alignment issues ({len(align_issues)}):\n"
                    for issue in align_issues[:3]:
                        prompt += f"    • {issue['hint']}\n"
                    if len(align_issues) > 3:
                        prompt += f"    • ... and {len(align_issues) - 3} more alignment issues\n"
            
            prompt += "\nRecommended actions: Align to unified left/right edges, standardize card/button styles (border-radius, padding), unify heading hierarchy\n\n"
        
        # P2 issues by section
        if p2_issues:
            prompt += "【P2 Minor Issues - Aspect Ratios】\n"
            p2_sections = group_by_section(p2_issues)
            for section, section_issues in p2_sections.items():
                prompt += f"\n{section.upper()} section:\n"
                for issue in section_issues[:3]:  # Limit to 3 issues per section
                    prompt += f"• {issue['hint']}\n"
                if len(section_issues) > 3:
                    prompt += f"• ... and {len(section_issues) - 3} more ratio issues\n"
            prompt += "\nRecommended actions: Unify image/card aspect ratios (e.g., 16:9), fine-tune with 4/8pt spacing system\n\n"
        
        # Summary statistics
        prompt += "【Issue Summary】\n"
        prompt += f"• P0 (Critical): {len(p0_issues)} overlap issues\n"
        prompt += f"• P1 (Important): {len(p1_issues)} alignment/spacing issues\n"
        prompt += f"• P2 (Minor): {len(p2_issues)} aspect ratio issues\n"
        prompt += f"• Total: {len(issues)} issues to address\n\n"
        
        # Acceptance criteria
        prompt += "【Acceptance Criteria】\n"
        prompt += "• No overlapping elements\n"
        prompt += "• Left/right alignment tolerance <= 8px\n"
        prompt += "• No horizontal scrolling on mobile\n"
        prompt += "• Consistent spacing\n"
        prompt += "• Unified image proportions\n"
        
        return prompt
    
    def compare_layouts(self, design_path: str, impl_path: str) -> Tuple[List[Dict], str]:
        """Main comparison function"""
        # Load and resize images
        design_img = self.resize_image(design_path)
        impl_img = self.resize_image(impl_path)
        
        print(f"    Image sizes: Design {design_img.shape}, Implementation {impl_img.shape}")
        
        # Extract contours
        design_boxes = self.extract_contours(design_img)
        impl_boxes = self.extract_contours(impl_img)
        
        print(f"    Contours found: Design {len(design_boxes)}, Implementation {len(impl_boxes)}")
        
        # Identify sections
        section_map = self.identify_sections(impl_boxes, impl_img.shape[0])
        
        # Detect text boxes (optional)
        text_boxes = self.detect_text_boxes(impl_img, impl_boxes)
        print(f"    Text boxes detected: {len(text_boxes)}")
        
        # Detect issues
        all_issues = []
        
        # P0: Overlaps
        p0_issues = self.detect_p0_overlaps(impl_boxes, text_boxes)
        # Add section information to P0 issues
        for issue in p0_issues:
            x, y, w, h = issue["bbox"]
            for i, (box_x, box_y, box_w, box_h) in enumerate(impl_boxes):
                if box_x == x and box_y == y and box_w == w and box_h == h:
                    issue["section"] = section_map.get(i, "unknown")
                    break
        all_issues.extend(p0_issues)
        print(f"    P0 issues found: {len(p0_issues)}")
        
        # P1: Alignment
        p1_issues = self.detect_p1_alignment(impl_boxes)
        # Add section information to P1 issues
        for issue in p1_issues:
            x, y, w, h = issue["bbox"]
            for i, (box_x, box_y, box_w, box_h) in enumerate(impl_boxes):
                if box_x == x and box_y == y and box_w == w and box_h == h:
                    issue["section"] = section_map.get(i, "unknown")
                    break
        all_issues.extend(p1_issues)
        print(f"    P1 issues found: {len(p1_issues)}")
        
        # P2: Proportions
        p2_issues = self.detect_p2_proportions(impl_boxes)
        # Add section information to P2 issues
        for issue in p2_issues:
            x, y, w, h = issue["bbox"]
            for i, (box_x, box_y, box_w, box_h) in enumerate(impl_boxes):
                if box_x == x and box_y == y and box_w == w and box_h == h:
                    issue["section"] = section_map.get(i, "unknown")
                    break
        all_issues.extend(p2_issues)
        print(f"    P2 issues found: {len(p2_issues)}")
        
        # Generate suggestion prompt
        suggestion_prompt = self.generate_suggestion_prompt(all_issues)
        
        return all_issues, suggestion_prompt

def find_image_pairs(samples_dir: str) -> List[Tuple[str, str, str]]:
    """Find design and implementation image pairs in samples directory"""
    pairs = []
    
    if not os.path.exists(samples_dir):
        return pairs
    
    # Get all files in samples directory
    files = os.listdir(samples_dir)
    
    # Find design files (ending with _design.png)
    design_files = [f for f in files if f.endswith('_design.png')]
    
    for design_file in design_files:
        # Extract the prefix (e.g., "1" from "1_design.png")
        prefix = design_file.replace('_design.png', '')
        impl_file = f"{prefix}_page.png"
        
        # Check if corresponding implementation file exists
        if impl_file in files:
            design_path = os.path.join(samples_dir, design_file)
            impl_path = os.path.join(samples_dir, impl_file)
            pairs.append((prefix, design_path, impl_path))
    
    return pairs

def main():
    parser = argparse.ArgumentParser(description="Compare design mockups with implementation screenshots")
    parser.add_argument("--samples", default="samples", help="Directory containing image pairs (default: samples)")
    parser.add_argument("--out", default="artifacts", help="Output directory (default: artifacts)")
    parser.add_argument("--width", type=int, default=1280, help="Target width for resizing (default: 1280)")
    parser.add_argument("--align_tol", type=int, default=8, help="Alignment tolerance in pixels (default: 8)")
    parser.add_argument("--ratio_tol", type=float, default=0.1, help="Ratio tolerance (default: 0.1)")
    parser.add_argument("--min_box", type=int, default=16, help="Minimum box size (default: 16)")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.out, exist_ok=True)
    
    # Initialize comparator
    comparator = LayoutComparator(
        width=args.width,
        align_tol=args.align_tol,
        ratio_tol=args.ratio_tol,
        min_box=args.min_box
    )
    
    # Find image pairs
    pairs = find_image_pairs(args.samples)
    
    if not pairs:
        print(f"No image pairs found in {args.samples}/")
        print("Expected format: 1_design.png + 1_page.png, 2_design.png + 2_page.png, etc.")
        return 1
    
    print(f"Found {len(pairs)} image pairs:")
    for prefix, design_path, impl_path in pairs:
        print(f"  {prefix}: {os.path.basename(design_path)} + {os.path.basename(impl_path)}")
    
    # Process each pair
    all_results = {}
    
    for prefix, design_path, impl_path in pairs:
        print(f"\nProcessing pair {prefix}...")
        
        try:
            issues, suggestion_prompt = comparator.compare_layouts(design_path, impl_path)
            
            # Save individual results
            pair_out_dir = os.path.join(args.out, f"pair_{prefix}")
            os.makedirs(pair_out_dir, exist_ok=True)
            
            # Save issues.json
            with open(os.path.join(pair_out_dir, "issues.json"), "w", encoding="utf-8") as f:
                json.dump(issues, f, ensure_ascii=False, indent=2)
            
            # Save suggestion_prompt.txt
            with open(os.path.join(pair_out_dir, "suggestion_prompt.txt"), "w", encoding="utf-8") as f:
                f.write(suggestion_prompt)
            
            # Store results for summary
            all_results[prefix] = {
                "issues": issues,
                "suggestion_prompt": suggestion_prompt,
                "summary": {
                    "total": len(issues),
                    "p0": len([i for i in issues if i['priority'] == 'P0']),
                    "p1": len([i for i in issues if i['priority'] == 'P1']),
                    "p2": len([i for i in issues if i['priority'] == 'P2'])
                }
            }
            
            print(f"  Pair {prefix} complete: {len(issues)} issues found")
            print(f"    P0: {all_results[prefix]['summary']['p0']}")
            print(f"    P1: {all_results[prefix]['summary']['p1']}")
            print(f"    P2: {all_results[prefix]['summary']['p2']}")
            
        except Exception as e:
            print(f"  Error processing pair {prefix}: {e}")
            all_results[prefix] = {"error": str(e)}
    
    # Generate summary report
    summary_path = os.path.join(args.out, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    # Generate combined suggestion prompt
    combined_prompt = "【Layout Analysis Summary】\n\n"
    
    for prefix in sorted(all_results.keys()):
        result = all_results[prefix]
        if "error" in result:
            combined_prompt += f"【Pair {prefix}】Error: {result['error']}\n\n"
        else:
            summary = result['summary']
            combined_prompt += f"【Pair {prefix}】Found {summary['total']} issues (P0: {summary['p0']}, P1: {summary['p1']}, P2: {summary['p2']})\n"
            
            if summary['total'] > 0:
                combined_prompt += result['suggestion_prompt']
            else:
                combined_prompt += "No issues detected - layout looks good!\n"
            combined_prompt += "\n"
    
    # Save combined suggestion prompt
    combined_prompt_path = os.path.join(args.out, "combined_suggestion_prompt.txt")
    with open(combined_prompt_path, "w", encoding="utf-8") as f:
        f.write(combined_prompt)
    
    print(f"\nAnalysis complete!")
    print(f"Results saved to: {args.out}/")
    print(f"Individual results: pair_1/, pair_2/, etc.")
    print(f"Summary: summary.json")
    print(f"Combined suggestions: combined_suggestion_prompt.txt")
    
    return 0

if __name__ == "__main__":
    exit(main())
