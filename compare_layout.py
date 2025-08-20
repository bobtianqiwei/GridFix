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
    def __init__(self, width: int = 1280, align_tol: int = 8, ratio_tol: float = 0.1, 
                 min_box: int = 16, thin_px: int = 4, min_area: int = 1500):
        self.width = width
        self.align_tol = align_tol
        self.ratio_tol = ratio_tol
        self.min_box = min_box
        self.thin_px = thin_px
        self.min_area = min_area
        
    def quantize_boxes(self, boxes: List[Tuple[int, int, int, int]], q: int = 2) -> List[Tuple[int, int, int, int]]:
        """Quantize box coordinates to reduce jitter"""
        quantized = []
        for x, y, w, h in boxes:
            qx = round(x / q) * q
            qy = round(y / q) * q
            qw = round(w / q) * q
            qh = round(h / q) * q
            quantized.append((qx, qy, qw, qh))
        return quantized
    
    def merge_near_duplicates(self, boxes: List[Tuple[int, int, int, int]], 
                             center_tol: int = 4, iou_thresh: float = 0.6) -> List[Tuple[int, int, int, int]]:
        """Merge boxes that are near duplicates"""
        if len(boxes) <= 1:
            return boxes
        
        merged = boxes.copy()
        changed = True
        
        while changed:
            changed = False
            new_merged = []
            used = set()
            
            for i in range(len(merged)):
                if i in used:
                    continue
                
                current_box = merged[i]
                cx1, cy1 = current_box[0] + current_box[2] // 2, current_box[1] + current_box[3] // 2
                
                for j in range(i + 1, len(merged)):
                    if j in used:
                        continue
                    
                    other_box = merged[j]
                    cx2, cy2 = other_box[0] + other_box[2] // 2, other_box[1] + other_box[3] // 2
                    
                    # Check center distance and IoU
                    center_dist = ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5
                    iou = self.calculate_iou(current_box, other_box)
                    
                    if center_dist < center_tol and iou > iou_thresh:
                        # Merge boxes
                        x1, y1, w1, h1 = current_box
                        x2, y2, w2, h2 = other_box
                        new_x = min(x1, x2)
                        new_y = min(y1, y2)
                        new_w = max(x1 + w1, x2 + w2) - new_x
                        new_h = max(y1 + h1, y2 + h2) - new_y
                        current_box = (new_x, new_y, new_w, new_h)
                        used.add(j)
                        changed = True
                
                new_merged.append(current_box)
                used.add(i)
            
            merged = new_merged
        
        return merged
    
    def nms_boxes(self, boxes: List[Tuple[int, int, int, int]], iou_thresh: float = 0.6) -> List[Tuple[int, int, int, int]]:
        """Non-maximum suppression to remove highly overlapping boxes"""
        if len(boxes) <= 1:
            return boxes
        
        # Calculate scores (area)
        scores = [w * h for _, _, w, h in boxes]
        
        # Sort by score (descending)
        indices = list(range(len(boxes)))
        indices.sort(key=lambda i: scores[i], reverse=True)
        
        keep = []
        while indices:
            current = indices.pop(0)
            keep.append(current)
            
            # Remove boxes with high IoU
            remaining = []
            for idx in indices:
                iou = self.calculate_iou(boxes[current], boxes[idx])
                if iou <= iou_thresh:
                    remaining.append(idx)
            indices = remaining
        
        return [boxes[i] for i in keep]
    
    def filter_small_and_thin(self, boxes: List[Tuple[int, int, int, int]], 
                             min_side: int = 4, min_area: int = 1500) -> List[Tuple[int, int, int, int]]:
        """Filter out small and thin elements"""
        filtered = []
        for x, y, w, h in boxes:
            if min(w, h) >= min_side and w * h >= min_area:
                filtered.append((x, y, w, h))
        return filtered
    
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
    
    def classify_boxes(self, image: np.ndarray, boxes: List[Tuple[int, int, int, int]], 
                      text_box_indices: List[int]) -> List[str]:
        """Classify boxes into text, media, icon, or other"""
        classifications = []
        
        for i, (x, y, w, h) in enumerate(boxes):
            if i in text_box_indices:
                classifications.append("text")
            else:
                area = w * h
                ratio = w / h if h > 0 else 0
                
                if area >= 4000 and 0.5 <= ratio <= 2.0:
                    classifications.append("media")
                elif area < 4000 and 0.75 <= ratio <= 1.33:
                    classifications.append("icon")
                else:
                    classifications.append("other")
        
        return classifications
    
    def row_tolerance(self, boxes: List[Tuple[int, int, int, int]]) -> float:
        """Calculate adaptive row tolerance based on median box height"""
        if not boxes:
            return 6.0
        
        heights = [h for _, _, _, h in boxes]
        median_height = sorted(heights)[len(heights) // 2]
        row_tol = 0.25 * median_height
        return max(6, min(row_tol, 20))  # Clamp between 6 and 20
    
    def estimate_content_edges(self, boxes: List[Tuple[int, int, int, int]], 
                              align_tol: int) -> Tuple[Optional[int], Optional[int]]:
        """Estimate content left and right edges"""
        if not boxes:
            return None, None
        
        left_edges = [x for x, _, _, _ in boxes]
        right_edges = [x + w for x, _, w, _ in boxes]
        
        # Create histograms
        left_hist = {}
        right_hist = {}
        
        for left in left_edges:
            bucket = (left // align_tol) * align_tol
            left_hist[bucket] = left_hist.get(bucket, 0) + 1
        
        for right in right_edges:
            bucket = (right // align_tol) * align_tol
            right_hist[bucket] = right_hist.get(bucket, 0) + 1
        
        # Find most common buckets
        content_left = max(left_hist.items(), key=lambda x: x[1])[0] if left_hist else None
        content_right = max(right_hist.items(), key=lambda x: x[1])[0] if right_hist else None
        
        return content_left, content_right
    
    def draw_overlay(self, image: np.ndarray, issues: List[Dict], out_path: str, show_ids: bool = True) -> None:
        """Draw overlay visualization with issue boxes"""
        overlay = image.copy()
        
        # Color mapping for priorities
        colors = {
            "P0": (0, 0, 255),    # Red
            "P1": (0, 165, 255),  # Orange
            "P2": (255, 0, 0)     # Blue
        }
        
        for i, issue in enumerate(issues):
            priority = issue["priority"]
            bbox = issue["bbox"]
            issue_type = issue["type"]
            
            x, y, w, h = bbox
            color = colors.get(priority, (128, 128, 128))
            
            # Draw rectangle
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
            
            # Draw label
            if show_ids:
                label = f"#{i+1} {priority} {issue_type}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1
                
                # Get text size
                (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
                
                # Draw background rectangle for text
                cv2.rectangle(overlay, (x, y - text_height - 5), (x + text_width, y), color, -1)
                
                # Draw text
                cv2.putText(overlay, label, (x, y - 5), font, font_scale, (255, 255, 255), thickness)
        
        cv2.imwrite(out_path, overlay)
    
    def save_issue_crops(self, image: np.ndarray, issues: List[Dict], crops_dir: str, crop_size: int = 160) -> None:
        """Save issue crops for detailed inspection"""
        os.makedirs(crops_dir, exist_ok=True)
        
        for i, issue in enumerate(issues):
            bbox = issue["bbox"]
            priority = issue["priority"]
            issue_type = issue["type"]
            
            x, y, w, h = bbox
            
            # Calculate crop boundaries
            center_x, center_y = x + w // 2, y + h // 2
            half_size = crop_size // 2
            
            crop_x1 = max(0, center_x - half_size)
            crop_y1 = max(0, center_y - half_size)
            crop_x2 = min(image.shape[1], center_x + half_size)
            crop_y2 = min(image.shape[0], center_y + half_size)
            
            # Extract crop
            crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
            
            # Save crop
            filename = f"{i+1:03d}_{priority}_{issue_type}.png"
            crop_path = os.path.join(crops_dir, filename)
            cv2.imwrite(crop_path, crop)
    
    def ssim_hotspots(self, design_img: np.ndarray, impl_img: np.ndarray, 
                     tile: int = 256, stride: int = 128, topk: int = 20) -> List[Tuple[int, int, int, int]]:
        """Calculate SSIM hotspots for focus scoring"""
        try:
            from skimage.metrics import structural_similarity
        except ImportError:
            return []
        
        # Convert to grayscale
        design_gray = cv2.cvtColor(design_img, cv2.COLOR_BGR2GRAY)
        impl_gray = cv2.cvtColor(impl_img, cv2.COLOR_BGR2GRAY)
        
        hotspots = []
        height, width = design_gray.shape
        
        for y in range(0, height - tile, stride):
            for x in range(0, width - tile, stride):
                # Extract tiles
                design_tile = design_gray[y:y+tile, x:x+tile]
                impl_tile = impl_gray[y:y+tile, x:x+tile]
                
                # Calculate SSIM
                ssim_score = structural_similarity(design_tile, impl_tile, data_range=255)
                dissimilarity = 1 - ssim_score
                
                hotspots.append((x, y, tile, tile, dissimilarity))
        
        # Sort by dissimilarity (highest first) and return top-k
        hotspots.sort(key=lambda x: x[4], reverse=True)
        return [(x, y, w, h) for x, y, w, h, _ in hotspots[:topk]]
        
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
        
        # Apply preprocessing pipeline
        boxes = self.quantize_boxes(boxes, q=2)
        boxes = self.merge_near_duplicates(boxes, center_tol=4, iou_thresh=0.6)
        boxes = self.nms_boxes(boxes, iou_thresh=0.6)
        boxes = self.filter_small_and_thin(boxes, min_side=self.min_box, min_area=self.min_area)
        
        return boxes
    
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
                          text_boxes: List[int], classifications: List[str]) -> List[Dict]:
        """Detect P0 overlap issues that affect readability"""
        issues = []
        
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                iou = self.calculate_iou(boxes[i], boxes[j])
                
                if iou > 0.1:  # Only consider significant overlaps
                    x1, y1, w1, h1 = boxes[i]
                    x2, y2, w2, h2 = boxes[j]
                    area1, area2 = w1 * h1, w2 * h2
                    
                    # Calculate overlap ratio relative to smaller element
                    intersection_area = iou * (area1 + area2) / (1 + iou)
                    overlap_ratio = intersection_area / min(area1, area2)
                    
                    # Determine if this overlap affects readability
                    is_text_involved = (i in text_boxes or j in text_boxes)
                    is_large_overlap = overlap_ratio > 0.30
                    is_both_large = (area1 > self.min_area * 2 and area2 > self.min_area * 2 and overlap_ratio > 0.50)
                    
                    if (is_text_involved and is_large_overlap) or is_both_large:
                        # Determine problem element (usually the smaller one)
                        problem_box = boxes[i] if area1 <= area2 else boxes[j]
                        other_box = boxes[j] if area1 <= area2 else boxes[i]
                        
                        x, y, w, h = problem_box
                        other_x, other_y, other_w, other_h = other_box
                        
                        # Create descriptive hint
                        if is_text_involved:
                            hint = f"Text element at ({x},{y}) overlaps with element at ({other_x},{other_y}), suggest using grid/flex/gap/min-height/wrap instead of z-index"
                        else:
                            hint = f"Large elements at ({x},{y}) and ({other_x},{other_y}) overlap significantly, suggest using grid/flex/gap/min-height/wrap instead of z-index"
                        
                        issues.append({
                            "priority": "P0",
                            "type": "overlap",
                            "bbox": [x, y, w, h],
                            "hint": hint,
                            "overlap_with": [other_x, other_y, other_w, other_h],
                            "iou": round(iou, 3),
                            "overlap_ratio": round(overlap_ratio, 3),
                            "score": overlap_ratio
                        })
        
        return issues
    
    def detect_p1_alignment(self, boxes: List[Tuple[int, int, int, int]], 
                          classifications: List[str]) -> List[Dict]:
        """Detect P1 alignment issues using robust methods"""
        issues = []
        
        if len(boxes) < 2:
            return issues
        
        # Estimate content edges
        content_left, content_right = self.estimate_content_edges(boxes, self.align_tol)
        
        # Collect alignment points
        left_points = [box[0] for box in boxes]
        center_points = [box[0] + box[2] // 2 for box in boxes]
        right_points = [box[0] + box[2] for box in boxes]
        
        # Check left alignment with content edge priority
        left_buckets = self._bucket_points(left_points, self.align_tol)
        
        for i, left in enumerate(left_points):
            if not self._is_aligned(left, left_buckets, self.align_tol):
                # Find closest alignment target
                closest_aligned = min(left_buckets, key=lambda bucket: abs(bucket - left))
                offset = left - closest_aligned
                
                # If content edge is available, prefer it
                if content_left is not None:
                    content_offset = abs(left - content_left)
                    if content_offset <= self.align_tol * 2:  # Within reasonable range
                        closest_aligned = content_left
                        offset = left - content_left
                
                # Only report if offset is significant
                if abs(offset) > self.align_tol:
                    x, y, w, h = boxes[i]
                    issues.append({
                        "priority": "P1",
                        "type": "misalign",
                        "bbox": [x, y, w, h],
                        "hint": f"Element at ({x},{y}) left edge not aligned, offset by {offset}px from grid at {closest_aligned}px, suggest aligning to main grid (tolerance {self.align_tol}px)",
                        "current_position": left,
                        "suggested_position": closest_aligned,
                        "offset": abs(offset),
                        "score": abs(offset)
                    })
        
        # Check spacing consistency within same class
        spacing_issues = self._check_spacing_consistency(boxes, classifications)
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
    
    def _check_spacing_consistency(self, boxes: List[Tuple[int, int, int, int]], 
                                 classifications: List[str]) -> List[Dict]:
        """Check spacing consistency between adjacent boxes in same horizontal band"""
        issues = []
        
        # Calculate adaptive row tolerance
        row_tol = self.row_tolerance(boxes)
        
        # Group boxes by horizontal bands and class
        bands_by_class = {}
        for i, (x, y, w, h) in enumerate(boxes):
            band_key = int(y // row_tol)
            class_type = classifications[i] if i < len(classifications) else "other"
            
            if class_type not in bands_by_class:
                bands_by_class[class_type] = {}
            if band_key not in bands_by_class[class_type]:
                bands_by_class[class_type][band_key] = []
            bands_by_class[class_type][band_key].append((i, x, y, w, h))
        
        # Check spacing in each band for each class
        for class_type, bands in bands_by_class.items():
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
                    # Use IQR for robust statistics
                    sorted_spacings = sorted(spacings)
                    median_spacing = sorted_spacings[len(sorted_spacings) // 2]
                    q1 = sorted_spacings[len(sorted_spacings) // 4]
                    q3 = sorted_spacings[3 * len(sorted_spacings) // 4]
                    iqr = q3 - q1
                    
                    # Check for inconsistent spacing using IQR
                    inconsistent_count = 0
                    for spacing in spacings:
                        if abs(spacing - median_spacing) > 0.4 * median_spacing:
                            inconsistent_count += 1
                    
                    if inconsistent_count > len(spacings) * 0.3:  # More than 30% inconsistent
                        for i, x, y, w, h in band_boxes:
                            issues.append({
                                "priority": "P1",
                                "type": "spacing",
                                "bbox": [x, y, w, h],
                                "hint": f"Element at ({x},{y}) has inconsistent spacing in {class_type} row at y={y}px, suggest using unified 8px or 16px spacing system",
                                "row_y": y,
                                "class_type": class_type,
                                "spacing_variance": round(iqr, 1),
                                "median_spacing": round(median_spacing, 1),
                                "score": iqr
                            })
        
        return issues
    
    def detect_p2_proportions(self, boxes: List[Tuple[int, int, int, int]], 
                            classifications: List[str]) -> List[Dict]:
        """Detect P2 proportion issues only in media elements"""
        issues = []
        
        # Filter for media elements only
        media_boxes = [(i, x, y, w, h) for i, (x, y, w, h) in enumerate(boxes) 
                      if i < len(classifications) and classifications[i] == "media"]
        
        if len(media_boxes) < 2:
            return issues
        
        # Calculate aspect ratios
        ratios = []
        for _, x, y, w, h in media_boxes:
            ratio = w / h if h > 0 else 0
            ratios.append(ratio)
        
        # Calculate mean ratio
        mean_ratio = sum(ratios) / len(ratios)
        
        # Check for outliers
        for i, (box_idx, x, y, w, h) in enumerate(media_boxes):
            ratio = ratios[i]
            deviation = abs(ratio - mean_ratio) / mean_ratio
            if deviation > self.ratio_tol:
                issues.append({
                    "priority": "P2",
                    "type": "ratio",
                    "bbox": [x, y, w, h],
                    "hint": f"Media element at ({x},{y}) has aspect ratio {ratio:.2f}:1, differs from group mean {mean_ratio:.2f}:1 by {deviation * 100:.1f}%, suggest unifying to {mean_ratio:.2f}:1 + object-fit: cover",
                    "current_ratio": round(ratio, 2),
                    "group_mean_ratio": round(mean_ratio, 2),
                    "deviation_percent": round(deviation * 100, 1),
                    "score": deviation * 100
                })
        
        return issues
    
    def generate_suggestion_prompt(self, issues: List[Dict], topk: int = 5) -> str:
        """Generate English suggestion prompt for frontend AI with Top-K limiting"""
        prompt = "【Objective】Focus on readability and human-like layout rather than pixel-perfect accuracy.\n\n"
        
        # Group issues by priority and section
        p0_issues = sorted([issue for issue in issues if issue["priority"] == "P0"], 
                          key=lambda x: x.get("score", 0), reverse=True)
        p1_issues = sorted([issue for issue in issues if issue["priority"] == "P1"], 
                          key=lambda x: x.get("score", 0), reverse=True)
        p2_issues = sorted([issue for issue in issues if issue["priority"] == "P2"], 
                          key=lambda x: x.get("score", 0), reverse=True)
        
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
                for issue in section_issues[:topk]:  # Limit to topk issues per section
                    prompt += f"• {issue['hint']}\n"
                if len(section_issues) > topk:
                    prompt += f"• ... and {len(section_issues) - topk} more overlap issues\n"
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
                    for issue in spacing_issues[:topk]:
                        prompt += f"    • {issue['hint']}\n"
                    if len(spacing_issues) > topk:
                        prompt += f"    • ... and {len(spacing_issues) - topk} more spacing issues\n"
                
                if align_issues:
                    prompt += f"  Alignment issues ({len(align_issues)}):\n"
                    for issue in align_issues[:topk]:
                        prompt += f"    • {issue['hint']}\n"
                    if len(align_issues) > topk:
                        prompt += f"    • ... and {len(align_issues) - topk} more alignment issues\n"
            
            prompt += "\nRecommended actions: Align to unified left/right edges, standardize card/button styles (border-radius, padding), unify heading hierarchy\n\n"
        
        # P2 issues by section
        if p2_issues:
            prompt += "【P2 Minor Issues - Aspect Ratios】\n"
            p2_sections = group_by_section(p2_issues)
            for section, section_issues in p2_sections.items():
                prompt += f"\n{section.upper()} section:\n"
                for issue in section_issues[:topk]:  # Limit to topk issues per section
                    prompt += f"• {issue['hint']}\n"
                if len(section_issues) > topk:
                    prompt += f"• ... and {len(section_issues) - topk} more ratio issues\n"
            prompt += "\nRecommended actions: Unify image/card aspect ratios (e.g., 16:9), fine-tune with 4/8pt spacing system\n\n"
        
        # Summary statistics
        prompt += "【Issue Summary】\n"
        prompt += f"• P0 (Critical): {len(p0_issues)} overlap issues\n"
        prompt += f"• P1 (Important): {len(p1_issues)} alignment/spacing issues\n"
        prompt += f"• P2 (Minor): {len(p2_issues)} aspect ratio issues\n"
        prompt += f"• Total: {len(issues)} issues to address\n"
        prompt += f"• Showing top {topk} issues per section\n\n"
        
        # Acceptance criteria
        prompt += "【Acceptance Criteria】\n"
        prompt += "• No overlapping elements\n"
        prompt += "• Left/right alignment tolerance <= 8px\n"
        prompt += "• No horizontal scrolling on mobile\n"
        prompt += "• Consistent spacing\n"
        prompt += "• Unified image proportions\n"
        
        return prompt
    
    def compare_layouts(self, design_path: str, impl_path: str, 
                       overlay: bool = True, save_crops: bool = True, 
                       crop_size: int = 160, topk: int = 5, 
                       use_ssim_focus: bool = False) -> Tuple[List[Dict], str]:
        """Main comparison function with enhanced features"""
        # Load and resize images
        design_img = self.resize_image(design_path)
        impl_img = self.resize_image(impl_path)
        
        print(f"    Image sizes: Design {design_img.shape}, Implementation {impl_img.shape}")
        
        # Extract contours
        design_boxes = self.extract_contours(design_img)
        impl_boxes = self.extract_contours(impl_img)
        
        print(f"    Contours found: Design {len(design_boxes)}, Implementation {len(impl_boxes)}")
        
        # Detect text boxes (optional)
        text_boxes = self.detect_text_boxes(impl_img, impl_boxes)
        print(f"    Text boxes detected: {len(text_boxes)}")
        
        # Classify boxes
        classifications = self.classify_boxes(impl_img, impl_boxes, text_boxes)
        
        # Identify sections
        section_map = self.identify_sections(impl_boxes, impl_img.shape[0])
        
        # Calculate SSIM hotspots if enabled
        ssim_hotspots = []
        if use_ssim_focus:
            ssim_hotspots = self.ssim_hotspots(design_img, impl_img)
            print(f"    SSIM hotspots calculated: {len(ssim_hotspots)}")
        
        # Detect issues
        all_issues = []
        
        # P0: Overlaps
        p0_issues = self.detect_p0_overlaps(impl_boxes, text_boxes, classifications)
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
        p1_issues = self.detect_p1_alignment(impl_boxes, classifications)
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
        p2_issues = self.detect_p2_proportions(impl_boxes, classifications)
        # Add section information to P2 issues
        for issue in p2_issues:
            x, y, w, h = issue["bbox"]
            for i, (box_x, box_y, box_w, box_h) in enumerate(impl_boxes):
                if box_x == x and box_y == y and box_w == w and box_h == h:
                    issue["section"] = section_map.get(i, "unknown")
                    break
        all_issues.extend(p2_issues)
        print(f"    P2 issues found: {len(p2_issues)}")
        
        # Add ID to all issues
        for i, issue in enumerate(all_issues):
            issue["id"] = str(i + 1)
            
            # Apply SSIM focus scoring if enabled
            if use_ssim_focus and ssim_hotspots:
                for hotspot in ssim_hotspots:
                    hx, hy, hw, hh = hotspot
                    # Check if issue bbox overlaps with hotspot
                    if (x < hx + hw and x + w > hx and 
                        y < hy + hh and y + h > hy):
                        # Add bonus score for SSIM hotspot overlap
                        overlap_area = min(x + w, hx + hw) - max(x, hx) * min(y + h, hy + hh) - max(y, hy)
                        issue["score"] = issue.get("score", 0) + overlap_area / (w * h) * 0.1
                        break
        
        # Generate suggestion prompt
        suggestion_prompt = self.generate_suggestion_prompt(all_issues, topk)
        
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
    parser.add_argument("--thin_px", type=int, default=4, help="Minimum side length for thin elements (default: 4)")
    parser.add_argument("--min_area", type=int, default=1500, help="Minimum area for elements (default: 1500)")
    parser.add_argument("--overlay", type=int, default=1, choices=[0, 1], help="Generate overlay visualization (default: 1)")
    parser.add_argument("--save_crops", type=int, default=1, choices=[0, 1], help="Save issue crops (default: 1)")
    parser.add_argument("--crop_size", type=int, default=160, help="Crop size for issue visualization (default: 160)")
    parser.add_argument("--topk", type=int, default=5, help="Top-K issues to show per section (default: 5)")
    parser.add_argument("--use_ssim_focus", type=int, default=0, choices=[0, 1], help="Use SSIM hotspots for scoring (default: 0)")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.out, exist_ok=True)
    
    # Initialize comparator with new parameters
    comparator = LayoutComparator(
        width=args.width,
        align_tol=args.align_tol,
        ratio_tol=args.ratio_tol,
        min_box=args.min_box,
        thin_px=args.thin_px,
        min_area=args.min_area
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
            # Compare layouts with enhanced features
            issues, suggestion_prompt = comparator.compare_layouts(
                design_path, impl_path,
                overlay=bool(args.overlay),
                save_crops=bool(args.save_crops),
                crop_size=args.crop_size,
                topk=args.topk,
                use_ssim_focus=bool(args.use_ssim_focus)
            )
            
            # Save individual results
            pair_out_dir = os.path.join(args.out, f"pair_{prefix}")
            os.makedirs(pair_out_dir, exist_ok=True)
            
            # Save issues.json
            with open(os.path.join(pair_out_dir, "issues.json"), "w", encoding="utf-8") as f:
                json.dump(issues, f, ensure_ascii=False, indent=2)
            
            # Save suggestion_prompt.txt
            with open(os.path.join(pair_out_dir, "suggestion_prompt.txt"), "w", encoding="utf-8") as f:
                f.write(suggestion_prompt)
            
            # Generate overlay visualization
            if args.overlay and issues:
                impl_img = comparator.resize_image(impl_path)
                overlay_path = os.path.join(pair_out_dir, "overlay.png")
                comparator.draw_overlay(impl_img, issues, overlay_path)
                print(f"    Overlay saved: {overlay_path}")
            
            # Save issue crops
            if args.save_crops and issues:
                impl_img = comparator.resize_image(impl_path)
                crops_dir = os.path.join(pair_out_dir, "crops")
                comparator.save_issue_crops(impl_img, issues, crops_dir, args.crop_size)
                print(f"    Crops saved: {crops_dir}/")
            
            # Generate CSV report
            if issues:
                csv_path = os.path.join(pair_out_dir, "issues.csv")
                with open(csv_path, "w", encoding="utf-8") as f:
                    f.write("id,priority,type,section,bbox,score,hint\n")
                    for issue in issues:
                        bbox_str = f"[{issue['bbox'][0]},{issue['bbox'][1]},{issue['bbox'][2]},{issue['bbox'][3]}]"
                        score = issue.get("score", 0)
                        hint = issue["hint"].replace('"', '""')  # Escape quotes
                        f.write(f'"{issue["id"]}","{issue["priority"]}","{issue["type"]}","{issue.get("section", "unknown")}","{bbox_str}",{score},"{hint}"\n')
                print(f"    CSV report saved: {csv_path}")
            
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
    
    # Print legend
    if args.overlay:
        legend_path = os.path.join(args.out, "legend.txt")
        with open(legend_path, "w", encoding="utf-8") as f:
            f.write("Overlay Color Legend:\n")
            f.write("• Red (P0): Critical overlap issues\n")
            f.write("• Orange (P1): Important alignment/spacing issues\n")
            f.write("• Blue (P2): Minor proportion issues\n")
        print(f"Legend saved: {legend_path}")
    
    for prefix in sorted(all_results.keys()):
        result = all_results[prefix]
        if "error" not in result:
            summary = result['summary']
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
