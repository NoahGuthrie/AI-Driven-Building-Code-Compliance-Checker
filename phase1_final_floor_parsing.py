"""
Phase 1: Floor Plan Parsing and Visualization with Evaluation Metrics
Author: Noah Guthrie
CS686 Final Project - AI-Driven Building Code Compliance Checker
Date: November 2025

Uses CubiCasa5k SVG annotations for accurate room and door detection
Includes F1, IoU, and other metrics for performance evaluation
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import os
from scipy.optimize import linear_sum_assignment
from collections import defaultdict

#configuration
BASE_PATH = r"C:\Users\guthr\OneDrive - University of Waterloo\Waterloo Homework\AI\Final Project\archive (3)\cubicasa5k\cubicasa5k\high_quality\17"
IMAGE_PATH = os.path.join(BASE_PATH, "F1_scaled.png")
SVG_PATH = os.path.join(BASE_PATH, "model.svg")

def load_image(path):
    """Load floor plan image"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    image = cv2.imread(path)
    if image is None:
        raise ValueError("Failed to load image")
    print(f"[✓] Loaded image: {image.shape}")
    return image

def parse_svg_file(svg_path):
    """
    Parse CubiCasa5k SVG file to extract rooms and doors
    """
    print(f"[INFO] Parsing SVG: {svg_path}")

    if not os.path.exists(svg_path):
        print(f"[ERROR] SVG file not found: {svg_path}")
        return [], []

    try:
        tree = ET.parse(svg_path)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"[ERROR] Malformed XML: {e}")
        return [], []

    rooms = []
    doors = []

    #parse all elements recursively
    for elem in root.iter():
        elem_class = elem.get('class', '')

        #check if this is a room (Space class), but exclude outdoor spaces
        if 'Space' in elem_class and 'Outdoor' not in elem_class:
            #find the polygon child element
            polygon = elem.find('.//{http://www.w3.org/2000/svg}polygon')
            if polygon is None:
                polygon = elem.find('.//polygon')

            if polygon is not None:
                points_str = polygon.get('points', '')
                if points_str:
                    points = parse_svg_points(points_str)
                    if len(points) >= 3:
                        bbox = get_bounding_box(points)
                        area = calculate_polygon_area(points)

                        #filter out very small spaces (< 0.5 m² or ~5000 sq units)
                        if area < 5000:
                            continue

                        #extract room type from class
                        room_type = extract_room_type(elem_class)

                        rooms.append({
                            'type': room_type,
                            'points': points,
                            'bbox': bbox,
                            'area': area,
                            'center': get_polygon_center(points),
                            'full_class': elem_class
                        })

        #check if this is a door (Door class)
        elif 'Door' in elem_class:
            #doors have a threshold polygon
            threshold = elem.find('.//{http://www.w3.org/2000/svg}*[@class="Threshold"]/{http://www.w3.org/2000/svg}polygon')
            if threshold is None:
                threshold = elem.find('.//*[@class="Threshold"]/polygon')

            if threshold is not None:
                points_str = threshold.get('points', '')
                if points_str:
                    points = parse_svg_points(points_str)
                    if len(points) >= 2:
                        bbox = get_bounding_box(points)
                        width = min(bbox[2], bbox[3])
                        length = max(bbox[2], bbox[3])

                        #determine orientation
                        orientation = 'horizontal' if bbox[2] > bbox[3] else 'vertical'

                        #extract door type
                        door_type = extract_door_type(elem_class)

                        doors.append({
                            'type': door_type,
                            'points': points,
                            'bbox': bbox,
                            'center': ((bbox[0] + bbox[2]/2), (bbox[1] + bbox[3]/2)),
                            'width': width,
                            'length': length,
                            'orientation': orientation,
                            'full_class': elem_class
                        })

    print(f"[✓] Parsed {len(rooms)} rooms and {len(doors)} doors from SVG")
    return rooms, doors

def extract_room_type(class_str):
    """Extract readable room type from class string"""
    types = {
        'LivingRoom': 'Living Room',
        'Bedroom': 'Bedroom',
        'Kitchen': 'Kitchen',
        'Bath': 'Bathroom',
        'Storage': 'Storage',
        'Entry': 'Entry',
        'Lobby': 'Lobby',
        'Outdoor': 'Outdoor',
        'Undefined': 'Room',
        'DraughtLobby': 'Vestibule'
    }

    for key, value in types.items():
        if key in class_str:
            return value
    return 'Room'

def extract_door_type(class_str):
    """Extract door type from class string"""
    if 'Swing' in class_str:
        return 'Swing Door'
    return 'Door'

def parse_svg_points(points_str):
    """
    Parse SVG points string to list of (x, y) tuples
    """
    points = []
    coords = points_str.strip().replace(',', ' ').split()

    for i in range(0, len(coords)-1, 2):
        try:
            x = float(coords[i])
            y = float(coords[i+1])
            points.append((x, y))
        except (ValueError, IndexError):
            continue

    return points

def get_bounding_box(points):
    """Calculate bounding box: (x, y, width, height)"""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    return (x_min, y_min, x_max - x_min, y_max - y_min)

def calculate_polygon_area(points):
    """Calculate area using shoelace formula"""
    if len(points) < 3:
        return 0
    area = 0
    for i in range(len(points)):
        j = (i + 1) % len(points)
        area += points[i][0] * points[j][1]
        area -= points[j][0] * points[i][1]
    return abs(area) / 2

def get_polygon_center(points):
    """Calculate centroid"""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (sum(xs) / len(xs), sum(ys) / len(ys))

# ============================================================================
# EVALUATION METRICS
# ============================================================================

def calculate_polygon_iou(poly1_points, poly2_points, image_shape=(2000, 2000)):
    """
    Calculate IoU between two polygons using rasterization
    """
    mask1 = np.zeros(image_shape, dtype=np.uint8)
    mask2 = np.zeros(image_shape, dtype=np.uint8)

    points1 = np.array(poly1_points, dtype=np.int32)
    points2 = np.array(poly2_points, dtype=np.int32)

    cv2.fillPoly(mask1, [points1], 1)
    cv2.fillPoly(mask2, [points2], 1)

    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    if union == 0:
        return 0.0

    return intersection / union

def calculate_bbox_iou(bbox1, bbox2):
    """
    Calculate IoU between two bounding boxes (x, y, w, h)
    Faster approximation for door matching
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    #convert to corner coordinates
    x1_max, y1_max = x1 + w1, y1 + h1
    x2_max, y2_max = x2 + w2, y2 + h2

    #intersection
    xi = max(x1, x2)
    yi = max(y1, y2)
    xi_max = min(x1_max, x2_max)
    yi_max = min(y1_max, y2_max)

    inter_area = max(0, xi_max - xi) * max(0, yi_max - yi)

    #union
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area

def match_detections_to_ground_truth(predicted, ground_truth, iou_threshold=0.5, use_polygon_iou=True):
    """
    Match predicted detections to ground truth using Hungarian algorithm
    Returns: matched_pairs, unmatched_pred, unmatched_gt, iou_scores
    """
    if len(predicted) == 0 or len(ground_truth) == 0:
        return [], list(range(len(predicted))), list(range(len(ground_truth))), []

    #compute IoU matrix
    iou_matrix = np.zeros((len(predicted), len(ground_truth)))

    for i, pred in enumerate(predicted):
        for j, gt in enumerate(ground_truth):
            if use_polygon_iou and 'points' in pred and 'points' in gt:
                iou = calculate_polygon_iou(pred['points'], gt['points'])
            else:
                iou = calculate_bbox_iou(pred['bbox'], gt['bbox'])
            iou_matrix[i, j] = iou

    #hungarian algorithm for optimal matching
    pred_indices, gt_indices = linear_sum_assignment(-iou_matrix)

    matched_pairs = []
    iou_scores = []

    for pred_idx, gt_idx in zip(pred_indices, gt_indices):
        iou = iou_matrix[pred_idx, gt_idx]
        if iou >= iou_threshold:
            matched_pairs.append((pred_idx, gt_idx))
            iou_scores.append(iou)

    #find unmatched
    matched_pred = set([p[0] for p in matched_pairs])
    matched_gt = set([p[1] for p in matched_pairs])

    unmatched_pred = [i for i in range(len(predicted)) if i not in matched_pred]
    unmatched_gt = [i for i in range(len(ground_truth)) if i not in matched_gt]

    return matched_pairs, unmatched_pred, unmatched_gt, iou_scores

def calculate_metrics(predicted_rooms, ground_truth_rooms, predicted_doors, ground_truth_doors,
                     iou_threshold=0.5):
    """
    Calculate comprehensive evaluation metrics
    """
    print("\n" + "="*70)
    print("EVALUATION METRICS")
    print("="*70)

    metrics = {}

    # ROOM DETECTION METRICS
    print(f"\n📦 ROOM DETECTION (IoU threshold = {iou_threshold})")

    room_matches, room_fp, room_fn, room_ious = match_detections_to_ground_truth(
        predicted_rooms, ground_truth_rooms, iou_threshold, use_polygon_iou=True
    )

    tp_rooms = len(room_matches)
    fp_rooms = len(room_fp)
    fn_rooms = len(room_fn)

    precision_rooms = tp_rooms / (tp_rooms + fp_rooms) if (tp_rooms + fp_rooms) > 0 else 0
    recall_rooms = tp_rooms / (tp_rooms + fn_rooms) if (tp_rooms + fn_rooms) > 0 else 0
    f1_rooms = 2 * (precision_rooms * recall_rooms) / (precision_rooms + recall_rooms) \
               if (precision_rooms + recall_rooms) > 0 else 0
    avg_iou_rooms = np.mean(room_ious) if len(room_ious) > 0 else 0

    print(f"  • True Positives:  {tp_rooms}")
    print(f"  • False Positives: {fp_rooms}")
    print(f"  • False Negatives: {fn_rooms}")
    print(f"  • Precision: {precision_rooms:.3f}")
    print(f"  • Recall:    {recall_rooms:.3f}")
    print(f"  • F1 Score:  {f1_rooms:.3f}")
    print(f"  • Avg IoU:   {avg_iou_rooms:.3f}")

    metrics['rooms'] = {
        'tp': tp_rooms, 'fp': fp_rooms, 'fn': fn_rooms,
        'precision': precision_rooms, 'recall': recall_rooms,
        'f1_score': f1_rooms, 'avg_iou': avg_iou_rooms
    }

    # ROOM TYPE CLASSIFICATION ACCURACY
    if len(room_matches) > 0:
        correct_types = 0
        confusion = defaultdict(lambda: defaultdict(int))

        for pred_idx, gt_idx in room_matches:
            pred_type = predicted_rooms[pred_idx]['type']
            gt_type = ground_truth_rooms[gt_idx]['type']
            confusion[gt_type][pred_type] += 1
            if pred_type == gt_type:
                correct_types += 1

        type_accuracy = correct_types / len(room_matches)
        print(f"\n  • Room Type Accuracy: {type_accuracy:.3f} ({correct_types}/{len(room_matches)})")

        metrics['rooms']['type_accuracy'] = type_accuracy
        metrics['rooms']['confusion_matrix'] = dict(confusion)

    # AREA ESTIMATION ERROR
    if len(room_matches) > 0:
        area_errors = []
        for pred_idx, gt_idx in room_matches:
            pred_area = predicted_rooms[pred_idx]['area']
            gt_area = ground_truth_rooms[gt_idx]['area']
            error = abs(pred_area - gt_area)
            area_errors.append(error)

        mae_area = np.mean(area_errors)
        mae_area_m2 = mae_area / 10000  #convert to m²
        print(f"  • Mean Area Error: {mae_area:.0f} sq units (~{mae_area_m2:.2f} m²)")

        metrics['rooms']['mae_area'] = mae_area

    # DOOR DETECTION METRICS
    print(f"\n🚪 DOOR DETECTION (IoU threshold = {iou_threshold})")

    door_matches, door_fp, door_fn, door_ious = match_detections_to_ground_truth(
        predicted_doors, ground_truth_doors, iou_threshold, use_polygon_iou=False
    )

    tp_doors = len(door_matches)
    fp_doors = len(door_fp)
    fn_doors = len(door_fn)

    precision_doors = tp_doors / (tp_doors + fp_doors) if (tp_doors + fp_doors) > 0 else 0
    recall_doors = tp_doors / (tp_doors + fn_doors) if (tp_doors + fn_doors) > 0 else 0
    f1_doors = 2 * (precision_doors * recall_doors) / (precision_doors + recall_doors) \
               if (precision_doors + recall_doors) > 0 else 0
    avg_iou_doors = np.mean(door_ious) if len(door_ious) > 0 else 0

    print(f"  • True Positives:  {tp_doors}")
    print(f"  • False Positives: {fp_doors}")
    print(f"  • False Negatives: {fn_doors}")
    print(f"  • Precision: {precision_doors:.3f}")
    print(f"  • Recall:    {recall_doors:.3f}")
    print(f"  • F1 Score:  {f1_doors:.3f}")
    print(f"  • Avg IoU:   {avg_iou_doors:.3f}")

    metrics['doors'] = {
        'tp': tp_doors, 'fp': fp_doors, 'fn': fn_doors,
        'precision': precision_doors, 'recall': recall_doors,
        'f1_score': f1_doors, 'avg_iou': avg_iou_doors
    }

    # DOOR WIDTH ESTIMATION ERROR
    if len(door_matches) > 0:
        width_errors = []
        for pred_idx, gt_idx in door_matches:
            pred_width = predicted_doors[pred_idx]['width']
            gt_width = ground_truth_doors[gt_idx]['width']
            error = abs(pred_width - gt_width)
            width_errors.append(error)

        mae_width = np.mean(width_errors)
        mae_width_cm = mae_width * 0.1  #rough conversion
        print(f"  • Mean Width Error: {mae_width:.1f} units (~{mae_width_cm:.1f} cm)")

        metrics['doors']['mae_width'] = mae_width

    # OVERALL METRICS
    print(f"\n📊 OVERALL PERFORMANCE")
    overall_f1 = (f1_rooms + f1_doors) / 2
    print(f"  • Combined F1 Score: {overall_f1:.3f}")
    metrics['overall_f1'] = overall_f1

    print("="*70 + "\n")

    return metrics

# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_results(image, rooms, doors, save_path=None):
    """Visualize detected rooms and doors"""
    vis_img = image.copy()

    print(f"\n[INFO] Visualizing {len(rooms)} rooms and {len(doors)} doors")

    # Colors for rooms
    np.random.seed(42)
    colors = [(100, 200, 100), (200, 100, 100), (100, 100, 200),
              (200, 200, 100), (200, 100, 200), (100, 200, 200),
              (150, 150, 100), (150, 100, 150), (100, 150, 150)]

    #draw rooms
    for i, room in enumerate(rooms):
        points = np.array(room['points'], dtype=np.int32)
        color = colors[i % len(colors)]

        #filled polygon with transparency
        overlay = vis_img.copy()
        cv2.fillPoly(overlay, [points], color)
        vis_img = cv2.addWeighted(vis_img, 0.6, overlay, 0.4, 0)

        #outline
        cv2.polylines(vis_img, [points], True, (0, 255, 0), 3)

        #label
        cx, cy = int(room['center'][0]), int(room['center'][1])
        label = f"{room['type']}"

        #text with background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(vis_img, (cx-tw//2-5, cy-th-5), (cx+tw//2+5, cy+5), (255, 255, 255), -1)
        cv2.putText(vis_img, label, (cx-tw//2, cy),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        #area
        area_m2 = room['area'] / 10000  #rough conversion (assuming cm units)
        area_text = f"{area_m2:.1f} m²"
        cv2.putText(vis_img, area_text, (cx-40, cy+25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 0), 2)

    #draw doors
    for i, door in enumerate(doors):
        x, y, w, h = door['bbox']
        x, y, w, h = int(x), int(y), int(w), int(h)
        cx, cy = int(door['center'][0]), int(door['center'][1])

        #door rectangle
        cv2.rectangle(vis_img, (x, y), (x+w, y+h), (0, 0, 255), 3)

        #center marker
        cv2.circle(vis_img, (cx, cy), 6, (255, 0, 0), -1)
        cv2.circle(vis_img, (cx, cy), 6, (255, 255, 255), 2)

        #label
        cv2.putText(vis_img, f"D{i+1}", (cx+10, cy-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    #display
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Floor Plan", fontsize=16, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f"Phase 1: {len(rooms)} Rooms, {len(doors)} Doors Detected",
                     fontsize=16, fontweight='bold')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

    if save_path:
        cv2.imwrite(save_path, vis_img)
        print(f"[✓] Saved to: {save_path}")

def generate_summary(rooms, doors):
    """Generate detailed summary report"""
    print("\n" + "="*70)
    print("PHASE 1: FLOOR PLAN ANALYSIS RESULTS")
    print("="*70)

    #sort rooms by area
    sorted_rooms = sorted(rooms, key=lambda r: r['area'], reverse=True)

    print(f"\n📦 ROOMS DETECTED: {len(rooms)}\n")

    total_area = 0
    for i, room in enumerate(sorted_rooms, 1):
        x, y, w, h = room['bbox']
        area = room['area']
        total_area += area
        area_m2 = area / 10000  #rough conversion
        cx, cy = room['center']

        print(f"Room {i}: {room['type']}")
        print(f"  • Bounding Box: ({x:.1f}, {y:.1f})")
        print(f"  • Dimensions: {w:.1f} × {h:.1f} units")
        print(f"  • Area: {area:.0f} sq units (~{area_m2:.1f} m²)")
        print(f"  • Center: ({cx:.0f}, {cy:.0f})")
        print()

    print(f"Total floor area: ~{total_area/10000:.1f} m²\n")

    print(f"🚪 DOORS DETECTED: {len(doors)}\n")

    for i, door in enumerate(doors, 1):
        x, y, w, h = door['bbox']
        width = door['width']
        width_cm = width * 0.1  #rough conversion to cm
        cx, cy = door['center']

        print(f"Door {i}: {door['type']}")
        print(f"  • Position: ({x:.0f}, {y:.0f})")
        print(f"  • Width: {width:.1f} units (~{width_cm:.0f} cm)")
        print(f"  • Orientation: {door['orientation']}")
        print(f"  • Center: ({cx:.0f}, {cy:.0f})")
        print()

    print("="*70)

def main():
    """Main execution pipeline"""
    print("\n" + "="*70)
    print("PHASE 1: FLOOR PLAN PARSING (SVG-BASED) WITH METRICS")
    print("="*70 + "\n")

    #check files
    if not os.path.exists(IMAGE_PATH):
        print(f"[ERROR] Image not found: {IMAGE_PATH}")
        return
    if not os.path.exists(SVG_PATH):
        print(f"[ERROR] SVG not found: {SVG_PATH}")
        return

    #load image
    image = load_image(IMAGE_PATH)

    #parse SVG (this acts as both prediction AND ground truth for demo)
    rooms, doors = parse_svg_file(SVG_PATH)

    if len(rooms) == 0:
        print("[WARNING] No rooms detected! Check SVG structure.")
        return

    #visualize
    visualize_results(image, rooms, doors, save_path="phase1_final.png")

    #summary
    generate_summary(rooms, doors)

    # EVALUATION METRICS
    # In a real scenario, you'd have separate predicted vs ground_truth data
    # For demonstration, we'll use the same data as both (perfect score expected)
    print("\n[INFO] Computing evaluation metrics...")
    print("[NOTE] Using same data as prediction and ground truth (demo only)")

    metrics = calculate_metrics(
        predicted_rooms=rooms,
        ground_truth_rooms=rooms,
        predicted_doors=doors,
        ground_truth_doors=doors,
        iou_threshold=0.5
    )

    print("✓ Phase 1 Complete!")
    print("="*70)
    print("\nNext Steps:")
    print("  → Phase 2: Implement constraint-based code checking")
    print("  → Phase 3: Add probabilistic reasoning for confidence scores")
    print("  → Phase 4: Implement RL for layout optimization")
    print("\n[TIP] To test metrics properly, run on separate test/validation sets")
    print()

if __name__ == "__main__":
    main()