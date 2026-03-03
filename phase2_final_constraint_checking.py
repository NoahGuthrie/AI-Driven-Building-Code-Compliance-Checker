"""
Phase 2: Constraint-Based Building Code Compliance Checker
Author: Noah Guthrie
CS686 Final Project - AI-Driven Building Code Compliance Checker
Date: November 2025

Uses constraint satisfaction to check building code compliance
Implements 3 critical building codes:
    1. Minimum room size requirements
    2. Minimum door width requirements
    3. Minimum number of exits
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from constraint import Problem
import os

#import Phase 1 functions
try:
    from phase1_final_floor_parsing import (
        load_image,
        parse_svg_file,
        visualize_results,
        generate_summary,
        IMAGE_PATH,
        SVG_PATH,
        BASE_PATH
    )
    PHASE1_AVAILABLE = True
    print("[✓] Phase 1 module imported successfully")
except ImportError as e:
    print(f"[WARNING] Could not import Phase 1: {e}")
    print("[INFO] Running in standalone mode")
    PHASE1_AVAILABLE = False

#building code definitions (International Building Code - simplified)
BUILDING_CODES = {
    #minimum room areas (in meters squared)
    'min_room_areas': {
        'Bedroom': 7.0,
        'Living Room': 12.0,
        'Kitchen': 4.2,
        'Bathroom': 2.5,
        'default': 4.0
    },

    #minimum door widths (in cm)
    'min_door_widths': {
        'entrance': 80,
        'interior': 70,
        #'bathroom': 60,
        'emergency_exit': 80
    },

    #minimum number of exists
    'min_exits': {
        'total_dwelling': 2
    }
}

#constraint checking functions
def check_room_size_constraints(rooms):
    """
    Check if rooms meet minimum size requirements
    Returns: list of violations
    """
    violations = []

    for ii, room in enumerate(rooms, 1):
        room_type = room['type']
        area_units = room['area']
        #convert to meters squared
        area_m2 = area_units / 10000

        #get min required area
        min_area = BUILDING_CODES['min_room_areas'].get(
            room_type,
            BUILDING_CODES['min_room_areas']['default']
        )

        if area_m2 < min_area:
            violations.append({
                'type': 'ROOM_SIZE',
                'severity': 'HIGH',
                'room_id': ii,
                'room_type': room_type,
                'actual_area': area_m2,
                'required_area': min_area,
                'deficit': min_area - area_m2,
                'message': f"{room_type} is undersized: {area_m2:.1f} m² < {min_area} m² required"
            })

    return violations

def check_door_width_constraints(doors, rooms):
    """
    Check if doors meet minimum width requirements
    Returns: list of violations
    """
    violations = []
    for ii, door in enumerate(doors, 1):
        #convert to cm
        width_cm = max(door['bbox'][2], door['bbox'][3])

        #determine door type based on position/context
        #for simplicity, classify as interior vs entrance

        #heuristic: doors near edges are likely entrances
        x, y = door['center']
        is_entrance = (y < 250 or y > 900 or x < 100 or x > 650)

        if is_entrance:
            min_width = BUILDING_CODES['min_door_widths']['entrance']
            door_class = 'Entrance/Exit'
        else:
            min_width = BUILDING_CODES['min_door_widths']['interior']
            door_class = 'Interior'

        if width_cm < min_width:
            violations.append({
                'type': 'DOOR_WIDTH',
                'severity': 'MEDIUM' if door_class == 'Interior' else 'HIGH',
                'door_id': ii,
                'door_class': door_class,
                'actual_width': width_cm,
                'required_width': min_width,
                'deficit': min_width - width_cm,
                'message': f"{door_class} door {ii} is too narrow: {width_cm:.1f} cm < {min_width} cm required"
            })
    return violations

def check_exit_requirements(doors, rooms):
    """
    Check if dwelling has minimum required exits
    Returns: list of all violations
    """
    violations = []

    #count potential exits (doors near perimeter)
    exit_doors = []
    for ii, door in enumerate(doors, 1):
        x, y = door['center']
        #check if door is on or near perimeter
        is_exit = (y < 250 or y > 900 or x < 100 or x > 650)
        if is_exit:
            exit_doors.append(ii)

    min_exits = BUILDING_CODES['min_exits']['total_dwelling']

    if len(exit_doors) < min_exits:
        violations.append({
            'type': 'EXIT_COUNT',
            'severity': 'CRITICAL',
            'actual_exits': len(exit_doors),
            'required_exits': min_exits,
            'deficit': min_exits - len(exit_doors),
            'message': f"Insufficient exits: {len(exit_doors)} found, {min_exits} required"
        })

    return violations

#constraint solver using python-constraint
def solve_constraints_csp(rooms, doors):
    """
    Use CSP solver to find if there is a valid configuration
    This demonstrates using an off-the-shelf constraint solver
    """
    problem = Problem()

    #variables: room areas (we'll see if they can meet constraints)
    for ii, room in enumerate(rooms):
        room_type = room['type']
        area_m2 = room['area'] / 10000
        min_area = BUILDING_CODES['min_room_areas'].get(
            room_type,
            BUILDING_CODES['min_room_areas']['default']
        )

        #add variable with domain [current_area, current_area]
        #in phase 4 RL, we'd expand this domain for optimization
        problem.addVariable(f"room_{ii}_area", [area_m2])

        #add constraint: area must be >= min
        problem.addConstraint(
            lambda a, min_a=min_area: a >= min_a,
            [f"room_{ii}_area"]
        )

    #try to find a solution
    solutions = problem.getSolutions()

    return len(solutions) > 0, solutions

#main compliance checker
def check_building_code_compliance(rooms, doors):
    """
    Main function to check all building code constraints
    """
    print("\n" + "="*70)
    print("PHASE 2: BUILDING CODE COMPLIANCE CHECK")
    print("="*70 + "\n")

    print(f"Checking {len(rooms)} rooms and {len(doors)} doors against building codes...\n")

    all_violations = []

    #1. check room sizes
    print("🏠 Checking Room Size Requirements")
    room_violations = check_room_size_constraints(rooms)
    all_violations.extend(room_violations)

    if room_violations:
        print(f"❌ Found {len(room_violations)} room size violations")
    else:
        print(f"✅ All rooms meet size requirements")

    #2. check door widths
    print("\n🚪 Checking Door Width Requirements...")
    door_violations = check_door_width_constraints(doors, rooms)
    all_violations.extend(door_violations)

    if door_violations:
        print(f"❌ Found {len(door_violations)} door width violations")
    else:
        print(f"✅ All doors meet width requirements")

    #3. check exit requirements
    print("\n🚨 Checking Exit Requirements...")
    exit_violations = check_exit_requirements(doors, rooms)
    all_violations.extend(exit_violations)

    if exit_violations:
        print(f"❌ Found {len(exit_violations)} exit violations")
    else:
        print(f"✅ All exit requirements met")

    #4. use csp solver (demonstration)
    print("\n🔍 Running CSP Solver...")
    has_solutions, solutions = solve_constraints_csp(rooms, doors)

    if has_solutions:
        print(f"✅ CSP solver found valid configuration")
    else:
        print(f"❌ CSP solver did not find valid configuration")

    #generate report
    print("\n" + "=" * 70)
    print("COMPLIANCE REPORT")
    print("=" * 70)

    if not all_violations:
        print("\n🎉 SUCCESS! Floor plan is COMPLIANT with all building codes.")
        print("\nAll checks passed:")
        print("  ✅ Room sizes meet minimum requirements")
        print("  ✅ Door widths meet minimum requirements")
        print("  ✅ Exit requirements satisfied")
    else:
        print(f"\n⚠️  VIOLATIONS FOUND: {len(all_violations)} total\n")

        #group by severity
        critical = [v for v in all_violations if v.get('severity') == 'CRITICAL']
        high = [v for v in all_violations if v.get('severity') == 'HIGH']
        medium = [v for v in all_violations if v.get('severity') == 'MEDIUM']

        if critical:
            print(f"🔴 CRITICAL ({len(critical)}):")
            for v in critical:
                print(f"   • {v['message']}")

        if high:
            print(f"\n🟠 HIGH SEVERITY ({len(high)}):")
            for v in high:
                print(f"   • {v['message']}")

        if medium:
            print(f"\n🟡 MEDIUM SEVERITY ({len(medium)}):")
            for v in medium:
                print(f"   • {v['message']}")

        print(f"\n📋 Detailed Violations:")
        for ii, violation in enumerate(all_violations, 1):
            print(f"\n  Violation {ii}:")
            print(f"    Type: {violation['type']}")
            print(f"    Severity: {violation['severity']}")
            print(f"    Details: {violation['message']}")
            if 'deficit' in violation:
                print(f"    Deficit: {violation['deficit']:.2f} {'m²' if violation['type'] == 'ROOM_SIZE' else 'cm' if violation['type'] == 'DOOR_WIDTH' else 'exits'}")

    print("\n" + "=" * 70)
    print("✓ Phase 2 Complete!")
    print("=" * 70)
    print("\nNext Steps:")
    print("  → Phase 3: Add probabilistic reasoning for confidence scores")
    print("  → Phase 4: Implement RL agent to suggest layout improvements\n")

    return all_violations

#visualize with violations highlighted
def visualize_with_violations(image, rooms, doors, violations, save_path="phase2_output.png"):
    """Visualize floor plan with violations highlighted"""
    vis_img = image.copy()

    # Get violation info
    violation_rooms = {v['room_id'] for v in violations if v['type'] == 'ROOM_SIZE'}
    violation_doors = {v['door_id'] for v in violations if v['type'] == 'DOOR_WIDTH'}

    # Draw rooms
    for i, room in enumerate(rooms, 1):
        points = np.array(room['points'], dtype=np.int32)

        # Red if violation, green otherwise
        color = (100, 100, 255) if i in violation_rooms else (100, 200, 100)

        overlay = vis_img.copy()
        cv2.fillPoly(overlay, [points], color)
        vis_img = cv2.addWeighted(vis_img, 0.6, overlay, 0.4, 0)

        # Thicker red outline for violations
        outline_color = (0, 0, 255) if i in violation_rooms else (0, 255, 0)
        thickness = 4 if i in violation_rooms else 2
        cv2.polylines(vis_img, [points], True, outline_color, thickness)

        # Label
        cx, cy = int(room['center'][0]), int(room['center'][1])
        label = f"{room['type']}"
        area_m2 = room['area'] / 10000

        cv2.putText(vis_img, label, (cx-40, cy),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(vis_img, f"{area_m2:.1f} m²", (cx-30, cy+20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        if i in violation_rooms:
            cv2.putText(vis_img, "⚠ VIOLATION", (cx-45, cy-15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)

    # Draw doors
    for ii, door in enumerate(doors, 1):
        x, y, w, h = door['bbox']
        x, y, w, h = int(x), int(y), int(w), int(h)
        cx, cy = int(door['center'][0]), int(door['center'][1])

        # Red if violation, blue otherwise
        color = (0, 0, 255) if ii in violation_doors else (255, 0, 0)
        thickness = 4 if ii in violation_doors else 2

        cv2.rectangle(vis_img, (x, y), (x+w, y+h), color, thickness)
        cv2.circle(vis_img, (cx, cy), 5, color, -1)

        label = f"D{ii}"
        if ii in violation_doors:
            label += " ⚠"
        cv2.putText(vis_img, label, (cx+8, cy-8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 2)

    # Display
    plt.figure(figsize=(16, 12))
    plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
    plt.title(f"Phase 2: Compliance Check - {len(violations)} Violation(s) Found",
             fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    if save_path:
        cv2.imwrite(save_path, vis_img)
        print(f"[✓] Saved visualization to: {save_path}")

#integration with phase 1
def main():
    """Main execution pipeline - Runs Phase 1 + Phase 2"""
    print("\n" + "=" * 70)
    print("INTEGRATED PIPELINE: PHASE 1 + PHASE 2")
    print("=" * 70 + "\n")

    # Check files exist
    if not os.path.exists(IMAGE_PATH):
        print(f"[ERROR] Image not found: {IMAGE_PATH}")
        return
    if not os.path.exists(SVG_PATH):
        print(f"[ERROR] SVG not found: {SVG_PATH}")
        return

    # ===== PHASE 1: Parse Floor Plan =====
    print("=" * 70)
    print("PHASE 1: FLOOR PLAN PARSING (SVG-BASED)")
    print("=" * 70 + "\n")

    # Load image
    image = load_image(IMAGE_PATH)

    # Parse SVG
    rooms, doors = parse_svg_file(SVG_PATH)

    if len(rooms) == 0:
        print("[WARNING] No rooms detected! Check SVG structure.")
        return

    print(f"\n[✓] Phase 1 Complete: {len(rooms)} rooms, {len(doors)} doors detected\n")

    # ===== PHASE 2: Check Building Codes =====
    violations = check_building_code_compliance(rooms, doors)

    #visualize results with violations highlighted
    print("\n[INFO] Generating visualization with violations...")
    visualize_with_violations(image, rooms, doors, violations)

    #summary
    print("\n" + "=" * 70)
    print("✓ PHASES 1 & 2 COMPLETE!")
    print("=" * 70)
    print("\nSummary:")
    print(f"  • Rooms detected: {len(rooms)}")
    print(f"  • Doors detected: {len(doors)}")
    print(f"  • Violations found: {len(violations)}")
    print(f"  • Compliance status: {'✅ PASS' if not violations else '⚠️ FAIL'}")

    print("\nNext Steps:")
    print("  → Phase 3: Add probabilistic reasoning for confidence scores")
    print("  → Phase 4: Implement RL agent to suggest layout improvements\n")

    return rooms, doors, violations

if __name__ == "__main__":
    main()