"""
Phase 3: Probabilistic Reasoning for Building Code Compliance
Author: Noah Guthrie
CS686 Final Project - AI-Driven Building Code Compliance Checker
Date: November 2025

Adds probabilistic reasoning to handle uncertainty in measurements:
- Confidence scores for room area calculations
- Uncertainty in door width measurements
- Probabilistic compliance assessment
- Bayesian inference for incomplete data
"""
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

#import Phase 2 (which includes Phase 1)
try:
    from phase2_final_constraint_checking import (
        BUILDING_CODES,
        load_image,
        parse_svg_file,
        IMAGE_PATH,
        SVG_PATH,
        check_building_code_compliance
    )
    PHASE2_AVAILABLE = True
    print("[✓] Phase 2 module (with Phase 1) imported successfully")
except ImportError as e:
    print(f"[WARNING] Could not import Phase 2: {e}")
    print("[INFO] Running in standalone mode")
    PHASE2_AVAILABLE = False
    # Define minimal BUILDING_CODES for standalone mode
    BUILDING_CODES = {
        'min_room_areas': {
            'Bedroom': 7.0,
            'Living Room': 12.0,
            'Kitchen': 4.2,
            'Bathroom': 2.5,
            'default': 4.0
        },
        'min_door_widths': {
            'entrance': 80,
            'interior': 70
        }
    }

#uncertainty modelling
class MeasurementUncertainty:
    """
    Models uncertainty in floor plan measurements
    """
    def __init__(self):
        #measurement noise parameters (based on typical svg parsing accuracy)
        self.area_uncertainty_pct = 0.05 #5% uncertainty in area
        self.door_uncertainty_cm = 2.0 #+- 2cm uncertainty in door widths
        self.position_uncertainty_px = 3.0 #+- pixels in position

    def calculate_area_confidence(self, room):
        """
        Calculate confidence score for room area measurement

        Factors affecting confidence:
        1. Number of corners (more complex = less confident)
        2. Room size (very small = less confident)
        3. Shape irregularity (irregular = less confident)
        """
        area = room['area']
        points = room['points']
        num_corners = len(points)

        #base confidence
        confidence = 1.0

        #penalty for complexity (many corners)
        if num_corners > 8:
            confidence *= 0.85
        elif num_corners > 6:
            confidence *= 0.90
        elif num_corners > 4:
            confidence *= 0.95

        #penalty for very small rooms (harder to measure accurately)
        area_m2 = area / 10000
        if area_m2 < 2.0:
            confidence *= 0.80
        elif area_m2 < 4.0:
            confidence *= 0.90

        #check shape irregularity
        bbox_area = room['bbox'][2] * room['bbox'][3]
        if bbox_area > 0:
            fill_ratio = area / bbox_area
            if fill_ratio < 0.6: #very irregular shape
                confidence *= 0.85

        return max(0.5, min(1.0, confidence)) #clamp between 0.5 and 1.0

    def calculate_door_confidence(self, door):
        """
        Calculate door confidence score for door width measurements

        Factors:
        1. Door orientation clarity
        2. Door size (very narrow = less confident)
        3. Position (near edges = more confident as likely main door)
        """
        width = max(door['bbox'][2], door['bbox'][3])

        #base confidence
        confidence = 1.0

        #penalty for very narrow doors (harder to measure)
        if width < 30:
            confidence *= 0.75
        elif width < 50:
            confidence *= 0.85

        #bonus for clear position (near perimeter = likely main door)
        x, y = door['center']
        is_perimeter = (y < 250 or y > 900 or x < 100 or x > 650)
        if is_perimeter:
            confidence *= 1.05

        return max(0.6, min(1.0, confidence))

    def get_area_distribution(self, room, confidence):
        """
        Get probability distribution for room area
        Returns (mean, std_dev) for normal distribution
        """
        area_m2 = room['area'] / 10000

        #standard deviation based on confidence
        #lower confidence = higher uncertainty
        base_std = area_m2 * self.area_uncertainty_pct
        std_dev = base_std / confidence

        return area_m2, std_dev

    def get_door_width_distribution(self, door, confidence):
        """
        Get probability distribution for door width
        Returns: (mean, std_dev) for normal distribution
        """
        width_cm = max(door['bbox'][2], door['bbox'][3])

        #standard deviation based on confidence
        std_dev = self.door_uncertainty_cm / confidence

        return width_cm, std_dev

#probabilistic compliance checking
def calculate_compliance_probability(measured_value, required_value, std_dev):
    """
    Calculate probability that actual value meets requirement

    Use normal distribution and cumulative distribution function
    P(actual >= required) = 1 - CDF(required)
    """
    if std_dev == 0:
        return 1.0 if measured_value >= required_value else 0.0

    #z-score
    z = (measured_value - required_value) / std_dev

    #probability of meeting requirement
    prob_compliant = stats.norm.cdf(z)

    return prob_compliant

def check_room_compliance_probabilistic(rooms, building_codes, uncertainty_model):
    """
    Check room compliance with probabilistic reasoning
    """
    results = []

    for ii, room in enumerate(rooms, 1):
        room_type = room['type']

        #get required area
        min_area = building_codes['min_room_areas'].get(
            room_type,
            building_codes['min_room_areas']['default']
        )

        #calculate confidence
        confidence = uncertainty_model.calculate_area_confidence(room)

        #get measurement distribution
        measured_area, std_dev = uncertainty_model.get_area_distribution(room, confidence)

        #calculate compliance probability
        prob_compliant = calculate_compliance_probability(
            measured_area, min_area, std_dev
        )

        #determine status
        if prob_compliant >= 0.95:
            status = 'COMPLIANT'
            severity = None
        elif prob_compliant >= 0.70:
            status = 'LIKELY_COMPLIANT'
            severity = 'LOW'
        elif prob_compliant >= 0.30:
            status = 'UNCERTAIN'
            severity = 'MEDIUM'
        else:
            status = 'LIKELY_VIOLATION'
            severity = 'HIGH'

        results.append({
            'room_id': ii,
            'room_type': room_type,
            'measured_area': measured_area,
            'required_area': min_area,
            'confidence': confidence,
            'std_dev': std_dev,
            'prob_compliant': prob_compliant,
            'status': status,
            'severity': severity
        })

    return results

def check_door_compliance_probabilistic(doors, building_codes, uncertainty_model):
    """
    Check door compliance with probabilistic reasoning
    """
    results = []

    for ii, door in enumerate(doors, 1):
        #determine door type
        x, y = door['center']
        is_entrance = (y < 250 or y > 900 or x < 100 or x > 650)

        if is_entrance:
            min_width = building_codes['min_door_widths']['entrance']
            door_class = 'Entrance/Exit'
        else:
            min_width = building_codes['min_door_widths']['interior']
            door_class = 'Interior'

        #calculate confidence
        confidence = uncertainty_model.calculate_door_confidence(door)

        #get measurement distribution
        measured_width, std_dev = uncertainty_model.get_door_width_distribution(door, confidence)

        #calculate compliance probability
        prob_compliant = calculate_compliance_probability(
            measured_width, min_width, std_dev
        )

        # determine status
        if prob_compliant >= 0.95:
            status = 'COMPLIANT'
            severity = None
        elif prob_compliant >= 0.70:
            status = 'LIKELY_COMPLIANT'
            severity = 'LOW'
        elif prob_compliant >= 0.30:
            status = 'UNCERTAIN'
            severity = 'MEDIUM'
        else:
            status = 'LIKELY_VIOLATION'
            severity = 'HIGH'

        results.append({
            'door_id': ii,
            'door_class': door_class,
            'measured_width': measured_width,
            'required_width': min_width,
            'confidence': confidence,
            'std_dev': std_dev,
            'prob_compliant': prob_compliant,
            'status': status,
            'severity': severity
        })

    return results

#visualization
def visualize_probability_distributions(room_results, door_results, show_plot=True):
    """
    Visualize probability distributions for uncertain measurements
    """
    #find items with uncertainty
    uncertain_rooms = [r for r in room_results if r['status'] in ['UNCERTAIN', 'LIKELY_VIOLATION', 'LIKELY_COMPLIANT']]
    uncertain_doors = [d for d in door_results if d['status'] in ['UNCERTAIN', 'LIKELY_VIOLATION', 'LIKELY_COMPLIANT']]

    if not uncertain_doors and not uncertain_rooms:
        print("[INFO] No uncertain measurements to visualize")
        return

    #create subplots
    n_plots = len(uncertain_rooms) + len(uncertain_doors)
    if n_plots == 0:
        return

    fig, axes = plt.subplots(min(n_plots, 4), 1, figsize=(10, 3 * min(n_plots, 4)))
    if n_plots == 1:
        axes = [axes]

    plot_idx = 0

    #plot uncertain rooms
    for result in uncertain_rooms[:2]: #show max 2
        if plot_idx >= len(axes):
            break

        ax = axes[plot_idx]
        mean = result['measured_area']
        std = result['std_dev']
        required = result['required_area']

        #generate distribution
        x = np.linspace(mean - 4 * std, mean + 4 * std, 200)
        y = stats.norm.pdf(x, mean, std)

        #plot
        ax.plot(x, y, 'b-', linewidth=2, label='Measured distribution')
        ax.axvline(required, color='r', linestyle='--', linewidth=2, label='Required minimum')
        ax.axvline(mean, color='g', linestyle='-', linewidth=1, alpha=0.5, label='Measured mean')
        ax.fill_between(x, 0, y, where=(x >= required), alpha=0.3, color='green', label='Compliant region')
        ax.fill_between(x, 0, y, where=(x < required), alpha=0.3, color='red', label='Non-compliant region')

        ax.set_title(f"Room {result['room_id']} ({result['room_type']}): {result['prob_compliant'] * 100:.1f}% compliant")
        ax.set_xlabel('Area (m²)')
        ax.set_ylabel('Probability Density')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        plot_idx += 1

    #plot uncertain doors
    for result in uncertain_doors[:2]: #show max 2
        if plot_idx >= len(axes):
            break

        ax = axes[plot_idx]
        mean = result['measured_width']
        std = result['std_dev']
        required = result['required_width']

        #generate distribution
        x = np.linspace(mean - 4 * std, mean + 4 * std, 200)
        y = stats.norm.pdf(x, mean, std)

        #plot
        ax.plot(x, y, 'b-', linewidth=2, label='Measured distribution')
        ax.axvline(required, color='r', linestyle='--', linewidth=2, label='Required minimum')
        ax.axvline(mean, color='g', linestyle='-', linewidth=1, alpha=0.5, label='Measured mean')
        ax.fill_between(x, 0, y, where=(x >= required), alpha=0.3, color='green')
        ax.fill_between(x, 0, y, where=(x < required), alpha=0.3, color='red')

        ax.set_title(
            f"Door {result['door_id']} ({result['door_class']}): {result['prob_compliant'] * 100:.1f}% compliant")
        ax.set_xlabel('Width (cm)')
        ax.set_ylabel('Probability Density')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        plot_idx += 1

    plt.tight_layout()
    if show_plot:
        plt.show()
    else:
        plt.close()

#reporting
def generate_probabilistic_report(room_results, door_results):
    """
    Generate comprehensive probabilistic compliance report
    """
    print("\n" + "=" * 70)
    print("PHASE 3: PROBABILISTIC COMPLIANCE ANALYSIS")
    print("=" * 70)

    #room analysis
    print("\n🏠 ROOM ANALYSIS (with confidence scores)\n")

    for result in room_results:
        status_symbol = {
            'COMPLIANT': '✅',
            'LIKELY_COMPLIANT': '✓',
            'UNCERTAIN': '❓',
            'LIKELY_VIOLATION': '⚠️'
        }.get(result['status'], '❓')

        print(f"{status_symbol} Room {result['room_id']}: {result['room_type']}")
        print(f"   Measured: {result['measured_area']:.2f} m² (±{result['std_dev']:.2f})")
        print(f"   Required: {result['required_area']:.2f} m²")
        print(f"   Confidence: {result['confidence'] * 100:.1f}%")
        print(f"   P(Compliant): {result['prob_compliant'] * 100:.1f}%")
        print(f"   Status: {result['status']}")
        print()

    #door analysis
    print("🚪 DOOR ANALYSIS (with confidence scores)\n")

    for result in door_results:
        status_symbol = {
            'COMPLIANT': '✅',
            'LIKELY_COMPLIANT': '✓',
            'UNCERTAIN': '❓',
            'LIKELY_VIOLATION': '⚠️'
        }.get(result['status'], '❓')

        print(f"{status_symbol} Door {result['door_id']}: {result['door_class']}")
        print(f"   Measured: {result['measured_width']:.1f} cm (±{result['std_dev']:.1f})")
        print(f"   Required: {result['required_width']:.1f} cm")
        print(f"   Confidence: {result['confidence'] * 100:.1f}%")
        print(f"   P(Compliant): {result['prob_compliant'] * 100:.1f}%")
        print(f"   Status: {result['status']}")
        print()

    #summary statistics
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    total_items = len(room_results) + len(door_results)
    compliant = len([r for r in room_results + door_results if r['status'] == 'COMPLIANT'])
    likely_compliant = len([r for r in room_results + door_results if r['status'] == 'LIKELY_COMPLIANT'])
    uncertain = len([r for r in room_results + door_results if r['status'] == 'UNCERTAIN'])
    violations = len([r for r in room_results + door_results if r['status'] == 'LIKELY_VIOLATION'])

    print(f"\nTotal measurements: {total_items}")
    print(f"  ✅ Compliant (>95%): {compliant}")
    print(f"  ✓  Likely compliant (70-95%): {likely_compliant}")
    print(f"  ❓ Uncertain (30-70%): {uncertain}")
    print(f"  ⚠️  Likely violations (<30%): {violations}")

    avg_confidence = np.mean([r['confidence'] for r in room_results + door_results])
    print(f"\nAverage measurement confidence: {avg_confidence * 100:.1f}%")

    print("\n" + "=" * 70)
    print("✓ Phase 3 Complete!")
    print("=" * 70)

#integration function
def run_probabilistic_analysis(rooms, doors, building_codes, show_plots=True, verbose=True):
    """
    Run complete probabilistic analysis

    Usage:
        from phase3 import run_probabilistic_analysis
        room_results, door_results = run_probabilistic_analysis(rooms, doors, BUILDING_CODES)
    """
    #initialize uncertainty model
    uncertainty_model = MeasurementUncertainty()

    #analyze rooms
    room_results = check_room_compliance_probabilistic(rooms, building_codes, uncertainty_model)

    #analyze doors
    door_results = check_door_compliance_probabilistic(doors, building_codes, uncertainty_model)

    #generate report (only if verbose)
    if verbose:
        generate_probabilistic_report(room_results, door_results)

    #visualize results (only if show_plots is True)
    if show_plots:
        print("\n[INFO] Generating probability distribution visualizations...")
        visualize_probability_distributions(room_results, door_results, show_plot=True)

    return room_results, door_results

def main():
    """
    Main execution pipeline - runs phases 1, 2, and 3
    """
    if not PHASE2_AVAILABLE:
        print("[ERROR] Phase 2 module not available. Cannot run integrated pipeline.")
        return

    print("\n" + "=" * 70)
    print("INTEGRATED PIPELINE: PHASE 1 + PHASE 2 + PHASE 3")
    print("=" * 70 + "\n")

    # ===== PHASE 1 & 2: Parse and Check Compliance =====
    print("Running Phase 1 & 2 (from phase2_constraint_checker)...")

    # Load image and parse SVG
    image = load_image(IMAGE_PATH)
    rooms, doors = parse_svg_file(SVG_PATH)

    if len(rooms) == 0:
        print("[WARNING] No rooms detected! Check SVG structure.")
        return

    # Run Phase 2 compliance check
    violations = check_building_code_compliance(rooms, doors)

    # ===== PHASE 3: Probabilistic Analysis =====
    print("\n" + "=" * 70)
    print("Starting Phase 3: Probabilistic Analysis")
    print("=" * 70)

    room_results, door_results = run_probabilistic_analysis(rooms, doors, BUILDING_CODES)

    # Final summary
    print("\n" + "=" * 70)
    print("✓ ALL PHASES COMPLETE!")
    print("=" * 70)
    print("\nSummary:")
    print(f"  • Rooms detected: {len(rooms)}")
    print(f"  • Doors detected: {len(doors)}")
    print(f"  • Deterministic violations (Phase 2): {len(violations)}")
    print(f"  • Probabilistic analysis (Phase 3): {len(room_results) + len(door_results)} items analyzed")
    print(f"  • Compliance status: {'✅ PASS' if not violations else '⚠️ FAIL'}")

    print("\nNext Steps:")
    print("  → Phase 4: Implement RL agent to suggest layout improvements\n")

    return rooms, doors, violations, room_results, door_results

if __name__ == "__main__":
    if PHASE2_AVAILABLE:
        main()
    else:
        print("Phase 3: Probabilistic Reasoning Module")
        print("Import this module with: from phase3_probabilistic import run_probabilistic_analysis")
        print("This module requires phase2_constraint_checker.py to run the full pipeline.")