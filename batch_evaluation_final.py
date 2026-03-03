"""
Batch Evaluation: Test Pipeline on Multiple Floor Plans
Author: Noah Guthrie
CS686 Final Project - AI-Driven Building Code Compliance Checker
Date: November 2025

Evaluates the complete pipeline (Phases 1-4) on multiple floor plans
from the CubiCasa5k dataset to provide statistical analysis.
"""
import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import random #commented out random.seed(42) to see different results

#import all phases
try:
    from phase3_final_uncertainty import (
        BUILDING_CODES,
        load_image,
        parse_svg_file,
        run_probabilistic_analysis
    )
    from phase1_final_floor_parsing import calculate_metrics as calculate_phase1_metrics
    from phase2_final_constraint_checking import check_building_code_compliance
    from phase4_final_rl_agent import run_rl_optimization

    PHASES_AVAILABLE = True
    print("[✓] All phases imported successfully")
except ImportError as e:
    print(f"[ERROR] Could not import phases: {e}")
    exit(1)


class BatchEvaluator:
    """Evaluate pipeline on multiple floor plans"""
    def __init__(self, dataset_path, num_samples=100, use_split=None, subset='all'):
        """
        Args:
            dataset_path: Path to cubicasa5k/cubicasa5k directory
            num_samples: Number of floor plans to evaluate
            use_split: 'train', 'test', or 'val' to use official splits, or None for random
            subset: 'high_quality', 'high_quality_architectural', 'colorful', or 'all'
        """
        self.dataset_path = dataset_path
        self.num_samples = num_samples
        self.use_split = use_split
        self.subset = subset
        self.results = []

    def find_floor_plans(self):
        """Find all floor plan directories in the dataset"""
        print(f"\n[INFO] Searching for floor plans in: {self.dataset_path}")

        floor_plans = []

        #if using official splits
        if self.use_split:
            return self._load_from_split()

        #otherwise, scan directories
        subdirs = []
        if self.subset == 'all':
            subdirs = ['high_quality', 'high_quality_architectural', 'colorful']
        else:
            subdirs = [self.subset]

        print(f"[INFO] Searching in subdirectories: {subdirs}")

        for subdir in subdirs:
            subdir_path = os.path.join(self.dataset_path, subdir)

            if not os.path.exists(subdir_path):
                print(f"[WARNING] Path not found: {subdir_path}")
                continue

            #find all numbered subdirectories with model.svg
            for item in os.listdir(subdir_path):
                floor_plan_dir = os.path.join(subdir_path, item)
                if os.path.isdir(floor_plan_dir):
                    svg_path = os.path.join(floor_plan_dir, "model.svg")
                    image_path = os.path.join(floor_plan_dir, "F1_scaled.png")

                    if os.path.exists(svg_path) and os.path.exists(image_path):
                        floor_plans.append({
                            'id': item,
                            'subset': subdir,
                            'svg_path': svg_path,
                            'image_path': image_path,
                            'dir': floor_plan_dir
                        })

        print(f"[✓] Found {len(floor_plans)} floor plans")

        #randomly sample if we have more than needed
        if len(floor_plans) > self.num_samples:
            #random.seed(42)  #reproducibility
            floor_plans = random.sample(floor_plans, self.num_samples)
            print(f"[INFO] Randomly selected {self.num_samples} floor plans for evaluation")

        return floor_plans

    def _load_from_split(self):
        """Load floor plans from train/test/val split files"""
        split_file = os.path.join(self.dataset_path, f"{self.use_split}.txt")

        if not os.path.exists(split_file):
            print(f"[ERROR] Split file not found: {split_file}")
            return []

        print(f"[INFO] Loading floor plans from {self.use_split}.txt")

        floor_plans = []
        with open(split_file, 'r') as f:
            for line in f:
                #line format: /high_quality_architectural/1191/
                path = line.strip().strip('/')
                parts = path.split('/')

                if len(parts) >= 2:
                    subset = parts[0]  #e.g., 'high_quality_architectural'
                    floor_id = parts[1]  #e.g., '1191'

                    floor_plan_dir = os.path.join(self.dataset_path, subset, floor_id)
                    svg_path = os.path.join(floor_plan_dir, "model.svg")
                    image_path = os.path.join(floor_plan_dir, "F1_scaled.png")

                    if os.path.exists(svg_path) and os.path.exists(image_path):
                        floor_plans.append({
                            'id': floor_id,
                            'subset': subset,
                            'svg_path': svg_path,
                            'image_path': image_path,
                            'dir': floor_plan_dir
                        })

        print(f"[✓] Found {len(floor_plans)} floor plans in {self.use_split} split")

        #sample if needed
        if len(floor_plans) > self.num_samples:
            #random.seed(42)
            floor_plans = random.sample(floor_plans, self.num_samples)
            print(f"[INFO] Randomly selected {self.num_samples} from {self.use_split} split")

        return floor_plans

    def evaluate_single_floor_plan(self, floor_plan, index, total):
        """Evaluate a single floor plan through all phases"""
        print(f"\n{'='*70}")
        print(f"Evaluating Floor Plan {index}/{total}: ID {floor_plan['id']}")
        print(f"{'='*70}")

        result = {
            'id': floor_plan['id'],
            'success': False,
            'error': None,
            'phase1': {},
            'phase2': {},
            'phase3': {},
            'phase4': {}
        }

        start_time = time.time()

        try:
            #phase 1: Parse
            print("\n[Phase 1] Parsing floor plan...")
            image = load_image(floor_plan['image_path'])
            rooms, doors = parse_svg_file(floor_plan['svg_path'])

            result['phase1'] = {
                'num_rooms': len(rooms),
                'num_doors': len(doors),
                'room_types': [r['type'] for r in rooms]
            }

            if len(rooms) == 0:
                result['error'] = 'No rooms detected'
                print(f"  ⚠️ Skipping: No rooms detected")
                return result

            print(f"  ✓ Detected {len(rooms)} rooms, {len(doors)} doors")

            #phase 2: Check compliance
            print("\n[Phase 2] Checking building code compliance...")
            violations = check_building_code_compliance(rooms, doors)

            room_violations = [v for v in violations if v['type'] == 'ROOM_SIZE']
            door_violations = [v for v in violations if v['type'] == 'DOOR_WIDTH']
            exit_violations = [v for v in violations if v['type'] == 'EXIT_COUNT']

            result['phase2'] = {
                'total_violations': len(violations),
                'room_violations': len(room_violations),
                'door_violations': len(door_violations),
                'exit_violations': len(exit_violations),
                'violation_details': violations
            }

            print(f"  ✓ Found {len(violations)} violations")

            #phase 3: Probabilistic analysis (suppress plots and verbose output)
            print("\n[Phase 3] Running probabilistic analysis...")
            room_results, door_results = run_probabilistic_analysis(
                rooms, doors, BUILDING_CODES,
                show_plots=False, #dont show plots during batch
                verbose=False #dont print reports during batch
            )

            compliant = len([r for r in room_results + door_results if r['status'] == 'COMPLIANT'])
            likely_compliant = len([r for r in room_results + door_results if r['status'] == 'LIKELY_COMPLIANT'])
            uncertain = len([r for r in room_results + door_results if r['status'] == 'UNCERTAIN'])
            likely_violations = len([r for r in room_results + door_results if r['status'] == 'LIKELY_VIOLATION'])

            avg_confidence = np.mean([r['confidence'] for r in room_results + door_results])

            result['phase3'] = {
                'compliant': compliant,
                'likely_compliant': likely_compliant,
                'uncertain': uncertain,
                'likely_violations': likely_violations,
                'avg_confidence': avg_confidence
            }

            print(f"  ✓ Average confidence: {avg_confidence*100:.1f}%")

            #phase 4: RL optimization (only if violations exist)
            if len(violations) > 0:
                print("\n[Phase 4] Running RL optimization...")
                #use fewer episodes for batch processing
                env, agent = run_rl_optimization(
                    rooms, doors, BUILDING_CODES,
                    num_episodes=100,
                    show_plots=False, #dont show plots during batch
                    verbose=False #dont print reports during batch
                )

                violations_after = len(env.get_violations())
                improvement = len(violations) - violations_after
                improvement_pct = (improvement / len(violations) * 100) if len(violations) > 0 else 0

                result['phase4'] = {
                    'violations_before': len(violations),
                    'violations_after': violations_after,
                    'improvement': improvement,
                    'improvement_pct': improvement_pct,
                    'num_modifications': len(env.action_history)
                }

                print(f"  ✓ Improved: {improvement} violations fixed ({improvement_pct:.1f}%)")
            else:
                result['phase4'] = {
                    'violations_before': 0,
                    'violations_after': 0,
                    'improvement': 0,
                    'improvement_pct': 100.0,
                    'num_modifications': 0
                }
                print(f"  ✓ Already compliant, no optimization needed")

            result['success'] = True
            result['processing_time'] = time.time() - start_time

        except Exception as e:
            result['error'] = str(e)
            result['processing_time'] = time.time() - start_time
            print(f"  ❌ Error: {e}")

        return result

    def run_evaluation(self):
        """Run evaluation on all floor plans"""
        print("\n" + "="*70)
        print("BATCH EVALUATION: TESTING PIPELINE ON MULTIPLE FLOOR PLANS")
        print("="*70)

        floor_plans = self.find_floor_plans()

        if len(floor_plans) == 0:
            print("[ERROR] No floor plans found!")
            return

        print(f"\n[INFO] Starting evaluation of {len(floor_plans)} floor plans...")
        print(f"[INFO] Using 50 RL episodes per floor plan for speed")

        for ii, floor_plan in enumerate(floor_plans, 1):
            result = self.evaluate_single_floor_plan(floor_plan, ii, len(floor_plans))
            self.results.append(result)

            #save intermediate results
            if ii % 10 == 0:
                self.save_results(f"batch_results_intermediate_{ii}.json")

        #generate final report
        self.generate_report()
        self.save_results("batch_results_final.json")
        self.visualize_results()

    def generate_report(self):
        """Generate comprehensive evaluation report"""
        print("\n" + "="*70)
        print("BATCH EVALUATION REPORT")
        print("="*70)

        successful = [r for r in self.results if r['success']]
        failed = [r for r in self.results if not r['success']]

        print(f"\n📊 OVERALL STATISTICS")
        print(f"{'='*70}")
        print(f"Total floor plans evaluated: {len(self.results)}")
        print(f"Successful: {len(successful)} ({len(successful)/len(self.results)*100:.1f}%)")
        print(f"Failed: {len(failed)} ({len(failed)/len(self.results)*100:.1f}%)")

        if len(failed) > 0:
            print(f"\nFailure reasons:")
            failure_reasons = defaultdict(int)
            for r in failed:
                failure_reasons[r['error']] += 1
            for reason, count in failure_reasons.items():
                print(f"  • {reason}: {count}")

        if len(successful) == 0:
            print("\n[WARNING] No successful evaluations!")
            return

        #phase 1 Statistics
        print(f"\n🏠 PHASE 1: FLOOR PLAN PARSING")
        print(f"{'='*70}")
        num_rooms = [r['phase1']['num_rooms'] for r in successful]
        num_doors = [r['phase1']['num_doors'] for r in successful]

        print(f"Rooms per floor plan:")
        print(f"  Average: {np.mean(num_rooms):.1f}")
        print(f"  Min: {np.min(num_rooms)}, Max: {np.max(num_rooms)}")
        print(f"  Median: {np.median(num_rooms):.1f}")

        print(f"\nDoors per floor plan:")
        print(f"  Average: {np.mean(num_doors):.1f}")
        print(f"  Min: {np.min(num_doors)}, Max: {np.max(num_doors)}")
        print(f"  Median: {np.median(num_doors):.1f}")

        #room type distribution
        all_room_types = []
        for r in successful:
            all_room_types.extend(r['phase1']['room_types'])
        room_type_counts = defaultdict(int)
        for rt in all_room_types:
            room_type_counts[rt] += 1

        print(f"\nMost common room types:")
        for rt, count in sorted(room_type_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  • {rt}: {count} ({count/len(all_room_types)*100:.1f}%)")

        #phase 2 Statistics
        print(f"\n⚠️  PHASE 2: BUILDING CODE COMPLIANCE")
        print(f"{'='*70}")
        total_violations = [r['phase2']['total_violations'] for r in successful]
        room_violations = [r['phase2']['room_violations'] for r in successful]
        door_violations = [r['phase2']['door_violations'] for r in successful]

        compliant_plans = sum(1 for v in total_violations if v == 0)

        print(f"Compliance rate: {compliant_plans}/{len(successful)} ({compliant_plans/len(successful)*100:.1f}%) fully compliant")
        print(f"\nViolations per floor plan:")
        print(f"  Average: {np.mean(total_violations):.1f}")
        print(f"  Min: {np.min(total_violations)}, Max: {np.max(total_violations)}")
        print(f"  Median: {np.median(total_violations):.1f}")

        print(f"\nViolation breakdown:")
        print(f"  Room size violations: {np.mean(room_violations):.1f} average")
        print(f"  Door width violations: {np.mean(door_violations):.1f} average")

        #most common violations
        violation_types = defaultdict(int)
        for r in successful:
            for v in r['phase2']['violation_details']:
                if v['type'] == 'ROOM_SIZE':
                    violation_types[f"Undersized {v['room_type']}"] += 1
                elif v['type'] == 'DOOR_WIDTH':
                    violation_types[f"Narrow {v['door_class']} door"] += 1
                else:
                    violation_types[v['type']] += 1

        print(f"\nMost common violations:")
        for vtype, count in sorted(violation_types.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  • {vtype}: {count}")

        #phase 3 Statistics
        print(f"\n🎲 PHASE 3: PROBABILISTIC ANALYSIS")
        print(f"{'='*70}")
        avg_confidences = [r['phase3']['avg_confidence'] for r in successful]

        print(f"Average measurement confidence: {np.mean(avg_confidences)*100:.1f}%")
        print(f"  Min: {np.min(avg_confidences)*100:.1f}%, Max: {np.max(avg_confidences)*100:.1f}%")

        #phase 4 Statistics
        print(f"\n🤖 PHASE 4: RL OPTIMIZATION")
        print(f"{'='*70}")

        #only consider floor plans that had violations
        with_violations = [r for r in successful if r['phase2']['total_violations'] > 0]

        if len(with_violations) > 0:
            improvements = [r['phase4']['improvement'] for r in with_violations]
            improvement_pcts = [r['phase4']['improvement_pct'] for r in with_violations]

            fully_fixed = sum(1 for r in with_violations if r['phase4']['violations_after'] == 0)

            print(f"Floor plans with violations: {len(with_violations)}")
            print(f"Fully fixed: {fully_fixed} ({fully_fixed/len(with_violations)*100:.1f}%)")
            print(f"\nAverage improvement: {np.mean(improvements):.1f} violations fixed")
            print(f"Average improvement rate: {np.mean(improvement_pcts):.1f}%")
            print(f"Best improvement: {np.max(improvements)} violations fixed ({np.max(improvement_pcts):.1f}%)")
            print(f"Worst improvement: {np.min(improvements)} violations fixed ({np.min(improvement_pcts):.1f}%)")

            #success rate by initial violation count
            print(f"\nImprovement by initial violation count:")
            violation_buckets = defaultdict(list)
            for r in with_violations:
                initial = r['phase2']['total_violations']
                bucket = (initial // 5) * 5  # Bucket into groups of 5
                violation_buckets[bucket].append(r['phase4']['improvement_pct'])

            for bucket in sorted(violation_buckets.keys()):
                avg_imp = np.mean(violation_buckets[bucket])
                print(f"  {bucket}-{bucket+4} violations: {avg_imp:.1f}% improvement rate")
        else:
            print("All floor plans were already compliant!")

        #processing time
        print(f"\n⏱️  PERFORMANCE")
        print(f"{'='*70}")
        processing_times = [r['processing_time'] for r in successful]
        print(f"Average processing time: {np.mean(processing_times):.1f} seconds per floor plan")
        print(f"Total evaluation time: {sum(processing_times)/60:.1f} minutes")

        print("\n" + "="*70)
        print("✓ Batch evaluation complete!")
        print("="*70)

    def save_results(self, filename):
        """Save results to JSON file"""
        output_path = filename
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n[✓] Results saved to: {output_path}")

    def visualize_results(self):
        """Create visualization of results"""
        successful = [r for r in self.results if r['success']]

        if len(successful) == 0:
            print("[WARNING] No successful results to visualize")
            return

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Batch Evaluation Results', fontsize=16, fontweight='bold')

        #1. Violations distribution
        ax = axes[0, 0]
        violations = [r['phase2']['total_violations'] for r in successful]
        ax.hist(violations, bins=20, color='orange', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Number of Violations')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Violations per Floor Plan')
        ax.axvline(np.mean(violations), color='red', linestyle='--', label=f'Mean: {np.mean(violations):.1f}')
        ax.legend()

        #2. Room and door counts
        ax = axes[0, 1]
        num_rooms = [r['phase1']['num_rooms'] for r in successful]
        num_doors = [r['phase1']['num_doors'] for r in successful]
        ax.scatter(num_rooms, num_doors, alpha=0.5)
        ax.set_xlabel('Number of Rooms')
        ax.set_ylabel('Number of Doors')
        ax.set_title('Rooms vs Doors Distribution')
        ax.grid(True, alpha=0.3)

        #3. Violation types
        ax = axes[0, 2]
        room_viols = sum(r['phase2']['room_violations'] for r in successful)
        door_viols = sum(r['phase2']['door_violations'] for r in successful)
        exit_viols = sum(r['phase2']['exit_violations'] for r in successful)

        ax.bar(['Room Size', 'Door Width', 'Exit Count'], [room_viols, door_viols, exit_viols],
               color=['#ff6b6b', '#4ecdc4', '#45b7d1'])
        ax.set_ylabel('Total Count')
        ax.set_title('Violation Types Across All Floor Plans')

        #4. RL Improvement
        ax = axes[1, 0]
        with_violations = [r for r in successful if r['phase2']['total_violations'] > 0]
        if len(with_violations) > 0:
            improvements = [r['phase4']['improvement_pct'] for r in with_violations]
            ax.hist(improvements, bins=20, color='green', alpha=0.7, edgecolor='black')
            ax.set_xlabel('Improvement Rate (%)')
            ax.set_ylabel('Frequency')
            ax.set_title('RL Agent Improvement Distribution')
            ax.axvline(np.mean(improvements), color='red', linestyle='--', label=f'Mean: {np.mean(improvements):.1f}%')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No violations to fix', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('RL Agent Improvement Distribution')

        #5. Before vs After
        ax = axes[1, 1]
        if len(with_violations) > 0:
            before = [r['phase4']['violations_before'] for r in with_violations]
            after = [r['phase4']['violations_after'] for r in with_violations]
            ax.scatter(before, after, alpha=0.5)
            ax.plot([0, max(before)], [0, max(before)], 'r--', label='No improvement')
            ax.set_xlabel('Violations Before RL')
            ax.set_ylabel('Violations After RL')
            ax.set_title('RL Optimization Effectiveness')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No violations to fix', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('RL Optimization Effectiveness')

        #6. Confidence scores
        ax = axes[1, 2]
        confidences = [r['phase3']['avg_confidence'] * 100 for r in successful]
        ax.hist(confidences, bins=20, color='purple', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Average Confidence (%)')
        ax.set_ylabel('Frequency')
        ax.set_title('Measurement Confidence Distribution')
        ax.axvline(np.mean(confidences), color='red', linestyle='--', label=f'Mean: {np.mean(confidences):.1f}%')
        ax.legend()

        plt.tight_layout()
        plt.savefig('batch_evaluation_results.png', dpi=300, bbox_inches='tight')
        print(f"[✓] Visualization saved to: batch_evaluation_results.png")
        plt.show()

def print_report_evidence(batch_results):
    """Print evidence for each report question"""
    successful = [r for r in batch_results if r['success']]
    print(f"\n=== EVIDENCE FOR REPORT QUESTIONS ===")
    print(f"Q1 (What are you trying to do?)")
    print(f"  ✓ Processed {len(successful)} floor plans")
    print(f"Q2 (How is it done before?)")
    print(f"  ✓ Manual: 30 min/plan × {len(successful)} = {len(successful)*0.5} hours")
    print(f"  ✓ Our system: < 1 minute ({len(successful)*30 / (len(successful)*0.5)}× faster)")
    print(f"Q4 (Who cares?)")
    print(f"  ✓ Architects: {len([r for r in successful if r['phase2']['total_violations'] > 0])} floor plans need fixes")

def main():
    """Main execution"""
    #configuration
    DATASET_PATH = r"C:\Users\guthr\OneDrive - University of Waterloo\Waterloo Homework\AI\Final Project\archive (3)\cubicasa5k\cubicasa5k"
    NUM_SAMPLES = 100  #adjust this based on time available

    print("\n" + "="*70)
    print("BATCH EVALUATION TOOL")
    print("AI-Driven Building Code Compliance Checker")
    print("="*70)
    print(f"\nDataset path: {DATASET_PATH}")
    print(f"Number of samples: {NUM_SAMPLES}")

    response = input("\nProceed with evaluation? (y/n): ")
    if response.lower() != 'y':
        print("Evaluation cancelled.")
        return

    #run evaluation
    evaluator = BatchEvaluator(DATASET_PATH, NUM_SAMPLES)
    evaluator.run_evaluation()


if __name__ == "__main__":
    main()