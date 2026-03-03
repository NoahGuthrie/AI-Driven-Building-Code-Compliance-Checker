"""
Phase 4: Reinforcement Learning Agent for Layout Optimization
Author: Noah Guthrie
CS686 Final Project - AI-Driven Building Code Compliance Checker
Date: November 2025

Uses RL to suggest layout modifications that improve building code compliance.
Implements Q-learning agent that learns to:
    - Expand undersized rooms
    - Widen narrow doors
    - Optimize overall compliance score
"""

import numpy as np
import random
from collections import defaultdict
import matplotlib.pyplot as plt

#import from Phase 3 (which imports Phase 2, which imports Phase 1)
try:
    from phase3_final_uncertainty import (
        BUILDING_CODES,
        load_image,
        parse_svg_file,
        IMAGE_PATH,
        SVG_PATH,
        run_probabilistic_analysis
    )
    #import from Phase 2 for compliance checking and visualization
    from phase2_final_constraint_checking import (
        check_building_code_compliance,
        visualize_with_violations
    )
    PHASES_AVAILABLE = True
    print("[✓] Phases 1, 2, 3 imported successfully")
except ImportError as e:
    print(f"[WARNING] Could not import previous phases: {e}")
    print("[INFO] Running in standalone mode")
    PHASES_AVAILABLE = False
    #minimal fallback
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

#environment: floor plan state
class FloorPlanEnvironment:
    """
    Environment for floor plan optimization
    State: current room sizes and door widths
    Actions: modify room dimensions or door widths
    Reward: improvement in compliance score
    """
    def __init__(self, rooms, doors, building_codes):
        #manual deep copy to avoid method reference issues
        self.original_rooms = []
        for room in rooms:
            room_copy = {
                'type': room['type'],
                'points': list(room['points']),
                'bbox': tuple(room['bbox']) if isinstance(room['bbox'], (tuple, list)) else room['bbox'],
                'area': float(room['area']),
                'center': tuple(room['center']),
                'full_class': room.get('full_class', '')
            }
            self.original_rooms.append(room_copy)

        self.original_doors = []
        for door in doors:
            door_copy = {
                'type': door['type'],
                'points': list(door['points']),
                'bbox': tuple(door['bbox']) if isinstance(door['bbox'], (tuple, list)) else door['bbox'],
                'center': tuple(door['center']),
                'width': float(door['width']),
                'length': float(door['length']),
                'orientation': door['orientation'],
                'full_class': door.get('full_class', '')
            }
            self.original_doors.append(door_copy)

        # Create working copies
        self.rooms = self._copy_rooms(self.original_rooms)
        self.doors = self._copy_doors(self.original_doors)
        self.building_codes = building_codes

        #action space
        self.actions = self.define_actions()

        #track history
        self.action_history = []
        self.reward_history = []

    def _copy_rooms(self, rooms):
        """Create a clean copy of rooms"""
        return [{
            'type': r['type'],
            'points': list(r['points']),
            'bbox': tuple(r['bbox']),
            'area': float(r['area']),
            'center': tuple(r['center']),
            'full_class': r.get('full_class', '')
        } for r in rooms]

    def _copy_doors(self, doors):
        """Create a clean copy of doors"""
        return [{
            'type': d['type'],
            'points': list(d['points']),
            'bbox': tuple(d['bbox']),
            'center': tuple(d['center']),
            'width': float(d['width']),
            'length': float(d['length']),
            'orientation': d['orientation'],
            'full_class': d.get('full_class', '')
        } for d in doors]

    def define_actions(self):
        """Define possible actions for modifying the floor plan"""
        actions = []

        #actions to expand undersized rooms only
        for ii, room in enumerate(self.rooms):
            area_m2 = room['area'] / 10000
            room_type = room['type']
            min_area = self.building_codes['min_room_areas'].get(
                room_type,
                self.building_codes['min_room_areas']['default']
            )
            #only add expand action if room is undersized or close
            if area_m2 < min_area * 1.1:
                actions.append(('expand_room', ii, 0.05))
                actions.append(('expand_room', ii, 0.10))

        #actions to widen narrow doors
        for ii, door in enumerate(self.doors):
            width_cm = max(door['bbox'][2], door['bbox'][3])
            if width_cm < 75:
                actions.append(('widen_door', ii, 5.0))
                actions.append(('widen_door', ii, 10.0))

        #if no targeted actions available, add some general ones
        if len(actions) == 0:
            for ii in range(len(self.rooms)):
                actions.append(('expand_room', ii, 0.05))
            for ii in range(len(self.doors)):
                actions.append(('widen_door', ii, 5.0))

        return actions

    def reset(self):
        """Reset environment to original state"""
        self.rooms = self._copy_rooms(self.original_rooms)
        self.doors = self._copy_doors(self.original_doors)
        self.action_history = []
        self.reward_history = []
        return self.get_state()

    def get_state(self):
        """Get current state representation"""
        #simple state: compliance status for each room and door
        state = []

        for room in self.rooms:
            area_m2 = room['area'] / 10000
            room_type = room['type']
            min_area = self.building_codes['min_room_areas'].get(
                room_type,
                self.building_codes['min_room_areas']['default']
            )
            compliance = 1.0 if area_m2 >= min_area else area_m2 / min_area
            state.append(compliance)

        for door in self.doors:
            width_cm = max(door['bbox'][2], door['bbox'][3])
            #simple heuristic for required width
            min_width = 70.0  # Assume interior doors
            compliance = 1.0 if width_cm >= min_width else width_cm / min_width
            state.append(compliance)

        return tuple(np.round(state, 2)) #discretize state

    def step(self, action_idx):
        """Execute action and return new state, reward, done"""
        action_type, target_idx, amount = self.actions[action_idx]

        #execute action
        if action_type == 'expand_room':
            self.rooms[target_idx]['area'] *= (1 + amount)
        elif action_type == 'widen_door':
            #increase the longer dimension (door width)
            bbox = list(self.doors[target_idx]['bbox'])
            if bbox[2] > bbox[3]:
                bbox[2] += amount
            else:
                bbox[3] += amount
            self.doors[target_idx]['bbox'] = tuple(bbox)

        #record action
        self.action_history.append((action_type, target_idx, amount))

        #calculate reward
        reward = self.calculate_reward()
        self.reward_history.append(reward)

        #check if done (max 100 actions or compliance achieved)
        done = len(self.action_history) >= 100 or self.check_compliance()

        new_state = self.get_state()

        return new_state, reward, done

    def calculate_reward(self):
        """Calculate reward based on compliance improvement"""
        compliance_score = 0
        num_violations = 0

        #check rooms
        for room in self.rooms:
            area_m2 = room['area'] / 10000
            room_type = room['type']
            min_area = self.building_codes['min_room_areas'].get(
                room_type,
                self.building_codes['min_room_areas']['default']
            )

            if area_m2 < min_area:
                deficit = min_area - area_m2
                #larger penalty for bigger deficits
                compliance_score -= deficit * 50
                num_violations += 1
            else:
                #small reward for being compliant
                excess = min(area_m2 - min_area, 2.0)
                compliance_score += excess * 5

        #check doors
        for door in self.doors:
            width_cm = max(door['bbox'][2], door['bbox'][3])
            min_width = 70.0

            if width_cm < min_width:
                deficit = min_width - width_cm
                #larger penalty for door violations
                compliance_score -= deficit * 3
                num_violations += 1
            else:
                #small reward for being compliant
                excess = min(width_cm - min_width, 10.0)
                compliance_score += excess * 0.5

        #full compliance bonus
        if num_violations == 0:
            compliance_score += 1000

        #penalty for excessive modifications (discourage unnecessary changes)
        compliance_score -= len(self.action_history) * 0.5

        return compliance_score

    def check_compliance(self):
        """Check if all requirements are met"""
        #check rooms
        for room in self.rooms:
            area_m2 = room['area'] / 10000
            room_type = room['type']
            min_area = self.building_codes['min_room_areas'].get(
                room_type,
                self.building_codes['min_room_areas']['default']
            )

            if area_m2 < min_area:
                return False

        #check doors
        for door in self.doors:
            width_cm = max(door['bbox'][2], door['bbox'][3])
            min_width = 70.0
            if width_cm < min_width:
                return False

        return True

    def get_violations(self):
        """Get current violations"""
        violations = []

        for ii, room in enumerate(self.rooms, 1):
            area_m2 = room['area'] / 10000
            room_type = room['type']
            min_area = self.building_codes['min_room_areas'].get(
                room_type,
                self.building_codes['min_room_areas']['default']
            )
            if area_m2 < min_area:
                violations.append({
                    'type': 'room',
                    'id': ii,
                    'name': room_type,
                    'actual': area_m2,
                    'required': min_area,
                    'deficit': min_area - area_m2
                })

        for ii, door in enumerate(self.doors, 1):
            width_cm = max(door['bbox'][2], door['bbox'][3])
            min_width = 70.0
            if width_cm < min_width:
                violations.append({
                    'type': 'door',
                    'id': ii,
                    'actual': width_cm,
                    'required': min_width,
                    'deficit': min_width - width_cm
                })

        return violations

    def deduplicate_actions(self):
        """Merge identical consecutive actions"""
        compacted = {}

        for action_type, target_idx, amount in self.action_history:
            key = (action_type, target_idx)
            compacted[key] = compacted.get(key, 0) + amount

        return [(k[0], k[1], v) for k, v in compacted.items()]

#Q-learning agent
class QLearningAgent:
    """
    Q-Learning agent for floor plan optimization
    """

    def __init__(self, num_actions, learning_rate=0.3, discount_factor=0.9,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        #Q-table: maps (state, action) -> Q-value
        self.q_table = defaultdict(lambda: np.zeros(num_actions))

        #training metrics
        self.episode_rewards = []
        self.episode_lengths = []

    def choose_action(self, state):
        """Choose action using epsilon-greedy policy"""
        if random.random() < self.epsilon:
            #explore: random action
            return random.randint(0, self.num_actions - 1)
        else:
            #exploit: best action
            return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state, done):
        """Update Q-table using Q-learning update rule"""
        current_q = self.q_table[state][action]

        if done:
            target_q = reward
        else:
            target_q = reward + self.discount_factor * np.max(self.q_table[next_state])

        #Q-learning update
        self.q_table[state][action] = current_q + self.learning_rate * (target_q - current_q)

    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train(self, env, num_episodes=100):
        """Train agent"""
        print("\n" + "=" * 70)
        print("PHASE 4: REINFORCEMENT LEARNING TRAINING")
        print("=" * 70 + "\n")

        print(f"Training Q-Learning agent for {num_episodes} episodes...")
        initial_violations = env.get_violations()
        print(f"Initial violations: {len(initial_violations)}\n")

        best_violations = len(initial_violations)
        best_episode = 0
        self.best_action_sequence = []

        for episode in range(num_episodes):
            state = env.reset()
            total_reward = 0
            steps = 0
            episode_actions = []

            while True:
                #choose and execute action
                action = self.choose_action(state)
                episode_actions.append(action)
                next_state, reward, done = env.step(action)

                #learn from experience
                self.learn(state, action, reward, next_state, done)

                total_reward += reward
                steps += 1
                state = next_state

                if done:
                    break

            #decay exploration
            self.decay_epsilon()

            #record metrics
            self.episode_rewards.append(total_reward)
            self.episode_lengths.append(steps)

            #track best performance
            violations = len(env.get_violations())
            if violations < best_violations:
                best_violations = violations
                best_episode = episode + 1
                self.best_action_sequence = episode_actions.copy()

            #print progress
            if (episode + 1) % 20 == 0:
                avg_reward = np.mean(self.episode_rewards[-20:])
                violations = len(env.get_violations())
                print(f"Episode {episode + 1}/{num_episodes} | "
                      f"Avg Reward: {avg_reward:.1f} | "
                      f"Violations: {violations} | "
                      f"Best: {best_violations} (ep {best_episode}) | "
                      f"Epsilon: {self.epsilon:.3f}")

        print("\n✓ Training complete!")
        print(f"Best performance: {best_violations} violations in episode {best_episode}")

    def get_best_policy(self, env):
        """Get best action sequence using learned policy"""
        env.reset()

        #use the best action sequence we found during training
        if hasattr(self, 'best_action_sequence') and self.best_action_sequence:
            print(f"[INFO] Using best action sequence from training ({len(self.best_action_sequence)} actions)")
            for action in self.best_action_sequence:
                env.step(action)
            return self.best_action_sequence

        #fallback: use learned Q-table with targeted actions
        state = env.get_state()
        actions = []

        for step in range(100):
            q_values = self.q_table[state]

            if np.any(q_values != 0):
                action = np.argmax(q_values)
            else:
                action = random.randint(0, self.num_actions - 1)

            actions.append(action)
            next_state, reward, done = env.step(action)
            state = next_state

            if done or len(env.get_violations()) == 0:
                break

        return actions

#reporting and visualization
def visualize_training(agent, show_plot=True):
    """Visualize training progress"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    #plot rewards
    ax1.plot(agent.episode_rewards, alpha=0.3, color='blue')
    window = 10
    if len(agent.episode_rewards) >= window:
        moving_avg = np.convolve(agent.episode_rewards,
                                 np.ones(window) / window, mode='valid')
        ax1.plot(range(window - 1, len(agent.episode_rewards)),
                 moving_avg, color='red', linewidth=2, label='Moving Average')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Training Progress: Rewards per Episode')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    #plot episode lengths
    ax2.plot(agent.episode_lengths, color='green')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps to Completion')
    ax2.set_title('Episode Lengths')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if show_plot:
        plt.show()
    else:
        plt.close()

def generate_optimization_report(env, agent):
    """Generate report of suggested improvements"""
    print("\n" + "=" * 70)
    print("RL AGENT RECOMMENDATIONS")
    print("=" * 70 + "\n")

    # Deduplicate actions
    dedup_actions = env.deduplicate_actions()

    print("📋 SUGGESTED MODIFICATIONS (deduplicated):\n")
    print(f"Total raw actions: {len(env.action_history)}")
    print(f"Deduplicated actions: {len(dedup_actions)}\n")

    for ii, (action_type, target_idx, amount) in enumerate(dedup_actions, 1):
        if action_type == 'expand_room':
            room = env.rooms[target_idx]
            print(f"{ii}. Expand {room['type']} (Room {target_idx + 1}) by {amount * 100:.0f}%")
        elif action_type == 'widen_door':
            print(f"{ii}. Widen Door {target_idx + 1} by {amount:.0f} cm")

    print("\n" + "=" * 70)
    print("BEFORE vs AFTER COMPARISON")
    print("=" * 70 + "\n")

    #compare violations
    env_original = FloorPlanEnvironment(env.original_rooms, env.original_doors, env.building_codes)
    violations_before = env_original.get_violations()
    violations_after = env.get_violations()

    print(f"Violations BEFORE: {len(violations_before)}")
    print(f"Violations AFTER:  {len(violations_after)}")
    print(f"Improvement: {len(violations_before) - len(violations_after)} violations fixed\n")

    if violations_after:
        print("⚠️  Remaining violations:")
        for v in violations_after:
            if v['type'] == 'room':
                print(f"  • {v['name']}: {v['actual']:.2f} m² < {v['required']:.2f} m² (deficit: {v['deficit']:.2f} m²)")
            else:
                print(f"  • Door {v['id']}: {v['actual']:.1f} cm < {v['required']:.1f} cm (deficit: {v['deficit']:.1f} cm)")
    else:
        print("🎉 All violations resolved! Floor plan is now COMPLIANT.")

    print("\n" + "=" * 70)

#main function
def run_rl_optimization(rooms, doors, building_codes, num_episodes=100, show_plots=True, verbose=True):
    """
    Run RL-based floor plan optimization
    """
    #create environment
    env = FloorPlanEnvironment(rooms, doors, building_codes)

    #create agent
    num_actions = len(env.actions)
    agent = QLearningAgent(num_actions)

    #train agent
    agent.train(env, num_episodes=num_episodes)

    #get best policy
    if verbose:
        print("\n[INFO] Applying learned policy to optimize floor plan...")
    best_actions = agent.get_best_policy(env)

    #generate report
    #if verbose:
    generate_optimization_report(env, agent)

    #visualize training (only if show_plots is True)
    if show_plots:
        if verbose:
            print("\n[INFO] Generating training visualization...")
        visualize_training(agent, show_plot=True)

    if verbose:
        print("\n✓ Phase 4 Complete!")
        print("=" * 70 + "\n")

    return env, agent

def main():
    """
    Main execution pipeline - runs all four phases
    """
    if not PHASES_AVAILABLE:
        print("[ERROR] Previous phases not available. Cannot run integrated pipeline.")
        return

    print("\n" + "=" * 70)
    print("COMPLETE PIPELINE: PHASES 1 + 2 + 3 + 4")
    print("AI-DRIVEN BUILDING CODE COMPLIANCE CHECKER")
    print("=" * 70 + "\n")

    # ===== PHASE 1: Parse Floor Plan =====
    print("=" * 70)
    print("PHASE 1: FLOOR PLAN PARSING")
    print("=" * 70 + "\n")

    image = load_image(IMAGE_PATH)
    rooms, doors = parse_svg_file(SVG_PATH)

    if len(rooms) == 0:
        print("[WARNING] No rooms detected!")
        return

    print(f"\n[✓] Phase 1 Complete: {len(rooms)} rooms, {len(doors)} doors detected")

    # ===== PHASE 2: Check Building Codes =====
    violations = check_building_code_compliance(rooms, doors)

    # ===== PHASE 3: Probabilistic Analysis =====
    room_results, door_results = run_probabilistic_analysis(rooms, doors, BUILDING_CODES)

    # ===== PHASE 4: RL Optimization =====
    if violations:
        print("\n" + "=" * 70)
        print("STARTING PHASE 4: RL-BASED OPTIMIZATION")
        print("=" * 70)
        print(f"\n{len(violations)} violations detected. Training RL agent to suggest improvements...\n")

        env, agent = run_rl_optimization(rooms, doors, BUILDING_CODES, num_episodes=100)
    else:
        print("\n✅ No violations found! Floor plan is already compliant.")
        print("   Skipping Phase 4 optimization.")

    # ===== Final Summary =====
    print("\n" + "=" * 70)
    print("✓ ALL PHASES COMPLETE!")
    print("=" * 70)
    print("\nComplete Analysis Summary:")
    print(f"  • Rooms detected: {len(rooms)}")
    print(f"  • Doors detected: {len(doors)}")
    print(f"  • Initial violations: {len(violations)}")

    if violations:
        print(f"  • RL agent trained for optimization")
        print(f"  • Suggested modifications: {len(env.action_history)}")
        final_violations = len(env.get_violations())
        print(f"  • Final violations: {final_violations}")
        print(f"  • Improvement: {len(violations) - final_violations} violations fixed")

    print("\n" + "=" * 70 + "\n")

if __name__ == "__main__":
    if PHASES_AVAILABLE:
        main()
    else:
        print("Phase 4: RL Agent Module")
        print("Import this module with: from phase4_rl_agent import run_rl_optimization")
        print("This module requires phase3_final_uncertainty.py to run the full pipeline.")