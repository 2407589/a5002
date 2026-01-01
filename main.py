
import sys
import os
import time
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Make sure we can import from subdirectories
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import simulation and learning modules
from core.simulation import Simulation, VerbosityLevel
from ai.q_learning import QLearningAgent, QLearningIntegration
from ai.adaptive_learning import AdaptiveLearning


class LearningSimulation(Simulation):
    """Enhanced simulation with Q-learning and adaptive learning"""

    def __init__(self, enable_q_learning=True, enable_adaptive=True, verbosity=VerbosityLevel.QUIET):
        super().__init__(verbosity=verbosity)
        self.enable_q_learning = enable_q_learning
        self.enable_adaptive = enable_adaptive

        # Q-learning agents
        self.q_agents = {}

        # Adaptive learning for adversary
        self.adaptive_system = None

        # Learning metrics
        self.learning_metrics = {
            'episode_rewards': [],
            'episode_honors': [],
            'episode_lengths': [],
            'episode_kills': [],
            'success_rate': [],
            'q_values': [],
            'exploration_rates': [],
            'adversary_strategies': [],
            'counter_strategies': []
        }

        self.current_episode_reward = 0
        self.current_episode_kills = 0
        self.previous_states = {}

    def initialize_learning_systems(self):
        """Initialize Q-learning and adaptive learning systems"""
        if self.enable_q_learning and self.dek:
            # Create Q-learning agent for Dek with better parameters
            self.q_agents['dek'] = QLearningAgent(
                agent_name='Dek',
                learning_rate=0.3,  # Increased for faster learning
                discount_factor=0.85,  # Reduced for immediate rewards
                exploration_rate=0.5,  # Start with some exploitation
                exploration_decay=0.99,
                min_exploration=0.15
            )

            # Try to load existing Q-table
            q_table_path = os.path.join(project_root, 'data', 'q_tables', 'dek_q_table.json')
            if os.path.exists(q_table_path):
                self.q_agents['dek'].load_q_table(q_table_path)
                if self.verbosity >= VerbosityLevel.QUIET:
                    print(f"Loaded existing Q-table for Dek")

        if self.enable_adaptive and self.adversary:
            # Create adaptive learning for adversary
            self.adaptive_system = AdaptiveLearning(memory_size=50, pattern_threshold=0.6)
            if self.verbosity >= VerbosityLevel.QUIET:
                print(f"Initialized adaptive learning for Ultimate Adversary")

    def execute_agent_turn(self, agent):
        """Override to integrate Q-learning"""
        if not agent.is_alive():
            return

        # SPECIAL HANDLING FOR DEK WITH Q-LEARNING
        if agent.name == 'Dek' and self.enable_q_learning and 'dek' in self.q_agents:
            self._execute_q_learning_turn(agent)
        else:
            # Standard agent turn for others
            super().execute_agent_turn(agent)

        # Observe Dek's actions for adaptive learning
        if self.enable_adaptive and self.adaptive_system and agent.name == 'Dek':
            action_decision = agent.decide_action(self.environment)
            self.adaptive_system.observe_player_action(agent, action_decision.get('action', 'idle'), self.environment)

    def _execute_q_learning_turn(self, agent):
        """Execute Q-learning turn with FIXED combat logic"""
        q_agent = self.q_agents['dek']

        # Get current state
        current_state = q_agent.get_state_representation(agent, self.environment)

        # Initialize previous state if needed
        if agent.name not in self.previous_states:
            self.previous_states[agent.name] = {
                'state': current_state,
                'action': None,
                'health': agent.health,
                'honor': agent.honor,
                'dishonor_acts': getattr(agent, 'dishonor_acts', 0),
                'kills': 0
            }

        prev_state_info = self.previous_states[agent.name]
        prev_state = prev_state_info['state']
        prev_action = prev_state_info['action']

        # Get available actions
        available_actions = q_agent.get_available_actions(agent)

        # Choose action using Q-learning
        chosen_action = q_agent.choose_action(current_state, available_actions)

        # EXECUTE THE ACTION
        action_success, killed_target = self._execute_q_action(agent, chosen_action)

        # Track kills
        if killed_target:
            self.current_episode_kills += 1
            prev_state_info['kills'] += 1

        # Calculate reward AFTER action execution
        reward = self._calculate_detailed_reward(
            agent,
            chosen_action,
            prev_state_info,
            action_success,
            killed_target
        )

        # Update Q-learning
        new_state = q_agent.get_state_representation(agent, self.environment)
        next_actions = q_agent.get_available_actions(agent)
        done = not agent.is_alive()

        # Learn from previous action if there was one
        if prev_action is not None:
            q_agent.learn(prev_state, prev_action, reward, new_state, next_actions, done)

        # Update metrics
        self.current_episode_reward += reward

        # Update previous state
        self.previous_states[agent.name] = {
            'state': new_state,
            'action': chosen_action,
            'health': agent.health,
            'honor': agent.honor,
            'dishonor_acts': getattr(agent, 'dishonor_acts', 0),
            'kills': prev_state_info['kills']
        }

    def _execute_q_action(self, agent, q_action):
        """Execute Q-learning action - FIXED VERSION"""
        x, y = agent.position
        action_success = False
        killed_target = False

        # Get nearby enemies
        nearby = self.environment.get_agents_in_range(x, y, 6)
        monsters = [m for m in nearby if m.agent_type == 'monster' and m.is_alive()]

        # Check for adversary if honor is high enough
        adversary = None
        if agent.honor >= 80:
            for a in self.environment.agents:
                if hasattr(a, 'agent_type') and a.agent_type == 'adversary' and a.is_alive():
                    adversary = a
                    break

        # PRIORITIZE ADVERSARY IF READY
        if q_action == 'challenge_adversary' and adversary:
            dist = self.environment.get_distance(x, y, adversary.position[0], adversary.position[1])
            if dist <= 1:
                killed_target = self._execute_attack(agent, adversary)
                action_success = True
            else:
                moved = self._move_towards(agent, adversary.position)
                action_success = moved

        # HUNT ACTION
        elif q_action in ['hunt', 'hunt_for_honor']:
            target = None

            # Prefer adversary if ready
            if adversary and agent.honor >= 80:
                target = adversary
            elif monsters:
                # Find target based on action type
                if q_action == 'hunt_for_honor':
                    # Target larger monsters for honor
                    large = [m for m in monsters if m.size == 'large']
                    medium = [m for m in monsters if m.size == 'medium']
                    target = large[0] if large else (medium[0] if medium else monsters[0])
                else:
                    # Target nearest monster
                    target = min(monsters, key=lambda m:
                    self.environment.get_distance(x, y, m.position[0], m.position[1]))

            if target:
                dist = self.environment.get_distance(x, y, target.position[0], target.position[1])
                if dist <= 1:
                    killed_target = self._execute_attack(agent, target)
                    action_success = True
                else:
                    moved = self._move_towards(agent, target.position)
                    action_success = moved

        # REST ACTION
        elif q_action == 'rest':
            agent.rest()
            action_success = True

        # RETREAT ACTION
        elif q_action == 'retreat':
            self._execute_flee(agent)
            action_success = True

        # DEFAULT: PATROL
        else:
            self._execute_patrol(agent)
            action_success = False

        return action_success, killed_target

    def _calculate_detailed_reward(self, agent, action, prev_state_info,
                                   action_success, killed_target):
        """Calculate reward for Q-learning"""
        reward = 0

        # HUGE REWARDS FOR KILLING
        if killed_target:
            honor_gained = agent.honor - prev_state_info['honor']
            reward += 100 + (honor_gained * 5)

        # Health damage penalty
        health_change = agent.health - prev_state_info['health']
        if health_change < 0:
            reward += health_change * 0.2

        # Death penalty
        if not agent.is_alive():
            reward -= 200

        # Honor milestones
        if agent.honor >= 50 and prev_state_info['honor'] < 50:
            reward += 100
        elif agent.honor >= 80 and prev_state_info['honor'] < 80:
            reward += 200

        # Quest completion
        if killed_target and hasattr(agent, 'kills') and agent.kills.get('adversary', 0) > 0:
            reward += 2000

        # Small penalty for inaction
        if action == 'rest' and agent.health > 150:
            reward -= 5

        return reward

    def save_learning_data(self):
        """Save Q-table and metrics"""
        # Save Q-table
        if 'dek' in self.q_agents:
            os.makedirs(os.path.join(project_root, 'data', 'q_tables'), exist_ok=True)
            q_table_path = os.path.join(project_root, 'data', 'q_tables', 'dek_q_table.json')
            self.q_agents['dek'].save_q_table(q_table_path)

            if self.verbosity >= VerbosityLevel.NORMAL:
                print(f"Saved Q-table: {len(self.q_agents['dek'].q_table)} entries")

        # Record metrics
        self.learning_metrics['episode_rewards'].append(self.current_episode_reward)
        self.learning_metrics['episode_honors'].append(self.dek.honor if self.dek else -10)
        self.learning_metrics['episode_lengths'].append(self.stats['turns_elapsed'])
        self.learning_metrics['episode_kills'].append(self.current_episode_kills)

        if 'dek' in self.q_agents:
            q_stats = self.q_agents['dek'].get_statistics()
            self.learning_metrics['exploration_rates'].append(q_stats['exploration_rate'])
            if len(self.q_agents['dek'].q_table) > 0:
                avg_q = sum(self.q_agents['dek'].q_table.values()) / len(self.q_agents['dek'].q_table)
                self.learning_metrics['q_values'].append(avg_q)

        # Calculate success rate (last 10 episodes)
        recent_episodes = self.learning_metrics['episode_honors'][-10:]
        successes = sum(1 for h in recent_episodes if h >= 100)
        self.learning_metrics['success_rate'].append(successes / len(recent_episodes) if recent_episodes else 0)


def visualize_learning_metrics(metrics, save_path=None):
    """Create comprehensive visualization of learning metrics"""
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle('Predator: Badlands - Learning Metrics', fontsize=16, fontweight='bold')

    episodes = list(range(1, len(metrics['episode_rewards']) + 1))

    # 1. Episode Rewards
    ax = axes[0, 0]
    ax.plot(episodes, metrics['episode_rewards'], 'b-', alpha=0.3, label='Raw')
    if len(metrics['episode_rewards']) >= 5:
        smoothed = np.convolve(metrics['episode_rewards'], np.ones(5) / 5, mode='valid')
        ax.plot(range(5, len(metrics['episode_rewards']) + 1), smoothed, 'b-', linewidth=2, label='Smoothed')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.set_title('Episode Rewards')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Honor Progression
    ax = axes[0, 1]
    ax.plot(episodes, metrics['episode_honors'], 'g-', alpha=0.3)
    if len(metrics['episode_honors']) >= 5:
        smoothed = np.convolve(metrics['episode_honors'], np.ones(5) / 5, mode='valid')
        ax.plot(range(5, len(metrics['episode_honors']) + 1), smoothed, 'g-', linewidth=2)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3, label='Neutral')
    ax.axhline(y=50, color='orange', linestyle='--', alpha=0.5, label='Respect')
    ax.axhline(y=100, color='red', linestyle='--', alpha=0.5, label='Warrior')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Honor')
    ax.set_title('Honor Progression')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Kills Per Episode
    ax = axes[0, 2]
    ax.plot(episodes, metrics['episode_kills'], 'r-', alpha=0.5, marker='o', markersize=3)
    if len(metrics['episode_kills']) >= 5:
        smoothed = np.convolve(metrics['episode_kills'], np.ones(5) / 5, mode='valid')
        ax.plot(range(5, len(metrics['episode_kills']) + 1), smoothed, 'r-', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Kills')
    ax.set_title('Monsters Killed Per Episode')
    ax.grid(True, alpha=0.3)

    # 4. Success Rate
    ax = axes[1, 0]
    if metrics['success_rate']:
        ax.plot(episodes, metrics['success_rate'], 'purple', linewidth=2)
        ax.fill_between(episodes, metrics['success_rate'], alpha=0.3)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Success Rate')
    ax.set_title('Success Rate (10-Episode Average)')
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3)

    # 5. Episode Lengths
    ax = axes[1, 1]
    ax.plot(episodes, metrics['episode_lengths'], 'orange', alpha=0.5, marker='o', markersize=3)
    if len(metrics['episode_lengths']) >= 5:
        smoothed = np.convolve(metrics['episode_lengths'], np.ones(5) / 5, mode='valid')
        ax.plot(range(5, len(metrics['episode_lengths']) + 1), smoothed, 'orange', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Turns Survived')
    ax.set_title('Episode Lengths')
    ax.grid(True, alpha=0.3)

    # 6. Exploration Rate
    ax = axes[1, 2]
    if metrics['exploration_rates']:
        ax.plot(episodes, metrics['exploration_rates'], 'cyan', linewidth=2)
        ax.fill_between(episodes, metrics['exploration_rates'], alpha=0.3)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Exploration Rate')
    ax.set_title('Q-Learning Exploration Decay')
    ax.grid(True, alpha=0.3)

    # 7. Average Q-Values
    ax = axes[2, 0]
    if metrics['q_values']:
        ax.plot(episodes, metrics['q_values'], 'magenta', linewidth=2)
        ax.fill_between(episodes, metrics['q_values'], alpha=0.3)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Average Q-Value')
    ax.set_title('Q-Value Evolution')
    ax.grid(True, alpha=0.3)

    # 8. Reward vs Honor Correlation
    ax = axes[2, 1]
    if len(metrics['episode_rewards']) > 0:
        ax.scatter(metrics['episode_honors'], metrics['episode_rewards'], alpha=0.5)
        ax.set_xlabel('Final Honor')
        ax.set_ylabel('Total Reward')
        ax.set_title('Reward vs Honor Correlation')
        ax.grid(True, alpha=0.3)

    # 9. Performance Summary
    ax = axes[2, 2]
    ax.axis('off')
    if len(metrics['episode_rewards']) > 0:
        summary_text = f"""
LEARNING SUMMARY

Episodes: {len(episodes)}
Avg Reward: {np.mean(metrics['episode_rewards']):.1f}
Avg Honor: {np.mean(metrics['episode_honors']):.1f}
Avg Kills: {np.mean(metrics['episode_kills']):.1f}
Avg Length: {np.mean(metrics['episode_lengths']):.1f}

Best Honor: {max(metrics['episode_honors'])}
Best Kills: {max(metrics['episode_kills'])}
Longest Run: {max(metrics['episode_lengths'])}

Final Success: {metrics['success_rate'][-1] * 100:.1f}%
        """
        ax.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                verticalalignment='center')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Saved: {save_path}")

    plt.close()  # Don't show, just save


def run_learning_session(num_episodes=10, verbosity=VerbosityLevel.QUIET):
    """Run multiple episodes with learning enabled"""
    print("\n" + "=" * 80)
    print("PREDATOR: BADLANDS - LEARNING SESSION")
    print("=" * 80)
    print(f"Episodes: {num_episodes} | Q-Learning: ENABLED | Adaptive: ENABLED")
    print(f"Verbosity: {['SILENT', 'QUIET', 'NORMAL', 'VERBOSE'][verbosity]}")
    print("=" * 80 + "\n")

    all_metrics = {
        'episode_rewards': [],
        'episode_honors': [],
        'episode_lengths': [],
        'episode_kills': [],
        'success_rate': [],
        'q_values': [],
        'exploration_rates': [],
        'adversary_strategies': [],
        'counter_strategies': []
    }

    for episode in range(num_episodes):
        print(f"\n{'=' * 60}\nEpisode {episode + 1}/{num_episodes}\n{'=' * 60}")

        sim = LearningSimulation(enable_q_learning=True, enable_adaptive=True, verbosity=verbosity)
        sim.start()
        sim.initialize_learning_systems()

        while sim.running and sim.stats['turns_elapsed'] < 200:
            sim.run_turn()

        # Episode summary
        print(f"\n✓ Episode {episode + 1} Complete:")
        print(f"  Success: {'YES' if sim.quest_completed else 'NO'}")
        print(f"  Turns: {sim.stats['turns_elapsed']}")
        print(f"  Honor: {sim.dek.honor if sim.dek else -10}")
        print(f"  Kills: {sim.current_episode_kills}")
        print(f"  Reward: {sim.current_episode_reward:.1f}")

        if 'dek' in sim.q_agents:
            q_stats = sim.q_agents['dek'].get_statistics()
            q_table_size = q_stats.get('q_table_size', len(sim.q_agents['dek'].q_table))
            print(f"  Q-Table: {q_table_size} entries")
            print(f"  Explore: {q_stats['exploration_rate']:.3f}")

        # Aggregate metrics - FIX: properly extend the lists
        all_metrics['episode_rewards'].append(sim.current_episode_reward)
        all_metrics['episode_honors'].append(sim.dek.honor if sim.dek else -10)
        all_metrics['episode_lengths'].append(sim.stats['turns_elapsed'])
        all_metrics['episode_kills'].append(sim.current_episode_kills)

        if 'dek' in sim.q_agents:
            q_stats = sim.q_agents['dek'].get_statistics()
            all_metrics['exploration_rates'].append(q_stats['exploration_rate'])
            if len(sim.q_agents['dek'].q_table) > 0:
                avg_q = sum(sim.q_agents['dek'].q_table.values()) / len(sim.q_agents['dek'].q_table)
                all_metrics['q_values'].append(avg_q)

        # Calculate success rate
        recent_honors = all_metrics['episode_honors'][-10:]
        successes = sum(1 for h in recent_honors if h >= 100)
        all_metrics['success_rate'].append(successes / len(recent_honors) if recent_honors else 0)

        sim.save_learning_data()

    # Generate visualizations
    print("\n" + "=" * 80)
    print("📊 Generating Visualizations...")
    print("=" * 80)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(project_root, 'output', 'visualizations')
    os.makedirs(output_dir, exist_ok=True)

    metrics_path = os.path.join(output_dir, f'learning_metrics_{timestamp}.png')
    visualize_learning_metrics(all_metrics, save_path=metrics_path)

    # Print final statistics
    print("\n" + "=" * 80)
    print("🎯 LEARNING SESSION COMPLETE")
    print("=" * 80)
    print(f"Episodes: {num_episodes}")

    if all_metrics['episode_rewards']:
        print(f"Avg Reward: {np.mean(all_metrics['episode_rewards']):.1f}")
        print(f"Avg Honor: {np.mean(all_metrics['episode_honors']):.1f}")
        print(f"Avg Kills: {np.mean(all_metrics['episode_kills']):.1f}")
        print(f"Best Honor: {max(all_metrics['episode_honors'])}")
        print(f"Best Kills: {max(all_metrics['episode_kills'])}")
        if all_metrics['success_rate']:
            print(f"Final Success Rate: {all_metrics['success_rate'][-1] * 100:.1f}%")
    else:
        print("No metrics data collected")

    print("=" * 80)


def main():
    """Main function"""
    print("\n" + "=" * 80)
    print("PREDATOR: BADLANDS - AI LEARNING SIMULATION")
    print("=" * 80)
    print("\nSelect Training Mode:")
    print("1. Quick Test (5 episodes)")
    print("2. Standard Training (10 episodes)")
    print("3. Extended Training (25 episodes)")
    print("4. Full Training (50 episodes)")

    choice = input("\nChoice (1-4): ").strip()

    episodes_map = {'1': 5, '2': 10, '3': 25, '4': 50}
    num_episodes = episodes_map.get(choice, 10)

    # Set verbosity to NORMAL by default
    verbosity = VerbosityLevel.NORMAL

    try:
        run_learning_session(num_episodes=num_episodes, verbosity=verbosity)
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()