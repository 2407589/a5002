
import random
import json
import os
from collections import defaultdict


class QLearningAgent:
    """Q-Learning agent for reinforcement learning"""

    def __init__(self, agent_name, learning_rate=0.1, discount_factor=0.95,
                 exploration_rate=1.0, exploration_decay=0.995, min_exploration=0.05):
        """
        Initialize Q-Learning agent

        Args:
            agent_name (str): Name of the agent
            learning_rate (float): Learning rate (alpha)
            discount_factor (float): Discount factor (gamma)
            exploration_rate (float): Initial exploration rate (epsilon)
            exploration_decay (float): Decay rate for exploration
            min_exploration (float): Minimum exploration rate
        """
        self.agent_name = agent_name
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration = min_exploration

        # Q-table: {(state, action): value}
        self.q_table = defaultdict(float)

        # Statistics
        self.total_episodes = 0
        self.total_rewards = []
        self.learning_history = []


    def get_state_representation(self, agent, environment):
        """Simplified state for better learning"""

        # Health buckets (fewer = faster learning)
        if agent.health > 150:
            health_state = "high"
        elif agent.health > 80:
            health_state = "medium"
        else:
            health_state = "low"

        # Stamina buckets
        stamina_state = "high" if agent.stamina > 50 else "low"

        # Honor buckets - CRITICAL for learning
        if agent.honor >= 80:
            honor_state = "ready"  # Can challenge adversary
        elif agent.honor >= 20:
            honor_state = "building"
        else:
            honor_state = "dishonored"

        # Threat assessment - SIMPLIFIED
        x, y = agent.position
        nearby = environment.get_agents_in_range(x, y, 4)  # Reduced range

        monsters = [a for a in nearby if a != agent and a.is_alive()
                    and a.agent_type == 'monster']

        if len(monsters) >= 2:
            threat = "high"
        elif len(monsters) == 1:
            threat = "medium"
        else:
            threat = "none"

        # Simple 4-tuple state (much faster learning)
        return (health_state, stamina_state, honor_state, threat)

    def get_available_actions(self, agent):
        """Simplified action set"""
        actions = ['hunt', 'rest']  # Always available

        if agent.health < 40:
            actions.append('retreat')

        if agent.honor >= 80 and agent.health > 100:
            actions.append('challenge_adversary')

        return actions

    def choose_action(self, state, available_actions):
        """
        Choose action using epsilon-greedy policy

        Args:
            state (tuple): Current state
            available_actions (list): Available actions

        Returns:
            str: Chosen action
        """
        # Exploration: random action
        if random.random() < self.exploration_rate:
            return random.choice(available_actions)

        # Exploitation: best known action
        q_values = {action: self.q_table[(state, action)] for action in available_actions}
        max_q = max(q_values.values())

        # Get all actions with max Q-value (handle ties)
        best_actions = [action for action, q in q_values.items() if q == max_q]

        return random.choice(best_actions)

    def learn(self, state, action, reward, next_state, next_available_actions, done=False):
        """
        Update Q-value using Q-learning update rule

        Args:
            state (tuple): Current state
            action (str): Action taken
            reward (float): Reward received
            next_state (tuple): Next state
            next_available_actions (list): Available actions in next state
            done (bool): Whether episode is done
        """
        current_q = self.q_table[(state, action)]

        # Calculate max Q-value for next state
        if done:
            max_next_q = 0
        else:
            next_q_values = [self.q_table[(next_state, a)] for a in next_available_actions]
            max_next_q = max(next_q_values) if next_q_values else 0

        # Q-learning update rule
        new_q = current_q + self.learning_rate * (
                reward + self.discount_factor * max_next_q - current_q
        )

        self.q_table[(state, action)] = new_q

        # Record learning
        self.learning_history.append({
            'state': state,
            'action': action,
            'reward': reward,
            'q_value': new_q
        })

    def decay_exploration(self):
        """Decay exploration rate"""
        self.exploration_rate = max(
            self.min_exploration,
            self.exploration_rate * self.exploration_decay
        )

    def _calculate_detailed_reward(self, agent, action, prev_state_info,
                                   action_success, killed_target):
        """Simplified reward for faster learning"""
        reward = 0

        # COMBAT REWARDS (Primary signal)
        if killed_target:
            honor_gained = agent.honor - prev_state_info['honor']
            reward += 50 + (honor_gained * 10)  # Huge positive signal

        # Damage penalty (but not too harsh)
        health_change = agent.health - prev_state_info['health']
        if health_change < 0:
            reward += health_change * 0.1  # Small penalty

        # Death penalty
        if not agent.is_alive():
            reward -= 100

        # Milestone bonuses
        if agent.honor == 50:  # Just reached respect
            reward += 50
        elif agent.honor == 80:  # Ready for adversary
            reward += 100

        # Quest completion
        if killed_target and hasattr(agent, 'kills') and agent.kills.get('adversary', 0) > 0:
            reward += 1000  # MASSIVE reward for winning

        return reward

    def end_episode(self, total_reward):
        """
        Mark end of episode and record statistics

        Args:
            total_reward (float): Total reward for episode
        """
        self.epsilon = max(self.min_exploration, self.exploration_rate * self.exploration_decay)
        self.total_episodes += 1
        self.total_rewards.append(total_reward)
        self.decay_exploration()

    def get_statistics(self):
        """Get learning statistics"""
        if not self.total_rewards:
            return {
                'episodes': 0,
                'avg_reward': 0,
                'exploration_rate': self.exploration_rate
            }

        return {
            'episodes': self.total_episodes,
            'avg_reward': sum(self.total_rewards) / len(self.total_rewards),
            'recent_avg_reward': sum(self.total_rewards[-10:]) / min(10, len(self.total_rewards)),
            'exploration_rate': self.exploration_rate,
            'q_table_size': len(self.q_table),
            'total_reward': sum(self.total_rewards)
        }

    def save_q_table(self, filepath):
        """
        Save Q-table to file

        Args:
            filepath (str): Path to save file
        """
        # Convert defaultdict to regular dict with string keys
        q_table_serializable = {
            str(k): v for k, v in self.q_table.items()
        }

        data = {
            'agent_name': self.agent_name,
            'q_table': q_table_serializable,
            'exploration_rate': self.exploration_rate,
            'total_episodes': self.total_episodes,
            'statistics': self.get_statistics()
        }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def load_q_table(self, filepath):
        """
        Load Q-table from file

        Args:
            filepath (str): Path to load file
        """
        if not os.path.exists(filepath):
            print(f"Q-table file not found: {filepath}")
            return False

        with open(filepath, 'r') as f:
            data = json.load(f)

        # Convert string keys back to tuples
        self.q_table = defaultdict(float)
        for k, v in data['q_table'].items():
            # Parse string representation back to tuple
            key = eval(k)
            self.q_table[key] = v

        self.exploration_rate = data.get('exploration_rate', self.exploration_rate)
        self.total_episodes = data.get('total_episodes', 0)

        print(f"Loaded Q-table for {self.agent_name}: {len(self.q_table)} entries")
        return True

    def get_best_action_for_state(self, state, available_actions):
        """
        Get best action for a state (no exploration)

        Args:
            state (tuple): State
            available_actions (list): Available actions

        Returns:
            str: Best action
        """
        q_values = {action: self.q_table[(state, action)] for action in available_actions}
        return max(q_values, key=q_values.get)

    def reset_learning(self):
        """Reset learning (for new training session)"""
        self.q_table = defaultdict(float)
        self.total_episodes = 0
        self.total_rewards = []
        self.learning_history = []
        self.exploration_rate = 0.3  # Reset to initial value


class QLearningIntegration:
    """Helper class to integrate Q-learning with simulation"""

    @staticmethod
    def create_learning_agent(agent):
        """
        Create Q-learning wrapper for an agent

        Args:
            agent: The agent to wrap

        Returns:
            QLearningAgent: Q-learning agent
        """
        return QLearningAgent(
            agent_name=agent.name,
            learning_rate=0.1,
            discount_factor=0.95,
            exploration_rate=0.3
        )

    @staticmethod
    def execute_q_learning_turn(agent, q_agent, environment, previous_state):
        """
        Execute one turn with Q-learning

        Args:
            agent: The game agent
            q_agent: The Q-learning agent
            environment: Game environment
            previous_state: Previous agent state

        Returns:
            dict: Turn results
        """
        # Get current state
        current_state = q_agent.get_state_representation(agent, environment)

        # Get available actions
        available_actions = q_agent.get_available_actions(agent)

        # Choose action
        action = q_agent.choose_action(current_state, available_actions)

        # Execute action (this should be done by the simulation)
        action_result = {'action': action, 'state': current_state}

        return action_result

    @staticmethod
    def update_q_learning(agent, q_agent, environment, previous_state,
                          action, current_state_dict):
        """
        Update Q-learning after action execution

        Args:
            agent: The game agent
            q_agent: The Q-learning agent
            environment: Game environment
            previous_state: Previous state tuple
            action: Action that was taken
            current_state_dict: Current agent state dictionary
        """
        # Get new state
        new_state = q_agent.get_state_representation(agent, environment)

        # Calculate reward
        reward = q_agent.calculate_reward(agent, action, current_state_dict, environment)

        # Get available actions for new state
        next_actions = q_agent.get_available_actions(agent)

        # Learn
        done = not agent.is_alive()
        q_agent.learn(previous_state, action, reward, new_state, next_actions, done)

        return reward