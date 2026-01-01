"""
Adaptive learning system for adversary that learns and counters player strategies
"""
import random
from collections import deque, Counter


class AdaptiveLearning:
    """Adaptive learning system for the Ultimate Adversary"""

    def __init__(self, memory_size=50, pattern_threshold=0.6):
        """
        Initialize adaptive learning system

        Args:
            memory_size (int): Number of recent actions to remember
            pattern_threshold (float): Threshold for pattern recognition (0-1)
        """
        self.memory_size = memory_size
        self.pattern_threshold = pattern_threshold

        # Action history memory
        self.player_action_history = deque(maxlen=memory_size)
        self.player_pattern_memory = []

        # Strategy tracking
        self.detected_strategies = {}
        self.current_player_strategy = "unknown"
        self.counter_strategy = "balanced"

        # Performance tracking
        self.strategy_effectiveness = {
            'area_denial': {'uses': 0, 'hits': 0, 'damage_dealt': 0},
            'burst_damage': {'uses': 0, 'hits': 0, 'damage_dealt': 0},
            'close_gap': {'uses': 0, 'hits': 0, 'damage_dealt': 0},
            'aggressive_push': {'uses': 0, 'hits': 0, 'damage_dealt': 0},
            'defensive': {'uses': 0, 'hits': 0, 'damage_dealt': 0},
            'balanced': {'uses': 0, 'hits': 0, 'damage_dealt': 0}
        }

        # Counter-strategy mappings
        self.counter_strategies = {
            'hit_and_run': 'area_denial',
            'frontal_assault': 'burst_damage',
            'ranged_attack': 'close_gap',
            'defensive_play': 'aggressive_push',
            'guerrilla': 'area_denial',
            'sustained_pressure': 'defensive',
            'unknown': 'balanced'
        }

        # Learning parameters
        self.adaptation_rate = 0.15
        self.pattern_window = 10  # Analyze last 10 actions

    def observe_player_action(self, player_agent, action, environment):
        """
        Observe and record player action

        Args:
            player_agent: The player agent (Dek)
            action (str): Action taken
            environment: Game environment
        """
        observation = {
            'action': action,
            'player_health': player_agent.health,
            'player_stamina': player_agent.stamina,
            'player_position': player_agent.position,
            'player_honor': getattr(player_agent, 'honor', 0),
            'distance_to_adversary': 0,  # Will be calculated
            'turn': environment.turn_count if hasattr(environment, 'turn_count') else 0
        }

        self.player_action_history.append(observation)

    def detect_player_strategy(self):
        """
        Analyze recent actions to detect player's strategy

        Returns:
            str: Detected strategy name
        """
        if len(self.player_action_history) < self.pattern_window:
            return "unknown"

        recent_actions = list(self.player_action_history)[-self.pattern_window:]

        # Extract action types
        actions = [obs['action'] for obs in recent_actions]
        action_counts = Counter(actions)

        # Analyze movement patterns
        attack_count = sum(1 for a in actions if a in ['attack', 'hunt'])
        retreat_count = sum(1 for a in actions if a in ['retreat', 'flee'])
        rest_count = sum(1 for a in actions if a == 'rest')
        patrol_count = sum(1 for a in actions if a == 'patrol')

        # Calculate health trend
        health_values = [obs['player_health'] for obs in recent_actions]
        health_decreasing = health_values[-1] < health_values[0]

        # Strategy detection logic
        total_actions = len(actions)

        # Hit and run: alternating attacks and retreats
        if attack_count > 3 and retreat_count > 3:
            return 'hit_and_run'

        # Frontal assault: high attacks, low retreats
        if attack_count >= total_actions * 0.7 and retreat_count <= 2:
            return 'frontal_assault'

        # Defensive play: lots of rest and retreat
        if (rest_count + retreat_count) >= total_actions * 0.6:
            return 'defensive_play'

        # Sustained pressure: consistent attacks with occasional rest
        if attack_count >= total_actions * 0.5 and rest_count > 0 and retreat_count <= 1:
            return 'sustained_pressure'

        # Guerrilla: mix of patrol and quick attacks
        if patrol_count >= total_actions * 0.4 and attack_count >= 2:
            return 'guerrilla'

        # Ranged/cautious: staying at distance
        avg_distance = sum(obs['distance_to_adversary'] for obs in recent_actions) / len(recent_actions)
        if avg_distance > 5 and attack_count > 2:
            return 'ranged_attack'

        return 'unknown'

    def select_counter_strategy(self, detected_strategy=None):
        """
        Select best counter-strategy based on detected player strategy

        Args:
            detected_strategy (str): Detected player strategy

        Returns:
            str: Counter-strategy to use
        """
        if detected_strategy is None:
            detected_strategy = self.detect_player_strategy()

        self.current_player_strategy = detected_strategy

        # Get base counter-strategy
        base_counter = self.counter_strategies.get(detected_strategy, 'balanced')

        # Adapt based on effectiveness
        if self.strategy_effectiveness[base_counter]['uses'] > 5:
            effectiveness = self._calculate_strategy_effectiveness(base_counter)

            # If current strategy is ineffective, try alternatives
            if effectiveness < 0.4:
                alternatives = self._get_alternative_strategies(base_counter)
                if alternatives:
                    base_counter = max(alternatives,
                                       key=lambda s: self._calculate_strategy_effectiveness(s))

        self.counter_strategy = base_counter
        return base_counter

    def _calculate_strategy_effectiveness(self, strategy):
        """
        Calculate effectiveness of a strategy

        Args:
            strategy (str): Strategy name

        Returns:
            float: Effectiveness score (0-1)
        """
        stats = self.strategy_effectiveness[strategy]

        if stats['uses'] == 0:
            return 0.5  # Neutral for untried strategies

        hit_rate = stats['hits'] / stats['uses']
        avg_damage = stats['damage_dealt'] / stats['uses']

        # Normalize and combine metrics
        effectiveness = (hit_rate * 0.6) + (min(avg_damage / 50, 1.0) * 0.4)

        return effectiveness

    def _get_alternative_strategies(self, current_strategy):
        """Get alternative strategies to try"""
        all_strategies = list(self.strategy_effectiveness.keys())
        all_strategies.remove(current_strategy)

        # Sort by least used (to encourage exploration)
        all_strategies.sort(key=lambda s: self.strategy_effectiveness[s]['uses'])

        return all_strategies[:3]  # Top 3 alternatives

    def record_strategy_result(self, strategy, hit, damage_dealt):
        """
        Record result of using a strategy

        Args:
            strategy (str): Strategy used
            hit (bool): Whether attack hit
            damage_dealt (int): Damage dealt
        """
        stats = self.strategy_effectiveness[strategy]
        stats['uses'] += 1
        if hit:
            stats['hits'] += 1
        stats['damage_dealt'] += damage_dealt

    def generate_adaptive_action(self, adversary, environment):
        """
        Generate action based on adaptive learning

        Args:
            adversary: The adversary agent
            environment: Game environment

        Returns:
            dict: Action decision with adaptive modifications
        """
        # Detect and counter player strategy
        player_strategy = self.detect_player_strategy()
        counter = self.select_counter_strategy(player_strategy)

        # Generate action based on counter-strategy
        action_decision = self._apply_counter_strategy(counter, adversary, environment)

        # Add learning metadata
        action_decision['learning_meta'] = {
            'detected_strategy': player_strategy,
            'counter_strategy': counter,
            'confidence': self._calculate_confidence()
        }

        return action_decision

    def _apply_counter_strategy(self, counter_strategy, adversary, environment):
        """
        Generate specific action for counter-strategy

        Args:
            counter_strategy (str): Counter strategy to apply
            adversary: The adversary agent
            environment: Game environment

        Returns:
            dict: Action decision
        """
        # Get nearby enemies
        enemies = adversary.detect_enemies(environment)

        if counter_strategy == 'area_denial':
            # Area attacks to prevent hit-and-run
            if len(enemies) > 0:
                return {
                    'action': 'area_attack',
                    'intensity': 'high',
                    'pattern': 'sweeping'
                }

        elif counter_strategy == 'burst_damage':
            # High damage special attacks
            target = adversary.select_target(enemies) if enemies else None
            if target:
                return {
                    'action': 'special_attack',
                    'target': target,
                    'intensity': 'maximum'
                }

        elif counter_strategy == 'close_gap':
            # Aggressive movement toward player
            target = adversary.select_target(enemies) if enemies else None
            if target:
                return {
                    'action': 'aggressive_advance',
                    'target': target,
                    'speed': 'fast'
                }

        elif counter_strategy == 'aggressive_push':
            # Constant pressure on defensive player
            return {
                'action': 'continuous_assault',
                'intensity': 'relentless'
            }

        elif counter_strategy == 'defensive':
            # Defensive stance with counter-attacks
            return {
                'action': 'defensive_counter',
                'regeneration': True
            }

        # Balanced/default
        return {
            'action': 'balanced_attack',
            'adaptability': 'high'
        }

    def _calculate_confidence(self):
        """
        Calculate confidence in current strategy detection

        Returns:
            float: Confidence score (0-1)
        """
        if len(self.player_action_history) < self.pattern_window:
            return 0.3  # Low confidence with insufficient data

        # Check consistency of detected strategy
        recent_detections = []
        for i in range(max(0, len(self.player_action_history) - 20),
                       len(self.player_action_history), 5):
            self.detect_player_strategy()
            recent_detections.append(self.current_player_strategy)

        if recent_detections:
            most_common = Counter(recent_detections).most_common(1)[0]
            consistency = most_common[1] / len(recent_detections)
            return consistency

        return 0.5

    def adapt_difficulty(self, player_performance):
        """
        Dynamically adjust difficulty based on player performance

        Args:
            player_performance (dict): Player performance metrics
        """
        success_rate = player_performance.get('success_rate', 0.5)

        # If player is doing too well, increase difficulty
        if success_rate > 0.7:
            self.adaptation_rate = min(1.0, self.adaptation_rate * 1.2)
        # If player is struggling, decrease difficulty slightly
        elif success_rate < 0.3:
            self.adaptation_rate = max(0.1, self.adaptation_rate * 0.9)

    def get_learning_statistics(self):
        """Get statistics about learning"""
        return {
            'observations': len(self.player_action_history),
            'detected_strategy': self.current_player_strategy,
            'counter_strategy': self.counter_strategy,
            'confidence': self._calculate_confidence(),
            'strategy_effectiveness': {
                k: self._calculate_strategy_effectiveness(k)
                for k in self.strategy_effectiveness.keys()
            },
            'adaptation_rate': self.adaptation_rate
        }

    def reset(self):
        """Reset learning for new game"""
        self.player_action_history.clear()
        self.current_player_strategy = "unknown"
        self.counter_strategy = "balanced"

        # Keep effectiveness data for meta-learning across games
        # But reset usage counters
        for strategy in self.strategy_effectiveness:
            self.strategy_effectiveness[strategy]['uses'] = 0
            self.strategy_effectiveness[strategy]['hits'] = 0


class PatternRecognition:
    """Advanced pattern recognition for player behavior"""

    @staticmethod
    def detect_sequence_pattern(action_sequence):
        """
        Detect repeating patterns in action sequence

        Args:
            action_sequence (list): List of actions

        Returns:
            dict: Detected patterns
        """
        patterns = {}

        # Check for simple repeating patterns (2-4 actions)
        for pattern_length in range(2, min(5, len(action_sequence) // 2 + 1)):
            for i in range(len(action_sequence) - pattern_length * 2 + 1):
                pattern = tuple(action_sequence[i:i + pattern_length])
                next_sequence = tuple(action_sequence[i + pattern_length:i + pattern_length * 2])

                if pattern == next_sequence:
                    patterns[pattern] = patterns.get(pattern, 0) + 1

        return patterns

    @staticmethod
    def predict_next_action(action_history, confidence_threshold=0.6):
        """
        Predict player's next action based on history

        Args:
            action_history (list): Recent action history
            confidence_threshold (float): Minimum confidence for prediction

        Returns:
            tuple: (predicted_action, confidence)
        """
        if len(action_history) < 3:
            return (None, 0.0)

        # Look for pattern in last 6 actions
        recent = action_history[-6:]

        # Find sequences that match current state
        current_sequence = tuple(recent[-3:])

        # Search history for matching sequences
        predictions = []
        for i in range(len(action_history) - 4):
            if tuple(action_history[i:i + 3]) == current_sequence:
                next_action = action_history[i + 3]
                predictions.append(next_action)

        if predictions:
            most_common = Counter(predictions).most_common(1)[0]
            predicted_action = most_common[0]
            confidence = most_common[1] / len(predictions)

            if confidence >= confidence_threshold:
                return (predicted_action, confidence)

        return (None, 0.0)