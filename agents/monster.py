"""
Monster creature agents
"""
import random
from .base_agents import BaseAgent
from config.settings import *


class Monster(BaseAgent):
    """Monster creature agent"""

    def __init__(self, name, position, size="small"):
        """
        Initialize Monster

        Args:
            name (str): Monster name
            position (tuple): Starting position
            size (str): Monster size (small, medium, large)
        """
        # Set health based on size
        health_map = {
            'small': SMALL_MONSTER_HEALTH,
            'medium': MEDIUM_MONSTER_HEALTH,
            'large': LARGE_MONSTER_HEALTH
        }
        health = health_map.get(size, SMALL_MONSTER_HEALTH)

        super().__init__(name, position, health, stamina=50)
        self.agent_type = "monster"
        self.size = size
        self.aggression = random.uniform(0.3, 0.9)
        self.attack_damage = self._calculate_attack_damage()
        self.territory_center = position
        self.territory_radius = random.randint(3, 7)
        self.target = None

    def _calculate_attack_damage(self):
        """Calculate attack damage based on size"""
        base_damage = {
            'small': 8,
            'medium': 15,
            'large': 25
        }
        return base_damage.get(self.size, 8)

    def is_in_territory(self, position):
        """Check if position is in monster's territory"""
        from math import sqrt
        dx = position[0] - self.territory_center[0]
        dy = position[1] - self.territory_center[1]
        distance = sqrt(dx * dx + dy * dy)
        return distance <= self.territory_radius

    def should_flee(self):
        """Determine if monster should flee based on health"""
        health_pct = self.get_health_percentage()

        # Small monsters flee earlier
        if self.size == 'small' and health_pct < 40:
            return True
        # Medium monsters flee when more damaged
        elif self.size == 'medium' and health_pct < 25:
            return True
        # Large monsters rarely flee
        elif self.size == 'large' and health_pct < 15:
            return True

        return False

    def detect_threat(self, environment):
        """
        Detect nearby threats

        Args:
            environment: Game environment

        Returns:
            Agent or None: Nearest threat
        """
        x, y = self.position
        nearby = environment.get_agents_in_range(x, y, AGGRO_RANGE)

        # Filter for predators and synthetics
        threats = [a for a in nearby if a.agent_type in ['predator', 'synthetic']
                   and a != self and a.is_alive()]

        if threats:
            # Return closest threat
            return min(threats, key=lambda t:
            environment.get_distance(x, y, t.position[0], t.position[1]))
        return None

    def should_attack(self, environment):
        """Decide if should attack based on aggression and threats"""
        threat = self.detect_threat(environment)
        if threat:
            return random.random() < self.aggression
        return False

    def decide_action(self, environment):
        """
        Enhanced AI for monsters - more aggressive and mobile

        Args:
            environment: Game environment

        Returns:
            dict: Action decision
        """
        x, y = self.position

        # Check if should flee (low health)
        if self.should_flee():
            return {'action': 'flee'}

        # Look for predators nearby to attack or avoid
        nearby = environment.get_agents_in_range(x, y, AGGRO_RANGE)
        predators = [a for a in nearby if a.agent_type == 'predator' and a.is_alive()]

        if predators:
            nearest = min(predators, key=lambda p:
            environment.get_distance(x, y, p.position[0], p.position[1]))

            distance = environment.get_distance(x, y,
                                                nearest.position[0],
                                                nearest.position[1])

            # If adjacent, attack
            if distance <= 1:
                return {'action': 'attack', 'target': nearest}

            # If in range, move toward (aggressive behavior)
            if self.size in ['medium', 'large']:  # Larger monsters are aggressive
                return {'action': 'hunt', 'target': nearest}
            else:
                # Small monsters are more cautious - only approach if healthy
                if self.get_health_percentage() > 60:
                    return {'action': 'hunt', 'target': nearest}

        # Patrol/wander
        return {'action': 'patrol'}

    def attack(self, target):
        """
        Attack a target

        Args:
            target: Agent to attack

        Returns:
            int: Damage dealt
        """
        if not self.is_alive():
            return 0

        # Calculate damage with some randomness
        damage = int(self.attack_damage * random.uniform(0.8, 1.2))
        target.take_damage(damage)
        return damage

    def get_trophy_value(self):
        """Get trophy value based on size"""
        trophy_values = {
            'small': 'skull_small',
            'medium': 'skull_medium',
            'large': 'skull_large'
        }
        return {
            'type': trophy_values.get(self.size, 'skull_small'),
            'size': self.size,
            'from': self.name
        }

    def get_status(self):
        """Get monster status"""
        status = super().get_status()
        status.update({
            'size': self.size,
            'aggression': self.aggression,
            'attack_damage': self.attack_damage,
            'territory_center': self.territory_center,
            'has_target': self.target is not None
        })
        return status

    def __repr__(self):
        """String representation"""
        return f"Monster({self.name}, {self.size}, HP={self.health}/{self.max_health}, pos={self.position})"