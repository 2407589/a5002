
import random
from .base_agents import BaseAgent
from config.settings import *


class UltimateAdversary(BaseAgent):
    """Ultimate boss creature that Dek must defeat"""

    def __init__(self, name, position):
        """
        Initialize Ultimate Adversary

        Args:
            name (str): Adversary name
            position (tuple): Starting position
        """
        super().__init__(name, position, ADVERSARY_HEALTH, stamina=200)
        self.agent_type = "adversary"
        self.attack_damage = 30  # REDUCED from 40
        self.area_of_effect = 2
        self.enraged = False
        self.rage_threshold = 0.5
        self.attack_pattern = "normal"
        self.targets = []
        self.defeated_by = None

        # Boss phases
        self.phase = 1
        self.max_phases = 3

    def get_current_phase(self):
        """Determine current phase based on health"""
        health_pct = self.get_health_percentage() / 100

        if health_pct > 0.66:
            return 1
        elif health_pct > 0.33:
            return 2
        else:
            return 3

    def update_phase(self):
        """Update boss phase and adjust behavior"""
        new_phase = self.get_current_phase()

        if new_phase != self.phase:
            self.phase = new_phase
            self._phase_transition()

    def _phase_transition(self):
        """Handle phase transition effects"""
        if self.phase == 2:
            self.attack_damage = int(self.attack_damage * 1.2)  # REDUCED from 1.3
            self.enraged = True
        elif self.phase == 3:
            self.attack_damage = int(self.attack_damage * 1.3)  # REDUCED from 1.5
            self.area_of_effect = 3

    def detect_enemies(self, environment):

        x, y = self.position
        detection_range = 4 if self.enraged else 3

        nearby = environment.get_agents_in_range(x, y, detection_range)

        enemies = [a for a in nearby if a.agent_type == 'predator'
                   and a != self and a.is_alive()
                   and getattr(a, 'honor', 0) >= 80]

        return enemies

    def select_target(self, enemies):
        """
        Select target based on threat assessment

        Args:
            enemies (list): List of potential targets

        Returns:
            Agent or None: Selected target
        """
        if not enemies:
            return None

        # In phase 1, target closest
        if self.phase == 1:
            return min(enemies, key=lambda e:
            ((e.position[0] - self.position[0]) ** 2 +
             (e.position[1] - self.position[1]) ** 2))

        # In later phases, target most threatening (predators with high honor)
        predators = [e for e in enemies if e.agent_type == 'predator']
        if predators:
            return max(predators, key=lambda p: getattr(p, 'honor', 0))

        return enemies[0]

    def area_attack(self, environment):
        """
        Perform area of effect attack

        Args:
            environment: Game environment

        Returns:
            list: List of (agent, damage) tuples
        """
        results = []
        x, y = self.position

        # Get all agents in AoE range
        for dx in range(-self.area_of_effect, self.area_of_effect + 1):
            for dy in range(-self.area_of_effect, self.area_of_effect + 1):
                if dx == 0 and dy == 0:
                    continue

                cell = environment.get_cell(x + dx, y + dy)
                if cell and cell.occupant and cell.occupant != self:
                    if cell.occupant.agent_type in ['predator', 'monster']:  # Removed 'synthetic'
                        damage = int(self.attack_damage * random.uniform(0.7, 1.0))
                        cell.occupant.take_damage(damage)
                        results.append((cell.occupant, damage))

        return results

    def special_attack(self, target, environment):
        """
        FIXED: Reduced damage multiplier

        Args:
            target: Target agent
            environment: Game environment

        Returns:
            int: Damage dealt
        """
        # REDUCED from 2.0 to 1.5
        damage = int(self.attack_damage * 1.5 * random.uniform(0.9, 1.1))
        target.take_damage(damage)
        return damage

    def regenerate(self):
        """Regenerate health slowly"""
        if self.phase == 1:
            regen_amount = 2
        elif self.phase == 2:
            regen_amount = 1
        else:
            regen_amount = 0

        self.heal(regen_amount)

    def decide_action(self, environment):
        """
        FIXED: Stay idle when no worthy opponents

        Args:
            environment: Game environment

        Returns:
            dict: Action decision
        """
        self.update_phase()

        enemies = self.detect_enemies(environment)

        if not enemies:
            # CRITICAL FIX: Don't patrol - stay put and wait for challenge
            if self.health < self.max_health * 0.8:
                return {'action': 'regenerate'}
            return {'action': 'idle'}  # CHANGED from 'patrol'

        # Select target
        target = self.select_target(enemies)
        self.targets = enemies

        # Decide attack type based on enemy positions
        close_enemies = [e for e in enemies if
                         abs(e.position[0] - self.position[0]) <= self.area_of_effect and
                         abs(e.position[1] - self.position[1]) <= self.area_of_effect]

        # REDUCED special attack chance from 0.3 to 0.15
        if len(close_enemies) >= 2 and random.random() < 0.4:
            return {'action': 'area_attack'}

        if self.phase >= 2 and random.random() < 0.15:  # REDUCED from 0.3
            return {'action': 'special_attack', 'target': target}

        return {'action': 'attack', 'target': target}

    def take_damage(self, damage):
        """
        Override take damage to handle rage

        Args:
            damage (int): Damage amount

        Returns:
            bool: True if still alive
        """
        result = super().take_damage(damage)

        # Check for rage
        if self.get_health_percentage() < self.rage_threshold * 100 and not self.enraged:
            self.enraged = True
            self.attack_damage = int(self.attack_damage * 1.1)  # REDUCED from 1.2

        return result

    def die(self):
        """Handle adversary death"""
        super().die()

    def get_trophy_value(self):
        """Get ultimate trophy for defeating adversary"""
        return {
            'type': 'adversary_skull',
            'size': 'legendary',
            'from': self.name,
            'honor_value': 100
        }

    def get_status(self):
        """Get adversary status"""
        status = super().get_status()
        status.update({
            'phase': self.phase,
            'enraged': self.enraged,
            'attack_damage': self.attack_damage,
            'area_of_effect': self.area_of_effect,
            'targets_count': len(self.targets)
        })
        return status

    def __repr__(self):
        """String representation"""
        phase_str = f"Phase {self.phase}"
        rage_str = "ENRAGED" if self.enraged else ""
        return f"Adversary({self.name}, {phase_str} {rage_str}, HP={self.health}/{self.max_health})"