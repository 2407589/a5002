"""
Base agent class for all entities in the simulation
"""
import uuid
from config.settings import *


class BaseAgent:
    """Base class for all agents in the simulation"""

    agent_counter = 0

    def __init__(self, name, position, health, stamina=None):
        """
        Initialize base agent

        Args:
            name (str): Agent name
            position (tuple): Starting (x, y) position
            health (int): Maximum and starting health
            stamina (int): Maximum and starting stamina (None for agents without stamina)
        """
        BaseAgent.agent_counter += 1
        self.agent_id = f"{name}_{BaseAgent.agent_counter}"
        self.name = name
        self.position = position
        self.max_health = health
        self.health = health
        self.max_stamina = stamina if stamina is not None else 0
        self.stamina = stamina if stamina is not None else 0
        self.alive = True
        self.agent_type = "base"

    def is_alive(self):
        """Check if agent is alive"""
        return self.alive and self.health > 0

    def take_damage(self, damage):
        """
        Apply damage to agent

        Args:
            damage (int): Amount of damage

        Returns:
            bool: True if agent is still alive
        """
        self.health = max(0, self.health - damage)
        if self.health <= 0:
            self.die()
        return self.is_alive()

    def heal(self, amount):
        """
        Heal the agent

        Args:
            amount (int): Amount to heal
        """
        self.health = min(self.max_health, self.health + amount)

    def use_stamina(self, amount):
        """
        Use stamina

        Args:
            amount (int): Stamina cost

        Returns:
            bool: True if had enough stamina
        """
        if self.stamina >= amount:
            self.stamina -= amount
            return True
        return False

    def recover_stamina(self, amount):
        """Recover stamina"""
        if self.max_stamina > 0:
            self.stamina = min(self.max_stamina, self.stamina + amount)

    def die(self):
        """Handle agent death"""
        self.alive = False
        self.health = 0

    def get_health_percentage(self):
        """Get health as percentage"""
        return (self.health / self.max_health) * 100 if self.max_health > 0 else 0

    def get_stamina_percentage(self):
        """Get stamina as percentage"""
        return (self.stamina / self.max_stamina) * 100 if self.max_stamina > 0 else 0

    def decide_action(self, environment):
        """
        Decide what action to take (override in subclasses)

        Args:
            environment: The game environment

        Returns:
            str: Action to take
        """
        return "idle"

    def execute_action(self, action, environment):
        """
        Execute chosen action (override in subclasses)

        Args:
            action (str): Action to execute
            environment: The game environment

        Returns:
            bool: True if action succeeded
        """
        return True

    def get_status(self):
        """Get agent status dict"""
        return {
            'id': self.agent_id,
            'name': self.name,
            'type': self.agent_type,
            'position': self.position,
            'health': self.health,
            'max_health': self.max_health,
            'health_pct': self.get_health_percentage(),
            'stamina': self.stamina,
            'max_stamina': self.max_stamina,
            'stamina_pct': self.get_stamina_percentage(),
            'alive': self.is_alive()
        }

    def __repr__(self):
        """String representation"""
        return f"{self.__class__.__name__}({self.name}, pos={self.position}, HP={self.health}/{self.max_health})"