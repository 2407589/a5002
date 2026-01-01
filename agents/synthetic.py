"""
Synthetic android agent (Thia and others)
"""
import random
from .base_agents import BaseAgent
from config.settings import *


class Synthetic(BaseAgent):
    """Synthetic android agent"""

    def __init__(self, name, position, damaged=False):
        """
        Initialize Synthetic

        Args:
            name (str): Synthetic name
            position (tuple): Starting position
            damaged (bool): Whether synthetic is damaged
        """
        super().__init__(name, position, SYNTHETIC_BASE_HEALTH)
        self.agent_type = "synthetic"
        self.damaged = damaged
        self.mobility = 0.3 if damaged else 1.0  # Mobility factor
        self.knowledge_database = {}
        self.map_knowledge = set()  # Cells explored
        self.warnings_given = []
        self.being_carried = False

        # Thia starts with some knowledge
        if name == "Thia":
            self._initialize_thia_knowledge()

    def _initialize_thia_knowledge(self):
        """Initialize Thia's starting knowledge"""
        self.knowledge_database = {
            'adversary_location_hint': "The ultimate adversary dwells in the far reaches",
            'adversary_weakness': "Unknown - requires reconnaissance",
            'terrain_hazards': ["Traps scattered across canyons", "Rough terrain in highlands"],
            'weyland_yutani_data': "Classified corporate information"
        }

    def can_move_independently(self):
        """Check if can move without assistance"""
        return not self.damaged or self.mobility > 0.5

    def provide_knowledge(self, topic):
        """
        Provide knowledge on a topic

        Args:
            topic (str): Knowledge topic

        Returns:
            str or None: Knowledge if available
        """
        return self.knowledge_database.get(topic, None)

    def add_knowledge(self, key, value):
        """Add knowledge to database"""
        self.knowledge_database[key] = value

    def scout_area(self, environment, position):
        """
        Scout an area and gather information

        Args:
            environment: Game environment
            position (tuple): Position to scout

        Returns:
            dict: Scouting report
        """
        x, y = position
        report = {
            'position': position,
            'agents_nearby': [],
            'terrain_info': {},
            'hazards': []
        }

        # Scan nearby cells
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                scan_x, scan_y = x + dx, y + dy
                cell = environment.get_cell(scan_x, scan_y)
                if cell:
                    self.map_knowledge.add((scan_x, scan_y))

                    if cell.occupant:
                        report['agents_nearby'].append({
                            'type': cell.occupant.agent_type,
                            'name': cell.occupant.name,
                            'position': (scan_x, scan_y)
                        })

                    if cell.terrain_type == TERRAIN_TRAP:
                        report['hazards'].append((scan_x, scan_y))

        return report

    def analyze_adversary(self, adversary):
        """
        Analyze the ultimate adversary

        Args:
            adversary: Adversary agent

        Returns:
            dict: Analysis results
        """
        analysis = {
            'health_estimate': adversary.health,
            'position': adversary.position,
            'threat_level': 'extreme',
            'recommended_strategy': self._recommend_strategy(adversary)
        }

        # Update knowledge
        self.add_knowledge('adversary_health', adversary.health)
        self.add_knowledge('adversary_position', adversary.position)

        return analysis

    def _recommend_strategy(self, adversary):
        """Recommend strategy for defeating adversary"""
        strategies = [
            "Hit-and-run tactics to wear down defenses",
            "Coordinate attacks from multiple angles",
            "Exploit environmental hazards",
            "Target weak points when exposed"
        ]
        return random.choice(strategies)

    def issue_warning(self, warning_type, details):
        """
        Issue a warning about dangers

        Args:
            warning_type (str): Type of warning
            details (str): Warning details
        """
        warning = {
            'type': warning_type,
            'details': details,
            'turn': 0  # Will be set by simulation
        }
        self.warnings_given.append(warning)
        return warning

    def repair(self, amount):
        """
        Repair damage

        Args:
            amount (float): Repair amount (0-1)
        """
        if self.damaged:
            self.mobility = min(1.0, self.mobility + amount)
            if self.mobility >= 0.8:
                self.damaged = False

    def get_map_coverage(self):
        """Get percentage of map explored"""
        return len(self.map_knowledge)

    def decide_action(self, environment):
        """
        Decide synthetic action

        Args:
            environment: Game environment

        Returns:
            dict: Action decision
        """
        # If being carried, no independent action
        if self.being_carried:
            return {'action': 'idle'}

        # If severely damaged, request help
        if self.damaged and self.mobility < 0.3:
            return {'action': 'request_help'}

        # Scout nearby area
        return {'action': 'scout'}

    def get_status(self):
        """Get synthetic status"""
        status = super().get_status()
        status.update({
            'damaged': self.damaged,
            'mobility': self.mobility,
            'can_move': self.can_move_independently(),
            'being_carried': self.being_carried,
            'knowledge_topics': list(self.knowledge_database.keys()),
            'map_coverage': self.get_map_coverage(),
            'warnings_given': len(self.warnings_given)
        })
        return status

    def __repr__(self):
        """String representation"""
        status = "damaged" if self.damaged else "functional"
        return f"Synthetic({self.name}, {status}, mobility={self.mobility:.1f}, pos={self.position})"