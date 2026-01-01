"""
Cell class representing individual grid cells in the environment
"""
from config.settings import *


class Cell:
    """Represents a single cell in the grid environment"""

    def __init__(self, x, y, terrain_type=TERRAIN_EMPTY):
        """
        Initialize a cell

        Args:
            x (int): X coordinate
            y (int): Y coordinate
            terrain_type (int): Type of terrain
        """
        self.x = x
        self.y = y
        self.terrain_type = terrain_type
        self.occupant = None  # Agent occupying this cell
        self.items = []  # Items in this cell (trophies, resources)
        self.visited_by = set()  # Track which agents have visited

    @property
    def position(self):
        """Return cell position as tuple"""
        return (self.x, self.y)

    def is_empty(self):
        """Check if cell has no occupant"""
        return self.occupant is None

    def is_passable(self):
        """Check if agents can move through this cell"""
        return self.terrain_type != TERRAIN_OBSTACLE

    def get_movement_cost(self):
        """Return stamina cost multiplier for moving through this cell"""
        if self.terrain_type == TERRAIN_ROUGH:
            return ROUGH_TERRAIN_MULTIPLIER
        elif self.terrain_type == TERRAIN_TRAP:
            return 2.0
        return 1.0

    def set_occupant(self, agent):
        """Set the agent occupying this cell"""
        if not self.is_empty() and self.occupant != agent:
            raise ValueError(f"Cell {self.position} already occupied by {self.occupant}")
        self.occupant = agent

    def clear_occupant(self):
        """Remove the occupant from this cell"""
        self.occupant = None

    def add_item(self, item):
        """Add an item to this cell"""
        self.items.append(item)

    def remove_item(self, item):
        """Remove an item from this cell"""
        if item in self.items:
            self.items.remove(item)

    def mark_visited(self, agent_id):
        """Mark this cell as visited by an agent"""
        self.visited_by.add(agent_id)

    def __repr__(self):
        """String representation"""
        occupant_str = f", occupant={self.occupant.name}" if self.occupant else ""
        return f"Cell({self.x}, {self.y}, terrain={self.terrain_type}{occupant_str})"