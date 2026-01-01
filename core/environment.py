"""
Environment grid for Predator: Badlands simulation
"""
import random
from core.cell import Cell
from config.settings import *


class Environment:
    """2D grid environment representing planet Kalisk"""

    def __init__(self, width=GRID_WIDTH, height=GRID_HEIGHT, wrapping=WRAPPING_ENABLED):
        """
        Initialize the environment

        Args:
            width (int): Grid width
            height (int): Grid height
            wrapping (bool): Whether edges wrap around
        """
        self.width = width
        self.height = height
        self.wrapping = wrapping
        self.grid = [[Cell(x, y) for y in range(height)] for x in range(width)]
        self.agents = []
        self.turn_count = 0

        # Initialize terrain
        self._generate_terrain()

    def _generate_terrain(self):
        """Generate varied terrain across the grid"""
        # Add some rough terrain patches
        for _ in range(int(self.width * self.height * 0.15)):
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            self.grid[x][y].terrain_type = TERRAIN_ROUGH

        # Add obstacles
        for _ in range(int(self.width * self.height * 0.08)):
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            self.grid[x][y].terrain_type = TERRAIN_OBSTACLE

        # Add traps
        for _ in range(int(self.width * self.height * 0.05)):
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            if self.grid[x][y].terrain_type == TERRAIN_EMPTY:
                self.grid[x][y].terrain_type = TERRAIN_TRAP

    def get_cell(self, x, y):
        """
        Get cell at position, handling wrapping

        Args:
            x (int): X coordinate
            y (int): Y coordinate

        Returns:
            Cell: The cell at the position
        """
        if self.wrapping:
            x = x % self.width
            y = y % self.height
        else:
            if not self.is_valid_position(x, y):
                return None
        return self.grid[x][y]

    def is_valid_position(self, x, y):
        """Check if position is within grid bounds"""
        return 0 <= x < self.width and 0 <= y < self.height

    def is_position_free(self, x, y):
        """Check if position is free for placement"""
        cell = self.get_cell(x, y)
        return cell is not None and cell.is_empty() and cell.is_passable()

    def place_agent(self, agent, x, y):
        """
        Place an agent at a specific position

        Args:
            agent: The agent to place
            x (int): X coordinate
            y (int): Y coordinate

        Returns:
            bool: True if placement successful
        """
        if not self.is_position_free(x, y):
            return False

        cell = self.get_cell(x, y)
        cell.set_occupant(agent)
        agent.position = (x, y)

        if agent not in self.agents:
            self.agents.append(agent)

        return True

    def move_agent(self, agent, new_x, new_y):
        """
        Move an agent to a new position

        Args:
            agent: The agent to move
            new_x (int): New X coordinate
            new_y (int): New Y coordinate

        Returns:
            bool: True if move successful
        """
        # Check if destination is valid
        new_cell = self.get_cell(new_x, new_y)
        if new_cell is None or not new_cell.is_passable():
            return False

        # Check if destination is occupied
        if not new_cell.is_empty():
            return False

        # Clear old position
        old_x, old_y = agent.position
        old_cell = self.get_cell(old_x, old_y)
        old_cell.clear_occupant()

        # Set new position
        new_cell.set_occupant(agent)
        agent.position = (new_x, new_y)
        new_cell.mark_visited(agent.agent_id)

        return True

    def remove_agent(self, agent):
        """Remove an agent from the environment"""
        x, y = agent.position
        cell = self.get_cell(x, y)
        cell.clear_occupant()

        if agent in self.agents:
            self.agents.remove(agent)

    def get_adjacent_positions(self, x, y):
        """
        Get all adjacent positions (8-directional)

        Args:
            x (int): X coordinate
            y (int): Y coordinate

        Returns:
            list: List of adjacent (x, y) tuples
        """
        adjacent = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                new_x, new_y = x + dx, y + dy

                if self.wrapping:
                    new_x = new_x % self.width
                    new_y = new_y % self.height
                    adjacent.append((new_x, new_y))
                elif self.is_valid_position(new_x, new_y):
                    adjacent.append((new_x, new_y))

        return adjacent

    def get_agents_in_range(self, x, y, range_val):
        """
        Get all agents within specified range

        Args:
            x (int): Center X coordinate
            y (int): Center Y coordinate
            range_val (int): Range radius

        Returns:
            list: List of agents within range
        """
        agents_in_range = []
        for agent in self.agents:
            ax, ay = agent.position
            distance = self.get_distance(x, y, ax, ay)
            if distance <= range_val:
                agents_in_range.append(agent)
        return agents_in_range

    def get_distance(self, x1, y1, x2, y2):
        """Calculate Manhattan distance between two positions"""
        if self.wrapping:
            dx = min(abs(x2 - x1), self.width - abs(x2 - x1))
            dy = min(abs(y2 - y1), self.height - abs(y2 - y1))
            return dx + dy
        return abs(x2 - x1) + abs(y2 - y1)

    def increment_turn(self):
        """Increment turn counter"""
        self.turn_count += 1

    def get_state_summary(self):
        """Get summary of current environment state"""
        return {
            'turn': self.turn_count,
            'agents_alive': len([a for a in self.agents if a.is_alive()]),
            'total_agents': len(self.agents)
        }

    def __repr__(self):
        """String representation"""
        return f"Environment({self.width}x{self.height}, {len(self.agents)} agents, turn {self.turn_count})"