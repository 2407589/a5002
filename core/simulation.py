"""
Main simulation controller for Predator: Badlands - IMPROVED VERSION
Added: Verbosity control, condensed logging, better combat tracking
"""
import random
import time
import sys
import os

# Ensure imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.environment import Environment
from agents.predator import Predator
from agents.synthetic import Synthetic
from agents.monster import Monster
from agents.adversary import UltimateAdversary
from config import settings

# Import specific constants we need
GRID_WIDTH = settings.GRID_WIDTH
GRID_HEIGHT = settings.GRID_HEIGHT
WRAPPING_ENABLED = settings.WRAPPING_ENABLED
DEK_START_POS = settings.DEK_START_POS
THIA_START_POS = settings.THIA_START_POS
FATHER_START_POS = settings.FATHER_START_POS
BROTHER_START_POS = settings.BROTHER_START_POS
SMALL_MONSTER_COUNT = settings.SMALL_MONSTER_COUNT
MEDIUM_MONSTER_COUNT = settings.MEDIUM_MONSTER_COUNT
LARGE_MONSTER_COUNT = settings.LARGE_MONSTER_COUNT
MAX_TURNS = settings.MAX_TURNS
MOVE_STAMINA_COST = settings.MOVE_STAMINA_COST
BASE_ATTACK_DAMAGE = settings.BASE_ATTACK_DAMAGE


# ================== VERBOSITY LEVELS ==================
class VerbosityLevel:
    """Control output verbosity"""
    SILENT = 0  # Only critical events
    QUIET = 1  # Major events + episode summary
    NORMAL = 2  # Combat outcomes + important events
    VERBOSE = 3  # Everything (original behavior)


class Simulation:
    """Main simulation controller with verbosity control"""

    def __init__(self, verbosity=VerbosityLevel.QUIET):
        """Initialize simulation

        Args:
            verbosity: Output level (0=SILENT, 1=QUIET, 2=NORMAL, 3=VERBOSE)
        """
        self.environment = Environment(GRID_WIDTH, GRID_HEIGHT, WRAPPING_ENABLED)
        self.dek = None
        self.thia = None
        self.father = None
        self.brother = None
        self.adversary = None
        self.monsters = []

        self.running = False
        self.paused = False
        self.quest_completed = False
        self.simulation_log = []

        # VERBOSITY CONTROL
        self.verbosity = verbosity

        # COMBAT TRACKING (for condensed output)
        self.combat_tracker = {}  # Track ongoing combats
        self.flee_tracker = {}  # Track flee sequences

        # Adaptive system support
        self.enable_adaptive = False
        self.adaptive_system = None

        # Statistics
        self.stats = {
            'turns_elapsed': 0,
            'total_kills': 0,
            'honor_gained': 0,
            'honor_lost': 0,
            'dek_deaths': 0,
            'adversary_defeated': False
        }

    def log_event(self, message, min_verbosity=VerbosityLevel.NORMAL):
        """Log a simulation event with verbosity control

        Args:
            message: Message to log
            min_verbosity: Minimum verbosity level required to show this message
        """
        event = {
            'turn': self.stats['turns_elapsed'],
            'message': message
        }
        self.simulation_log.append(event)

        # Only print if verbosity is high enough
        if self.verbosity >= min_verbosity:
            print(f"[Turn {self.stats['turns_elapsed']}] {message}")

    def initialize_agents(self):
        """Initialize all agents in the simulation"""
        # Create Dek (the protagonist - exiled runt)
        self.dek = Predator("Dek", DEK_START_POS, role="runt")
        self.dek.honor = -10  # Starts dishonored
        self.dek.max_health = 400
        self.dek.health = 400
        self.dek.max_stamina = 200
        self.dek.stamina = 200
        self.environment.place_agent(self.dek, *DEK_START_POS)
        self.log_event(f"🎬 Dek exiled at position {DEK_START_POS}", VerbosityLevel.QUIET)

        # Create Thia (damaged synthetic)
        self.thia = Synthetic("Thia", THIA_START_POS, damaged=True)
        self.environment.place_agent(self.thia, *THIA_START_POS)
        self.log_event(f"🤖 Thia (damaged synthetic) found at {THIA_START_POS}", VerbosityLevel.QUIET)

        # Thia's first interaction with Dek
        self.log_event(f"💬 THIA: 'My systems are damaged, but I can still help. Together, we might survive this.'",
                       VerbosityLevel.NORMAL)
        self.log_event(
            f"💬 THIA: 'Hunt worthy prey to restore your honor. Only then can you face the ultimate adversary.'",
            VerbosityLevel.NORMAL)

        # Create Dek's father (elder predator)
        self.father = Predator("Father", FATHER_START_POS, role="elder")
        self.father.honor = 150
        self.environment.place_agent(self.father, *FATHER_START_POS)

        # Create Dek's brother (warrior predator)
        self.brother = Predator("Brother", BROTHER_START_POS, role="warrior")
        self.brother.honor = 80
        self.environment.place_agent(self.brother, *BROTHER_START_POS)

        # Create monsters
        self._spawn_monsters()

        # Create ultimate adversary
        self._spawn_adversary()

    def _spawn_monsters(self):
        """Spawn monster creatures across the map"""
        monster_id = 1

        # Small monsters
        for _ in range(SMALL_MONSTER_COUNT):
            pos = self._find_random_empty_position()
            if pos:
                monster = Monster(f"SmallMonster_{monster_id}", pos, size="small")
                self.environment.place_agent(monster, *pos)
                self.monsters.append(monster)
                monster_id += 1

        # Medium monsters
        for _ in range(MEDIUM_MONSTER_COUNT):
            pos = self._find_random_empty_position()
            if pos:
                monster = Monster(f"MediumMonster_{monster_id}", pos, size="medium")
                self.environment.place_agent(monster, *pos)
                self.monsters.append(monster)
                monster_id += 1

        # Large monsters
        for _ in range(LARGE_MONSTER_COUNT):
            pos = self._find_random_empty_position()
            if pos:
                monster = Monster(f"LargeMonster_{monster_id}", pos, size="large")
                self.environment.place_agent(monster, *pos)
                self.monsters.append(monster)
                monster_id += 1

        self.log_event(f"Spawned {len(self.monsters)} monsters across Kalisk", VerbosityLevel.QUIET)

    def _spawn_adversary(self):
        """Spawn the ultimate adversary"""
        # Place in far corner from Dek
        pos = (GRID_WIDTH - 3, GRID_HEIGHT - 3)
        pos = self._find_empty_near(pos, radius=3)

        if pos:
            self.adversary = UltimateAdversary("Ultimate_Adversary", pos)
            self.environment.place_agent(self.adversary, *pos)
            self.log_event(f"Ultimate Adversary spawned at {pos}", VerbosityLevel.QUIET)

    def _find_random_empty_position(self):
        """Find a random empty position on the grid"""
        attempts = 0
        max_attempts = 100

        while attempts < max_attempts:
            x = random.randint(0, self.environment.width - 1)
            y = random.randint(0, self.environment.height - 1)

            if self.environment.is_position_free(x, y):
                return (x, y)
            attempts += 1

        return None

    def _find_empty_near(self, position, radius=2):
        """Find empty position near a target position"""
        x, y = position

        for r in range(radius):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    check_x, check_y = x + dx, y + dy
                    if self.environment.is_position_free(check_x, check_y):
                        return (check_x, check_y)

        return self._find_random_empty_position()

    def execute_agent_turn(self, agent):
        """
        Execute one turn for an agent with CONDENSED LOGGING
        """
        if not agent.is_alive():
            return

        # ====== ADAPTIVE ADVERSARY SYSTEM ======
        if (agent == self.adversary and
                hasattr(self, 'enable_adaptive') and self.enable_adaptive and
                hasattr(self, 'adaptive_system') and self.adaptive_system):

            # Use adaptive learning to select action
            action_decision = self.adaptive_system.generate_adaptive_action(agent, self.environment)
            action = action_decision.get('action', 'patrol')
            target = action_decision.get('target')

            # Execute adaptive action
            if action == 'area_attack':
                results = agent.area_attack(self.environment)
                if results:
                    self.log_event(f"⚡ {agent.name} performs ADAPTIVE AREA ATTACK, hitting {len(results)} targets",
                                   VerbosityLevel.NORMAL)
            elif action == 'special_attack' and target:
                damage = agent.special_attack(target, self.environment)
                self.log_event(f"⚡ {agent.name} unleashes ADAPTIVE SPECIAL ATTACK on {target.name} ({damage} damage)",
                               VerbosityLevel.NORMAL)
            elif action == 'attack' and target:
                self._execute_attack(agent, target)
            else:
                # Default behavior
                standard_action = agent.decide_action(self.environment)
                self._handle_agent_action(agent, standard_action)

            return

        # ====== STANDARD AGENT BEHAVIOR ======
        action_decision = agent.decide_action(self.environment)
        self._handle_agent_action(agent, action_decision)

    def _handle_agent_action(self, agent, action_decision):
        """Handle agent actions with CONDENSED LOGGING"""
        action = action_decision.get('action', 'idle')
        target = action_decision.get('target')

        if action == 'attack' and target:
            self._execute_attack(agent, target)
        elif action == 'hunt' and target:
            # CONDENSED: Only log if Dek is hunting adversary or first hunt
            if agent.name == 'Dek' and target.agent_type == 'adversary':
                dist = self.environment.get_distance(agent.position[0], agent.position[1],
                                                     target.position[0], target.position[1])
                self.log_event(f"🎯 {agent.name} pursues the Ultimate Adversary (distance: {dist})",
                               VerbosityLevel.NORMAL)
            self._move_towards(agent, target.position)
        elif action == 'flee':
            # Track flees but don't log every one
            if agent.name not in self.flee_tracker:
                self.flee_tracker[agent.name] = {'start_turn': self.stats['turns_elapsed'], 'count': 0}
            self.flee_tracker[agent.name]['count'] += 1
            self._execute_flee(agent)
        elif action == 'rest':
            agent.rest()
        elif action == 'patrol':
            self._execute_patrol(agent)
        elif action == 'retreat':
            self._execute_flee(agent)

    def _execute_attack(self, attacker, target):
        """Execute attack with COMBAT TRACKING for condensed output"""
        if not attacker.is_alive() or not target.is_alive():
            return False

        # Calculate damage
        damage = int(BASE_ATTACK_DAMAGE * random.uniform(0.8, 1.2))

        # Critical hit chance
        if random.random() < 0.15:
            damage = int(damage * 2.0)
            crit = True
        else:
            crit = False

        # Apply damage
        target.take_damage(damage)

        # COMBAT TRACKING
        combat_key = f"{attacker.name}_vs_{target.name}"
        if combat_key not in self.combat_tracker:
            self.combat_tracker[combat_key] = {
                'start_turn': self.stats['turns_elapsed'],
                'rounds': 0,
                'total_damage': 0
            }

        self.combat_tracker[combat_key]['rounds'] += 1
        self.combat_tracker[combat_key]['total_damage'] += damage

        # Check for death
        if not target.is_alive():
            # Log complete combat summary
            combat_info = self.combat_tracker[combat_key]
            self.log_event(
                f"💀 {attacker.name} defeats {target.name} " +
                f"({combat_info['rounds']} rounds, {combat_info['total_damage']} total damage)",
                VerbosityLevel.NORMAL
            )
            self._handle_death(target, attacker)
            # Clear combat tracker
            if combat_key in self.combat_tracker:
                del self.combat_tracker[combat_key]
            return True
        else:
            # Only log attacks in VERBOSE mode or first round
            if self.verbosity >= VerbosityLevel.VERBOSE or self.combat_tracker[combat_key]['rounds'] == 1:
                crit_str = " CRITICAL!" if crit else ""
                self.log_event(
                    f"⚔️ {attacker.name} attacks {target.name} for {damage} damage{crit_str} ({target.health}/{target.max_health} HP remaining)",
                    VerbosityLevel.VERBOSE
                )
            return False

    def _execute_flee(self, agent):
        """Execute flee action - ONLY LOG PERIODICALLY"""
        x, y = agent.position
        adjacent = self.environment.get_adjacent_positions(x, y)

        valid_moves = []
        for pos in adjacent:
            cell = self.environment.get_cell(*pos)
            if cell and cell.is_passable() and cell.is_empty():
                valid_moves.append(pos)

        if valid_moves:
            new_pos = random.choice(valid_moves)
            self.environment.move_agent(agent, *new_pos)

            # Only log in VERBOSE mode
            self.log_event(f"{agent.name} flees to {new_pos}", VerbosityLevel.VERBOSE)

    def _execute_patrol(self, agent):
        """Execute patrol/exploration - SILENT"""
        x, y = agent.position
        adjacent = self.environment.get_adjacent_positions(x, y)

        valid_moves = []
        for pos in adjacent:
            cell = self.environment.get_cell(*pos)
            if cell and cell.is_passable() and cell.is_empty():
                valid_moves.append(pos)

        if valid_moves:
            new_pos = random.choice(valid_moves)
            if hasattr(agent, 'use_stamina'):
                if agent.use_stamina(MOVE_STAMINA_COST):
                    self.environment.move_agent(agent, *new_pos)
            else:
                self.environment.move_agent(agent, *new_pos)

    def _move_towards(self, agent, target_pos):
        """
        Move agent towards target position - SILENT except for important moves
        """
        x, y = agent.position
        tx, ty = target_pos

        # Calculate direction
        dx = 0 if x == tx else (1 if tx > x else -1)
        dy = 0 if y == ty else (1 if ty > y else -1)

        # Try to move in both directions (diagonal movement)
        moves_to_try = []

        # Priority 1: Diagonal move (both dx and dy)
        if dx != 0 and dy != 0:
            moves_to_try.append((x + dx, y + dy))

        # Priority 2: Horizontal move
        if dx != 0:
            moves_to_try.append((x + dx, y))

        # Priority 3: Vertical move
        if dy != 0:
            moves_to_try.append((x, y + dy))

        # Try each move in priority order
        for new_x, new_y in moves_to_try:
            if self.environment.is_position_free(new_x, new_y):
                cell = self.environment.get_cell(new_x, new_y)
                if cell and cell.is_passable():
                    # Check stamina if agent has it
                    movement_cost = MOVE_STAMINA_COST
                    if hasattr(agent, 'get_movement_stamina_cost'):
                        movement_cost = agent.get_movement_stamina_cost(
                            MOVE_STAMINA_COST,
                            cell.get_movement_cost()
                        )

                    if hasattr(agent, 'use_stamina'):
                        if agent.use_stamina(movement_cost):
                            self.environment.move_agent(agent, new_x, new_y)
                            return True
                    else:
                        self.environment.move_agent(agent, new_x, new_y)
                        return True

        return False

    def _handle_death(self, dead_agent, killer):
        """Handle agent death with CONDENSED logging"""
        self.log_event(f"☠️ {dead_agent.name} has been slain by {killer.name}!", VerbosityLevel.NORMAL)

        # Award honor and trophies if killer is a predator
        if isinstance(killer, Predator):
            if isinstance(dead_agent, Monster):
                honor = killer.record_kill(dead_agent.size)
                trophy = dead_agent.get_trophy_value()
                killer.add_trophy(trophy)
                self.stats['total_kills'] += 1
                self.stats['honor_gained'] += honor

                # CONDENSED log
                self.log_event(
                    f"🏆 {killer.name} claims {dead_agent.size} trophy! Honor +{honor} (Total: {killer.honor})",
                    VerbosityLevel.NORMAL
                )

                # Thia comments on MILESTONES only
                if killer.name == "Dek" and self.thia and self.thia.is_alive():
                    if killer.honor >= 50 and killer.honor - honor < 50:
                        self.log_event(f"💬 THIA: 'Impressive, Dek. You're earning respect. Keep this momentum.'",
                                       VerbosityLevel.NORMAL)
                    elif killer.honor >= 100 and killer.honor - honor < 100:
                        self.log_event(f"💬 THIA: 'Your honor is substantial now. The clan will take notice.'",
                                       VerbosityLevel.NORMAL)

            elif isinstance(dead_agent, UltimateAdversary):
                honor = killer.record_kill('adversary')
                trophy = dead_agent.get_trophy_value()
                killer.add_trophy(trophy)
                self.quest_completed = True
                self.stats['adversary_defeated'] = True
                self.stats['honor_gained'] += honor
                self.log_event(f"🎉 QUEST COMPLETE! {killer.name} has defeated the Ultimate Adversary!",
                               VerbosityLevel.QUIET)

                if self.thia and self.thia.is_alive():
                    self.log_event(
                        f"💬 THIA: 'Incredible! Against all odds, you've prevailed. Your honor is restored, Dek.'",
                        VerbosityLevel.QUIET
                    )

        # Remove from environment
        self.environment.remove_agent(dead_agent)

    def run_turn(self):
        """Execute one full simulation turn"""
        if not self.running or self.paused:
            return

        self.stats['turns_elapsed'] += 1
        self.environment.increment_turn()

        # Thia provides strategic updates - REDUCED FREQUENCY (every 100 turns)
        if self.stats[
            'turns_elapsed'] % 100 == 0 and self.thia and self.thia.is_alive() and self.dek and self.dek.is_alive():
            monsters_alive = len([m for m in self.monsters if m.is_alive()])
            self.log_event(
                f"💬 THIA STATUS [Turn {self.stats['turns_elapsed']}]: {monsters_alive} hostiles, Honor: {self.dek.honor}",
                VerbosityLevel.QUIET
            )

            # Strategic advice based on situation
            if self.dek.honor >= 80 and self.adversary and self.adversary.is_alive():
                dist = self.environment.get_distance(self.dek.position[0], self.dek.position[1],
                                                     self.adversary.position[0], self.adversary.position[1])
                self.log_event(
                    f"💬 THIA: 'Ready for adversary. Distance: {dist} units.'",
                    VerbosityLevel.NORMAL
                )

        # Execute turns for all living agents
        for agent in list(self.environment.agents):
            if agent.is_alive():
                self.execute_agent_turn(agent)

        # Check end conditions
        if not self.dek.is_alive():
            self.stats['dek_deaths'] += 1
            self.log_event("💀 Dek has fallen. Quest failed.", VerbosityLevel.QUIET)
            self.running = False

        if self.quest_completed:
            self.log_event(f"🎉 Quest completed in {self.stats['turns_elapsed']} turns!", VerbosityLevel.QUIET)
            self.running = False

        if self.stats['turns_elapsed'] >= MAX_TURNS:
            self.log_event("⏰ Maximum turns reached.", VerbosityLevel.QUIET)
            self.running = False

    def start(self):
        """Start the simulation"""
        self.initialize_agents()
        self.running = True
        self.log_event("=== Simulation Started ===", VerbosityLevel.QUIET)

    def pause(self):
        """Pause the simulation"""
        self.paused = not self.paused

    def stop(self):
        """Stop the simulation"""
        self.running = False

    def get_statistics(self):
        """Get current simulation statistics"""
        return {
            **self.stats,
            'dek_status': self.dek.get_status() if self.dek else None,
            'thia_status': self.thia.get_status() if self.thia else None,
            'monsters_alive': len([m for m in self.monsters if m.is_alive()]),
            'adversary_alive': self.adversary.is_alive() if self.adversary else False
        }