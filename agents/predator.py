"""
Predator agent class (Yautja species)
"""
import random
from .base_agents import BaseAgent
from config.settings import *


class Predator(BaseAgent):
    """Predator/Yautja agent class"""

    def __init__(self, name, position, role="warrior"):
        """
        Initialize Predator agent

        Args:
            name (str): Predator name
            position (tuple): Starting position
            role (str): Clan role (runt, warrior, elite, elder)
        """
        super().__init__(name, position, PREDATOR_BASE_HEALTH, PREDATOR_BASE_STAMINA)
        self.agent_type = "predator"
        self.role = role
        self.honor = PREDATOR_BASE_HONOR
        self.trophies = []
        self.kills = {
            'small': 0,
            'medium': 0,
            'large': 0,
            'adversary': 0
        }
        self.carrying = None  # Can carry Thia or items
        self.territory = None  # Claimed territory
        self.clan_standing = "outcast" if role == "runt" else "member"
        self.dishonor_acts = 0

    def add_honor(self, amount, reason=""):
        """
        Add honor points

        Args:
            amount (int): Honor to add
            reason (str): Reason for honor gain
        """
        self.honor += amount
        self._update_clan_standing()
        return amount

    def remove_honor(self, amount, reason=""):
        """
        Remove honor points

        Args:
            amount (int): Honor to remove
            reason (str): Reason for honor loss
        """
        self.honor -= amount
        if amount > 0:
            self.dishonor_acts += 1
        self._update_clan_standing()
        return -amount

    def _update_clan_standing(self):
        """Update clan standing based on honor"""
        if self.honor >= HONOR_ELDER_THRESHOLD:
            self.clan_standing = "elder"
        elif self.honor >= HONOR_WARRIOR_THRESHOLD:
            self.clan_standing = "respected_warrior"
        elif self.honor >= HONOR_RESPECT_THRESHOLD:
            self.clan_standing = "warrior"
        elif self.honor >= 0:
            self.clan_standing = "member"
        else:
            self.clan_standing = "dishonored"

    def add_trophy(self, trophy):
        """
        Add a trophy

        Args:
            trophy (dict): Trophy item

        Returns:
            bool: True if added successfully
        """
        if len(self.trophies) < TROPHY_SLOTS:
            self.trophies.append(trophy)
            return True
        return False

    def record_kill(self, creature_type):
        """
        Record a kill and award appropriate honor

        Args:
            creature_type (str): Type of creature killed

        Returns:
            int: Honor gained
        """
        honor_gain = 0

        if creature_type == 'small':
            self.kills['small'] += 1
            honor_gain = HONOR_SMALL_KILL
        elif creature_type == 'medium':
            self.kills['medium'] += 1
            honor_gain = HONOR_MEDIUM_KILL
        elif creature_type == 'large':
            self.kills['large'] += 1
            honor_gain = HONOR_LARGE_KILL
        elif creature_type == 'adversary':
            self.kills['adversary'] += 1
            honor_gain = HONOR_ADVERSARY_KILL

        self.add_honor(honor_gain, f"Killed {creature_type} creature")
        return honor_gain

    def can_carry(self, item_or_agent):
        """Check if can carry something"""
        return self.carrying is None

    def pick_up(self, item_or_agent):

        if self.can_carry(item_or_agent):
            self.carrying = item_or_agent
            return True
        return False

    def put_down(self):
        """Put down carried object"""
        carried = self.carrying
        self.carrying = None
        return carried

    def is_carrying(self):
        """Check if carrying something"""
        return self.carrying is not None

    def get_movement_stamina_cost(self, base_cost, terrain_multiplier=1.0):
        """
        Calculate stamina cost for movement

        Args:
            base_cost (int): Base movement cost
            terrain_multiplier (float): Terrain difficulty multiplier

        Returns:
            int: Total stamina cost
        """
        cost = base_cost * terrain_multiplier
        if self.is_carrying():
            cost *= CARRY_STAMINA_MULTIPLIER
        return int(cost)

    def can_move(self, base_cost, terrain_multiplier=1.0):
        """Check if has enough stamina to move"""
        cost = self.get_movement_stamina_cost(base_cost, terrain_multiplier)
        return self.stamina >= cost

    def rest(self):
        """Rest to recover stamina and health"""
        self.recover_stamina(REST_STAMINA_RECOVERY)
        self.heal(REST_HEALTH_RECOVERY)

    def violate_code(self, violation_type):
        """
        Handle Yautja code violation

        Args:
            violation_type (str): Type of violation

        Returns:
            int: Honor penalty
        """
        penalties = {
            'unworthy_prey': UNWORTHY_PREY_PENALTY,
            'dishonor_kill': DISHONOR_KILL_PENALTY,
            'territory_violation': TERRITORY_VIOLATION_PENALTY
        }

        penalty = penalties.get(violation_type, HONOR_DISHONOR_PENALTY)
        self.remove_honor(abs(penalty), f"Code violation: {violation_type}")
        return penalty

    def challenge_predator(self, other_predator):
        """
        Challenge another predator

        Args:
            other_predator (Predator): Predator to challenge

        Returns:
            str: Challenge result
        """
        # Honor-based challenge system
        if self.honor > other_predator.honor:
            return "dominant"
        elif self.honor < other_predator.honor:
            return "submit"
        else:
            return "equal"

    def decide_action(self, environment):
        """
        Enhanced AI decision making for predator with active hunting

        Args:
            environment: Game environment

        Returns:
            dict: Action decision
        """

        if self.name == "Dek" and self.honor >= 80:
            # Find adversary manually by checking all agents
            adversary = None
            for agent in environment.agents:
                if hasattr(agent, 'agent_type') and agent.agent_type == 'adversary' and agent.is_alive():
                    adversary = agent
                    break

            if adversary:
                return {"action": "hunt", "target": adversary}

        # PRIORITY 1: Critical health - rest or retreat
        if self.get_health_percentage() < 30:
            if self.stamina < self.max_stamina * 0.5:
                return {'action': 'rest'}
            return {'action': 'retreat'}

        # PRIORITY 2: Low stamina - rest
        if self.get_stamina_percentage() < 20:
            return {'action': 'rest'}

        # PRIORITY 3: Look for prey in immediate range (adjacent cells)
        x, y = self.position
        adjacent_agents = environment.get_agents_in_range(x, y, 1)
        adjacent_monsters = [a for a in adjacent_agents if a.agent_type == 'monster' and a.is_alive()]

        if adjacent_monsters:
            # Attack if monster is adjacent
            target = adjacent_monsters[0]
            return {'action': 'attack', 'target': target}

        # PRIORITY 4: Look for prey in detection range
        from config.settings import AGGRO_RANGE
        nearby_agents = environment.get_agents_in_range(x, y, AGGRO_RANGE)
        nearby_monsters = [a for a in nearby_agents if a.agent_type == 'monster' and a.is_alive()]

        if nearby_monsters:
            # Move toward nearest monster
            target = min(nearby_monsters, key=lambda m:
            environment.get_distance(x, y, m.position[0], m.position[1]))
            return {'action': 'hunt', 'target': target}

        # PRIORITY 5: ACTIVE HUNTING - Find nearest monster on entire map
        all_monsters = [a for a in environment.agents
                        if a.agent_type == 'monster' and a.is_alive()]

        if all_monsters:
            # Find nearest monster anywhere on the map
            target = min(all_monsters, key=lambda m:
            environment.get_distance(x, y, m.position[0], m.position[1]))

            distance = environment.get_distance(x, y, target.position[0], target.position[1])

            # If target is far, actively hunt it
            if distance > 2:
                return {'action': 'hunt', 'target': target}
            else:
                return {'action': 'attack', 'target': target}

        # PRIORITY 6: If honor is high enough, seek the adversary
        if self.honor >= 80:
            adversary = None
            for agent in environment.agents:
                if hasattr(agent, 'agent_type') and agent.agent_type == 'adversary' and agent.is_alive():
                    adversary = agent
                    break

            if adversary:
                distance = environment.get_distance(x, y,
                                                    adversary.position[0],
                                                    adversary.position[1])
                if distance <= 1:
                    return {'action': 'attack', 'target': adversary}
                else:
                    return {'action': 'hunt', 'target': adversary}

        # PRIORITY 7: No targets - rest to recover
        if self.stamina < self.max_stamina or self.health < self.max_health:
            return {'action': 'rest'}

        # PRIORITY 8: Explore/patrol
        return {'action': 'patrol'}

    def get_status(self):
        """Get detailed predator status"""
        status = super().get_status()
        status.update({
            'role': self.role,
            'honor': self.honor,
            'clan_standing': self.clan_standing,
            'trophies': len(self.trophies),
            'kills': self.kills,
            'carrying': self.carrying.name if self.carrying else None,
            'dishonor_acts': self.dishonor_acts
        })
        return status

    def __repr__(self):
        """String representation"""
        return f"Predator({self.name}, {self.role}, honor={self.honor}, pos={self.position})"