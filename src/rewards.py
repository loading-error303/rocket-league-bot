"""Custom reward functions for Rocket League RL training."""

import numpy as np
from typing import Any, Dict, List

from rlgym.api import RewardFunction, AgentID
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league import common_values


# 50 km/h in unreal units per second (uu/s)
MIN_SPEED_THRESHOLD = 730.0

class SpeedReward(RewardFunction[AgentID, GameState, float]):
    """Reward for moving fast - encourages speed flips and fast play."""

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(
        self, agents: List[AgentID], state: GameState,
        is_terminated: Dict[AgentID, bool], is_truncated: Dict[AgentID, bool],
        shared_info: Dict[str, Any]
    ) -> Dict[AgentID, float]:
        rewards = {}
        for agent in agents:
            car = state.cars[agent]
            velocity = car.physics.linear_velocity
            speed = np.sqrt(velocity[0]**2 + velocity[1]**2 + velocity[2]**2)
            
            if speed < MIN_SPEED_THRESHOLD:
                # Penalty scales with how slow - max penalty of -1.0 when stationary
                rewards[agent] = -1.0 * (1.0 - speed / MIN_SPEED_THRESHOLD)
            else:
                speed_ratio = speed / common_values.CAR_MAX_SPEED
                rewards[agent] = speed_ratio
        return rewards


class TurnLeftReward(RewardFunction[AgentID, GameState, float]):
    """Reward for turning left — measures yaw angular velocity.

    In RocketSim, angular_velocity[2] (z-axis) is negative when turning left.
    We negate it so the reward is positive for left turns.
    """

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(
        self, agents: List[AgentID], state: GameState,
        is_terminated: Dict[AgentID, bool], is_truncated: Dict[AgentID, bool],
        shared_info: Dict[str, Any]
    ) -> Dict[AgentID, float]:
        rewards = {}
        for agent in agents:
            car = state.cars[agent]
            # Only reward turning when wheels are on the ground
            if not car.on_ground:
                rewards[agent] = 0.0
                continue
            # angular_velocity[2] = yaw rate. Negative = turning left in RocketSim.
            yaw_rate = -car.physics.angular_velocity[2]
            # Normalize: max angular vel is ~5.5 rad/s
            rewards[agent] = float(np.clip(yaw_rate / common_values.CAR_MAX_ANG_VEL, -1.0, 1.0))
        return rewards
