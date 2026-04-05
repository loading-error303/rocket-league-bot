"""Custom reward functions for Rocket League RL training."""

import numpy as np
from typing import Any, Dict, List

from rlgym.api import RewardFunction, AgentID
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league import common_values

ORANGE_TEAM = 1


# 50 km/h in unreal units per second (uu/s)
MIN_SPEED_THRESHOLD = 730.0


class SpeedReward(RewardFunction[AgentID, GameState, float]):
    """Reward for moving fast - encourages speed flips and fast play."""

    def reset(
        self,
        agents: List[AgentID],
        initial_state: GameState,
        shared_info: Dict[str, Any],
    ) -> None:
        pass

    def get_rewards(
        self,
        agents: List[AgentID],
        state: GameState,
        is_terminated: Dict[AgentID, bool],
        is_truncated: Dict[AgentID, bool],
        shared_info: Dict[str, Any],
    ) -> Dict[AgentID, float]:
        rewards = {}
        for agent in agents:
            car = state.cars[agent]
            velocity = car.physics.linear_velocity
            speed = np.sqrt(velocity[0] ** 2 + velocity[1] ** 2 + velocity[2] ** 2)

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

    def reset(
        self,
        agents: List[AgentID],
        initial_state: GameState,
        shared_info: Dict[str, Any],
    ) -> None:
        pass

    def get_rewards(
        self,
        agents: List[AgentID],
        state: GameState,
        is_terminated: Dict[AgentID, bool],
        is_truncated: Dict[AgentID, bool],
        shared_info: Dict[str, Any],
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
            rewards[agent] = float(
                np.clip(yaw_rate / common_values.CAR_MAX_ANG_VEL, -1.0, 1.0)
            )
        return rewards


class ForwardReward(RewardFunction[AgentID, GameState, float]):
    """
    Reward for moving in the direction the car is facing.
    Encourages intentional driving rather than just chaotic speed.
    """

    def reset(
        self,
        agents: List[AgentID],
        initial_state: GameState,
        shared_info: Dict[str, Any],
    ) -> None:
        pass

    def get_rewards(
        self,
        agents: List[AgentID],
        state: GameState,
        is_terminated: Dict[AgentID, bool],
        is_truncated: Dict[AgentID, bool],
        shared_info: Dict[str, Any],
    ) -> Dict[AgentID, float]:
        rewards = {}
        for agent in agents:
            car = state.cars[agent]

            # 1. Get the car's forward vector (where the nose is pointing)
            # In most Rocket League sims, this is the first column of the orientation matrix
            forward_vector = car.physics.rotation_mtx[:, 0]  # Shape (3,) unit vector pointing forward

            # 2. Get the current velocity vector
            velocity_vector = car.physics.linear_velocity

            # 3. Calculate Dot Product to find component of velocity along forward vector
            # Positive value = moving forward, Negative value = moving backward
            forward_speed = np.dot(forward_vector, velocity_vector)

            # 4. Normalize based on max car speed
            # We clip at 0 because this specific reward is for "Going Forward"
            reward = forward_speed / common_values.CAR_MAX_SPEED
            rewards[agent] = float(np.clip(reward, 0.0, 1.0))

        return rewards


class DriveToOpponentGoalReward(RewardFunction[AgentID, GameState, float]):
    """
    Reward for having velocity directed toward the opponent's goal.
    Blue team attacks orange goal (+Y), orange team attacks blue goal (-Y).
    """

    def __init__(self):
        self.orange_goal = np.array(common_values.ORANGE_GOAL_CENTER, dtype=np.float32)
        self.blue_goal = np.array(common_values.BLUE_GOAL_CENTER, dtype=np.float32)

    def reset(
        self,
        agents: List[AgentID],
        initial_state: GameState,
        shared_info: Dict[str, Any],
    ) -> None:
        pass

    def get_rewards(
        self,
        agents: List[AgentID],
        state: GameState,
        is_terminated: Dict[AgentID, bool],
        is_truncated: Dict[AgentID, bool],
        shared_info: Dict[str, Any],
    ) -> Dict[AgentID, float]:
        rewards = {}
        for agent in agents:
            car = state.cars[agent]
            pos = car.physics.position
            vel = car.physics.linear_velocity

            # Determine which goal to attack
            target_goal = self.blue_goal if car.team_num == ORANGE_TEAM else self.orange_goal

            # Direction from car to opponent goal (normalized)
            to_goal = target_goal - pos
            dist = np.linalg.norm(to_goal)
            if dist < 1e-5:
                rewards[agent] = 1.0
                continue
            to_goal_dir = to_goal / dist

            # Component of velocity toward the goal
            speed_toward_goal = np.dot(vel, to_goal_dir)

            # Normalize by max car speed, clip to [0, 1]
            rewards[agent] = float(np.clip(speed_toward_goal / common_values.CAR_MAX_SPEED, 0.0, 1.0))

        return rewards
