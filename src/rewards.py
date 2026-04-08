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


class JumpReward(RewardFunction[AgentID, GameState, float]):
    """Reward for performing a jump.

    Only rewards genuine jumps initiated via the jump mechanic (has_jumped=True).
    Falling from the ceiling or driving off a wall without jumping gives zero reward.

    Tracks each agent's ground→air transition and only grants reward when
    the car left the ground through a jump. Reward is proportional to the
    height gained above takeoff and upward velocity.
    """

    # Approximate z of car centre when sitting on the field
    GROUND_Z = 17.01

    def __init__(self):
        self._prev_on_ground: Dict[AgentID, bool] = {}
        self._jump_origin_z: Dict[AgentID, float] = {}
        self._jump_initiated: Dict[AgentID, bool] = {}

    def reset(
        self,
        agents: List[AgentID],
        initial_state: GameState,
        shared_info: Dict[str, Any],
    ) -> None:
        self._prev_on_ground.clear()
        self._jump_origin_z.clear()
        self._jump_initiated.clear()
        for agent in agents:
            car = initial_state.cars[agent]
            self._prev_on_ground[agent] = car.on_ground
            self._jump_origin_z[agent] = car.physics.position[2]
            self._jump_initiated[agent] = False

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
            z = car.physics.position[2]
            vz = car.physics.linear_velocity[2]
            was_on_ground = self._prev_on_ground.get(agent, True)

            # Detect jump takeoff: was on ground, now airborne, jump button was used
            if was_on_ground and not car.on_ground and car.has_jumped:
                self._jump_origin_z[agent] = self.GROUND_Z
                self._jump_initiated[agent] = True

            # Landing resets the jump tracker
            if car.on_ground:
                self._jump_initiated[agent] = False
                self._jump_origin_z[agent] = z

            if self._jump_initiated.get(agent, False) and not car.on_ground:
                # Height gained since takeoff, normalised by ceiling height
                height_gained = max(0.0, z - self._jump_origin_z.get(agent, z))
                height_reward = min(height_gained / common_values.CEILING_Z, 1.0)

                # Bonus for upward velocity (rewards the rising phase more)
                vel_reward = max(0.0, vz / common_values.CAR_MAX_SPEED)

                rewards[agent] = float(height_reward + 0.5 * vel_reward)
            else:
                rewards[agent] = 0.0

            self._prev_on_ground[agent] = car.on_ground

        return rewards


class BoostPickupReward(RewardFunction[AgentID, GameState, float]):
    """Reward for picking up boost pads.

    Detects pickups by comparing each agent's boost amount to the previous step.
    Big pads (100 boost) give 5x the reward of small pads (12 boost).
    """

    # Big pads grant 100 boost, small pads grant 12 boost.
    # We use a threshold to distinguish the two.
    BIG_PAD_THRESHOLD = 50.0

    def __init__(self):
        self._prev_boost: Dict[AgentID, float] = {}

    def reset(
        self,
        agents: List[AgentID],
        initial_state: GameState,
        shared_info: Dict[str, Any],
    ) -> None:
        self._prev_boost.clear()
        for agent in agents:
            self._prev_boost[agent] = initial_state.cars[agent].boost_amount

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
            current_boost = car.boost_amount
            prev_boost = self._prev_boost.get(agent, current_boost)

            boost_gained = current_boost - prev_boost

            if boost_gained > 0:
                if boost_gained >= self.BIG_PAD_THRESHOLD:
                    # Big pad pickup — 5x reward
                    rewards[agent] = 1.0
                else:
                    # Small pad pickup — base reward (1/5 of big)
                    rewards[agent] = 0.2
            else:
                rewards[agent] = 0.0

            self._prev_boost[agent] = current_boost

        return rewards


class SpeedTowardBallReward(RewardFunction[AgentID, GameState, float]):
    """Reward for moving toward the ball.

    Measures the component of velocity in the direction of the ball.
    Returns 0-1 scaled by max car speed.
    """

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(
        self,
        agents: List[AgentID],
        state: GameState,
        is_terminated: Dict[AgentID, bool],
        is_truncated: Dict[AgentID, bool],
        shared_info: Dict[str, Any],
    ) -> Dict[AgentID, float]:
        ball_pos = state.ball.position
        rewards = {}
        for agent in agents:
            car = state.cars[agent]
            pos = car.physics.position
            vel = car.physics.linear_velocity

            to_ball = ball_pos - pos
            dist = np.linalg.norm(to_ball)
            if dist < 1e-5:
                rewards[agent] = 1.0
                continue
            to_ball_dir = to_ball / dist

            speed_toward_ball = np.dot(vel, to_ball_dir)
            rewards[agent] = float(np.clip(speed_toward_ball / common_values.CAR_MAX_SPEED, 0.0, 1.0))
        return rewards


class TouchBallReward(RewardFunction[AgentID, GameState, float]):
    """One-time reward for touching the ball.

    Uses the car's ball_touches counter — gives a reward each time
    the counter increments compared to the previous step.
    """

    def __init__(self):
        self._prev_touches: Dict[AgentID, int] = {}

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self._prev_touches.clear()
        for agent in agents:
            self._prev_touches[agent] = initial_state.cars[agent].ball_touches

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
            current_touches = car.ball_touches
            prev_touches = self._prev_touches.get(agent, current_touches)

            if current_touches > prev_touches:
                rewards[agent] = 1.0
            else:
                rewards[agent] = 0.0

            self._prev_touches[agent] = current_touches
        return rewards


class BallSpeedAfterTouchReward(RewardFunction[AgentID, GameState, float]):
    """Reward for how fast the ball is moving after the agent touches it.

    Only fires on the step a touch is detected. Encourages powerful hits
    rather than weak dribbles. Scaled by ball max speed.
    """

    def __init__(self):
        self._prev_touches: Dict[AgentID, int] = {}

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self._prev_touches.clear()
        for agent in agents:
            self._prev_touches[agent] = initial_state.cars[agent].ball_touches

    def get_rewards(
        self,
        agents: List[AgentID],
        state: GameState,
        is_terminated: Dict[AgentID, bool],
        is_truncated: Dict[AgentID, bool],
        shared_info: Dict[str, Any],
    ) -> Dict[AgentID, float]:
        ball_vel = state.ball.linear_velocity
        ball_speed = np.linalg.norm(ball_vel)

        rewards = {}
        for agent in agents:
            car = state.cars[agent]
            current_touches = car.ball_touches
            prev_touches = self._prev_touches.get(agent, current_touches)

            if current_touches > prev_touches:
                rewards[agent] = float(np.clip(ball_speed / common_values.BALL_MAX_SPEED, 0.0, 1.0))
            else:
                rewards[agent] = 0.0

            self._prev_touches[agent] = current_touches
        return rewards


class GoalScoredReward(RewardFunction[AgentID, GameState, float]):
    """Reward for scoring a goal, penalty for getting scored on.

    Uses the termination signal from GoalCondition. When an episode ends
    via goal, checks ball Y position to determine who scored.
    Blue attacks +Y (orange goal), orange attacks -Y (blue goal).
    """

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
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

        # Check if any agent got terminated (goal scored)
        goal_scored = any(is_terminated.get(a, False) for a in agents)

        if not goal_scored:
            for agent in agents:
                rewards[agent] = 0.0
            return rewards

        # Ball Y determines who scored: +Y = scored in orange goal (blue scores)
        ball_y = state.ball.position[1]
        blue_scored = ball_y > 0

        for agent in agents:
            car = state.cars[agent]
            is_blue = car.team_num != ORANGE_TEAM
            if (is_blue and blue_scored) or (not is_blue and not blue_scored):
                rewards[agent] = 1.0   # Scored
            else:
                rewards[agent] = -1.0  # Got scored on

        return rewards
