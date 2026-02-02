"""Custom reward functions for Rocket League RL training."""

import numpy as np
from typing import Any, Dict, List

from rlgym.api import RewardFunction, AgentID
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league import common_values


# =============================================================================
# CONSTANTS
# =============================================================================

# 50 km/h in unreal units per second (uu/s)
MIN_SPEED_THRESHOLD = 730.0


# Boost pad locations in Rocket League (x, y, z)
# Big boost pads (100 boost)
BIG_BOOST_PADS = np.array([
    [-3584, 0, 73],      # Left mid
    [3584, 0, 73],       # Right mid
    [-3072, -4096, 73],  # Blue left corner
    [3072, -4096, 73],   # Blue right corner
    [-3072, 4096, 73],   # Orange left corner
    [3072, 4096, 73],    # Orange right corner
])

# Small boost pads (12 boost) - simplified subset
SMALL_BOOST_PADS = np.array([
    [0, -4240, 70],      # Blue goal area
    [0, 4240, 70],       # Orange goal area
    [-1792, -4184, 70],
    [1792, -4184, 70],
    [-1792, 4184, 70],
    [1792, 4184, 70],
    [0, -2816, 70],
    [0, 2816, 70],
    [-940, -1100, 70],
    [940, -1100, 70],
    [-940, 1100, 70],
    [940, 1100, 70],
    [-1788, -2300, 70],
    [1788, -2300, 70],
    [-1788, 2300, 70],
    [1788, 2300, 70],
    [0, -1024, 70],
    [0, 1024, 70],
    [-1024, 0, 70],
    [1024, 0, 70],
    [-512, -512, 70],
    [512, -512, 70],
    [-512, 512, 70],
    [512, 512, 70],
])

ALL_BOOST_PADS = np.vstack([BIG_BOOST_PADS, SMALL_BOOST_PADS])


# =============================================================================
# SPEED REWARDS
# =============================================================================

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


class SpeedTowardBallReward(RewardFunction[AgentID, GameState, float]):
    """Reward for moving quickly toward the ball - key for learning kickoffs and challenges."""

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
            car_physics = car.physics if car.is_orange else car.inverted_physics
            ball_physics = state.ball if car.is_orange else state.inverted_ball
            
            player_vel = np.array(car_physics.linear_velocity)
            pos_diff = np.array(ball_physics.position) - np.array(car_physics.position)
            dist_to_ball = np.linalg.norm(pos_diff)
            
            if dist_to_ball > 0:
                dir_to_ball = pos_diff / dist_to_ball
                speed_toward_ball = np.dot(player_vel, dir_to_ball)
                
                if speed_toward_ball < MIN_SPEED_THRESHOLD:
                    rewards[agent] = 0.0
                else:
                    rewards[agent] = speed_toward_ball / common_values.CAR_MAX_SPEED
            else:
                rewards[agent] = 0.0
        return rewards


class BallSpeedReward(RewardFunction[AgentID, GameState, float]):
    """Reward for ball moving fast - encourages powerful hits and shots."""

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(
        self, agents: List[AgentID], state: GameState,
        is_terminated: Dict[AgentID, bool], is_truncated: Dict[AgentID, bool],
        shared_info: Dict[str, Any]
    ) -> Dict[AgentID, float]:
        ball_vel = state.ball.linear_velocity
        ball_speed = np.sqrt(ball_vel[0]**2 + ball_vel[1]**2 + ball_vel[2]**2)
        speed_ratio = ball_speed / common_values.BALL_MAX_SPEED
        return {agent: speed_ratio for agent in agents}


# =============================================================================
# BOOST REWARDS
# =============================================================================

class BoostPickupReward(RewardFunction[AgentID, GameState, float]):
    """Reward for collecting boost - tracks boost amount changes."""

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.prev_boost = {agent: initial_state.cars[agent].boost_amount for agent in agents}

    def get_rewards(
        self, agents: List[AgentID], state: GameState,
        is_terminated: Dict[AgentID, bool], is_truncated: Dict[AgentID, bool],
        shared_info: Dict[str, Any]
    ) -> Dict[AgentID, float]:
        rewards = {}
        for agent in agents:
            current_boost = state.cars[agent].boost_amount
            prev = self.prev_boost.get(agent, current_boost)
            
            boost_gained = max(current_boost - prev, 0)
            rewards[agent] = boost_gained / 100.0  # Normalize to 0-1
            
            self.prev_boost[agent] = current_boost
        return rewards


class BoostUsageReward(RewardFunction[AgentID, GameState, float]):
    """Reward for using boost - encourages aggressive boost usage."""

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.prev_boost = {agent: initial_state.cars[agent].boost_amount for agent in agents}

    def get_rewards(
        self, agents: List[AgentID], state: GameState,
        is_terminated: Dict[AgentID, bool], is_truncated: Dict[AgentID, bool],
        shared_info: Dict[str, Any]
    ) -> Dict[AgentID, float]:
        rewards = {}
        for agent in agents:
            current_boost = state.cars[agent].boost_amount
            prev = self.prev_boost.get(agent, current_boost)
            
            boost_used = max(prev - current_boost, 0)
            rewards[agent] = boost_used / 100.0  # Normalize to 0-1
            
            self.prev_boost[agent] = current_boost
        return rewards


class BoostPadDirectionReward(RewardFunction[AgentID, GameState, float]):
    """Reward for moving toward the closest boost pad in the car's forward direction.
    
    Optimized with vectorized numpy operations for ~10x speed improvement.
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
            
            # Only care about boost when low
            if car.boost_amount > 50:
                rewards[agent] = 0.0
                continue
            
            # Extract values once
            car_pos = np.asarray(car.physics.position)
            car_forward_xy = np.asarray(car.physics.forward[:2])
            car_vel = np.asarray(car.physics.linear_velocity)
            
            # Vectorized: compute to_pad for all pads at once (30 x 3)
            to_pads = ALL_BOOST_PADS - car_pos
            
            # Vectorized: compute distances (30,)
            dists = np.sqrt(np.sum(to_pads * to_pads, axis=1))
            
            # Mask out very close pads
            valid_mask = dists > 50
            if not np.any(valid_mask):
                rewards[agent] = 0.0
                continue
            
            # Vectorized: normalize directions (only for valid pads)
            # Add small epsilon to avoid division by zero
            dir_to_pads = to_pads / (dists[:, np.newaxis] + 1e-8)
            
            # Vectorized: facing check (XY plane only) (30,)
            facing_pads = dir_to_pads[:, 0] * car_forward_xy[0] + dir_to_pads[:, 1] * car_forward_xy[1]
            
            # Mask: must be facing pad and valid distance
            good_mask = valid_mask & (facing_pads > 0.5)
            if not np.any(good_mask):
                rewards[agent] = 0.0
                continue
            
            # Vectorized: speed toward each pad (30,)
            speed_toward = dir_to_pads[:, 0] * car_vel[0] + dir_to_pads[:, 1] * car_vel[1] + dir_to_pads[:, 2] * car_vel[2]
            
            # Final mask: must be moving toward pad fast enough
            final_mask = good_mask & (speed_toward > MIN_SPEED_THRESHOLD)
            if not np.any(final_mask):
                rewards[agent] = 0.0
                continue
            
            # Vectorized: compute rewards for valid pads
            distance_factors = 1.0 / (1.0 + dists / 1000.0)
            pad_rewards = (speed_toward / common_values.CAR_MAX_SPEED) * distance_factors * facing_pads
            
            # Take max of valid rewards
            pad_rewards[~final_mask] = 0.0
            rewards[agent] = float(np.max(pad_rewards))
        
        return rewards


# =============================================================================
# MOVEMENT REWARDS
# =============================================================================

class AirReward(RewardFunction[AgentID, GameState, float]):
    """Small reward for being in the air - encourages aerial play."""

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(
        self, agents: List[AgentID], state: GameState,
        is_terminated: Dict[AgentID, bool], is_truncated: Dict[AgentID, bool],
        shared_info: Dict[str, Any]
    ) -> Dict[AgentID, float]:
        return {agent: float(not state.cars[agent].on_ground) for agent in agents}


class FlipPenalty(RewardFunction[AgentID, GameState, float]):
    """Penalty for flipping that causes speed loss - only punish bad flips."""

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.prev_has_flipped = {agent: initial_state.cars[agent].has_flipped for agent in agents}
        self.prev_speed = {}
        for agent in agents:
            vel = initial_state.cars[agent].physics.linear_velocity
            self.prev_speed[agent] = np.sqrt(vel[0]**2 + vel[1]**2 + vel[2]**2)

    def get_rewards(
        self, agents: List[AgentID], state: GameState,
        is_terminated: Dict[AgentID, bool], is_truncated: Dict[AgentID, bool],
        shared_info: Dict[str, Any]
    ) -> Dict[AgentID, float]:
        rewards = {}
        for agent in agents:
            car = state.cars[agent]
            prev_flipped = self.prev_has_flipped.get(agent, False)
            
            vel = car.physics.linear_velocity
            current_speed = np.sqrt(vel[0]**2 + vel[1]**2 + vel[2]**2)
            prev_speed = self.prev_speed.get(agent, current_speed)
            
            # Penalty only when car flips AND loses speed
            if car.has_flipped and not prev_flipped:
                if current_speed < prev_speed:
                    rewards[agent] = -2.0  # Big penalty for speed-losing flip
                elif current_speed > prev_speed:
                    rewards[agent] = 1.0   # Reward for speed-gaining flip
                else:
                    rewards[agent] = 0.0
            else:
                rewards[agent] = 0.0
            
            self.prev_has_flipped[agent] = car.has_flipped
            self.prev_speed[agent] = current_speed
        return rewards
