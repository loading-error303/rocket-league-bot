"""RLGym environment wrapper for Stable Baselines3 compatibility."""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Any, Dict, List, Optional, Tuple

from rlgym.api import RLGym, RewardFunction, AgentID
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
from rlgym.rocket_league.done_conditions import (
    AnyCondition,
    GoalCondition,
    NoTouchTimeoutCondition,
    TimeoutCondition,
)
from rlgym.rocket_league.obs_builders import DefaultObs
from rlgym.rocket_league.reward_functions import CombinedReward, GoalReward, TouchReward
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.state_mutators import (
    FixedTeamSizeMutator,
    KickoffMutator,
    MutatorSequence,
)
from rlgym.rocket_league import common_values


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
            # Get car speed as fraction of max speed (2300 uu/s)
            velocity = car.physics.linear_velocity
            speed = np.sqrt(velocity[0]**2 + velocity[1]**2 + velocity[2]**2)
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
                rewards[agent] = max(speed_toward_ball / common_values.CAR_MAX_SPEED, 0.0)
            else:
                rewards[agent] = 0.0
        return rewards


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
        # Normalize by max ball speed (6000 uu/s)
        speed_ratio = ball_speed / common_values.BALL_MAX_SPEED
        # All agents get same reward for ball speed
        return {agent: speed_ratio for agent in agents}


class RLGymWrapper(gym.Env):
    """Gymnasium wrapper for RLGym v2 to work with Stable Baselines3.
    
    This wrapper adapts the multi-agent RLGym environment to a single-agent
    Gymnasium interface, training one agent while using random actions for opponents.
    """

    def __init__(self, rlgym_env: RLGym):
        super().__init__()
        self.rlgym_env = rlgym_env
        self._agents: list = []

        # Initialize by doing a reset to get observation shape
        obs_dict = self.rlgym_env.reset()
        self._agents = list(obs_dict.keys())

        # Use first agent (blue team) for training
        sample_obs = obs_dict[self._agents[0]]

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=sample_obs.shape,
            dtype=np.float32,
        )

        # LookupTableAction uses 90 discrete actions by default
        action_space = self.rlgym_env.action_parser.get_action_space(self._agents[0])
        if isinstance(action_space, spaces.Discrete):
            self.action_space = action_space
        else:
            self.action_space = spaces.Discrete(90)

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        """Reset the environment and return initial observation."""
        if seed is not None:
            np.random.seed(seed)

        obs_dict = self.rlgym_env.reset()
        self._agents = list(obs_dict.keys())

        obs = obs_dict[self._agents[0]]
        return obs.astype(np.float32), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one environment step."""
        # Build action dict: trained agent uses given action, opponents use random
        actions = {}
        for i, agent in enumerate(self._agents):
            if i == 0:
                actions[agent] = np.array([action], dtype=np.int64)
            else:
                actions[agent] = np.array([self.action_space.sample()], dtype=np.int64)

        obs_dict, reward_dict, terminated_dict, truncated_dict = self.rlgym_env.step(
            actions
        )

        agent_id = self._agents[0]
        obs = obs_dict[agent_id].astype(np.float32)
        reward = float(reward_dict[agent_id])
        terminated = terminated_dict[agent_id]
        truncated = truncated_dict[agent_id]

        return obs, reward, terminated, truncated, {}

    def close(self):
        """Clean up resources."""
        pass


def make_env(
    spawn_opponents: bool = True,
    team_size: int = 1,
    action_repeat: int = 8,
    no_touch_timeout: float = 30.0,
    game_timeout: float = 300.0,
) -> RLGymWrapper:
    """Create and configure the RLGym environment.

    Args:
        spawn_opponents: Whether to spawn opponent bots.
        team_size: Number of players per team.
        action_repeat: Number of physics ticks per action.
        no_touch_timeout: Seconds without ball touch before truncation.
        game_timeout: Maximum game duration in seconds.

    Returns:
        Wrapped environment compatible with Stable Baselines3.
    """
    blue_team_size = team_size
    orange_team_size = team_size if spawn_opponents else 0

    action_parser = RepeatAction(LookupTableAction(), repeats=action_repeat)

    termination_condition = GoalCondition()
    truncation_condition = AnyCondition(
        NoTouchTimeoutCondition(timeout_seconds=no_touch_timeout),
        TimeoutCondition(timeout_seconds=game_timeout),
    )

    # Reward function with speed multipliers for learning speed flips
    reward_fn = CombinedReward(
        (TouchReward(), 0.1),           # +0.1 for touching ball
        (GoalReward(), 10.0),           # +10 for scoring
        (SpeedReward(), 0.01),          # Small continuous reward for going fast
        (SpeedTowardBallReward(), 0.05),# Reward for speed toward ball (kickoffs!)
        (AirReward(), 0.002),           # Tiny reward for being airborne
        (BallSpeedReward(), 0.02),      # Reward for ball moving fast (powerful hits!)
    )

    obs_builder = DefaultObs(
        zero_padding=None,
        pos_coef=np.asarray([
            1 / common_values.SIDE_WALL_X,
            1 / common_values.BACK_NET_Y,
            1 / common_values.CEILING_Z,
        ]),
        ang_coef=1 / np.pi,
        lin_vel_coef=1 / common_values.CAR_MAX_SPEED,
        ang_vel_coef=1 / common_values.CAR_MAX_ANG_VEL,
        boost_coef=1 / 100.0,
    )

    state_mutator = MutatorSequence(
        FixedTeamSizeMutator(blue_size=blue_team_size, orange_size=orange_team_size),
        KickoffMutator(),
    )

    rlgym_env = RLGym(
        state_mutator=state_mutator,
        obs_builder=obs_builder,
        action_parser=action_parser,
        reward_fn=reward_fn,
        termination_cond=termination_condition,
        truncation_cond=truncation_condition,
        transition_engine=RocketSimEngine(),
    )

    return RLGymWrapper(rlgym_env)
