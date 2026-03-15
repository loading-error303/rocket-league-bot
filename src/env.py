"""RLGym environment wrapper for Stable Baselines3 compatibility."""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Any, Dict, List, Optional, Tuple

from rlgym.api import RLGym
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

from .rewards import (
    SpeedReward,
    SpeedTowardBallReward,
    BallSpeedReward,
    BoostPickupReward,
    BoostUsageReward,
    BoostPadDirectionReward,
    SupersonicReward,
    AirReward,
    FlipPenalty,
    OwnGoalPenalty,
    WiffOrWeakShotPenalty,
)


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
    action_repeat: int = 12,  # Increased from 8 to 12 (50% speed boost)
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
        # (TouchReward(), 1),             # +0.1 for touching ball
        # (GoalReward(), 10.0),             # +10 for scoring
        # (OwnGoalPenalty(), 10.0),          # -30 when conceding
        # (WiffOrWeakShotPenalty(), 3.0),   # Penalty for wiffs/weak touches
        # (SpeedReward(), 2.0),            # Small continuous reward for going fast
        (SupersonicReward(), 1.0),        # Reward for maintaining supersonic speed
        # # (SpeedTowardBallReward(), 0.05),  # Reward for speed toward ball (kickoffs!)
        # # (AirReward(), 0.002),             # Tiny reward for being airborne
        # (BallSpeedReward(), 0.05),        # Reward for ball moving fast (powerful hits!)
        # (BoostPickupReward(), 6.1),       # Reward for collecting boost pads
        # (BoostUsageReward(), 5.0),        # High reward for using boost!
        # #(BoostPadDirectionReward(), 1.0),# Reward for moving toward boost in front of car
        # (FlipPenalty(), 5.0),             # -0.1 penalty each time car flips
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
