# noqa: D212, D415
"""
# Simple Tag [CUSTOMIZED]

This environment is part of the <a href='..'>MPE environments</a>. Please read that page first for general information.

This is a predator-signal environment. Prey agents (green) are faster and receive a negative reward for being hit by agents (red) (-10 for each collision). Predators are slower and are rewarded for hitting signal agents (+10 for each collision). Obstacle (large black circles) block the way. By
default, there is 1 signal agent, 3 agents and 2 obstacle.

So that signal agents don't run to infinity, they are also penalized for exiting the area by the following function:

``` python
def bound(x):
      if x < 0.9:
          return 0
      if x < 1.0:
          return (x - 0.9) * 10
      return min(np.exp(2 * x - 2), 10)
```

Agent and adversary observations: `[self_vel, self_pos, obstacle_rel_positions, other_agent_rel_positions, other_agent_velocities]`

Agent and adversary action space: `[no_action, move_left, move_right, move_down, move_up]`

### Arguments

``` python
simple_tag_v3.env(num_signal=1, num_agents=3, num_noise=2, max_cycles=25, continuous_actions=False)
```

`num_signal`:  number of signal agents

`num_agents`:  number of agents

`num_noise`:  number of obstacle

`max_cycles`:  number of frames (a step for each agent) until game terminates

`continuous_actions`: Whether agent action spaces are discrete(default) or continuous

"""

import numpy as np
from gymnasium.utils import EzPickle

from pettingzoo.mpe._mpe_utils.core import Agent, Landmark, World
from pettingzoo.mpe._mpe_utils.scenario import BaseScenario
from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv, make_env
from pettingzoo.utils.conversions import parallel_wrapper_fn


class raw_env(SimpleEnv, EzPickle):
    def __init__(
        self,
        num_agents=1,
        num_signal=8,
        num_noise=8,
        max_cycles=128,
        continuous_actions=False,
        render_mode=None,
        **kwargs,
    ):
        EzPickle.__init__(
            self,
            num_signal=num_signal,
            num_agents=num_agents,
            num_noise=num_noise,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            render_mode=render_mode,
        )
        scenario = Scenario()
        world = scenario.make_world(num_agents, num_signal, num_noise, **kwargs)
        SimpleEnv.__init__(
            self,
            scenario=scenario,
            world=world,
            render_mode=render_mode,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
        )
        self.metadata["name"] = "mpe_centroid"


env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)


def bound(x):
    if x < 0.9:
        return 0
    if x < 1.0:
        return (x - 0.9) * 10
    return min(np.exp(2 * x - 2), 10)


class Scenario(BaseScenario):
    def make_world(
            self, 
            num_agents=1, 
            num_signal=8, 
            num_noise=8, 
            M = 3.0,
            eps_s=None,
            eps_n=None,
            rew_coeff_centroid=0.2,
            rew_positive=False,
            collidable_agents=False,
            collidable_signal=True,
            collidable_noise=True,
            norm_constant=1.0,
            **kwargs
        ):
        ## SET ENTITY AND FEATURE COUNT
        self.num_entities = num_agents + num_signal + num_noise
        self.num_features = 5  # FEATURES PER TOKEN: [x, y, vx, vy, role]
        self.agents = None
        self.signal = None

        ## PARAMETERS FOR SIGNAL AND NOISE DISTRIBUTIONS
        self.eps_s = eps_s
        self.eps_n = eps_n
        self.M = M
        
        ## ROLE IDXS            # ROLE INTERPRETATION
        self.own_role = 0       # This token is my state
        self.agent_role = 1     # This token is an agent's state
        self.signal_role = 2    # This token is an signal entity's state
        self.noise_role = 3     # This token is an noise entity's state

        ## CUSTOM REWARD ARGUMENTS
        self.rew_coeff_centroid = rew_coeff_centroid
        self.rew_positive = rew_positive
        
        ## CONSTANT FOR OBS NORMALIZATION
        self.norm_constant = norm_constant

        ## DEFAULT SIMPLE TAG INITIALIZEATION
        world = World()
        world.dim_c = 2
        num_players = num_agents + num_signal

        ## INIT AGENTS
        world.agents = [Agent() for i in range(num_players)]
        for i, entity in enumerate(world.agents):
            entity.adversary = True if i < num_agents else False
            base_name = "agent" if entity.adversary else "signal"
            base_index = i if i < num_agents else i - num_agents
            entity.name = f"{base_name}_{base_index}"
            entity.collide = collidable_agents if entity.adversary else collidable_signal
            entity.size = 0.075 if entity.adversary else 0.05
            entity.accel = 3.0 if entity.adversary else 5.0
            entity.max_speed = 6.0 
            entity.silent = True
            entity.color = (
                np.array([0.35, 0.85, 0.35])
                if not entity.adversary
                else np.array([0.85, 0.35, 0.35])
            )
        ## ADD LANDMARKS
        world.landmarks = [Landmark() for i in range(num_noise)]
        for i, noise in enumerate(world.landmarks):
            noise.name = "noise %d" % i
            noise.collide = collidable_noise
            noise.size = 0.05
            noise.movable = False
            noise.boundary = False
            noise.color = np.array([0.25, 0.25, 0.25])
        return world


    def reset_world(self, world, np_random):
        ## SPAWN SIGNAL & AGENTS
        signal_spread = self.eps_s or max(np.log(self.num_entities) /  np.log(4), 2.0)
        for entity in world.agents:   
            entity.state.p_pos = signal_spread * np.random.uniform(-1., 1., world.dim_p)
            entity.state.p_vel = np.random.uniform(-2., +2., world.dim_p)
            entity.state.c = np.zeros(world.dim_c)

        if len(world.landmarks) == 0:
            return
        
        ## SPAWN NOISE
        noise_spread = self.eps_n or max(np.log(self.num_entities) /  np.log(4), 2.0)
        distance = np.random.rand() * self.M + 0.5*signal_spread + 0.5*noise_spread
        theta = np.random.rand() * 2 * np.pi
        direction = np.array((np.sin(theta), np.cos(theta))) * distance
        for noise in world.landmarks:
            noise.state.p_vel = np.zeros(world.dim_p)         
            noise.state.p_pos = direction + noise_spread * np.random.uniform(-1., 1., world.dim_p)
        return
    

    def reward(self, agent, world):
        if not agent.adversary:
            return 0.0
        ## MINIMIZE DISTANCE TO THE SIGNAL CENTROID
        signal_pos = np.array([a.state.p_pos for a in world.agents if not a.adversary])
        signal_centroid = np.mean(signal_pos, axis=0)
        signal_centroid_dist = np.linalg.norm(agent.state.p_pos - signal_centroid)
        if self.rew_positive:
            rew = min(1.0, self.rew_coeff_centroid / (signal_centroid_dist**0.5))
        else:
            # rew = -self.rew_coeff_centroid * (signal_centroid_dist**0.5)
            rew = -self.rew_coeff_centroid * (signal_centroid_dist**2)
            # rew = -signal_centroid_dist * self.rew_coeff_centroid
        return rew
        

    def observation(self,agent, world):
        own_token = np.hstack((agent.state.p_pos, agent.state.p_vel, [self.own_role]))
        tokens = [own_token]
        for lm in world.landmarks:
            if not lm.boundary:
                tokens.append(
                    np.hstack((lm.state.p_pos, lm.state.p_vel, [self.noise_role]))
                )        
        for entity in world.agents:
            if entity is agent: 
                continue
            role = self.agent_role if entity.adversary else self.signal_role
            tokens.append(
                np.hstack((entity.state.p_pos, entity.state.p_vel, [role]))
            )  
        tokens = np.stack(tokens)
        ## SHIFT INTO RELATIVE COORDINATE FRAME
        tokens[:, :4] -= tokens[0, :4]
        ## NORMALIZE
        tokens[:, :4] /= self.norm_constant ## ~5
        return tokens
    