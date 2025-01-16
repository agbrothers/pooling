# noqa: D212, D415
"""
# Simple Tag [CUSTOMIZED]

This environment is part of the <a href='..'>MPE environments</a>. Please read that page first for general information.

This is a predator-prey environment. Prey agents (green) are faster and receive a negative reward for being hit by predators (red) (-10 for each collision). Predators are slower and are rewarded for hitting prey agents (+10 for each collision). Obstacle (large black circles) block the way. By
default, there is 1 prey agent, 3 predators and 2 obstacle.

So that prey agents don't run to infinity, they are also penalized for exiting the area by the following function:

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
simple_tag_v3.env(num_prey=1, num_predators=3, num_obstacles=2, max_cycles=25, continuous_actions=False)
```

`num_prey`:  number of prey agents

`num_predators`:  number of predators

`num_obstacles`:  number of obstacle

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
        num_prey=1,
        num_predators=3,
        num_obstacles=2,
        max_cycles=25,
        continuous_actions=False,
        render_mode=None,
        **kwargs,
    ):
        EzPickle.__init__(
            self,
            num_prey=num_prey,
            num_predators=num_predators,
            num_obstacles=num_obstacles,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            render_mode=render_mode,
        )
        scenario = Scenario()
        world = scenario.make_world(num_prey, num_predators, num_obstacles, **kwargs)
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
            num_prey=1, 
            num_predators=3, 
            num_obstacles=2, 
            rew_coeff_centroid=0.2,
            rew_pred_hit_obstacle=0.0,
            collidable_pred=False,
            collidable_prey=True,
            collidable_obstacles=True,
            norm_constant=1.0,
            obstacle_ics="spread",
            **kwargs
        ):
        ## SET ENTITY AND FEATURE COUNT
        self.num_entities = num_prey + num_predators + num_obstacles
        self.num_features = 5  # FEATURES PER TOKEN: [x, y, vx, vy, role]
        self.predators = None
        self.prey = None
        
        ## ROLE IDXS            # ROLE INTERPRETATION
        self.own_role = 0       # This token is my state
        self.obstacle_role = 1  # This token is an obstacle's state
        self.pred_role = 2      # This token is an predator's state
        self.prey_role = 3      # This token is an prey's state

        ## CUSTOM REWARD ARGUMENTS
        self.rew_pred_hit_obstacle = rew_pred_hit_obstacle
        self.rew_coeff_centroid = rew_coeff_centroid
        
        ## CONSTANT FOR OBS NORMALIZATION
        self.norm_constant = norm_constant

        ## CUSTOM INITIAL CONDITIONS
        self.obstacle_ics = obstacle_ics

        ## DEFAULT SIMPLE TAG INITIALIZEATION
        world = World()
        world.dim_c = 2
        num_prey_agents = num_prey
        num_agents = num_predators + num_prey_agents
        
        ## INIT AGENTS
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.adversary = True if i < num_predators else False
            base_name = "adversary" if agent.adversary else "agent"
            base_index = i if i < num_predators else i - num_predators
            agent.name = f"{base_name}_{base_index}"
            agent.collide = collidable_pred if agent.adversary else collidable_prey
            agent.silent = True
            agent.size = 0.075 if agent.adversary else 0.05
            agent.accel = 6.0 #3.0 if agent.adversary else 4.0
            agent.max_speed = 3.0 #1.0 if agent.adversary else 1.3
            agent.color = (
                np.array([0.35, 0.85, 0.35])
                if not agent.adversary
                else np.array([0.85, 0.35, 0.35])
            )        
        ## ADD LANDMARKS
        world.landmarks = [Landmark() for i in range(num_obstacles)]
        for i, obstacle in enumerate(world.landmarks):
            obstacle.name = "obstacle %d" % i
            obstacle.collide = collidable_obstacles
            obstacle.movable = False
            obstacle.size = 0.2
            obstacle.boundary = False
            obstacle.color = np.array([0.25, 0.25, 0.25])
        return world


    def reset_world(self, world, np_random):
        ## TODO: ALLOW FOR RESETTING RANDOM NUMBERS OF ENTITIES TYPES
        self.spawn_agents(world)
        self.spawn_obstacles(world)


    def spawn_agents(self, world):
        spawn_min = 0.1
        spawn_max = np.log(len(world.agents)) /  np.log(4)
        spawn_dist = np.random.uniform(low=spawn_min, high=spawn_max)
        for agent in world.agents:   
            agent.state.p_pos = spawn_dist * np.random.uniform(-1., 1., world.dim_p)
            agent.state.p_vel = np.random.uniform(-0.5, +0.5, world.dim_p)
            agent.state.c = np.zeros(world.dim_c)        
        return


    def spawn_obstacles(self, world):
        if len(world.landmarks) == 0:
            return
        spawn_dist = max(np.log(len(world.landmarks)) /  np.log(4), 2.0)
        for i, obstacle in enumerate(world.landmarks):
            ## ZERO VELOCITY
            obstacle.state.p_vel = np.zeros(world.dim_p)         
            ## DEFAULT SPAWNING 
            if self.obstacle_ics == "default":
                obstacle.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
            ## SPREAD
            elif self.obstacle_ics == "spread":
                obstacle.state.p_pos = spawn_dist * np.random.uniform(-1., 1., world.dim_p)
            ## OUT-OF-BOUNDS
            elif self.obstacle_ics == "out_of_bounds":
                noise = 2*np.random.rand() - 1
                obstacle.state.p_pos = np.ones(world.dim_p) * 6. + (noise*1e-3)
            ## RADIAL
            elif self.obstacle_ics == "radial":
                r = 8.
                theta = 2*np.pi*np.random.rand()
                obstacle.state.p_pos = np.array((r*np.sin(theta), r*np.cos(theta)))
            ## ORIGIN
            elif self.obstacle_ics == "origin":
                obstacle.state.p_pos = np.zeros(world.dim_p)
        return
    

    def reward(self, agent, world):
        return self.centroid_reward(agent, world) if agent.adversary else 0.0

    def centroid_reward(self, agent, world):
        ## MINIMIZE DISTANCE TO THE PREY CENTROID
        prey_pos = np.array([a.state.p_pos for a in world.agents if not a.adversary])
        prey_centroid = np.mean(prey_pos, axis=0)
        prey_centroid_dist = np.linalg.norm(agent.state.p_pos - prey_centroid)
        # rew = min(1.0, self.rew_coeff_centroid / (prey_centroid_dist**0.5))
        rew = -prey_centroid_dist * self.rew_coeff_centroid
        
        ## COLLIDE WITH LANDMARK
        if self.rew_pred_hit_obstacle != 0 and len(world.landmarks) > 0:
            obstacle_min = agent.size + world.landmarks[-1].size
            obstacles_pos = np.array([lm.state.p_pos for lm in world.landmarks])
            obstacles_dist = np.linalg.norm(obstacles_pos - agent.state.p_pos, axis=1)
            obstacle_collisions = obstacles_dist <= obstacle_min
            rew += sum(obstacle_collisions*self.rew_pred_hit_obstacle)
        return rew
        
    def observation(self,agent, world):
        own_token = np.hstack((agent.state.p_pos, agent.state.p_vel, [self.own_role]))
        tokens = [own_token]

        for lm in world.landmarks:
            if not lm.boundary:
                tokens.append(
                    np.hstack((lm.state.p_pos, lm.state.p_vel, [self.obstacle_role]))
                )        
        
        for entity in world.agents:
            if entity is agent: 
                continue
            role = self.pred_role if entity.adversary else self.prey_role
            tokens.append(
                np.hstack((entity.state.p_pos, entity.state.p_vel, [role]))
            )  
        tokens = np.stack(tokens)
        ## SHIFT INTO RELATIVE COORDINATE FRAME
        tokens[:, :4] -= tokens[0, :4]
        ## NORMALIZE
        tokens[:, :4] /= self.norm_constant ## ~5
        return tokens
    