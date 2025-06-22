import random
import numpy as np
from typing import Tuple
from collections import deque
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium.spaces import Box, Discrete
from ray.rllib import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict
# np.set_printoptions(precision=3)

from pooling.envs.boxworld_gen import world_gen, is_empty, update_color, goal_color, wall_color, agent_color, grid_color


class BoxWorld(gym.Env):
    """Boxworld representation
    Args:
      n (int): Size of the field (n x n)
      goal_length (int): Number of keys to collect to solve the level
      num_distractor (int): Number of distractor trajectories
      distractor_length (int): Number of distractor keys in each distractor trajectory
      max_steps (int): Maximum number of env step for a given level
      collect_key (bool): If true, a key is collected immediately when its corresponding lock is opened
      world: an existing level. If None, generates a new level by calling the world_gen() function
    """

    def __init__(
            self, 
            n, 
            goal_length, 
            num_distractor, 
            distractor_length, 
            reward_gem=10,
            reward_key=1,
            reward_distractor=-1,
            reward_invalid_move=0.0,
            reward_step=0.0,
            max_steps=10**6, 
            mask=False,
            world=None, 
            render_mode="rgb_array",
            **kwargs,
        ):
        self.n = n
        self.goal_length = goal_length
        self.num_distractor = num_distractor
        self.distractor_length = distractor_length
        self.num_pairs = goal_length - 1 + distractor_length * num_distractor
        self.agents = ["agent"]
        self.upscale = 30
        self.render_mode = render_mode
        self.metadata = {"render.modes": [render_mode], "render_fps": 10}
        self.mask = mask
        self.done = False
        
        ## GET COORDINATE IDXS
        self.num_pixels = (n+2) ** 2
        self.num_features = 5  ## (r, g, b, x, y)
        dim = np.arange(n+2)
        self.coords = np.stack(np.meshgrid(dim, dim), axis=-1)[..., ::-1] / (n+1)
        self.max_tokens = self.num_pairs * 2 + 2 if mask else self.num_pixels - 4*(self.n+1) + 1 #self.num_pixels

        ## SET REWARDS
        self.reward_gem = reward_gem
        self.reward_key = reward_key
        self.reward_distractor = reward_distractor
        self.reward_invalid_move = reward_invalid_move
        self.step_cost = reward_step

        ## SET OBSERVATION & ACTION SPACES
        self.observation_space = Box(low=-1, high=1, shape=(self.max_tokens, self.num_features), dtype=np.float32)
        self.action_space = Discrete(len(ACTIONS))
        self.max_steps = max_steps

        # Game initialization
        self.owned_key = grid_color
        self.reset(world=world)
        self.num_env_steps = 0
        self.episode_reward = 0
        return
    

    def observe(self):
        ## ADD COORDINATES TO PIXELS
        obs = np.concatenate((
            self.world / 255, 
            self.coords - self.coords[*self.player_position],
        ), axis=2)

        ## SWAP PLAYER POSITION COORD AT END OF OBS LIST
        own_token = obs[*self.player_position].copy()
        obs = obs.reshape(-1, self.num_features)
        own_idx = list(np.where(np.all(obs[...,:3] == np.array(agent_color)/255, axis=-1))[0])
        wall_idxs = list(np.where(np.all(obs[...,:3] == wall_color, axis=-1))[0])
        mask_idxs = own_idx + wall_idxs

        ## MASK EMPTY SPACE PIXELS
        if self.mask:
            empty_idxs = list(np.where(np.all(obs[...,:3] == np.array(grid_color)/255, axis=-1))[0])
            mask_idxs += wall_idxs + empty_idxs

        obs = np.delete(obs, mask_idxs, axis=0)

        ## PAD OBS TO MATCH THE OBS SPACE
        if len(obs) < self.max_tokens-1:
            pad_num = self.max_tokens-1 - len(obs)
            pad = np.zeros((pad_num, self.num_features))
            pad[:, -2:] = 1.0
            obs = np.vstack((pad, obs))

        ## ADD OWN TOKEN TO THE BOTTOM OF THE STACK AND RETURN
        return np.vstack((obs, own_token))


    def reset(self, seed=None, world=None, **kwargs):
        if world is None:
            world = world_gen(
                n=self.n, 
                goal_length=self.goal_length,
                num_distractor=self.num_distractor,
                distractor_length=self.distractor_length,
                seed=seed
            )
        self.world, self.player_position, self.world_dict = world
        self.num_env_steps = 0
        self.episode_reward = 0
        self.owned_key = grid_color
        self.done = False
        return self.observe()


    def step(self, action):
        new_position = self.player_position + ACTIONS[action]
        current_position = self.player_position.copy()
        reward = self.step_cost
        self.num_env_steps += 1
        truncated = self.num_env_steps == self.max_steps
        solved = False

        # Move player if the field in the moving direction is either
        ## TRIED TO STEP THROUGH BOUNDARY
        if np.any(new_position < 1) or np.any(new_position >= self.n + 1):
            possible_move = False
            # self.world.transpose(2,0,1)

        ## STEPPED ON EMPTY SPACE
        elif is_empty(self.world[*new_position]):
            # No key, no lock
            possible_move = True

        ## STEPPED ON LOOSE KEY
        elif new_position[1] == 1 or is_empty(self.world[new_position[0], new_position[1] - 1]):
            
            ## KEY IS NOT LOCKED
            if is_empty(self.world[new_position[0], new_position[1] + 1]):
                possible_move = True
                self.owned_key = self.world[*new_position].copy()
                self.world[0, 0] = self.owned_key
                ## KEY IS THE GOAL
                if np.array_equal(self.world[*new_position], goal_color):
                    self.world[0, 0] = goal_color
                    reward += self.reward_gem
                    solved = True
                ## KEY IS NOT THE GOAL
                else:
                    reward += self.reward_key
            ## KEY IS LOCKED
            else:
                possible_move = False
        
        ## STEPPED ON LOCK
        else:
            ## HELD KEY COLOR MATCHES LOCK, BOX OPENED
            if np.array_equal(self.world[*new_position], self.owned_key):
                possible_move = True
                ## GOAL REACHED
                if np.array_equal(self.world[new_position[0], new_position[1]-1], goal_color):
                    self.world[new_position[0], new_position[1] - 1] = grid_color
                    self.world[0, 0] = goal_color
                    reward += self.reward_gem
                    solved = True
                ## NEW KEY ACQUIRED
                else:
                    # lose old key and collect new one
                    self.owned_key = np.copy(self.world[new_position[0], new_position[1] - 1])
                    self.world[new_position[0], new_position[1] - 1] = grid_color
                    self.world[0, 0] = self.owned_key
                    if self.world_dict[*new_position] == 0:
                        reward += self.reward_distractor
                        solved = True
                    else:
                        reward += self.reward_key
            ## WRONG KEY FOR LOCK
            else:
                possible_move = False
                # print("lock color is {}, but owned key is {}".format(
                #     self.world[*new_position], self.owned_key))

        if possible_move:
            self.player_position = new_position
            reward += self.reward_invalid_move
            update_color(self.world, previous_agent_loc=current_position, new_agent_loc=new_position)

        self.episode_reward += reward

        info = {
            "action.name": ACTION_LOOKUP[action],
            "action.moved_player": possible_move,
            "bad_transition": self.max_steps == self.num_env_steps,
        }
        self.done = solved or truncated
        if self.done:
            info["episode"] = {"r": self.episode_reward, "length": self.num_env_steps, "solved": solved}

        if solved:
            pass
        if truncated:
            pass
        obs = self.observe()
        return obs, reward, solved, truncated, info


    def seed(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)
        return seed


    def render(self, mode="rgb_array"):
        if mode == "rgb_array":
            img = self.world.astype(np.uint8)
            return img.repeat(self.upscale, axis=0).repeat(self.upscale, axis=1)
        return


    def save(self):
        np.save('box_world.npy', self.world)


class BoxWorldRllib(BoxWorld, MultiAgentEnv):

    def step(self, action_dict: MultiAgentDict) -> Tuple[MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:
        obs, reward, terminated, truncated, info = super().step(action_dict["agent"])
        obs_dict = {"agent":obs}
        reward_dict = {"agent":reward}
        term_dict = {"agent":terminated, "__all__":terminated}
        trun_dict = {"agent":truncated, "__all__":truncated}
        info_dict = {"agent":info}
        return obs_dict, reward_dict, term_dict, trun_dict, info_dict
    
    def reset(self, **kwargs):
        return {"agent":super().reset(**kwargs)}, {}
    

ACTION_LOOKUP = {
    0: 'move up',
    1: 'move down',
    2: 'move left',
    3: 'move right',
}
ACTIONS = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1)
}
ENTITY_LOOKUP = {
    "padding": 0,
    "agent": 1,
    "lock": 2,
    "key": 3,
    "owned_key": 4,
}

if __name__ == "__main__":
    # import pickle

    # execute only if run as a script
    env = BoxWorld(6, 2, 1, 1)
    env.seed(10)

    env.reset()
    env.step(0)
    env.render()

    env.reset()
    env.render()
    # with open('/home/nathan/PycharmProjects/relational_RL_graphs/images/ex_world.pkl', 'wb') as file:
    #     pickle.dump([env.world, env.player_position, env.world_dict], file)


# TO DO : impossible lvls ? (keys stacked right made inaccessible)
