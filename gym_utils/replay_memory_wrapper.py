from gym import Wrapper
from .replay_memory import ReplayMemory

class ReplayMemoryWrapper(Wrapper):
    '''
      Wrapper for OpenAI gym which automatically adds observations to a replay
      memory which can then be sampled by the agent as needed.
    '''

    def __init__(self, env, mem_size=1000):
        super(ReplayMemoryWrapper, self).__init__(env)
        obs_size = env.observation_space.shape
        self.memory = ReplayMemory(mem_size, obs_size)
        self._curr_obs = None
        
    def _step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.memory.add(self._curr_obs, action, reward, done)
        self._curr_obs = observation
        return observation, reward, done, info

    def _reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        self._curr_obs = observation
        return observation
        
    def get_memory(self):
        return self.memory
        
        

