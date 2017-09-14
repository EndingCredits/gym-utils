from gym import Wrapper
from .rollout import Rollout

class RolloutWrapper(Wrapper):
'''
  Wrapper for OpenAI gym which automatically keeps track of episode history and
  can be used to automatically calulate things like n-step return.
'''

    def __init__(self, env):
        Super(RolloutWrapper, self).__init__(env)
        self.episode = Rollout()
        
    def _step(self, action, value=None):
        self.episode.update_action(action)
        observation, reward, done, info = self.env.step(action)
        self.episode.update_reward(reward, done)
        self.episode.step()
        self.episode.update_state(observation)
        return observation, reward, done, info

    def _reset(self, **kwargs):
        self.episode.clear()
        observation = self.env.reset(**kwargs)
        self.episode.update_state(observation)
        return observation
        
        

