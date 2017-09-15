from gym import Wrapper
from collections import deque


class FrameHistoryWrapper(Wrapper):
    '''
    Gym wrapper implementing frame history, as published
    by e.g. Mnih et al. in 'Playing Atari with deep reinforcement learning'.

    Returns a FrameBuffer instance holding the past n frames.

    Credits to openai baselines - the implementation is basically the same
    '''
    def __init__(self, env, hl):
        assert env is not None
        Wrapper.__init__(self, env)

        # Set `hl`, init `frames` deque, override `observation_space`
        self.hl = hl
        self.frames = deque([], maxlen=hl)
        os = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(os[0], os[1], os[2]*hl))

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_state(), reward, done, info

    def _reset(self):
        obs = self.env.reset()
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_state()

    def _get_state(self):
        assert len(self.frames) == self.hl
        return FrameBuffer(list(self.frames))
