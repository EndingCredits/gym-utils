from gym import Wrapper


class FrameSkipWrapper(Wrapper):
    '''
    Implements a frame kipping wrapper.
    As described by Mnih et al., it returns only every 'k-th' frame
    '''
    def __init__(self, env=None, k=4):
        super(FrameSkipWrapper, self).__init__(env)
        self._k = k

    def _step(self, action):
        r = 0.0
        done = False

        # Step the env `_k` times and return the last obs
        for _ range(self._k):
            obs, reward, done, info = self.env.step(action)
            r += reward
            if done:
                break
        return obs, r, done, info
