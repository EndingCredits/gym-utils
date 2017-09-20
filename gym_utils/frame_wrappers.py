from gym import ObservationWrapper
import gym.spaces
import numpy as np
import scipy

from frame_history_wrapper import FrameHistoryWrapper

'''
Collection of wrappers specifically tailored to images. All inputs and outputs
are of the form
'''


class FrameGreyScaleWrapper(ObservationWrapper):
    def __init__(self, env, keep_dim=False):
        super(ImgGreyScale, self).__init__(env)
        env_shape = list(self.observation_space.shape)
        env_shape[-1] = 1
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=env_shape)

    def _observation(self, obs):
        return ImgGreyScale.process(obs)

    @staticmethod
    def process(input_image):
        greys = np.dot(input_image[..., :3], [0.299, 0.587, 0.114])
        return np.expand_dims(greys, 2)


class FrameResizeWrapper(ObservationWrapper):
    def __init__(self, env=None, x=84, y=84):
        super(ImgResize, self).__init__(env)
        self.x = x
        self.y = y
        env_shape = list(self.observation_space.shape)
        # N.B: Need to sort out for differetn obs shapes
        env_shape[0] = self.x
        env_shape[1] = self.y
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=env_shape)

    def _observation(self, obs):
        return ImgResize.process(obs, self.x, self.y)

    @staticmethod
    def process(input_image, x, y):
        return scipy.misc.imresize(input_image, (y, x))


class FrameHistoryWrapper(FrameHistoryWrapper):
    '''
    Same as vanilla history wrapper but instead of giving frames as a list,
    the frames are concatenated along the last axis.
    '''
    def __init__(self, env, hl=4):
        super(ImgFrameHistoryWrapper, self).__init__(env, hl)
        os = list(env.observation_space.shape)
        os[2] = os[2]*hl
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=tuple(os))

    def _get_state(self):
        assert len(self.frames) == self._hl
        # return FrameBuffer(list(self.frames)) # Not working
        return ImgFrameHistoryWrapper.concat_img_history(self.frames)

    @staticmethod
    def concat_img_history(frames):
        return np.concatenate(frames, axis=2)
