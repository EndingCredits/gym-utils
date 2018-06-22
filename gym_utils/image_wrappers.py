from gym import ObservationWrapper
import gym.spaces
import numpy as np
import scipy

from .frame_history_wrapper import FrameHistoryWrapper

'''
Collection of wrappers specifically tailored to images. All inputs and outputs
are of the form 
'''

class ImgGreyScale(ObservationWrapper):
    def __init__(self, env, keep_dim=False):
        super(ImgGreyScale, self).__init__(env)
        env_shape = list(self.observation_space.shape)
        self.keep_dim = keep_dim
        if keep_dim:
            env_shape[-1] = 1
        else:
            env_shape = env_shape[:-1]
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=env_shape)

    def _observation(self, obs):
        return ImgGreyScale.process(obs, self.keep_dim)
        
    @staticmethod
    def process(input_image, keep_dim=False):
        greys = np.dot(input_image[...,:3], [0.299, 0.587, 0.114])
        if keep_dim:
            return np.expand_dims(greys, 2)
        else:
            return greys
        
class ImgResize(ObservationWrapper):
    def __init__(self, env=None, x=84, y=84):
        super(ImgResize, self).__init__(env)
        self.x = x ; self.y = y
        env_shape = list(self.observation_space.shape)
        # N.B: Need to sort out for different obs shapes
        env_shape[0] = self.x ; env_shape[1] = self.y
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=env_shape)

    def _observation(self, obs):
        return ImgResize.process(obs, self.x, self.y)
        
    @staticmethod
    def process(input_image, x, y):
        return scipy.misc.imresize(input_image, (y, x))


class ImgCrop(ObservationWrapper):
    def __init__(self, env=None, x=84, y=84):
        super(ImgCrop, self).__init__(env)
        self.x = x ; self.y = y
        env_shape = list(self.observation_space.shape)
        self.old_x = x ; self.old_y = y
        self.side_crop = (self.old_x - self.x) // 2
        self.top_crop = (self.old_y - self.y) // 2
        env_shape[0] = self.x ; env_shape[1] = self.y
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=env_shape)

    def _observation(self, obs):
        return ImgCrop.process(obs, self.side_crop, self.old_x - self.side_crop,
                                    self.top_crop, self.old_y - self.top_crop)
        
    @staticmethod
    def process(input_image, x_1, x_2, y_1, y_2):
        return input_image[ x_1:x_2, y_1:y_2  ]
        
        
class ImgFrameHistoryWrapper(FrameHistoryWrapper):
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
        #return FrameBuffer(list(self.frames)) # Not working
        return ImgFrameHistoryWrapper.concat_img_history(self.frames)
        
    @staticmethod
    def concat_img_history(frames):
        return np.concatenate(frames, axis=2)

