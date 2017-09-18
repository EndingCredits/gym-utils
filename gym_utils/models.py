import numpy as np


class FrameBuffer(object):
    '''
    FrameBuffer implements a helper class which prevents that gym observations
    are constantly reinstantiated. It will prevent your python process
    from constantly allocating memory and thus being killed eventually.
    Only call `__array__` on it before you pass it into your model.
    '''
    def __init__(self, frames):
        '''
        Keep a reference of the framebuffer in that instance
        '''
        self._frames = frames

    def __array__(self, dtype=None):
        '''
        This is where the magic happens. All frames are concatenated along
        the second axis and returned in the prefered datatype.
        '''
        out = np.concatenate(self._frames, axis=2)
        if dtype is not None:
            out = out.astype(dtype)
        return out
