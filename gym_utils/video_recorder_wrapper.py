from gym import Wrapper, error, version
from .third_party.video_recorder import VideoRecorder
from gym.utils import closer
import os
import logging
import six

logger = logging.getLogger(__name__)


class VideoRecorderWrapper(Wrapper):
    '''
    VideoRecorderWrapper records episodes for you.
    Original implementation by openai gym monitoring, basically extracted the
    video recording part.
    '''

    def __init__(self, env, directory, should_capture=None, force=False,
                 resume=False, uid=None):
        super(VideoRecorderWrapper, self).__init__(env)

        self.videos = []
        self.enabled = False
        self.video_recorder = None
        self._rec_id = None
        self.episode_id = 0
        self.env_semantics_autoreset = env.metadata.get('semantics.autoreset')

        self._start(directory, should_capture, force, resume, uid)

    def _step(self, action):
        observation, reward, done, info = self.env.step(action)
        done = self._after_step(observation, reward, done, info)
        return observation, reward, done, info

    def _after_step(self, observation, reward, done, info):
        if not self.enabled:
            return done
        if done and self.env_semantics_autoreset:
            self._reset_video_recorder()
            self.episode_id += 1
        # Record video
        self.video_recorder.capture_frame()

        return done

    def _reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        self._after_reset(observation)

        return observation

    def _after_reset(self, observation):
        if not self.enabled:
            return
        # Reset video rectorder
        self._reset_video_recorder()

        # Increment episode_id
        self.episode_id += 1

    def _reset_video_recorder(self):
        if self.video_recorder:
            self._close_video_recorder()

        self.video_recorder = VideoRecorder(
            env=self.env,
            base_path=os.path.join(
                        self.directory,
                        '{}.video{:06}'
                        .format(self.file_prefix, self.episode_id)),
            metadata={'episode_id': self.episode_id},
            enabled=self._capture_sheduled()
        )
        self.video_recorder.capture_frame()

    def _start(self, directory, should_capture=None, force=False, resume=False, uid=None):
        """Start monitoring.
        Args:
            directory (str): A per-training run directory where to record stats.

            should_capture (Optional[function, False]): function that takes in the
            index of the episode and outputs a boolean, indicating whether we should
            record a video on this episode. The default (for should_capture is None)
            is to take perfect cubes, capped at 1000. False disables video recording.

            force (bool): Clear out existing training data from this directory
            (by deleting every file prefixed with "openaigym.").
            resume (bool): Retain the training data already in this directory,
            which will be merged with our new data

            uid (Optional[str]): A unique id used as part of the suffix for the file.
            By default, uses os.getpid().
        """
        if self.env.spec is None:
            env_id = '(unknown)'
        else:
            env_id = self.env.spec.id

        if not os.path.exists(directory):
            logger.info('Creating video directory %s', directory)
            if six.PY3:
                os.makedirs(directory, exist_ok=True)
            else:
                os.makedirs(directory)

        if should_capture is None:
            should_capture = self.default_capture_schedule
        elif should_capture is False:
            should_capture = no_capture
        elif not callable(should_capture):
            raise error.Error('You must provide a function, None, or False for should_capture,'
                              'not {}: {}'.format(type(should_capture), should_capture))
        self.should_capture = should_capture

        self.directory = os.path.abspath(directory)
        # We use the 'openai-gym' prefix to determine if a file is
        # ours
        self.file_prefix = 'openai.gym'

        if not os.path.exists(directory):
            os.mkdir(directory)

        self._rec_id = clsr.register(self)
        self.enabled = True

    def _close(self):
        super(VideoRecorderWrapper, self)._close()

        if getattr(self, '_videorecorderwrapper', None):
            self.close()

    def _close_video_recorder(self):
        self.video_recorder.close()
        if self.video_recorder.functional:
            self.videos.append((self.video_recorder.path, self.video_recorder.metadata_path))

    def _capture_sheduled(self):
        return self.should_capture(self.episode_id)

    def _env_info(self):
        env_info = {
            'gym_version': version.VERSION
        }
        if self.env.spec:
            env_info['env_id'] = self.env.spec.id
        return env_info

    def __del__(self):
        self.close()

    def default_capture_schedule(self, episode_id):
        '''
        Implementation from gym.monitoring.monitoring.py
        '''
        if episode_id < 1000:
            return int(round(episode_id ** (1. / 3))) ** 3 == episode_id
        else:
            return episode_id % 1000 == 0

    def no_capture(self, episode_id):
        return False

    def close(self):
        '''Close any open rending windows.'''
        if not self.enabled:
            return
        if self.video_recorder is not None:
            self._close_video_recorder()

        clsr.unregister(self._rec_id)
        self.enabled = False

clsr = closer.Closer()
