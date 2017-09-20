import numpy as np


class ReplayMemory:
    def __init__(self, memory_size, obs_size):
        self.memory_size = memory_size
        self.obs_size = list(obs_size)

        if self.obs_size[0] is None:
            self.observations = [None]*self.memory_size
        else:
            self.observations = np.empty(
                [self.memory_size]+list(self.obs_size), dtype=np.float16)
        self.actions = np.empty(self.memory_size, dtype=np.int16)
        self.returns = np.empty(self.memory_size, dtype=np.float16)
        self.terminal = np.empty(self.memory_size, dtype=np.bool_)

        self.count = 0
        self.current = 0

    def add(self, obs, action, returns, terminal):
        self.observations[self.current] = obs
        self.actions[self.current] = action
        self.returns[self.current] = returns
        self.terminal[self.current] = terminal

        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.memory_size

    def _get_state(self, index, seq_len):
        # normalize index to expected range, allows negative indexes
        if seq_len == 0:
            index = index % self.count
            state = self.observations[index]
        else:
            if self.obs_size[0] is None:
                state = []
                for i in range(seq_len):
                    ind = (index-i) % self.count
                    state.append(self.observations[ind])
            else:
                state = np.zeros([seq_len]+self.obs_size)
                for i in range(seq_len):
                    ind = (index-i) % self.count
                    state[i] = self.observations[ind]
        return state

    def _uninterrupted(self, start, final):
        if self.current in range(start+1, final):
            return False
        for i in range(start, final-1):
            if self.terminal[i] is True:
                return False
        return True

    def sample(self, batch_size, seq_len=0):
        # sample random indexes
        indexes = []
        states = []
        poststates = []
        watchdog = 0
        threshold = 100
        while len(indexes) < batch_size:
            while watchdog < threshold:
                # find random index
                index = np.random.randint(1, self.count - 1)
                if seq_len is not 0:
                    start = index-seq_len
                    if not self._uninterrupted(start, index):
                        watchdog += 1
                        continue
                break
            indexes.append(index)
            states.append(self._get_state(index, seq_len))
            poststates.append(self._get_state(index+1, seq_len))

        return states, self.actions[indexes],\
            self.returns[indexes], poststates, self.terminal[indexes]
