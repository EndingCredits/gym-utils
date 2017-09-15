class Rollout():
  '''
    Class storing current agent trajectory.
  '''

  def __init__(self):
    self.clear()
    
  def __len__(self):
    return self.current_step
    
  def clear(self):
    self.states = []
    self.actions = []
    self.values = []
    self.rewards = []
    self.terminals = []
    self.terminal = False
    self.last_state = None
    
    self.current_step = 0
    
    self._clear_current()

  def _clear_current(self):    
    self._curr_state = None
    self._curr_value = None
    self._curr_action = None
    self._curr_reward = None
    self._curr_terminal = False

  # These functions are a bit strange, but want to keep all values in sync
  def update_state(self, state):
    self._curr_state = state

  def update_action(self, action, value=None):
    self._curr_action = action
    self._curr_value = value
      
  def update_reward(self, reward, terminal=False):
    self._curr_reward = reward
    self._curr_terminal = terminal
      
  def step(self):
    self.current_step += 1
    self.states.append(self._curr_state)
    self.actions.append(self._curr_action)
    self.values.append(self._curr_value)
    self.rewards.append(self._curr_reward)
    self.terminals.append(self._curr_terminal)
    
    self.last_state = self._curr_state
    
    self.terminal = self.terminal or self._curr_terminal
    self._clear_current()
      
  # Helper Utils
  def get_state(self, seq_len=1, use_current=True):
    if seq_len == 1:
      return self._curr_state if use_current else self.states[-1]
    else:
      assert self.current_step >= seq_len, "Not enough steps in episode"
      return self.states[-(seq_len-1):] + [self._curr_state]
      
  def n_step_return(self):
    pass #Not implemented yet
  
      
      

