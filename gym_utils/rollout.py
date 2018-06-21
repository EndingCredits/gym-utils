
STATE = 0
ACTION = 1
REWARD = 2
TERMINAL = 3

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
    self.rewards = []
    self.terminals = []
    self.other = []
    self.terminal = False
    self.last_state = None
    self.current_step = 0
    
    self._clear_current()

  def _clear_current(self):    
    self._curr_sart = [ None ] * 4
    self._curr_other = {}

  # These functions are a bit strange, but want to keep all values in sync
  def update_state(self, state):
    self._curr_sart[STATE] = state

  def update_action(self, action):
    self._curr_sart[ACTION] = action
      
  def update_reward(self, reward, terminal=False):
    self._curr_sart[REWARD] = reward
    self._curr_sart[TERMINAL] = terminal
    
  def update_other(self, **kwargs):
    self._curr_other.update(kwargs)
      
  def step(self):
    self.current_step += 1
    
    # Check if any values are missing
    assert all( self._curr_sart )
    
    self.last_state = self._curr_sart[STATE]
    self.terminal = self.terminal or self._curr_sart[TERMINAL]
    
    # Update entries
    self.states.append(self._curr_sart[STATE])
    self.actions.append(self._curr_sart[ACTION])
    self.rewards.append(self._curr_sart[REWARD])
    self.terminals.append(self._curr_sart[TERMINAL])
    self.others.append(self._curr_other)
    
    self._clear_current()
      
      
  # Helper Utils
  def get_state(self, seq_len=1, use_current=True):
    if seq_len == 1:
      return self._curr_sart[STATE] if use_current else self.states[-1]
    else:
      assert self.current_step >= seq_len, "Not enough steps in episode"
      return [entry.state for entry in self.rollout[-(seq_len-1):] ] \
                + [self._curr_sart[STATE]]
      
  def n_step_return(self):
    pass #Not implemented yet
  
      
      

