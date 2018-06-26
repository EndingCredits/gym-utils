
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
    self.others = []
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
    
  def update_other(self, other):
    self._curr_other = other
      
  def step(self):
    self.current_step += 1
    
    # Check if any values are missing
    assert not [ c for c in self._curr_sart if c is None ]
    
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
  def get_state(self, seq_len=0, use_current=True, tile=True):
    if seq_len == 0:
      return self._curr_sart[STATE] if use_current else self.states[-1]
    elif seq_len == 1:
      return [ self.get_state(use_current=use_current) ]
    else:
      if self.current_step >= seq_len:
          return self.states[-(seq_len-1):] + [self._curr_sart[STATE]]
      else:
          assert tile, "Not enough steps in episode"
          diff = seq_len - self.current_step - 1
          try:
              first = self.states[0]
          except:
              first = self._curr_sart[STATE]
          states = [ first ]*diff + self.states \
                     + [ self._curr_sart[STATE] ]
          return states
      
      
  def n_step_return(self, n_steps, discount=1, value_index=None):
    '''
    Calculates the n-step return based on value predictions saved in self.others
    
    n_steps is the number of steps before truncation
    discount is the discount factor
    value_index is the index in the list of others where the values are stored,
    leave none is each others is a raw float value corresponding to the value
    '''
    
    if value_index is None:
        values = self.others
    else:
        values = [ o[value_index] for o in self.others ]

    returns = []
    
    for t in range(len(self)):
      if self.current_step - t > n_steps:
        #Get truncated return
        start_t = t + n_steps
        R_t = values[start_t]
      else:
        start_t = self.current_step
        R_t = 0
          
      for i in range(start_t-1, t, -1):
        R_t = R_t * discount + self.rewards[i]
      returns.append(R_t)
    return returns


  
      
      

