import numpy as np
import tensorflow as tf

class SimpleQNetAgent:
    def __init__(self, 
                 obs_size, 
                 num_actions,
                 model_fn,
                 discount = 0.9,
                 epsilon = 0.1,
                 learning_rate = 0.00025,
                 double_q = True,
                 target_network_update_step = 1000):
                 
        # Agent params
        self.input_shape = list(obs_size)
        self.num_actions = num_actions
        self.discount = discount
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.double_q = double_q
        self.target_network_update_step = target_network_update_step
        
        # Initialise train step
        self.train_step = 0
        
        # Build Graph
        self.build_graph(model_fn)
        
        
    def build_graph(self, model_fn):
        # Set up tensorflow session
        self.session = tf.Session()
        
        # Placeholders
        self.state = tf.placeholder("float", [None] + self.input_shape)
        self.action = tf.placeholder('int64', [None])
        self.poststate = tf.placeholder("float", [None] + self.input_shape)
        self.reward = tf.placeholder("float", [None])
        self.terminal = tf.placeholder('float', [None])
        
        # Apply model to get output action values:
        ##################################
        
        # Build Networks
        with tf.variable_scope('prediction'):
            self.pred_qs = \
                model_fn(self.state, self.num_actions)
        with tf.variable_scope('prediction', reuse=True):
            self.pred_post_qs = \
                model_fn(self.state, self.num_actions)
        with tf.variable_scope('target'):
            self.target_post_qs = \
                model_fn(self.poststate, self.num_actions)
                
        self.pred_weights = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES,scope='prediction')
        self.target_weights = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES,scope='target')
        
        self.target_network_update = [tf.assign(t, e) for t, e in
            zip(self.target_weights, self.pred_weights)]
             
             
        # Calculate TD error
        ##################################
        
        # Get relevant action
        action_one_hot = tf.one_hot(self.action, self.num_actions, 1.0, 0.0)
        q_acted = tf.reduce_sum(self.pred_qs * action_one_hot, axis=1)
        self.pred_q = q_acted
        
        # Get target value
        if self.double_q:
            # Predict action with current network
            pred_action = tf.argmax(self.pred_post_qs, axis=1)
            pred_action_one_hot = tf.one_hot(pred_action, self.num_actions)
            # Get value of action from target network
            V_t1 = tf.reduce_sum(self.target_post_qs * pred_action_one_hot, 1)
        else:
            V_t1 = tf.reduce_max(self.target_post_qs, axis=1)
            
        # Zero out target values for terminal states
        V_t1 = V_t1 * (1.0-self.terminal)
        
        # Bellman equation
        self.target_q = tf.stop_gradient(self.reward + self.discount * V_t1)
        
        self.td_err = self.target_q - self.pred_q
        
        
        # Loss Function:
        ##################################

        # Huber loss, from baselines
        total_loss = tf.where(
          tf.abs(self.td_err) < 1.0,
          tf.square(self.td_err) * 0.5,
          (tf.abs(self.td_err) - 0.5))
                         
        # Optimiser
        self.optim = tf.train.AdamOptimizer(
                        self.learning_rate).minimize(total_loss)
                        
        self.session.run(tf.global_variables_initializer())
                        
                        
        
    def getAction(self, state):
        # Get Q values from network
        Q_t = self.session.run(self.pred_qs, feed_dict={self.state: [state]})
        action = np.argmax(Q_t, axis=1)[0]
        
        # With prob epsilon select random action
        if np.random.rand() < self.epsilon:
            action = np.random.randint(0, self.num_actions)
        return action
        
        
    def tdUpdate(self, s_t0, a_t0, r_t0, s_t1, t_t1):
        # Run optimiser
        feed_dict={
          self.state: s_t0, 
          self.action: a_t0,
          self.reward: r_t0,
          self.poststate: s_t1,
          self.terminal: t_t1
        }
        self.session.run([self.optim],feed_dict=feed_dict)
        
        # Update target network weights every k steps
        if self.train_step % self.target_network_update_step == 0:
            self.session.run(self.target_network_update)
        
        # Increment train step counter
        self.train_step += 1

