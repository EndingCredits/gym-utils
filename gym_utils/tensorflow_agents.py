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
        self.q_target = tf.placeholder("float", [None])
        
        # Build Networks
        with tf.variable_scope('prediction'):
            self.pred_q = \
                model_fn(self.state, self.num_actions)
        with tf.variable_scope('target'):
            self.target_pred_q = \
                model_fn(self.state, self.num_actions)
                
        self.pred_weights = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES,scope='prediction')
        self.target_weights = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES,scope='target') 
            
        # Loss function
        action_one_hot = tf.one_hot(self.action, self.num_actions, 1.0, 0.0)
        q_acted = tf.reduce_sum(self.pred_q * action_one_hot, axis=1)
        self.td_err = self.q_target - q_acted
        td_loss = tf.reduce_mean(tf.square(self.td_err))
        self.optim = tf.train.AdamOptimizer(self.learning_rate).minimize(td_loss)
        
        # Initialise variables
        self.session.run(tf.global_variables_initializer())
    
        
    def getAction(self, state):
        # Get Q values from network
        Q_t = self.session.run(self.pred_q, feed_dict={self.state: [state]})
        action = np.argmax(Q_t, axis=1)[0]
        
        # With prob epsilon select random action
        if np.random.rand() < self.epsilon:
            action = np.random.randint(0, self.num_actions)
            
        return action
        
        
    def tdUpdate(self, s_t0, a_t0, r_t0, s_t1, t_t1):
    
        # Get estimate of value, V, of s_(t+1)
        Q_t1 = self.session.run(self.pred_q, feed_dict={self.state: s_t1})
        if self.double_q:
            # Predict action with current network
            a_t1_max = np.argmax(Q_t1, axis=1)
            a_t1_max_one_hot = np.eye(self.num_actions)[a_t1_max]

            # Get value of this action from target network
            targ_Q_t1 = self.session.run(self.target_pred_q, 
                feed_dict={self.state: s_t1})
            V_t1 = np.sum(np.multiply(targ_Q_t1, a_t1_max_one_hot), axis=1)
        else:
            # Get max value from target network
            V_t1 = np.max(Q_t1, axis=1)
        
        # Set V to zero if episode has ended
        V_t1 = np.multiply(np.ones(shape=np.shape(t_t1)) - t_t1, V_t1)

        # Bellman Equation
        target_q_t = self.discount * V_t1 + r_t0
        
        # Run optimiser
        feed_dict={
          self.state: s_t0, 
          self.q_target: target_q_t, 
          self.action: a_t0
        }
        self.session.run([self.optim],feed_dict=feed_dict)
        
        # Update target network weights every k steps
        if self.train_step % self.target_network_update_step == 0:
          ops = [ self.target_weights[i].assign(self.pred_weights[i]) 
            for i in range(len(self.target_weights))]
          self.session.run(ops)
        
        # Increment train step counter
        self.train_step += 1

