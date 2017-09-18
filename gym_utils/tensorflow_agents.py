import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

def Q_network(state, num_outputs, hiddens = [20]):
    layer_dimensions = \
        state.get_shape().as_list()[1:] + hiddens + [num_outputs]
    d = layer_dimensions
    num_hidden_dim = len(d)-1
    weights = [None]*num_hidden_dim
    biases = [None]*num_hidden_dim

    # Create params
    with tf.variable_scope("params") as vs:
      for i in range(num_hidden_dim):
        weights[i] = tf.Variable(
            tf.random_normal((d[i],d[i+1])), name='weights'+str(i+1))
        biases[i] = tf.Variable(
            tf.zeros(d[i+1]), name='biases'+str(i+1))
    
    # Build graph
    fc = state
    for i in range(num_hidden_dim - 1):
        fc = tf.nn.relu(tf.matmul(fc, weights[i]) + biases[i]) 
    Qs = tf.matmul(fc, weights[-1]) + biases[-1]

    # Returns the output Q-values and network params
    return Qs
    
    
def mlp(inpt, num_outputs, hiddens = [20], activation_fn=tf.nn.relu):
    out = inpt
    for hidden in hiddens:
        out = layers.fully_connected(
            out, num_outputs=hidden, activation_fn=activation_fn)
    out = layers.fully_connected(
        out, num_outputs=num_outputs, activation_fn=None)
    return out
    
def deepmind_CNN(inpt, num_outputs):
    #initializer = tf.truncated_normal_initializer(0, 0.1, seed=seed)
    activation_fn = tf.nn.relu
    
    inpt = tf.transpose(inpt, perm=[0, 2, 3, 1])

    l1, w['l1_w'], w['l1_b'] = layers.convolution2d(inpt,
      32, [8, 8], [4, 4], activation_fn)
    l2, w['l2_w'], w['l2_b'] = layers.convolution2d(l1,
      64, [4, 4], [2, 2], activation_fn)
    l3, w['l3_w'], w['l3_b'] = layers.convolution2d(l2, 
      64, [3, 3], [1, 1], activation_fn)

    shape = l3.get_shape().as_list()
    l3_flat = tf.reshape(l3, [-1, reduce(lambda x, y: x * y, shape[1:])])

    out = mlp(l3_flat, num_outputs, [128])

    # Returns the network output, parameters
    return embedding

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

