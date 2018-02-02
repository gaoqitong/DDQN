import tensorflow as tf
import numpy as np
import random
from collections import deque

global num_epi

class ReplayBuffer(object):

    def __init__(self, buffer_size, random_seed = 123):
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        random.seed(random_seed)

    def add(self, s, a, r, t, s2):
        experience = (s, a, r, t, s2)
        if self.count < self.buffer_size: 
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.count

    def sample_batch(self, batch_size):

        batch = []

        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch])
        t_batch = np.array([_[3] for _ in batch])
        s2_batch = np.array([_[4] for _ in batch])

        return s_batch, a_batch, r_batch, t_batch, s2_batch

    def clear(self):
        self.buffer.clear()
        self.count = 0

class Actor(object):
    
    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, batch_size, save_path, action_bound=1):
        self.sess = sess
        self.save_path = save_path
        self.last_num_epi = -1
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.tau = tau
        self.batch_size = batch_size
        
        # Actor
        self.inputs, self.out, self.scaled_out = self.create_actor()
        self.network_params = tf.trainable_variables()
        
        # Target
        self.target_inputs, self.target_out, self.target_scaled_out = self.create_actor()
        self.target_network_params = tf.trainable_variables()[len(self.network_params):]
        self.update_target_network_params = [self.target_network_params[i]
                                             .assign(tf.multiply(self.network_params[i], self.tau) 
                                                     + tf.multiply(self.target_network_params[i], 1. - self.tau))
                                             for i in range(len(self.target_network_params))]
        
        # Action Gradient
        self.action_gradient = tf.placeholder(shape=[None, self.a_dim], dtype=tf.float32)
        
        # Policy Gradient
        self.unnormalized_actor_gradients = tf.gradients(self.out, self.network_params, -self.action_gradient)
        self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))
        
        # Train
        self.optimize = tf.train.AdamOptimizer(self.learning_rate)
        self.optimize = self.optimize.apply_gradients(zip(self.actor_gradients, self.network_params))
        
        self.num_trainable_vars = len(self.network_params) + len(self.target_network_params)
    
        self.saver = tf.train.Saver()

    def create_actor(self):
        inputs = tf.placeholder(shape=[None, self.s_dim], dtype=tf.float32)
        W1 = tf.Variable(tf.random_uniform([self.s_dim, 400], -0.125, 0.125))
        B1 = tf.Variable(tf.zeros([400]))
        L1 = tf.add(tf.matmul(inputs, W1), B1)
        L1 = tf.layers.batch_normalization(L1)
        L1 = tf.nn.relu(L1)
        W2 = tf.Variable(tf.random_uniform([400, 300], -0.05, 0.05))
        B2 = tf.Variable(tf.zeros([300]))
        L2 = tf.add(tf.matmul(L1, W2), B2)
        L2 = tf.layers.batch_normalization(L2)
        L2 = tf.nn.relu(L2)
        W3 = tf.Variable(tf.random_uniform([300, self.a_dim], -0.003, 0.003))
        B3 = tf.Variable(tf.random_uniform([self.a_dim], -0.003, 0.003))
        Out = tf.add(tf.matmul(L2, W3), B3)
        Out = tf.nn.softmax(Out)
        scaled_out = tf.one_hot(tf.argmax(Out, 1), self.a_dim)
        return inputs, Out, scaled_out
    
    def train(self, inputs, a_gradient, num_epi):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient
        })

        if num_epi%20 == 0 and num_epi!=self.last_num_epi:
            self.saver.save(self.sess, self.save_path)
            print "Actor Saved"
            self.last_num_epi = num_epi

    def predict(self, inputs):
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs
        })

    def predict_probability(self, inputs):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs
            })

    def predict_target(self, inputs):
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_inputs: inputs
        })

    def predict_from_save(self, inputs):
        self.saver.restore(self.sess, self.save_path)
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars

class Critic(object):
    
    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, gamma, num_actor_vars, save_path):
        self.sess = sess
        self.last_num_epi = -1
        self .save_path = save_path
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.gamma = gamma
        
        # Critic
        self.inputs, self.action, self.out = self.create_critic()
        self.network_params = tf.trainable_variables()[num_actor_vars:]
        
        # Target
        self.target_inputs, self.target_action, self.target_out = self.create_critic()
        self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]
        
        self.update_target_network_params = [self.target_network_params[i]
                                             .assign(tf.multiply(self.network_params[i], self.tau) 
                                                     + tf.multiply(self.target_network_params[i], 1.-self.tau)) 
                                             for i in range(len(self.target_network_params))]
        
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        self.loss = tf.losses.mean_squared_error(self.predicted_q_value, self.out)
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        
        self.action_grads = tf.gradients(self.out, self.action)

        self.saver = tf.train.Saver()
    
    def create_critic(self):
        s_inputs = tf.placeholder(shape=[None, self.s_dim], dtype=tf.float32)
        a_inputs = tf.placeholder(shape=[None, self.a_dim], dtype=tf.float32)
        W1 = tf.Variable(tf.random_uniform([self.s_dim, 400], -0.125, 0.125))
        B1 = tf.Variable(tf.zeros([400]))
        L1 = tf.add(tf.matmul(s_inputs, W1), B1)
        L1 = tf.layers.batch_normalization(L1)
        L1 = tf.nn.relu(L1)
        W2_1 = tf.Variable(tf.random_uniform([400, 300], -0.05, 0.05))
        W2_2 = tf.Variable(tf.random_uniform([self.a_dim, 300], -0.125, 0.125))
        B2 = tf.Variable(tf.zeros([300]))
        L2 = tf.add(tf.add(tf.matmul(L1, W2_1), tf.matmul(a_inputs, W2_2)), B2)
        L2 = tf.nn.relu(L2)
        W3 = tf.Variable(tf.random_uniform([300, 1], -0.003, 0.003))
        B3 = tf.Variable(tf.random_uniform([1], -0.003, 0.003))
        Out = tf.add(tf.matmul(L2, W3), B3)
#         Out = tf.nn.(Out)
        
        regularizer = tf.contrib.layers.l2_regularizer(0.01)
        tf.contrib.layers.apply_regularization(regularizer,[W1, B1, W2_1, W2_2, B2, W3, B3, L1, L2, Out])
        
        return s_inputs, a_inputs, Out
    
    def train(self, inputs, action, predicted_q_value, num_epi):
        if num_epi%20 == 0 and num_epi!=self.last_num_epi:
            self.saver.save(self.sess, self.save_path)
            print "Critic Saved"
            self.last_num_epi = num_epi

        return self.sess.run([self.out, self.optimize], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value
        })

        

    def predict(self, inputs, action):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def predict_target(self, inputs, action):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action
        })

    def predict_from_save(self, inputs, action):
        self.saver.restore(self.sess, self.save_path)
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def action_gradients(self, inputs, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

def build_summaries():
    episode_reward = tf.Variable(0.)
    tf.summary.scalar("Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0.)
    tf.summary.scalar("Qmax_Value", episode_ave_max_q)
    exploration_rate = tf.Variable(0.)
    tf.summary.scalar("Qmax_Value", exploration_rate)

    summary_vars = [episode_reward, episode_ave_max_q, exploration_rate]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars

