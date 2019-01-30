import tensorflow as tf
import numpy as np
from collections import deque
import random

ACTOR_LR = 0.001
CRITIC_LR = 0.001
GAMMA = 0.9
TAU = 0.01
BATCH_SIZE = 32
OUTPUT_GRAPH = False


class DDPG:
    def __init__(self, s_dim, a_dim, a_bound, sess,
                 actor_lr=ACTOR_LR,
                 critic_lr=CRITIC_LR,
                 gamma=GAMMA,
                 tau=TAU,
                 batch_size=BATCH_SIZE):
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.a_bound = a_bound
        self.sess = sess
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        # Actor Network
        with tf.variable_scope('Actor'):
            with tf.variable_scope('eval'):
                self.actor_s = tf.placeholder(tf.float32, shape=[None, self.s_dim], name='state')
                self.actor_a = self.actor_network(self.actor_s, name='action', trainable=True)
                self.actor_eval_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor/eval')
            with tf.variable_scope('target'):
                self.actor_s_ = tf.placeholder(tf.float32, shape=[None, self.s_dim], name='state')
                self.actor_a_ = self.actor_network(self.actor_s_, name='action', trainable=True)
                self.actor_target_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor/target')

            self.update_actor_params = [self.actor_target_params[i].assign(
                tf.multiply(self.actor_eval_params[i], self.tau) +
                tf.multiply(self.actor_target_params[i], 1. - self.tau)
            ) for i in range(len(self.actor_target_params))]
        # Critic network
        with tf.variable_scope('Critic'):
            with tf.variable_scope('eval'):
                self.critic_s = tf.placeholder(tf.float32, shape=[None, self.s_dim], name='state')
                self.critic_a = tf.placeholder(tf.float32, shape=[None, self.a_dim], name='action')
                self.policy = self.critic_network(self.critic_s, self.critic_a, name='policy', trainable=True)
                self.critic_eval_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Critic/eval')
            with tf.variable_scope('target'):
                self.critic_s_ = tf.placeholder(tf.float32, shape=[None, self.s_dim], name='state')
                self.critic_a_ = tf.placeholder(tf.float32, shape=[None, self.a_dim], name='action')
                self.policy_ = self.critic_network(self.critic_s_, self.critic_a_, name='policy', trainable=True)
                self.critic_target_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Critic/target')

            self.update_critic_params = [self.critic_target_params[i].assign(
                tf.multiply(self.critic_eval_params[i], self.tau) +
                tf.multiply(self.critic_target_params[i], 1. - self.tau)
            ) for i in range(len(self.critic_target_params))]

        # optimize action
        self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim], name='action_gradient')
        self.unnormalized_actor_gradients = tf.gradients(
            self.actor_a,
            self.actor_eval_params,
            -self.action_gradient,
            name='unnormalized_actor_gradients')
        self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))
        self.optimize_action = tf.train.AdamOptimizer(
            self.actor_lr).apply_gradients(zip(self.actor_gradients, self.actor_eval_params))
        # optimize policy
        self.predicted_policy = tf.placeholder(tf.float32, shape=[None, 1], name='predicted_policy')
        self.td_error = tf.losses.mean_squared_error(self.predicted_policy, self.policy)
        self.optimize_policy = tf.train.AdamOptimizer(self.critic_lr).minimize(self.td_error)
        # update action gradient
        self.action_grads = tf.gradients(self.policy, self.critic_a)
        # global initializer
        self.sess.run(tf.global_variables_initializer())

        if OUTPUT_GRAPH:
            tf.summary.FileWriter("logs/", self.sess.graph)

    def actor_network(self, state, name, trainable):
        net = tf.layers.batch_normalization(state)
        net = tf.layers.dense(net, 300, activation=tf.nn.relu, name='hidden_layer', trainable=trainable)
        action = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, name='action', trainable=trainable)
        action = tf.multiply(action, self.a_bound, name=name)
        return action

    def critic_network(self, state, action, name, trainable):
        net_s = tf.layers.batch_normalization(state)
        net_a = tf.layers.batch_normalization(action)
        state_weights = tf.get_variable('state_weights', [self.s_dim, 300], trainable=trainable)
        action_weights = tf.get_variable('action_weights', [self.a_dim, 300], trainable=trainable)
        action_biases = tf.get_variable('action_biases', [1, 300], trainable=trainable)
        net = tf.nn.relu(tf.matmul(net_s, state_weights) + tf.matmul(net_a, action_weights) + action_biases)
        policy = tf.layers.dense(net, 1, activation=tf.nn.tanh, trainable=trainable, name=name)
        return policy

    def update_target_network(self):
        self.sess.run(self.update_actor_params)
        self.sess.run(self.update_critic_params)

    def train(self, scope, state, action, policy=None):
        if scope == 'Actor':
            return self.sess.run(self.optimize_action, {self.actor_s: state, self.action_gradient: action})
        else:
            return self.sess.run(self.optimize_policy, {self.critic_s: state,
                                                        self.critic_a: action,
                                                        self.predicted_policy: policy})

    def evaluate(self, scope, state, action=None):
        if scope == 'Actor':
            return self.sess.run(self.actor_a, {self.actor_s: state})
        else:
            return self.sess.run(self.policy, {self.critic_s: state, self.critic_a: action})

    def target(self, scope, state, action=None):
        if scope == 'Actor':
            return self.sess.run(self.actor_a_, {self.actor_s_: state})
        else:
            return self.sess.run(self.policy_, {self.critic_s_: state, self.critic_a_: action})

    def action_gradients(self, state, action):
        return self.sess.run(self.action_grads, {self.critic_s: state, self.critic_a: action})

    def save(self):
        saver = tf.train.Saver()
        saver.save(self.sess, 'params/', write_meta_graph=False)

    def restore(self):
        saver = tf.train.Saver()
        saver.restore(self.sess, 'params/')


class ReplayBuffer(object):

    def __init__(self, buffer_size, random_seed=123):
        """
        The right side of the deque contains the most recent experiences
        """
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
