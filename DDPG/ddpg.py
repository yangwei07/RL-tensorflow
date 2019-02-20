import tensorflow as tf
import numpy as np
from collections import deque
import random

ACTOR_LR = 0.0001
CRITIC_LR = 0.001
GAMMA = 0.99
TAU = 0.01
BATCH_SIZE = 64
SUMMARY_OUTPUT = True


class DDPG:
    def __init__(self, s_dim, a_dim, a_bound,
                 actor_lr=ACTOR_LR,
                 critic_lr=CRITIC_LR,
                 gamma=GAMMA,
                 tau=TAU,
                 batch_size=BATCH_SIZE,
                 summary_output=SUMMARY_OUTPUT):

        self.s_dim = s_dim
        self.a_dim = a_dim
        self.a_bound = a_bound
        self.gamma = gamma
        self.batch_size = batch_size
        self.summary_output = summary_output

        self.sess = tf.Session()
        # define inputs for neural network
        self.s = tf.placeholder(tf.float32, shape=[None, self.s_dim], name='s')
        self.s_ = tf.placeholder(tf.float32, shape=[None, self.s_dim], name='s_')
        self.r = tf.placeholder(tf.float32, shape=[None, 1], name='r')
        self.t = tf.placeholder(tf.bool, shape=[None, 1], name='t')

        # actor Network (deep q learning)
        with tf.variable_scope('Actor'):
            with tf.variable_scope('eval'):
                self.a = self.actor_network(self.s, trainable=True, name='a')
            with tf.variable_scope('target'):
                a_ = self.actor_network(self.s_, trainable=False)
        # critic network (policy gradient)
        with tf.variable_scope('Critic'):
            with tf.variable_scope('eval'):
                self.q = self.critic_network(self.s, self.a, trainable=True, name='q_value')
            with tf.variable_scope('target'):
                q_ = self.critic_network(self.s_, a_, trainable=False)

        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')
        # update network parameters
        self.update_params = [[tf.assign(at, (1 - tau) * at + tau * ae),
                               tf.assign(ct, (1 - tau) * ct + tau * ce)] for at, ae, ct, ce in zip(
                self.at_params,
                self.ae_params,
                self.ct_params,
                self.ce_params
            )]
        # critic train
        # q_target = self.r + gamma * q_
        q_target = self.skip_done(q_, self.r, self.t)
        td_error = tf.losses.mean_squared_error(q_target, self.q)
        self.critic_train = tf.train.AdamOptimizer(critic_lr).minimize(
            td_error, var_list=self.ce_params, name='tf_error')
        # actor train
        a_loss = -tf.reduce_mean(self.q)  # maximize the q
        self.actor_train = tf.train.AdamOptimizer(actor_lr).minimize(
            a_loss, var_list=self.ae_params, name='a_loss')
        # summary
        if summary_output:
            with tf.variable_scope('summary', reuse=True):
                self.ep_r = tf.Variable(0., name='ep_r')
                self.ep_q = tf.Variable(0., name='ep_q')
            tf.summary.scalar("total_reward", self.ep_r)
            tf.summary.scalar("average_q_Value", self.ep_q)
            self.summary_ops = tf.summary.merge_all()
            self.writer = tf.summary.FileWriter("logs/", self.sess.graph)
        # global initializer
        self.sess.run(tf.global_variables_initializer())

    def actor_network(self, s, trainable, name=None):
        net = tf.layers.batch_normalization(s)
        net = tf.layers.dense(net, 300, activation=tf.nn.relu, name='hidden_layer', trainable=trainable)
        net = tf.layers.batch_normalization(net)
        action = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, name='output_layer', trainable=trainable)
        action = tf.multiply(action, self.a_bound, name=name)
        return action

    def critic_network(self, s, a, trainable, name=None):
        net_s = tf.layers.batch_normalization(s)
        net_a = tf.layers.batch_normalization(a)
        s_w = tf.get_variable('state_weights', [self.s_dim, 300], trainable=trainable)
        a_w = tf.get_variable('action_weights', [self.a_dim, 300], trainable=trainable)
        a_b = tf.get_variable('action_biases', [1, 300], trainable=trainable)
        net = tf.matmul(net_s, s_w) + tf.matmul(net_a, a_w) + a_b
        net = tf.layers.batch_normalization(net)
        net = tf.nn.relu(net)
        q_value = tf.layers.dense(net, 1, trainable=trainable, name=name)
        return q_value

    def update_target_network(self):
        self.sess.run(self.update_params)

    """
    if the agent succeed to finish task with [done == True], the policy doesn't need to update
    """

    def skip_done(self, q, r, t):
        q_target = []
        for i in range(self.batch_size):
            if t[i] is True:
                q_target.append(r[i])
            else:
                q_target.append(r[i] + self.gamma * q[i])

        return q_target

    def train(self, scope, s, a, s_=None, r=None, t=None):
        if scope == 'Actor':
            return self.sess.run(self.actor_train, {self.s: s, self.a: a})
        else:
            return self.sess.run(self.critic_train,
                                 {self.s: s, self.a: a, self.s_: s_, self.r: r, self.t: t})

    def evaluate(self, scope, s, a=None):
        if scope == 'Actor':
            return self.sess.run(self.a, {self.s: s})
        else:
            return self.sess.run(self.q, {self.s: s, self.a: a})

    def save(self):
        saver = tf.train.Saver()
        saver.save(self.sess, 'params/', write_meta_graph=False)

    def restore(self):
        saver = tf.train.Saver()
        saver.restore(self.sess, 'params/')

    def summary(self, r, q, episode):
        if self.summary_output:
            record = self.sess.run(self.summary_ops, {self.ep_r: r, self.ep_q: q})
            self.writer.add_summary(record, episode)
            self.writer.flush()


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


class ActionNoise:
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
        return 'ActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)
