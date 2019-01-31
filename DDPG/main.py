from DDPG.ddpg import DDPG, ReplayBuffer, OrnsteinUhlenbeckActionNoise
import tensorflow as tf
import numpy as np
from DDPG.car_env import CarEnv
# from DDPG.env import ArmEnv
import matplotlib.pyplot as plt

BUFFER_SIZE = 100000
RANDOM = 1234
MAX_EPISODES = 3000
MAX_STEPS = 400
RENDER = True
ON_TRAIN = True

np.random.seed(RANDOM)
tf.set_random_seed(RANDOM)

env = CarEnv()
# env = ArmEnv()
s_dim, a_dim, a_bound = env.state_dim, env.action_dim, env.action_bound['d'][1]
# s_dim, a_dim, a_bound = env.state_dim, env.action_dim, env.action_bound[1]
rl = DDPG(s_dim, a_dim, a_bound)
rl.update_network()
actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(a_dim))
replay_buffer = ReplayBuffer(BUFFER_SIZE, RANDOM)


def train():
    reward = []
    policy = []
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    plt.ion()
    plt.show()
    # main loop
    for i in range(MAX_EPISODES):
        s = env.reset()
        ep_r = 0
        ep_q = 0
        for j in range(MAX_STEPS):
            if RENDER:
                env.render()
            # Added exploration noise
            a = rl.evaluate('Actor', [s])# + actor_noise()
            s_, r, t, c = env.step(a[0])
            # s_, r, t = env.step(a[0])
            # store replay buffer
            replay_buffer.add(
                np.reshape(s, (s_dim,)),
                np.reshape(s_, (s_dim,)),
                np.reshape(a, (a_dim,)),
                r, t,
            )

            if replay_buffer.size() > rl.batch_size:
                s_batch, s2_batch, a_batch, r_batch, t_batch = replay_buffer.sample_batch(rl.batch_size)
                rl.train('Actor', s_batch, a_batch)
                rl.train('Critic', s_batch, a_batch, s2_batch,
                         np.reshape(r_batch, (rl.batch_size, 1)))
                # Update target networks
                rl.update_network()

                ep_r += r
                ep_q += np.amax(rl.evaluate('Critic', s_batch, a_batch))


            s = s_
            if t or c or j == MAX_STEPS - 1:
            # if t or j == MAX_STEPS - 1:
                print('episode: %i | step: %i | %s | reward: %.1f | q_value: %.4f' % (
                    i, j, '---' if not t else 'done', ep_r, ep_q / j))
                reward = np.append(reward, ep_r)
                policy = np.append(policy, ep_q / j)
                ax1.cla()
                ax2.cla()
                ax1.plot(reward, 'b')
                ax2.plot(policy, 'b')
                break

    rl.save()

def eval():
    rl.restore()
    env.render()
    env.viewer.set_vsync(True)
    while True:
        s = env.reset()
        for _ in range(MAX_STEPS):
            env.render()
            a = rl.evaluate('Actor', [s]) + actor_noise()
            s, r, done = env.step(a[0])
            if done:
                break


if ON_TRAIN:
    train()
else:
    eval()
