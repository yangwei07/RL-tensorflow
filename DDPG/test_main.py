"""
Make it more robust.
Stop episode once the finger stop at the final position for 50 steps.
Feature & reward engineering.
"""
from DDPG.env.arm_env import ArmEnv
from DDPG.test_rl import DDPG
import matplotlib.pyplot as plt
import numpy as np

MAX_EPISODES = 500
MAX_EP_STEPS = 200
ON_TRAIN = True

# set env
env = ArmEnv()
s_dim = env.state_dim
a_dim = env.action_dim
a_bound = env.action_bound

# set RL method (continuous)
rl = DDPG(a_dim, s_dim, a_bound)

steps = []
def train():
    reward = []
    policy = []
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    plt.ion()
    plt.show()
    # start training
    for i in range(MAX_EPISODES):
        s = env.reset()
        ep_r = 0.
        ep_q = 0.
        for j in range(MAX_EP_STEPS):
            env.render()

            a = rl.choose_action(s)

            s_, r, t = env.step(a)

            rl.store_transition(s, a, r, s_)

            if rl.memory_full:
                # start to learn once has fulfilled the memory
                ep_r += r
                ep_q += np.amax(rl.learn())

            s = s_
            if t or j == MAX_EP_STEPS-1:
                print('episode: %i | step: %i | %s | reward: %.1f | q_value: %.4f' % (
                    i, j, '---' if not t else 'done', ep_r, ep_q / j))
                reward = np.append(reward, ep_r)
                policy = np.append(policy, ep_q / j)
                ax1.cla()
                ax2.cla()
                ax1.plot(reward, 'b')
                ax2.plot(policy, 'b')
    rl.save()


def eval():
    rl.restore()
    env.render()
    env.viewer.set_vsync(True)
    while True:
        s = env.reset()
        for _ in range(200):
            env.render()
            a = rl.choose_action(s)
            s, r, done = env.step(a)
            if done:
                break


if ON_TRAIN:
    train()
else:
    eval()



