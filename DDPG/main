from ddpg import DDPG, ReplayBuffer, OrnsteinUhlenbeckActionNoise
import tensorflow as tf
import numpy as np
from part5.env import ArmEnv

BUFFER_SIZE = 10000
RANDOM = 1234
MAX_EPISODES = 500
MAX_STEPS = 200
RENDER = True

np.random.seed(RANDOM)
tf.set_random_seed(RANDOM)

env = ArmEnv()
s_dim, a_dim, a_bound = env.state_dim, env.action_dim, env.action_bound[1]
actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(a_dim))
replay_buffer = ReplayBuffer(BUFFER_SIZE, RANDOM)

with tf.Session() as sess:
    rl = DDPG(s_dim, a_dim, a_bound, sess)
    rl.update_target_network()
    reward = []
    policy = []
    # main loop
    for i in range(MAX_EPISODES):
        state = env.reset()
        ep_reward = 0
        ep_ave_max_q = 0

        for j in range(MAX_STEPS):
            if RENDER:
                env.render()

            # Added exploration noise
            a = rl.evaluate('Actor', [state])# + actor_noise()

            s2, r, terminal = env.step(a[0])

            replay_buffer.add(
                np.reshape(state, (s_dim,)),
                np.reshape(a, (a_dim,)),
                r,
                terminal,
                np.reshape(s2, (s_dim,))
            )

            if replay_buffer.size() > rl.batch_size:
                s_batch, a_batch, r_batch, t_batch, s2_batch = replay_buffer.sample_batch(rl.batch_size)

                # Calculate targets
                target_a = rl.target('Actor', s2_batch)
                target_q = rl.target('Critic', s2_batch, target_a)
                y_i = []
                for k in range(rl.batch_size):
                    if t_batch[k]:
                        y_i.append(r_batch[k])
                    else:
                        y_i.append(r_batch[k] + rl.gamma * target_q[k])

                # Update the critic given the targets
                predicted_policy = rl.evaluate('Critic', s_batch, a_batch)
                rl.train('Critic', s_batch, a_batch, np.reshape(y_i, (rl.batch_size, 1)))
                ep_ave_max_q += np.amax(predicted_policy)

                # Update the actor policy using the sampled gradient
                a_outs = rl.evaluate('Actor', s_batch)
                grads = rl.action_gradients(s_batch, a_outs)
                rl.train('Actor', s_batch, grads[0])

                # Update target networks
                rl.update_target_network()

            state = s2
            ep_reward += r
            if terminal or j == MAX_STEPS - 1:
                reward = np.append(reward, ep_reward)
                policy = np.append(policy, ep_ave_max_q / j)
                print('Episode: %i | Step: %i | Reward: %.2f | Qmax: %.4f' %
                      (i, j, ep_reward, ep_ave_max_q / j))
                break

    rl.save()


def eval():
    rl.restore()
    env.render()
    env.viewer.set_vsync(True)
    while True:
        state = env.reset()
        for _ in range(200):
            env.render()
            a = rl.evaluate('Actor', [state]) + actor_noise()
            s, r, done = env.step(a)
            if done:
                break
