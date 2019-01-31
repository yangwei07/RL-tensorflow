import numpy as np
import pyglet

WIDTH = 500
HEIGHT = 500


class CarEnv(object):
    def __init__(self):
        self.viewer = None
        self.dt = 0.1
        self.action_bound = {'x': [0.0, WIDTH],
                       'y': [0.0, HEIGHT],
                       't': [-2 * 3.14, 2 * 3.14],
                       's': [-0.52, 0.52],
                       'v': [0.0, 30.0],
                       'd': [-0.1, 0.1],
                       'a': [-5.0, 5.0]}
        self.ego = {'x': 250, 'y': 250, 't': 0, 's': 0, 'v': 10, 'L': 50, 'W': 20, 'l': 20}
        self.goal = {'x': 400, 'y': 350, 't': 0, 'L': 50, 'W': 50}
        self.state_dim = 8
        self.action_dim = 1
        self.on_goal = 0

    def reset(self):
        # self.ego['x'] = np.array([150], dtype=np.float32)
        # self.ego['y'] = np.array([100], dtype=np.float32)
        # self.ego['t'] = np.array([0], dtype=np.float32)
        # self.ego['s'] = np.array([0], dtype=np.float32)
        self.ego['v'] = np.array([30], dtype=np.float32)
        self.ego['x'] = self.action_bound['x'][1] * np.random.rand(1)
        self.ego['y'] = self.action_bound['y'][1] * np.random.rand(1)
        self.ego['t'] = self.action_bound['t'][1] * np.random.randn(1)
        self.ego['s'] = 0. * np.random.rand(1)
        # self.ego['v'] = self.bounds['v'][1] * np.random.rand(1)

        relative = [self.ego['x'] - self.goal['x'], self.ego['y'] - self.goal['y']]
        dist = np.sqrt(relative[0] ** 2 + relative[1] ** 2)
        states = np.concatenate((self.ego['x'],
                                 self.ego['y'],
                                 self.ego['t'],
                                 self.ego['s'],
                                 # self.ego['v'] / self.bounds['v'][1],
                                 relative[0], relative[1], dist,
                                 [1. if self.on_goal else 0.]))
        return states

    def step(self, action):
        terminal = False
        collision = False
        reward = 0.

        d = np.clip(action[0], *self.action_bound['d'])
        # a = np.clip(action[1], *self.bounds['a'])
        a = 0
        self.ego['x'] += self.dt * self.ego['v'] * np.cos(self.ego['t'])
        self.ego['y'] += self.dt * self.ego['v'] * np.sin(self.ego['t'])
        self.ego['t'] += self.dt * self.ego['v'] * np.tan(self.ego['s']) / self.ego['l']
        self.ego['s'] += self.dt * d
        self.ego['v'] += self.dt * a

        self.ego['x'] = np.clip(self.ego['x'], *self.action_bound['x'])
        self.ego['y'] = np.clip(self.ego['y'], *self.action_bound['y'])
        self.ego['t'] %= 2 * 3.14 if self.ego['t'] >= 0 else -2 * 3.14
        self.ego['s'] = np.clip(self.ego['s'], *self.action_bound['s'])
        self.ego['v'] = np.clip(self.ego['v'], *self.action_bound['v'])

        relative = [self.ego['x'] - self.goal['x'], self.ego['y'] - self.goal['y']]
        dist = np.sqrt(relative[0] ** 2 + relative[1] ** 2)

        # reward
        reward += 1 / (np.sqrt((relative[0] / WIDTH) ** 2 + (relative[1] / HEIGHT) ** 2) + 1)
        # reward += self.ego['v'] / self.bounds['v'][1]
        reward += 1 / (self.ego['s'] ** 2 / self.action_bound['s'][1] ** 2 + 1)
        reward += 1 / (d ** 2 / self.action_bound['d'][1] ** 2 + 1)
        if self.ego['x'] <= self.action_bound['x'][0] or self.ego['x'] >= self.action_bound['x'][1]:
            reward += -1
            collision = True
        if self.ego['y'] <= self.action_bound['y'][0] or self.ego['y'] >= self.action_bound['y'][1]:
            reward += -1
            collision = True
        if  (self.goal['x'] - self.goal['L'] / 2 < self.ego['x'] < self.goal['x'] + self.goal['L'] / 2
        ) and (self.goal['y'] - self.goal['W'] / 2 < self.ego['y'] < self.goal['y'] + self.goal['W'] / 2):
            reward += 1
            self.on_goal += 1
            if self.on_goal > 0:
                terminal = True
        else:
            self.on_goal = 0

        states = np.concatenate((self.ego['x'],
                                 self.ego['y'],
                                 self.ego['t'],
                                 self.ego['s'],
                                 # self.ego['v'] / self.bounds['v'][1],
                                 relative[0], relative[1], dist,
                                 [1. if self.on_goal else 0.]))
        return states, reward, terminal, collision

    def render(self):
        if self.viewer is None:
            self.viewer = Viewer(self.ego, self.goal)
        self.viewer.render(self.ego)

    def sample_action(self):
        d = self.action_bound['d'][1] * np.random.randn(1)
        # a = self.bounds['a'][1] * np.random.rand(1)
        return d


class Viewer(pyglet.window.Window):
    def __init__(self, ego, goal):
        super(Viewer, self).__init__(width=WIDTH, height=HEIGHT, resizable=False, caption='Car', vsync=False)
        self.ego = ego
        self.goal = goal

        pyglet.gl.glClearColor(1, 1, 1, 1)  # background color
        self.batch = pyglet.graphics.Batch()  # display whole batch at once

        self.ego_graph = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', draw_block(self.ego)),
            ('c3B', (255, 0, 0) * 4))  # color

        self.target = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', draw_block(self.goal)),
            ('c3B', (0, 0, 255) * 4))  # color

    def render(self, ego):
        self.ego = ego
        self._update_states()
        self.switch_to()
        self.dispatch_events()
        self.dispatch_event('on_draw')
        self.flip()

    def on_draw(self):
        self.clear()
        self.batch.draw()

    def _update_states(self):
        self.ego_graph.vertices = draw_block(self.ego)


def draw_block(obj):
    l = obj['L']
    w = obj['W']
    x1 = obj['x'] + l / 2 * np.cos(obj['t']) - w / 2 * np.sin(obj['t'])
    y1 = obj['y'] + l / 2 * np.sin(obj['t']) + w / 2 * np.cos(obj['t'])
    x2 = obj['x'] + l / 2 * np.cos(obj['t']) + w / 2 * np.sin(obj['t'])
    y2 = obj['y'] + l / 2 * np.sin(obj['t']) - w / 2 * np.cos(obj['t'])
    x3 = obj['x'] - l / 2 * np.cos(obj['t']) + w / 2 * np.sin(obj['t'])
    y3 = obj['y'] - l / 2 * np.sin(obj['t']) - w / 2 * np.cos(obj['t'])
    x4 = obj['x'] - l / 2 * np.cos(obj['t']) - w / 2 * np.sin(obj['t'])
    y4 = obj['y'] - l / 2 * np.sin(obj['t']) + w / 2 * np.cos(obj['t'])
    return [x1, y1, x2, y2, x3, y3, x4, y4]


if __name__ == '__main__':
    env = CarEnv()
    while True:
        env.reset()
        for i in range(500):
            env.render()
            s, r, done, collision = env.step(env.sample_action())
