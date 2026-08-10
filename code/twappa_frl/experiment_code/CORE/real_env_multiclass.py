import numpy as np


class CICIDSMultiClassEnv:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.n_samples = len(X)
        self.idx = 0

    def reset(self):
        self.idx = 0
        return self.X[self.idx]

    def step(self, action):
        true_label = int(self.y[self.idx])

        # Reward design:
        # correct classification => +3
        # wrong classification => -2
        if action == true_label:
            reward = 3
        else:
            reward = -2

        self.idx += 1
        done = self.idx >= self.n_samples

        if done:
            next_state = np.zeros_like(self.X[0], dtype=np.float32)
        else:
            next_state = self.X[self.idx]

        info = {"true_label": true_label}
        return next_state, reward, done, info