import random
import hashlib
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)


class DQNAgentMultiClass:
    def __init__(self, state_dim, action_dim, lr=1e-3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05

        self.batch_size = 128
        self.memory = []
        self.max_memory = 50000
        self.action_dim = action_dim
        self.last_sampled_indices = []
        self.memory_digest = hashlib.sha256()

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return int(torch.argmax(q_values, dim=1).item())

    def predict(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return int(torch.argmax(q_values, dim=1).item())

    def store(self, transition):
        self.memory.append(transition)
        state, action, reward, next_state, done = transition
        self.memory_digest.update(np.asarray(state, dtype=np.float32).tobytes())
        self.memory_digest.update(np.asarray([action, reward, float(done)], dtype=np.float64).tobytes())
        self.memory_digest.update(np.asarray(next_state, dtype=np.float32).tobytes())
        if len(self.memory) > self.max_memory:
            self.memory.pop(0)

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return None

        self.last_sampled_indices = random.sample(range(len(self.memory)), self.batch_size)
        batch = [self.memory[index] for index in self.last_sampled_indices]
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        current_q = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(dim=1)[0]
            target_q = rewards + self.gamma * max_next_q * (1 - dones)

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

        return float(loss.item())

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def get_weights(self):
        return {k: v.detach().cpu().clone() for k, v in self.q_net.state_dict().items()}

    def set_weights(self, weights):
        self.q_net.load_state_dict(weights)
        self.target_net.load_state_dict(weights)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        state = {}
        for k, v in self.q_net.state_dict().items():
            state[f"q_net.{k}"] = v.detach().cpu().clone()
        for k, v in self.target_net.state_dict().items():
            state[f"target_net.{k}"] = v.detach().cpu().clone()
        return state

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        q_state = {k.replace("q_net.", ""): v for k, v in state_dict.items() if k.startswith("q_net.")}
        target_state = {k.replace("target_net.", ""): v for k, v in state_dict.items() if k.startswith("target_net.")}
        self.q_net.load_state_dict(q_state)
        self.target_net.load_state_dict(target_state)

    def parameters(self):
        return self.q_net.parameters()
