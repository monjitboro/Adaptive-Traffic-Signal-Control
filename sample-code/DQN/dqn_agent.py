import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

class DQNNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )

    def forward(self, x):
        return self.network(x)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size  = state_size
        self.action_size = action_size

        # Better hyperparameters
        self.gamma         = 0.99
        self.epsilon       = 1.0
        self.epsilon_min   = 0.01
        self.epsilon_decay = 0.998
        self.lr            = 0.0005
        self.batch_size    = 64
        self.target_update = 20
        self.update_counter = 0

        self.memory = deque(maxlen=20000)

        self.device       = torch.device("cpu")
        self.model        = DQNNetwork(state_size, action_size).to(self.device)
        self.target_model = DQNNetwork(state_size, action_size).to(self.device)
        self.optimizer    = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion    = nn.MSELoss()

        print("\nDQN Agent initialized")
        print("  State size   : {}".format(state_size))
        print("  Action size  : {}".format(action_size))
        print("  Network      : {} -> 128 -> 128 -> {}".format(state_size, action_size))
        print("  Batch size   : {}".format(self.batch_size))
        print("  Memory size  : 20000")
        print("  Epsilon      : {}".format(self.epsilon))
        print("  LR           : {}".format(self.lr))
        print("  Gamma        : {}".format(self.gamma))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return np.argmax(q_values.cpu().numpy())

    def replay(self):
        if len(self.memory) < self.batch_size:
            return 0.0

        batch       = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states      = torch.FloatTensor(np.array(states)).to(self.device)
        actions     = torch.LongTensor(actions).to(self.device)
        rewards     = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones       = torch.FloatTensor(dones).to(self.device)

        current_q   = self.model(states).gather(1, actions.unsqueeze(1))

        with torch.no_grad():
            next_q   = self.target_model(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = self.criterion(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_counter += 1
        if self.update_counter % self.target_update == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        return loss.item()

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, filepath):
        torch.save(self.model.state_dict(), filepath)
        print("  Model saved to {}".format(filepath))

    def load(self, filepath):
        self.model.load_state_dict(torch.load(filepath))
        self.target_model.load_state_dict(torch.load(filepath))
        print("  Model loaded from {}".format(filepath))