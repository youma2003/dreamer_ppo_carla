"""MLP world model predicting next state, VRU risk, and route progress."""
import torch
import torch.nn as nn


class WorldModel(nn.Module):
    def __init__(self, state_dim=28, action_dim=4, hidden=256):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.fc1 = nn.Linear(state_dim + action_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.head_state = nn.Linear(hidden, state_dim)
        self.head_risk = nn.Linear(hidden, 1)
        self.head_progress = nn.Linear(hidden, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        next_state_hat = self.head_state(x)
        risk_hat = self.sigmoid(self.head_risk(x))
        progress_hat = self.head_progress(x)
        return next_state_hat, risk_hat, progress_hat
