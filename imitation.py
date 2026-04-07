import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Load demos
with open('demos.pkl', 'rb') as f:
    data = pickle.load(f)
X = torch.FloatTensor(data['X'])
y = torch.FloatTensor(data['y'])

# Network: 2 hidden layers of 64 with ReLU, no activation on output
obs_dim = X.shape[1]
act_dim = y.shape[1]

policy = nn.Sequential(
    nn.Linear(obs_dim, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, act_dim),
)

optimizer = optim.Adam(policy.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# Train
for epoch in range(500):
    pred = policy(X)
    loss = loss_fn(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 50 == 0:
        print(f'Epoch {epoch+1}, Loss: {loss.item():.6f}')

torch.save(policy, 'imitation_policy.pt')
print('Saved imitation_policy.pt')
