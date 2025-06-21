import torch
import torch.nn as nn
import torch.optim as optim

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim=3):
        super().__init__()
        self.fc = nn.Linear(state_dim, 128)
        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)

    def forward(self, state):
        x = torch.tanh(self.fc(state))
        return torch.softmax(self.actor(x), dim=-1), self.critic(x)

def train_ac(model, env, episodes=200, gamma=0.99):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(episodes):
        state = env.reset()
        done = False
        log_probs, values, rewards = [], [], []

        while not done:
            state_t = torch.tensor(state, dtype=torch.float32)
            probs, value = model(state_t)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            next_state, reward, done = env.step(action.item())

            log_probs.append(dist.log_prob(action))
            values.append(value)
            rewards.append(reward)
            state = next_state

        R = 0
        returns = []
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        values = torch.cat(values).squeeze()

        actor_loss = 0
        critic_loss = 0
        for log_prob, value, R in zip(log_probs, values, returns):
            advantage = R - value.item()
            actor_loss += -log_prob * advantage
            critic_loss += (advantage ** 2)

        loss = actor_loss + critic_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if ep % 10 == 0:
            print(f"Episode {ep} — loss {loss.item():.4f}")
