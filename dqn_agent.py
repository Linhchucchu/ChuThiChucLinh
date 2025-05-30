import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import cv2
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    def __init__(self, input_channels, n_actions):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),  # (4,84,84) -> (32,20,20)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),             # -> (64,9,9)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),             # -> (64,7,7)
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*7*7, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states, dtype=np.float32),
                np.array(actions, dtype=np.int64),
                np.array(rewards, dtype=np.float32),
                np.array(next_states, dtype=np.float32),
                np.array(dones, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)

def preprocess_state(state):
    gray = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    normalized = resized / 255.0
    return normalized.astype(np.float32)

def stack_frames(frames, stack_size=4):
    return np.stack(frames[-stack_size:], axis=0)

class DQNAgent:
    def __init__(self, n_actions, replay_buffer):
        self.n_actions = n_actions
        self.replay_buffer = replay_buffer

        self.policy_net = DQN(4, n_actions).to(device)
        self.target_net = DQN(4, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-4)
        self.criteria = nn.MSELoss()

        self.batch_size = 32
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.update_target_every = 1000
        self.learn_step_counter = 0

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        else:
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = self.policy_net(state_t)
            return torch.argmax(q_values).item()

    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states_t = torch.tensor(states, dtype=torch.float32).to(device)
        actions_t = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32).to(device)
        next_states_t = torch.tensor(next_states, dtype=torch.float32).to(device)
        dones_t = torch.tensor(dones, dtype=torch.float32).to(device)

        q_pred = self.policy_net(states_t).gather(1, actions_t).squeeze(1)

        with torch.no_grad():
            q_next = self.target_net(next_states_t).max(1)[0]

        q_target = rewards_t + self.gamma * q_next * (1 - dones_t)

        loss = self.criteria(q_pred, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.learn_step_counter += 1
        if self.learn_step_counter % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)

        return loss.item()

def train_dqn(env, num_episodes=500, max_steps=200):
    replay_buffer = ReplayBuffer(10000)
    agent = DQNAgent(env.get_action_space(), replay_buffer)

    scores = []
    losses = []
    epsilons = []
    reward_window = deque(maxlen=10)

    for ep in range(num_episodes):
        state = env.reset()
        processed = preprocess_state(state)
        frames = [processed for _ in range(4)]
        state_stack = stack_frames(frames)

        total_reward = 0
        done = False
        step_count = 0

        while not done:
            action = agent.select_action(state_stack)
            next_state, reward, done, _ = env.step(action)

            processed_next = preprocess_state(next_state)
            frames.append(processed_next)
            frames.pop(0)
            next_state_stack = stack_frames(frames)

            replay_buffer.push(state_stack, action, reward, next_state_stack, done)
            loss = agent.learn()
            if loss is not None:
                losses.append(loss)

            state_stack = next_state_stack
            total_reward += reward
            step_count += 1

            if step_count >= max_steps:
                break

        scores.append(total_reward)
        reward_window.append(total_reward)
        avg_reward = np.mean(reward_window)
        epsilons.append(agent.epsilon)

        print(f"Episode {ep+1}/{num_episodes} - Reward: {total_reward:.2f} - Avg Reward (last 10): {avg_reward:.2f} - Epsilon: {agent.epsilon:.3f}")

    return scores, losses, epsilons

def plot_metrics(scores, losses, epsilons, window_size=10, save_path="dqn_training_result.png"):
    rewards_moving_avg = pd.Series(scores).rolling(window=window_size).mean()

    plt.figure(figsize=(18,5))

    plt.subplot(1,4,1)
    plt.title("Total Reward per Episode")
    plt.plot(scores, label="Reward per episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()

    plt.subplot(1,4,2)
    plt.title(f"Moving Average Reward (window={window_size})")
    plt.plot(rewards_moving_avg, color='orange', label='Moving Avg Reward')
    plt.xlabel("Episode")
    plt.ylabel("Avg Reward")
    plt.legend()

    plt.subplot(1,4,3)
    plt.title("Loss during Training")
    plt.plot(losses)
    plt.xlabel("Training Step")
    plt.ylabel("Loss")

    plt.subplot(1,4,4)
    plt.title("Epsilon Decay")
    plt.plot(epsilons, color='green')
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    set_seed(42)
    from sokoban import Sokoban

    env = Sokoban(400, 400, 9, 10, mode="train")
    scores, losses, epsilons = train_dqn(env, num_episodes=200, max_steps=200)
    plot_metrics(scores, losses, epsilons, window_size=10, save_path="dqn_training_result.png")
    env.close()
