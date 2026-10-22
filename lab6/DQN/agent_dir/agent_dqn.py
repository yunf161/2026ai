import os
import random
import copy
import numpy as np
import torch
from pathlib import Path
from tensorboardX import SummaryWriter
from torch import nn, optim
from agent_dir.agent import Agent
from collections import deque


 
class QNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, inputs):
        x = torch.relu(self.fc1(inputs))
        x = self.fc2(x)
        return x
    



class ReplayBuffer:
    def __init__(self, buffer_size, n_step, gamma):
        self.buffer = deque(maxlen=buffer_size)
        self.n_step = n_step
        self.gamma = gamma
        self.n_step_buffer = deque(maxlen=n_step)

    def __len__(self):
        return len(self.buffer)

    def _get_n_step_info(self):
        reward, next_state, done = 0.0, None, False
        for idx, transition in enumerate(self.n_step_buffer):
            _, _, r, next_s, d = transition
            reward += (self.gamma ** idx) * r
            next_state = next_s
            done = d
            if done:
                break
        return reward, next_state, done

    def push(self, state, action, reward, next_state, done):
        self.n_step_buffer.append((state, action, reward, next_state, done))
        if len(self.n_step_buffer) < self.n_step:
            if done:
                while self.n_step_buffer:
                    n_reward, n_next_state, n_done = self._get_n_step_info()
                    s, a, _, _, _ = self.n_step_buffer[0]
                    self.buffer.append((s, a, n_reward, n_next_state, n_done))
                    self.n_step_buffer.popleft()
            return

        n_reward, n_next_state, n_done = self._get_n_step_info()
        s, a, _, _, _ = self.n_step_buffer[0]
        self.buffer.append((s, a, n_reward, n_next_state, n_done))

        if done:
            self.n_step_buffer.popleft()
            while self.n_step_buffer:
                n_reward, n_next_state, n_done = self._get_n_step_info()
                s, a, _, _, _ = self.n_step_buffer[0]
                self.buffer.append((s, a, n_reward, n_next_state, n_done))
                self.n_step_buffer.popleft()

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        return transitions

    def clean(self):
        self.buffer.clear()


class AgentDQN(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """
        super(AgentDQN, self).__init__(env)
        self.args = args
        self.env = env
        self.q_network = QNetwork(env.observation_space.shape[0], args.hidden_size, env.action_space.n)
        self.replay_buffer = ReplayBuffer(args.buffer_size, args.n_step, args.gamma)
        self.q_network_target = copy.deepcopy(self.q_network)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=args.lr)
        self.n_step = args.n_step
        self.epsilon_start = args.epsilon_start
        self.epsilon = args.epsilon_start
        self.epsilon_end = args.epsilon_end
        self.epsilon_decay_steps = args.epsilon_decay_steps
        self.epsilon_accelerate_step = args.epsilon_accelerate_step
        self.gamma = args.gamma
        self.grad_norm_clip = args.grad_norm_clip
        self.device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
        self.q_network.to(self.device)
        self.q_network_target.to(self.device)
        self.batch_size = args.batch_size
        self.target_update_freq = args.target_update_freq
        self.train_every = args.train_every
        self.warmup_steps = args.warmup_steps
        self.lr_decay_rate_slow = args.lr_decay_rate_slow
        self.lr_decay_rate_fast = args.lr_decay_rate_fast
        self.lr_accelerate_step = args.lr_accelerate_step
        self.lr_decay_every = args.lr_decay_every

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    
    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        ##################
        # YOUR CODE HERE #
        ##################
        pass

    def train(self):
        """
        Implement your training algorithm here
        """
        transitions = self.replay_buffer.sample(self.batch_size)
        batch=list(zip(*transitions))
        states = torch.FloatTensor(np.array(batch[0])).to(self.device)
        actions = torch.LongTensor(batch[1]).to(self.device)
        rewards = torch.FloatTensor(batch[2]).to(self.device)
        next_states = torch.FloatTensor(np.array(batch[3])).to(self.device)
        dones = torch.FloatTensor(batch[4]).to(self.device)
        Q_values = self.q_network(states)
        Q_values = Q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Double DQN: Q-Network selects action, Target-Network evaluates it
            best_actions = self.q_network(next_states).argmax(1, keepdim=True)
            next_Q_values = self.q_network_target(next_states).gather(1, best_actions).squeeze(1)
            q_target = rewards + (self.gamma ** self.n_step) * next_Q_values * (1 - dones)

        q_loss = nn.MSELoss()(Q_values, q_target)
        self.optimizer.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.grad_norm_clip)
        self.optimizer.step()

        return q_loss.item()
       

    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent
        Input:observation
        Return:action
        """
         # 测试模式 → 纯贪心；训练模式 → ε-greedy
        if test or random.random() > self.epsilon:
            state = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state)
            action = q_values.argmax().item()
        else:
            action = self.env.action_space.sample()  # 随机动作
        return action

    def run(self):
        """
        Implement the interaction between agent and environment here
        """
        total_steps = 0
        episode = 0
        while total_steps < self.args.n_frames:
            state = self.env.reset()
            total_reward = 0 
            done = False
            while not done:
                action = self.make_action(state, test=False)
                next_state, reward, done, _ = self.env.step(action)
                self.replay_buffer.push(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                total_steps += 1
                # 线性 ε 衰减
                if total_steps <= self.epsilon_accelerate_step:
                    frac = total_steps / self.epsilon_accelerate_step
                    self.epsilon = self.epsilon_start - (self.epsilon_start - self.epsilon_end) * 0.5 * frac
                else:
                    frac = (total_steps - self.epsilon_accelerate_step) / max(1, self.epsilon_decay_steps - self.epsilon_accelerate_step)
                    self.epsilon = self.epsilon_start - (self.epsilon_start - self.epsilon_end) * (0.5 + 0.5 * min(1.0, frac))
                self.epsilon = max(self.epsilon_end, self.epsilon)

                if len(self.replay_buffer) >= self.batch_size and total_steps > self.warmup_steps and total_steps % self.train_every == 0:
                    self.train()

                # 学习率衰减
                if total_steps % self.lr_decay_every == 0:
                    decay_rate = self.lr_decay_rate_fast if total_steps >= self.lr_accelerate_step else self.lr_decay_rate_slow
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] *= decay_rate

                if total_steps % self.target_update_freq == 0:
                    self.q_network_target.load_state_dict(self.q_network.state_dict())

            episode += 1
            if episode % 10 == 0:
                print(f"Episode {episode}, Steps {total_steps}, Reward {total_reward}, Epsilon {self.epsilon:.3f}")
           
