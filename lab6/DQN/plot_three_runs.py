import os
import random
import numpy as np
import matplotlib.pyplot as plt
from argparse import Namespace
from agent_dir.agent_dqn import AgentDQN
import gym


def run_one(args, seed):
    random.seed(seed)
    np.random.seed(seed)

    env = gym.make(args.env_name)
    agent = AgentDQN(env, args)
    rewards = []
    steps = []
    total_steps = 0
    episode = 0
    first_200_step = None
    stable_200_step = None
    reward_history = []
    stable_threshold = 195

    while total_steps < args.n_frames:
        state = env.reset()
        episode_reward = 0
        done = False
        while not done and total_steps < args.n_frames:
            action = agent.make_action(state, test=False)
            next_state, reward, done, _ = env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            total_steps += 1

            if total_steps <= agent.epsilon_accelerate_step:
                frac = total_steps / agent.epsilon_accelerate_step
                agent.epsilon = agent.epsilon_start - (agent.epsilon_start - agent.epsilon_end) * 0.5 * frac
            else:
                frac = (total_steps - agent.epsilon_accelerate_step) / max(1, agent.epsilon_decay_steps - agent.epsilon_accelerate_step)
                agent.epsilon = agent.epsilon_start - (agent.epsilon_start - agent.epsilon_end) * (0.5 + 0.5 * min(1.0, frac))
            agent.epsilon = max(agent.epsilon_end, agent.epsilon)

            if len(agent.replay_buffer) >= agent.batch_size and total_steps > agent.warmup_steps and total_steps % agent.train_every == 0:
                agent.train()

            if total_steps % agent.lr_decay_every == 0:
                decay_rate = agent.lr_decay_rate_fast if total_steps >= agent.lr_accelerate_step else agent.lr_decay_rate_slow
                for param_group in agent.optimizer.param_groups:
                    param_group['lr'] *= decay_rate

            if total_steps % agent.target_update_freq == 0:
                agent.q_network_target.load_state_dict(agent.q_network.state_dict())
        
        episode += 1
        rewards.append(episode_reward)
        steps.append(total_steps)

        if episode_reward >= 200 and first_200_step is None:
            first_200_step = total_steps

        reward_history.append(episode_reward)
        if len(reward_history) >= 10 and stable_200_step is None:
            recent_avg = np.mean(reward_history[-10:])
            if recent_avg >= stable_threshold:
                stable_200_step = total_steps

    env.close()
    return steps, rewards, first_200_step, stable_200_step


if __name__ == '__main__':
    args = Namespace(
        env_name='CartPole-v0',
        seed=11037,
        hidden_size=64,
        lr=0.0012,
        gamma=0.99,
        n_step=2,
        grad_norm_clip=5.0,
        batch_size=256,
        buffer_size=32000,
        target_update_freq=50,
        epsilon_start=1.0,
        epsilon_end=0.02,
        epsilon_decay_steps=5000,
        epsilon_accelerate_step=2500,
        train_every=1,
        warmup_steps=200,
        lr_decay_rate_slow=0.98,
        lr_decay_rate_fast=0.9,
        lr_accelerate_step=5000,
        lr_decay_every=1000,
        use_cuda=False,
        n_frames=12000,
    )

    seeds = [11037, 22037, 33037]
    results = []
    metrics = []
    for i, seed in enumerate(seeds, 1):
        print(f'Run {i} with seed {seed}')
        steps, rewards, first_200_step, stable_200_step = run_one(args, seed)
        results.append((steps, rewards))
        metrics.append((first_200_step, stable_200_step))
        print(f' Run {i} first reward=200 step: {first_200_step}')
        print(f' Run {i} stable reward=200 step: {stable_200_step}')

    plt.figure(figsize=(10, 6))
    for i, (steps, rewards) in enumerate(results, 1):
        plt.plot(steps, rewards, marker='o', label=f'Run {i} (seed={seeds[i-1]})')

    plt.xlabel('Cumulative Steps')
    plt.ylabel('Episode Reward')
    plt.title('DQN Reward Curve over Steps (3 Runs)')
    plt.legend()
    plt.grid(True)
    out_path = os.path.join(os.path.dirname(__file__), 'reward_curves.png')
    plt.tight_layout()
    plt.savefig(out_path)
    print(f'Saved plot to {out_path}')

    avg_first200 = np.mean([m[0] for m in metrics if m[0] is not None])
    avg_stable200 = np.mean([m[1] for m in metrics if m[1] is not None])
    print('Metrics summary:')
    for i, (first_step, stable_step) in enumerate(metrics, 1):
        print(f'  Run {i}: first reward=200 at step {first_step}, stable 200 at step {stable_step}')
    print(f'  Average first 200 step: {avg_first200}')
    print(f'  Average stable 200 step: {avg_stable200}')
