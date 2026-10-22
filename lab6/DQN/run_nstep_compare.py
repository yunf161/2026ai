import random
import numpy as np
from argparse import Namespace
from agent_dir.agent_dqn import AgentDQN
import gym


def run_one(args, seed):
    random.seed(seed)
    np.random.seed(seed)

    env = gym.make(args.env_name)
    agent = AgentDQN(env, args)
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
        reward_history.append(episode_reward)

        if episode_reward >= 200 and first_200_step is None:
            first_200_step = total_steps

        if len(reward_history) >= 10 and stable_200_step is None:
            recent_avg = np.mean(reward_history[-10:])
            if recent_avg >= stable_threshold:
                stable_200_step = total_steps

    env.close()
    return first_200_step, stable_200_step


def run_compare():
    seeds = [11037, 22037, 33037]
    n_steps = [2, 3, 4]
    results = {}

    for n_step in n_steps:
        print(f"=== n_step = {n_step} ===")
        args = Namespace(
            env_name='CartPole-v0',
            seed=11037,
            hidden_size=64,
            lr=0.0012,
            gamma=0.99,
            n_step=n_step,
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
        metrics = []
        for i, seed in enumerate(seeds, 1):
            print(f'Run {i} seed {seed}')
            first, stable = run_one(args, seed)
            metrics.append((first, stable))
            print(f'  first 200 step: {first}, stable 200 step: {stable}')

        avg_first = np.mean([m[0] for m in metrics if m[0] is not None])
        avg_stable = np.mean([m[1] for m in metrics if m[1] is not None])
        print(f'  average first 200 step: {avg_first}')
        print(f'  average stable 200 step: {avg_stable}')
        results[n_step] = (metrics, avg_first, avg_stable)
        print()

    return results


if __name__ == '__main__':
    run_compare()
