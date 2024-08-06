import numpy as np
import random
import matplotlib.pyplot as plt
import json
import copy


class RewardDistribution:
    def __init__(
            self, 
            k, 
            sampling_variance, 
            is_nonstationary=None, 
            dist_mean_drift_fn=None,
            **kwargs,
        ):
        self.k = k
        self.initial_means = np.random.normal(size=k)
        self.means = np.copy(self.initial_means)
        self.variance = sampling_variance
        if is_nonstationary:
            self.update_fn = dist_mean_drift_fn
            self.random_state = np.random.RandomState(seed=42)

    def generate_rewards(self):
        return np.random.normal(self.means, np.sqrt(self.variance))

    def update_means(self, **kwargs):
        if self.update_fn=="standard_normal_increments":
            self.means += self.random_state.normal(size=self.k, **kwargs)
        else:
            raise NotImplementedError   


class EpsilonGreedyPolicy:
    def __init__(
            self, 
            k, 
            epsilon, 
            initial_action_values, 
            action_update_fn
        ):
        self.k = k
        if not (0 <= epsilon <= 1):
            raise ValueError("Epsilon must be between 0 and 1")
        else:
            self.epsilon = epsilon
        self.Q = np.ones(k) * initial_action_values
        self.N = np.zeros(k)
        self.update_fn = action_update_fn

    def select_action(self):
        if random.random() < self.epsilon:
            return np.random.randint(self.k)
        else:
            max_indices = np.where(self.Q == self.Q.max())[0]
            random_max_action = np.random.choice(max_indices)
            return random_max_action

    def update_action_value(self, action, reward, **kwargs):
        self.N[action] += 1
        if self.update_fn=="sample_average":
            self.Q[action] += online_mean_update(
                old_estimate=self.Q[action],
                target=reward,
                step_size=1.0/self.N[action],
            )
        elif self.update_fn=="constant_step_size":
            self.Q[action] += online_mean_update(
                old_estimate=self.Q[action],
                target=reward,
                **kwargs
            )
        else:
            raise NotImplementedError


class Metrics:
    def __init__(self, num_runs):
        self.rewards = np.zeros(num_runs)
        self.optimal_actions = np.zeros(num_runs)
        self.run = 0
    
    def update(self, rewards, optimal_actions):
        self.rewards += online_mean_update(
            old_estimate=self.rewards,
            target=rewards,
            step_size=1.0/(self.run + 1)
        )
        self.optimal_actions += online_mean_update(
            old_estimate=self.optimal_actions,
            target=optimal_actions,
            step_size=1.0/(self.run + 1)
        )
        self.run += 1


def read_config(file_path):
    with open(file_path, 'r') as file:
        config = json.load(file)
    return config


def online_mean_update(old_estimate, target, step_size):
    return (target - old_estimate) * step_size


def run_bandits(num_steps, k, reward_distribution, **kwargs):
    policy = EpsilonGreedyPolicy(k, **kwargs["policy"])
    rewards = np.zeros(num_steps)
    optimal_actions = np.zeros(num_steps)

    for step in range(num_steps):
        optimal_action = np.argmax(reward_distribution.means)
        action = policy.select_action()
        reward = reward_distribution.generate_rewards()[action]
        policy.update_action_value(action, reward)
        # print("Reward means:", reward_distribution.means)
        # print("Action value", policy.Q)
        # print("\n")
        rewards[step] = reward
        optimal_actions[step] = (action == optimal_action)
        if kwargs["reward_dist"]["is_nonstationary"]:
            nonstationary_kwargs = kwargs["reward_dist"]["nonstationary_kwargs"]
            reward_distribution.update_means(**nonstationary_kwargs)
    
    return rewards, optimal_actions


def run_simulation(config):
    k = config.pop('k')
    num_steps = config.pop('num_steps')
    num_runs = config.pop('num_runs')
    epsilon = config["policy"].pop("epsilon")
    
    if not isinstance(epsilon, list):
        epsilon = [epsilon]

    reward_distribution = RewardDistribution(k, **config["reward_dist"])
    results = {}

    for eps in epsilon:
        config["policy"]["epsilon"] = eps
        metrics = Metrics(num_steps)
        reward_distribution_initial = copy.deepcopy(reward_distribution)
        
        for run in range(num_runs):
            rewards, optimal_actions = run_bandits(
                num_steps, k, reward_distribution, **config
            )
            metrics.update(rewards, optimal_actions)
            reward_distribution = copy.deepcopy(reward_distribution_initial)
        
        results[eps] = [metrics.rewards, metrics.optimal_actions, reward_distribution.means]
    
    return results


def plot_results(data_dict, dpi=500):
    colors = ['green', 'red', 'blue']
    
    # Plot Average Reward Over Time
    plt.figure(figsize=(12, 5), dpi=dpi)
    for i, (key, (avg_rewards, avg_optimal_actions)) in enumerate(data_dict.items()):
        plt.plot(avg_rewards, color=colors[i % len(colors)], label=key)
    plt.xlabel('Steps')
    plt.ylabel('Average Reward')
    plt.title('Average Reward Over Time')
    plt.legend()
    avg_reward_filename = 'average_reward_over_time.png'
    plt.savefig(avg_reward_filename)

    # Plot % Optimal Action Over Time
    plt.figure(figsize=(12, 5), dpi=dpi)
    for i, (key, (avg_rewards, avg_optimal_actions)) in enumerate(data_dict.items()):
        plt.plot(avg_optimal_actions * 100, color=colors[i % len(colors)], label=key)
    plt.ylim(0,100)
    plt.xlabel('Steps')
    plt.ylabel('% Optimal Action')
    plt.title('% Optimal Action Over Time')
    plt.legend()
    optimal_action_filename = 'optimal_action_over_time.png'
    plt.savefig(optimal_action_filename)


if __name__ == '__main__':
    config = read_config('config.json')
    results = run_simulation(config)
    metrics = {k: v[:2] for k, v in results.items()}
    dist_means = {k: v[2] for k, v in results.items()}
    plot_results(metrics)
