import numpy as np
import random
import copy
import argparse
from tqdm import tqdm
from logger_config import logger
from utils import *


class RewardDistribution:
    def __init__(
        self,
        k,
        sampling_variance,
        init_mean_fn,
        is_nonstationary=None,
        mean_update_fn=None,
        **kwargs,
    ):
        logger.debug(
            "Initializing RewardDistribution with k=%d, sampling_variance=%f",
            k,
            sampling_variance,
        )
        self.k = k
        self._init_mean(init_mean_fn)
        self.means = np.copy(self.initial_means)
        self.variance = sampling_variance
        if is_nonstationary:
            self.update_fn = mean_update_fn
            self.random_state = np.random.RandomState(seed=42)

    def _init_mean(self, init_mean_fn):
        logger.debug("Initializing means with init_mean_fn=%s", init_mean_fn)
        if init_mean_fn == "equal":
            self.initial_means = np.ones(self.k) * np.random.normal()
        elif init_mean_fn == "standard_normal":
            self.initial_means = np.random.normal(size=self.k)
        else:
            raise NotImplementedError

    def update_means(self, **kwargs):
        if self.update_fn == "standard_normal_increments":
            self.means += self.random_state.normal(size=self.k, **kwargs)
        else:
            raise NotImplementedError

    def generate_rewards(self):
        rewards = np.random.normal(self.means, np.sqrt(self.variance))
        logger.debug("Generated rewards: %s", rewards)
        return rewards


class EpsilonGreedyPolicy:
    def __init__(self, k, epsilon, initial_action_values, action_update_fn, **kwargs):
        logger.debug(
            "Initializing EpsilonGreedyPolicy with k=%d, epsilon=%f", k, epsilon
        )
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
            action = np.random.randint(self.k)
        else:
            max_indices = np.where(self.Q == self.Q.max())[0]
            action = np.random.choice(max_indices)
        logger.debug("Selected action: %d", action)
        return action

    def update_action_value(self, action, reward, **kwargs):
        self.N[action] += 1
        if self.update_fn == "sample_average":
            self.Q[action] = online_mean_update(
                old_estimate=self.Q[action],
                target=reward,
                step_size=1.0 / self.N[action],
            )
        elif self.update_fn == "constant_step_size":
            self.Q[action] = online_mean_update(
                old_estimate=self.Q[action], target=reward, **kwargs
            )
        else:
            raise NotImplementedError
        logger.debug("Updated action value for action %d: %f", action, self.Q[action])


class Metrics:
    def __init__(self, num_steps):
        self.rewards = np.zeros(num_steps)
        self.optimal_actions = np.zeros(num_steps)
        self.run = 0

    def update(self, rewards, optimal_actions):
        self.rewards = online_mean_update(
            old_estimate=self.rewards, target=rewards, step_size=1.0 / (self.run + 1)
        )
        self.optimal_actions = online_mean_update(
            old_estimate=self.optimal_actions,
            target=optimal_actions,
            step_size=1.0 / (self.run + 1),
        )
        self.run += 1
        logger.debug("Updated metrics for run %d", self.run)


def run_bandits(k, num_steps, reward_distribution, policy_kwargs, reward_kwargs):
    policy = EpsilonGreedyPolicy(k, **policy_kwargs)
    rewards = np.zeros(num_steps)
    optimal_actions = np.zeros(num_steps)

    for step in range(num_steps):
        optimal_action = np.argmax(reward_distribution.means)
        action = policy.select_action()
        reward = reward_distribution.generate_rewards()[action]
        policy.update_action_value(action, reward, **policy_kwargs["update_kwargs"])
        rewards[step] = reward
        optimal_actions[step] = action == optimal_action
        if reward_kwargs["is_nonstationary"]:
            reward_distribution.update_means(**reward_kwargs["update_kwargs"])
        logger.debug(
            "Step %d: action=%d, reward=%f, optimal_action=%d",
            step,
            action,
            reward,
            optimal_action,
        )

    return rewards, optimal_actions


def run_simulation(config):
    k = config.pop("k")
    num_steps = config.pop("num_steps")
    num_runs = config.pop("num_runs")
    reward_kwargs = config["reward"]

    # Use same reward distribution for all simulations
    reward_distribution = RewardDistribution(k, **reward_kwargs)
    results = {}

    for policy_name, policy_kwargs in config["policies"].items():
        metrics = Metrics(num_steps)
        reward_distribution_initial = copy.deepcopy(reward_distribution)

        for run in tqdm(range(num_runs), desc=f"Running {policy_name}"):
            rewards, optimal_actions = run_bandits(
                k=k,
                num_steps=num_steps,
                reward_distribution=reward_distribution,
                policy_kwargs=policy_kwargs,
                reward_kwargs=reward_kwargs,
            )
            metrics.update(rewards, optimal_actions)

            # Reset reward distribution for next run
            reward_distribution = copy.deepcopy(reward_distribution_initial)

        results[policy_name] = [metrics.rewards, metrics.optimal_actions]
        logger.info("Completed runs for policy: %s", policy_name)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run bandit simulations.")
    parser.add_argument("config_file", type=str, help="Path to the configuration file.")
    args = parser.parse_args()

    config = read_config(args.config_file)
    config_prefix = get_config_prefix(args.config_file)
    results = run_simulation(config)
    metrics = {k: v[:2] for k, v in results.items()}
    plot_results(metrics, prefix=config_prefix)

    save_metrics(
        metrics_dict=metrics,
        filename=f"{config_prefix}_metrics.csv",
        dir="results",
    )
