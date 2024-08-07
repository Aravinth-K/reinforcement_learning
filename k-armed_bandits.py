import numpy as np
import random
import copy
import argparse
from tqdm import tqdm
from logger_config import logger
from utils import *
from reward import RewardDistribution
from policy import EpsilonGreedy


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
    policy = EpsilonGreedy(k, **policy_kwargs)
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

        results[policy_name] = [
            metrics.rewards, 
            metrics.optimal_actions,
            metrics.rewards[-num_steps//2:].mean()
        ]
        logger.info("Completed runs for policy: %s", policy_name)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run bandit simulations.")
    parser.add_argument("config_file", type=str, help="Path to the configuration file.")
    args = parser.parse_args()

    config = read_config(args.config_file)
    config_prefix = get_config_prefix(args.config_file)
    metrics = run_simulation(config)
    plot_results(
        {k: v[:2] for k, v in metrics.items()}, 
        prefix=config_prefix
    )
    print("Avg. reward for final 50% of steps: \n")
    for k, v in metrics.items():
        print(f"Policy {k}: ", v[-1])
    save_metrics(
        metrics_dict={k: v[:2] for k, v in metrics.items()},
        filename=f"{config_prefix}_metrics.csv",
        dir="results",
    )
