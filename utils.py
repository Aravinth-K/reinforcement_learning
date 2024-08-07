import json
import matplotlib.pyplot as plt
import pandas as pd
import os
from logger_config import logger


def plot_results(data_dict, prefix="", dpi=500):
    colors = ["green", "red", "blue"]

    # Plot Average Reward Over Time
    plt.figure(figsize=(12, 5), dpi=dpi)
    for i, (key, (avg_rewards, avg_optimal_actions)) in enumerate(data_dict.items()):
        plt.plot(avg_rewards, color=colors[i % len(colors)], label=key)
    plt.xlabel("Steps")
    plt.ylabel("Average Reward")
    plt.title("Average Reward Over Time")
    plt.legend()
    avg_reward_filename = f"{prefix}_average_reward_over_time.png"
    plt.savefig(avg_reward_filename)
    logger.info("Saved plot: %s", avg_reward_filename)

    # Plot % Optimal Action Over Time
    plt.figure(figsize=(12, 5), dpi=dpi)
    for i, (key, (avg_rewards, avg_optimal_actions)) in enumerate(data_dict.items()):
        plt.plot(avg_optimal_actions * 100, color=colors[i % len(colors)], label=key)
    plt.ylim(0, 100)
    plt.xlabel("Steps")
    plt.ylabel("% Optimal Action")
    plt.title("% Optimal Action Over Time")
    plt.legend()
    optimal_action_filename = f"{prefix}_optimal_action_over_time.png"
    plt.savefig(optimal_action_filename)
    logger.info("Saved plot: %s", optimal_action_filename)


def save_metrics(metrics_dict, filename, dir=None):
    data = {}
    for policy_name, (rewards, optimal_actions) in metrics_dict.items():
        data[f"{policy_name}_rewards"] = rewards
        data[f"{policy_name}_optimal_actions"] = optimal_actions
    df = pd.DataFrame(data)

    if dir and not os.path.exists(dir):
        os.makedirs(dir)

    full_path = os.path.join(dir, filename) if dir else filename
    df.to_csv(full_path, index=False)
    logger.info("Saved metrics to %s", full_path)


def read_config(file_path):
    with open(file_path, "r") as file:
        config = json.load(file)
    logger.info("Read configuration from %s", file_path)
    return config


def get_config_prefix(file):
    return os.path.splitext(os.path.basename(file))[0]


def online_mean_update(old_estimate, target, step_size):
    return old_estimate + (target - old_estimate) * step_size
