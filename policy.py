from abc import ABC, abstractmethod
import numpy as np
from logger_config import logger
import random
from utils import online_mean_update


class Policy(ABC):
    @abstractmethod
    def select_action(self):
        pass

    @abstractmethod
    def update_action_value(self):
        pass


class EpsilonGreedy(Policy):
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
            raise NotImplementedError(f"Unknown update function: {self.update_fn}")
        logger.debug("Updated action value for action %d: %f", action, self.Q[action])