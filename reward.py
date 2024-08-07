from logger_config import logger
import numpy as np

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