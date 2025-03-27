import numpy as np
import scipy
from retry import retry


class DistributionOutOfRangeException(ValueError):
    def __init__(self, message: str):
        self.message = message

    def __str__(self):
        return self.message


class Distribution:
    def __init__(
        self,
        lower_limit: float = -float("inf"),
        upper_limit: float = float("inf"),
        params_corrected: bool = True,
    ):
        """
        create a Distribution

        Args:
            lower_limit: the lower limit of generated number. Defaults to negative infinity.
            upper_limit: the upper limit of generated number. Defaults to positive infinity.
            params_corrected: A boolean value indicating whether the parameters have been corrected or not.
                              Defaults to False.
        """
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        if lower_limit > upper_limit:
            raise ValueError("upper_limit must be larger than lower_limit")
        if params_corrected:
            self.correct_params()
        else:
            self.check_range()

    def out_of_range(self) -> bool:
        """
        Returns: true if the probability distribution can generate values in the given range, otherwise false.
        """
        return True

    def correct_params(self):
        assert False

    def check_range(self):
        """
        Check if the probability distribution can generate values in the given range.
        """
        if self.out_of_range():
            raise DistributionOutOfRangeException(
                "the probability distribution can generate values in the given range"
            )

    def basely_generate(self):
        assert False

    def restrictively_generate(self) -> float:
        res = self.basely_generate()
        while res < self.lower_limit or res > self.upper_limit:
            res = self.basely_generate()
        return res


class GaussianDistribution(Distribution):
    def __init__(
        self,
        mu,
        sigma,
        lower_limit: float = -float("inf"),
        upper_limit: float = float("inf"),
        params_corrected: bool = True,
    ):
        self.mu = mu
        self.sigma = sigma
        Distribution.__init__(self, lower_limit, upper_limit, params_corrected)

    def out_of_range(self) -> bool:
        """
        Checks if the integral of the normal distribution within the given limits is less than 0.5.

        Returns:
        bool: True if the integral is less than 0.5, False otherwise.
        """
        result = scipy.stats.norm.cdf(
            self.upper_limit, loc=self.mu, scale=self.sigma
        ) - scipy.stats.norm.cdf(self.lower_limit, loc=self.mu, scale=self.sigma)
        if result < 0.5:
            return True
        return False

    def correct_params(self):
        mu_hat = np.mean([self.lower_limit, self.upper_limit])
        if self.mu < mu_hat:
            self.mu = mu_hat - (self.upper_limit - self.lower_limit) * 0.5 / (
                1 + np.exp((self.mu - mu_hat) / 10)
            )
        else:
            self.mu = mu_hat + (self.upper_limit - self.lower_limit) * 0.5 / (
                1 + np.exp((mu_hat - self.mu) / 10)
            )

    def basely_generate(self):
        return np.random.normal(self.mu, self.sigma)


class UniformDistribution(Distribution):
    def __init__(
        self,
        low,
        high,
        lower_limit: float = -float("inf"),
        upper_limit: float = float("inf"),
        params_corrected: bool = True,
    ):
        self.low = low
        self.high = high
        Distribution.__init__(self, lower_limit, upper_limit, params_corrected)

    def out_of_range(self) -> bool:
        """
        Checks if the limits are out of range or if the range is too small.

        Returns:
        bool: True if the limits are out of range or the range is too small, False otherwise.
        """
        if (
            self.upper_limit <= self.low
            or self.lower_limit >= self.high
            or (self.high - self.lower_limit < 0.5 * (self.high - self.low))
        ):
            return True
        return False

    def correct_params(self):
        self.low = self.lower_limit
        self.high = self.upper_limit

    def basely_generate(self):
        return np.random.uniform(self.low, self.high)


# class ExponentialDistribution(Distribution):
#     def __init__(self, scale: float, lower_limit: float = -float('inf'), upper_limit: float = float('inf')):
#         Distribution.__init__(self, lower_limit, upper_limit)
#         self.scale = scale
#
#     def out_of_range(self) -> bool:
#         """
#         Define "in range" to mean :math:`0.05 < \int^{t}_{0} e^{-x}dx < 0.95`
#         """
#         a = self.lower_limit if self.lower_limit > 0. else 0.
#         b = self.upper_limit
#         result, error = scipy.integrate.quad(lambda x: 1. / self.scale * np.exp(-(x / self.scale)), a, b)
#         if result < 0.5:
#             return True
#         return False
#
#     def basely_generate(self):
#         return np.random.exponential(self.scale)


class ConstantDistribution(Distribution):
    def __init__(
        self,
        c: float,
        lower_limit: float = -float("inf"),
        upper_limit: float = float("inf"),
        params_corrected: bool = True,
    ):
        assert c >= 0
        self.c = c
        Distribution.__init__(self, lower_limit, upper_limit, params_corrected)

    def out_of_range(self) -> bool:
        if self.lower_limit > self.c or self.upper_limit < self.c:
            return True
        return False

    def correct_params(self):
        self.c = np.random.uniform(self.lower_limit, self.upper_limit)

    def basely_generate(self):
        return self.c


class MetaDistribution:
    def __init__(
        self,
        lower_limit: float = -float("inf"),
        upper_limit: float = float("inf"),
        params_corrected: bool = True,
    ):
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self.params_corrected = params_corrected

    def basely_generate(self) -> Distribution:
        assert False

    @retry(DistributionOutOfRangeException)
    def generate(self) -> Distribution:
        # print("retry...")
        return self.basely_generate()


class MetaDistribution4Gaussian(MetaDistribution):
    def __init__(
        self,
        mu_low,
        mu_high,
        sigma_low,
        sigma_high,
        lower_limit: float = -float("inf"),
        upper_limit: float = float("inf"),
        params_corrected: bool = True,
    ):
        MetaDistribution.__init__(self, lower_limit, upper_limit, params_corrected)
        self.mu_low = mu_low
        self.mu_high = mu_high
        self.sigma_low = sigma_low
        self.sigma_high = sigma_high
        self.mu = None
        self.sigma = None

    def basely_generate(self) -> Distribution:
        self.mu = np.random.uniform(self.mu_low, self.mu_high)
        self.sigma = np.random.uniform(self.sigma_low, self.sigma_high)
        return GaussianDistribution(
            self.mu,
            self.sigma,
            self.lower_limit,
            self.upper_limit,
            self.params_corrected,
        )


class MetaDistribution4Uniform(MetaDistribution):
    def __init__(
        self,
        low_lower_limit,
        low_upper_limit,
        high_lower_limit,
        high_upper_limit,
        lower_limit: float = -float("inf"),
        upper_limit: float = float("inf"),
        params_corrected: bool = True,
    ):
        MetaDistribution.__init__(self, lower_limit, upper_limit, params_corrected)
        self.low_lower_limit = low_lower_limit
        self.low_upper_limit = low_upper_limit
        self.high_lower_limit = high_lower_limit
        self.high_upper_limit = high_upper_limit
        self.low = None
        self.high = None

    def basely_generate(self) -> Distribution:
        self.low = np.random.uniform(self.low_lower_limit, self.low_upper_limit)
        self.high = np.random.uniform(self.high_lower_limit, self.high_upper_limit)
        return UniformDistribution(
            self.low,
            self.high,
            self.lower_limit,
            self.upper_limit,
            self.params_corrected,
        )


# class MetaDistribution4Exponential(MetaDistribution):
#     def __init__(self, scale_low, scale_high, lower_limit: float = -float('inf'), upper_limit: float = float('inf')):
#         MetaDistribution.__init__(self, lower_limit, upper_limit)
#         self.scale_low = scale_low
#         self.scale_high = scale_high
#         self.scale = None
#
#     def basely_generate(self) -> Distribution:
#         self.scale = np.random.uniform(self.scale_low, self.scale_high)
#         return ExponentialDistribution(self.scale, self.lower_limit, self.upper_limit)


class MetaDistribution4Constant(MetaDistribution):
    def __init__(
        self,
        constant_low,
        constant_high,
        lower_limit: float = -float("inf"),
        upper_limit: float = float("inf"),
        params_corrected: bool = True,
    ):
        MetaDistribution.__init__(self, lower_limit, upper_limit, params_corrected)
        self.constant_low = constant_low
        self.constant_high = constant_high
        self.constant = None

    def basely_generate(self) -> Distribution:
        self.constant = np.random.uniform(self.constant_low, self.constant_high)
        return ConstantDistribution(
            self.constant, self.lower_limit, self.upper_limit, self.params_corrected
        )
