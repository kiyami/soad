import numpy as np
import matplotlib.pyplot as plt
from scripts import generator


class VariableGaussian:
    def __init__(self, mu=10.0, sigma_n=1.0, sigma_p=1.5, N=10000):
        """
        :param mu: Mode of the distribution, the most probable value
        :param sigma_n: Pegative sigma
        :param sigma_p: Positive sigma
        :param N: Sample size
        """
        self.mu = mu
        self.sigma_n = sigma_n
        self.sigma_p = sigma_p
        self.N = int(N)

        """
        x_limits: Limit values of x-axis for plotting or integrating.
        x_values: Array of equally seperated x values. (Array length = N)
        norm: Normalisation constant for the PDF. It is being updated in 'integrate' method.
        pdf_values: Probability density values of the x_values.
        """
        self.x_limits = [self.mu - 5.0*self.sigma_n, self.mu + 5.0*sigma_p]
        self.x_values = np.linspace(self.x_limits[0], self.x_limits[1], self.N)

        self.norm = 1.0

        self.pdf_values = np.asarray(self.pdf(self.x_values))
        self.integrate()

        self.data = np.asarray([])

    @classmethod
    def new(cls, mu=10.0, sigma_n=1.0, sigma_p=1.5, N=10000):
        return cls(mu, sigma_n, sigma_p, N)

    def pdf(self, x):
        A = self.norm / (2.0 * np.pi) ** 0.5
        B = (2.0 * self.sigma_p * self.sigma_n) / (self.sigma_p + self.sigma_n)
        C = (self.sigma_p - self.sigma_n) / (self.sigma_p + self.sigma_n)
        D = (self.mu - x) / (B + (C * (x - self.mu)))
        value = A * np.exp(-D ** 2.0)
        return value

    """
    def integrate(self):
        x_values = np.linspace(self.x_limits[0], self.x_limits[1], self.N, dtype=float)
        pdf_values = np.asarray(self.pdf(self.x_values))
        c = (self.x_limits[1] - self.x_limits[0]) / float(self.N)
        sum = np.sum(c * pdf_values)
        norm = 1.0 / sum
        return sum
    """
    def integrate(self):
        delta_x = self.x_limits[1] - self.x_limits[0]
        c = delta_x / (self.N - 1)
        x_values = np.linspace(self.x_limits[0], self.x_limits[1], self.N, dtype=float)
        area = 0.0
        cdf_values = np.asarray([])
        for i in range(self.N):
            area += self.pdf(x_values[i]) * c
            cdf_values = np.append(cdf_values, area)
        norm = 1 / area
        self.norm = norm
        self.cdf_values = cdf_values * norm
        return area

    def plot_pdf(self):
        plt.plot(self.x_values, self.pdf_values)
        plt.show()

    # bozuk
    def sample_data(self):
        generator.uniform_sampling(self)

    def plot_data(self, bin=50):
        plt.hist(self.data, bins=bin, density=True)
        plt.show()

    def plot_data_and_pdf(self, bin=50):
        plt.hist(self.data, bins=bin, density=True)
        plt.plot(self.x_values, self.pdf_values)
        plt.show()

    def plot_pdf_cdf(self):
        plt.plot(self.x_values, self.cdf_values)
        plt.plot(self.x_values, self.pdf_values)
        plt.show()