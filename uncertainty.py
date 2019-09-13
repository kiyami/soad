import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
from scipy.optimize import curve_fit
import random
import sys


class AsymmetricData:
    def __init__(self, mu=10.0, sigma_n=1.0, sigma_p=1.0, N=10000, creation_type='by_constructor', data=[]):
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
        self.confidence = 1.0  # sigma
        self.creation_type = creation_type

        if not any(data):
            self.data = np.asarray([])
        else:
            self.data = np.asarray(data)
            #self.creation_type = 'by_operation'

        self.bin_value = 50

        if str(self.creation_type) == 'by_constructor':
            self.x_limits = [self.mu - 5.0*self.sigma_n, self.mu + 5.0*sigma_p]
            self.x_values = np.linspace(self.x_limits[0], self.x_limits[1], self.N)

            self.norm = 1.0
            self.norm = self.calculate_norm()

            self.pdf_values = np.asarray(self.pdf(self.x_values))
            self.cdf_values = self.calculate_cdf_values()
            self.cdf = self.calculate_cdf()
            self.inverse_cdf = self.calculate_inverse_cdf()

            self.log_likelihood_values = self.log_likelihood(self.x_values)

            self.generate()

        elif str(self.creation_type) == 'by_operation':
            self.N = self.data.size

            self.fit()
            #self.sigma_n, self.sigma_p = self.estimate()

            self.x_limits = [self.mu - 5.0*self.sigma_n, self.mu + 5.0*self.sigma_p]
            self.x_values = np.linspace(self.x_limits[0], self.x_limits[1], self.N)

            self.norm = 1.0
            self.norm = self.calculate_norm()

            self.pdf_values = np.asarray(self.pdf(self.x_values))
            self.cdf_values = self.calculate_cdf_values()
            self.cdf = self.calculate_cdf()
            self.inverse_cdf = self.calculate_inverse_cdf()

    def __str__(self):
        output = "Value = {:.2e} ( - {:.2e} , + {:.2e} )\n({:.1f} sigma confidence interval)"
        return output.format(self.mu, self.sigma_n, self.sigma_p, self.confidence)

    @classmethod
    def new(cls, mu=10.0, sigma_n=1.0, sigma_p=1.0, N=10000):
        return cls(mu, sigma_n, sigma_p, N)

    def integrate(self):
        delta_x = self.x_limits[1] - self.x_limits[0]
        c = delta_x / (self.N - 1)
        # x_values = np.linspace(self.x_limits[0], self.x_limits[1], self.N, dtype=float)
        area = np.sum(c * self.pdf(self.x_values))
        return area

    def calculate_norm(self):
        area = self.integrate()
        norm = 1/area
        return norm

    def pdf(self, x):
        par_1 = (2.0 * self.sigma_p * self.sigma_n) / (self.sigma_p + self.sigma_n)
        par_2 = (self.sigma_p - self.sigma_n) / (self.sigma_p + self.sigma_n)
        par_3 = (-1.0/2.0) * ((self.mu - x)/(par_1 + par_2*(x - self.mu)))**2.0
        par_4 = self.norm / (2.0 * np.pi)**0.5
        value = par_4 * np.exp(par_3)
        """
        A = self.norm / (2.0 * np.pi) ** 0.5
        B = (2.0 * self.sigma_p * self.sigma_n) / (self.sigma_p + self.sigma_n)
        C = (self.sigma_p - self.sigma_n) / (self.sigma_p + self.sigma_n)
        D = (self.mu - x) / ((B + (C * (x - self.mu)))*(2.0**0.5))
        value = A * np.exp(-D ** 2.0)

        B = (2.0 * self.sigma_p * self.sigma_n) / (self.sigma_p + self.sigma_n)
        C = (self.sigma_p - self.sigma_n) / (self.sigma_p + self.sigma_n)
        D = (self.mu - x) / (B + (C * (x - self.mu)))
        A = self.norm / (2.0 * np.pi * (B + (C * (x - self.mu)))**2.0) ** 0.5
        value = A * np.exp(-D ** 2.0)
        """
        return value

    def log_likelihood(self, x):
        par_1 = (2.0 * self.sigma_p * self.sigma_n) / (self.sigma_p + self.sigma_n)
        par_2 = (self.sigma_p - self.sigma_n) / (self.sigma_p + self.sigma_n)
        value = (-1.0/2.0) * ((self.mu - x)/(par_1 + par_2*(x - self.mu)))**2.0
        """
        A = -1.0 / 2.0
        B = (2.0 * self.sigma_p * self.sigma_n) / (self.sigma_p + self.sigma_n)
        C = (self.sigma_p - self.sigma_n) / (self.sigma_p + self.sigma_n)
        D = (self.mu - x) / (B + C * (x - self.mu))
        value = A * D ** 2.0
        return value
        """
        return value

    def calculate_cdf_values(self):
        delta_x = self.x_limits[1] - self.x_limits[0]
        c = delta_x / (self.N - 1)
        area = 0.0
        cdf_values = np.asarray([])
        for i in range(self.N):
            area += self.pdf_values[i] * c
            cdf_values = np.append(cdf_values, area)
        return cdf_values

    def calculate_cdf(self):
        cdf = interpolate.interp1d(self.x_values, self.cdf_values, kind='nearest')
        return cdf

    def calculate_inverse_cdf(self):
        inverse_cdf = interpolate.interp1d(self.cdf_values, self.x_values, kind='nearest')
        return inverse_cdf

    def generate(self):
        for i in range(self.N):
            rnd_prob = random.uniform(0, 1)
            self.data = np.append(self.data, self.inverse_cdf(rnd_prob))

    @staticmethod
    def fit_func(x, norm, mu, sigma_n, sigma_p):
        par_1 = (2.0 * sigma_p * sigma_n) / (sigma_p + sigma_n)
        par_2 = (sigma_p - sigma_n) / (sigma_p + sigma_n)
        par_3 = (-1.0 / 2.0) * ((mu - x) / (par_1 + par_2 * (x - mu))) ** 2.0
        par_4 = norm / (2.0 * np.pi) ** 0.5
        value = par_4 * np.exp(par_3)
        """
        A = norm / (2.0 * np.pi) ** 0.5
        B = (2.0 * sigma_p * sigma_n) / (sigma_p + sigma_n)
        C = (sigma_p - sigma_n) / (sigma_p + sigma_n)
        D = (mu - x) / ((B + (C * (x - mu)))*(2.0**0.5))
        gau = A * np.exp(-D ** 2.0)
        return gau
        """
        return value

    def fit(self, expected_values=None):
        y, x, _ = plt.hist(self.data, bins=int(self.N/250))
        plt.clf()
        x = (x[1:] + x[:-1]) / 2  # for len(x)==len(y)

        mod = None
        max_y = max(y)
        for i in range(len(y)):
            if y[i] == max_y:
                mod = x[i]

        print("mod", mod)
        #print(len(self.data))
        min_data = min(self.data)
        max_data = max(self.data)
        norm = 1000.0

        if not expected_values:
            expected_values = norm, mod, (mod - min_data) * 0.1, (max_data - mod) * 0.1

        expected = (expected_values[0], expected_values[1], expected_values[2], expected_values[3])
        params, cov = curve_fit(self.fit_func, x, y, expected, method='trf')
        self.norm = params[0]
        self.mu = params[1]
        print("params", params)
        if params[2] > 0.0:
            self.sigma_n = (params[2])
            self.sigma_p = (params[3])
        else:
            self.sigma_n = (params[3])
            self.sigma_p = (params[2])

    def estimate(self, confidence=1.0):
        target_likelihood = -0.5 * float(confidence)
        delta_steps = 1e-5

        current_value = self.mu
        delta = abs(self.mu - self.sigma_p) * delta_steps
        current_likelihood = self.log_likelihood(current_value)

        while abs(current_likelihood) < abs(target_likelihood):
            current_value += delta
            current_likelihood = self.log_likelihood(current_value)
        positive_limit = current_value

        current_value = self.mu
        delta = abs(self.mu - self.sigma_n) * delta_steps
        current_likelihood = self.log_likelihood(current_value)

        while abs(current_likelihood) < abs(target_likelihood):
            current_value -= delta
            current_likelihood = self.log_likelihood(current_value)
        negative_limit = current_value
        print("interval found")
        return [self.mu - negative_limit, positive_limit - self.mu]

    def plot_pdf(self, show=True, save=False):

        plt.clf()
        plt.plot(self.x_values, self.pdf_values, color="blue")

        plt.xlabel("x")
        plt.ylabel("prob")

        if save:
            plt.savefig("plot_pdf.png", dpi=300)

        if show:
            plt.show()

    def plot_log_likelihood(self, show=True, save=False):
        plt.clf()
        plt.plot(self.x_values, self.log_likelihood(self.x_values))
        plt.ylim([-5, 0.5])

        plt.xlabel("x")
        plt.ylabel("ln L")

        plt.axhline(y=-0.5, color="black", ls="--")

        if save:
            plt.savefig("plot_log_likelihood.png", dpi=300)

        if show:
            plt.show()

    def plot_cdf(self, show=True, save=False):
        plt.plot(self.x_values, self.cdf(self.x_values))

        if save:
            plt.savefig("plot_cdf.png", dpi=300)

        if show:
            plt.show()

    def plot_data(self, bins=None, show=True, save=False):
        if not bins:
            bins = self.bin_value

        plt.clf()
        plt.hist(self.data, bins=bins, density=True, color="green", alpha=0.7)

        if save:
            plt.savefig("plot_data.png", dpi=300)

        if show:
            plt.show()

    def plot_data_and_pdf(self, bins=None, show=True, save=False):
        if not bins:
            bins = self.bin_value

        plt.clf()
        plt.hist(self.data, bins=bins, density=True, color="green", alpha=0.6)
        plt.plot(self.x_values, self.pdf_values, color="blue")

        plt.plot([10, 10], [0, 0.4], ls="--", color="black", lw=2)
        plt.plot([9, 9], [0, 0.245], ls="--", color="black", lw=1.2)
        plt.plot([11, 11], [0, 0.245], ls="--", color="black", lw=1.2)
        plt.xticks([5,6,7,8,9,10,11,12,13,14,15])
        plt.xlabel("x")
        plt.ylabel("prob")

        if save:
            plt.savefig("plot_data_and_pdf.png", dpi=300)

        if show:
            plt.show()

    def plot_pdf_cdf(self, show=True, save=False):
        plt.plot(self.x_values, self.cdf_values)
        plt.plot(self.x_values, self.pdf_values)

        if save:
            plt.savefig("plot_pdf_cdf.png", dpi=300)

        if show:
            plt.show()

    def __add__(self, other):
        if isinstance(other, self.__class__):
            add = self.data + other.data
            print(len(add))
        elif isinstance(other, (int, float)):
            add = self.data + float(other)
        else:
            print("Unindentified input type! ({}, {})".format(other, type(other)))
            sys.exit()
        temp_obj = AsymmetricData(creation_type='by_operation', data=add)
        return temp_obj

    def __radd__(self, other):
        if isinstance(other, self.__class__):
            add = other.data + self.data
        elif isinstance(other, (int, float)):
            add = float(other) + self.data
        else:
            print("Unindentified input type! ({}, {})".format(other, type(other)))
            sys.exit()
        temp_obj = AsymmetricData(creation_type='by_operation', data=add)
        return temp_obj

    def __sub__(self, other):
        if isinstance(other, self.__class__):
            add = self.data - other.data
        elif isinstance(other, (int, float)):
            add = self.data - float(other)
        else:
            print("Unindentified input type! ({}, {})".format(other, type(other)))
            sys.exit()
        temp_obj = AsymmetricData(creation_type='by_operation', data=add)
        return temp_obj

    def __rsub__(self, other):
        if isinstance(other, self.__class__):
            add = other.data - self.data
        elif isinstance(other, (int, float)):
            add = float(other) - self.data
        else:
            print("Unindentified input type! ({}, {})".format(other, type(other)))
            sys.exit()
        temp_obj = AsymmetricData(creation_type='by_operation', data=add)
        return temp_obj

    def __mul__(self, other):
        if isinstance(other, self.__class__):
            add = self.data * other.data
        elif isinstance(other, (int, float)):
            add = self.data * float(other)
        else:
            print("Unindentified input type! ({}, {})".format(other, type(other)))
            sys.exit()
        temp_obj = AsymmetricData(creation_type='by_operation', data=add)
        return temp_obj

    def __rmul__(self, other):
        if isinstance(other, self.__class__):
            add = other.data * self.data
        elif isinstance(other, (int, float)):
            add = float(other) * self.data
        else:
            print("Unindentified input type! ({}, {})".format(other, type(other)))
            sys.exit()
        temp_obj = AsymmetricData(creation_type='by_operation', data=add)
        return temp_obj

    def __truediv__(self, other):
        if isinstance(other, self.__class__):
            add = self.data / other.data
        elif isinstance(other, (int, float)):
            add = self.data / float(other)
        else:
            print("Unindentified input type! ({}, {})".format(other, type(other)))
            sys.exit()
        temp_obj = AsymmetricData(creation_type='by_operation', data=add)
        return temp_obj

    def __rtruediv__(self, other):
        if isinstance(other, self.__class__):
            add = other.data / self.data
        elif isinstance(other, (int, float)):
            add = float(other) / self.data
        else:
            print("Unindentified input type! ({}, {})".format(other, type(other)))
            sys.exit()
        temp_obj = AsymmetricData(creation_type='by_operation', data=add)
        return temp_obj

    def __pow__(self, other):
        if isinstance(other, self.__class__):
            add = self.data ** other.data
        elif isinstance(other, (int, float)):
            add = self.data ** float(other)
        else:
            print("Unindentified input type! ({}, {})".format(other, type(other)))
            sys.exit()
        temp_obj = AsymmetricData(creation_type='by_operation', data=add)
        return temp_obj

    def __rpow__(self, other):
        if isinstance(other, self.__class__):
            add = other.data ** self.data
        elif isinstance(other, (int, float)):
            add = float(other) ** self.data
        else:
            print("Unindentified input type! ({}, {})".format(other, type(other)))
            sys.exit()
        temp_obj = AsymmetricData(creation_type='by_operation', data=add)
        return temp_obj
