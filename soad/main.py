import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
from scipy.optimize import curve_fit
import sys


class AsymmetricData:
    def __init__(self, mu=10.0, sigma_n=1.0, sigma_p=1.0, N=10000, confidence=1.0, creation_type='by_constructor', data=[]):
        """
        :param mu: Mode of the distribution, the most probable value
        :param sigma_n: Pegative sigma
        :param sigma_p: Positive sigma
        :param N: Sample size
        """
        self.mu = mu
        self.confidence = confidence  # sigma

        self.sigma_n, self.sigma_p = sigma_n, sigma_p
        self.sigma2_n, self.sigma2_p = None, None
        self.sigma3_n, self.sigma3_p = None, None

        self.N = int(N)
        self.creation_type = creation_type

        if not any(data):
            self.data = np.asarray([])
        else:
            self.data = np.asarray(data)
            #self.creation_type = 'by_operation'

        self.bin_value = 50

        if str(self.creation_type) == 'by_constructor':
            if confidence == 1.0:
                self.sigma_n, self.sigma_p = sigma_n, sigma_p
                self.sigma2_n, self.sigma2_p = self.convert_from_1_sigma(self.mu, sigma_n, sigma_p, 2.0)
                self.sigma3_n, self.sigma3_p = self.convert_from_1_sigma(self.mu, sigma_n, sigma_p, 3.0)
            elif confidence == 2.0:
                self.sigma_n, self.sigma_p = self.convert_to_1_sigma(self.mu, sigma_n, sigma_p, 2.0)
                self.sigma2_n, self.sigma2_p = sigma_n, sigma_p
                self.sigma3_n, self.sigma3_p = self.convert_from_1_sigma(self.mu, self.sigma_n, self.sigma_p, 3.0)
            elif confidence == 3.0:
                self.sigma_n, self.sigma_p = self.convert_to_1_sigma(self.mu, sigma_n, sigma_p, 3.0)
                self.sigma2_n, self.sigma2_p = self.convert_from_1_sigma(self.mu, self.sigma_n, self.sigma_p, 2.0)
                self.sigma3_n, self.sigma3_p = sigma_n, sigma_p
            else:
                raise ValueError

            self.x_limits = [self.mu - 5.0*self.sigma_n, self.mu + 5.0*self.sigma_p]
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

            self.sigma2_n, self.sigma2_p = self.convert_from_1_sigma(self.mu, self.sigma_n, self.sigma_p, 2.0)
            self.sigma3_n, self.sigma3_p = self.convert_from_1_sigma(self.mu, self.sigma_n, self.sigma_p, 3.0)

            self.x_limits = [self.mu - 5.0*self.sigma_n, self.mu + 5.0*self.sigma_p]
            self.x_values = np.linspace(self.x_limits[0], self.x_limits[1], self.N)

            self.norm = 1.0
            self.norm = self.calculate_norm()

            self.pdf_values = np.asarray(self.pdf(self.x_values))
            self.cdf_values = self.calculate_cdf_values()
            self.cdf = self.calculate_cdf()
            self.inverse_cdf = self.calculate_inverse_cdf()

    def __str__(self):
        output = f"Value = {self.mu:.4f} (-{self.sigma_n:.4f}, +{self.sigma_p:.4f}) (1 sigma)"
        output2 = f"Value = {self.mu:.4f} (-{self.sigma2_n:.4f}, +{self.sigma2_p:.4f}) (2 sigma)"
        output3 = f"Value = {self.mu:.4f} (-{self.sigma3_n:.4f}, +{self.sigma3_p:.4f}) (3 sigma)"

        result = "{}\n{}\n{}".format(output, output2, output3)
        return result

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

        return value

    def log_likelihood(self, x):
        par_1 = (2.0 * self.sigma_p * self.sigma_n) / (self.sigma_p + self.sigma_n)
        par_2 = (self.sigma_p - self.sigma_n) / (self.sigma_p + self.sigma_n)
        value = (-1.0/2.0) * ((self.mu - x)/(par_1 + par_2*(x - self.mu)))**2.0

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
        rnd_prob = np.random.uniform(0, 1, self.N)
        self.data = self.inverse_cdf(rnd_prob)

    @staticmethod
    def fit_func(x, norm, mu, sigma_n, sigma_p):
        par_1 = (2.0 * sigma_p * sigma_n) / (sigma_p + sigma_n)
        par_2 = (sigma_p - sigma_n) / (sigma_p + sigma_n)
        par_3 = (-1.0 / 2.0) * ((mu - x) / (par_1 + par_2 * (x - mu))) ** 2.0
        par_4 = norm / (2.0 * np.pi) ** 0.5
        value = par_4 * np.exp(par_3)

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

        #print("mod", mod)
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
        #print("params", params)
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
        plt.ylim([-5, 1.5])

        plt.xlabel("x")
        plt.ylabel("ln L")

        plt.axhline(y=-0.5, color="black", ls="--", lw="2.0", label=f"Value = {self.mu:.4f} (-{self.sigma_n:.4f}, +{self.sigma_p:.4f}) (1 sigma)")
        plt.axhline(y=-2.0, color="black", ls="--", lw="1.5", label=f"Value = {self.mu:.4f} (-{self.sigma2_n:.4f}, +{self.sigma2_p:.4f}) (2 sigma)")
        plt.axhline(y=-4.5, color="black", ls="--", lw="1.0", label=f"Value = {self.mu:.4f} (-{self.sigma3_n:.4f}, +{self.sigma3_p:.4f}) (3 sigma)")

        plt.legend()

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

        plt.xlabel("x")
        plt.ylabel("Prob.")

        plt.axvline(x=self.mu - self.sigma_n, color="black", ls="--", lw="1.5",
                    label=f"Value = {self.mu:.4f} (-{self.sigma_n:.4f}, +{self.sigma_p:.4f}) (1 sigma)")
        plt.axvline(x=self.mu + self.sigma_p, color="black", ls="--", lw="1.5")

        plt.axvline(x=self.mu - self.sigma2_n, color="black", ls="--", lw="1.0",
                    label=f"Value = {self.mu:.4f} (-{self.sigma2_n:.4f}, +{self.sigma2_p:.4f}) (2 sigma)")
        plt.axvline(x=self.mu + self.sigma2_p, color="black", ls="--", lw="1.0")

        plt.axvline(x=self.mu - self.sigma3_n, color="black", ls="--", lw="0.5",
                    label=f"Value = {self.mu:.4f} (-{self.sigma3_n:.4f}, +{self.sigma3_p:.4f}) (3 sigma)")
        plt.axvline(x=self.mu + self.sigma3_p, color="black", ls="--", lw="0.5")

        plt.legend()

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

    @staticmethod
    def lnL(x, mu, sigma_n, sigma_p):
        par_1 = (2.0 * sigma_p * sigma_n) / (sigma_p + sigma_n)
        par_2 = (sigma_p - sigma_n) / (sigma_p + sigma_n)
        value = (-1.0 / 2.0) * ((mu - x) / (par_1 + par_2 * (x - mu))) ** 2.0

        return value

    @staticmethod
    def residual(params1, mu, n3, p3, confidence):
        n1, p1 = params1

        if confidence == 1.0:
            target_likelihood = -0.5
        elif confidence == 2.0:
            target_likelihood = -2.0
        elif confidence == 3.0:
            target_likelihood = -4.5
        else:
            target_likelihood = -0.5
            print("Something went wrong!")

        resid = (AsymmetricData.lnL(mu - n3, mu, n1, p1) - target_likelihood) ** 2.0 + (
                    AsymmetricData.lnL(mu + p3, mu, n1, p1) - target_likelihood) ** 2.0
        return resid

    @staticmethod
    def convert_to_1_sigma(mu, n, p, confidence):
        N = 500
        n_range = np.linspace(1e-5, n, N)
        p_range = np.linspace(1e-5, p, N)

        np_matrix = np.zeros([n_range.shape[0], p_range.shape[0]])

        for i in range(n_range.shape[0]):
            for j in range(p_range.shape[0]):
                np_matrix[i, j] = np.log(AsymmetricData.residual([n_range[i], p_range[j]], mu, n, p, confidence))

        min_val = np_matrix.min()
        index_n, index_p = np.where(np_matrix == min_val)

        n_new, p_new = n_range[index_n[0]], p_range[index_p[0]]

        #print("")
        #print("# Converting to 1 sigma")
        #print("# {} (-{},+{}) ({} sigma) -> {} (-{},+{}) ({} sigma)".format(mu, n, p, confidence, mu, n_new, p_new, 1.0))

        return [n_new, p_new]

    @staticmethod
    def convert_from_1_sigma(mu, sigma_n, sigma_p, confidence):
        if confidence == 1.0:
            target_likelihood = -0.5
        elif confidence == 2.0:
            target_likelihood = -2.0
        elif confidence == 3.0:
            target_likelihood = -4.5
        else:
            target_likelihood = -0.5

        delta_steps = 1e-4

        current_value = mu
        delta = abs(mu - sigma_p) * delta_steps
        current_likelihood = AsymmetricData.lnL(current_value, mu, sigma_n, sigma_p)

        while abs(current_likelihood) < abs(target_likelihood):
            current_value += delta
            current_likelihood = AsymmetricData.lnL(current_value, mu, sigma_n, sigma_p)
        positive_limit = current_value

        current_value = mu
        delta = abs(mu - sigma_n) * delta_steps
        current_likelihood = AsymmetricData.lnL(current_value, mu, sigma_n, sigma_p)

        while abs(current_likelihood) < abs(target_likelihood):
            current_value -= delta
            current_likelihood = AsymmetricData.lnL(current_value, mu, sigma_n, sigma_p)
        negative_limit = current_value

        n_new, p_new = mu - negative_limit, positive_limit - mu

        #print("")
        #print("# Converting from 1 sigma")
        #print("# {} (-{},+{}) ({} sigma) -> {} (-{},+{}) ({} sigma)".format(mu, sigma_n, sigma_p, 1.0, mu, n_new, p_new, confidence))

        return [n_new, p_new]
