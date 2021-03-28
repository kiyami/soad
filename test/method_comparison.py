from soad import AsymmetricData as asyd
import matplotlib.pyplot as plt


# This script is prepared for showing the difference between methods of handling asymmetric errors.

class Data:
    control_variable_parameters = [10.0, 1.0, 1.0]

    control_variable = []
    variable_list = []

    def __init__(self, mu, sigma_n, sigma_p):
        self.mu = float(mu)
        self.sigma_n = float(sigma_n)
        self.sigma_p = float(sigma_p)
        self.avg_std = (self.sigma_n + self.sigma_p) * 0.5

    def get_params(self):
        return [float(self.mu), float(self.sigma_n), float(self.sigma_p)]

    @classmethod
    def set_control_variable(cls):
        cls.control_variable = Data(*cls.control_variable_parameters)

    @classmethod
    def print_variables(cls):
        for variable in cls.variable_list:
            print(variable.get_params())

    @staticmethod
    def calculate_asym_index(sigma_n, sigma_p):
        sigma_n = float(sigma_n)
        sigma_p = float(sigma_p)
        return float((sigma_p - sigma_n) / (sigma_p + sigma_n))

    @staticmethod
    def calculate_sigma_p(sigma_n, asym_index):
        sigma_n = float(sigma_n)
        asym_index = float(asym_index)
        return float(sigma_n * (1.0 + asym_index) / (1.0 - asym_index))

    @staticmethod
    def calculate_sigma_n(sigma_p, asym_index):
        sigma_p = float(sigma_p)
        asym_index = float(asym_index)
        return float(sigma_p * (1.0 - asym_index) / (1.0 + asym_index))


def generate_single_variable(*args):
    Data.variable_list.append(Data(*args))


def generate_control_variable(*args):
    Data.control_variable = Data(*args)


def generate_multiple_variable():
    n = 15
    asym_index = 0.2
    mu, sigma_n, sigma_p = Data.control_variable.get_params()

    start = float(sigma_p)
    stop = float(Data.calculate_sigma_p(sigma_n, asym_index))
    step = (stop - start) / float(n)

    for i in range(n+1):
        temp_sigma_p = float(sigma_p) + (float(i)*float(step))
        print("###### New sigma_p: ", temp_sigma_p)
        generate_single_variable(mu, sigma_n, temp_sigma_p)



class AverageMethod:
    result_list = []

    @classmethod
    def sum(cls, val_1):
        val_2 = Data.control_variable
        mu_result = val_1.mu + val_2.mu
        std_result = (val_1.avg_std**2.0 + val_2.avg_std**2.0)**0.5
        cls.result_list.append(Data(mu_result, std_result, std_result))

    @classmethod
    def mul(cls, val_1):
        val_2 = Data.control_variable
        mu_result = val_1.mu * val_2.mu
        std_result = mu_result * ((val_1.avg_std / val_1.mu)**2.0 + (val_2.avg_std / val_2.mu)**2.0)**0.5
        cls.result_list.append(Data(mu_result, std_result, std_result))

    @classmethod
    def print_results(cls):
        print("Results for AverageMethod")
        for result in cls.result_list:
            print(result.get_params())

class MonteCarloMethod:
    N = 50000
    result_list = []
    control_variable = []

    @classmethod
    def generate_control_variable(cls):
        mu, sigma_n, sigma_p = Data.control_variable.get_params()
        cls.control_variable = asyd(mu, sigma_n, sigma_p, N=cls.N)

    @classmethod
    def sum(cls, val):
        if not cls.control_variable:
            cls.generate_control_variable()

        mu, sigma_n, sigma_p = val.get_params()
        asym_val = asyd(mu, sigma_n, sigma_p, N=cls.N)

        result = cls.control_variable + asym_val
        result_val = [result.mu, result.sigma_n, result.sigma_p]
        cls.result_list.append(result)

    @classmethod
    def mul(cls, val):
        if not cls.control_variable:
            cls.generate_control_variable()

        mu, sigma_n, sigma_p = val.get_params()
        asym_val = asyd(mu, sigma_n, sigma_p, N=cls.N)

        result = cls.control_variable * asym_val
        cls.result_list.append(result)

    @classmethod
    def print_results(cls):
        print("Results for MonteCarloMethod")
        for result in cls.result_list:
            print([result.mu, result.sigma_n, result.sigma_p])

class CompareMethods:
    methods = [AverageMethod, MonteCarloMethod]

    @classmethod
    def calculate_sum(cls):
        for variable in Data.variable_list:
            for method in cls.methods:
                method.sum(variable)

    @classmethod
    def calculate_mul(cls):
        for variable in Data.variable_list:
            for method in cls.methods:
                method.mul(variable)

    @classmethod
    def print_results(cls):
        print("Result Comparison")
        for method in cls.methods:
            method.print_results()

    # plotu düzelt
    @classmethod
    def plot_results(cls, save=True):
        plt.clf()
        fig, ax = plt.subplots(figsize=(8, 5.5))

        plot_counter = 0
        colors = ["deepskyblue", "tomato"]
        ecolors = ["lightskyblue", "salmon"]
        plot_shift_delta = 0.002
        for method in cls.methods:
            plot_shift = plot_counter * plot_shift_delta
            print(plot_counter, plot_shift)
            x = [(Data.calculate_asym_index(x.sigma_n, x.sigma_p)+plot_shift) for x in Data.variable_list]
            print("x", x)
            y = [x.mu for x in method.result_list]
            yerr_neg = [x.sigma_n for x in method.result_list]
            yerr_poz = [x.sigma_p for x in method.result_list]
            plt.errorbar(x, y, yerr=[yerr_neg, yerr_poz], fmt='o', color=colors[plot_counter], ecolor=ecolors[plot_counter], elinewidth=3, capsize=0)
            plot_counter += 1

        plt.axhline(y=AverageMethod.result_list[0].mu, color="black", linewidth=2)

        plt.title("Method Comparison")
        plt.xlabel("Asymmetry Index")
        plt.ylabel("Result")
        plt.grid("True")

        ax.set_facecolor('whitesmoke')

        if save:
            plt.savefig("trial-1.png", dpi=300)

        plt.show()


if __name__ == "__main__":

    Data.set_control_variable()
    generate_multiple_variable()
    Data.print_variables()

    CompareMethods.calculate_sum()
    #CompareMethods.calculate_mul()

    CompareMethods.print_results()
    CompareMethods.plot_results(save=True)
