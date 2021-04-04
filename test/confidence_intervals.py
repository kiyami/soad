from soad import AsymmetricData as asyd
import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import leastsq
import matplotlib.pyplot as plt

a=asyd(10,1.0,1.5,10000)

#a.plot_log_likelihood(show=True, save=False)

def lnL(x, mu, sigma_n, sigma_p):
    par_1 = (2.0 * sigma_p * sigma_n) / (sigma_p + sigma_n)
    par_2 = (sigma_p - sigma_n) / (sigma_p + sigma_n)
    value = (-1.0 / 2.0) * ((mu - x) / (par_1 + par_2 * (x - mu))) ** 2.0

    return value

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

    resid = (lnL(mu-n3, mu, n1, p1) - target_likelihood)**2.0 + (lnL(mu+p3, mu, n1, p1) - target_likelihood)**2.0
    return resid

def convert_to_1_sigma(mu, n, p, confidence):
    N = 500
    n_range = np.linspace(1e-5, n, N)
    p_range = np.linspace(1e-5, p, N)

    np_matrix = np.zeros([n_range.shape[0], p_range.shape[0]])

    for i in range(n_range.shape[0]):
        for j in range(p_range.shape[0]):
            np_matrix[i, j] = np.log(residual([n_range[i], p_range[j]], mu, n, p, confidence))

    min_val = np_matrix.min()
    index_n, index_p = np.where(np_matrix == min_val)

    n_new, p_new = n_range[index_n[0]], p_range[index_p[0]]

    print("")
    print("# Converting to 1 sigma")
    print("# {} (-{},+{}) ({} sigma) -> {:.3} (-{:.3},+{:.3}) ({} sigma)".format(mu,n,p,confidence,mu,n_new,p_new,1.0))

    # print(np_matrix)
    #print(min_val, index_n[0], n_range[index_n[0]], index_p[0], p_range[index_p[0]])

    #plt.imshow(np_matrix)
    #plt.show()

    return [n_new, p_new]

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
    current_likelihood = lnL(current_value, mu, sigma_n, sigma_p)

    while abs(current_likelihood) < abs(target_likelihood):
        current_value += delta
        current_likelihood = lnL(current_value, mu, sigma_n, sigma_p)
    positive_limit = current_value

    current_value = mu
    delta = abs(mu - sigma_n) * delta_steps
    current_likelihood = lnL(current_value, mu, sigma_n, sigma_p)

    while abs(current_likelihood) < abs(target_likelihood):
        current_value -= delta
        current_likelihood = lnL(current_value, mu, sigma_n, sigma_p)
    negative_limit = current_value

    n_new, p_new = mu - negative_limit, positive_limit - mu

    print("")
    print("# Converting from 1 sigma")
    print("# {} (-{},+{}) ({} sigma) -> {:.3} (-{:.3},+{:.3}) ({} sigma)".format(mu,sigma_n,sigma_p,1.0,mu,n_new,p_new,confidence))

    return [n_new, p_new]


#convert_to_1_sigma(10.0,1.71,4.0,2.0)
#convert_to_1_sigma(10.0,2.25,9.0,3.0)

#convert_from_1_sigma(10.0,1.0,1.0,3.0)
#convert_from_1_sigma(10.0,1.0,1.1,3.0)
#convert_from_1_sigma(10.0,1.0,1.2,3.0)
#convert_from_1_sigma(10.0,1.0,1.3,3.0)
#convert_from_1_sigma(10.0,1.0,1.4,3.0)
#convert_from_1_sigma(10.0,1.0,1.5,3.0)


def convert_to_1_sigma_fast(mu, nerr, perr, confidence):
    if confidence == 1.0:
        target_likelihood = -0.5
    elif confidence == 2.0:
        target_likelihood = -2.0
    elif confidence == 3.0:
        target_likelihood = -4.5
    else:
        target_likelihood = -0.5
        print("Something went wrong!")

    def func(x, sigma_n, sigma_p):
        par_1 = (2.0 * sigma_p * sigma_n) / (sigma_p + sigma_n)
        par_2 = (sigma_p - sigma_n) / (sigma_p + sigma_n)
        value = (-1.0 / 2.0) * ((mu - x) / (par_1 + par_2 * (x - mu))) ** 2.0

        return value

    xdata = [mu-nerr, mu, mu+perr]
    ydata = [target_likelihood, 0.0, target_likelihood]

    popt, pcov = curve_fit(func, xdata, ydata)

    print(popt)
    print(pcov)

    return popt


convert_from_1_sigma(10.0, 1.0, 1.1, 2.0)
convert_from_1_sigma(10.0, 1.0, 1.1, 3.0)

convert_to_1_sigma_fast(10.0, 1.91, 2.32, 2.0)
convert_to_1_sigma_fast(10.0, 2.75, 3.67, 3.0)
