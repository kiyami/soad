import random
import numpy as np
import scipy.interpolate as interpolate
from scipy.interpolate import UnivariateSpline

def uniform_sampling(object):
    while len(object.data) < object.N:
        rnd_val = random.randint(0, object.x_values.size - 1)
        rnd_prob = random.uniform(0, 1.0/2.0)
        if object.pdf_values[rnd_val] >= rnd_prob:
            object.data = np.append(object.data, object.x_values[rnd_val])
    np.random.shuffle(object.data)
    return object.data

def find_cdf(object):
    #UnivariateSpline
    cdf = interpolate.interp1d(object.cdf_values, object.x_values, kind = 'cubic')
    for i in range(object.N):
        rnd_prob = random.uniform(0, 1)
        object.data = np.append(object.data, cdf(rnd_prob))
    return cdf



"""
def generate_data(self):
    print("Generating data.. (N={})".format(self.N))
    while len(self.data) < self.N:
        rnd_val = random.randint(0, self.x_values.size - 1)
        rnd_prob = random.uniform(0, self.norm)
        if self.pdf_values_norm[rnd_val] >= rnd_prob:
            self.data = np.append(self.data, self.x_values[rnd_val])
    np.random.shuffle(self.data)
    print("data set created! (N=%d) --> %s" % (self.data.size, self.__str__()))
    return self.data
"""