import numpy as np

import matplotlib.pyplot as plt

from scipy.optimize import fmin
from scipy.stats import skewnorm

import sys
import os


class AsymmetricData:
    
    confidence_conversion = {
        "1 sigma": 0.6827,
        "2 sigma": 0.9545,
        "3 sigma": 0.9973,
    }
    

    def __init__(self, 
                 mode=None, sigma_n=None, sigma_p=None, N=50_000, 
                 skew=None, loc=None, scale=None,
                 confidence=None, data=None):
        

        # if confidence is None, set it to 1 sigma value.
        if confidence is None:
            self.confidence = self.confidence_conversion["1 sigma"]
        
        else:
            # if confidence can be convertale to float and between 0 and 1, it is ok.
            # else, raise Value Error
            try:
                confidence = float(confidence)
                
                if 0.0 < confidence < 1.0:
                    self.confidence = confidence
                    
                else:
                    raise Exception('Value Error: confidence value should be a number between 0 and 1 (not inclusive)!!!')
                
            except:
                raise Exception('Value Error: confidence value should be a number between 0 and 1 (not inclusive)!!!')
                
        
        # asymmetric distribution parameters
        self.mode = mode
        
        # input sigma values (can be for any confidence interval)
        self.sigma_n, self.sigma_p = sigma_n, sigma_p
        
        # sigma values for different confidence intervals
        self.sigma1_n, self.sigma1_p = None, None
        self.sigma2_n, self.sigma2_p = None, None
        self.sigma3_n, self.sigma3_p = None, None

        
        # skewnorm parameters
        self.skew, self.loc, self.scale = skew, loc, scale
        
        # set data input
        self.data = data
        
        
        # we use 'sigma_n' and 'sigma_p' instead of 'self.sigma1_n' and 'self.sigma1_p',
        # because these input values can be in any confidence value.
        # (e.g. 'sigma2_n' and 'sigma2_p')
        asymmetric_input = (self.mode is not None) and (self.sigma_n is not None) and (self.sigma_p is not None)
        skewnorm_input = (self.skew is not None) and (self.loc is not None) and (self.scale is not None)
        data_input = (not asymmetric_input) and (not skewnorm_input) and (self.data is not None) and isinstance(self.data, (np.ndarray, list))
        
        
        # if the input is given as only as 'mode, sigma_n, sigma_p',
        # calculate 'skew, loc, scale' parameters
        if asymmetric_input and (not skewnorm_input) and (not data_input):
            
            self.skew, self.loc, self.scale = self.sigma_to_skew(self.mode, self.sigma_n, self.sigma_p, confidence=self.confidence)
                            
            _, self.sigma1_n, self.sigma1_p = self.skew_to_sigma(self.skew, self.loc, self.scale, confidence=self.confidence_conversion["1 sigma"], 
                                                                 xmin=None, xmax=None)

            _, self.sigma2_n, self.sigma2_p = self.skew_to_sigma(self.skew, self.loc, self.scale, confidence=self.confidence_conversion["2 sigma"], 
                                                                 xmin=None, xmax=None)

            _, self.sigma3_n, self.sigma3_p = self.skew_to_sigma(self.skew, self.loc, self.scale, confidence=self.confidence_conversion["3 sigma"], 
                                                                 xmin=None, xmax=None)
            
            # sample size for Monte Carlo simulation
            self.N = int(N)
            
            # generated data with skewnorm parameters
            self.data = skewnorm.rvs(self.skew, self.loc, self.scale, self.N)

                
        # if the input is given as only as 'skew, loc, scale',
        # calculate 'mode, sigma_n, sigma_p' parameters
        elif skewnorm_input and (not asymmetric_input) and (not data_input):
            
            self.mode, self.sigma_n, self.sigma_p = self.skew_to_sigma(self.skew, self.loc, self.scale, confidence=self.confidence, 
                                                                       xmin=None, xmax=None)
                            
            _, self.sigma1_n, self.sigma1_p = self.skew_to_sigma(self.skew, self.loc, self.scale, confidence=self.confidence_conversion["1 sigma"], 
                                                                 xmin=None, xmax=None)

            _, self.sigma2_n, self.sigma2_p = self.skew_to_sigma(self.skew, self.loc, self.scale, confidence=self.confidence_conversion["2 sigma"], 
                                                                 xmin=None, xmax=None)

            _, self.sigma3_n, self.sigma3_p = self.skew_to_sigma(self.skew, self.loc, self.scale, confidence=self.confidence_conversion["3 sigma"], 
                                                                 xmin=None, xmax=None)
            
            # sample size for Monte Carlo simulation
            self.N = int(N)
            
            # generated data with skewnorm parameters
            self.data = skewnorm.rvs(self.skew, self.loc, self.scale, self.N)
   

        elif data_input and (not asymmetric_input) and (not skewnorm_input):
        
            self.N = len(self.data)
            self.skew, self.loc, self.scale = skewnorm.fit(self.data)
            
            self.mode, self.sigma_n, self.sigma_p = self.skew_to_sigma(self.skew, self.loc, self.scale, confidence=self.confidence, 
                                                                 xmin=None, xmax=None)
            
            _, self.sigma1_n, self.sigma1_p = self.skew_to_sigma(self.skew, self.loc, self.scale, confidence=self.confidence_conversion["1 sigma"], 
                                                                 xmin=None, xmax=None)

            _, self.sigma2_n, self.sigma2_p = self.skew_to_sigma(self.skew, self.loc, self.scale, confidence=self.confidence_conversion["2 sigma"], 
                                                                 xmin=None, xmax=None)

            _, self.sigma3_n, self.sigma3_p = self.skew_to_sigma(self.skew, self.loc, self.scale, confidence=self.confidence_conversion["3 sigma"], 
                                                                 xmin=None, xmax=None)

        else:
            sys.exit("Wrong input!!!")
            
    def __str__(self):
        
        conf = self.confidence
        conf_1s = self.confidence_conversion['1 sigma']
        conf_2s = self.confidence_conversion['2 sigma']
        conf_3s = self.confidence_conversion['3 sigma']
        
        output = f"Value = {self.mode:.4f} (-{self.sigma_n:.4f}, +{self.sigma_p:.4f}) (conf: {conf:.4f}) (input)"
        seperator = "-"*len(output)
        output1 = f"Value = {self.mode:.4f} (-{self.sigma1_n:.4f}, +{self.sigma1_p:.4f}) (conf: {conf_1s:.4f}) (1 sigma)"
        output2 = f"Value = {self.mode:.4f} (-{self.sigma2_n:.4f}, +{self.sigma2_p:.4f}) (conf: {conf_2s:.4f}) (2 sigma)"
        output3 = f"Value = {self.mode:.4f} (-{self.sigma3_n:.4f}, +{self.sigma3_p:.4f}) (conf: {conf_3s:.4f}) (3 sigma)"

        result = "{}\n{}\n{}\n{}\n{}\n{}".format(output, seperator, output1, output2, output3, seperator)
        return result 
    
    def __repr__(self):
        
        conf = self.confidence
        conf_1s = self.confidence_conversion['1 sigma']
        conf_2s = self.confidence_conversion['2 sigma']
        conf_3s = self.confidence_conversion['3 sigma']
        
        output = f"Value = {self.mode:.4f} (-{self.sigma_n:.4f}, +{self.sigma_p:.4f}) (conf: {conf:.4f}) (input)"
        seperator = "-"*len(output)
        output1 = f"Value = {self.mode:.4f} (-{self.sigma1_n:.4f}, +{self.sigma1_p:.4f}) (conf: {conf_1s:.4f}) (1 sigma)"
        output2 = f"Value = {self.mode:.4f} (-{self.sigma2_n:.4f}, +{self.sigma2_p:.4f}) (conf: {conf_2s:.4f}) (2 sigma)"
        output3 = f"Value = {self.mode:.4f} (-{self.sigma3_n:.4f}, +{self.sigma3_p:.4f}) (conf: {conf_3s:.4f}) (3 sigma)"

        result = "{}\n{}\n{}\n{}\n{}\n{}".format(output, seperator, output1, output2, output3, seperator)
        return result 
    
    @staticmethod
    def find_mode(skew, loc, scale, xmin, xmax):
    
        if xmin is None:
            xmin = loc*0.7

        if xmax is None:
            xmax = loc*1.3
            
        n = 50_000

        x = np.linspace(xmin, xmax, n)

        scale = max(scale, loc * 1e-10)
        
        y = skewnorm.pdf(x, skew, loc, scale)

        mode_idx = np.where(y == y.max())
        mode = x[mode_idx[0][0]]

        return mode

    
    @staticmethod
    def find_bounds(skew, loc, scale, confidence):
        left_bound = skewnorm.ppf((1 - confidence) / 2,  skew, loc, scale)
        right_bound = skewnorm.ppf(1 - (1 - confidence) / 2,  skew, loc, scale)

        return left_bound, right_bound
    
    @staticmethod
    def find_sigmas(mode, left_bound, right_bound):
        return mode-left_bound, right_bound-mode
    
    @staticmethod
    def skew_to_sigma(skew, loc, scale, confidence=0.683, xmin=None, xmax=None):
        mode = AsymmetricData.find_mode(skew, loc, scale, xmin, xmax)
        left_bound, right_bound = AsymmetricData.find_bounds(skew, loc, scale, confidence)
        sigma_n, sigma_p = AsymmetricData.find_sigmas(mode, left_bound, right_bound)

        return np.array([mode, sigma_n, sigma_p], dtype=np.float64)

    @staticmethod
    def fit_func(x, mode, sigma_n, sigma_p, confidence, xmin=None, xmax=None):
        skew, loc, scale = x

        if xmin is None:
            xmin = mode - sigma_n

        if xmax is None:
            xmax = mode + sigma_p 

        calculated = AsymmetricData.skew_to_sigma(skew, loc, scale, confidence=confidence, xmin=xmin, xmax=xmax)
        expected = np.array([mode, sigma_n, sigma_p], dtype=np.float64)

        return np.sqrt(((calculated - expected)**2.0).sum())
    
    @staticmethod
    def sigma_to_skew(mode, sigma_n, sigma_p, confidence=0.683):
        print("Calculating 'skew, loc, scale' parameters for given 'mu, sigma_n, sigma_p' values:")
        
        initial_skew = (sigma_p - sigma_n) / 2
        initial_loc = mode
        initial_scale = (sigma_n + sigma_p) / 2

        print("Initial guesses:\n", initial_skew, initial_loc, initial_scale)

        x0 = np.array([initial_skew, initial_loc, initial_scale])

        fitParams = fmin(AsymmetricData.fit_func, x0, args=(mode, sigma_n, sigma_p, confidence, None, None), 
                         xtol=1e-25, ftol=1e-25, maxiter=1000, maxfun=1000, full_output=True)

        params = fitParams[0]
        func_val = fitParams[1]

        fit_repeat_count = 0

        print("Initial fit results:", params)
        print("\tFit stat:", func_val)

        while (func_val > 1e-3) and (fit_repeat_count < 2):

            # update limits of loc parameter according to previous fit results
            xmin = params[1] * 0.9
            xmax = params[1] * 1.1

            # update initial parameter guesses
            x0_rep = params

            fitParams_rep= fmin(AsymmetricData.fit_func, x0_rep, args=(mode, sigma_n, sigma_p, confidence, xmin, xmax), 
                             xtol=1e-25, ftol=1e-25, maxiter=1000, maxfun=1000, full_output=True)

            params = fitParams_rep[0]
            func_val = fitParams_rep[1]

            fit_repeat_count += 1

            print(f"{fit_repeat_count}. fit results:", params)
            print("\tFit stat:", func_val)
            
        skew, loc, scale = params

        return skew, loc, scale
    
    def pdf(self, x):
        return skewnorm(self.skew, self.loc, self.scale).pdf(x)
    
    def get_pdf(self):
        return skewnorm(self.skew, self.loc, self.scale).pdf
    
    def cdf(self, x):
        return skewnorm(self.skew, self.loc, self.scale).cdf(x)
    
    def get_cdf(self):
        return skewnorm(self.skew, self.loc, self.scale).cdf
    
    def plot(self, show=True, save=False, filename=None):
        
        #xmin = self.mode - 5*self.sigma_n
        #xmax = self.mode + 5*self.sigma_p
        
        xmin = 0.99 * self.data.min()
        xmax = 1.01 * self.data.max()
        
        xrange = np.linspace(xmin, xmax, 1000)
        pdf_values = skewnorm.pdf(xrange, self.skew, self.loc, self.scale)
        
        plt.clf()
        plt.hist(self.data, bins=100, density=True, color="teal", alpha=0.6)
        plt.plot(xrange, pdf_values, color="black")

        plt.xlabel("x")
        plt.ylabel("Prob.")

        plt.axvline(x=self.mode - self.sigma_n, color="black", ls="--", lw="1.5")
        plt.axvline(x=self.mode + self.sigma_p, color="black", ls="--", lw="1.5")
        
        numeric_scale = round(-np.log10(self.mode))
        numeric_num = max(2, numeric_scale+2)
        
        plt.axvline(x=self.mode, color="black", lw="1.5",
                    label=f" $\mu$={self.mode:.{numeric_num}f}(-{self.sigma_n:.{numeric_num}f},+{self.sigma_p:.{numeric_num}f}) \n(conf.: {self.confidence:.3f})")

        plt.ylim(top=1.35 * pdf_values.max())

        plt.legend()
        
        
        if show:
            plt.show()
            
            
        if save:
            
            if filename is None:
                
                fig_count = 1
                filename = f"figure{fig_count}.jpg"
                
                while filename in os.listdir():
                    fig_count += 1
                    filename = f"figure{fig_count}.jpg"
                    
            else:
                
                if filename in os.listdir():
                    print(f"SAVE ERROR: File name already exist! ({filename})\nSkipping..")
                    return

                
            try:    
                plt.savefig(filename, dpi=150)
                print(f"Plot saved succesfully. ({filename})")
                
            except:
                print("SAVE ERROR: Couldn't save plot!!!")
    
    def __add__(self, other):
        
        if isinstance(other, self.__class__):
            
            N_self = len(self.data)
            N_other = len(other.data)
            
            if N_self != N_other:
                N = min(N_self, N_other)
            
            else:
                N = N_self
            
            calc_data = self.data[:N] + other.data[:N]
            
        elif isinstance(other, (int, float)):
            calc_data = self.data + float(other)
            
        else:
            print("Unindentified input type! ({}, {})".format(other, type(other)))
            sys.exit()
            
        temp_obj = AsymmetricData(data=calc_data)
        
        return temp_obj
    
    def __radd__(self, other):
        
        if isinstance(other, self.__class__):
            
            N_self = len(self.data)
            N_other = len(other.data)
            
            if N_self != N_other:
                N = min(N_self, N_other)
            
            else:
                N = N_self
            
            calc_data = other.data[:N] + self.data[:N]
            
        elif isinstance(other, (int, float)):
            calc_data = self.data + float(other)
            
        else:
            print("Unindentified input type! ({}, {})".format(other, type(other)))
            sys.exit()
            
        temp_obj = AsymmetricData(data=calc_data)
        
        return temp_obj
    
    def __sub__(self, other):
        
        if isinstance(other, self.__class__):
            
            N_self = len(self.data)
            N_other = len(other.data)
            
            if N_self != N_other:
                N = min(N_self, N_other)
            
            else:
                N = N_self
            
            calc_data = self.data[:N] - other.data[:N]
            
        elif isinstance(other, (int, float)):
            calc_data = self.data - float(other)
            
        else:
            print("Unindentified input type! ({}, {})".format(other, type(other)))
            sys.exit()
            
        temp_obj = AsymmetricData(data=calc_data)
        
        return temp_obj
    
    def __rsub__(self, other):
        
        if isinstance(other, self.__class__):
            
            N_self = len(self.data)
            N_other = len(other.data)
            
            if N_self != N_other:
                N = min(N_self, N_other)
            
            else:
                N = N_self
            
            calc_data = other.data[:N] - self.data[:N]
            
        elif isinstance(other, (int, float)):
            calc_data = float(other) - self.data
            
        else:
            print("Unindentified input type! ({}, {})".format(other, type(other)))
            sys.exit()
            
        temp_obj = AsymmetricData(data=calc_data)
        
        return temp_obj
    
    def __mul__(self, other):
        
        if isinstance(other, self.__class__):
            
            N_self = len(self.data)
            N_other = len(other.data)
            
            if N_self != N_other:
                N = min(N_self, N_other)
            
            else:
                N = N_self
            
            calc_data = self.data[:N] * other.data[:N]
            
        elif isinstance(other, (int, float)):
            calc_data = self.data * float(other)
            
        else:
            print("Unindentified input type! ({}, {})".format(other, type(other)))
            sys.exit()
            
        temp_obj = AsymmetricData(data=calc_data)
        
        return temp_obj
    
    def __rmul__(self, other):
        
        if isinstance(other, self.__class__):
            
            N_self = len(self.data)
            N_other = len(other.data)
            
            if N_self != N_other:
                N = min(N_self, N_other)
            
            else:
                N = N_self
            
            calc_data = other.data[:N] * self.data[:N]
            
        elif isinstance(other, (int, float)):
            calc_data = float(other) * self.data
            
        else:
            print("Unindentified input type! ({}, {})".format(other, type(other)))
            sys.exit()
            
        temp_obj = AsymmetricData(data=calc_data)
        
        return temp_obj
    
    def __div__(self, other):
        
        if isinstance(other, self.__class__):
            
            N_self = len(self.data)
            N_other = len(other.data)
            
            if N_self != N_other:
                N = min(N_self, N_other)
            
            else:
                N = N_self
            
            calc_data = self.data[:N] / other.data[:N]
            
        elif isinstance(other, (int, float)):
            calc_data = self.data / float(other)
            
        else:
            print("Unindentified input type! ({}, {})".format(other, type(other)))
            sys.exit()
            
        temp_obj = AsymmetricData(data=calc_data)
        
        return temp_obj
    
    def __rdiv__(self, other):
        
        if isinstance(other, self.__class__):
            
            N_self = len(self.data)
            N_other = len(other.data)
            
            if N_self != N_other:
                N = min(N_self, N_other)
            
            else:
                N = N_self
            
            calc_data = other.data[:N] / self.data[:N]
            
        elif isinstance(other, (int, float)):
            calc_data = float(other) / self.data
            
        else:
            print("Unindentified input type! ({}, {})".format(other, type(other)))
            sys.exit()
            
        temp_obj = AsymmetricData(data=calc_data)
        
        return temp_obj
    
    def __truediv__(self, other):
        
        if isinstance(other, self.__class__):
            
            N_self = len(self.data)
            N_other = len(other.data)
            
            if N_self != N_other:
                N = min(N_self, N_other)
            
            else:
                N = N_self
            
            calc_data = self.data[:N] / other.data[:N]
            
        elif isinstance(other, (int, float)):
            calc_data = self.data / float(other)
            
        else:
            print("Unindentified input type! ({}, {})".format(other, type(other)))
            sys.exit()
            
        temp_obj = AsymmetricData(data=calc_data)
        
        return temp_obj
    
    def __rtruediv__(self, other):
        
        if isinstance(other, self.__class__):
            
            N_self = len(self.data)
            N_other = len(other.data)
            
            if N_self != N_other:
                N = min(N_self, N_other)
            
            else:
                N = N_self
            
            calc_data = other.data[:N] / self.data[:N]
            
        elif isinstance(other, (int, float)):
            calc_data = float(other) / self.data
            
        else:
            print("Unindentified input type! ({}, {})".format(other, type(other)))
            sys.exit()
            
        temp_obj = AsymmetricData(data=calc_data)
        
        return temp_obj
    
    def __pow__(self, other):
        
        if isinstance(other, self.__class__):
            
            N_self = len(self.data)
            N_other = len(other.data)
            
            if N_self != N_other:
                N = min(N_self, N_other)
            
            else:
                N = N_self
            
            calc_data = self.data[:N] ** other.data[:N]
            
        elif isinstance(other, (int, float)):
            calc_data = self.data ** float(other)
            
        else:
            print("Unindentified input type! ({}, {})".format(other, type(other)))
            sys.exit()
            
        temp_obj = AsymmetricData(data=calc_data)
        
        return temp_obj
    
    def __rpow__(self, other):
        
        if isinstance(other, self.__class__):
            
            N_self = len(self.data)
            N_other = len(other.data)
            
            if N_self != N_other:
                N = min(N_self, N_other)
            
            else:
                N = N_self
            
            calc_data = other.data[:N] ** self.data[:N]
            
        elif isinstance(other, (int, float)):
            calc_data = float(other) ** self.data
            
        else:
            print("Unindentified input type! ({}, {})".format(other, type(other)))
            sys.exit()
            
        temp_obj = AsymmetricData(data=calc_data)
        
        return temp_obj
    
