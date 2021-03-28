**Note:** If you use this code for a publication please cite:

- <em>Erdіm, M. Kıyami, and Murat Hüdaverdі. "Asymmetric uncertainties in measurements: SOAD a Python package based on Monte Carlo Simulations." AIP Conference Proceedings. Vol. 2178. No. 1. AIP Publishing LLC, 2019.</em>

https://doi.org/10.1063/1.5135421

Any bug reports and suggestions are all welcomed.


> Type the following commands in terminal:
>
    $ git clone https://github.com/kiyami/soad.git
    $ pip3 install ./soad

> (Alternative way)
>
    $ git clone https://github.com/kiyami/soad.git
    $ cd soad
    $ python3 setup.py install

> After installation, you can remove the downloaded sneratio folder.

#### Usage example:
    >>> from soad import AsymmetricData as asyd
    >>>
    >>> a = asyd(10.0, 1.0, 1.2, N=10000)
    >>> b = asyd(15.0, 2.4, 2.1, N=10000)
    >>>
    >>> c = a * b
    >>> print(c)
    >>> c.plot_data_and_pdf(show=True, save=False)
    >>> c.plot_log_likelihood(show=True, save=False)
    >>>


# SOAD (Statistics of Asymmetric Distributions)

"soad" module is for handling measurement values with asymmetric uncertainties.

Measurement values are sampling with a pre-defined Probability Density Function (PDF).

And the result of the desired mathematical operation is applied to the values by Monte Carlo simulation.

Selected PDF to represent asymmetric data is "Variable Gaussian" which is defined by "Roger Barlow" in the article "Asymmetric Statistical Errors" (2004).


Defined operations:
[sum, subtract, multiply, divide, power], [+, -, x, /, **]

These operations can be used between "AsymmetricData" objects
or between an "AsymmetricData" object and an integer/float value.


#### Example:
Let "A" and "B" two measured quantity with asymmetric uncertainties:

A=10(-0.5,+1.2)

B=20(-0.4,0.9)

And we want to calculate "C=A+B" value.

Steps:
- Define the values
- Define the mathematical operation
- Print / Plot the results

#### How to use in Python interactively:

from soad import AsymmetricData

####### Initial definitions and random sample generation

A = AsymmetricData(10,0.5,1.2,N=100000) # (mean, negative sigma, positive sigma, sample size)

B = AsymmetricData(10,0.5,1.2,N=100000)

####### Do the calculation

C = A + B

####### Print / Plot the results

print(C)

C.plot_data(show=True, save=False)

C.plot_pdf(show=True, save=True)

C.plot_log_likelihood()

####### Another calculation

D = (0.5 * A)**0.5 + (B / 2)**2.0

print(D)



#### Author

> M. Kıyami ERDİM
> 
> mkiyami@yildiz.edu.tr
> 
> kiyami_erdim@outlook.com
