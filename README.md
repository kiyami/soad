# astrostat

"astrostat.uncertainty" module is for handling measurement values with asymmetric uncertainties.

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

from astrostat.uncertainty import AsymmetricData

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

M. Kıyami ERDİM

kiyami_erdim@hotmail.com
