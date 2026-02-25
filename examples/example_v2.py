from soad import AsymmetricData

#----------------------------------------
# Example 1: 
# Creating asymmetric variables
# by using skew, loc and scale values
# with a confidence interval
#----------------------------------------
def example_1():
    a = AsymmetricData(skew=5, loc=100, scale=10, confidence=0.9, N=50_000)
    b = AsymmetricData(skew=-1, loc=5, scale=0.3, confidence=0.9, N=50_000)

    print(a)
    print(b)

#----------------------------------------
# Example 2: 
# Creating asymmetric variables
# by using mode, negative sigma and 
# positive sigma values
# with a confidence interval
#----------------------------------------
def example_2():
    c = AsymmetricData(mode=30, sigma_n=1, sigma_p=2, confidence=0.68)
    d = AsymmetricData(mode=20, sigma_n=2, sigma_p=1, confidence=0.68)

    print(c)
    print(d)

#-----------------------------------
# Example 3: Plotting a parameter
#-----------------------------------
def example_3():
    c = AsymmetricData(mode=30, sigma_n=1, sigma_p=2, confidence=0.68)
    c.plot(show=True, save=False)

#-----------------------------------
# Example 4: Basic operations
#-----------------------------------
def example_4():
    c = AsymmetricData(mode=30, sigma_n=1, sigma_p=2, confidence=0.68)
    d = AsymmetricData(mode=20, sigma_n=2, sigma_p=1, confidence=0.68)

    e = c + d
    print(e)


#-----------------------------------
# Example 5: Complicated operations
#-----------------------------------
def example_5():
    c = AsymmetricData(mode=30, sigma_n=1, sigma_p=2, confidence=0.95)
    d = AsymmetricData(mode=20, sigma_n=2, sigma_p=1, confidence=0.95)

    f = ( (c+d)**2 + (c-d)**2)**0.5
    print(f)
