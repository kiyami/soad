from soad import AsymmetricData as asyd

# Example 1: (Basic operations)

a=asyd(mu=40, sigma_n=1, sigma_p=1.5, N=10000, confidence=1.0)
a.plot_pdf(show=True, save=False)

b=asyd(mu=20, sigma_n=1.5, sigma_p=1.2, N=10000, confidence=1.0)
b.plot_log_likelihood(show=True, save=False)

c = a * b
print(c)
c.plot_data_and_pdf(show=True, save=False)

d = ( (a+b)**2.0 / (a-b)**2.0 )**0.5
print(d)
d.plot_data_and_pdf(show=True, save=False)


# Example 2: (Different confidence levels)

x1 = asyd(mu=10, sigma_n=1, sigma_p=1.5, N=10000, confidence=1.0)
print("x1")
print(x1)

x2 = asyd(mu=10, sigma_n=1, sigma_p=1.5, N=10000, confidence=2.0)
print("x2")
print(x2)

x3 = asyd(mu=10, sigma_n=1, sigma_p=1.5, N=10000, confidence=3.0)
print("x3")
print(x3)


# Example 3: (Small numbers)

x4 = asyd(mu=0.01, sigma_n=0.001, sigma_p=0.0011, N=10000, confidence=1.0)
print("x4")
print(x4)

x5 = x4 * 0.1
print("x5")
print(x5)

x5.plot_data_and_pdf()