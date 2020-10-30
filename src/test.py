from uncertainty import AsymmetricData as Asym

N = 50000

a = Asym(0.1, 0.05, 0.1, N)
b = Asym(100, 5.0, 25.0, N)

c = a/b
print(c)

c.plot_data_and_pdf(save=False, show=True)
