from soad import AsymmetricData as asyd


a=asyd(10,1,1.5,50000)
b=asyd(20,1.5,1.0,50000)
c = a / b

a.plot_pdf(show=True, save=False)

a.plot_log_likelihood(show=True, save=False)

print(c)
c.plot_data_and_pdf(show=True, save=False)
