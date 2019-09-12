from uncertainty import AsymmetricData as asyd

#a=asyd(10,1,1,30000)
#a.plot_data_and_pdf(show=True, save=False)
#a.plot_pdf(show=True, save=False)
#a.plot_log_likelihood(show=False, save=True)

a=asyd(10.05,1.53,2.06,30000)
b=asyd(4.95,0.66,0.87,30000)
c=a/b
print(c)
c.plot_data_and_pdf()