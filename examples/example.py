from uncertainty import AsymmetricData as asyd

a=asyd(10,1,1.5,100000)
b=asyd(20,1.5,1.0,100000)
c = a / b
#c.plot_data_and_pdf(show=True, save=False)
#a.plot_pdf(show=True, save=False)
#a.plot_log_likelihood(show=True, save=False)

#a=asyd(10.05,1.53,2.06,30000)
#b=asyd(4.95,0.66,0.87,30000)
#c=a/b
#print(c)
#c.plot_data_and_pdf()