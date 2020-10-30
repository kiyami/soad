from src.uncertainty import AsymmetricData

a=AsymmetricData(10,1,1.5,10000)
b=AsymmetricData(20,1.5,1.0,10000)
c = a / b
#c.plot_data_and_pdf(show=True, save=False)
#a.plot_pdf(show=True, save=False)
#a.plot_log_likelihood(show=True, save=False)

#a=asyd(10.05,1.53,2.06,30000)
#b=asyd(4.95,0.66,0.87,30000)
#c=a/b
#print(c)
#c.plot_data_and_pdf()