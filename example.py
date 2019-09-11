from uncertainty import AsymmetricData as asyd

a=asyd(10,1,1,10000)
a.plot_data_and_pdf(show=False, save=True)
a.plot_pdf(show=False, save=True)
#a.plot_log_likelihood(show=False, save=True)