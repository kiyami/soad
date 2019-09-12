from uncertainty import AsymmetricData as asyd

a=asyd(10,1,1,10000)
a.plot_data_and_pdf(show=True, save=False)
a.plot_pdf(show=True, save=False)
#a.plot_log_likelihood(show=False, save=True)