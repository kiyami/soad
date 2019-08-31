from scripts import distributions as dist

var_gau = dist.VariableGaussian

a = var_gau.new(30, 0.5, 0.6, 100000)
b = var_gau.new(30, 0.5, 0.6, 100000)
#c = (2.5*a + (b + 5.0))**0.5
c = a + b
#a.plot_pdf()
#a.plot_data_and_pdf(bin=30)
#a.plot_pdf_cdf()
#a.plot_data_and_pdf(bin=50)
#a.plot_cdf()
#a.plot_data()
#a.plot_log_likelihood()
#a.fit()
c.plot_data_and_pdf()
