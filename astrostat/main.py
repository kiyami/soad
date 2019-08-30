from scripts import distributions as dist, generator as gen

var_gau = dist.VariableGaussian

a = var_gau.new(10,1,1.5,50000)
print(a.norm)
area = a.integrate()
print(a.norm, area)

#a.sample_data()

#a.plot_pdf()
#a.plot_data_and_pdf(bin=30)
#a.plot_pdf_cdf()
cdf = gen.find_cdf(a)
a.plot_data_and_pdf(bin=100)
