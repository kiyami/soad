from soad import AsymmetricData as asyd
import numpy as np


N=50000

data_input = list(np.random.normal(10,1,N))

a = asyd(creation_type='by_operation', data=data_input)


print(a)
a.plot_data_and_pdf(show=True, save=False)
a.plot_pdf()


#******* Ignore this part **********

#c=a**0.1
#print("print c", c)
#c.plot_data_and_pdf()
#plt.hist(a.data, bins=80, density=True, color='red', alpha=0.5)
#plt.hist(b.data, bins=80, density=True, color='blue', alpha=0.5)
#plt.hist(c.data, bins=100, density=True, color='cyan', alpha=0.5)

#plt.show()

#a = asyd(10,1,1.2,N=N)
#b = asyd(10,1,1.2,N=N)
"""
c = asyd(10,1,1.2,N=N)
d = asyd(10,1,1.2,N=N)
e = asyd(10,1,1.2,N=N)
f = asyd(10,1,1.2,N=N)
g = asyd(10,1,1.2,N=N)
h = asyd(10,1,1.2,N=N)
print(a,b,c,d,e,f,g,h,sep="\n")
data_input = np.concatenate((a.data, b.data, c.data, d.data, e.data, f.data, g.data, h.data))
"""
#data_input = np.concatenate((a.data, b.data))
#x = asyd(creation_type='by_operation', data=data_input)
#x = (a+b+c+d+e+f+g+h+a+b+c+d+e+f+g+h+a+b+c+d+e+f+g+h+a+b+c+d+e+f+g+h)/32.0

#x = (a+b)/2.0

#print(x)
#print("asym a: {}".format((a.sigma_p-a.sigma_n)/(a.sigma_p+a.sigma_n)))
#print("asym x: {}".format((x.sigma_p-x.sigma_n)/(x.sigma_p+x.sigma_n)))
#x.plot_data_and_pdf()

#means = []

#for i in range(5000):
#    temp = asyd(10,1.0,1.5,500)
#    temp_mean = np.sum(temp.data)/temp.N
#    means.append(temp_mean)

#plt.hist(means, bins=30)
#plt.show()