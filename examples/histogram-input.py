from uncertainty import AsymmetricData as asyd
import random
import matplotlib.pyplot as plt
import numpy as np

N=250000
#data_input = [random.normalvariate(20,2) for i in range(N)]
data_input = list(np.random.normal(10,1,N))

#a = asyd(creation_type='by_operation', data=data_input)
a = asyd(10,1,1,N=N)
#b = asyd(10,1,1.5,N=N)
print(a)
a.plot_data_and_pdf(show=True, save=True)
#a.plot_pdf()
#print(a)

#c=a**0.1
#print("print c", c)
#c.plot_data_and_pdf()
#plt.hist(a.data, bins=80, density=True, color='red', alpha=0.5)
#plt.hist(b.data, bins=80, density=True, color='blue', alpha=0.5)
#plt.hist(c.data, bins=100, density=True, color='cyan', alpha=0.5)

#plt.show()
