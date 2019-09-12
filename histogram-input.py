from uncertainty import AsymmetricData as asyd
import random
import matplotlib.pyplot as plt
import numpy as np

N=100000
#data_input = [random.normalvariate(20,2) for i in range(N)]
data_input = list(np.random.normal(20,2,N))

a = asyd(data=data_input)
b = asyd(20,2,2,N=N)
print(a)
a.plot_pdf()
#print(a)

#plt.hist(b.data, bins=100, density=True, color='blue')
#plt.hist(a.data, bins=100, density=True, color='red')
#plt.show()