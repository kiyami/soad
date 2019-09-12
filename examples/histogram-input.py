from uncertainty import AsymmetricData as asyd
import random
import matplotlib.pyplot as plt
import numpy as np

N=100000
#data_input = [random.normalvariate(20,2) for i in range(N)]
data_input = list(np.random.normal(20,2,N))

a = asyd(creation_type='by_operation', data=data_input)
#a = asyd(20,2,2,N=N)
#b = asyd(10,1,2,N=N)
print(a)
#a.plot_pdf()
#print(a)

c=a**0.1
print("print c", c)
c.plot_data_and_pdf()
#plt.hist(a.data, bins=80, density=True, color='red', alpha=0.5)
#plt.hist(b.data, bins=80, density=True, color='blue', alpha=0.5)
#plt.hist(c.data, bins=100, density=True, color='cyan', alpha=0.5)

#plt.show()

#params [2037.08581748   64.77188903    7.60323176    8.52804339]
#interval found
#Value = 6.48e+01 ( - 7.60e+00 , + 8.55e+00 )

#params [2185.44835775   64.7150203     7.59951938    8.59122004]
#interval found
#Value = 6.47e+01 ( - 7.65e+00 , + 8.64e+00 )