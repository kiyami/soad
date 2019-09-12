import uncertainty as asym
import matplotlib.pyplot as plt
import numpy as np

a = asym.AsymmetricData(2.03, 0.48, 0.48,N=50000)
b = asym.AsymmetricData(1.93e+00, 4.17e-01 , 5.32e-01,N=50000)


plt.plot(a.x_values,a.pdf_values)
plt.plot(b.x_values,b.pdf_values)

plt.ylim(ymin=0)
plt.xlabel("x")
plt.ylabel("Probability")
plt.savefig("diff.png", dpi=300)
plt.show()

def diff(a, b):
    sum = 0.0
    c = (a.x_limits[1]-a.x_limits[0])/float(a.N)
    for i in a.x_values:
        if (a.pdf(i) >= b.pdf(i)):
            sum += c * b.pdf(i)
        else:
            sum += c * a.pdf(i)
    return sum



area = diff(a, b)
print(area)
