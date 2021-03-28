from soad import uncertainty as asym
import matplotlib.pyplot as plt

a = asym.AsymmetricData(20.0, 1.6007810593582121, 1.6007810593582121,N=100000)
b = asym.AsymmetricData(20.27602675930529, 1.521759265471423, 1.916585620152389,N=100000)


plt.plot(a.x_values,a.pdf_values, color="deepskyblue")
plt.plot(b.x_values,b.pdf_values, color="tomato")

plt.ylim(ymin=0)
plt.xlabel("x")
plt.ylabel("prob")
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
