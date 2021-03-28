from soad import AsymmetricData as asyd
import matplotlib.pyplot as plt


# This script is prepared for calculating the difference between
# PDFs (Probability Density Function) of two variable.


a = asyd(20.0, 1.6007810593582121, 1.6007810593582121,N=50000)
b = asyd(20.27602675930529, 1.521759265471423, 1.916585620152389,N=50000)

plt.plot(a.x_values,a.pdf_values, color="deepskyblue")
plt.plot(b.x_values,b.pdf_values, color="tomato")

plt.ylim(ymin=0)
plt.xlabel("x")
plt.ylabel("prob")
plt.savefig("diff.png", dpi=300)

def find_match_percent(a, b):
    sum = 0.0
    c = (a.x_limits[1]-a.x_limits[0])/float(a.N)
    for i in a.x_values:
        if (a.pdf(i) >= b.pdf(i)):
            sum += c * b.pdf(i)
        else:
            sum += c * a.pdf(i)
    return sum

area = find_match_percent(a, b)
print("Matching area ratio is: {:.3}".format(area))

plt.show()
