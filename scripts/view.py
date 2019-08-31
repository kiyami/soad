import matplotlib.pyplot as plt


class View:
    def plot_pdf(object):
        plt.clf()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(object.x_values, object.pdf_values, color='lightblue', linewidth=2)
        plt.show()

    def plot_cdf(object):
        plt.clf()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(object.x_values, object.cdf(object.x_values), color='lightblue', linewidth=2)
        plt.show()

    def hist_data(object):
        plt.clf()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(object.data, bins=50, density=True, color='blue', alpha=0.6)
        plt.show()