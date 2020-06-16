from scipy import stats
import matplotlib.pyplot as plt

def display_graph(xPoints, yPoints):
    slope, intercept, r, p, stdErr = stats.linregress(xPoints, yPoints)

    def myfunc(x):
        return slope * x + intercept

    mymodel = list(map(myfunc, xPoints))

    plt.scatter(xPoints, yPoints)
    plt.plot(xPoints, mymodel)
    plt.show()

def get_rsquared_coefficient(xPoints, yPoints):
    slope, intercept, r, p, stdErr = stats.linregress(xPoints, yPoints)
    return r
