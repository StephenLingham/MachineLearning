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

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.linregress.html
# returns the correlation coefficient (r, not r squared)
def get_r_coefficient(xPoints, yPoints):
    slope, intercept, r, p, stdErr = stats.linregress(xPoints, yPoints)
    return r

# returns the coefficient of determination (r squared)
def get_rsquared_coefficient(xPoints, yPoints):
    slope, intercept, r, p, stdErr = stats.linregress(xPoints, yPoints)
    return r**2
