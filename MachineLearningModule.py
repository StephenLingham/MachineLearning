from scipy import stats
import matplotlib.pyplot as plt
import requests
import json

url = 'https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&starttime=2020-05-30'

response = requests.get(url)

magnitudes = []
latitudes = []
longitudes = []
depths = []
jsonData = json.loads(response.content)
for feature in jsonData["features"]:
    magnitudes.append(feature["properties"]["mag"])
    latitudes.append(feature["geometry"]["coordinates"][0])
    longitudes.append(feature["geometry"]["coordinates"][1])
    depths.append(feature["geometry"]["coordinates"][2])

# print("Magnitudes: ")
# print(magnitudes)
# print("")
# print("Latitudes: ")
# print(latitudes)
# print()
print(longitudes)

slopeLat, interceptLat, rLat, pLat, stdErrLat = stats.linregress(
    magnitudes, latitudes)
slopeLong, interceptLong, rLong, pLong, stdErrLong = stats.linregress(
    magnitudes, longitudes)
slopeDepth, interceptDepth, rDepth, pDepth, stdErrDepth = stats.linregress(
    magnitudes, depths)

def myfunc(x):
    return slopeLat * x + interceptLat

mymodel = list(map(myfunc, magnitudes))

plt.scatter(magnitudes, latitudes)
plt.plot(magnitudes, mymodel)
plt.show()

print("R Squared coefficient for latitude correlated with magnitude: ")
print(rLat)
print("R Squared coefficient for longitude correlated with magnitude: ")
print(rLong)
print("R Squared coefficient for depths correlated with magnitude: ")
print(rDepth)

# import pandas
# from sklearn import linear_model

# df = pandas.read_csv("cars.csv")

# X = df[['Weight', 'Volume']]
# y = df['CO2']

# regr = linear_model.LinearRegression()
# regr.fit(X, y)

# #predict the CO2 emission of a car where the weight is 2300kg, and the volume is 1300ccm:
# predictedCO2 = regr.predict([[2300, 1300]])

# print(predictedCO2) 

# print(regr.coef_) 

# import pandas
# from sklearn import linear_model

# X = [longitudes, latitudes]
# y = magnitudes

# regr = linear_model.LinearRegression()
# regr.fit(X, y)

# print(regr.coef_) 