from scipy import stats
import matplotlib.pyplot as plt
import requests
import json
import pandas
from sklearn import linear_model
from linearregression import get_rsquared_coefficient as r2, display_graph
import multipleregression

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

#display_graph(magnitudes, latitudes)
r2Lat = r2(magnitudes, latitudes)
r2Long = r2(magnitudes, longitudes)
r2Depth = r2(magnitudes, depths)

print("R Squared coefficient for latitude correlated with magnitude: ")
print(r2Lat)
print("R Squared coefficient for longitude correlated with magnitude: ")
print(r2Long)
print("R Squared coefficient for depths correlated with magnitude: ")
print(r2Depth)

df = pandas.read_csv("cars.csv")



X = df[['Weight', 'Volume']]
y = df['CO2']
# test = y.to_numpy()
# print(type(test))
# test2 = []
# print(type(test2))
# test3 = y.tolist()
# print(type(test3))
#print(type(X))
#print(X)
# for item in X:
#     print(item[0])

#print(X["Weight"].tolist())



dependentVariable = y.tolist()
# print("dependant variable")
# print(dependantVariable)
# print(len(dependantVariable))

independentVariable1 = X["Weight"].tolist()
# print("independant variable 1")
# print(independantVariable1)
# print(len(independantVariable1))

independentVariable2 = X["Volume"].tolist()
# print("independant variable 2")
# print(independantVariable2)
# print(len(independantVariable2))

result = multipleregression.get_rsquared_coefficient(dependentVariable, independentVariable1, independentVariable2)
print("\nR squared coefficient for multiple regression:")
print(result[0])

print("Adjusted R squared coefficient for multiple regression:")
print(result[1])
