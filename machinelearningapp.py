from scipy import stats
import matplotlib.pyplot as plt
import requests
import json
import pandas
from sklearn import linear_model
from linearregression import get_rsquared_coefficient as r2, display_graph

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

display_graph(magnitudes, latitudes)
r2Lat = r2(magnitudes, latitudes)
r2Long = r2(magnitudes, longitudes)
r2Depth = r2(magnitudes, depths)

print("R Squared coefficient for latitude correlated with magnitude: ")
print(r2Lat)
print("R Squared coefficient for longitude correlated with magnitude: ")
print(r2Long)
print("R Squared coefficient for depths correlated with magnitude: ")
print(r2Depth)

hihi
