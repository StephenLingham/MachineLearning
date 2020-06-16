from sklearn import linear_model
import json
import requests
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import numpy as np


url = 'https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&starttime=2020-06-10'


response = requests.get(url)

quakeInfo = ["magnitude", "latitude", "longitude", "depth"]
allQuakes = []

jsonData = json.loads(response.content)
for feature in jsonData["features"]:
    singleQuake = []
    singleQuake.append(feature["properties"]["mag"])
    singleQuake.append(feature["geometry"]["coordinates"][0])
    singleQuake.append(feature["geometry"]["coordinates"][1])
    singleQuake.append(feature["geometry"]["coordinates"][2])
    allQuakes.append(singleQuake)


df = pd.DataFrame(data=allQuakes, columns=quakeInfo)
X = df[["latitude", "longitude"]]
y = df["magnitude"]

regr = linear_model.LinearRegression()
regr.fit(X, y)


def predict_magnitude(lat, long):
    predicted = regr.predict([[lat, long]])
    return predicted


# print(predict_magnitude(-110, 32))

# TESTINGTESTING123

testingUrl = "https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&starttime=2016-06-01&endtime=2016-06-05"


testingResponse = requests.get(testingUrl)

quakeInfoTesting = ["magnitude", "latitude", "longitude",
                    "depth", "predicted mag", "difference"]
allQuakesTesting = []

jsonDataTesting = json.loads(testingResponse.content)

for feature in jsonDataTesting["features"]:
    singleQuake = []
    singleQuake.append(feature["properties"]["mag"])
    singleQuake.append(feature["geometry"]["coordinates"][0])
    singleQuake.append(feature["geometry"]["coordinates"][1])
    singleQuake.append(feature["geometry"]["coordinates"][2])
    allQuakesTesting.append(singleQuake)


for testingQuake in allQuakesTesting:
    [guessMag] = predict_magnitude(testingQuake[1], testingQuake[2])

    testingQuake.append(guessMag)
    diffInGuess = testingQuake[0] - guessMag
    testingQuake.append(diffInGuess)


dfTest = pd.DataFrame(data=allQuakesTesting, columns=quakeInfoTesting)

print(dfTest)
