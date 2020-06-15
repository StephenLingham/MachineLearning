import requests
import json

url = 'https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&starttime=2020-06-14'

response = requests.get(url)

#print(response.content)

jsonData = json.loads(response.content)
for feature in jsonData["features"]:
    print(feature["properties"]["mag"])

