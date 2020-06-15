import requests
url = 'https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&starttime=2020-06-10'

response = requests.get(url)

print(response.content)
