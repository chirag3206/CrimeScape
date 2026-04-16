import requests

url = 'https://raw.githubusercontent.com/geohacker/india/master/state/india_state.geojson'
resp = requests.get(url)
data = resp.json()

names = [f['properties']['NAME_1'] for f in data['features']]
print("Telangana" in names)
print("Andhra Pradesh" in names)
print("Telangana" in str(data))
