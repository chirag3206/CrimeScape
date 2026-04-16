import requests

url = 'https://raw.githubusercontent.com/PratapVardhan/india-geojson/master/india_state.geojson'
resp = requests.get(url)
data = resp.json()

geojson_names = sorted([f['properties']['st_nm'] for f in data['features']])
print("GEOJSON NAMES (PratapVardhan):")
print(geojson_names)
print(f"Total: {len(geojson_names)}")
