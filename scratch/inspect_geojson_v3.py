import requests

url = 'https://raw.githubusercontent.com/Subhash0/India-State-and-District-GeoJSON/master/India_States.json'
resp = requests.get(url)
data = resp.json()

# Look for Telangana and the property name for states
print("Telangana" in str(data))
print(data['features'][0]['properties'])
