import requests

url = 'https://raw.githubusercontent.com/geohacker/india/master/state/india_state.geojson'
resp = requests.get(url)
data = resp.json()

geojson_names = set([f['properties']['NAME_1'] for f in data['features']])
data_names = set(['Andaman & Nicobar Islands', 'Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar', 'Chandigarh', 'Chhattisgarh', 'Dadra & Nagar Haveli and Daman & Diu', 'Delhi (UT)', 'Goa', 'Gujarat', 'Haryana', 'Himachal Pradesh', 'Jammu & Kashmir', 'Jharkhand', 'Karnataka', 'Kerala', 'Ladakh', 'Lakshadweep', 'Madhya Pradesh', 'Maharashtra', 'Manipur', 'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha', 'Puducherry', 'Punjab', 'Rajasthan', 'Sikkim', 'Tamil Nadu', 'Telangana', 'Tripura', 'Uttar Pradesh', 'Uttarakhand', 'West Bengal'])

print("NAME MISMATCH ANALYSIS:")
print("-" * 30)

missing_in_geo = data_names - geojson_names
print(f"Data Names needing mapping ({len(missing_in_geo)}):")
for name in sorted(list(missing_in_geo)):
    print(f" - {name}")

print("\nPotential GeoJSON Candidates:")
for name in sorted(list(geojson_names)):
    if name not in data_names:
        print(f" ? {name}")
