import os
import sys
import json
from fastapi.testclient import TestClient

# Add app dir to path
sys.path.insert(0, r"D:\projects\solar\app")
from server import app

client = TestClient(app)

print("1. Testing GET /api/presets...")
res = client.get("/api/presets")
assert res.status_code == 200, f"Presets failed: {res.text}"
presets = res.json()["presets"]
print(f"   Success! Found {len(presets)} presets:")
for p in presets:
    print(f"   - {p['filename']} ({p['title']})")

print("\n2. Testing POST /api/predict with preset_1_residential_dense.jpg...")
res_pred = client.post("/api/predict", data={
    "preset_filename": "preset_1_residential_dense.jpg",
    "confidence_threshold": 0.35,
    "min_area": 5
})
assert res_pred.status_code == 200, f"Prediction failed: {res_pred.text}"
pred_data = res_pred.json()
print(f"   Inference Success in {pred_data['inference_time_ms']} ms!")
print(f"   - Total Arrays:   {pred_data['total_arrays']}")
print(f"   - Total Area:     {pred_data['total_area_m2']} m2")
print(f"   - Est. Capacity:  {pred_data['total_capacity_kw']} kW")
print(f"   - Annual Yield:   {pred_data['annual_yield_kwh']} kWh/yr")
print(f"   - Images returned: {list(pred_data['images'].keys())}")

print("\n3. Testing GET /api/export_geojson...")
res_geo = client.get("/api/export_geojson")
assert res_geo.status_code == 200, f"GeoJSON failed: {res_geo.text}"
geo_data = res_geo.json()
print(f"   GeoJSON Success! Features count: {len(geo_data['features'])}")

print("\n4. Testing GET / (Static Frontend)...")
res_html = client.get("/")
assert res_html.status_code == 200
assert "Solar PV Detection Engine" in res_html.text
print("   Frontend HTML loaded successfully!")

print("\nAll verification tests passed with 100% success!")
