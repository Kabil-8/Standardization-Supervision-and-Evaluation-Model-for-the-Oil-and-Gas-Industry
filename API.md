# Prediction API

`POST /api/predict` with JSON:

```json
{
  "years": [{"active_months": 12}, {"active_months": 12}, {"active_months": 12}],
  "static": {"water_depth": 62, "latitude": 28.95, "longitude": -91.01},
  "categorical": {"area_code": "GC", "region_code": "G", "structure_type": "FIXED"}
}
```

All 26 dynamic names, static fields, categorical options, defaults, seeds, and the threshold are available from `GET /api/schema`. Missing numerical values are imputed with the saved training median; unseen category strings use code zero.

`GET /api/health` confirms that all seven seed models loaded.
