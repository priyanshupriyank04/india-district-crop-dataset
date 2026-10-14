"""Fetch NASA POWER monthly T2M, T2M_MAX, T2M_MIN, RH2M, PRECTOTCORR, ALLSKY_SFC_SW_DWN, WS2M
for each district centroid (representative point), 1997-2015 (+2016 spillover), save JSON per district."""
import json, os, time, sys
import geopandas as gpd, requests
OUT = "data/raw/climate/power"
os.makedirs(OUT, exist_ok=True)
g = gpd.read_file("data/raw/shapefile/in_district.shp")
pts = g.geometry.representative_point()
params = "T2M,T2M_MAX,T2M_MIN,RH2M,PRECTOTCORR,ALLSKY_SFC_SW_DWN,WS2M"
done = 0
for i, (idx, row) in enumerate(g.iterrows()):
    key = f"{row['stcode11']}_{row['dtcode11']}"
    fn = f"{OUT}/{key}.json"
    if os.path.exists(fn):
        continue
    lon, lat = pts.iloc[i].x, pts.iloc[i].y
    url = (f"https://power.larc.nasa.gov/api/temporal/monthly/point?parameters={params}"
           f"&community=AG&longitude={lon:.4f}&latitude={lat:.4f}&start=1997&end=2016&format=JSON")
    for attempt in range(5):
        try:
            r = requests.get(url, timeout=120)
            if r.status_code == 200:
                d = r.json()
                d["_district"] = {"stname": row["stname"], "dtname": row["dtname"], "stcode11": row["stcode11"],
                                  "dtcode11": row["dtcode11"], "lon": lon, "lat": lat}
                json.dump(d, open(fn, "w"))
                break
            elif r.status_code == 429:
                time.sleep(30)
            else:
                time.sleep(5)
        except Exception as e:
            time.sleep(10)
    done += 1
    if done % 25 == 0:
        print(f"{i+1}/{len(g)} fetched", flush=True)
print("DONE", flush=True)
