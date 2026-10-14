"""Aggregate NASA POWER monthly data per district into season-year features aligned with crop_production seasons.
Crop_Year Y = agricultural year Y/Y+1. Windows (month indices, year offset):
 Kharif: Jun-Oct Y; Rabi: Nov Y - Mar Y+1; Whole Year: Jun Y - May Y+1; Summer: Mar-May Y+1; Autumn: Sep-Nov Y; Winter: Dec Y - Feb Y+1"""
import json, glob, pandas as pd, numpy as np, calendar
SEASONS = {"Kharif":[(6,0),(7,0),(8,0),(9,0),(10,0)], "Rabi":[(11,0),(12,0),(1,1),(2,1),(3,1)],
           "Whole Year":[(m,0) for m in range(6,13)]+[(m,1) for m in range(1,6)],
           "Summer":[(3,1),(4,1),(5,1)], "Autumn":[(9,0),(10,0),(11,0)], "Winter":[(12,0),(1,1),(2,1)]}
rows=[]
for fn in glob.glob("data/raw/climate/power/*.json"):
    d=json.load(open(fn)); meta=d["_district"]; P=d["properties"]["parameter"]
    key=f"{meta['stcode11']}_{meta['dtcode11']}"; elev=d["geometry"]["coordinates"][2]
    def val(par,y,m):
        k=f"{y}{m:02d}"; v=P[par].get(k); return np.nan if v is None or v==-999 else v
    for y in range(1997,2016):
        for s,months in SEASONS.items():
            r={"district_key":key,"Crop_Year":y,"Season":s,"elevation_m":elev,"centroid_lon":meta["lon"],"centroid_lat":meta["lat"]}
            for par,nm in [("T2M","temp_mean_c"),("T2M_MAX","temp_max_c"),("T2M_MIN","temp_min_c"),("RH2M","humidity_pct"),
                           ("ALLSKY_SFC_SW_DWN","solar_mj_m2_day"),("WS2M","wind_m_s")]:
                r[nm]=np.nanmean([val(par,y+o,m) for m,o in months])
            r["rainfall_mm"]=np.nansum([val("PRECTOTCORR",y+o,m)*calendar.monthrange(y+o,m)[1] for m,o in months])
            r["rain_days_proxy_mm_per_day"]=np.nanmean([val("PRECTOTCORR",y+o,m) for m,o in months])
            # annual (agri-year Jun Y - May Y+1) context features
            ann=SEASONS["Whole Year"]
            r["annual_rainfall_mm"]=np.nansum([val("PRECTOTCORR",y+o,m)*calendar.monthrange(y+o,m)[1] for m,o in ann])
            r["annual_temp_mean_c"]=np.nanmean([val("T2M",y+o,m) for m,o in ann])
            r["annual_humidity_pct"]=np.nanmean([val("RH2M",y+o,m) for m,o in ann])
            rows.append(r)
df=pd.DataFrame(rows); df.to_csv("data/processed/climate_features_district_year_season.csv",index=False)
print(df.shape, df.district_key.nunique()); print(df.describe().T)
