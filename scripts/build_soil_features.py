"""Zonal mean of SoilGrids 1 km layers per district polygon; average 0-5,5-15,15-30 cm into 0-30 cm (depth-weighted)."""
import geopandas as gpd, pandas as pd, numpy as np, rasterio, os
from rasterstats import zonal_stats
g = gpd.read_file("data/raw/shapefile/in_district.shp"); g['dtname']=g.dtname.str.strip()
g['district_key']=g.stcode11+'_'+g.dtcode11
props = {"phh2o":("ph_h2o",0.1),"nitrogen":("total_n_g_kg",0.01),"soc":("soc_g_kg",0.1),"cec":("cec_cmol_kg",0.1),
         "clay":("clay_pct",0.1),"sand":("sand_pct",0.1),"silt":("silt_pct",0.1),"bdod":("bulk_density_kg_dm3",0.01),
         "ocd":("org_c_density_kg_m3",0.1),"cfvo":("coarse_frag_pct",0.1)}
depths = {"0-5cm":5,"5-15cm":10,"15-30cm":15}
out = g[['district_key','stname','dtname','Dist_LGD']].copy()
for p,(name,scale) in props.items():
    acc = np.zeros(len(g)); wacc = np.zeros(len(g))
    for d,w in depths.items():
        fn=f"data/raw/soilgrids/india_{p}_{d}_1km.tif"
        if not os.path.exists(fn): print("missing",fn); continue
        zs = zonal_stats(g, fn, stats=["mean"], nodata=0, all_touched=True)
        vals = np.array([z["mean"] if z["mean"] is not None else np.nan for z in zs])
        ok = ~np.isnan(vals)
        acc[ok] += vals[ok]*w; wacc[ok] += w
        if d=="0-5cm": out[f"soil_{name}_0_5cm"]=vals*scale
    with np.errstate(invalid='ignore', divide='ignore'):
        out[f"soil_{name}_0_30cm"]=np.where(wacc>0, acc/np.where(wacc>0,wacc,1), np.nan)*scale
# districts with no soil pixels (tiny/urban polygons): fill from nearest district polygon
soil_cols=[c for c in out.columns if c.startswith('soil_')]
miss = out[soil_cols].isna().any(axis=1)
if miss.any():
    have = g[~miss.values]; cent = have.geometry.centroid
    for i in out.index[miss]:
        j = cent.distance(g.geometry.iloc[i].centroid).idxmin()
        out.loc[i, soil_cols] = out.loc[j, soil_cols].values
        out.loc[i, 'soil_filled_from_nearest'] = g.dtname.iloc[j]
        print("filled", g.dtname.iloc[i], "from", g.dtname.iloc[j])
out['soil_filled_from_nearest']=out.get('soil_filled_from_nearest', pd.Series([None]*len(out))).fillna('')
out.to_csv("data/processed/soil_features_district.csv", index=False)
print(out.describe().T)
