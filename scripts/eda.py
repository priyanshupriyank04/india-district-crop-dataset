"""EDA for the merged India district crop dataset. Writes tables + PNG figures to data/final/eda/ and EDA_SUMMARY.txt."""
import pandas as pd, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, os, textwrap
OUT="data/final/eda"; os.makedirs(OUT, exist_ok=True)
BLUE="#3B6FB6"; GREY="#8A8F98"
plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,"grid.color":"#E5E7EB","grid.linewidth":0.6,"axes.axisbelow":True})
df=pd.read_csv("data/final/india_district_crop_dataset_1997_2015.csv")
kag=pd.read_csv("supplementary_datasets/kaggle_ingle/Crop_recommendation.csv")
lines=[]
def P(s=""): lines.append(s); print(s)

P("="*78); P("EDA SUMMARY: India District Crop Dataset 1997-2015"); P("="*78)
P(f"File: data/final/india_district_crop_dataset_1997_2015.csv")
P(f"Size on disk: {os.path.getsize('data/final/india_district_crop_dataset_1997_2015.csv')/1e6:.1f} MB")
P(f"Rows: {len(df):,}   Columns: {df.shape[1]}   Duplicate rows: {df.duplicated().sum()}")
P(f"Memory in pandas: {df.memory_usage(deep=True).sum()/1e6:.1f} MB")
P(f"Districts: {df.district_key.nunique()}   States/UTs: {df.state.nunique()}   Years: {df.crop_year.min()}-{df.crop_year.max()}   Seasons: {df.season.nunique()}")
P()
P("-"*78); P("1. COLUMN INVENTORY (name, dtype, source, non-null, nulls)"); P("-"*78)
src={}
for c in df.columns:
    if c.startswith("soil_"): src[c]="SoilGrids 2.0 (ISRIC)"
    elif c in ["temp_mean_c","temp_max_c","temp_min_c","humidity_pct","rainfall_mm","rain_days_proxy_mm_per_day","solar_mj_m2_day","wind_m_s","annual_rainfall_mm","annual_temp_mean_c","annual_humidity_pct","elevation_m"]: src[c]="NASA POWER"
    elif c in ["district_key","Dist_LGD","centroid_lat","centroid_lon"]: src[c]="india_shp_2020 shapefile"
    else: src[c]="data.gov.in crop production"
inv=pd.DataFrame({"dtype":df.dtypes.astype(str),"source":pd.Series(src),"non_null":df.notna().sum(),"nulls":df.isna().sum(),"unique":df.nunique()})
inv.to_csv(f"{OUT}/column_inventory.csv")
P(inv.to_string())
P(); P("Columns by source: "+str(pd.Series(src).value_counts().to_dict()))
P()
P("-"*78); P("2. NULL VALUES"); P("-"*78)
dfn=df.drop(columns=["soil_filled_from_nearest"]); nn=dfn.isna().sum(); nn=nn[nn>0]
P(f"(soil_filled_from_nearest is a provenance flag, empty by design except Mumbai; excluded below)")
P(f"Total null cells: {dfn.isna().sum().sum()} of {dfn.size:,} ({dfn.isna().sum().sum()/dfn.size*100:.4f}%)")
P(f"Rows with any null: {dfn.isna().any(axis=1).sum()}")
P("Columns with nulls: "+str(nn.to_dict()))
P("Feature columns (soil/climate/location) null count: "+str(df[[c for c in df.columns if c.startswith('soil_') and c!='soil_filled_from_nearest' or src[c]=='NASA POWER']].isna().sum().sum()))
P("Note: soil_filled_from_nearest is a flag; it is intentionally empty except Mumbai. dominant_yield null = production not reported by DES.")
P()
P("-"*78); P("3. NUMERIC DESCRIPTIVE STATISTICS"); P("-"*78)
num=df.select_dtypes("number").drop(columns=["crop_year","Dist_LGD"])
desc=num.describe().T; desc["skew"]=num.skew(); desc.to_csv(f"{OUT}/numeric_describe.csv")
P(desc.round(3).to_string())
P()
P("-"*78); P("4. TARGET: dominant_crop"); P("-"*78)
vc=df.dominant_crop.value_counts(); vc.to_csv(f"{OUT}/class_counts.csv")
P(f"Distinct classes: {vc.size}; classes with >=100 rows: {(vc>=100).sum()}; >=500: {(vc>=500).sum()}; >=1000: {(vc>=1000).sum()}")
P(f"Majority class share: {vc.iloc[0]/len(df):.3f} ({vc.index[0]}); imbalance ratio top/32nd: {vc.iloc[0]/vc.iloc[31]:.1f}")
P("Top 32 classes:"); P(vc.head(32).to_string())
P(); P("Dominant crop by season (top 5 each):")
for s,g in df.groupby("season"): P(f"  {s:11s} n={len(g):5d}: "+", ".join(f"{k} {v}" for k,v in g.dominant_crop.value_counts().head(5).items()))
P(); P(f"dominant_share: mean {df.dominant_share.mean():.3f}, median {df.dominant_share.median():.3f}; share of rows where dominant crop holds >50% of season area: {(df.dominant_share>0.5).mean():.3f}")
P()
P("-"*78); P("5. CATEGORICAL COVERAGE"); P("-"*78)
P("Rows per season: "+str(df.season.value_counts().to_dict()))
P("Rows per year: "+str(df.crop_year.value_counts().sort_index().to_dict()))
st=df.groupby("state").agg(rows=("district","size"),districts=("district_key","nunique")).sort_values("rows",ascending=False); st.to_csv(f"{OUT}/rows_per_state.csv")
P("Rows and districts per state:"); P(st.to_string())
P()
P("-"*78); P("6. FEATURE CORRELATIONS"); P("-"*78)
feat=[c for c in df.columns if (c.startswith("soil_") and c.endswith("0_30cm")) or src[c]=="NASA POWER"]
corr=df[feat].corr(); corr.to_csv(f"{OUT}/feature_correlation.csv")
pairs=corr.where(np.triu(np.ones_like(corr,dtype=bool),1)).stack().sort_values(key=abs,ascending=False)
P("Strongest absolute correlations (|r|>0.7):"); P(pairs[abs(pairs)>0.7].round(3).to_string())
P("-> expect to drop one of each near-duplicate pair (0_5cm vs 0_30cm depths, temp_mean vs temp_max, rainfall vs rain_days_proxy).")
P()
P("-"*78); P("7. COMPARISON WITH THE KAGGLE (INGLE) DATASET"); P("-"*78)
cmp=pd.DataFrame({"Kaggle":[len(kag),kag.shape[1]-1,kag.label.nunique(),"balanced, 100/class","none","none","augmented/unknown",0],
                  "New":[len(df),"28 (model inputs)",vc.size,f"imbalanced, top={vc.iloc[0]/len(df):.0%}","652 districts, lat/lon","1997-2015, 6 seasons","4 documented public sources","0 in features"]},
                 index=["rows","features","classes","class balance","geography","time","provenance","missing values"])
P(cmp.to_string())
P(); P("Comparable columns: Kaggle ph~soil_ph_h2o_0_30cm, temperature~temp_mean_c, humidity~humidity_pct, rainfall~rainfall_mm, N~soil_total_n_g_kg (different quantity). P and K have no counterpart.")
P(f"pH:        Kaggle {kag.ph.mean():.2f} (sd {kag.ph.std():.2f}) | New {df.soil_ph_h2o_0_30cm.mean():.2f} (sd {df.soil_ph_h2o_0_30cm.std():.2f})")
P(f"Temp C:    Kaggle {kag.temperature.mean():.1f} (sd {kag.temperature.std():.1f}) | New {df.temp_mean_c.mean():.1f} (sd {df.temp_mean_c.std():.1f})")
P(f"Humidity:  Kaggle {kag.humidity.mean():.1f} (sd {kag.humidity.std():.1f}) | New {df.humidity_pct.mean():.1f} (sd {df.humidity_pct.std():.1f})")
P(f"Rainfall:  Kaggle {kag.rainfall.mean():.0f} mm (sd {kag.rainfall.std():.0f}) | New season {df.rainfall_mm.mean():.0f} mm (sd {df.rainfall_mm.std():.0f}), annual {df.annual_rainfall_mm.mean():.0f} mm")
P()
P("-"*78); P("8. HOW THE DATASET WAS BUILT (procedure)"); P("-"*78)
P(textwrap.dedent("""
 Step 0  Environment: Python venv (.venv) with pandas, geopandas, rasterio, rasterstats, scikit-learn.
 Step 1  Crop statistics (label source): data.gov.in "District-wise, season-wise crop production statistics"
         1997-98 to 2014-15 (246,091 rows: State, District, Crop_Year, Season, Crop, Area, Production).
         Downloaded from a public GitHub mirror because the data.gov.in API is rate-limited.
 Step 2  District polygons: india_shp_2020 (GitHub) - 735 districts with Census-2011 and LGD codes.
 Step 3  Soil: ISRIC SoilGrids 2.0, requested through its public WCS service, clipped to the India
         bounding box (68-98E, 6-38N) at 1 km, for 10 properties x 3 depths (0-5, 5-15, 15-30 cm)
         = 30 GeoTIFFs. Script: scripts/fetch_soilgrids.py
 Step 4  Climate: NASA POWER monthly API (no key) at each district's representative point, 1997-2016,
         parameters T2M, T2M_MAX, T2M_MIN, RH2M, PRECTOTCORR, ALLSKY_SFC_SW_DWN, WS2M; elevation from
         the API response. 735 JSON files. Script: scripts/fetch_nasa_power.py
 Step 5  Crosswalk: match the 652 (state, district) names in the crop table to polygons:
         559 exact, 54 fuzzy (spelling), 39 manual (renamed/split districts), 0 unmatched.
         Script: scripts/build_district_crosswalk.py -> data/processed/district_crosswalk.csv
 Step 6  Labels: per district x crop_year x season, exclude plantation crops and aggregate categories,
         compute area share per crop, take the largest as dominant_crop; also top3, multi-label (>=10%),
         yield. Paddy merged into Rice. Script: scripts/build_crop_labels.py -> 34,903 rows
 Step 7  Soil features: zonal mean of every 1 km pixel inside each polygon, per depth; 0-30 cm =
         depth-weighted mean (5:10:15). Values rescaled to physical units (pH, g/kg, %, cmol/kg).
         Mumbai (urban polygon, no pixels) filled from Mumbai Suburban. Script: build_soil_features.py
 Step 8  Climate features: monthly values aggregated into the season window of that crop year
         (Kharif Jun-Oct; Rabi Nov-Mar next yr; Whole Year Jun-May; Summer Mar-May next yr;
         Autumn Sep-Nov; Winter Dec-Feb). Rainfall = sum of mm; others = mean.
         Script: build_climate_features.py
 Step 9  Merge: labels LEFT JOIN soil ON district_key, LEFT JOIN climate ON (district_key, crop_year,
         season). Script: build_merged_dataset.py -> data/final/india_district_crop_dataset_1997_2015.csv
 Step 10 Sanity baseline: RandomForest, 32 classes: random 5-fold 0.824; district-grouped 0.709;
         state-grouped 0.526; majority 0.309.
"""))
P("-"*78); P("9. RECOMMENDATIONS BEFORE MODELLING"); P("-"*78)
P(textwrap.dedent("""
 - Restrict to classes with >=100 (32 classes) or >=500 rows (16 classes) for a stable multi-class task.
 - Use ONLY *_0_30cm soil columns plus the climate columns as model features (drop *_0_5cm duplicates).
 - Never use random k-fold alone: use GroupKFold by district (and report state-grouped and year-split).
 - Do not feed total_area_ha, n_crops, dominant_share, dominant_yield, top3_crops as features: they leak the label.
 - Consider 'Whole Year' season rows separately (perennials such as sugarcane, banana dominate there).
 - Report macro-F1 and per-class recall; accuracy alone is dominated by Rice.
"""))
open("data/final/EDA_SUMMARY.txt","w").write("\n".join(lines))

# ---------- figures ----------
fig,ax=plt.subplots(figsize=(8,7)); top=vc.head(32)[::-1]
ax.barh(top.index,top.values,color=BLUE,height=0.7); ax.set_xlabel("rows (district-year-season)"); ax.set_title("Dominant crop: top 32 classes")
for i,v in enumerate(top.values): ax.text(v+80,i,f"{v:,}",va="center",fontsize=7,color="#374151")
plt.tight_layout(); plt.savefig(f"{OUT}/01_class_distribution.png",dpi=150); plt.close()

fig,axes=plt.subplots(3,4,figsize=(13,8.5)); axes=axes.ravel()
cols=["soil_ph_h2o_0_30cm","soil_total_n_g_kg_0_30cm","soil_soc_g_kg_0_30cm","soil_cec_cmol_kg_0_30cm","soil_clay_pct_0_30cm","soil_sand_pct_0_30cm",
      "temp_mean_c","humidity_pct","rainfall_mm","annual_rainfall_mm","solar_mj_m2_day","elevation_m"]
for a,c in zip(axes,cols):
    a.hist(df[c].dropna(),bins=40,color=BLUE,edgecolor="white",linewidth=0.3); a.set_title(c,fontsize=8); a.tick_params(labelsize=7)
fig.suptitle("Feature distributions (all 34,903 rows)"); plt.tight_layout(); plt.savefig(f"{OUT}/02_feature_histograms.png",dpi=150); plt.close()

fig,ax=plt.subplots(figsize=(11,9)); im=ax.imshow(corr.values,cmap="Blues",vmin=-1,vmax=1)
ax.set_xticks(range(len(feat))); ax.set_xticklabels(feat,rotation=90,fontsize=7); ax.set_yticks(range(len(feat))); ax.set_yticklabels(feat,fontsize=7); ax.grid(False)
plt.colorbar(im,fraction=0.03); ax.set_title("Pearson correlation between model features"); plt.tight_layout(); plt.savefig(f"{OUT}/03_feature_correlation.png",dpi=150); plt.close()

fig,axes=plt.subplots(1,2,figsize=(12,4.5))
df.season.value_counts().plot.bar(ax=axes[0],color=BLUE,width=0.7); axes[0].set_title("Rows per season"); axes[0].tick_params(axis='x',rotation=0)
df.crop_year.value_counts().sort_index().plot.bar(ax=axes[1],color=BLUE,width=0.7); axes[1].set_title("Rows per crop year"); axes[1].tick_params(axis='x',rotation=90)
plt.tight_layout(); plt.savefig(f"{OUT}/04_season_year_coverage.png",dpi=150); plt.close()

top8=vc.head(8).index; fig,axes=plt.subplots(2,3,figsize=(13,7)); axes=axes.ravel()
for a,c in zip(axes,["soil_ph_h2o_0_30cm","soil_clay_pct_0_30cm","temp_mean_c","humidity_pct","rainfall_mm","annual_rainfall_mm"]):
    data=[df.loc[df.dominant_crop==k,c].dropna().values for k in top8]
    a.boxplot(data,labels=[k[:10] for k in top8],showfliers=False,medianprops=dict(color=BLUE,linewidth=2),boxprops=dict(color=GREY),whiskerprops=dict(color=GREY),capprops=dict(color=GREY))
    a.set_title(c,fontsize=9); a.tick_params(axis='x',rotation=45,labelsize=7)
fig.suptitle("Feature ranges by dominant crop (top 8 classes)"); plt.tight_layout(); plt.savefig(f"{OUT}/05_features_by_crop.png",dpi=150); plt.close()

fig,ax=plt.subplots(figsize=(7,7.5)); d1=df.drop_duplicates("district_key")
sc=ax.scatter(d1.centroid_lon,d1.centroid_lat,c=d1.annual_rainfall_mm,cmap="Blues",s=14,edgecolors="none"); ax.set_aspect("equal"); ax.grid(False)
plt.colorbar(sc,label="mean annual rainfall (mm), NASA POWER",fraction=0.04); ax.set_title("652 district centroids in the dataset"); plt.tight_layout(); plt.savefig(f"{OUT}/06_district_map.png",dpi=150); plt.close()

df.head(20).to_csv(f"{OUT}/sample_first_20_rows.csv",index=False)
df.sample(20,random_state=1).to_csv(f"{OUT}/sample_random_20_rows.csv",index=False)
print("\nFigures + tables written to", OUT)
