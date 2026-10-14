"""From data.gov.in district crop production (1997-2015) build:
 - crop_long: one row per district-year-season-crop with area, production, yield, area_share
 - labels: one row per district-year-season with dominant crop (by area share), top-3 crops, and multi-label list."""
import pandas as pd, numpy as np
c = pd.read_csv("data/raw/crops/crop_production_datagovin_1997_2015.csv")
xw = pd.read_csv("data/processed/district_crosswalk.csv", dtype={'stcode11':str,'dtcode11':str})
c['Season']=c.Season.str.strip(); c['Crop']=c.Crop.str.strip()
c = c.merge(xw, on=['State_Name','District_Name'], how='left')
assert c.shp_dtname.notna().all()
c['district_key']=c.stcode11+'_'+c.dtcode11
c = c[(c.Area>0)].copy()
c['Production']=pd.to_numeric(c.Production, errors='coerce')
c['yield_t_per_ha']=c.Production/c.Area
# exclude non-field / plantation / minor categories that are not "crop recommendation" targets
exclude = {'Coconut','Arecanut','Cashewnut','Rubber','Tea','Coffee','Other  Rabi pulses','Other Kharif pulses','other oilseeds',
           'Other Cereals & Millets','other misc. pulses','Other Fresh Fruits','Other Vegetables','Other Citrus Fruit','Other Dry Fruit',
           'Pulses total','Total foodgrain','Oilseeds total','Cereals total','other fibres','Other Summer Pulses','Peas & beans (Pulses)',
           'Beans & Mutter(Vegetable)','Citrus Fruit','Fruits total','Vegetables total','Condiments and spices total','Other misc. pulses'}
c['is_excluded']=c.Crop.isin(exclude)
tot = c[~c.is_excluded].groupby(['district_key','Crop_Year','Season']).Area.transform('sum')
c['area_share']=np.where(c.is_excluded, np.nan, c.Area/tot)
c.to_csv("data/processed/crop_long_1997_2015.csv", index=False)
f = c[~c.is_excluded].copy()
f = f.sort_values(['district_key','Crop_Year','Season','area_share'], ascending=[True,True,True,False])
grp = f.groupby(['district_key','Crop_Year','Season'])
lab = grp.agg(shp_stname=('shp_stname','first'), shp_dtname=('shp_dtname','first'), Dist_LGD=('Dist_LGD','first'),
              total_area_ha=('Area','sum'), n_crops=('Crop','nunique'),
              dominant_crop=('Crop','first'), dominant_share=('area_share','first'),
              dominant_yield=('yield_t_per_ha','first')).reset_index()
top3 = grp.Crop.apply(lambda s: '|'.join(s.head(3))).rename('top3_crops').reset_index()
multi = f[f.area_share>=0.10].groupby(['district_key','Crop_Year','Season']).Crop.apply(lambda s:'|'.join(s)).rename('crops_share_ge_10pct').reset_index()
lab = lab.merge(top3, on=['district_key','Crop_Year','Season']).merge(multi, on=['district_key','Crop_Year','Season'], how='left')
lab.to_csv("data/processed/crop_labels_district_year_season.csv", index=False)
print(lab.shape); print(lab.Season.value_counts()); print(lab.dominant_crop.value_counts().head(25)); print("districts:", lab.district_key.nunique())
