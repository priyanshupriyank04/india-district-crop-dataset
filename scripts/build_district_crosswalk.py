"""Build crosswalk from data.gov.in crop_production (State_Name, District_Name) to the
india_shp_2020 district polygons (stname, dtname, stcode11, dtcode11, Dist_LGD)."""
import pandas as pd, geopandas as gpd, re, difflib
g = gpd.read_file("data/raw/shapefile/in_district.shp")
g['dtname']=g.dtname.str.strip()
c = pd.read_csv("data/raw/crops/crop_production_datagovin_1997_2015.csv")
def norm(s): return re.sub(r'[^a-z]', '', str(s).lower())
state_alias = {'andamanandnicobarislands':'andamannicobar','dadraandnagarhaveli':'dadranagarhave','jammuandkashmir':'jammukashmir'}
# (State_Name, District_Name) -> (shp stname, shp dtname)
manual = {
 ('Andhra Pradesh','KADAPA'):('ANDHRA PRADESH','Y.S.R.'),
 ('Andhra Pradesh','SPSR NELLORE'):('ANDHRA PRADESH','Sri Potti Sriramulu Nell*'),
 ('Chhattisgarh','DANTEWADA'):('CHHATTISGARH','Dakshin Bastar Dantewada'),
 ('Chhattisgarh','KANKER'):('CHHATTISGARH','Uttar Bastar Kanker'),
 ('Chhattisgarh','KOREA'):('CHHATTISGARH','Koriya'),
 ('Gujarat','DANG'):('GUJARAT','The Dangs'),
 ('Haryana','GURGAON'):('HARYANA','Gurugram'),
 ('Haryana','MEWAT'):('HARYANA','Nuh'),
 ('Jammu and Kashmir ','KARGIL'):('LADAKH','Kargil'),
 ('Jammu and Kashmir ','LEH LADAKH'):('LADAKH','Leh'),
 ('Jammu and Kashmir ','POONCH'):('JAMMU & KASHMIR','Punch'),
 ('Jharkhand','EAST SINGHBUM'):('JHARKHAND','Purbi Singhbhum'),
 ('Jharkhand','WEST SINGHBHUM'):('JHARKHAND','Pashchimi Singhbhum'),
 ('Karnataka','BENGALURU URBAN'):('KARNATAKA','Bangalore'),
 ('Karnataka','GULBARGA'):('KARNATAKA','Kalaburagi'),
 ('Karnataka','MYSORE'):('KARNATAKA','Mysuru'),
 ('Karnataka','BELGAUM'):('KARNATAKA','Belagavi'),
 ('Karnataka','BELLARY'):('KARNATAKA','Ballari'),
 ('Karnataka','BIJAPUR'):('KARNATAKA','Vijayapura'),
 ('Madhya Pradesh','KHANDWA'):('MADHYA PRADESH','East Nimar'),
 ('Madhya Pradesh','KHARGONE'):('MADHYA PRADESH','West Nimar'),
 ('Maharashtra','BEED'):('MAHARASHTRA','Bid'),
 ('Maharashtra','RAIGAD'):('MAHARASHTRA','Raigarh'),
 ('Odisha','SONEPUR'):('ODISHA','Subarnapur'),
 ('Punjab','MUKTSAR'):('PUNJAB','Sri Muktsar Sahib'),
 ('Punjab','NAWANSHAHR'):('PUNJAB','Shahid Bhagat Singh Nagar'),
 ('Punjab','S.A.S NAGAR'):('PUNJAB','Sahibzada Ajit Singh Nag*'),
 ('Tamil Nadu','TUTICORIN'):('TAMIL NADU','Thoothukkudi'),
 ('Telangana ','WARANGAL'):('TELANGANA','Warangal Rural'),
 ('Uttar Pradesh','ALLAHABAD'):('UTTAR PRADESH','Prayagraj'),
 ('Uttar Pradesh','SANT RAVIDAS NAGAR'):('UTTAR PRADESH','Bhadohi'),
 ('Uttarakhand','PAURI GARHWAL'):('UTTARAKHAND','Garhwal'),
 ('West Bengal','24 PARAGANAS NORTH'):('WEST BENGAL','North Twenty Four Pargan*'),
 ('West Bengal','24 PARAGANAS SOUTH'):('WEST BENGAL','South Twenty Four Pargan*'),
 ('West Bengal','BARDHAMAN'):('WEST BENGAL','Purba Bardhaman'),
 ('West Bengal','DINAJPUR DAKSHIN'):('WEST BENGAL','Dakshin Dinajpur'),
 ('West Bengal','DINAJPUR UTTAR'):('WEST BENGAL','Uttar Dinajpur'),
 ('West Bengal','MEDINIPUR EAST'):('WEST BENGAL','Purba Medinipur'),
 ('Gujarat','AHMADABAD'):('GUJARAT','Ahmadabad'),
}
g['st_n']=g.stname.map(norm); g['dt_n']=g.dtname.map(norm)
cd = c[['State_Name','District_Name']].drop_duplicates().reset_index(drop=True)
cd['st_n']=cd.State_Name.map(norm).replace(state_alias); cd['dt_n']=cd.District_Name.map(norm)
out=[]
for _,r in cd.iterrows():
    key=(r.State_Name, r.District_Name); method=None; row=None
    if key in manual:
        st,dt=manual[key]; hit=g[(g.stname==st)&(g.dtname==dt)]; method='manual'
    else:
        hit=g[(g.st_n==r.st_n)&(g.dt_n==r.dt_n)]; method='exact'
        if len(hit)==0:
            sub=g[g.st_n==r.st_n]
            best=difflib.get_close_matches(r.dt_n, sub.dt_n.tolist(), n=1, cutoff=0.75)
            if best: hit=sub[sub.dt_n==best[0]]; method='fuzzy'
    if len(hit)==0: out.append((*key,None,None,None,None,None,'UNMATCHED')); continue
    h=hit.iloc[0]
    out.append((*key,h.stname,h.dtname,h.stcode11,h.dtcode11,h.Dist_LGD,method))
x=pd.DataFrame(out,columns=['State_Name','District_Name','shp_stname','shp_dtname','stcode11','dtcode11','Dist_LGD','match_method'])
x.to_csv("data/processed/district_crosswalk.csv",index=False)
print(x.match_method.value_counts()); print(x[x.match_method=='UNMATCHED']); print(x[x.match_method=='fuzzy'][['District_Name','shp_dtname']].to_string())
