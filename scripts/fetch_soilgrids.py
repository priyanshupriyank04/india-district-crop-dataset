"""Download SoilGrids 2.0 mean layers for India bbox at 1 km via ISRIC WCS."""
import os, subprocess, time
OUT="data/raw/soilgrids"; os.makedirs(OUT, exist_ok=True)
props = ["phh2o","nitrogen","soc","cec","clay","sand","silt","bdod","ocd","cfvo"]
depths = ["0-5cm","5-15cm","15-30cm"]
for p in props:
    for d in depths:
        fn=f"{OUT}/india_{p}_{d}_1km.tif"
        if os.path.exists(fn) and os.path.getsize(fn)>100000: continue
        url=(f"https://maps.isric.org/mapserv?map=/map/{p}.map&SERVICE=WCS&VERSION=2.0.1&REQUEST=GetCoverage"
             f"&COVERAGEID={p}_{d}_mean&FORMAT=image/tiff&SUBSET=long(68,98)&SUBSET=lat(6,38)"
             f"&SUBSETTINGCRS=http://www.opengis.net/def/crs/EPSG/0/4326&OUTPUTCRS=http://www.opengis.net/def/crs/EPSG/0/4326"
             f"&SCALESIZE=long(3000),lat(3200)")
        for a in range(4):
            r=subprocess.run(["curl","-sS","-m","900","-o",fn,url])
            if os.path.exists(fn) and os.path.getsize(fn)>100000:
                print("ok",fn,os.path.getsize(fn),flush=True); break
            time.sleep(10)
print("DONE",flush=True)
