import cdsapi
import numpy as np
import pandas as pd
import pygrib
import pdb
import math

site_ids = ['nhdhr_143249470']
metadata = pd.read_csv('../../metadata/lake_metadata.csv')
metadata.set_index('site_id',inplace=True)
pdb.set_trace()
factor = 2 
for site_id in site_ids:

    lat = metadata.loc[site_id]['lake_lat_deg']
    lon = metadata.loc[site_id]['lake_lon_deg']
    lat_upper = math.ceil(lat * factor) / factor
    lat_lower = math.floor(lat * factor) / factor
    lon_upper = math.ceil(lon * factor) / factor
    lon_lower = math.floor(lon * factor) / factor
    c = cdsapi.Client()

    fn = 'download.grib'

    c.retrieve(
        'reanalysis-era5-land',
        {
            'format': 'grib',
            'variable': [
                'lake_mix_layer_temperature',
            ],
            'year': '2020',
            'month': '07',
            'day': '01',
            'time': '00:00',
            'area': [
                lat_upper, lon_upper, lat_lower,
                lon_lower
            ],
        },
        fn)

    gr = pygrib.open(fn)
    for g in gr:
        pdb.set_trace()
        print(g)
        print(g.values)