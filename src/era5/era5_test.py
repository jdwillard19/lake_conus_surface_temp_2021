import cdsapi
import numpy as np
import pandas as pd
import pygrib

site_ids = ['nhdhr_']

metadata = pd.read_csv('../../metadata/lake_metadata.csv')
metadata.set_index('site_id',inplace=True)
pdb.set_trace()
for site_id in site_ids:


    c = cdsapi.Client()

    fn = 'download.grib'

    c.retrieve(
        'reanalysis-era5-land',
        {
            'format': 'grib',
            'variable': [
                'lake_mix_layer_temperature', 'skin_temperature',
            ],
            'year': '2020',
            'month': '07',
            'day': '01',
            'time': '00:00',
            'area': [
                45, -93.21, 44.99,
                -93.22,
            ],
        },
        fn)

    gr = pygrib.open(fn)
    for g in gr:
        print(g)
        print(g.values)