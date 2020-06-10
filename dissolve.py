import csv
import geopandas as gpd
import numpy as np
import os
import pandas as pd

def dissolve_and_save(gdf, col_key, output_fname='output.shp'):
	"""Dissolves the given GeoDataFrame using the given column key and
	   saves the result to a shapefile."""
	districts = gdf.dissolve(by=col_key)
	# import pdb; pdb.set_trace()
	districts.to_file(output_fname)


if __name__ == '__main__':
	# Dissolve WI maps to show only districts
	HOME_DIR = os.path.expanduser("~")
	DATA_DIR = '{0}/{1}'.format(HOME_DIR, 'Documents/Data') 
	tracts_fpath = 'zip://{0}/{1}'.format(DATA_DIR, 'Census/Wisconsin/tl_2013_55_tract.zip')
	gdf = gpd.read_file(tracts_fpath)
	gdf = gdf.set_index('GEOID')

	pop_fpath = '{0}/Census/Wisconsin/2010_CensusTractPopulations/DEC_10_SF1_P1_with_ann_modified.csv'.format(DATA_DIR)

	with open(pop_fpath, 'r') as pop_file:
		pop_reader = csv.reader(pop_file)
		pop_raw = list(pop_reader)

	pop_headers = pop_raw.pop(0)
	pop_df = pd.DataFrame(pop_raw, columns=pop_headers).astype({"POP": int})
	pop_df = pop_df.drop(columns=['GEOIDLONG', 'DISPLAYNAME'])
	pop_df = pop_df.set_index('GEOID')
	gdf = gdf.join(pop_df)
	# import pdb; pdb.set_trace()

	map_a_filepath = '{0}/ICOR/Wisconsin/wi-gerrymander-dem.csv'.format(DATA_DIR)
	map_b_filepath = '{0}/ICOR/Wisconsin/wi-gerrymander-rep.csv'.format(DATA_DIR)

	for map_fpath in [map_a_filepath, map_b_filepath]:
		with open(map_fpath) as map_file:
			map_reader = csv.reader(map_file)
			map_raw = list(map_reader)

		map_headers = map_raw.pop(0)
		map_df = pd.DataFrame(map_raw, columns=map_headers)
		map_df = map_df.set_index('GEOID')

		map_gdf = gdf.join(map_df)
		output_fname = '{0}-districts.shp'.format(map_fpath[-22:-4])
		dissolve_and_save(map_gdf, 'district', output_fname)