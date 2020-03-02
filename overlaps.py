import csv
import geopandas as gpd
import numpy as np
import os
import pandas as pd

from gerrychain import Graph, Partition

######################################################################
# Author: Ian Ludden
# Date:   29 Feb 2020
# 
# Class and functions for examining the overlap between 
# two district maps. 
######################################################################

class MapMerger(object):
	"""Manages the merging of two district maps drawn at the same 
	level of granularity as the given shapefile."""
	def __init__(self, units_shapefile_path, key='GEOID'):
		units_gdf = gpd.read_file(units_shapefile_path)
		self.key = key
		self.units_gdf = units_gdf.set_index(self.key)

	def merge_maps(self, map_a_df, map_b_df):
		"""Given two district maps represented as 
		pandas DataFrame objects,  
		joins both with the units_gdf GeoDataFrame 
		to add columns 'district_a' and 'district_b'.
		Returns the joined GeoDataFrame."""
		joined_a_df = self.units_gdf.join(map_a_df, rsuffix='_a')
		return joined_a_df.join(map_b_df, rsuffix='_b')

		
	def read_map(self, map_filepath):
		"""Reads a district map from a csv file with 
		columns 'GEOID' and 'district'. 
		Returns a pandas DataFrame object representing the map, where
		'GEOID' is the key."""
		with open(map_filepath) as map_file:
			map_reader = csv.reader(map_file)
			map_raw = list(map_reader)

		map_headers = map_raw.pop(0)
		map_df = pd.DataFrame(map_raw, columns=map_headers)
		return map_df.set_index(self.key)

		
if __name__ == '__main__':
	HOME_DIR = os.path.expanduser("~")
	DATA_DIR = '{0}/{1}'.format(HOME_DIR, 'Documents/Data') 
	tracts_fpath = 'zip://{0}/{1}'.format(DATA_DIR, 'Census/Wisconsin/tl_2013_55_tract.zip')
	
	merger = MapMerger(tracts_fpath)

	map_a_filepath = '{0}/{1}'.format(DATA_DIR, 'ICOR/Wisconsin/icor-wi-01.csv')
	map_b_filepath = '{0}/{1}'.format(DATA_DIR, 'ICOR/Wisconsin/icor-wi-02.csv')
	
	map_a_df = merger.read_map(map_a_filepath)
	map_b_df = merger.read_map(map_b_filepath)

	import pdb; pdb.set_trace()
	merged_map_a_b_df = merger.merge_maps(map_a_df, map_b_df)

	merged_map_a_b_df.to_file("merged-maps-01-and-02.shp")