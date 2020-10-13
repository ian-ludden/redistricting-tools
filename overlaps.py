import csv
import geopandas as gpd
import networkx as nx
import numpy as np
import os
import pandas as pd

from gerrychain import Graph, Partition


class MapMerger(object):
	"""Manages the merging of two district maps drawn at the same 
	level of granularity as the given shapefile."""
	def __init__(self, units_shapefile_path=None, units_gdf_given=None, key='GEOID', populations_file_path=None):
		if units_shapefile_path is not None:
			units_gdf = gpd.read_file(units_shapefile_path)
		elif units_gdf_given is None:
			raise ValueError('At least one of units_shapefile_path and units_gdf must be given.')
		else:
			units_gdf = units_gdf_given.copy()
		self.key = key
		self.units_gdf = units_gdf.reset_index()
		self.units_gdf = units_gdf.set_index(self.key)
		self.merged_gdf = None
		self.pop_df = None
		if populations_file_path is not None:
			with open(populations_file_path, 'r') as pop_file:
				pop_reader = csv.reader(pop_file)
				pop_raw = list(pop_reader)

			pop_headers = pop_raw.pop(0)
			self.pop_df = pd.DataFrame(pop_raw, columns=pop_headers).astype({"population": int})
			self.pop_df = self.pop_df.drop(columns=['GEOIDLONG', 'DISPLAYNAME'])
			self.pop_df = self.pop_df.set_index('GEOID')


	def merge_maps(self, map_a_df, map_b_df):
		"""Given two district maps represented as 
		pandas DataFrame objects,  
		joins both with the units_gdf GeoDataFrame 
		to add columns 'district_a' and 'district_b'.
		Returns the joined GeoDataFrame."""
		joined_a_df = self.units_gdf.join(map_a_df)
		joined_a_df = joined_a_df.rename(columns={'district': 'district_a'})
		joined_df = joined_a_df.join(map_b_df)
		joined_df = joined_df.rename(columns={'district': 'district_b'})
		self.merged_gdf = joined_df # Save latest merge result
		return joined_df


	def reference_zones(self, map_a_df=None, map_b_df=None, property_name=None):
		"""Compute the reference zones for map A in reference to map B
		   using the definition of Pereira et al. (2009). 

		   The default property, used when property_name is None, 
		   is the number of units, i.e., p_i := 1 for every unit i. 
		   Other supported property_name values include:
		    - 'population': p_i := the population of unit i

		   Returns a list ref_zones representing the mapping of 
		   zones in map A to zones in map B, i.e., 
		   ref_zones[a] = R_B(a), where a is a zone in A.
		   Note ref_zones[0] is unused. 
		   Also returns a matrix of overlap amounts. 
		"""
		# Merge the two given maps if necessary. 
		if map_a_df is not None and map_b_df is not None:
			self.merge_maps(map_a_df, map_b_df)

		# Drop unnecessary columns and handle NaN entries.  
		df = pd.DataFrame(self.merged_gdf.drop(columns=['geometry', 'INTPTLON', 'INTPTLAT']))
		df = df.fillna(value=-1).astype({'district_a': 'int32', 'district_b': 'int32'})

		# Add property column, if missing. 
		if property_name is not None and not (property_name in df.columns):
			if property_name == 'population':
				df = df.join(self.pop_df)
			else:
				raise NotImplementedError('Property name \'{0}\' is not yet supported.'.format(property_name))

		# Overview:
		# 1. Extract numbers of zones from the DataFrame. 
		# 2. For each zone a in A:
		# 3.     For each zone b in B:
		# 4.         Count how many units match both zones.
		# 5.         Update ref_zones[a] if b has more overlap than previous max.
		# Note: Can probably do some clever pandas trick to 
		#       avoid the nested for loop. 
		num_zones_a = df.max().district_a
		num_zones_b = df.max().district_b

		ref_zones = np.arange(num_zones_a + 1)

		overlap_counts = np.zeros((num_zones_a + 1, num_zones_b + 1))

		for a in range(1, num_zones_a + 1):
			max_overlap = -1
			for b in range(1, num_zones_b + 1):
				rows = df.loc[(df['district_a'] == a) & (df['district_b'] == b)]
				
				overlap_counts[a, b] = self.sum_property(rows, property_name=property_name)

				if len(rows.index) > max_overlap:
					max_overlap = len(rows.index)
					ref_zones[a] = b
		
		return ref_zones, overlap_counts[1:, 1:]


	def reference_zones_by_max_wt_matching(self, overlap_counts):
		"""
		Determines reference zones by computing a maximum weight bipartite matching 
		given a matrix (k x k numpy array) of district overlaps.
		"""
		k = len(overlap_counts)
		G = nx.Graph()
		for map_name in ['A', 'B']:
			for i in range(1, k + 1):
				G.add_node('A{0}'.format(i))
				G.add_node('B{0}'.format(i))

		for i in range(k):
			u = 'A{0}'.format(i + 1)
			for j in range(k):
				v = 'B{0}'.format(j + 1)
				# Negate weights to use Networkx implementation of minimum_weight_full_matching
				G.add_edge(u, v, weight=-overlap_counts[i, j])

		M = nx.bipartite.minimum_weight_full_matching(G)

		ref_zones = [0]
		for i in range(k):
			u = 'A{0}'.format(i + 1) # 'Ai'
			v = M[u] # 'Bj', neighbor of 'Ai' in optimal matching
			ref_zones.append(int(v[1])) # j, as int

		return ref_zones


	def sum_property(self, df, property_name=None):
		if property_name is None:
			return len(df.index)
		elif property_name in df.columns:
			return df[property_name].sum()
		else:
			raise NotImplementedError('Property name \'{0}\' is not among the columns.'.format(property_name))


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


	def read_merge_and_save(self, map_a_filepath, map_b_filepath, output_filename='merged-maps.shp'):
		"""Reads the maps at the given filepaths, merges them into 
		   a GeoDataFrame, and saves them to a shapefile with 
		   the given output filename (or default name). 

		   Returns the merged GeoDataFrame. 
		"""
		map_a_df = self.read_map(map_a_filepath)
		map_b_df = self.read_map(map_b_filepath)
		self.merged_gdf = self.merge_maps(map_a_df, map_b_df)
		self.merged_gdf['district_a'] = self.merged_gdf['district_a'].fillna(value=-1)
		self.merged_gdf['district_b'] = self.merged_gdf['district_b'].fillna(value=-1)
		self.merged_gdf = self.merged_gdf.join(self.pop_df)

		# Compute reference zones, and verify they are unique (a permutation).
		ref_zones, overlap_counts = self.reference_zones(property_name='population')

		if set(ref_zones[1:]) != set(np.arange(1, 9)):
			print('Collision in reference zones:')
			print(list(ref_zones[1:]))
			ref_zones = self.reference_zones_by_max_wt_matching(overlap_counts)
			print('New reference zones from maximum-weight matching:')
			print(ref_zones[1:])

		# Add column 'IS_CORE' to merged GeoDataFrame indicating whether 
		# each unit is in its overlap. 
		# Also add column combining district_a with core_zone, 
		# setting to -1 if 'IS_CORE' is 0. 
		is_core = np.zeros(len(self.merged_gdf))
		core_zone = np.zeros(len(self.merged_gdf))
		i = 0
		for index, row in self.merged_gdf.iterrows():
			is_core[i] = ref_zones[int(row['district_a'])] == int(row['district_b'])
			core_zone[i] = is_core[i] * int(row['district_a'])
			if core_zone[i] <= 0:
				core_zone[i] = -1 # Replace both 0 and -0 with -1
			i += 1

		self.merged_gdf = self.merged_gdf.assign(IS_CORE = is_core)
		self.merged_gdf = self.merged_gdf.assign(COREZONE = core_zone)

		# Save to shapefile and return merged GeoDataFrame
		self.merged_gdf.to_file(output_filename)
		return self.merged_gdf


if __name__ == '__main__':
	# Examine overlaps between Rep./Dem. gerrymanders for Wisconsin. 
	HOME_DIR = os.path.expanduser("~")
	DATA_DIR = '{0}/{1}'.format(HOME_DIR, 'Documents/Data') 
	tracts_fpath = 'zip://{0}/{1}'.format(DATA_DIR, 'Census/Wisconsin/tl_2013_55_tract.zip')
	pop_fpath = '{0}/Census/Wisconsin/2010_CensusTractPopulations/DEC_10_SF1_P1_with_ann_modified.csv'.format(DATA_DIR)
	
	merger = MapMerger(tracts_fpath, populations_file_path=pop_fpath)

	map_a_filepath = '{0}/ICOR/Wisconsin/wi-gerrymander-dem.csv'.format(DATA_DIR)
	map_b_filepath = '{0}/ICOR/Wisconsin/wi-gerrymander-rep.csv'.format(DATA_DIR)
	merger.read_merge_and_save(map_a_filepath, map_b_filepath, 'merged-maps-wi-gerrymanders.shp')
