import csv
import geopandas
import numpy as np
import os
import pandas as pd

from gerrychain import Graph, Partition


HOME = os.path.expanduser("~")
DIR = "Documents/Data/Census/Wisconsin"
tractsZipFname = "tl_2013_55_tract.zip"

# Load Wisconsin tracts as geopandas.GeoDataFrame
shapefilePath = 'zip://{0}/{1}/{2}'.format(HOME, DIR, tractsZipFname)
gdf = geopandas.read_file(shapefilePath)

# Build adjacency graph
graph = Graph.from_geodataframe(gdf, reproject=True)

# Create initial district assignment
with open('init-partition.csv') as csv_file:
	csv_reader = csv.reader(csv_file, delimiter=',')
	initPartitionRaw = list(csv_reader)

partition_cols = initPartitionRaw.pop(0)
partition_df = pd.DataFrame(initPartitionRaw, columns=partition_cols)
joined_df = gdf.set_index('GEOID').join(partition_df.set_index('GEOID'))

# Write updated shapefile with district assignments
joined_df.to_file("init-wi-district-plan.shp")
