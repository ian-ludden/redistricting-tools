import geopandas
import numpy as np
import os

from gerrychain import Graph, Partition

######################################################################
# Author: Ian Ludden
# Date:   26 Feb 2020
# 
# GerryChain test with Wisconsin data
######################################################################
HOME = os.path.expanduser("~")
DIR = "Documents/Data/Census/Wisconsin"
tractsZipFname = "tl_2013_55_tract.zip"

# Load Wisconsin tracts as geopandas.GeoDataFrame
shapefilePath = 'zip://{0}/{1}/{2}'.format(HOME, DIR, tractsZipFname)
gdf = geopandas.read_file(shapefilePath)

# Build adjacency graph
graph = Graph.from_geodataframe(gdf, reproject=True)

# Create initial district assignment

# import pdb; pdb.set_trace()