import csv
import geopandas
import numpy as np
import os
import pandas as pd
import random

from gerrychain import Graph, Partition
from gerrychain.updaters import cut_edges, Tally


def read_map(map_fname):
    """
    Reads a district map from a csv file with 
    columns 'GEOID' and 'district'. 
    Returns a pandas DataFrame object representing the map, where
    'GEOID' is the key.
    """
    with open(map_fname, 'r') as map_file:
        map_reader = csv.reader(map_file)
        map_raw = list(map_reader)

    map_headers = map_raw.pop(0)
    map_df = pd.DataFrame(map_raw, columns=map_headers)
    return map_df.set_index('GEOID')


def tally_property(gdf, units_subset, compat_property='units'):
    """
    Tallies the value of the given property 
    over a subset of units (GEOIDs) in the given GeoDataFrame. 
    """
    if compat_property == 'units': # Easy case; no need to manipulate DataFrame
        return len(units_subset)

    df = gdf.reset_index() # Make sure GEOID is a column for lookup
    units_subset_series = pd.Series(list(units_subset), name='GEOID', dtype='str')

    if compat_property in df.columns:
        return df.merge(units_subset_series, on='GEOID')[compat_property].sum()
    else:
        raise Exception('Property \'{}\' is not a column of the given GeoDataFrame.'.format(compat_property))


def get_subset(zone_1, zone_2, operation='intersection'):
    """
    Returns the subset defined by the given operation on the given zones. 

    Supported "operation" values include: 
     - 'intersection', the units in both zone_1 and zone_2 (default)
     - 'difference', the units in zone_1 but not zone_2
    """
    subset = None
    if operation == 'intersection':
        subset = zone_1.intersection(zone_2)
    elif operation == 'difference':
        subset = zone_1.difference(zone_2)
    else:
        raise Exception('Unsupported operation: {}'.format(operation))
    return subset


def get_sample_wi_maps(num_flips=-1):
    """
    Loads the republican and democratic gerrymanders for Wisconsin. 
    
    If num_flips is a positive integer, 
    returns the republican map and a map with num_flips random one-swaps
    applied to the republican map. 
    
    If num_flips is -1 (the default), 
    returns the republican and democratic gerrymander maps. 

    Maps are returned as Partition objects from the MGGG code. 

    In either case, a third object is returned: the geopandas.GeoDataFrame object
    representing the Wisconsin census tracts, supplemented with population data. 
    """
    HOME = os.path.expanduser("~")
    DIR = "Documents/Data/Census/Wisconsin"
    tracts_fname = "tl_2013_55_tract.zip"
    pop_fpath = '{0}/{1}/2010_CensusTractPopulations/DEC_10_SF1_P1_with_ann_modified.csv'.format(HOME, DIR)

    # Load Wisconsin tracts as geopandas.GeoDataFrame
    shapefile_path = 'zip://{0}/{1}/{2}'.format(HOME, DIR, tracts_fname)
    gdf = geopandas.read_file(shapefile_path)
    gdf.set_index('GEOID', inplace=True)

    # Add populations to gdf
    with open(pop_fpath, 'r') as pop_file:
        pop_reader = csv.reader(pop_file)
        pop_raw = list(pop_reader)

    pop_headers = pop_raw.pop(0)
    pop_df = pd.DataFrame(pop_raw, columns=pop_headers).astype({"population": int})
    pop_df = pop_df.drop(columns=['GEOIDLONG', 'DISPLAYNAME'])
    pop_df = pop_df.set_index('GEOID')
    gdf = gdf.join(pop_df)

    # Remove units with zero population
    gdf = gdf.loc[gdf["population"]!=0]

    # Load sample Wisconsin maps
    map_1_fname = 'wi-gerrymander-rep.csv' #'icor-wi-03.csv'
    map_1_basic = read_map('./{0}'.format(map_1_fname))
    map_1_gdf = gdf.join(map_1_basic)
    map_1_gdf['district'] = map_1_gdf['district'].fillna(value=-1)

    map_1_graph = Graph.from_geodataframe(map_1_gdf)
    map_1_graph.add_data(map_1_gdf)
    map_1 = Partition(
        map_1_graph, 
        'district',
        updaters={"cut_edges": cut_edges, "population": Tally("population")})

    # Generate new_map by applying num_flips one-swaps to map_1
    print(map_1["population"])
    new_map = Partition(
        map_1_graph, 'district',
        updaters={"cut_edges": cut_edges, "population": Tally("population")})
    if num_flips > 0:
        for i in range(num_flips):
            edge = random.choice(list(new_map["cut_edges"]))
            flipped_node, other_node = edge[0], edge[1]
            flip = {flipped_node: new_map.assignment[other_node]}
            new_map = new_map.flip(flip)
            print(new_map["population"])

    map_2_fname = 'wi-gerrymander-dem.csv' #'icor-wi-04.csv'
    map_2_basic = read_map('./{0}'.format(map_2_fname))
    map_2_gdf = gdf.join(map_2_basic)
    map_2_gdf['district'] = map_2_gdf['district'].fillna(value=-1)

    map_2_graph = Graph.from_geodataframe(map_2_gdf)
    map_2_graph.add_data(map_2_gdf)
    map_2 = Partition(
        map_2_graph, 
        'district',
        updaters={"cut_edges": cut_edges, "population": Tally("population")})

    target_map = new_map if num_flips > 0 else map_2
    return [map_1, target_map, gdf]


if __name__ == "__main__":
    rep_map, dem_map, gdf = get_sample_wi_maps()

    rep_map, rep_map_modified, gdf = get_sample_wi_maps(num_flips=10)
