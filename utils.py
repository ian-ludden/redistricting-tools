import csv
import descartes
import geopandas
import json
import numpy as np
import os
import pandas as pd
import random

from gerrychain import Graph, MarkovChain, Partition
from gerrychain.accept import always_accept
from gerrychain.constraints import single_flip_contiguous, within_percent_of_ideal_population, Validator
from gerrychain.proposals import propose_random_flip
from gerrychain.updaters import cut_edges, Tally

# Constants
DEFAULT_POP_BAL_THRESHOLD = 0.05

DEBUG = False


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
    ICOR_DIR = "Documents/Data/ICOR/Wisconsin"
    tracts_fname = "tl_2013_55_tract.zip"
    pop_fpath = '{0}/{1}/2010_CensusTractPopulations/DEC_10_SF1_P1_with_ann_modified.csv'.format(HOME, DIR)
    votes_2016_fpath = '{0}/{1}/E_55_tract_votes2016.csv'.format(HOME, ICOR_DIR)
    votes_2012_fpath = '{0}/{1}/F_55_tract_votes2012.csv'.format(HOME, ICOR_DIR)

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

    # Add average of 2012 and 2016 presidential election votes to gdf
    votes_2016_df = pd.read_csv(votes_2016_fpath)
    votes_2016_df = votes_2016_df.rename(columns={"GEOID10": "GEOID", "votes_dem": "votes_2016_dem", "votes_gop": "votes_2016_gop"})
    votes_2016_df = votes_2016_df.astype({'GEOID': str}).set_index('GEOID')

    votes_2012_df = pd.read_csv(votes_2012_fpath)
    votes_2012_df = votes_2012_df.rename(columns={"GEOID10": "GEOID", "votes_dem": "votes_2012_dem", "votes_gop": "votes_2012_gop"})
    votes_2012_df = votes_2012_df.astype({'GEOID': str}).set_index('GEOID')

    votes_df = votes_2016_df.join(votes_2012_df)
    votes_df['votes_dem'] = (votes_df['votes_2016_dem'] + votes_df['votes_2012_dem']) / 2.
    votes_df['votes_gop'] = (votes_df['votes_2016_gop'] + votes_df['votes_2012_gop']) / 2.
    votes_df = votes_df.drop(columns=['votes_2016_dem', 'votes_2012_dem', 'votes_2016_gop', 'votes_2012_gop'])

    gdf = gdf.join(votes_df)

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
        updaters={"cut_edges": cut_edges, "population": Tally("population"), 
                "votes_dem": Tally("votes_dem"), "votes_gop": Tally("votes_gop")})

    if DEBUG:
        print(map_1["population"])
    
    # Generate new_map by applying num_flips one-swaps to map_1
    if num_flips > 0:
        new_map = make_random_flips(map_1, num_flips)

    map_2_fname = 'wi-gerrymander-dem.csv' #'icor-wi-04.csv'
    map_2_basic = read_map('./{0}'.format(map_2_fname))
    map_2_gdf = gdf.join(map_2_basic)
    map_2_gdf['district'] = map_2_gdf['district'].fillna(value=-1)

    map_2_graph = Graph.from_geodataframe(map_2_gdf)
    map_2_graph.add_data(map_2_gdf)
    map_2 = Partition(
        map_2_graph, 
        'district',
        updaters={"cut_edges": cut_edges, "population": Tally("population"), 
                "votes_dem": Tally("votes_dem"), "votes_gop": Tally("votes_gop")})

    target_map = new_map if num_flips > 0 else map_2
    return [map_1, target_map, gdf]


def make_random_flips(partition, num_flips=1):
    """
    Randomly chooses a cut-edge in the given district map
    and swaps one of its endpoints (units) to the other district. 
    
    Performs a total of num_flips such single-unit swaps.

    Returns the resulting Partition object. 
    """
    is_valid = Validator([single_flip_contiguous, within_percent_of_ideal_population(partition, percent=DEFAULT_POP_BAL_THRESHOLD)])
    chain = MarkovChain(
        proposal=propose_random_flip,
        constraints=is_valid,
        accept=always_accept,
        initial_state=partition,
        total_steps=num_flips+1
    )

    for index, current_partition in enumerate(chain):
        if index == len(chain) - 1:
            new_partition = current_partition

    # new_partition = Partition(
    #     partition.graph, 'district',
    #     updaters={"cut_edges": cut_edges, "population": Tally("population"), 
    #             "votes_dem": Tally("votes_dem"), "votes_gop": Tally("votes_gop")})
    
    # num_flips_remaining = num_flips

    # while num_flips_remaining > 0:
    #     edge = random.choice(list(new_partition["cut_edges"]))
    #     flipped_node, other_node = edge[0], edge[1]
    #     previous_district = new_partition.assignment[flipped_node]
    #     flip = {flipped_node: new_partition.assignment[other_node]}
    #     new_partition_candidate = new_partition.flip(flip)

    #     is_pop_balanced = True
    #     district_pops_dict = new_partition_candidate["population"]
    #     num_parts = len(new_partition_candidate.parts)
    #     total_pop = np.sum([district_pops_dict[str(i)] for i in range(1, num_parts + 1)])
    #     ideal_pop = total_pop * 1. / num_parts
    #     pop_bal_threshold = DEFAULT_POP_BAL_THRESHOLD

    #     for district in district_pops_dict:
    #         district_pop = district_pops_dict[district]
    #         if (district_pop < (1 - pop_bal_threshold) * ideal_pop
    #             or district_pop > (1 + pop_bal_threshold) * ideal_pop):
    #             is_pop_balanced = False

    #     if is_pop_balanced:
    #         new_partition = new_partition.flip(flip)
    #         num_flips_remaining -= 1
        
    #         if DEBUG:
    #             print('Flipped unit {0} from {1} to {2}.'.format(flipped_node, previous_district, new_partition.assignment[flipped_node]))
    #             print('\t{0}'.format(new_partition["population"]))

    return new_partition


def save_path_of_maps(path, fname='path_out.json'):
    """
    Saves the given path (list) of district maps to a file. 
    """
    path_dict = {}

    # Add initial map (set of units for each district)
    parts = path[0].partition.parts
    map_dict = {}
    for key in parts.keys():
        units = list(parts[key])
        map_dict[key] = units

    path_dict['initial_map'] = map_dict

    path_dict['flips'] = []
    for node in path:
        if node.flip is not None:
            path_dict['flips'].append(node.flip)

    with open(fname, 'w') as outfile:
        json.dump(path_dict, outfile)


if __name__ == "__main__":
    # rep_map, dem_map, gdf = get_sample_wi_maps()

    rep_map, rep_map_modified, gdf = get_sample_wi_maps(num_flips=100)


    rep_map.plot()
    plt.axis('off')
    plt.show()

    # rep_map_modified.plot()
    # plt.axis('off')
    # plt.show()
