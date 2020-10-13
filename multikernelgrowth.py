#!/usr/bin/python
# --------------------------------------------------------------------
# multikernelgrowth.py
# --------------------------------------------------------------------
"""
Generates a hybrid of two given maps by a multi-kernel growth approach
starting with a bijection of the districts that maximizes total overlap.
"""
import os
import pandas as pd
import numpy as np

from gerrychain import Graph, Partition

from overlaps import MapMerger

def find_kernels(merger, plan_a, plan_b):
    """
    Takes two Partition objects and finds kernels, that is, 
    a best bijection of overlaps, for starting a 
    multi-kernel growth approach to generating a hybrid map. 
    
    Returns a Partition object with the overlaps assigned to 
    districts 1 through k (where k is the number of districts in 
    each of the given maps) and assigning all remaining units to -1. 
    """
    # Extract the DataFrames of node attributes
    df_a = plan_a.graph.data
    df_b = plan_b.graph.data

    # Determine the number of districts
    k = df_a.max().district
    if not (k == df_b.max().district):
        raise ValueError('The given district plans must have the same number of districts.')

    # Find the best overlaps w.r.t. number of units
    # Compute reference zones, and verify they are unique (a permutation).
    ref_zones, overlap_counts = merger.reference_zones(df_a, df_b, property_name='population')
        
    if set(ref_zones[1:]) != set(np.arange(1, k + 1)):
        print('Collision in reference zones:')
        print(list(ref_zones[1:]))
        ref_zones = merger.reference_zones_by_max_wt_matching(overlap_counts)
        print('New reference zones from maximum-weight matching:')
        print(ref_zones[1:])

    return ref_zones


def generate_hybrid():
    # Find kernels

    # Fill in rest
    pass


if __name__ == '__main__':
    HOME_DIR = os.path.expanduser("~")
    DATA_DIR = '{0}/{1}'.format(HOME_DIR, 'Documents/Data') 
    tracts_fpath = 'zip://{0}/{1}'.format(DATA_DIR, 'Census/Wisconsin/tl_2013_55_tract.zip')
    pop_fpath = '{0}/Census/Wisconsin/2010_CensusTractPopulations/DEC_10_SF1_P1_with_ann_modified.csv'.format(DATA_DIR)
    
    merger = MapMerger(tracts_fpath, populations_file_path=pop_fpath)

    map_a_filepath = '{0}/ICOR/Wisconsin/wi-gerrymander-dem.csv'.format(DATA_DIR)
    map_b_filepath = '{0}/ICOR/Wisconsin/wi-gerrymander-rep.csv'.format(DATA_DIR)

    map_a_df = merger.read_map(map_a_filepath)
    map_a_df = map_a_df.astype({'district': int})
    map_b_df = merger.read_map(map_b_filepath)
    map_b_df = map_b_df.astype({'district': int})
    merger.merge_maps(map_a_df, map_b_df)

    gdf = merger.units_gdf

    graph_a = Graph.from_geodataframe(gdf)
    graph_a.add_data(map_a_df)
    graph_a.data['district'].fillna(value=-1, inplace=True)

    for node in graph_a:
        try:
            asst = {node: graph_a.nodes[node]['district']}
        except Exception as e:
            graph_a.nodes[node]['district'] = -1

    plan_a = Partition(graph_a, 'district')

    graph_b = Graph.from_geodataframe(gdf)
    graph_b.add_data(map_b_df)
    graph_b.data['district'].fillna(value=-1, inplace=True)

    for node in graph_b:
        try:
            asst = {node: graph_b.nodes[node]['district']}
        except Exception as e:
            graph_b.nodes[node]['district'] = -1

    plan_b = Partition(graph_b, 'district')

    ref_zones = find_kernels(merger, plan_a, plan_b)
