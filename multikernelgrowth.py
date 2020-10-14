#!/usr/bin/python
# --------------------------------------------------------------------
# multikernelgrowth.py
# --------------------------------------------------------------------
"""
Generates a hybrid of two given maps by a multi-kernel growth approach
starting with a bijection of the districts that maximizes total overlap.
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random

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


def generate_hybrid(merger, plan_a, plan_b):
    # Find kernels
    ref_zones = find_kernels(merger, plan_a, plan_b)

    # Determine unassigned nodes
    df = merger.merged_gdf.copy()
    df['is_overlap'] = False
    df['district_a'].fillna(value=-1, inplace=True)
    df['district_b'].fillna(value=-1, inplace=True)
    df = df.astype({'district_a': int, 'district_b': int})
    k = len(ref_zones) - 1

    for a in range(1, k + 1):
        df['is_overlap'] = (df['is_overlap']) | ((df['district_a'] == a) & (df['district_b'] == ref_zones[a]))

    df['district'] = -1
    df.loc[df['is_overlap'], 'district'] = df['district_a']
    unassigned_nodes = set(df.loc[~df['is_overlap']].index)
    print(len(unassigned_nodes), 'unassigned (non-overlap)')
    print(df['is_overlap'].sum(), 'assigned (overlap)')

    # Create partial hybrid plan
    graph = plan_a.graph.copy()
    graph.geometry = plan_a.graph.geometry
    graph.add_data(df)
    hybrid = Partition(graph, 'district')

    # Check whether overlap zones are contiguous
    # import networkx as nx
    # for part in hybrid.parts:
    #     subgraph = hybrid.subgraphs[part]
    #     if subgraph:
    #         print('Zone', part, 'has', subgraph.number_of_nodes(), 'nodes. Contiguous:', nx.is_connected(subgraph))
    #     else:
    #         print('Zone', part, 'has null subgraph:', subgraph)

    # Assign remaining units
    free_nodes = list(unassigned_nodes)
    count_no_neighbors = 0
    MAX_ALLOWED_RETRIES = 1000

    while free_nodes:
        node = free_nodes.pop()
        neighbor_zones = set([hybrid.assignment[y] for y in hybrid.graph.neighbors(node)]).difference({-1})
        
        if neighbor_zones:
            zone = random.sample(neighbor_zones, 1)[0]
            hybrid = hybrid.flip({node: zone})
        else:
            count_no_neighbors += 1
            if count_no_neighbors < MAX_ALLOWED_RETRIES: # prevent infinite loop
                free_nodes.insert(0, node)

    return hybrid


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

    hybrid = generate_hybrid(merger, plan_a, plan_b)

    hybrid.plot()
    plt.axis('off')
    plt.show()
