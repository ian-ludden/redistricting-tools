#!/usr/bin/python
# --------------------------------------------------------------------
# multikernelgrowth.py
# --------------------------------------------------------------------
"""
Generates a hybrid of two given maps by a multi-kernel growth approach
starting with a bijection of the districts that maximizes total overlap.
"""
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import pandas as pd
import random

from gerrychain import Graph, Partition
from gerrychain.constraints import within_percent_of_ideal_population
from gerrychain.constraints.contiguity import single_flip_contiguous
from gerrychain.grid import Grid
from gerrychain.updaters import Tally, cut_edges

from overlaps import MapMerger

tolerance = 0.02 # population deviation tolerance

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
    k = len(ref_zones) - 1 # This assumes some nodes are unassigned, i.e., assigned to part -1

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
    hybrid = Partition(graph, 'district', 
        updaters={
            'population': Tally('population', alias='population')
    }) # The updater {'cut_edges': cut_edges} is included by default

    # Check whether overlap zones are contiguous
    # import networkx as nx
    # for part in hybrid.parts:
    #     subgraph = hybrid.subgraphs[part]
    #     if subgraph:
    #         print('Zone', part, 'has', subgraph.number_of_nodes(), 'nodes. Contiguous:', nx.is_connected(subgraph))
    #     else:
    #         print('Zone', part, 'has null subgraph:', subgraph)

    df = hybrid.graph.data
    total_pop = df['population'].sum()
    avg_pop = total_pop * 1. / k

    # Assign each unassigned node to its neighboring zone with the least population
    free_nodes = list(hybrid.subgraphs[-1].nodes())
    random.shuffle(free_nodes)
    count_invalid = 0
    MAX_ALLOWED_RETRIES = 1000

    while free_nodes and count_invalid < MAX_ALLOWED_RETRIES:
        node = free_nodes.pop()
        neighbor_zones = set([hybrid.assignment[y] for y in hybrid.graph.neighbors(node)]).difference({-1})

        if neighbor_zones:
            # Pick neighbor zone with minimum population
            min_pop = total_pop
            new_zone = -1
            for neighbor_zone in neighbor_zones:
                if hybrid.population[neighbor_zone] < min_pop:
                    min_pop = hybrid.population[neighbor_zone]
                    new_zone = neighbor_zone
            
            # Check population balance, flip if acceptable
            node_pop = hybrid.graph.nodes[node]['population']
            if hybrid.population[new_zone] + node_pop <= (1 + tolerance) * avg_pop:
                hybrid = hybrid.flip({node: new_zone})
                hybrid.parent = None # Erase parent to avoid memory leak/self-reference
            else:
                count_invalid += 1
                free_nodes.insert(0, node)
        else:
            count_invalid += 1
            free_nodes.insert(0, node)


    # Randomly assign remaining nodes without regard for population balance
    while free_nodes:
        node = free_nodes.pop()
        neighbor_zones = set([hybrid.assignment[y] for y in hybrid.graph.neighbors(node)]).difference({-1})
        
        if neighbor_zones:
            # Pick random neighbor zone
            zone = random.sample(neighbor_zones, 1)[0]
            hybrid = hybrid.flip({node: zone})
            hybrid.parent = None
        else:
            free_nodes.insert(0, node)


    # Fix population imbalances
    # Delete -1 part to make pop. balance checks correct
    hybrid.parts.pop(-1, None)
    hybrid.population.pop(-1, None)

    # Build a function for checking population balance
    pop_bounds = within_percent_of_ideal_population(hybrid, percent=tolerance)

    # 
    MAX_RETRIES = 1000
    count_retries = 0
    while not pop_bounds(hybrid) and count_retries < MAX_RETRIES:
        pop_dev = pop_dev = [(hybrid.population[i] - avg_pop) / avg_pop for i in range(1, k + 1)]
        
        smallest_dev = min(pop_dev)
        smallest_part = pop_dev.index(smallest_dev) + 1
        
        # Find boundary units of other districts that are adjacent to smallest_part
        neighboring_units = []

        for edge in hybrid.cut_edges:
            # By definition of cut-edge, at most one of these conditions will hold
            if edge[0] in hybrid.parts[smallest_part]:
                neighboring_units.append(edge[1])
            if edge[1] in hybrid.parts[smallest_part]:
                neighboring_units.append(edge[0])
        
        # #  Try moving a neighboring unit, uniformly at random, into the smallest district. 
        # unit_to_add = random.sample(neighboring_units, 1)[0]

        # Pick a neighboring unit from the largest neighboring district. 
        unit_to_add = neighboring_units[0]
        largest_neighbor_district = hybrid.assignment[unit_to_add]
        largest_neighbor_district_pop = hybrid.population[largest_neighbor_district]

        for neighboring_unit in neighboring_units:
            neighbor_district = hybrid.assignment[neighboring_unit]
            neighbor_district_pop = hybrid.population[neighbor_district]
            if neighbor_district_pop > largest_neighbor_district_pop:
                unit_to_add = neighboring_unit
                largest_neighbor_district = neighbor_district
                largest_neighbor_district_pop = neighbor_district_pop

        candidate = hybrid.flip({unit_to_add: smallest_part})
        
        # Check contiguity before finalizing the flip. 
        # If it fails, try to flip a random neighboring unit instead. 
        if single_flip_contiguous(candidate):
            hybrid = candidate
            hybrid.parent = None
        else:
            unit_to_add = random.sample(neighboring_units, 1)[0]
            candidate = hybrid.flip({unit_to_add: smallest_part})
            if single_flip_contiguous(candidate):
                hybrid = candidate
                hybrid.parent = None
        
        count_retries += 1

    print('Local flips made in pursuit of population balance:', count_retries)

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

    pop_bounds = within_percent_of_ideal_population(hybrid, percent=tolerance)
    print('\nDoes the hybrid plan have population balance (tol={0:.3f})?'.format(tolerance), 'Yes' if pop_bounds(hybrid) else 'No')
    is_contiguous = True
    for part in range(1, len(hybrid.parts) + 1):
        is_contiguous = is_contiguous and nx.is_connected(hybrid.subgraphs[part])
    
    print('\nDoes the hybrid plan have contiguity?', 'Yes' if is_contiguous else 'No')

    hybrid.plot()
    plt.axis('off')
    plt.savefig('hybrid_plan_tol={0}.png'.format(tolerance))
    plt.show()


    grid = Grid((6, 6))
    print(grid.parts)
