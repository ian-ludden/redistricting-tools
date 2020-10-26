from collections import deque
import csv
import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import pandas as pd
import random

import gerrychain

def add_population_data(gdf, populations_file_path=None):
    """
    Reads a population data file (CSV) and 
    joins it to the given GeoDataFrame using
    'GEOID' as the index/key. 

    The CSV file must have at least a 'GEOID' column and
    a 'population' column. Any other columns will be ignored.

    If populations_file_path is None, 
    then populations are all set to 1. 

    Returns the new GeoDataFrame with population data. 
    The given gdf is modified. 
    """
    if populations_file_path is None:
        gdf['population'] = 1.
        return gdf
    
    pop_df = pd.read_csv(populations_file_path)
    pop_df = pop_df[['GEOID', 'population']]
    pop_df = pop_df.astype({'GEOID': str})
    pop_df.set_index('GEOID', inplace=True)
    
    return gdf.join(pop_df)


def load_shapefile(shapefile_path):
    """
    Loads the shapefile at the given path,
    likely a zip folder, 
    to a GeoDataFrame using GeoPandas. 

    The shapefile must include a column with 'GEOID'
    to be used as the GeoDataFrame index. 

    Returns the GeoDataFrame with 'GEOID' set as the index. 
    """
    gdf = gpd.read_file(shapefile_path)
    return gdf.set_index('GEOID')


def build_district_plan(gdf, assignment_file_path):
    """
    Loads a CSV representing a district plan as 
    a mapping of 'GEOID' to 'district'. 

    Creates a gerrychain.Partition object using
    the graph of the given GeoDataFrame and 
    the assignment mapping. 
    """
    with open(assignment_file_path, 'r') as file:
        reader = csv.reader(file)
        data = list(reader)

    headers = data.pop(0)
    assignment = {data[i][0]: int(data[i][1]) for i in range(len(data))}

    graph = gerrychain.Graph.from_geodataframe(gdf)
    nodes = list(graph.nodes)

    unassigned = []
    # Resolve any issues with unassigned units
    for node in nodes:
        if node not in assignment.keys():
            unassigned.append(node)

    while unassigned:
        node = unassigned.pop()
        for neighbor in graph.neighbors(node):
            if neighbor in assignment:
                assignment[node] = assignment[neighbor]
    
    # Add district assignment to GeoDataFrame
    assignment_array = [[node, assignment[node]] for node in assignment]
    assignment_df = pd.DataFrame(assignment_array, columns=headers)
    assignment_df.set_index('GEOID', inplace=True)
    gdf = gdf.join(assignment_df)
    graph.add_data(gdf)

    return gerrychain.Partition(graph, assignment)


def zone_max_matching(plan_a, plan_b):
    """
    Determines the optimal bipartite matching of
    districts in plan_a to districts in plan_b 
    to maximize the total population overlap. 

    Both plans should have districts indexed from 1 to k, 
    where k is some positive integer. 

    Based on the concept of "reference zones"
    as defined in Pereira et al. (2009). 

    Returns a list ref_zones representing 
    a bijection from zones in plan_a to zones in plan_b.

    Note ref_zones is indexed 0 to k, whereas
    the districts are indexed 1 to k, so
    ref_zones[0] is unused. 
    """
    assert(len(plan_a.parts) == len(plan_b.parts))
    k = len(plan_a.parts)
    overlap_counts = np.zeros((len(plan_a.parts), len(plan_b.parts)))

    for zone_a in plan_a.parts:
        for zone_b in plan_b.parts:
            intersection = plan_a.parts[zone_a].intersection(plan_b.parts[zone_b])
            df = plan_a.graph.data
            df = df.loc[df.index.isin(intersection)]
            overlap_counts[zone_a - 1, zone_b - 1] = df['population'].sum()

    max_matching = max_matching_helper(overlap_counts)
    ref_zones = [0] + max_matching
    return ref_zones


def max_matching_helper(edge_weights):
    """
    Finds a max-weight matching given
    a (symmetric) 2-D array of edge weights. 
    """
    k = len(edge_weights)
    G = nx.Graph()
    for map_name in ['A', 'B']:
        for i in range(1, k + 1):
            G.add_node('A{0}'.format(i))
            G.add_node('B{0}'.format(i))

    for i in range(k):
        u = 'A{0}'.format(i + 1)
        for j in range(k):
            v = 'B{0}'.format(j + 1)
            # Negate weights to use Networkx min-weight matching
            G.add_edge(u, v, weight=-edge_weights[i, j])

    M = nx.bipartite.minimum_weight_full_matching(G)

    max_matching = []
    for i in range(k):
        u = 'A{0}'.format(i + 1) # 'Ai'
        v = M[u] # 'Bj', neighbor of 'Ai' in optimal matching
        max_matching.append(int(v[1:])) # j, as int

    return max_matching


def generate_hybrid(plan_a, plan_b, pop_bal_tolerance=0.05):
    """
    Generates a hybrid of two given maps by
    starting with a bijection of the districts 
    that maximizes total overlap, then
    assigning unassigned units and fixing population balance
    via local-search. 
    """
    ref_zones = zone_max_matching(plan_a, plan_b)
    df = plan_a.graph.data.join(plan_b.graph.data['district'], lsuffix='_a', rsuffix='_b')
    df['is_overlap'] = False
    df['district_a'].fillna(value=-1, inplace=True)
    df['district_b'].fillna(value=-1, inplace=True)
    k = len(ref_zones) - 1

    # Add columns for hybrid district assignments and 
    # whether each node is in the overlap defined by ref_zones
    df['district'] = -1

    for a in range(1, k + 1):
        df['is_overlap'] = (df['is_overlap']) | ((df['district_a'] == a) & (df['district_b'] == ref_zones[a]))
    
    unassigned_nodes = set(df.loc[~df['is_overlap']].index)
    df.loc[df['is_overlap'], 'district'] = df['district_a']

    # Create partial hybrid plan based on plan_a
    graph = plan_a.graph.copy()
    graph.geometry = plan_a.graph.geometry
    graph.add_data(df)
    hybrid = gerrychain.Partition(graph, 'district', updaters={
        'population': gerrychain.updaters.Tally('population')
        }) # The updater {'cut_edges': cut_edges} is included by default

    # Assign each unassigned node to min pop. neighboring zone
    hybrid = assign_gaps(hybrid, tolerance=pop_bal_tolerance)
    # Assign remaining nodes without regard for population balance
    hybrid = assign_gaps(hybrid, tolerance=1000)
    hybrid = rebalance_populations(hybrid, tolerance=pop_bal_tolerance)
    return hybrid


def assign_gaps(hybrid, max_retries=1000, tolerance=0.05):
    """
    Assign each unassigned node to its neighboring zone 
    with the least population. 

    The unassigned nodes are randomly shuffled at the start
    to eliminate any spatial correlations in the initial order. 

    If a zone has no assigned neighbors, 
    or if adding it to the neighboring zone 
    with minimum population would cause that zone 
    to exceed the max allowable population 
    (using the given tolerance from ideal, default 5%), 
    then it is added back to the queue of unassigned nodes 
    to be considered again later. 

    At most max_retries such retries are allowed 
    before giving up and leaving the remaining nodes unassigned. 

    Returns the modified hybrid Partition. 
    """
    free_nodes = deque(hybrid.subgraphs[-1].nodes())
    random.shuffle(free_nodes)

    count_invalid = 0
    k = max(hybrid.parts.keys())
    ideal_pop = hybrid.graph.data['population'].sum() * 1. / k

    while free_nodes and count_invalid < max_retries:
        node = free_nodes.popleft()
        neighbors = hybrid.graph.neighbors(node)
        neighbor_zones = set([hybrid.assignment[y] for y in neighbors])
        neighbor_zones = list(neighbor_zones.difference({-1})) # Don't consider dummy zone

        # Try again if no assigned neighbors
        if not neighbor_zones:
            count_invalid += 1
            free_nodes.append(node)
            continue

        # Find neighbor with minimum population
        neighbor_zone_pops = [hybrid.population[zone] for zone in neighbor_zones]
        min_pop_index = neighbor_zone_pops.index(min(neighbor_zone_pops))
        new_zone = neighbor_zones[min_pop_index]

        # Check population balance, flip if acceptable
        node_pop = hybrid.graph.nodes[node]['population']
        if hybrid.population[new_zone] + node_pop <= (1 + tolerance) * ideal_pop:
            hybrid = hybrid.flip({node: new_zone})
            hybrid.parent = None # Erase parent to avoid memory leak/self-reference
        else:
            count_invalid += 1
            free_nodes.append(node)

    return hybrid


def rebalance_populations(hybrid, max_retries=1000, tolerance=0.05):
    """
    Fix population imbalances by local search. 

    The input hybrid Partition should have no unassigned units; 
    otherwise, these units will be ignored/lost. 
    """
    # Delete -1 part to make pop. balance checks correct
    hybrid.parts.pop(-1, None)
    hybrid.population.pop(-1, None)
    k = len(hybrid.parts)

    # Build a function for checking population balance
    is_pop_bal = gerrychain.constraints.within_percent_of_ideal_population(hybrid, percent=tolerance)
    
    count_retries = 0
    while not is_pop_bal(hybrid) and count_retries < max_retries:
        # Determine which zone has the least population
        zone_pops = [hybrid.population[i] for i in range(1, k + 1)]
        min_pop_zone = zone_pops.index(min(zone_pops)) + 1

        # Find boundary units of adjacent zones
        neighbor_units = []
        for edge in hybrid.cut_edges:
            # By definition of cut-edge, at most one of these conditions will hold
            if edge[0] in hybrid.parts[min_pop_zone]:
                neighbor_units.append(edge[1])
            if edge[1] in hybrid.parts[min_pop_zone]:
                neighbor_units.append(edge[0])

        # Pick a neighboring unit from the largest neighboring district, 
        # uniformly at random from all units on that border.        
        neighbor_zones = list(set([hybrid.assignment[unit] for unit in neighbor_units]))
        neighbor_zone_pops = [hybrid.population[i] for i in neighbor_zones]
        largest_neighbor_zone = neighbor_zones[neighbor_zone_pops.index(
            max(neighbor_zone_pops))]

        candidate_units = set(neighbor_units).intersection(hybrid.parts[largest_neighbor_zone])
        unit_to_flip = random.sample(candidate_units, 1)[0]

        candidate_plan = hybrid.flip({unit_to_flip: min_pop_zone})

        # Check contiguity before finalizing the flip. 
        if gerrychain.constraints.contiguity.single_flip_contiguous(candidate_plan):
            hybrid = candidate_plan
            hybrid.parent = None # Erase parent to avoid memory leak/self-reference
        else:
            # Try to flip a random neighboring unit instead. 
            unit_to_add = random.sample(neighbor_units, 1)[0]
            candidate_plan = hybrid.flip({unit_to_add: min_pop_zone})
            if gerrychain.constraints.contiguity.single_flip_contiguous(candidate_plan):
                hybrid = candidate_plan
                hybrid.parent = None
        
        count_retries += 1

    # Raise exception if we stopped due to exceeding max_retries
    if not is_pop_bal(hybrid):
        raise Exception('Failed to rebalance populations after {0} retries.\nPopulations: {1}'.format(count_retries, hybrid.population))

    return hybrid


if __name__ == '__main__':
    # Examine overlaps between Rep./Dem. gerrymanders for Wisconsin. 
    HOME_DIR = os.path.expanduser("~")
    DATA_DIR = '{0}/{1}'.format(HOME_DIR, 'Documents/Data') 
    tracts_fpath = 'zip://{0}/{1}'.format(DATA_DIR, 'Census/Wisconsin/tl_2013_55_tract.zip')
    pop_fpath = '{0}/Census/Wisconsin/2010_CensusTractPopulations/DEC_10_SF1_P1_with_ann_modified.csv'.format(DATA_DIR)
    
    gdf = load_shapefile(tracts_fpath)
    gdf = add_population_data(gdf, pop_fpath)

    plan_a = build_district_plan(gdf, '{0}/ICOR/Wisconsin/wi-gerrymander-dem.csv'.format(DATA_DIR))
    # plan_a.plot()
    # plt.axis('off')
    # plt.show()

    plan_b = build_district_plan(gdf, '{0}/ICOR/Wisconsin/wi-gerrymander-rep.csv'.format(DATA_DIR))
    # plan_b.plot()
    # plt.axis('off')
    # plt.show()

    ref_zones = zone_max_matching(plan_a, plan_b)
    print(ref_zones)

    hybrid = generate_hybrid(plan_a, plan_b, pop_bal_tolerance=0.02)

    # hybrid.plot()
    # plt.axis('off')
    # plt.show()

    print('-' * 40)
    print('District plan summary')
    print('-' * 40)
    for i in range(1, len(hybrid.parts) + 1):
        print('Zone', i, ':', hybrid.population[i], 'people,', len(hybrid.parts[i]), 'units')
