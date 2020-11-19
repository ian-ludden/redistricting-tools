from collections import deque
import csv
import geopandas as gpd
import json
import math
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import pandas as pd
import random
import re
import xml.dom.minidom
import xml.etree.ElementTree as ET

import gerrychain
from gerrychain.accept import always_accept
from gerrychain.proposals import propose_random_flip

# Constants
DEFAULT_POP_BAL_THRESHOLD = 0.05

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


def add_voteshare_data(gdf, voteshares_file_path=None):
    """
    Reads a voteshares data file (CSV) and 
    joins it to the given GeoDataFrame using
    'GEOID' as the index/key. 

    The CSV file must have at least a 'GEOID' column, 
    a 'gop_voteshare' column, and 
    a 'dem_voteshare' column. 
    Any other columns will be ignored.

    If voteshares_file_path is None, 
    then voteshares are all set to 0.5. 

    Returns the new GeoDataFrame with voteshare data. 
    The given gdf is modified. 
    """
    if voteshares_file_path is None:
        gdf['gop_voteshare'] = 0.5
        gdf['dem_voteshare'] = 0.5
        return gdf

    voteshare_df = pd.read_csv(voteshares_file_path)
    voteshare_df = voteshare_df[['GEOID', 'gop_voteshare', 'dem_voteshare']]
    voteshare_df = voteshare_df.astype({'GEOID': str})
    voteshare_df.set_index('GEOID', inplace=True)

    gdf = gdf.join(voteshare_df)
    return gdf


def add_assignment_as_district_col(partition):
    """
    Adds the given Partition object's assignment
    as a column named 'district'. 

    If there's already a column named 'district', 
    it is dropped.

    Returns the modified paritition. 
    """
    df = partition.graph.data
    if 'district' in df.columns:
        df.drop(columns=['district'], inplace=True)

    assignment_series = partition.assignment.to_series().rename('district')
    df = df.join(assignment_series)
    partition.graph.add_data(df)
    return partition


def build_partition(gdf, assignment_file_path):
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
    graph.add_data(gdf)
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
    
    partition = gerrychain.Partition(graph, assignment, updaters={
        'population': gerrychain.updaters.Tally('population')
        }) # The updater {'cut_edges': cut_edges} is included by default

    partition = add_assignment_as_district_col(partition)
    return partition


def build_grid_graph(rows, cols):
    """
    Builds a rows-by-cols grid graph with 
    each unit having population 1. 
    """
    G = nx.Graph()
    G.add_nodes_from(np.arange(1, rows * cols + 1))

    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            if i < rows:
                G.add_edge(j + cols * (i - 1), j + cols * i)

            if j < cols:
                G.add_edge(j + cols * (i - 1), j + cols * (i - 1) + 1)

    graph = gerrychain.Graph(G)
    df = pd.DataFrame(graph.nodes)
    df.rename(columns={0: 'Name'}, inplace=True)
    df.set_index('Name', inplace=True)
    df['population'] = 1.
    graph.add_data(df)

    return graph


def draw_grid_plan(partition):
    """
    Prints a visual representation of the given partition
    of an r x r grid graph. 

    The input partition must be of a square grid graph 
    with nodes named 1 to r^2. 
    """
    r = int(math.sqrt(partition.graph.number_of_nodes()))
    
    print('-', '----' * r, sep='')
    for i in range(r ** 2):
        index = i + 1
        row = i // r
        col = i % r

        print('| {0} '.format(partition.assignment[index]), end='')
        
        if col == r - 1:
            print('|\n-', '----' * r, sep='')


def pereira_index(p, q, property_name=None):
    """
    Given two Partition objects, 
    computes and returns the Tavares-Pereira et al. (2009)
    distance index. 

    If property_name is None, 
    then the edge weights delta_e are all 1. 
    Otherwise, 
    the given node property_name is used to compute
    the edge weights delta_e = min{p_i, p_j} 
    for each edge e = ij. 
    """
    if property_name is None:
        return pereira_index_unweighted(p, q)
    else:
        raise NotImplementedError()


def pereira_index_unweighted(p, q):
    """
    Given two Partition objects, 
    computes and returns the Tavares-Pereira et al. (2009)
    distance index (unweighted). 
    """
    num_disagree_edges = 0

    for e in p.graph.edges():
        x = min(e)
        y = max(e)
        px = p.assignment[x]
        qx = q.assignment[x]
        py = p.assignment[y]
        qy = q.assignment[y]

        num_disagree_edges += ((px == py and not(qx == qy)) or (not(px == py) and qx == qy))

    num_edges = p.graph.number_of_edges()
    assert(q.graph.number_of_edges() == num_edges) # Need to have same underlying graph
    
    return 1. / num_edges * (num_disagree_edges), num_disagree_edges, num_edges


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


def build_district_plan(tracts_fname, assignment_fname, pop_fname=None, voteshares_fname=None):
    """
    Loads a sample Wisconsin district plan. 

    Parameters:
        tracts_fname: the name of the Wisconsin census tracts file (zipped shapefile)
        partition_fname: the name of the initial partition file (.csv)
        pop_fname: (optional) the name of the population data file (.csv)
        voteshares_fname: (optional) the name of the voteshares data file (.csv)

    Returns:
        the given Wisconsin district plan as a GerryChain Partition object
    """
    tracts_fname = tracts_fname if 'zip://' in tracts_fname else 'zip://' + tracts_fname
    gdf = load_shapefile(tracts_fname)
    gdf = add_population_data(gdf, pop_fname)
    gdf = add_voteshare_data(gdf, voteshares_fname)
    plan = build_partition(gdf, assignment_fname)
    return plan


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


# Functions related to MIP warm-starts
def load_warmstarts_xml(file='warmstarts.mst'):
    if os.path.exists(file):
        try:
            tree = ET.parse(file)
        except ET.ParseError as e:
            print(e)
            tree = create_empty_warmstarts_xml()
    else:
        tree = create_empty_warmstarts_xml()

    root = tree.getroot()
    if not root.tag == 'CPLEXSolutions':
        raise ValueError('Invalid warmstarts xml file. Must have CPLEXSolutions tag as root.')

    return tree


def create_empty_warmstarts_xml():
    return ET.ElementTree(ET.Element('CPLEXSolutions'))


def save_warmstarts_xml(tree, file='warmstarts.mst'):
    raw = ET.tostring(tree.getroot(), encoding='utf-8').decode(encoding='utf-8')
    # Remove newlines and extra whitespaces so they don't accumulate
    raw = re.sub('\n\s*', '', raw).encode(encoding='utf-8')
    # Add indentation using built-in xml library
    pretty = xml.dom.minidom.parseString(raw).toprettyxml(indent='  ')
    with open(file, 'wb') as f:
        f.write(pretty.encode(encoding='utf-8'))
    return


# Functions related to setting Shirabe flow variables
def compute_feasible_flows(partition):
    """
    Given a Partition object, 
    computes feasible flow variable assignments 
    for the Shirabe flow constraints.
    (Helper function for the midpoint MIP warm-start.) 

    Returns f, an n-by-2m NumPy array 
    representing flow variable assignments, 
    where n is the number of nodes and 
    m is the number of undirected edges. 
    """
    n = partition.graph.number_of_nodes()
    m = partition.graph.number_of_edges()
    nodes = list(partition.graph.nodes())
    edges = list(partition.graph.edges())
    k = len(partition.parts)
    f = np.zeros((n, 2 * m)) # Flows are on directed edges, hence 2 * m

    for part in partition.parts:
        # The min unit is set as the center
        center = min(partition.parts[part])
        center_index = nodes.index(center)

        # Build a spanning tree of the part, and
        # label each node with its descendant count 
        # (treating center as root). 
        graph = partition.subgraphs[part]
        mst = nx.minimum_spanning_tree(graph)
        label_num_descendants(mst, center)

        # Set flow value of edge from parent to number of descendants
        stack = deque()
        stack.append(center)

        while stack:
            node = stack.pop()
            children = mst.nodes[node]['children']
            for child in children:
                stack.append(child)
                edge = (node, child)

                # Exact edge index depends on which directed version
                # of the edge has the flow
                if edge in edges:
                    edge_index = 2 * edges.index(edge)
                else:
                    edge_index = 2 * edges.index((edge[1], edge[0])) + 1

                f[center_index, edge_index] = mst.nodes[child]['num_descendants']

    return f


def label_num_descendants(tree, root):
    """
    Given a tree (Networkx Graph object)
    and its root, 
    label each node with its number of descendants, 
    including itself, 
    as a 'num_descendants' attribute. 
    """
    stack = deque()
    stack.append(root)

    postorder_stack = deque()

    for node in tree.nodes:
        tree.nodes[node]['visited'] = False # Use visited flags to avoid revisiting parents
        tree.nodes[node]['num_descendants'] = -1

    # In the first pass, determine parent-child relationships and
    # set num_descendants for leaves
    while stack:
        node = stack.pop()
        tree.nodes[node]['visited'] = True
        postorder_stack.append(node)
        
        children = []

        for neighbor in tree.neighbors(node):
            if not tree.nodes[neighbor]['visited']:
                stack.append(neighbor)
                children.append(neighbor)

        tree.nodes[node]['children'] = children

        if not children: # If children is empty, then node is a leaf
            tree.nodes[node]['num_descendants'] = 1

    # Use the post-order traversal to update num_descendants
    while postorder_stack:
        node = postorder_stack.pop()
        if tree.nodes[node]['num_descendants'] < 0:
            # Add up all num_descendants values of children, plus one to count node itself
            children_descendants = [tree.nodes[child]['num_descendants'] for child in tree.nodes[node]['children']]
            tree.nodes[node]['num_descendants'] = 1 + sum(children_descendants)
    return


# Functions for getting/generating sample data for Wisconsin
def get_sample_wi_plans(num_flips=-1):
    """
    Loads the republican and democratic gerrymanders for Wisconsin. 
    
    If num_flips is a positive integer, 
    returns the republican map and a map with num_flips random one-swaps
    applied to the republican map. 
    
    If num_flips is -1 (the default), 
    returns the republican and democratic gerrymander maps. 

    Maps are returned as Partition objects from the MGGG code. 
    """
    tracts_fname = 'data/tl_2013_55_tract.zip'
    dem_assignment_fname = 'data/wi_gerrymander_dem.csv'
    gop_assignment_fname = 'data/wi_gerrymander_rep.csv'
    population_fname = 'data/wi_tract_populations_census_2010.csv'
    voteshares_fname = 'data/wi_voteshares.csv'

    gop_plan = build_district_plan(tracts_fname, gop_assignment_fname, population_fname, voteshares_fname)
    dem_plan = build_district_plan(tracts_fname, dem_assignment_fname, population_fname, voteshares_fname)
    
    # Generate new_map by applying num_flips one-swaps to map_1
    if num_flips > 0:
        gop_plan_after_flips = make_random_flips(gop_plan, num_flips)

    second_plan = gop_plan_after_flips if num_flips > 0 else dem_plan
    return [gop_plan, second_plan]


def make_random_flips(partition, num_flips=1):
    """
    Randomly chooses a cut-edge in the given district map
    and swaps one of its endpoints (units) to the other district. 
    
    Performs a total of num_flips such single-unit swaps.

    Returns the resulting Partition object. 
    """
    is_valid = gerrychain.constraints.Validator(
        [gerrychain.constraints.single_flip_contiguous, 
        gerrychain.constraints.within_percent_of_ideal_population(
            partition, percent=DEFAULT_POP_BAL_THRESHOLD)])
    
    chain = gerrychain.MarkovChain(
        proposal=propose_random_flip,
        constraints=is_valid,
        accept=always_accept,
        initial_state=partition,
        total_steps=num_flips+1 # The initial plan counts as a step
    )

    # Iterate over chain to extract last partition 
    # (doesn't support subscript, i.e., chain[-1])
    for index, current_partition in enumerate(chain):
        if index == len(chain) - 1:
            new_partition = current_partition

    return new_partition


def grid_partition_from_string(partition_string, rows=None, cols=None):
    """
    Given a serialized string representing a partitioned grid graph, 
    constructs a gerrychain Partition object with nodes named 
    1 through rows * cols. 
    """
    assignments = partition_string.split(',')
    n = len(assignments)

    if rows is None and cols is None:
        r = int(math.sqrt(n))
        rows = r
        cols = r
    elif rows is None:
        rows = n // cols
    elif cols is None:
        cols = n // rows

    assert(n == rows * cols)

    graph = build_grid_graph(rows, cols)
    assignment_dict = {i: int(assignments[i-1]) for i in range(1, n + 1)}
    return gerrychain.Partition(graph, assignment_dict)


def load_enumerated_grid_partitions(enumerated_partitions_file):
    """
    Loads enumerated grid partitions from a file produced by 
    https://github.com/zschutzman/enumerator. 

    Returns a list of Partition objects. 
    """
    with open(enumerated_partitions_file, 'r') as f:
        lines = list(f)

    partitions = []
    index = 0
    for line in lines:
        partitions.append(grid_partition_from_string(line))
        index += 1
        # Temporary logging prints for long loads:
        print('Loaded partition {index} of {total}.'.format(index=index, total=len(lines)))

    print('Returning list of partitions.', flush=True)
    return partitions


def search_for_midpoint_special_cases(partitions, distances, fix_start_end=False):
    """
    Given partitions (a list of Partition objects output by 
    load_enumerated_grid_partitions) and 
    distances (a numpy array of pairwise Pereira distances), 
    searches for start/middle/end tuples that 
    achieve certain special cases for the midpoint problem. 

    If fix_start_end is True, then the start/end plans are fixed to be 
    vertical stripes and horizontal stripes, respectively. 
    Otherwise, all possible start/middle/end tuples are considered 
    (where start, middle, and end are distinct).  

    The special cases sought are:

    1. Perfect midpoint: 
            D(start, middle) = D(middle, end) = 0.5 * D(start, end)
    
    2. Equidistant, but slack in triangle inequality:
            D(start, middle) = D(middle, end) > 0.5 * D(start, end)
    
    3. Not equidistant, but triangle inequality is tight:
            D(start, middle) =/= D(middle, end)
            D(start, middle) + D(middle, end) = D(start, end)

    Returns a list of three lists of tuples of indices 
    representing instances of each of the special cases. 

    For example, if the tuple (0, 19, 4) appears in the second list, 
    then plan 19 is equidistant from plans 0 and 4 but has slack in 
    the triangle inequality. 
    """
    n = len(partitions)
    assert(distances.shape[0] == n and distances.shape[1] == n)

    perfect = []
    equidistant = []
    tight_triangle = []
    
    if fix_start_end:
        # Currently hard-coded for enumerated 4x4 and 5x5 grid partitions
        if n == 117: # 4x4
            start_end_pairs = [(95, 21)]
        elif n == 4006: # 5x5
            start_end_pairs = [(1774, 558)]

    else:
        start_end_pairs = [(i, j) for i in range(n) for j in range(n)]
        for i in range(n):
            start_end_pairs.remove((i, i))

    for pair in start_end_pairs:
        start = pair[0]
        end = pair[1]

        for middle in range(n):
            if middle == start or middle == end:
                continue

            is_equidistant = distances[start, middle] == distances[middle, end]
            is_tight_triangle = distances[start, middle] + distances[middle, end] == distances[start, end]

            if is_equidistant and is_tight_triangle:
                perfect.append((start, middle, end))
            elif is_equidistant:
                equidistant.append((start, middle, end))
            elif is_tight_triangle:
                tight_triangle.append((start, middle, end))

    return [perfect, equidistant, tight_triangle]


if __name__ == "__main__":
    ### Test set 1: loading and plotting maps

    # # Test loading of Wisconsin plans
    # gop_plan, dem_plan = get_sample_wi_plans()

    # # Test GerryChain's built-in plot features (using matplotlib)
    # dem_plan.plot()
    # plt.axis('off')
    # plt.title('Democratic Party Gerrymander')
    # plt.show()

    # gop_plan.plot()
    # plt.axis('off')
    # plt.title('Republican Party Gerrymander')
    # plt.show()

    # gop_plan, gop_plan_500_flips = get_sample_wi_plans(num_flips=500)

    # gop_plan_500_flips.plot()
    # plt.axis('off')
    # plt.title('Republican plan after 500 flips')
    # plt.show()


    ### Test set 2: loading enumerated 4x4 grid plans and computing all pairwise distances
    # enumerated_partitions_file = 'enum_[4,4]_[4]_4_rc.txt'
    # partitions = load_enumerated_grid_partitions(enumerated_partitions_file)
    # n = len(partitions)
    # distances = np.zeros((n, n))

    # for i in range(n):
    #     for j in range(n):
    #         distances[i, j] = pereira_index(partitions[i], partitions[j])[0]

    # np.savetxt('4x4_distances.csv', distances, fmt='%.8f', delimiter=',')


    # ### Test set 3: finding special cases of midpoints among 4x4 grid plans
    # enumerated_partitions_file = 'enum_[4,4]_[4]_4_rc.txt'
    # partitions = load_enumerated_grid_partitions(enumerated_partitions_file)
    # distances = np.loadtxt('4x4_distances.csv', delimiter=',')
    # n = len(partitions)

    # # Results where start is vertical stripes and end is horizontal stripes
    # perfect, equidistant, tight_triangle = search_for_midpoint_special_cases(partitions, distances, fix_start_end=True)

    # print('\nPerfect: {0} tuples'.format(len(perfect)))
    # print(perfect)
    # print('\nEquidistant: {0} tuples'.format(len(equidistant)))
    # print(equidistant)
    # print('\nTight triangle: {0} tuples'.format(len(tight_triangle)))
    # print(tight_triangle)

    # perfect_midpoints = [triple[1] for triple in perfect]

    # midpoint_distances = set()
    # for a in perfect_midpoints:
    #     for b in perfect_midpoints:
    #         if b == a:
    #             continue

    #         midpoint_distances.add(distances[a, b])
    #         if distances[a, b] == 0.5:
    #             print('Strange pair (0.5 from each other and from start/end):', a, b)

    # midpoint_distances = list(midpoint_distances)
    # midpoint_distances.sort()

    # # Determine max number of partitions that are all distance 0.5 from each other
    # adjacency_matrix = (distances == 0.5) * 1.
    # G = nx.from_numpy_matrix(adjacency_matrix)
    # clique_number = nx.graph_clique_number(G)
    # max_clique_nodelist, weight = nx.max_weight_clique(G, weight=None)
    # print()
    # print('The maximum number of 4x4 grid partitions that are all distance 0.5 from each other is:', clique_number)
    # print('(From nx.max_weight_clique:', weight, ')')
    # print('These maps are:', max_clique_nodelist)

    # print('\nPerfect midpoints:')
    # for midpoint in perfect_midpoints:
    #     print('Plan #{0}:'.format(midpoint))
    #     draw_grid_plan(partitions[midpoint])

    # print('\nPerfect midpoint distances:')
    # print(midpoint_distances)


    # # Results for all start/end pairs
    # perfect, equidistant, tight_triangle = search_for_midpoint_special_cases(partitions, distances)

    # # Count the number of perfect midpoints for each start/end pair
    # perfect_counts = np.zeros((n, n))
    # for start in range(n):
    #     for end in range(n):
    #         if end == start:
    #             continue

    #         for triple in perfect:
    #             if triple[0] == start and triple[2] == end:
    #                 perfect_counts[start, end] += 1

    # np.savetxt('perfect_counts.csv', perfect_counts, fmt='%d', delimiter=',')


    ### Test set 4: finding special cases of midpoints among 5x5 grid plans
    enumerated_partitions_file = 'enum_[5,5]_[5]_5_rc.txt'
    partitions = load_enumerated_grid_partitions(enumerated_partitions_file)
    n = len(partitions)
    print('Finished loading all {num_partitions} partitions.'.format(num_partitions=n), flush=True)
    
    # # Compute all pairwise distances and save to file
    # distances = np.zeros((n, n))

    # print('Computing distances between all pairs.', flush=True)
    # for i in range(n):
    #     for j in range(n):
    #         distances[i, j] = pereira_index(partitions[i], partitions[j])[0]

    # np.savetxt('5x5_distances.csv', distances, fmt='%.8f', delimiter=',')
    distances = np.loadtxt('5x5_distances.csv', delimiter=',')

    # Results where start is vertical stripes and end is horizontal stripes
    perfect, equidistant, tight_triangle = search_for_midpoint_special_cases(partitions, distances, fix_start_end=True)

    print('\nPerfect: {0} tuples'.format(len(perfect)))
    print(perfect)
    print('\nEquidistant: {0} tuples'.format(len(equidistant)))
    print(equidistant)
    print('\nTight triangle: {0} tuples'.format(len(tight_triangle)))
    print(tight_triangle)

    perfect_midpoints = [triple[1] for triple in perfect]

    midpoint_distances = set()
    for a in perfect_midpoints:
        for b in perfect_midpoints:
            if b == a:
                continue

            midpoint_distances.add(distances[a, b])
            # if distances[a, b] == 0.5:
            #     print('Strange pair (0.5 from each other and from start/end):', a, b)

    midpoint_distances = list(midpoint_distances)
    midpoint_distances.sort()

    # Determine max number of partitions that are all distance 0.5 from each other
    adjacency_matrix = (distances == 0.5) * 1.
    G = nx.from_numpy_matrix(adjacency_matrix)
    clique_number = nx.graph_clique_number(G)
    max_clique_nodelist, weight = nx.max_weight_clique(G, weight=None)
    print()
    print('The maximum number of 5x5 grid partitions that are all distance 0.5 from each other is:', clique_number)
    print('(From nx.max_weight_clique:', weight, ')')
    print('These maps are:', max_clique_nodelist)

    print('\nPerfect midpoints:')
    for midpoint in perfect_midpoints:
        print('Plan #{0}:'.format(midpoint))
        draw_grid_plan(partitions[midpoint])

    print('\nPerfect midpoint distances:')
    print(midpoint_distances)


    # # Results for all start/end pairs (TOO SLOW for 5x5)
    # perfect, equidistant, tight_triangle = search_for_midpoint_special_cases(partitions, distances)

    # # Count the number of perfect midpoints for each start/end pair
    # perfect_counts = np.zeros((n, n))
    # for start in range(n):
    #     for end in range(n):
    #         if end == start:
    #             continue

    #         for triple in perfect:
    #             if triple[0] == start and triple[2] == end:
    #                 perfect_counts[start, end] += 1

    # np.savetxt('5x5_perfect_counts.csv', perfect_counts, fmt='%d', delimiter=',')

    # TODO: add more tests
