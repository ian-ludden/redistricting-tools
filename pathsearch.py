from collections import deque
import descartes
import math
import numpy as np
import time

from utils import get_sample_wi_maps
from partitionnode import PartitionNode

# Constants
DEFAULT_POP_BAL_THRESHOLD = 0.05

# Global variables
visited_nodes = 0
time_generating_successors = 0


def ida_star(root):
    """Implementation of iterative-deepening-A* (IDA*) search based on 
       https://en.wikipedia.org/wiki/Iterative_deepening_A*.
    
    root - starting PartitionNode
    """
    start_time = time.time()

    bound = root.distance_heuristic()
    path = deque()
    path.append(root)

    MAX_ITER = 100
    i = 0
    while i < MAX_ITER:
        iter_start_time = time.time()
        t = search(path, 0, bound)
        if t == 'FOUND':
            print('Elapsed time: {0:.3f} s'.format(time.time() - start_time))
            print('Time generating successors: {0:.3f} s'.format(time_generating_successors))
            return [path, bound]
        elif math.isinf(t):
            print('Elapsed time: {0:.3f} s'.format(time.time() - start_time))
            print('Time generating successors: {0:.3f} s'.format(time_generating_successors))
            return 'NOT_FOUND'
        else:
            self.bound = t
        print('Iteration {0} complete.\n\tIteration time: {1:.3f} s'.format(i, time.time() - iter_start_time))
        i += 1

    print('Elapsed time: {0:.3f} s'.format(time.time() - start_time))
    print('Time generating successors: {0:.3f} s'.format(time_generating_successors))
    return 'TIME_OUT' # Did not find goal after MAX_ITER iterations


def search(path, cost, bound):
    """
    Helper function for IDA* search.
    """
    global visited_nodes
    global time_generating_successors

    node = path[-1]
    visited_nodes += 1

    new_cost = cost + node.distance_heuristic()

    if new_cost > bound:
        return new_cost

    if node.is_target():
        return 'FOUND'

    current_time = time.time()
    successors = node.successors()
    time_generating_successors += time.time() - current_time
    min_val = float('inf')

    for successor in successors:
        if successor not in path and successor.is_pop_balanced(DEFAULT_POP_BAL_THRESHOLD):
            path.append(successor)
            t = search(path, cost + node_to_node_cost(node, successor), bound)
            if t == 'FOUND':
                return 'FOUND'
            if t < min_val:
                min_val = t
            path.pop()

    return min_val


def node_to_node_cost(node, successor):
    """
    Returns the cost for moving from a given PartitionNode
    to its successor.
    """
    return abs(node.compatibility - successor.compatibility)


if __name__ == '__main__':
    # Test with sample GOP-/Dem.-gerrymandered Wisconsin maps
    init_partition, target_partition, gdf = get_sample_wi_maps(num_flips=30)

    root = PartitionNode(
        partition=init_partition,
        target=target_partition,
        map_gdf=gdf,
        compat_property='population')

    results = ida_star(root)

    if results == 'NOT_FOUND':
        print('No path found.')
    elif results == 'TIME_OUT':
        print('Max iteration reached.')
    else:
        print('Path found.')
        print('\tPath length: {0} maps (including start and end)'.format(len(results[0])))
        print('\tValue (sum of compatibility differences): {0:.5f}'.format(results[1]))
        print('\tInitial incompatibility: {0:.5f}'.format(root.distance_heuristic()))
        print('\tVisited nodes: {0}'.format(visited_nodes))

        path = results[0]
        efficiency_gap = np.zeros((len(path),))
        negative_variance = np.zeros((len(path),))
        mean_median_gap = np.zeros((len(path),))

        for index, partition_node in enumerate(path):
            print(partition_node.get_dem_vote_shares())
            # partition_node.partition.plot()

            efficiency_gap[index] = partition_node.efficiency_gap()
            negative_variance[index] = partition_node.negative_variance()
            mean_median_gap[index] = partition_node.mean_median_gap()



        print('Efficiency gap:\n{0}'.format(np.around(efficiency_gap, decimals=4)))
        print('Negative variance:\n{0}'.format(np.around(negative_variance, decimals=4)))
        print('Mean median gap:\n{0}'.format(np.around(mean_median_gap, decimals=4)))
