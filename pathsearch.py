from collections import deque
import math
import time

from compatibility import get_sample_wi_maps
from partitionnode import PartitionNode

# Constants
DEFAULT_POP_BAL_THRESHOLD = 0.05

# Global variables
visited_nodes = 0


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
            return [path, bound]
        elif math.isinf(t):
            print('Elapsed time: {0:.3f} s'.format(time.time() - start_time))
            return 'NOT_FOUND'
        else:
            self.bound = t
        print('Iteration {0} complete.\n\tIteration time: {1:.3f} s'.format(i, time.time() - iter_start_time))
        i += 1

    print('Elapsed time: {0:.3f} s'.format(time.time() - start_time))
    return 'TIME_OUT' # Did not find goal after MAX_ITER iterations


def search(path, cost, bound):
    """
    Helper function for IDA* search.
    """
    global visited_nodes

    node = path[-1]
    visited_nodes += 1

    new_cost = cost + node.distance_heuristic()

    if new_cost > bound:
        return new_cost

    if node.is_target():
        return 'FOUND'

    min_val = float('inf')
    for successor in node.successors():
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
    init_partition, target_partition, gdf = get_sample_wi_maps(num_flips=10)

    root = PartitionNode(
        partition=init_partition,
        target=target_partition,
        map_gdf=gdf,
        compat_property='units')

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
