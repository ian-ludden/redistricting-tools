import numpy as np

from gerrychain import Graph, Partition
from utils import tally_property, get_subset, get_sample_wi_maps


DEBUG = False


class PartitionNode(object):
    """
    Wrapper for MGGG's Partition class to represent 
    a node in the IDA* search.

    Fields:
    parent - PartitionNode from which this node was created as a successor (if any)
    partition - current district map, instance of Partition
    target - target district map, instance of Partition
    compat_property - the property used for the compatibility index ('units' or 'population')
    flip - the single-unit flip used to obtain this node's partition from that of its parent (or None, if parent is None)
    num_parts - the number of districts
    total_pop - the sum of all unit populations
    P_total - the sum of compat_property for all units
    P_self_minus_target - a 2-D numpy array; 
                        entry (i,j) contains the sum of compat_property for all units in 
                        the i-th district of partition and not in the j-th district of target
    P_self_intersect_target - a 2-D numpy array; 
                        entry (i,j) contains the sum of compat_property for all units in both 
                        the i-th district of partition and the j-th district of target
    P_target_minus_self - a 2-D numpy array; 
                        entry (i,j) contains the sum of compat_property for all units in 
                        the i-th district of target and not in the j-th district of partition
    min_P - a 2-D numpy array;
            entry (i,j) contains the minimum of P_self_minus_target[i,j], 
            P_self_intersect_target[i,j], and P_target_minus_self[i,j]
    compatibility - the compatibility index for partition against target
    """
    def __init__(self, partition=None, target=None, 
        map_gdf=None, compat_property='units', 
        parent=None, flip=None):
        if parent is None:
            self._first_time(partition, target, map_gdf, compat_property)
        else:
            self._from_parent(parent, flip)
        
        self.flip = flip


    def _first_time(self, partition, target, map_gdf, compat_property):
        """
        Helper function for __init__ when creating a root PartitionNode, 
        that is, one with no parent. 
        """
        self.partition = partition
        self.target = target
        self.compat_property = compat_property

        self.total_pop = map_gdf["population"].sum()
        self.num_parts = len(self.partition.parts)

        # Initialize fields for maintaining compatibility index
        self.P_self_minus_target = np.zeros((self.num_parts, self.num_parts))
        self.P_target_minus_self = np.zeros((self.num_parts, self.num_parts))
        self.P_self_intersect_target = np.zeros((self.num_parts, self.num_parts))
        self.min_P = np.zeros((self.num_parts, self.num_parts))

        # Tally compatibility property across all units
        reset_gdf = map_gdf.reset_index() # Ensure GEOID is a column
        all_units = set(reset_gdf['GEOID'])
        self.P_total = tally_property(map_gdf, all_units, self.compat_property)

        # Compute individual components of compatibility index 
        for a_part in self.partition.parts:
            i = int(a_part) - 1 # Array indices are one less than district indices
            a = self.partition.parts[a_part]
            
            for b_part in self.target.parts:
                j = int(b_part) - 1 # Array indices are one less than district indices
                b = self.target.parts[b_part]
                a_minus_b = get_subset(a, b, operation='difference')
                a_intersect_b = get_subset(a, b, operation='intersection')
                b_minus_a = get_subset(b, a, operation='difference')

                self.P_self_minus_target[i,j] = tally_property(map_gdf, a_minus_b, self.compat_property)
                self.P_self_intersect_target[i,j] = tally_property(map_gdf, a_intersect_b, self.compat_property)
                self.P_target_minus_self[i,j] = tally_property(map_gdf, b_minus_a, self.compat_property)
                self.min_P[i,j] = min(
                    self.P_self_minus_target[i,j], 
                    self.P_self_intersect_target[i,j], 
                    self.P_target_minus_self[i,j])

        min_P_sum = np.sum(self.min_P)
        self.compatibility = 1 - min_P_sum * 1. / self.P_total


    def _from_parent(self, parent, flip):
        """
        Helper function for __init__ when creating a new PartitionNode
        from a parent by a single flip (one unit across a district boundary). 
        """
        # Directly reuse some static fields from parent
        self.target = parent.target
        self.total_pop = parent.total_pop
        self.P_total = parent.P_total
        self.num_parts = parent.num_parts
        self.compat_property = parent.compat_property

        # Copy compatibility fields before updating
        self.P_self_minus_target = parent.P_self_minus_target.copy()
        self.P_target_minus_self = parent.P_target_minus_self.copy()
        self.P_self_intersect_target = parent.P_self_intersect_target.copy()
        self.min_P = parent.min_P.copy()

        # Apply flip to parent's partition
        self.partition = parent.partition.flip(flip)
        flipped_unit_geoid = list(flip.keys())[0]
        flipped_unit_vertex = self.partition.graph.nodes[flipped_unit_geoid]
        if self.compat_property == 'units':
            P_unit = 1
        else:
            P_unit = flipped_unit_vertex[self.compat_property]

        old_part = int(parent.partition.assignment[flipped_unit_geoid]) - 1
        new_part = int(self.partition.assignment[flipped_unit_geoid]) - 1
        target_part = int(self.target.assignment[flipped_unit_geoid]) - 1

        # Update intersections
        self.P_self_intersect_target[old_part, target_part] = parent.P_self_intersect_target[old_part, target_part] - P_unit
        self.P_self_intersect_target[new_part, target_part] = parent.P_self_intersect_target[new_part, target_part] + P_unit

        # Update differences
        self.P_target_minus_self[target_part, old_part] = parent.P_target_minus_self[target_part, old_part] + P_unit
        self.P_self_minus_target[new_part, target_part] = parent.P_self_minus_target[new_part, target_part] - P_unit

        # Update min_P & compatibility index
        self.min_P[old_part, target_part] = min(
            self.P_self_minus_target[old_part, target_part], 
            self.P_self_intersect_target[old_part, target_part], 
            self.P_target_minus_self[target_part, old_part])
        self.min_P[new_part, target_part] = min(
            self.P_self_minus_target[new_part, target_part],
            self.P_self_intersect_target[new_part, target_part],
            self.P_target_minus_self[target_part, new_part])
        new_min_P_sum = (np.sum(parent.min_P)
                        - parent.min_P[old_part, target_part]
                        - parent.min_P[new_part, target_part]
                        + self.min_P[old_part, target_part]
                        + self.min_P[new_part, target_part])

        self.compatibility = 1 - new_min_P_sum * 1. / self.P_total


    def is_pop_balanced(self, pop_bal_threshold):
        """
        Returns True iff all districts in self.partition have populations within
        the given threshold of the average. 
        """
        district_pops_dict = self.partition["population"]
        ideal_pop = self.total_pop * 1. / self.num_parts

        for district in district_pops_dict:
            district_pop = district_pops_dict[district]
            if (district_pop < (1 - pop_bal_threshold) * ideal_pop
                or district_pop > (1 + pop_bal_threshold) * ideal_pop):
                return False
        
        return True


    def distance_heuristic(self):
        """
        Returns the "incompatibility" index, 
        that is, one minus the compatibility index with the target,
        to be used as a distance heuristic. 
        """
        return 1 - self.compatibility


    def get_dem_vote_shares(self):
        """
        Returns the democratic vote shares in each of the districts as fractions, 
        sorted by increasing district number.
        """
        return [self.partition['votes_dem'][str(i)] / (self.partition['votes_dem'][str(i)] + self.partition['votes_gop'][str(i)]) for i in range(1, self.num_parts + 1)]


    def mean_median_gap(self):
        """
        Computes the difference between 
        the mean democratic vote share and 
        the median democratic vote share, 
        treated as fractions for each of the districts. 

        Returns this difference as a float. 
        """
        dem_fracs = self.get_dem_vote_shares()

        if DEBUG:
            dem_fracs.sort()
            print(dem_fracs)

        return np.mean(dem_fracs) - np.median(dem_fracs)


    def efficiency_gap(self):
        """
        Computes the following ratio (the efficiency gap):
            (total wasted dem votes - total wasted gop votes) / (total votes cast)

        Since votes are fractional due to disaggregation, 
        wasted votes include all votes above 50% in won districts and
        all votes in lost districts. 

        Returns the efficiency gap as a float.
        """
        dem_fracs = self.get_dem_vote_shares()
        total_votes = [(self.partition['votes_dem'][str(i)] + self.partition['votes_gop'][str(i)]) for i in range(1, self.num_parts + 1)]

        wasted_dem_votes = [dem_fracs[i] * total_votes[i] if dem_fracs[i] < 0.5 else (dem_fracs[i] - 0.5) for i in range(len(dem_fracs))]
        wasted_gop_votes = [(1 - dem_fracs[i]) * total_votes[i] if dem_fracs[i] >= 0.5 else (1 - dem_fracs[i] - 0.5) for i in range(len(dem_fracs))]
        eff_gap = (np.sum(wasted_dem_votes) - np.sum(wasted_gop_votes)) / np.sum(total_votes)

        if DEBUG:
            print('Dem vote shares:', dem_fracs)
            print('Total votes:', np.sum(total_votes))
            print('Total wasted Dem votes:', np.sum(wasted_dem_votes))
            print('Total wasted GOP votes:', np.sum(wasted_gop_votes))
            print('Efficiency gap:', eff_gap)

        return eff_gap


    def negative_variance(self):
        """
        Computes the negative variance of the democratic vote shares.

        Returns as a float. 
        """
        dem_fracs = self.get_dem_vote_shares()
        return -1. * np.var(dem_fracs)


    def is_target(self, tol=1e-7):
        """
        Returns True iff this node's partition is 
        within the given compatibility tolerance of 
        perfect compatibility (value of 1) with the target.
        """
        return self.distance_heuristic() <= tol


    def successors(self):
        """
        Returns new PartitionNodes for all possible single-unit flips.
        """
        flips = []
        for edge in self.partition['cut_edges']:
            for index in [0, 1]:
                flipped_node, other_node = edge[index], edge[1 - index]
                flips.append({flipped_node: self.partition.assignment[other_node]})
        
        successors = []
        for flip in flips:
            successors.append(PartitionNode(parent=self, flip=flip))

        return successors


    def __eq__(self, other):
        """
        Returns True if self and other are equal, False otherwise. 

        Two PartitionNode objects are considered equal
        if they have the same districts (partition), 
        up to relabeling the districts.
        """
        if not isinstance(other, PartitionNode):
            return NotImplemented

        self_parts = self.partition.parts
        other_parts = other.partition.parts

        # Compare unit sets for each pair of districts (one from self, one from other)
        # and count number of matches
        count_matches = 0
        for self_part in self_parts:
            for other_part in other_parts:
                if DEBUG:
                    int_set = self_parts[self_part].intersection(other_parts[other_part])
                    if int_set:
                        print('There are {0} shared units between {1} in self and {2} in other.'.format(len(int_set), self_part, other_part))
                if self_parts[self_part] == other_parts[other_part]:
                    count_matches += 1
        
        return count_matches == self.num_parts


if __name__ == '__main__':
    # Test with Wisconsin maps
    init_partition, target_partition, gdf = get_sample_wi_maps()
    
    root = PartitionNode(
        partition=init_partition, 
        target=target_partition,
        map_gdf=gdf,
        compat_property='units')

    cut_edge = ('55073001800', '55073002100') # Just a random cut edge in init_partition

    child = PartitionNode(
        parent=root,
        flip={cut_edge[0]: init_partition.assignment[cut_edge[1]]})

    print('root:\n\t{0}\n\tCompatibility, distance heuristic: {1:.3f}, {2:.3f}\n\tPop bal: {3}'.format(
        root, 
        root.compatibility, 
        root.distance_heuristic(), 
        root.is_pop_balanced(pop_bal_threshold=0.05)))
    print('child:\n\t{0}\n\tCompatibility, distance heuristic: {1:.3f}, {2:.3f}\n\tPop bal: {3}'.format(
        child, 
        child.compatibility, 
        child.distance_heuristic(), 
        child.is_pop_balanced(pop_bal_threshold=0.05)))

