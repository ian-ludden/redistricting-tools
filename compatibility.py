import csv
import geopandas
import numpy as np
import os
import pandas as pd

from gerrychain import Graph, Partition


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
        raise Error('Unsupported operation: {}'.format(operation))

    return subset


def tally_property(map_df, units_subset, prop='units'):
    """
    Tallies the value of the given property 
    over a subset of units (indices) in a map (GeoDataFrame). 
    
    See `compatibility` function for supported "prop" values. 
    """
    if prop == 'units': # Easy case; no need to manipulate DataFrame
        return len(units_subset)

    df = map_df.reset_index() # Make sure GEOID is a column for lookup
    units_subset_ser = pd.Series(np.array(list(units_subset)), name='GEOID')

    if prop in df.columns:
        return df.merge(units_subset_ser, on='GEOID')[prop].sum()
    else:
        raise Error('Property \'{}\' is not a column of the given GeoDataFrame.'.format(prop))


def compatibility(map_1, map_2, map_gdf, prop='units'):
    """
    Computes the compatibility index defined by Pereira et al. (2009)
    between the two district maps, given as Partition objects.

    Supported "prop" values for the property by which to measure zone intersections/differnces include: 
     - 'units', the number of units (default)
     - 'POP', the sum of unit populations
    """
    sum_min_pieces = 0
    sum_intersections = 0

    for i in map_1.parts:
        if int(i) < 0:
            continue
        y = map_1.parts[i]
        for j in map_2.parts:
            if int(j) < 0:
                continue
            y_prime = map_2.parts[j]
            y_minus_y_prime = get_subset(y, y_prime, operation='difference')
            y_intersect_y_prime = get_subset(y, y_prime, operation='intersection')
            y_prime_minus_y = get_subset(y_prime, y, operation='difference')

            P_y_minus_y_prime = tally_property(map_gdf, y_minus_y_prime, prop=prop)
            P_y_intersect_y_prime = tally_property(map_gdf, y_intersect_y_prime, prop=prop)
            P_y_prime_minus_y = tally_property(map_gdf, y_prime_minus_y, prop=prop)

            sum_min_pieces += min(P_y_minus_y_prime, P_y_intersect_y_prime, P_y_prime_minus_y)
            sum_intersections += P_y_intersect_y_prime
    print(sum_min_pieces)
    print(sum_intersections)
    return 1 - (sum_min_pieces / sum_intersections)


class CompatibilityManager(object):
    """
    Maintains intermediate values of a computation index calculation
    to support efficient calculation of the compatibility index after a one-swap.

    Fields:
        target_map - the (fixed) target district map with which to calculate compatibility (map B)
        current_map - the (dynamic) current district map (map A)
        num_parts - the number of districts in the map/partition
        P_a_minus_b - a num_parts-by-num_parts array of the property totals 
            for each unit a in the current map, A, minus each unit b in target_map
        P_b_minus_a - a num_parts-by-num_parts array of the property totals 
            for each unit b in target_map minus each unit a in the current map, A
        P_a_int_b - a num_parts-by-num_parts array of the property totals 
            for each unit a in the current map, A, intersected with each unit b in target_map
        min_P - a num_parts-by-num_parts array of the min of P_a_minus_b, P_b_minus_a, and P_a_int_b
        P_total - the (fixed) total sum of the property across all units
        min_P_sum - the (dynamic) sum of all (num_parts)^2 entries in min_P
        compat_index - the current compatibility index (between current_map and target_map)
    """
    def __init__(self, target_map, initial_map, map_gdf, prop='units'):
        super(CompatibilityManager, self).__init__()

        # Initialize fields
        self.target_map = target_map
        self.current_map = initial_map
        self.map_gdf = map_gdf
        self.prop = prop

        # Parts are indexed 1 to num_parts; there may be a dummy part -1 for unassigned units
        self.num_parts = 0
        for part in self.target_map.parts:
            self.num_parts = max(self.num_parts, int(part))

        self.P_a_minus_b = np.zeros((self.num_parts, self.num_parts))
        self.P_b_minus_a = np.zeros((self.num_parts, self.num_parts))
        self.P_a_int_b = np.zeros((self.num_parts, self.num_parts))
        self.min_P = np.zeros((self.num_parts, self.num_parts))

        # Tally property across all units assigned to real districts
        # (i.e., not unassigned; unassigned units are in part '-1')
        all_units = set()
        for part in target_map.parts:
            if int(part) > 0:
                all_units = all_units.union(target_map.parts[part])
        self.P_total = self.tally_property(all_units)

        # Update fields with compatibility values for initial_map
        for a_part in initial_map.parts:
            i = int(a_part) - 1 # Array indices are one less than district indices
            if i < 0: 
                continue
            a = initial_map.parts[a_part]
            for b_part in target_map.parts:
                j = int(b_part) - 1 # Array indices are one less than district indices
                if j < 0: 
                    continue
                b = target_map.parts[b_part]
                a_minus_b = self.get_subset(a, b, operation='difference')
                a_intersect_b = self.get_subset(a, b, operation='intersection')
                b_minus_a = self.get_subset(b, a, operation='difference')

                self.P_a_minus_b[i,j] = self.tally_property(a_minus_b)
                self.P_a_int_b[i,j] = self.tally_property(a_intersect_b)
                self.P_b_minus_a[i,j] = self.tally_property(b_minus_a)
                self.min_P[i,j] = min(self.P_a_minus_b[i,j], self.P_a_int_b[i,j], self.P_b_minus_a[i,j])

        self.min_P_sum = np.sum(self.min_P)
        self.compat_index = 1 - self.min_P_sum * 1. / self.P_total


    def calc_compatibility(self, flip, replace=False):
        """
        Calculates the compatibility (w.r.t. target_map) of 
        the map resulting from the given flip applied to current_map. 

        If replace is True, current_map is replaced by the new map and
        all relevant fields are updated. 
        """
        unit = list(flip.keys())[0]
        p_unit = self.tally_property(set([unit]))
        old_part = int(self.current_map.assignment[unit]) - 1
        new_part = int(flip[unit]) - 1
        target_part = int(self.target_map.assignment[unit]) - 1

        # Update intersections
        P_old_int_target = self.P_a_int_b[old_part, target_part] - p_unit
        P_new_int_target = self.P_a_int_b[new_part, target_part] + p_unit

        # Update differences
        P_target_minus_old = self.P_b_minus_a[target_part, old_part] + p_unit
        P_new_minus_target = self.P_a_minus_b[new_part, target_part] - p_unit

        # Update minima & compatibility index
        min_P_old_target = min(self.P_a_minus_b[old_part, target_part], 
                               P_old_int_target, 
                               P_target_minus_old)
        min_P_new_target = min(P_new_minus_target,
                               P_new_int_target,
                               self.P_b_minus_a[target_part, new_part])
        new_min_P_sum = (self.min_P_sum 
                        - self.min_P[old_part, target_part]
                        - self.min_P[new_part, target_part]
                        + min_P_old_target
                        + min_P_new_target)
        new_compat_index = 1 - new_min_P_sum * 1. / self.P_total

        if replace: # Save updates, including current_map
            self.current_map = self.current_map.flip(flip)
            self.P_a_int_b[old_part, target_part] = P_old_int_target
            self.P_a_int_b[new_part, target_part] = P_new_int_target
            self.P_b_minus_a[target_part, old_part] = P_target_minus_old
            self.P_a_minus_b[new_part, target_part] = P_new_minus_target
            self.min_P[old_part, target_part] = min_P_old_target
            self.min_P[new_part, target_part] = min_P_new_target
            self.min_P_sum = new_min_P_sum
            self.compat_index = new_compat_index
        
        return new_compat_index


    def tally_property(self, units_subset):
        """
        Tallies the value of the given property 
        over a subset of units (indices) in the current district map. 
        """
        if self.prop == 'units': # Easy case; no need to manipulate DataFrame
            return len(units_subset)

        df = self.map_gdf.reset_index() # Make sure GEOID is a column for lookup

        # Turn units_subset into a pandas Series for merging with df
        units_subset_series = pd.Series(list(units_subset), name='GEOID', dtype='str')

        if self.prop in df.columns:
            return df.merge(units_subset_series, on='GEOID')[self.prop].sum()
        else:
            raise Error('Property \'{}\' is not a column of the given GeoDataFrame.'.format(self.prop))


    def get_subset(self, zone_1, zone_2, operation='intersection'):
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
            raise Error('Unsupported operation: {}'.format(operation))
        return subset


if __name__ == "__main__":
    # Test with Wisconsin maps
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
    pop_df = pd.DataFrame(pop_raw, columns=pop_headers).astype({"POP": int})
    pop_df = pop_df.drop(columns=['GEOIDLONG', 'DISPLAYNAME'])
    pop_df = pop_df.set_index('GEOID')
    gdf = gdf.join(pop_df)

    # Load sample Wisconsin maps
    map_1_fname = 'wi-gerrymander-rep.csv' #'icor-wi-03.csv'
    map_1_basic = read_map('./{0}'.format(map_1_fname))
    map_1_gdf = gdf.join(map_1_basic)
    map_1_gdf['district'] = map_1_gdf['district'].fillna(value=-1)

    map_1_graph = Graph.from_geodataframe(map_1_gdf)
    map_1_graph.add_data(map_1_gdf)
    map_1 = Partition(map_1_graph, 'district')

    map_2_fname = 'wi-gerrymander-dem.csv' #'icor-wi-04.csv'
    map_2_basic = read_map('./{0}'.format(map_2_fname))
    map_2_gdf = gdf.join(map_2_basic)
    map_2_gdf['district'] = map_2_gdf['district'].fillna(value=-1)

    map_2_graph = Graph.from_geodataframe(map_2_gdf)
    map_2_graph.add_data(map_2_gdf)
    map_2 = Partition(map_2_graph, 'district')


    # Attempt to construct a CompatibilityManager
    compat_manager = CompatibilityManager(map_1, map_2, map_1_gdf)
    print(compat_manager.compat_index)

    compat_index = compatibility(map_1, map_2, gdf)
    print('Compatibility (w.r.t. no. of units) between {0} and {1}: {2:.3f}'.format(map_1_fname, map_2_fname, compat_index))
    print()

    compat_manager_pop = CompatibilityManager(map_1, map_2, map_1_gdf, prop='POP')
    print(compat_manager_pop.compat_index)

    compat_index_pop = compatibility(map_1, map_2, gdf, prop='POP')
    print('Compatibility (w.r.t. population) between {0} and {1}: {2:.3f}'.format(map_1_fname, map_2_fname, compat_index_pop))
