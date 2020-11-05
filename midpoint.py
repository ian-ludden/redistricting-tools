#!/usr/bin/python
# --------------------------------------------------------------------
# midpoint.py
# --------------------------------------------------------------------
"""
Find a midpoint of two political districting plans by 
building and solving a MIP (mixed-integer (linear) program). 
"""

import cplex
from cplex.exceptions import CplexError
import itertools
from math import sqrt
import numpy as np
import pandas as pd
import random
import sys
import traceback
import xml.etree.ElementTree as ET

import gerrychain

import helpers
import hybrid

def extract_plan_constants(plan):
    """
    Extracts the cut-edge constants (indicator vector)
    of the given district plan. 
    """
    edges = [e for e in plan.graph.edges()]
    cut_edges = plan.cut_edges
    is_cut_edge = np.zeros(len(edges))
    for index, e in enumerate(edges):
        if e in cut_edges:
            is_cut_edge[index] = 1

    return is_cut_edge


def build_midpoint_milp(plan_a, plan_b, tau=0.03):
    """
    Builds and returns a CPLEX model object representing 
    a MILP for finding the midpoint of the given two district plans.

    The parameter tau is the population balance tolerance
    with a default value of 3%. 
    """
    model = cplex.Cplex()
    model.set_problem_name("midpoint_py")
    # Objective: minimize total moment-of-inertia (from Hess model)
    model.objective.set_sense(model.objective.sense.minimize)

    n = plan_a.graph.number_of_nodes()
    k = len(plan_a.parts)

    n_xvars = n * n
    edges = [e for e in plan_a.graph.edges()]
    a = extract_plan_constants(plan_a)
    b = extract_plan_constants(plan_b)
    D_ab = helpers.pereira_index(plan_a, plan_b)

    def x_varindex(i, j):
        return i * n + j

    d = np.zeros((n, n)) # Squared distances between units
    x = np.zeros((n,))
    y = np.zeros((n,))
    
    for v in range(n): # Distances for square grid graph are based on x,y coords
        x[v] = v // int(sqrt(n))
        y[v] = v % int(sqrt(n))

    for u in range(n):
        for v in range(n):
            d[u, v] = (x[u] - x[v])**2 + (y[u] - y[v])**2

    # Create x variables. x[i, j] is 1 if and only if
    # unit i is assigned to a district whose center is j. 
    colname_x = ["x{0}".format(i + 1) for i in range(n_xvars)]
    model.variables.add(obj=[0] * n_xvars, lb=[0] * n_xvars, # obj=list(np.reshape(d, (n**2,)))
        ub=[1] * n_xvars, names=colname_x, types=["N"] * n_xvars)

    # Create flow variables. f^v_{ij} is the amount of
    # (nonnegative) flow from the district centered at v (if one exists)
    # through edge ij. 
    dir_edges = [] # Consider a bidirected version of the undirected graph
    for e in edges:
        dir_edges.append(e)
        dir_edges.append((e[1], e[0]))

    colname_f = ["f{0}_{1}".format(v, edge) for v, edge in itertools.product(np.arange(1, n + 1), dir_edges)]
    model.variables.add(obj=[0] * len(colname_f), lb=[0] * len(colname_f), 
        names=colname_f)

    # Create y and z variables to represent cut-edges
    colname_y = ["y{0}".format(e) for e in edges]
    model.variables.add(obj=[0] * len(edges), lb=[0] * len(edges), 
        ub=[1] * len(edges), names=colname_y, types=["N"] * len(edges))

    def z_varindex(v, edge_index):
        return v * len(edges) + edge_index

    colname_z = []
    for v in range(n):
        for edge_index in range(len(edges)):
            colname_z.append('z{0}'.format(z_varindex(v, edge_index)))
    model.variables.add(obj=[0] * len(colname_z), lb=[0] * len(colname_z), 
        ub=[1] * len(colname_z), names=colname_z, types=["N" * len(colname_z)])

    # Create slack variables for second objective function
    # with weights of 0.5 each for the objective function
    model.variables.add(obj=[0.5, 0.5], lb=[0, 0], names=['c', 'd'])


    ### Add Hess constraints
    # sum_j x_jj = k (there are exactly k centers)
    indices = [x_varindex(j, j) for j in range(n)]
    coeffs = [1] * n

    model.linear_constraints.add(lin_expr=[cplex.SparsePair(indices, coeffs)],
        senses=["E"], rhs=[k])

    for i in range(n):
        for j in range(n): 
            if j == i:
                continue

            # x_ij <= x_jj for all i,j in V
            indices = [x_varindex(i, j), x_varindex(j, j)]
            coeffs = [1, -1]
            model.linear_constraints.add(lin_expr=[cplex.SparsePair(indices, coeffs)],
                senses="L", rhs=[0])

        # sum_j x_ij = 1 for all i in V (every unit assigned)
        indices = [x_varindex(i, j) for j in range(n)]
        coeffs = [1] * n
        model.linear_constraints.add(lin_expr=[cplex.SparsePair(indices, coeffs)], 
            senses=["E"], rhs=[1])

    # Determine ideal (average) district population and upper/lower bounds
    avg_pop = plan_a.graph.data['population'].sum() / k
    pop_lb = (1 - tau) * avg_pop
    pop_ub = (1 + tau) * avg_pop

    for j in range(n):
        indices = [x_varindex(i, j) for i in range(n)]
        lb_coeffs = [1] * n
        lb_coeffs[j] -= pop_lb # Subtract lower-bound from x_jj coeff
        model.linear_constraints.add(lin_expr=[cplex.SparsePair(indices, lb_coeffs)],
            senses=["G"], rhs=[0])

        ub_coeffs = [1] * n
        ub_coeffs[j] -= pop_ub # Subtract upper-bound from x_jj coeff
        model.linear_constraints.add(lin_expr=[cplex.SparsePair(indices, ub_coeffs)], 
            senses=["L"], rhs=[0])


    ### Add Shirabe flow-based contiguity constraints (using Validi et al. notation)
    # Compute in-/out-adjacency (really, edge) lists for all vertices
    in_edges = [set() for v in range(n)]
    out_edges = [set() for v in range(n)]
    for e in dir_edges:
        in_edges[e[1] - 1].add(e)
        out_edges[e[0] - 1].add(e)

    # (2b) f^j (\delta^-(i)) - f^j (\delta^+(i)) = x_ij (move x_ij to LHS)
    for j in range(n):
        for i in range(n):
            if i == j: continue
            names = [x_varindex(i, j)]
            coeffs = [-1]

            for e in in_edges[i]:
                names.append('f{0}_{1}'.format(j + 1, e))
                coeffs.append(1)
            for e in out_edges[i]:
                names.append('f{0}_{1}'.format(j + 1, e))
                coeffs.append(-1)

            model.linear_constraints.add(lin_expr=[cplex.SparsePair(names, coeffs)], 
                senses=["E"], rhs=[0])

    # (2c) f^j (\delta^-(i)) <= (n - 1) * x_ij (move (n-1) * x_ij to LHS)
    for j in range(n):
        for i in range(n):
            if i == j: continue
            names = [x_varindex(i, j)]
            coeffs = [1 - n] # Subtract (n - 1) x_ij

            for e in in_edges[i]:
                names.append('f{0}_{1}'.format(j + 1, e))
                coeffs.append(1)

            model.linear_constraints.add(lin_expr=[cplex.SparsePair(names, coeffs)], 
                senses=["L"], rhs=[0])

    # (2d) f^j (\delta^-(j)) = 0
    for j in range(n):
        names = ['f{0}_{1}'.format(j + 1, e) for e in in_edges[j]]
        coeffs = [1] * len(names)
        model.linear_constraints.add(lin_expr=[cplex.SparsePair(names, coeffs)], 
            senses=["E"], rhs=[0])


    ### Add cut-edge constraints
    for index, e in enumerate(edges):
        y_name = colname_y[index]
        names = [y_name]
        i, j = e
        i -= 1
        j -= 1

        for v in range(n):
            z_name = colname_z[z_varindex(v, index)]
            names.append(z_name)

            xi_name = colname_x[x_varindex(i, v)]
            xj_name = colname_x[x_varindex(j, v)]
            # z^v_{ij} >= x_{iv} + x_{jv} - 1
            model.linear_constraints.add(lin_expr=[cplex.SparsePair([z_name, xi_name, xj_name], [1, -1, -1])], 
                senses=["G"], rhs=[-1])
            # z^v_{ij} <= x_{iv}
            model.linear_constraints.add(lin_expr=[cplex.SparsePair([z_name, xi_name], [1, -1])], 
                senses=["L"], rhs=[0])
            # z^v_{ij} <= x_{jv}
            model.linear_constraints.add(lin_expr=[cplex.SparsePair([z_name, xj_name], [1, -1])], 
                senses=["L"], rhs=[0])
        
        coeffs = [1] * len(names)
        model.linear_constraints.add(lin_expr=[cplex.SparsePair(names, coeffs)], 
            senses=["E"], rhs=[1])


    ### Add alpha and beta variables and constraints
    colname_alpha = ["alpha{0}".format(e) for e in edges]
    # These variables are included in the objective function
    # to capture D(A, Y) + D(Y, B). Since D(A, B) is constant w.r.t. Y, 
    # we don't need to explicitly include it in the objective function. 
    model.variables.add(obj=[0.5 / len(edges)] * len(colname_alpha), lb=[0] * len(colname_alpha), 
        ub=[1] * len(colname_alpha), names=colname_alpha, types=["N" * len(colname_alpha)])
    colname_beta = ["beta{0}".format(e) for e in edges]
    model.variables.add(obj=[0.5 / len(edges)] * len(colname_beta), lb=[0] * len(colname_beta), 
        ub=[1] * len(colname_beta), names=colname_beta, types=["N" * len(colname_beta)])

    for index, e in enumerate(edges):
        alpha_name = colname_alpha[index]
        beta_name = colname_beta[index]
        y_name = colname_y[index]
        
        for var_name, indicator_vector in zip([alpha_name, beta_name], [a, b]):
            if indicator_vector[index] == 1:
                # alpha/beta_e = 1 XOR y_e = 1 - y_e
                model.linear_constraints.add(lin_expr=[cplex.SparsePair([var_name, y_name], [1, 1])], 
                    senses=["E"], rhs=[1])
            else:
                # alpha/beta_e = 0 XOR y_e = y_e
                model.linear_constraints.add(lin_expr=[cplex.SparsePair([var_name, y_name], [1, -1])], 
                    senses=["E"], rhs=[0])

    ### Add c and d slack variables constraint
    names = ['c', 'd']
    coeffs = [1, -1]
    recip_num_edges = 1. / len(edges)
    neg_recip_num_edges = -1. / len(edges)
    for index, e in enumerate(edges):
        names.append(colname_alpha[index])
        coeffs.append(recip_num_edges)
        names.append(colname_beta[index])
        coeffs.append(neg_recip_num_edges)
    
    # D(A, Y) + c = D(Y, B) + d
    model.linear_constraints.add(lin_expr=[cplex.SparsePair(names, coeffs)], 
        senses=["E"], rhs=[0])

    return model, n


def find_midpoint(plan_a, plan_b, num_hybrids=0, sol_file=None):
    """
    Finds the midpoint of two district plans by building and solving a MIP. 

    Generates num_hybrids randomized hybrid Partition objects
    to warm-start the MIP solver.

    If sol_file is given, it's a path to a .sol XML file
    containing feasible solution(s) used to warm-start the MIP solver. 

    Returns the midpoint plan as a Partition object. 
    """
    model, n = build_midpoint_milp(plan_a, plan_b)

    hybrids = []
    index = 0
    while (index < num_hybrids):
        hybrids.append(hybrid.generate_hybrid(plan_a, plan_b))
        index += 1

    add_warmstarts(model, plan_a, plan_b, hybrids=hybrids, sol_file=sol_file)

    try:
        model.solve()
        model.write('midpoint_py.lp')
        model.solution.write('midpoint_py_solution.sol')

        # Create a Partition object from the model's solution
        graph = plan_a.graph.copy()
        n = plan_a.graph.number_of_nodes()
        nodes = [node for node in graph.nodes()]

        assignment = {}

        def x_varindex(i, j):
            return i * n + j

        district_index = 0
        for i in range(n):
            if model.solution.get_values('x{0}'.format(x_varindex(i, i) + 1)) >= 1:
                district_index += 1

                for j in range(n):
                    if model.solution.get_values('x{0}'.format(x_varindex(j, i) + 1)) >= 1:
                        assignment[nodes[j]] = district_index

        try:
            midpoint = gerrychain.Partition(graph, assignment, updaters={
                'population': gerrychain.updaters.Tally('population')
                }) # The updater {'cut_edges': cut_edges} is included by default)
            midpoint.graph.add_data(plan_a.graph.data)
        except Exception as e:
            print(e)
            traceback.print_exc()
        return helpers.add_assignment_as_district_col(midpoint)

    except CplexError as exception:
        print(sys.exc_info())
        print(exception)
        sys.exit(-1)


def add_warmstarts(model, plan_a, plan_b, hybrids=[], sol_file=None, warmstart_name_prefix=None):
    """
    Wrapper for add_warmstart to support multiple warmstarts. 
    """
    if not hybrids: # If hybrids is empty, still attempt to use sol_file
        add_warmstart(model, plan_a, plan_b, sol_file=sol_file, warmstart_name=warmstart_name_prefix)
        return

    for index, hybrid in enumerate(hybrids):
        warmstart_name = None if warmstart_name_prefix is None else '{0}_{1}'.format(warmstart_name_prefix, index)
        add_warmstart(model, plan_a, plan_b, hybrid=hybrid, sol_file=sol_file, warmstart_name=warmstart_name)


def add_warmstart(model, plan_a, plan_b, hybrid=None, sol_file=None, warmstart_name=None):
    """
    Uses the given hybrid plan as a feasible solution to 
    warm-start the given CPLEX MIP model. 

    If sol_file is provided and hybrid is None, 
    then sol_file is read in as the MIP start(s).

    If both are provided, then hybrid is added to 
    the existing warmstarts in sol_file. 

    model - a Cplex object
    hybrid - a Partition object
    sol_file - the path of a .sol file (XML) with feasible solution(s)
    warmstart_name - a name for the new warm-start produced by hybrid. 
                     If not provided, a name is randomly generated. 
    """
    if sol_file is None and hybrid is None:
        return

    if sol_file is not None and hybrid is None:
        model.MIP_starts.read(sol_file)
        return

    n = hybrid.graph.number_of_nodes()
    m = hybrid.graph.number_of_edges()
    nodes = list(hybrid.graph.nodes())
    edges = list(hybrid.graph.edges())
    a = extract_plan_constants(plan_a)
    b = extract_plan_constants(plan_b)

    var_names = [model.variables.get_names(i) for i in range(model.variables.get_num())]

    # Initialize variable assignments
    x = np.zeros((n, n))
    f = np.zeros((n, 2 * m))
    y = np.ones(m)
    z = np.zeros((m, n))
    c = 0
    d = 0
    alpha = np.zeros(m)
    beta = np.zeros(m)
    
    # 1. Determine the x variable values
    for part in hybrid.parts:
        # For simplicity, make the "minimum" unit in each part the center
        center = min(hybrid.parts[part])
        center_index = nodes.index(center)
        for unit in hybrid.parts[part]:
            unit_index = nodes.index(unit)
            x[unit_index, center_index] = 1

    # 2. Determine the z variable values
    for edge_index, e in enumerate(edges):
        i, j = e
        i_index = nodes.index(i)
        j_index = nodes.index(j)

        for node_index, v in enumerate(nodes):
            # z^v_{ij} should be x_iv AND x_jv
            if (x[i_index, node_index] == 1 and x[j_index, node_index] == 1):
                z[edge_index, node_index] = 1

    # 3. Determine the y variable values
    for edge_index, e in enumerate(edges):
        i, j = e
        i_index = nodes.index(i)
        j_index = nodes.index(j)
        for node_index, v in enumerate(nodes):
            # If v is the center of a district 
            # to which both i and j are assigned, then
            # e is not a cut edge. 
            if z[edge_index, node_index] == 1:
                y[edge_index] = 0.

    # 4. Determine the f (flow) variable values
    f = helpers.compute_feasible_flows(hybrid)

    # 5. Determine the c/d variable values
    dist_a_y = helpers.pereira_index(plan_a, hybrid)[0]
    dist_y_b = helpers.pereira_index(hybrid, plan_b)[0]
    
    # Recall: D(A, Y) + c = D(Y, B) + d
    if dist_a_y < dist_y_b:
        c = dist_y_b - dist_a_y
    else:
        d = dist_a_y - dist_y_b

    # 6. Determine the alpha/beta variable values
    for edge_index, e in enumerate(edges):
        if y[edge_index] + a[edge_index] == 1: # XOR
            alpha[edge_index] = 1.

        if y[edge_index] + b[edge_index] == 1: # XOR
            beta[edge_index] = 1.

    # 7. Save these variable assignments to a MIP start XML file (.sol or .mst)
    warmstarts_file = 'warmstarts.mst' if sol_file is None else sol_file
    tree = helpers.load_warmstarts_xml(file=warmstarts_file)
    cplex_sol = ET.SubElement(tree.getroot(), 'CPLEXSolution')
    cplex_sol.attrib['version'] = '1.2'
    header = ET.SubElement(cplex_sol, 'header')

    # Extract names of existing warm-starts from "header" tag, 
    # the first child of each CPLEXSolution
    warmstart_names = [child[0].attrib.get('solutionName', None) for child in tree.getroot()]
    new_warmstart_name = 'warmstart' if warmstart_name is None else warmstart_name

    # Ensure new warmstart name is unused
    while new_warmstart_name in warmstart_names:
        new_warmstart_name = '{0}_{1:04.0f}'.format(new_warmstart_name, 10000 * random.random())

    header.attrib = {'problemName': 'midpoint_py', 'solutionName': new_warmstart_name}

    variables = ET.SubElement(cplex_sol, 'variables')

    def add_variable_to_xml(var_name, value):
        """
        Creates an xml SubElement of the given parent 
        for the variable with the given name and value, 
        fetching the index from var_names and 
        casting the value to an integer if it is integral. 
        """
        if round(value) == value:
            value = int(value)
        
        variable = ET.SubElement(variables, 'variable')
        variable.attrib = {
            'name': '{0}'.format(var_name),
            'index': '{0}'.format(var_names.index(var_name)),
            'value': '{0}'.format(value)}
        return

    # Write variable assignments
    ## x:
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            var_name = 'x{0}'.format(i * n + j + 1)
            value = x[i, j]
            add_variable_to_xml(var_name, value)                

    ## f: 
    for node_index in range(n):
        flow_node_index = node_index + 1 # In flow variable names, nodes are indexed 1 to n
        for edge_index, edge in enumerate(edges):
            # Edge in given direction
            var_name = 'f{0}_{1}'.format(flow_node_index, edge)
            value = f[node_index, 2 * edge_index]
            add_variable_to_xml(var_name, value)
            
            # Edge in reverse direction
            var_name = var_name = 'f{0}_{1}'.format(flow_node_index, (edge[1], edge[0]))
            value = f[node_index, 2 * edge_index + 1]                
            add_variable_to_xml(var_name, value)

    ## y:
    for edge_index, edge in enumerate(edges):
        var_name = 'y{0}'.format(edge)
        value = y[edge_index]
        add_variable_to_xml(var_name, value)

    ## z:
    for node_index in range(n):
        for edge_index, edge in enumerate(edges):
            var_name = 'z{0}'.format(node_index * m + edge_index)
            value = z[edge_index, node_index]
            add_variable_to_xml(var_name, value)

    ## c/d:
    add_variable_to_xml('c', c)
    add_variable_to_xml('d', d)

    ## alpha/beta:
    for prefix in ['alpha', 'beta']:
        for edge_index, edge in enumerate(edges):
            var_name = '{0}{1}'.format(prefix, edge)
            value = alpha[edge_index] if prefix == 'alpha' else beta[edge_index]
            add_variable_to_xml(var_name, value)

    # Save updated warm-starts XML file
    helpers.save_warmstarts_xml(tree=tree, file=warmstarts_file)

    # 8. Set the start values for the MIP model
    model.MIP_starts.read(warmstarts_file)

    model.parameters.output.intsolfileprefix.set('midpoint_int_solns')
    # ^ uncomment to save feasible integer solutions found during branch & cut to files
    # with the naming scheme [prefix]-[five-digit index, starting at 1].sol
    return


def build_ruler_sequence(start, end, depth=1, num_hybrids=1):
    """
    Recursively build a "ruler sequence" of district plans, 
    that is, find the midpoint of the start and end plans (depth 1), 
    then find the "quarterpoints" as 
    midpoints of the midpoint and each of start/end (depth 2), 
    and so on. 

    Uses num_hybrids auto-generated hybrids as a warm-start
    for use in the midpoint MIP solver. 

    Returns the sequence (list) of plans,
    beginning with start and ending with end. 
    """
    midpoint_plan = find_midpoint(plan_a=start, plan_b=end, num_hybrids=num_hybrids)
    
    # Base case: depth is 1
    if depth == 1:
        return [start, midpoint_plan, end]

    first_half = build_ruler_sequence(start, midpoint_plan, depth=depth - 1, num_hybrids=num_hybrids)
    second_half = build_ruler_sequence(midpoint_plan, end, depth=depth - 1, num_hybrids=num_hybrids)
    return first_half + second_half[1:]


if __name__ == '__main__':
    # Run simple test with vertical/horizontal stripes on r x r grid
    r = 8

    # Flag for toggling custom hybrid
    USE_SPECIAL_HYBRID = False

    # Vertical stripes:
    graph = helpers.build_grid_graph(r, r)
    assignment = {}
    for i in range(1, graph.number_of_nodes() + 1):
        assignment[i] = r if i % r == 0 else i % r
    
    vert_stripes = gerrychain.Partition(graph, assignment, updaters={
        'population': gerrychain.updaters.Tally('population')
        }) # The updater {'cut_edges': cut_edges} is included by default

    helpers.add_assignment_as_district_col(vert_stripes)

    # Horizontal stripes:
    graph = helpers.build_grid_graph(r, r)
    assignment = {}
    for i in range(1, graph.number_of_nodes() + 1):
        assignment[i] = (i - 1) // r + 1

    horiz_stripes = gerrychain.Partition(graph, assignment, updaters={
        'population': gerrychain.updaters.Tally('population')
        }) # The updater {'cut_edges': cut_edges} is included by default

    helpers.add_assignment_as_district_col(horiz_stripes)

    # graph = helpers.build_grid_graph(r, r)
    # assignment = {}

    # # feas_hybrid = hybrid.generate_hybrid(vert_stripes, horiz_stripes, pop_bal_tolerance=0.02)

    # # 4 x 4, squares:
    # if r == 4:
    #     for i in range(1, graph.number_of_nodes() + 1):
    #         if i in [1, 2, 5, 6]:
    #             assignment[i] = 1
    #         elif i in [3, 4, 7, 8]:
    #             assignment[i] = 2
    #         elif i in [9, 10, 13, 14]:
    #             assignment[i] = 3
    #         else: # [11, 12, 15, 16]
    #             assignment[i] = 4

    # # 8 x 8, rectangles:
    # if r == 8:
    #     for i in range(1, graph.number_of_nodes() + 1):
    #         if i in [1, 2, 3, 4, 9, 10, 11, 12]:
    #             assignment[i] = 1
    #         elif i in [5, 6, 7, 8, 13, 14, 15, 16]:
    #             assignment[i] = 2
    #         elif i in [17, 18, 19, 20, 25, 26, 27, 28]:
    #             assignment[i] = 3
    #         elif i in [21, 22, 23, 24, 29, 30, 31, 32]:
    #             assignment[i] = 4
    #         elif i in [33, 34, 41, 42, 49, 50, 57, 58]:
    #             assignment[i] = 5
    #         elif i in [35, 36, 43, 44, 51, 52, 59, 60]:
    #             assignment[i] = 6
    #         elif i in [37, 38, 45, 46, 53, 54, 61, 62]:
    #             assignment[i] = 7
    #         else:
    #             assignment[i] = 8
    
    # if USE_SPECIAL_HYBRID:
    #     feas_hybrid = gerrychain.Partition(graph, assignment, updaters={
    #         'population': gerrychain.updaters.Tally('population')
    #         }) # The updater {'cut_edges': cut_edges} is included by default)
    
    # helpers.draw_grid_plan(feas_hybrid)

    # print('The given hybrid is {0:.2f} from vert_stripes, {1:.2f} from horiz_stripes.\n\n'.format(helpers.pereira_index(feas_hybrid, vert_stripes)[0], helpers.pereira_index(feas_hybrid, horiz_stripes)[0]))
    # midpoint_plan = find_midpoint(vert_stripes, horiz_stripes, hybrid=feas_hybrid, sol_file=None) #hybrid=feas_hybrid, sol_file='midpoint_warmstarts.mst')
    # print('\nThe computed midpoint is {0:.2f} from vert_stripes, {1:.2f} from horiz_stripes.\n\n'.format(helpers.pereira_index(vert_stripes, midpoint_plan)[0], helpers.pereira_index(horiz_stripes, midpoint_plan)[0]))

    # helpers.draw_grid_plan(vert_stripes)
    # helpers.draw_grid_plan(midpoint_plan)
    # helpers.draw_grid_plan(horiz_stripes)
    # print()

    ruler_sequence = build_ruler_sequence(vert_stripes, horiz_stripes, depth=2, num_hybrids=5)
    helpers.draw_grid_plan(ruler_sequence[0]) # should be vert_stripes

    for i in range(1, len(ruler_sequence)):
        distance = helpers.pereira_index(ruler_sequence[i - 1], ruler_sequence[i])[0]
        print('\ndistance: {0:.3f}\n'.format(distance))
        helpers.draw_grid_plan(ruler_sequence[i])

    # print('The given first-quarter hybrid is {0:.2f} from vert_stripes, {1:.2f} from midpoint.\n\n'.format(helpers.pereira_index(feas_hybrid, vert_stripes)[0], helpers.pereira_index(feas_hybrid, midpoint_plan)[0]))
    # firstquarter_hybrid = hybrid.generate_hybrid(vert_stripes, midpoint_plan, pop_bal_tolerance=0.02)
    
    # sol_file = 'warmstarts.mst'

    # firstquarter_plan = find_midpoint(vert_stripes, midpoint_plan, hybrid=firstquarter_hybrid, sol_file=sol_file)
    # print('First quarter (between vert_stripes and midpoint_plan):')
    
    # helpers.draw_grid_plan(firstquarter_plan)
