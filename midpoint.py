#!/usr/bin/python
# --------------------------------------------------------------------
# midpoint.py
# --------------------------------------------------------------------
"""
Finding the midpoint of two political districting plans by 
building and solving a MILP. 
"""

import cplex
from cplex.exceptions import CplexError
import itertools
import numpy as np

from distances import pereira_index, build_grid_graph
from gerrychain import Graph, Partition

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
    D_ab = pereira_index(plan_a, plan_b)

    print('D(A, B) =', D_ab)

    def x_varindex(i, j):
        return i * n + j

    d = np.zeros((n, n)) # Squared distances between units
    x = np.zeros((n,))
    y = np.zeros((n,))
    
    for v in range(n): # Distances for square grid graph are based on x,y coords
        x[v] = v // 4
        y[v] = v % 4

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

    colname_f = ["f{0}_({1},{2})".format(v, edge[0], edge[1]) for v, edge in itertools.product(np.arange(1, n + 1), dir_edges)]
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

    # TODO: incorporate actual populations in avg_pop and pop. balance
    avg_pop = n * 1. / k # TODO: sum of actual pops divided by k
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
                names.append('f{0}_({1},{2})'.format(j + 1, e[0], e[1]))
                coeffs.append(1)
            for e in out_edges[i]:
                names.append('f{0}_({1},{2})'.format(j + 1, e[0], e[1]))
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
                names.append('f{0}_({1},{2})'.format(j + 1, e[0], e[1]))
                coeffs.append(1)

            model.linear_constraints.add(lin_expr=[cplex.SparsePair(names, coeffs)], 
                senses=["L"], rhs=[0])

    # (2d) f^j (\delta^-(j)) = 0
    for j in range(n):
        names = ['f{0}_({1},{2})'.format(j + 1, e[0], e[1]) for e in in_edges[j]]
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
    
    model.linear_constraints.add(lin_expr=[cplex.SparsePair(names, coeffs)], 
        senses=["E"], rhs=[0])

    return model, n


if __name__ == '__main__':
    # Run simple test with vertical/horizontal stripes on r x r grid
    r = 4

    # Vertical stripes:
    graph = build_grid_graph(r, r)
    assignment = {}
    for i in range(1, graph.number_of_nodes() + 1):
        assignment[i] = r if i % r == 0 else i % r
    
    vert_stripes = Partition(graph, assignment)

    # import pdb; pdb.set_trace()

    # Horizontal stripes:
    graph = build_grid_graph(r, r)
    assignment = {}
    for i in range(1, graph.number_of_nodes() + 1):
        assignment[i] = (i - 1) // r + 1

    horiz_stripes = Partition(graph, assignment)

    model, n = build_midpoint_milp(vert_stripes, horiz_stripes)
    print(n)

    try:
        model.solve()
        model.write('midpoint_py.lp')
    except CplexError as exception:
        print(exception)
        sys.exit(-1)

    # Display solution
    print()
    print("Solution status :", model.solution.get_status(), model.solution.status[model.solution.get_status()])
    print("Objective value : {0:.2f}".format(
        model.solution.get_objective_value()))
    print("c = {0:.2f}, d = {1:.2f}".format(model.solution.get_values('c'), model.solution.get_values('d')))
    print()

    print('edges:  ', end='')
    for e in graph.edges:
        print('{0}'.format(e), end=' ')
    print()

    print('alpha: ', end='')
    for e in graph.edges:
        print('{0:6.0f}'.format(model.solution.get_values('alpha{0}'.format(e))), end=' ')
    print()

    print('beta: ', end='')
    for e in graph.edges:
        print('{0:6.0f}'.format(model.solution.get_values('beta{0}'.format(e))), end=' ')
    print()

    print("Districts (center : units)")
    print('-' * 50)
    for j in range(n):
        # if model.solution.get_values(j * n + j) <= 0:
        #     continue

        print("   {0} : ".format(j + 1), end='')
        for i in range(n):
            if model.solution.get_values(i * n + j) > 0:
                print('{0}  '.format(i + 1), end='')
                # pass
            # print('{0}\t'.format(model.solution.get_values(i * n + j)), end='')

        print()


