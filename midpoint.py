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
import numpy as np

from distances import pereira_index, build_grid_graph
from gerrychain import Graph, Partition

def extract_plan_constants(plan):
    """
    TODO: implement

    Extracts the cut-edge constants of the given district plan. 
    """
    pass


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
    a = extract_plan_constants(plan_a)
    b = extract_plan_constants(plan_b)
    D_ab = pereira_index(plan_a, plan_b)

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
    model.variables.add(obj=list(np.reshape(d, (n**2,))), lb=[0] * n_xvars,
        ub=[1] * n_xvars, names=colname_x, types=["N"] * n_xvars)
    # TODO: remove x variables from objective function

    # Create flow variables. f^v_{ij} is the amount of
    # (nonnegative) flow to the district centered at v
    # through arc ij. 
    # TODO

    # Create y variables...
    # TODO


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
    avg_pop = n * 1. / k
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

    return model, n


if __name__ == '__main__':
    # Run simple test with vertical/horizontal stripes on 4 x 4 grid

    # Vertical stripes:
    graph = build_grid_graph(4, 4)
    assignment = {}
    for i in range(1, graph.number_of_nodes() + 1):
        assignment[i] = 4 if i % 4 == 0 else i % 4
    
    vert_stripes = Partition(graph, assignment)

    # Horizontal stripes:
    graph = build_grid_graph(4, 4)
    assignment = {}
    for i in range(1, graph.number_of_nodes() + 1):
        assignment[i] = (i - 1) // 4 + 1

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
    print("Solution status :", model.solution.get_status())
    print("Objective value : {0:.2f}".format(
        model.solution.get_objective_value()))
    print()

    print("Solution values:")
    for j in range(n):
        # if model.solution.get_values(j * n + j) <= 0:
        #     continue

        print("   {0}: ".format(j + 1), end='')
        for i in range(n):
            if model.solution.get_values(i * n + j) > 0:
                print('{0}  '.format(i + 1), end='')
                # pass
            # print('{0}\t'.format(model.solution.get_values(i * n + j)), end='')

        print()
