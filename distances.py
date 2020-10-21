import networkx as nx
import numpy as np
import pandas as pd

from gerrychain import Graph, Partition


def build_grid_graph(rows, cols):
	G = nx.Graph()
	G.add_nodes_from(np.arange(1, rows * cols + 1))

	for i in range(1, rows + 1):
		for j in range(1, cols + 1):
			if i < rows:
				G.add_edge(j + cols * (i - 1), j + cols * i)

			if j < cols:
				G.add_edge(j + cols * (i - 1), j + cols * (i - 1) + 1)

	graph = Graph(G)
	df = pd.DataFrame(graph.nodes)
	df.rename(columns={0: 'Name'}, inplace=True)
	df.set_index('Name', inplace=True)
	df['population'] = 1.
	graph.add_data(df)

	return graph


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


def rand_index(p, q):
	"""
	Given Partition objects p and q, 
	computes and returns the Rand index, i.e., 
	the fraction of pairs for which p and q agree. 

	See section 4.1 of Denoeud and Guenoche (2006). 
	"""
	pnodes = p.graph.nodes()
	qnodes = q.graph.nodes()
	assert(pnodes == qnodes) # Must have same nodes

	r = 0; s = 0; u = 0; v = 0 # Initialize counters

	for x in pnodes:
		px = p.assignment[x]
		qx = q.assignment[x]

		for y in pnodes:
			if x <= y: 
				continue # Count each distinct pair only once

			py = p.assignment[y]
			qy = q.assignment[y]

			r += (px == py and qx == qy)
			s += (not(px == py) and not(qx == qy))
			u += (px == py and not(qx == qy))
			v += (not(px == py) and qx == qy)

	total_pairs = r + s + u + v
	assert(total_pairs == len(pnodes) * (len(pnodes) - 1) / 2)

	index = (r + s) * 1. / (r + s + u + v)
	return index, r, s, u, v


def jaccard_index(p, q):
	"""
	Given Partition objects p and q, 
	computes and returns the Jaccard index, i.e., 
	the fraction of pairs for which p and q agree, 
	excluding pairs separated by both partitions. 

	See section 4.2 of Denoeud and Guenoche (2006). 
	"""
	# Reuse rand_index code, but ignore s
	_, r, s, u, v = rand_index(p, q)
	index = r * 1. / (r + u + v)
	return index, r, s, u, v


if __name__ == '__main__': 
	# Generate sample partitions of grid graphs, compute various distance metrics
	
	# Vertical stripes:
	graph = build_grid_graph(4, 4)
	assignment = {}
	for i in range(1, graph.number_of_nodes() + 1):
		assignment[i] = 4 if i % 4 == 0 else i % 4
	
	vert_stripes = Partition(graph, assignment)
	print(vert_stripes.parts)

	# Horizontal stripes:
	graph = build_grid_graph(4, 4)
	assignment = {}
	for i in range(1, graph.number_of_nodes() + 1):
		assignment[i] = (i - 1) // 4 + 1

	horiz_stripes = Partition(graph, assignment)
	print(horiz_stripes.parts)

	# Squares:
	graph = build_grid_graph(4, 4)
	assignment = {}
	for i in range(1, graph.number_of_nodes() + 1):
		if i in [1, 2, 5, 6]:
			assignment[i] = 1
		elif i in [3, 4, 7, 8]:
			assignment[i] = 2
		elif i in [9, 10, 13, 14]:
			assignment[i] = 3
		else: # i in [11, 12, 15, 16]
			assignment[i] = 4

	squares = Partition(graph, assignment)
	print(squares.parts, '\n')

	print(rand_index(squares, horiz_stripes))
	print(rand_index(squares, vert_stripes))
	print(rand_index(vert_stripes, horiz_stripes))
	print()

	print(jaccard_index(squares, horiz_stripes))
	print(jaccard_index(squares, vert_stripes))
	print(jaccard_index(vert_stripes, horiz_stripes))
	print()

	print(pereira_index(squares, horiz_stripes))
	print(pereira_index(squares, vert_stripes))
	print(pereira_index(vert_stripes, horiz_stripes))
	print()
