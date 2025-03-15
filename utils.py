import networkx as nx
from networkx.algorithms import approximation as approx


def get_density(g):
    return nx.density(g)


def get_number_of_edges(g):
    return nx.number_of_edges(g)

def get_number_of_nodes(g):
    return nx.number_of_nodes(g)

def get_node_connectivity(g):
    return approx.node_connectivity(g)

def get_avg_clustering(g):
    return approx.average_clustering(g)

def get_closeness_centrality(g):
    
    closeness_centrality = nx.algorithms.closeness_centrality(g)
    return sum(list(closeness_centrality.values()))/(len(list(closeness_centrality.values()))*1.0)

# newely added 

def get_len_local_bridges(g):
    return len(list(nx.algorithms.bridges(g)))
    
def get_transitivity(g):
    return nx.algorithms.transitivity(g)
    
def get_edge_connectivity(g):
    return nx.algorithms.connectivity.edge_connectivity(g)

def get_len_find_cliques(g):
    return len(list(nx.algorithms.find_cliques(g)))

def get_estrada_index(g):
    return nx.algorithms.estrada_index(g)

def get_treewidth_min_degree(g):
    return approx.treewidth_min_degree(g)[0]

def get_diameter(g):
    return nx.diameter(g)


def calculate_indices(g):
    connected_components = [c for c in sorted(nx.connected_components(g), key=len, reverse=True)]
    lcc = connected_components[0]
    g = g.subgraph(lcc)
    d = get_density(g)
    ne = get_number_of_edges(g)
    nn = get_number_of_nodes(g)
    nc = get_node_connectivity(g)
    ac = get_avg_clustering(g)
    cc = get_closeness_centrality(g)
    lb = get_len_local_bridges(g)
    t  = get_transitivity(g)
    ec = get_edge_connectivity(g)
    lc = get_len_find_cliques(g)
    ei = get_estrada_index(g)
    td = get_treewidth_min_degree(g)
    dia = get_diameter(g)
    
    return d,ne,nn,nc,ac,cc, lb,t,ec,lc,ei,td,dia




