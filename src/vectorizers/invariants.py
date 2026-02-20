## logging
import logging
logger = logging.getLogger(__name__)

## libraries
import igraph as ig
import numpy as np
import scipy.stats as stats
from scipy.sparse import csc_matrix, diags, eye

## modules
from src.data.helpers import _force_finite

## compute graph invariant vector
class GraphInvariants:

    """
    Desc:
        Computes true graph-wise global invariants for igraph.Graph objects without 
        aggregating individual node or edge properties. Only includes pure classical 
        global features. Guarantees returned values are finite (no inf, -inf, or nan).

    Args:
        graph (igraph.Graph): An igraph object representing the graph.

    Returns:
        dict: Each method returns a dictionary containing the extracted graph-level 
        features. All values are guaranteed to be finite.

    Raises:
        TypeError: If the input is not an igraph.Graph object.
    """

    ## init input
    def __init__(self, graph):
        if not isinstance(graph, ig.Graph):
            raise TypeError("Input graph must be an igraph.Graph object")
        self.graph = graph
        self._degree = None

    ## compute simple (linear-time) graph invariants
    def simple(self) -> dict:
        graph = self.graph
        features = {}

        ## number of nodes and edges (cardinality of graph)
        features['n_nodes'] = graph.vcount()
        features['n_edges'] = graph.ecount()

        ## articulation points (vertex cut set of size 1)
        articulation_points = graph.articulation_points()
        features['n_articulation_points'] = len(articulation_points)

        ## bridges (edges whose removal increases number of components)
        bridges = graph.bridges()
        features['n_bridges'] = len(bridges)

        ## simple pure graph invariants
        return features

    ## compute cohesion (cubic-time) graph invariants
    def cohesion(self) -> dict:
        graph = self.graph
        features = {}

        ## operate on the largest connected component for canonical diameter/radius on disconnected graphs
        H = graph.components().giant() if graph.vcount() > 0 else graph
        if H.vcount() < 2:
            return {
                'diameter': 0.0,
                'radius': 0.0,
                'degeneracy': 0.0,
                'k_core_size': 0.0
            }

        ## diameter (longest shortest-path distance among all vertex pairs)
        try:
            diam = H.diameter(directed = False, unconn = False)
            features['diameter'] = _force_finite(float(diam), 0.0)
        except ig.InternalError:
            features['diameter'] = 0.0
        
        ## radius (minimum eccentricity among vertices)
        try:
            ecc = np.array(H.eccentricity())
            finite_ecc = ecc[np.isfinite(ecc)]
            features['radius'] = _force_finite(float(finite_ecc.min()), 0.0) if finite_ecc.size > 0 else 0.0
        except ig.InternalError:
            features['radius'] = 0.0

        ## degeneracy (largest core number)
        core_nums = graph.coreness()
        if core_nums:
            d = max(core_nums)
            features['degeneracy'] = d
            features['k_core_size'] = core_nums.count(d)
        else:
            features['degeneracy'] = 0.0
            features['k_core_size'] = 0.0

        return features

    ## compute extremal graph invariants
    def extremal(self) -> dict:
        features = {}
        
        ## maximum degree (extremum of degree sequence)
        if self._degree is None:
            self._degree = self.graph.degree()
        degrees = self._degree
        features['maximum_degree'] = max(degrees) if degrees else 0.0

        ## extremal pure graph invariant
        return features

    ## compute statistical graph invariants
    def statistical(self) -> dict:
        graph = self.graph
        features = {}

        ## degree sequence variance
        if self._degree is None:
            self._degree = self.graph.degree()
        degrees = self._degree
        features['degree_variance'] = np.var(degrees) if degrees else 0.0

        ## global clustering coefficient (fraction of triangles among connected triples)
        clustering = graph.transitivity_undirected()
        features['global_clustering'] = _force_finite(clustering, 0.0)

        ## degree assortativity (correlation of connected node degrees)
        assortativity = graph.assortativity_degree(directed = False)
        features['degree_assortativity'] = _force_finite(assortativity, 0.0)

        ## shannon entropy of degree sequence
        if len(degrees) > 0:
            degree_counts = np.bincount(degrees)
            degree_probs = degree_counts[degree_counts > 0] / len(degrees)
            features['degree_entropy'] = -np.sum(degree_probs * np.log(degree_probs + 1e-16))
        else:
            features['degree_entropy'] = 0.0

        ## joint degree entropy (unordered degree pairs across undirected edges)
        if graph.ecount() > 0:
            deg = np.array(degrees)
            ## make each edge’s degree pair unordered by sorting the pair
            sources, targets = np.array(graph.get_edgelist(), dtype=int).T
            deg_pairs = np.sort(np.column_stack((deg[sources], deg[targets])), axis=1)
            pairs, counts = np.unique(deg_pairs, axis = 0, return_counts = True)
            probs = counts / counts.sum()
            features['joint_degree_entropy'] = -np.sum(probs * np.log(probs + 1e-16))
        else:
            features['joint_degree_entropy'] = 0.0

        ## degree skewness (fisher–pearson, bias-corrected)
        if len(degrees) < 3 or len(set(degrees)) <= 1:
            features['degree_skewness'] = 0.0
        else:
            features['degree_skewness'] = _force_finite(stats.skew(degrees, bias = False), 0.0)

        ## degree kurtosis (fisher, bias-corrected)
        if len(degrees) < 4 or len(set(degrees)) <= 1:
            features['degree_kurtosis'] = 0.0
        else:
            features['degree_kurtosis'] = _force_finite(stats.kurtosis(degrees, fisher = True, bias = False), 0.0)

        return features

    ## compute spectral invariants (trace-polynomial, no eigendecomposition)
    def spectral(self) -> dict:
        graph = self.graph
        features = {}
        n_nodes = graph.vcount()
        
        if n_nodes < 2:
            return {
                'normalized_laplacian_second_moment': 0.0,
                'normalized_laplacian_third_moment': 0.0,
                'random_walk_triangle_weight': 0.0,
                'random_walk_fourth_moment': 0.0,
                'adjacency_fourth_moment_per_node': 0.0,
            }
        
        ## precompute degrees and adjacency
        degrees = np.array(graph.degree(), dtype = float)
        n_edges = graph.ecount()
        edges = graph.get_edgelist()
        has_isolated = np.any(degrees == 0)
        
        ## adjacency as sparse matrix
        A = csc_matrix(graph.get_adjacency_sparse(), dtype = float)
        
        ## 1. normalized laplacian second moment: O(E) - scales well
        if n_edges > 0 and not has_isolated:
            edge_array = np.array(edges, dtype = int)
            sum_inv = np.sum(
                1.0 / (degrees[edge_array[:, 0]] * degrees[edge_array[:, 1]])
            )
            features['normalized_laplacian_second_moment'] = _force_finite(
                1.0 + (2.0 / n_nodes) * sum_inv, 0.0
            )
        else:
            features['normalized_laplacian_second_moment'] = 0.0
        
        ## precompute degree-normalized matrices if no isolated vertices
        if n_edges > 0 and not has_isolated:
            D_inv_sqrt = diags(1.0 / np.sqrt(degrees))
            D_inv = diags(1.0 / degrees)
            I = eye(n_nodes, format = 'csr')
        
        ## normalized laplacian third moment: O(E²)
        if n_edges > 0 and not has_isolated:
            L_norm = I - D_inv_sqrt @ A @ D_inv_sqrt
            L2 = L_norm @ L_norm
            
            ## monitor densification
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"L_norm nnz: {L_norm.nnz}, L2 nnz: {L2.nnz}")
            
            L3 = L2 @ L_norm
            trace_L3 = L3.diagonal().sum()
            features['normalized_laplacian_third_moment'] = _force_finite(
                trace_L3 / n_nodes, 0.0
            )
        else:
            features['normalized_laplacian_third_moment'] = 0.0
        
        ## random walk moments: O(E²)
        if n_edges > 0 and not has_isolated:
            P = D_inv @ A
            P2 = P @ P
            
            ## monitor densification
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"P nnz: {P.nnz}, P2 nnz: {P2.nnz}")
            
            ## random walk triangle weight: tr(P³)/n
            P3 = P2 @ P
            trace_P3 = P3.diagonal().sum()
            features['random_walk_triangle_weight'] = _force_finite(
                trace_P3 / n_nodes, 0.0
            )
            
            ## random walk fourth moment: tr(P⁴)/n
            P4 = P2 @ P2
            trace_P4 = P4.diagonal().sum()
            features['random_walk_fourth_moment'] = _force_finite(
                trace_P4 / n_nodes, 0.0
            )
        else:
            features['random_walk_triangle_weight'] = 0.0
            features['random_walk_fourth_moment'] = 0.0
        
        ## adjacency fourth moment
        A2 = A @ A
        
        ## monitor densification
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"A nnz: {A.nnz}, A2 nnz: {A2.nnz}")
        
        A4 = A2 @ A2
        trace_A4 = A4.diagonal().sum()
        features['adjacency_fourth_moment_per_node'] = _force_finite(
            trace_A4 / n_nodes, 0.0
        )
        
        return features

    ## compute all invariants and ensure all are finite
    def all(self):
        features = {}
        features.update(self.simple())
        features.update(self.cohesion())
        features.update(self.extremal())
        features.update(self.statistical())
        features.update(self.spectral())
        
        ## final results check - should never trigger with proper implementation
        for key, value in features.items():
            if not np.isfinite(value):
                raise ValueError(f"Feature '{key}' is non-finite ({value}).")

        return features


## compute graph invariants for fully connected bipartite graph
class BipartiteInvariants:
    """
    Desc:
        Computes analytic graph-wise invariants for a complete bipartite graph
        K_{m,n}. This uses direct mathematical formulas for efficiency and does 
        not require constructing a graph object. It computes the same set of 
        invariants as the GraphInvariants class.

    Args:
        m (int): The number of vertices in the first partition.
        n (int): The number of vertices in the second partition.
    
    Returns:
        dict: Each method returns a dictionary containing the graph-level features.
    """
    
    ## init input
    def __init__(self, m: int, n: int):
        if not isinstance(m, int) or not isinstance(n, int):
            raise TypeError("Inputs m and n must be integers.")
        if m < 0 or n < 0:
            raise ValueError("Inputs m and n must be non-negative.")
        self.m = m
        self.n = n
        self.is_trivial = (self.m == 0 or self.n == 0)
        self.is_star = (self.m == 1 and self.n > 1) or (self.n == 1 and self.m > 1)
        self.is_edge = (self.m == 1 and self.n == 1)
    
    ## compute simple bipartite invariants
    def simple(self) -> dict:
        if self.is_trivial:
            return {'n_nodes': 0, 'n_edges': 0, 'n_articulation_points': 0, 'n_bridges': 0}
        
        ## articulation points: 1 if it's a star graph (k_1,n with n>1), otherwise 0.
        n_articulation = 1 if self.is_star else 0
        
        ## bridges: n if it's a star graph (k_1,n), 1 if it's a single edge (k_1,1), otherwise 0.
        n_bridges = self.n if self.m == 1 else (self.m if self.n == 1 else 0)

        return {
            'n_nodes': self.m + self.n, 
            'n_edges': self.m * self.n,
            'n_articulation_points': n_articulation,
            'n_bridges': n_bridges
        }

    ## compute cohesion bipartite invariants
    def cohesion(self) -> dict:
        if self.is_trivial:
            return {'diameter': 0, 'radius': 0, 'degeneracy': 0, 'k_core_size': 0}
        
        ## diameter: 1 for a single edge (k_1,1), 2 for star graphs and other k_m,n.
        diameter = 1 if self.is_edge else 2
        
        ## radius: 1 for single edge or star graphs, 2 otherwise.
        radius = 1 if self.is_edge or self.is_star else 2
        
        degeneracy = min(self.m, self.n)
        
        ## k-core size: the k-core (for k=min(m,n)) is the entire graph.
        k_core_size = self.m + self.n

        return {
            'diameter': diameter, 
            'radius': radius, 
            'degeneracy': degeneracy,
            'k_core_size': k_core_size
        }

    ## compute extremal bipartite invariants
    def extremal(self) -> dict:
        if self.is_trivial:
            return {'maximum_degree': 0}
        return {'maximum_degree': max(self.m, self.n)}

    ## compute statistical bipartite invariants
    def statistical(self) -> dict:
        if self.is_trivial:
            return {
                'degree_variance': 0.0,
                'global_clustering': 0.0,
                'degree_assortativity': 0.0,
                'degree_entropy': 0.0,
                'joint_degree_entropy': 0.0,
                'degree_skewness': 0.0,
                'degree_kurtosis': 0.0
            }
        
        N = self.m + self.n
        M = self.m * self.n
        
        ## degree variance
        mean_k = (2 * M) / N
        deg_var = ((self.m * (self.n ** 2) + self.n * (self.m ** 2)) / N) - (mean_k ** 2)
        
        ## degree assortativity
        assortativity = -1.0 if self.m != self.n else 0.0
        
        ## degree entropy
        p_m = self.m / N  ## probability of having degree n
        p_n = self.n / N  ## probability of having degree m
        degree_entropy = - (p_m * np.log(p_m) + p_n * np.log(p_n)) if self.m != self.n else -np.log(0.5) * 2
        
        ## joint degree entropy: all edges connect a degree-m node to a degree-n node.
        ## there is only one type of edge, so probability is 1. log(1) = 0.
        joint_degree_entropy = 0.0
        
        ## degree skewness
        if self.m == self.n:
            skewness = 0.0
            kurtosis = 0.0
        else:
            std_dev = np.sqrt(deg_var)
            if std_dev < 1e-9:
                skewness = 0.0
                kurtosis = 0.0
            else:
                term_m = self.m * (self.n - mean_k)**3
                term_n = self.n * (self.m - mean_k)**3
                skewness = (term_m + term_n) / (N * std_dev**3)
                
                ## degree kurtosis (excess)
                ## for a two-point distribution (Bernoulli-like), excess kurtosis is (1/(pq)) - 6
                p = self.m / N
                q = self.n / N
                kurtosis = (1.0 / (p * q)) - 6.0

        return {
            'degree_variance': _force_finite(float(deg_var)),
            'global_clustering': 0.0,
            'degree_assortativity': _force_finite(assortativity),
            'degree_entropy': _force_finite(float(degree_entropy)),
            'joint_degree_entropy': _force_finite(float(joint_degree_entropy)),
            'degree_skewness': _force_finite(float(skewness)),
            'degree_kurtosis': _force_finite(float(kurtosis))
        }

    ## compute spectral invariants (trace-polynomial, no eigendecomposition)
    def spectral(self) -> dict:

        ## trivial cases: no edges
        if self.is_trivial:
            return {
                'normalized_laplacian_second_moment': 0.0,
                'normalized_laplacian_third_moment': 0.0,
                'random_walk_triangle_weight': 0.0,
                'random_walk_fourth_moment': 0.0,
                'adjacency_fourth_moment_per_node': 0.0,
            }
        
        ## non-trivial case: K_{m,n}
        m, n = self.m, self.n
        N = m + n

        ## normalized laplacian second moment: tr(L^2)/N = [4 + (N-2)*1]/N = (N+2)/N
        nl2 = 1.0 + 2.0 / N

        ## normalized laplacian third moment: tr(L^3)/N = [8 + (N-2)*1]/N = (N+6)/N
        nl3 = 1.0 + 6.0 / N

        ## random-walk triangle weight: tr(P^3)/N = 0 for bipartite graphs
        rw3 = 0.0

        ## random-walk fourth moment: tr(P^4)/N
        rw4 = (2.0 * (m/n + n/m)) / N if m > 0 and n > 0 else 0.0

        ## adjacency fourth moment per node: tr(A^4)/N
        a4 = 2.0 * (m**2) * (n**2) / N

        return {
            'normalized_laplacian_second_moment': _force_finite(float(nl2), 0.0),
            'normalized_laplacian_third_moment': _force_finite(float(nl3), 0.0),
            'random_walk_triangle_weight': _force_finite(float(rw3), 0.0),
            'random_walk_fourth_moment': _force_finite(float(rw4), 0.0),
            'adjacency_fourth_moment_per_node': _force_finite(float(a4), 0.0),
        }

    ## compute all bipartite invariants
    def all(self) -> dict:
        if self.is_trivial:
            return {
                'n_nodes': 0,
                'n_edges': 0,
                'n_articulation_points': 0,
                'n_bridges': 0,
                'diameter': 0,
                'radius': 0,
                'degeneracy': 0,
                'k_core_size': 0,
                'maximum_degree': 0,
                'degree_variance': 0.0,
                'global_clustering': 0.0,
                'degree_assortativity': 0.0,
                'degree_entropy': 0.0,
                'joint_degree_entropy': 0.0,
                'degree_skewness': 0.0,
                'degree_kurtosis': 0.0,
                'normalized_laplacian_second_moment': 0.0,
                'normalized_laplacian_third_moment': 0.0,
                'random_walk_triangle_weight': 0.0,
                'random_walk_fourth_moment': 0.0,
                'adjacency_fourth_moment_per_node': 0.0,
            }
        features = {}
        features.update(self.simple())
        features.update(self.cohesion())
        features.update(self.extremal())
        features.update(self.statistical())
        features.update(self.spectral())

        for k, v in features.items():
            if not np.isfinite(v):
                raise ValueError(f"Feature '{k}' is non-finite ({v}).")
        return features
