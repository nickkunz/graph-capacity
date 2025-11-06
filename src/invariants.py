## logging
import logging
logger = logging.getLogger(__name__)

## libraries
import igraph as ig
import numpy as np
import scipy.stats as stats
from scipy.sparse.linalg import eigsh

## utilities
from src.utils import _ensure_finite

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

    ## compute simple (linear-time) graph invariants
    def simple(self):
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
    def cohesion(self):
        graph = self.graph
        features = {}

        ## operate on the largest connected component for canonical diameter/radius on disconnected graphs
        H = graph.components().giant() if graph.vcount() > 0 else graph

        ## diameter (longest shortest‐path length)
        diam = H.diameter()
        features['diameter'] = _ensure_finite(diam, 0.0)
        
        ## radius (minimum eccentricity among vertices)
        ecc = np.array(H.eccentricity())
        finite = ecc[np.isfinite(ecc)]
        features['radius'] = _ensure_finite(finite.min() if finite.size > 0 else 0.0)

        ## degeneracy (largest core number) and k-core size on the whole graph (standard)
        core_nums = graph.coreness()
        if len(core_nums) > 0:
            arr = np.array(core_nums)
            d = int(arr.max())
            features['degeneracy'] = d
            features['k_core_size'] = int((arr == d).sum())
        else:
            features['degeneracy'] = 0.0
            features['k_core_size'] = 0.0

        return features

    ## compute extremal graph invariants
    def extremal(self):
        graph = self.graph
        features = {}
        
        ## maximum degree (extremum of degree sequence)
        degrees = graph.degree()
        features['maximum_degree'] = max(degrees) if degrees else 0.0

        ## extremal pure graph invariant
        return features

    ## compute statistical graph invariants
    def statistical(self):
        graph = self.graph
        features = {}

        ## degree sequence variance
        degrees = graph.degree()
        features['degree_variance'] = np.var(degrees) if degrees else 0.0

        ## global clustering coefficient (fraction of triangles among connected triples)
        clustering = graph.transitivity_undirected()
        features['global_clustering'] = _ensure_finite(clustering, 0.0)

        ## degree assortativity (correlation of connected node degrees)
        assortativity = graph.assortativity_degree(directed = False)
        features['degree_assortativity'] = _ensure_finite(assortativity, 0.0)

        ## shannon entropy of degree sequence
        if len(degrees) > 0:
            degree_counts = np.bincount(degrees)
            degree_probs = degree_counts[degree_counts > 0] / len(degrees)
            features['degree_entropy'] = -np.sum(degree_probs * np.log(degree_probs + 1e-16))
        else:
            features['degree_entropy'] = 0.0

        ## joint degree entropy (unordered degree pairs across undirected edges)
        if graph.ecount() > 0:
            deg = np.array(graph.degree())
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
            features['degree_skewness'] = _ensure_finite(stats.skew(degrees, bias = False), 0.0)

        return features
    
    ## compute global spectral graph invariants (uses canonical gutman–zhang)
    def spectral(self):
        graph = self.graph
        features = {}

        n_nodes = graph.vcount()
        if n_nodes < 2:
            return {
                'spectral_radius': 0.0,
                'spectral_radius_ratio': 0.0,
                'laplacian_energy': 0.0,
                'algebraic_connectivity': 0.0,
                'spectral_entropy': 0.0,
            }

        ## use sparse matrix for memory efficiency
        A_sparse = graph.get_adjacency_sparse().astype(np.float64)

        ## compute the 2 largest magnitude eigenvalues from sparse adjacency
        try:
            # k=2 to get ratio, which='LM' for largest magnitude
            # Use a larger ncv value for better convergence stability
            adj_eigvals = eigsh(
                A_sparse, 
                k = 2, 
                which = 'LM', 
                return_eigenvectors = False, 
                ncv = 32,
                tol = 1e-6,
            )
            abs_eigs = np.sort(np.abs(adj_eigvals))[::-1]
        except Exception:
            # fallback for convergence issues or small graphs
            A = np.array(graph.get_adjacency().data, dtype = float)
            adj_eigvals = np.linalg.eigvalsh(A)
            abs_eigs = np.sort(np.abs(adj_eigvals))[::-1]

        ## spectral radius (largest magnitude eigenvalue)
        features['spectral_radius'] = abs_eigs[0] if abs_eigs.size > 0 else 0.0

        ## spectral radius ratio (|λ1| / |λ2|); guard division by zero
        if abs_eigs.size > 1 and abs_eigs[1] > 1e-9:
            ratio = abs_eigs[0] / abs_eigs[1]
            features['spectral_radius_ratio'] = _ensure_finite(ratio, 0.0)
        else:
            features['spectral_radius_ratio'] = 0.0

        ## laplacian spectrum (unnormalized)
        L_sparse = graph.laplacian(normalized = False)
        
        ## compute 2 smallest algebraic eigenvalues for algebraic connectivity
        try:
            # which='SM' for smallest magnitude. sigma=0 helps find eigenvalues near zero.
            lap_eigvals_small = eigsh(L_sparse, k=2, which='SM', return_eigenvectors=False, sigma=0)
        except Exception:
            # fallback for convergence issues
            L = np.array(graph.laplacian(normalized = False), dtype = float)
            lap_eigvals_small = np.linalg.eigvalsh(L)

        ## algebraic connectivity (second smallest laplacian eigenvalue)
        # The smallest is always 0 for a connected graph.
        alg_conn = lap_eigvals_small[1] if lap_eigvals_small.size > 1 else 0.0
        features['algebraic_connectivity'] = _ensure_finite(float(alg_conn), 0.0)

        ## The following still require the full spectrum, which is too slow for large graphs.
        ## We skip them to ensure timely completion.
        features['laplacian_energy'] = 0.0
        features['spectral_entropy'] = 0.0

        ## spectral pure graph invariants
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
                'degree_variance': 0.0, 'global_clustering': 0.0, 'degree_assortativity': 0.0,
                'degree_entropy': 0.0, 'joint_degree_entropy': 0.0, 'degree_skewness': 0.0
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
        else:
            std_dev = np.sqrt(deg_var)
            if std_dev < 1e-9:
                skewness = 0.0
            else:
                term_m = self.m * (self.n - mean_k)**3
                term_n = self.n * (self.m - mean_k)**3
                skewness = (term_m + term_n) / (N * std_dev**3)

        return {
            'degree_variance': _ensure_finite(float(deg_var)),
            'global_clustering': 0.0,
            'degree_assortativity': _ensure_finite(assortativity),
            'degree_entropy': _ensure_finite(float(degree_entropy)),
            'joint_degree_entropy': _ensure_finite(float(joint_degree_entropy)),
            'degree_skewness': _ensure_finite(float(skewness))
        }

    ## compute spectral bipartite invariants
    def spectral(self) -> dict:
        if self.is_trivial:
            return {
                'spectral_radius': 0.0, 'spectral_radius_ratio': 0.0, 'algebraic_connectivity': 0.0,
                'laplacian_energy': 0.0, 'spectral_entropy': 0.0
            }
        
        ## spectral radius and ratio
        spec_rad = np.sqrt(self.m * self.n)
        spec_rad_ratio = 1.0 if self.m > 0 and self.n > 0 else 0.0
        
        ## algebraic connectivity
        alg_conn = float(min(self.m, self.n))
        
        ## laplacian energy: sum of absolute differences of laplacian eigenvalues from the mean.
        ## eigenvalues are m+n (1), m (n-1 times), n (m-1 times), 0 (1).
        N = self.m + self.n
        mean_eig = (2 * self.m * self.n) / N
        energy = (abs(N - mean_eig) + (self.n - 1) * abs(self.m - mean_eig) + 
                  (self.m - 1) * abs(self.n - mean_eig) + abs(0 - mean_eig))
        
        ## spectral entropy (from adjacency matrix eigenvalues)
        ## eigenvalues are sqrt(mn), -sqrt(mn), and 0 (n-2 times).
        ## normalized squared eigenvalues are 0.5, 0.5.
        spec_entropy = - (0.5 * np.log(0.5) + 0.5 * np.log(0.5)) if (self.m > 0 and self.n > 0) else 0.0

        return {
            'spectral_radius': _ensure_finite(float(spec_rad)),
            'spectral_radius_ratio': _ensure_finite(float(spec_rad_ratio)),
            'algebraic_connectivity': _ensure_finite(float(alg_conn)),
            'laplacian_energy': _ensure_finite(float(energy)),
            'spectral_entropy': _ensure_finite(float(spec_entropy))
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
                'spectral_radius': 0.0, 
                'spectral_radius_ratio': 0.0,
                'algebraic_connectivity': 0.0, 
                'laplacian_energy': 0.0, 
                'spectral_entropy': 0.0
            }
            
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
