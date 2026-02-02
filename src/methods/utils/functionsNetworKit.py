import networkx as nx
import pandas as pd

def assign_att(u, att, val):
    att[u] = val

def betweenness_nx(G_nx):
    print("Calculating betweenness...")
    betweenness_full = nx.betweenness_centrality(G_nx, normalized=True)
    print("Done")
    
    psp_list = list(G_nx.nodes())
    betweenness_list = [betweenness_full[u] for u in psp_list]
    
    betweenness_df = pd.DataFrame({"PSP": psp_list, "Betweenness": betweenness_list})
    return betweenness_df

def closeness_nx(G_nx):
    print("Calculating closeness...")
    closeness_full = nx.closeness_centrality(G_nx)
    print("Done")
    
    psp_list = list(G_nx.nodes())
    closeness_list = [closeness_full[u] for u in psp_list]
    
    closeness_df = pd.DataFrame({"PSP": psp_list, "Closeness": closeness_list})
    return closeness_df

def eigenvector_nx(G_nx):
    print("Calculating eigenvector centrality...")
    eigen_full = nx.eigenvector_centrality(G_nx, max_iter=1000)
    print("Done")
    
    psp_list = list(G_nx.nodes())
    eigen_list = [eigen_full[u] for u in psp_list]
    
    eigen_df = pd.DataFrame({"PSP": psp_list, "Eigenvector": eigen_list})
    return eigen_df

def features_nx_calculations(G_nx):
    betweenness = betweenness_nx(G_nx)
    closeness = closeness_nx(G_nx)
    eigenvector = eigenvector_nx(G_nx)
    
    features_df = betweenness.merge(closeness, on="PSP").merge(eigenvector, on="PSP")
    return features_df

def features_nx(G_nx, ntw_name):
    location = 'res/' + ntw_name + '_features_nx.csv'
    try:
        features_df = pd.read_csv(location, index_col=0)
    except FileNotFoundError:
        features_df = features_nx_calculations(G_nx)
        try:
            features_df.to_csv(location)
        except Exception:
            # If saving fails, continue without stopping
            pass

    return features_df


def features_nk(G_nk_or_nx, ntw_name):
    """Compatibility shim for Networkit-based features.
    Accepts either a networkit Graph or a networkx Graph (or an edge-list-like object).
    Attempts to convert networkit -> networkx using nk.nxadapter when available,
    otherwise falls back to treating the input as a networkx graph or building one
    from edges. Returns a pandas DataFrame (same shape as features_nx).
    """
    # Prefer networkx representation for downstream processing
    G_nx = None
    try:
        # Try to import networkit locally — if unavailable this will raise
        import networkit as nk
        # If the object is a networkit Graph, try to convert to networkx
        if hasattr(nk, 'graph') and isinstance(G_nk_or_nx, nk.graph.Graph):
            try:
                # networkit provides an adapter to networkx
                G_nx = nk.nxadapter.toNetworkX(G_nk_or_nx)
            except Exception:
                # If adapter fails, try to build networkx from edges
                try:
                    G_nx = nx.Graph()
                    # networkit Graph doesn't expose a simple iterator in a stable way,
                    # so try common methods; fall back silently if unavailable
                    if hasattr(G_nk_or_nx, 'iterEdges'):
                        for u, v in G_nk_or_nx.iterEdges():
                            G_nx.add_edge(u, v)
                    elif hasattr(G_nk_or_nx, 'edges'):
                        for e in G_nk_or_nx.edges():
                            G_nx.add_edge(e[0], e[1])
                except Exception:
                    G_nx = None
    except Exception:
        # networkit not available — proceed
        G_nx = None

    # If not converted yet, check if input is already networkx
    if G_nx is None:
        if isinstance(G_nk_or_nx, nx.Graph) or isinstance(G_nk_or_nx, nx.DiGraph):
            G_nx = G_nk_or_nx
        else:
            # Try to build networkx from an iterable of edges
            try:
                G_nx = nx.Graph()
                for u, v in G_nk_or_nx:
                    G_nx.add_edge(u, v)
            except Exception:
                raise ValueError("Could not interpret input as a networkit or networkx graph")

    # Reuse the existing NetworkX feature generator
    return features_nx(G_nx, ntw_name)
