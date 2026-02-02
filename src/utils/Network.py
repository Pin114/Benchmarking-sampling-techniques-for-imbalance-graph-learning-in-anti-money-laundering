import torch
from torch_geometric.data import Data
import networkx as nx

# --- Try import networkit, fallback if unavailable ---
try:
    import networkit as nk
    HAS_NK = True
except Exception:
    nk = None
    HAS_NK = False

import numpy as np
import pandas as pd


class network_AML():
    def __init__(self, df_features, df_edges, directed=False,
                 train_mask=None, val_mask=None, test_mask=None,
                 fraud_dict=None, name=None):

        self.name = name
        self.df_features = df_features
        self.df_edges = df_edges
        self.directed = directed

        # Build mapping + edges
        self.nodes, self.edges, self.map_id = self._set_up_network_info()

        # Fraud labels mapped to 0...N indexing
        self.fraud_dict = dict(
            zip(
                df_features["txId"].map(self.map_id),
                df_features["class"]
            )
        )

        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask


    # ----------------------------------------------------
    # Mapping features + edges
    # ----------------------------------------------------
    def _set_up_network_info(self):
        nodes = self.df_features['txId']

        # Map txId → 0..N
        map_id = {j: i for i, j in enumerate(nodes)}

        if self.directed:
            edges = self.df_edges[['txId1', 'txId2']]
        else:
            edges_direct = self.df_edges[['txId1', 'txId2']]
            edges_rev = edges_direct[['txId2', 'txId1']]
            edges_rev.columns = ['txId1', 'txId2']
            edges = pd.concat([edges_direct, edges_rev])

        nodes = nodes.map(map_id)
        edges.txId1 = edges.txId1.map(map_id)
        edges.txId2 = edges.txId2.map(map_id)

        edges = edges.astype(int)
        return nodes, edges, map_id


    # ----------------------------------------------------
    # NETWORKX GRAPH
    # ----------------------------------------------------
    def get_network_nx(self):
        edges_zipped = zip(self.edges['txId1'], self.edges['txId2'])

        G_nx = nx.DiGraph() if self.directed else nx.Graph()
        G_nx.add_nodes_from(self.nodes)
        G_nx.add_edges_from(edges_zipped)
        return G_nx


    # ----------------------------------------------------
    # NETWORKIT GRAPH (with fallback)
    # ----------------------------------------------------
    def get_network_nk(self):
        """
        Try networkit. If unavailable, fallback to NetworkX.
        """
        edges_zipped = zip(self.edges['txId1'], self.edges['txId2'])

        if HAS_NK:
            try:
                G_nk = nk.Graph(len(self.nodes), directed=self.directed)
                for u, v in edges_zipped:
                    G_nk.addEdge(u, v)
                return G_nk
            except Exception:
                print("[Warning] Networkit failed, falling back to NetworkX.")

        # Fallback
        return self.get_network_nx()


    # ----------------------------------------------------
    # PyTorch Geometric Data
    # ----------------------------------------------------
    def get_network_torch(self):
        labels = self.df_features['class']
        features = self.df_features[self.df_features.columns.drop(['txId', 'class'])]

        x = torch.tensor(np.array(features.values, dtype=float), dtype=torch.float)
        if x.size()[1] == 0:
            x = torch.ones(x.size()[0], 1)

        y = torch.tensor(np.array(labels.values, dtype=np.int64), dtype=torch.long)

        # PyG edge_index [2, num_edges]
        edge_index = torch.tensor(np.array(self.edges.values).T, dtype=torch.long)

        data = Data(x=x, y=y, edge_index=edge_index)

        if self.train_mask is not None:
            data.train_mask = torch.tensor(self.train_mask, dtype=torch.bool)
        if self.val_mask is not None:
            data.val_mask = torch.tensor(self.val_mask, dtype=torch.bool)
        if self.test_mask is not None:
            data.test_mask = torch.tensor(self.test_mask, dtype=torch.bool)

        return data


    # ----------------------------------------------------
    # Feature extraction
    # ----------------------------------------------------
    def get_features(self, full=False):
        if self.name == 'elliptic':
            if full:
                columns = [i for i in range(2, 167)]
            else:
                columns = [i for i in range(2, 95)]
        else:
            columns = self.df_features.columns.drop(['txId', 'class'])

        return self.df_features[columns]


    def get_features_torch(self, full=False):
        X = self.get_features(full)
        return torch.tensor(X.values, dtype=torch.float32)


    # ----------------------------------------------------
    # Intrinsic Train/Test split for IF
    # ----------------------------------------------------
    def get_train_test_split_intrinsic(self, train_mask, test_mask, device='cpu'):
        X = self.get_features()
        y = self.df_features['class']

        X_train = X[train_mask.numpy()]
        y_train = y[train_mask.numpy()]

        X_test = X[test_mask.numpy()]
        y_test = y[test_mask.numpy()]

        X_train = torch.tensor(X_train.values, dtype=torch.float32).to(device)
        y_train = torch.tensor(y_train.values, dtype=torch.long).to(device)
        X_test = torch.tensor(X_test.values, dtype=torch.float32).to(device)
        y_test = torch.tensor(y_test.values, dtype=torch.long).to(device)

        return X_train, y_train, X_test, y_test


    # ----------------------------------------------------
    def get_fraud_dict(self):
        return self.fraud_dict

    def get_masks(self):
        return self.train_mask, self.val_mask, self.test_mask
