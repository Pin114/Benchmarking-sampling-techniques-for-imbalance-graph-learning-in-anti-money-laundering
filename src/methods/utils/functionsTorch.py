from typing import List, Optional, Tuple, Union
import torch
from torch import Tensor
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader, NeighborLoader
from multiprocessing import cpu_count
import sys
import numpy as np

# --- Node2Vec import handling ---
Node2VecPyG = None
Node2VecPure = None
try:
    from torch_geometric.nn import Node2Vec
    Node2VecPyG = Node2Vec
except Exception:
    Node2VecPyG = None

try:
    from node2vec import Node2Vec as Node2VecPure
except Exception:
    Node2VecPure = None


def node2vec_representation_torch(
    G_torch: Data,
    train_mask: Tensor,
    test_mask: Tensor,
    embedding_dim: int = 128,
    walk_length: int = 20,
    context_size: int = 10,
    walks_per_node: int = 10,
    num_negative_samples: int = 1,
    p: float = 1.0,
    q: float = 1.0,
    batch_size: int = 128,
    lr: float = 0.01,
    max_iter: int = 150,
    n_epochs: int = 100,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_obs = G_torch.num_nodes

    # -----------------------------
    # Use PyG Node2Vec if available
    # -----------------------------
    if Node2VecPyG is not None:
        try:
            model = Node2VecPyG(
                G_torch.edge_index,
                embedding_dim=embedding_dim,
                walk_length=walk_length,
                context_size=context_size,
                walks_per_node=walks_per_node,
                num_negative_samples=num_negative_samples,
                p=p,
                q=q,
                sparse=True,
                num_nodes=n_obs,
            ).to(device)
        except Exception as e:
            print(f"PyG Node2Vec failed: {e}. Falling back to pure-Python Node2Vec.")
            model = None
        else:
            num_workers = int(cpu_count() / 2) if sys.platform == "linux" else 0
            batch_size = int(n_obs / 10)
            loader = model.loader(batch_size=batch_size, shuffle=True, num_workers=num_workers)
            optimizer = torch.optim.SparseAdam(model.parameters(), lr=lr)

            def train():
                model.train()
                total_loss = 0
                for pos_rw, neg_rw in loader:
                    optimizer.zero_grad()
                    loss = model.loss(pos_rw.to(device), neg_rw.to(device))
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                return total_loss / len(loader)

            print("Epochs: ", n_epochs)
            for epoch in range(n_epochs):
                loss = train()
            return model

    # -----------------------------------------
    # Fallback: pure-Python Node2Vec + networkx
    # -----------------------------------------
    if Node2VecPure is None:
        raise ImportError("Neither PyG Node2Vec nor node2vec (pure Python) is available")

    import networkx as nx

    edge_index = G_torch.edge_index.cpu().numpy()
    u = edge_index[0]
    v = edge_index[1]
    G_nx = nx.Graph()
    G_nx.add_nodes_from(range(n_obs))
    G_nx.add_edges_from(zip(u.tolist(), v.tolist()))

    # Start with multiple workers but gracefully fall back to single-worker
    # if the worker processes are killed (SIGKILL) or an OOM/segfault occurs.
    n_workers = max(1, int(cpu_count() / 2))
    try:
        n2v = Node2VecPure(
            G_nx,
            dimensions=embedding_dim,
            walk_length=walk_length,
            num_walks=walks_per_node,
            p=p,
            q=q,
            workers=n_workers,
        )
        model_w2v = n2v.fit(window=context_size, min_count=1, batch_words=4)
    except Exception as e:
        print(f"[Warning] Node2Vec (multi-worker={n_workers}) failed: {e}. Retrying with workers=1.")
        try:
            n2v = Node2VecPure(
                G_nx,
                dimensions=embedding_dim,
                walk_length=walk_length,
                num_walks=walks_per_node,
                p=p,
                q=q,
                workers=1,
            )
            model_w2v = n2v.fit(window=context_size, min_count=1, batch_words=4)
        except Exception as e2:
            print(f"[Error] Node2Vec fallback (workers=1) also failed: {e2}")
            raise

    embeds = np.zeros((n_obs, embedding_dim), dtype=float)
    for i in range(n_obs):
        key = str(i)
        if key in model_w2v.wv:
            embeds[i] = model_w2v.wv[key]
        else:
            try:
                embeds[i] = model_w2v.wv[i]
            except Exception:
                embeds[i] = np.zeros(embedding_dim, dtype=float)

    embeds_t = torch.tensor(embeds, dtype=torch.float32)

    class SimpleN2V:
        def __init__(self, emb_tensor: Tensor):
            self._emb = emb_tensor

        def eval(self):
            return None

        def __call__(self):
            return self._emb

    return SimpleN2V(embeds_t)


# -----------------------------
# GNN train/test helpers
# -----------------------------
def train_GNN_old(data: Data, model: nn.Module, loader: DataLoader = None, lr: float = 0.02, train_mask: Tensor = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    if loader is None:
        optimizer.zero_grad()
        y_hat, h = model(data.x, data.edge_index.to(device))
        y = data.y
        loss = criterion(y_hat[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()
    else:
        for batch in loader:
            optimizer.zero_grad()
            out, h = model(batch.x, batch.edge_index.to(device))
            y_hat = out[:batch.batch_size]
            y = batch.y[:batch.batch_size]
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()

    return loss


def test_GNN(data: Data, model: nn.Module, test_mask: Tensor = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    criterion = nn.CrossEntropyLoss()
    out, h = model(data.x, data.edge_index.to(device))
    if test_mask is None:
        y_hat = out
        y = data.y
    else:
        y_hat = out[test_mask].squeeze()
        y = data.y[test_mask].squeeze()
    loss = criterion(y_hat, y)
    return loss
