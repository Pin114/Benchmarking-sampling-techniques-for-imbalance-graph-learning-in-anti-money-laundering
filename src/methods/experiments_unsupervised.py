# src/methods/experiments_unsupervised.py
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.methods.utils.functionsNetworkX import *
from src.methods.utils.functionsTorch import *
from src.methods.utils.isolation_forest import *

# ======================
# Intrinsic Features
# ======================
def intrinsic_features(ntw, train_mask, test_mask, n_estimators, max_samples, max_features, bootstrap):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    X_train, y_train, X_test, y_test = ntw.get_train_test_split_intrinsic(train_mask, test_mask, device=device)
    
    # combine train and test for Isolation Forest
    X_train = torch.cat((X_train, X_test), 0)
    y_test = torch.cat((y_train, y_test), 0)
    
    X_train = X_train.cpu().detach().numpy()
    y_test = y_test.cpu().detach().numpy()
    
    max_features_int = int(np.ceil(max_features * X_train.shape[1] / 100))
    
    y_pred = isolation_forest(X_train, n_estimators, max_samples, max_features_int, bootstrap)
    ap_score = average_precision_score(y_test, y_pred)
    return ap_score

# ======================
# Positional Features
# ======================
def positional_features(
        ntw, train_mask, test_mask,
        alpha_pr: float, alpha_ppr: float,
        n_estimators: int, max_samples: float,
        max_features: int, bootstrap: bool = False,
        fraud_dict_train: dict = None, fraud_dict_test: dict = None,
        ntw_name: str = None):
    
    # 取得 intrinsic + summary features
    X = ntw.get_features(full=True)
    
    # NetworkX features
    ntw_nx = ntw.get_network_nx()
    features_nx_df = local_features_nx(ntw_nx, alpha_pr, alpha_ppr,
                                       fraud_dict_train=fraud_dict_train,
                                       ntw_name=ntw_name)
    
    # Combine features
    features_df = pd.concat([X, features_nx_df], axis=1)
    features_df["fraud"] = [fraud_dict_test[x] for x in features_df.index]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    '''x_train = torch.tensor(features_df.iloc[train_mask.numpy()].drop(["PSP", "fraud"], axis=1).values,
                           dtype=torch.float32).to(device)
    y_train = torch.tensor(features_df.iloc[train_mask.numpy()]["fraud"].values,
                           dtype=torch.long).to(device)
    
    x_test = torch.tensor(features_df.iloc[test_mask.numpy()].drop(["PSP", "fraud"], axis=1).values,
                          dtype=torch.float32).to(device)
    y_test = torch.tensor(features_df.iloc[test_mask.numpy()]["fraud"].values,
                          dtype=torch.long).to(device)'''
    drop_cols = [col for col in ["PSP", "fraud"] if col in features_df.columns]

    x_train = torch.tensor(features_df.iloc[train_mask.numpy()].drop(drop_cols, axis=1).values,
                        dtype=torch.float32).to(device)
    y_train = torch.tensor(features_df.iloc[train_mask.numpy()]["fraud"].values,
                        dtype=torch.long).to(device)

    x_test = torch.tensor(features_df.iloc[test_mask.numpy()].drop(drop_cols, axis=1).values,
                        dtype=torch.float32).to(device)
    y_test = torch.tensor(features_df.iloc[test_mask.numpy()]["fraud"].values,
                        dtype=torch.long).to(device)

    
    # combine train and test
    X_train_comb = torch.cat((x_train, x_test), 0).cpu().detach().numpy()
    y_test_comb = torch.cat((y_train, y_test), 0).cpu().detach().numpy()
    
    max_features_int = int(np.ceil(max_features * X_train_comb.shape[1] / 100))
    y_pred = isolation_forest(X_train_comb, n_estimators, max_samples, max_features_int, bootstrap)
    
    ap_score = average_precision_score(y_test_comb, y_pred)
    return ap_score

# ======================
# Node2Vec Features
# ======================
def node2vec_features(
        ntw_torch, train_mask, test_mask,
        embedding_dim, walk_length, context_size,
        walks_per_node, num_negative_samples,
        p, q, lr, n_epochs,
        n_estimators, max_samples, max_features, bootstrap):
    
    # Node2Vec embeddings
    model_n2v = node2vec_representation_torch(
        ntw_torch,
        train_mask=train_mask,
        test_mask=test_mask,
        embedding_dim=embedding_dim,
        walk_length=walk_length,
        context_size=context_size,
        walks_per_node=walks_per_node,
        num_negative_samples=num_negative_samples,
        p=p,
        q=q,
        lr=lr,
        n_epochs=n_epochs
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model_n2v.eval()
    x_node2vec = model_n2v().detach().to('cpu')
    x_intrinsic = ntw_torch.x.detach().to('cpu')
    
    # concat intrinsic + node2vec
    x_all = torch.cat((x_node2vec, x_intrinsic), dim=1)
    
    x_train = x_all[train_mask].to(device)
    x_test = x_all[test_mask].to(device)
    y_train = ntw_torch.y[train_mask].to(device)
    y_test = ntw_torch.y[test_mask].to(device)
    
    X_train_comb = torch.cat((x_train, x_test), 0).cpu().detach().numpy()
    y_test_comb = torch.cat((y_train, y_test), 0).cpu().detach().numpy()
    
    max_features_int = int(np.ceil(max_features * X_train_comb.shape[1] / 100))
    y_pred = isolation_forest(X_train_comb, n_estimators, max_samples, max_features_int, bootstrap)
    
    ap_score = average_precision_score(y_test_comb, y_pred)
    return ap_score

# ======================
# Test imports
# ======================
if __name__ == "__main__":
    print("intrinsic_features defined:", callable(intrinsic_features))
    print("positional_features defined:", callable(positional_features))
    print("node2vec_features defined:", callable(node2vec_features))
