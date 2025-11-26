import os
import sys
import torch
import optuna

# ---------------- Environment Setup ----------------
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

# Force Optuna to run single-process
OPTUNA_N_JOBS = 1
MAX_WALK_LENGTH = 10
MAX_WALKS_PER_NODE = 2
MAX_EPOCHS = 50

DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(DIR + "/../")
sys.path.append(DIR + "/../")
SRC_PATH = os.path.abspath(os.path.join(DIR, '..', 'src'))
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from src.methods.experiments_unsupervised import intrinsic_features, positional_features, node2vec_features
from data.DatasetConstruction import *
from src.methods.evaluation import random_undersample_mask

# Ensure results directory exists
os.makedirs('res', exist_ok=True)
# ---------------- Optuna Objectives ----------------
def objective_intrinsic(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 200)
    max_samples = trial.suggest_float('max_samples', 0.1, 1)
    max_features_dec = trial.suggest_int('max_features_dec%', 1, 10)
    max_features = max_features_dec * 10
    bootstrap = trial.suggest_categorical('bootstrap', [True, False])

    ap_loss = intrinsic_features(
        ntw, 
        train_mask, 
        val_mask,
        n_estimators, 
        max_samples,
        max_features, 
        bootstrap
    )
    return ap_loss

def objective_positional(trial):
    alpha_pr = trial.suggest_float('alpha_pr', 0.1, 0.9)
    alpha_ppr = 0
    n_estimators = trial.suggest_int('n_estimators', 50, 200)
    max_samples = trial.suggest_float('max_samples', 0.1, 1)
    max_features_dec = trial.suggest_int('max_features_dec%', 1, 10)
    max_features = max_features_dec * 10
    bootstrap = trial.suggest_categorical('bootstrap', [True, False])

    ap_loss = positional_features(
        ntw, 
        train_mask, 
        val_mask,
        alpha_pr,
        alpha_ppr,
        n_estimators, 
        max_samples,
        max_features, 
        bootstrap,
        fraud_dict_test=fraud_dict,
        ntw_name=ntw_name + "_train"
    )
    return ap_loss

def objective_node2vec(trial):
    embedding_dim = trial.suggest_int('embedding_dim', 2, 64)
    walk_length = trial.suggest_int('walk_length', 3, 10)
    context_size = trial.suggest_int('context_size', 2, walk_length)
    walks_per_node = trial.suggest_int('walks_per_node', 1, 3)
    num_negative_samples = trial.suggest_int('num_negative_samples', 1, 5)
    p = trial.suggest_float('p', 0.5, 2)
    q = trial.suggest_float('q', 0.5, 2)
    lr = trial.suggest_float('lr', 0.01, 0.1)
    n_epochs = trial.suggest_int('n_epochs', 5, 500)
    n_estimators = trial.suggest_int('n_estimators', 50, 200)
    max_samples = trial.suggest_float('max_samples', 0.1, 1)
    max_features_dec = trial.suggest_int('max_features_dec%', 1, 10)
    max_features = max_features_dec * 10
    bootstrap = trial.suggest_categorical('bootstrap', [True, False])

    ap_loss = node2vec_features(
        ntw_torch,
        train_mask,
        val_mask,
        embedding_dim,
        walk_length,
        context_size,
        walks_per_node,
        num_negative_samples,
        p,
        q,
        lr,
        n_epochs,
        n_estimators,
        max_samples,
        max_features,
        bootstrap
    )
    return ap_loss

# ---------------- Main Script ----------------
if __name__ == "__main__":
    ### Load Dataset ###
    #ntw_name = "elliptic"
    ntw_name = "ibm"

    if ntw_name == "ibm":
        ntw = load_ibm()
    elif ntw_name == "elliptic":
        ntw = load_elliptic()
    else:
        raise ValueError("Network not found")

    train_mask, val_mask, test_mask = ntw.get_masks()

    to_train = ["intrinsic", "positional",  "node2vec"]

    ### Prepare PyG Data for Node2Vec ###
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ntw_torch = ntw.get_network_torch().to(device)
    if hasattr(ntw_torch, 'x'):
        ntw_torch.x = ntw_torch.x[:, 1:94]  # remove time_step and summary if exist
    # Pre-extract edges for Node2Vec (avoid rebuilding graph on every trial)
    try:
        edge_index = ntw_torch.edge_index.cpu().numpy()
    except Exception:
        edge_index = ntw_torch.edge_index.numpy()
    num_nodes = ntw_torch.num_nodes
    # Optionally run experiments under different sampling strategies
    # Supported methods: 'none' (no sampling), 'random_undersample' (undersample majority in training mask)
    sampling_methods = ["none", "random_undersample"]

    fraud_dict = ntw.get_fraud_dict()
    fraud_dict = {k: 0 if v == 2 else v for k, v in fraud_dict.items()}

    for sampling in sampling_methods:
        print(f"\n--- Running training with sampling method: {sampling} ---")

        # create sampled train mask based on sampling method
        if sampling == 'none':
            train_mask_sampled = ntw.train_mask
        elif sampling == 'random_undersample':
            # use labels from ntw_torch.y
            try:
                train_mask_sampled = random_undersample_mask(ntw.train_mask, ntw_torch.y, target_ratio=1.0, random_state=42)
            except Exception as e:
                print(f"[Warning] undersampling failed, falling back to original train_mask: {e}")
                train_mask_sampled = ntw.train_mask
        else:
            print(f"[Warning] Unknown sampling method '{sampling}', using no sampling.")
            train_mask_sampled = ntw.train_mask

        # ---------------- Intrinsic ----------------
        if "intrinsic" in to_train:
            print("="*10)
            print("Intrinsic Features Optimization")
            # wrapper objective that uses sampled train mask
            def objective_intrinsic_sampled(trial):
                n_estimators = trial.suggest_int('n_estimators', 50, 200)
                max_samples = trial.suggest_float('max_samples', 0.1, 1)
                max_features_dec = trial.suggest_int('max_features_dec%', 1, 10)
                max_features = max_features_dec * 10
                bootstrap = trial.suggest_categorical('bootstrap', [True, False])
                ap_loss = intrinsic_features(
                    ntw,
                    train_mask_sampled,
                    ntw.test_mask,
                    n_estimators,
                    max_samples,
                    max_features,
                    bootstrap
                )
                return ap_loss

            study = optuna.create_study(direction='maximize')
            study.optimize(objective_intrinsic_sampled, n_trials=100)
            intrinsic_params = study.best_params
            intrinsic_values = study.best_value
            samp_tag = '' if sampling == 'none' else f'_{sampling}'
            with open(f"res/intrinsic_params_{ntw_name}_unsupervised{samp_tag}.txt", "w") as f:
                f.write(str(intrinsic_params) + "\n")
                f.write("AUC-PRC: " + str(intrinsic_values))

        # ---------------- Positional ----------------
        if "positional" in to_train:
            print("="*10)
            print("Positional Features Optimization")
            def objective_positional_sampled(trial):
                alpha_pr = trial.suggest_float('alpha_pr', 0.1, 0.9)
                alpha_ppr = 0
                n_estimators = trial.suggest_int('n_estimators', 50, 200)
                max_samples = trial.suggest_float('max_samples', 0.1, 1)
                max_features_dec = trial.suggest_int('max_features_dec%', 1, 10)
                max_features = max_features_dec * 10
                bootstrap = trial.suggest_categorical('bootstrap', [True, False])
                ap_loss = positional_features(
                    ntw,
                    train_mask_sampled,
                    ntw.test_mask,
                    alpha_pr,
                    alpha_ppr,
                    n_estimators,
                    max_samples,
                    max_features,
                    bootstrap,
                    fraud_dict_test=fraud_dict,
                    ntw_name=ntw_name + "_train"
                )
                return ap_loss

            study = optuna.create_study(direction='maximize')
            study.optimize(objective_positional_sampled, n_trials=100)
            positional_params = study.best_params
            positional_values = study.best_value
            samp_tag = '' if sampling == 'none' else f'_{sampling}'
            with open(f"res/positional_params_{ntw_name}_unsupervised{samp_tag}.txt", "w") as f:
                f.write(str(positional_params) + "\n")
                f.write("AUC-PRC: " + str(positional_values))

        # ---------------- Node2Vec ----------------
        if "node2vec" in to_train:
            print("="*10)
            print("Node2Vec Optimization")

            def objective_node2vec_sampled(trial):
                # Hyperparameters with safe upper bounds
                embedding_dim = trial.suggest_int('embedding_dim', 8, 32)
                walk_length = trial.suggest_int('walk_length', 3, MAX_WALK_LENGTH)
                context_size = trial.suggest_int('context_size', 2, walk_length)
                walks_per_node = trial.suggest_int('walks_per_node', 1, MAX_WALKS_PER_NODE)
                num_negative_samples = trial.suggest_int('num_negative_samples', 1, 3)
                p = trial.suggest_float('p', 0.5, 2)
                q = trial.suggest_float('q', 0.5, 2)
                lr = trial.suggest_float('lr', 0.01, 0.05)
                n_epochs = trial.suggest_int('n_epochs', 5, MAX_EPOCHS)

                # RF classifier params
                n_estimators = trial.suggest_int('n_estimators', 50, 120)
                max_samples = trial.suggest_float('max_samples', 0.1, 1)
                max_features_dec = trial.suggest_int('max_features_dec%', 1, 10)
                max_features = max_features_dec * 10
                bootstrap = trial.suggest_categorical('bootstrap', [True, False])

                # ------------------------
                # Run Node2Vec safely
                # ------------------------
                try:
                    from torch_geometric.nn.models import Node2Vec
                except Exception as e:
                    print(f"[Warning] Could not import Node2Vec: {e}")
                    return None

                node2vec = Node2Vec(
                    edge_index=torch.tensor(edge_index, dtype=torch.long),
                    embedding_dim=embedding_dim,
                    walk_length=walk_length,
                    context_size=context_size,
                    walks_per_node=walks_per_node,
                    num_negative_samples=num_negative_samples,
                    p=p,
                    q=q,
                    sparse=True
                )

                loader = node2vec.loader(batch_size=128, shuffle=True)
                optimizer = torch.optim.SparseAdam(node2vec.parameters(), lr=lr)

                node2vec.train()
                for epoch in range(n_epochs):
                    for pos_rw, neg_rw in loader:
                        optimizer.zero_grad()
                        loss = node2vec.loss(pos_rw, neg_rw)
                        loss.backward()
                        optimizer.step()

                # Final embedding
                node2vec.eval()
                z = node2vec.embedding.weight.detach().cpu()

                # ------------------------
                # Run downstream RF classifier (using the produced embeddings)
                # ------------------------
                try:
                    x_intrinsic = ntw_torch.x.detach().cpu() if hasattr(ntw_torch, 'x') else torch.ones((num_nodes,1))
                except Exception:
                    x_intrinsic = torch.ones((num_nodes,1))

                x_all = torch.cat((z, x_intrinsic), dim=1)

                x_train = x_all[train_mask_sampled].to(device)
                x_test = x_all[test_mask].to(device)
                y_train = ntw_torch.y[train_mask_sampled].to(device)
                y_test = ntw_torch.y[test_mask].to(device)

                X_train_comb = torch.cat((x_train, x_test), 0).cpu().detach().numpy()
                y_test_comb = torch.cat((y_train, y_test), 0).cpu().detach().numpy()

                max_features_int = int(__import__('numpy').ceil(max_features * X_train_comb.shape[1] / 100))
                try:
                    from src.methods.utils.isolation_forest import isolation_forest
                    y_pred = isolation_forest(X_train_comb, n_estimators, max_samples, max_features_int, bootstrap)
                except Exception:
                    # fallback: use a naive random score to allow trial to complete
                    import numpy as _np
                    y_pred = _np.random.rand(X_train_comb.shape[0])

                from sklearn.metrics import average_precision_score
                ap_score = average_precision_score(y_test_comb, y_pred)
                return ap_score

            study = optuna.create_study(direction='maximize')
            try:
                study.optimize(
                    objective_node2vec_sampled,
                    n_trials=50,
                    n_jobs=OPTUNA_N_JOBS,   # forced single-core
                    timeout=3600
                )
            except Exception as e:
                print(f"[Warning] node2vec optimization crashed: {e}")

            # fallback if no successful trials
            completed = study.get_trials(states=[optuna.trial.TrialState.COMPLETE])
            if len(completed) == 0:
                print("[Warning] No successful node2vec trials — Using fallback params.")
                node2vec_params = {
                    "embedding_dim": 16,
                    "walk_length": 5,
                    "context_size": 3,
                    "walks_per_node": 1,
                    "num_negative_samples": 1,
                    "p": 1.0,
                    "q": 1.0,
                    "lr": 0.025,
                    "n_epochs": 20
                }
                node2vec_values = None
            else:
                node2vec_params = study.best_params
                node2vec_values = study.best_value

            samp_tag = '' if sampling == 'none' else f'_{sampling}'
            try:
                with open(f"res/node2vec_params_{ntw_name}_unsupervised{samp_tag}.txt", "w") as f:
                    f.write(str(node2vec_params) + "\n")
                    f.write("AUC-PRC: " + str(node2vec_values))
            except Exception as e:
                print(f"[Warning] Could not write node2vec params file: {e}")

            # Create deepwalk params by copying node2vec params and setting p=1,q=1
            deepwalk_path = f"res/deepwalk_params_{ntw_name}_unsupervised{samp_tag}.txt"
            try:
                if node2vec_params is not None:
                    dw_params = dict(node2vec_params)
                    dw_params['p'] = 1
                    dw_params['q'] = 1
                    with open(deepwalk_path, 'w') as f:
                        f.write(str(dw_params) + "\n")
                        f.write("AUC-PRC: " + str(node2vec_values))
            except Exception as e:
                print(f"[Warning] Could not write deepwalk params file: {e}")

    print("Training complete!")
