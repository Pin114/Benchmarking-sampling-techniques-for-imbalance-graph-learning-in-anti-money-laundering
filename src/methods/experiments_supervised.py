import torch

import torch.nn as nn

from torch.utils.data import DataLoader



from src.methods.utils.functionsNetworkX import *

from src.methods.utils.functionsNetworKit import *

from src.methods.utils.functionsTorch import *

from src.methods.utils.GNN import *

from src.utils.Network import *



from src.methods.utils.decoder import *

from src.methods.evaluation import smote_mask, graph_smote_mask



from sklearn.metrics import average_precision_score



def intrinsic_features(

        ntw, train_mask, test_mask,

        n_layers_decoder, hidden_dim_decoder, lr, n_epochs_decoder

):

    device_decoder = (

        "cuda"

        if torch.cuda.is_available()

        else "mps"

        if torch.backends.mps.is_available()

        else "cpu"

    )



    X_train, y_train, X_test, y_test = ntw.get_train_test_split_intrinsic(train_mask, test_mask, device=device_decoder)



    decoder = Decoder_deep_norm(X_train.shape[1], n_layers_decoder, hidden_dim_decoder).to(device_decoder)



    optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)

    criterion = nn.CrossEntropyLoss()



    for epoch in range(n_epochs_decoder):

        decoder.train()

        optimizer.zero_grad()

        output = decoder(X_train)

        loss = criterion(output, y_train)

        loss.backward()

        optimizer.step()



    decoder.eval()

    y_pred = decoder(X_test)

    y_pred = y_pred.softmax(dim=1)

    ap_score = average_precision_score(y_test.cpu().detach().numpy(), y_pred.cpu().detach().numpy()[:,1])

    return(ap_score)



def positional_features(

        ntw, train_mask, test_mask,

        alpha_pr: float,

        alpha_ppr: float,

        n_epochs_decoder: list, 

        lr: float,

        fraud_dict_train: dict = None, 

        fraud_dict_test: dict = None,

        n_layers_decoder: int = 2,

        hidden_dim_decoder: int = 5,

        ntw_name: str = None,

        use_intrinsic: bool = False

        ):

    

    print("intrinsic and summary: ")

    X = ntw.get_features(full=True)

    

    print("networkx: ")

    ntw_nx = ntw.get_network_nx()

    features_nx_df = local_features_nx(ntw_nx, alpha_pr, alpha_ppr, fraud_dict_train=fraud_dict_train, ntw_name=ntw_name)



    ## Train NetworkKit

    print("networkit: ")

    ntw_nk = ntw.get_network_nk()

    features_nk_df = features_nk(ntw_nk, ntw_name=ntw_name)



    ## Concatenate features

    if use_intrinsic:

        features_df = pd.concat([X, features_nx_df, features_nk_df], axis=1)

    else:

        features_df = pd.concat([features_nx_df, features_nk_df], axis=1)

    features_df["fraud"] = [fraud_dict_test[x] for x in features_df.index]



    device_decoder = (

        "cuda"

        if torch.cuda.is_available()

        else "mps"

        if torch.backends.mps.is_available()

        else "cpu"

    )



    features_df_train = features_df[train_mask.numpy()]


    # FIX 1: Add errors='ignore' in drop function
    # This allows Pandas to ignore errors if PSP or fraud columns don't exist and continue execution
    x_train_features = features_df_train.drop(["PSP", "fraud"], axis=1, errors='ignore').values
    x_train = torch.tensor(x_train_features, dtype=torch.float32).to(device_decoder)
    y_train = torch.tensor(features_df_train["fraud"].values, dtype=torch.long).to(device_decoder)



    features_df_test = features_df[test_mask.numpy()]


    # FIX 2: Add errors='ignore' in drop function for test set
    x_test_features = features_df_test.drop(["PSP", "fraud"], axis=1, errors='ignore').values
    x_test = torch.tensor(x_test_features, dtype=torch.float32).to(device_decoder)
    y_test = torch.tensor(features_df_test["fraud"].values, dtype=torch.long).to(device_decoder)



    decoder = Decoder_deep_norm(x_train.shape[1], n_layers_decoder, hidden_dim_decoder).to(device_decoder)



    optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)

    criterion = nn.CrossEntropyLoss()



    for epoch in range(n_epochs_decoder):

        decoder.train()

        optimizer.zero_grad()

        output = decoder(x_train)

        loss = criterion(output, y_train)

        loss.backward()

        optimizer.step()

        #print(f"Epoch {epoch+1}: Loss: {loss.item()}")

    

    decoder.eval()

    y_pred = decoder(x_test)

    y_pred = y_pred.softmax(dim=1)



    ap_score = average_precision_score(y_test.cpu().detach().numpy(), y_pred.cpu().detach().numpy()[:,1])

    return(ap_score)



def node2vec_features(

        ntw_torch, train_mask, test_mask,

        embedding_dim, 

        walk_length, 

        context_size,

        walks_per_node,

        num_negative_samples,

        p,

        q,

        lr=0.01,

        n_epochs=1,

        n_epochs_decoder=1,

        ntw_nx = None,

        use_torch = False, 

        use_intrinsic=True

):

    if use_torch:

        model_n2v = node2vec_representation_torch(

            ntw_torch,

            train_mask = train_mask,

            test_mask = test_mask,

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

        

        model_n2v.eval()

        x = model_n2v()

        # For ease of use, move both tensors to the same device (cpu will always work)

        x = x.detach().to('cpu')



    else:

        model_n2v = node2vec_representation_nx(

            ntw_nx,

            embedding_dim=embedding_dim,

            walk_length=walk_length,

            context_size=context_size,

            walks_per_node=walks_per_node,

            num_negative_samples=num_negative_samples,

            p=p,

            q=q

            )



        print("Saving node2vec embedding...")

        x_df = pd.DataFrame([model_n2v(n) for n in ntw_nx.nodes()], index=ntw_nx.nodes())

        x = torch.tensor(x_df.values).to('cpu')

    

    x_intrinsic = ntw_torch.x.detach().to('cpu')

    if use_intrinsic:

        x = torch.cat((x, x_intrinsic), 1) # Concatenate node2vec and intrinsic features



    device_decoder = (

            "cuda"

            if torch.cuda.is_available()

            else "mps"

            if torch.backends.mps.is_available()

            else "cpu"

        )



    x_train = x[train_mask].to(device_decoder).squeeze()

    x_test = x[test_mask].to(device_decoder).squeeze()



    y_train = ntw_torch.y[train_mask].to(device_decoder).squeeze()

    y_test = ntw_torch.y[test_mask].to(device_decoder).squeeze()



    decoder = Decoder_deep_norm(x_train.shape[1], 2, 10).to(device_decoder)



    optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)

    criterion = nn.CrossEntropyLoss()



    for epoch in range(n_epochs_decoder):

        decoder.train()

        optimizer.zero_grad()

        output = decoder(x_train)

        loss = criterion(output, y_train)

        loss.backward()

        optimizer.step()

    decoder.eval()

    y_pred = decoder(x_test)

    y_pred = y_pred.softmax(dim=1)



    ap_score = average_precision_score(y_test.cpu().detach().numpy(), y_pred.cpu().detach().numpy()[:,1])

    return(ap_score)



def GNN_features(

        ntw_torch: Data,

        model: nn.Module,

        lr: float,

        n_epochs: int, 

        train_loader: DataLoader =None,

        test_loader: DataLoader =None,

        train_mask: torch.Tensor = None,

        test_mask: torch.Tensor = None,

        use_intrinsic: bool = True

):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)



    #Here training should not be a whole other function, but should be part of this function



    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)  # Define optimizer.

    criterion = nn.CrossEntropyLoss()  # Define loss function.



    def train_GNN_feat():

        model.train()

        if train_loader is None:

            optimizer.zero_grad()

            y_hat, h = model(ntw_torch.x, ntw_torch.edge_index.to(device))

            y = ntw_torch.y.long()

            loss = criterion(y_hat[train_mask.bool()], y[train_mask.bool()])

            loss.backward()

            optimizer.step()

        else:

            loader = train_loader #User-specified loader. Intetended mainly for GraphSAGE.

            for batch in loader:

                optimizer.zero_grad()

                out, h = model(batch.x, batch.edge_index.to(device))

                y_hat = out[:batch.batch_size]

                y = batch.y[:batch.batch_size]

                loss = criterion(y_hat, y)

                loss.backward()

                optimizer.step()



        return(loss)

    

    def train_GNN_ones():

        model.train()

        if train_loader is None:

            optimizer.zero_grad()

            # Vector of ones for training

            ones = torch.ones((ntw_torch.x.shape[0], 1),dtype=torch.float32).to(device)

            y_hat, h = model(ones, ntw_torch.edge_index.to(device))

            y = ntw_torch.y

            loss = criterion(y_hat[train_mask], y[train_mask])

            loss.backward()

            optimizer.step()

        else:

            loader = train_loader #User-specified loader. Intetended mainly for GraphSAGE.

            for batch in loader:

                # Vector of ones for training

                ones = torch.ones((ntw_torch.x.shape[0], 1),dtype=torch.float32).to(device)

                optimizer.zero_grad()

                out, h = model(ones, batch.edge_index.to(device))

                y_hat = out[:batch.batch_size]

                y = batch.y[:batch.batch_size]

                loss = criterion(y_hat, y)

                loss.backward()

                optimizer.step()



        return(loss)



    if use_intrinsic:

        for epoch in range(n_epochs):

            loss_train = train_GNN_feat()

            #print('epoch: ', epoch, 'train loss: ', loss_train.item())

    else:

        for epoch in range(n_epochs):

            loss_train = train_GNN_ones()

            #print('epoch: ', epoch, 'train loss: ', loss_train.item())

    

    model.eval()

    if test_loader is None:

        if use_intrinsic:

            out, h = model(ntw_torch.x, ntw_torch.edge_index.to(device))

        else:

            # Vector of ones for testing

            ones = torch.ones((ntw_torch.x.shape[0], 1),dtype=torch.float32).to(device)

            out, h = model(ones, ntw_torch.edge_index.to(device))



        if test_mask is None: # If no test_mask is provided, use all data

            y_hat = out

            y = ntw_torch.y.long()

        else:

            y_hat = out[test_mask.bool()].squeeze()

            y = ntw_torch.y.long()[test_mask.bool()].squeeze()

    else:

        for batch in test_loader:

            batch = batch.to(device, 'edge_index')

            if use_intrinsic:

                out, h = model(batch.x, batch.edge_index)

            else:

                # Vector of ones for testing

                ones = torch.ones((ntw_torch.x.shape[0], 1),dtype=torch.float32).to(device)

                out, h = model(ones, batch.edge_index)

            y_hat = out[:batch.batch_size]

            y = batch.y[:batch.batch_size]
            
    y_hat = y_hat.softmax(dim=1)

    try:

        ap_score = average_precision_score(y.cpu().detach().numpy(), y_hat.cpu().detach().numpy()[:,1])

    except: # Just test accuarcy (more than one class)

        pred = out.argmax(dim=1)

        test_correct = pred[ntw_torch.test_mask] == ntw_torch.y[ntw_torch.test_mask]  # Check against ground-truth labels.

        ap_score = int(test_correct.sum()) / int(ntw_torch.test_mask.sum())
    
    return ap_score


def intrinsic_features_smote(
        ntw, train_mask, test_mask,
        n_layers_decoder, hidden_dim_decoder, lr, n_epochs_decoder,
        k_neighbors=5, random_state=None
):
    """
    Intrinsic features with SMOTE over-sampling for minority class.
    """
    device_decoder = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    X_train, y_train, X_test, y_test = ntw.get_train_test_split_intrinsic(train_mask, test_mask, device=device_decoder)

    # Apply SMOTE to training data
    # Create a temporary mask for training samples
    train_mask_temp = torch.ones(X_train.shape[0], dtype=torch.bool)
    X_train_smote, y_train_smote, _ = smote_mask(
        train_mask_temp, X_train, y_train, 
        k_neighbors=k_neighbors, 
        random_state=random_state
    )

    # Convert back to torch tensors if needed
    if not isinstance(X_train_smote, torch.Tensor):
        X_train_smote = torch.from_numpy(X_train_smote).to(device=device_decoder, dtype=X_train.dtype)
    if not isinstance(y_train_smote, torch.Tensor):
        y_train_smote = torch.from_numpy(y_train_smote).to(device=device_decoder)

    decoder = Decoder_deep_norm(X_train_smote.shape[1], n_layers_decoder, hidden_dim_decoder).to(device_decoder)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(n_epochs_decoder):
        decoder.train()
        optimizer.zero_grad()
        output = decoder(X_train_smote)
        loss = criterion(output, y_train_smote)
        loss.backward()
        optimizer.step()

    decoder.eval()
    y_pred = decoder(X_test)
    y_pred = y_pred.softmax(dim=1)
    ap_score = average_precision_score(y_test.cpu().detach().numpy(), y_pred.cpu().detach().numpy()[:,1])

    return ap_score


def positional_features_smote(
        ntw, train_mask, test_mask,
        alpha_pr: float,
        alpha_ppr: float,
        n_epochs_decoder: int,
        lr: float,
        fraud_dict_train: dict = None,
        fraud_dict_test: dict = None,
        n_layers_decoder: int = 2,
        hidden_dim_decoder: int = 5,
        ntw_name: str = None,
        use_intrinsic: bool = False,
        k_neighbors: int = 5,
        random_state: int = None
):
    """
    Positional features with SMOTE over-sampling for minority class.
    """
    print("intrinsic and summary: ")
    X = ntw.get_features(full=True)

    print("networkx: ")
    ntw_nx = ntw.get_network_nx()
    features_nx_df = local_features_nx(ntw_nx, alpha_pr, alpha_ppr, fraud_dict_train=fraud_dict_train, ntw_name=ntw_name)

    print("networkit: ")
    ntw_nk = ntw.get_network_nk()
    features_nk_df = features_nk(ntw_nk, ntw_name=ntw_name)

    if use_intrinsic:
        features_df = pd.concat([X, features_nx_df, features_nk_df], axis=1)
    else:
        features_df = pd.concat([features_nx_df, features_nk_df], axis=1)
    features_df["fraud"] = [fraud_dict_test[x] for x in features_df.index]

    device_decoder = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    # Extract train features
    features_df_train = features_df[train_mask.numpy()]
    x_train_features = features_df_train.drop(["PSP", "fraud"], axis=1, errors='ignore').values
    x_train = torch.tensor(x_train_features, dtype=torch.float32).to(device_decoder)
    y_train = torch.tensor(features_df_train["fraud"].values, dtype=torch.long).to(device_decoder)

    # Extract test features
    features_df_test = features_df[test_mask.numpy()]
    x_test_features = features_df_test.drop(["PSP", "fraud"], axis=1, errors='ignore').values
    x_test = torch.tensor(x_test_features, dtype=torch.float32).to(device_decoder)
    y_test = torch.tensor(features_df_test["fraud"].values, dtype=torch.long).to(device_decoder)

    # Apply SMOTE to training data
    train_mask_temp = torch.ones(x_train.shape[0], dtype=torch.bool)
    x_train_smote, y_train_smote, _ = smote_mask(
        train_mask_temp, x_train, y_train,
        k_neighbors=k_neighbors,
        random_state=random_state
    )

    if not isinstance(x_train_smote, torch.Tensor):
        x_train_smote = torch.from_numpy(x_train_smote).to(device=device_decoder, dtype=x_train.dtype)
    if not isinstance(y_train_smote, torch.Tensor):
        y_train_smote = torch.from_numpy(y_train_smote).to(device=device_decoder)

    decoder = Decoder_deep_norm(x_train_smote.shape[1], n_layers_decoder, hidden_dim_decoder).to(device_decoder)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(n_epochs_decoder):
        decoder.train()
        optimizer.zero_grad()
        output = decoder(x_train_smote)
        loss = criterion(output, y_train_smote)
        loss.backward()
        optimizer.step()

    decoder.eval()
    y_pred = decoder(x_test)
    y_pred = y_pred.softmax(dim=1)
    ap_score = average_precision_score(y_test.cpu().detach().numpy(), y_pred.cpu().detach().numpy()[:,1])

    return ap_score


def GNN_features_graphsmote(
        ntw_torch,
        model: nn.Module,
        lr: float,
        n_epochs: int,
        train_loader: DataLoader = None,
        test_loader: DataLoader = None,
        train_mask: torch.Tensor = None,
        test_mask: torch.Tensor = None,
        use_intrinsic: bool = True,
        k_neighbors: int = 5,
        random_state: int = None
):
    """
    GNN features with GraphSMOTE over-sampling for minority class.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Apply GraphSMOTE to expand training data
    x_smote, y_smote, train_mask_smote, edge_index_smote = graph_smote_mask(
        train_mask, ntw_torch.x, ntw_torch.y, ntw_torch.edge_index,
        k_neighbors=k_neighbors,
        random_state=random_state
    )
    
    # Ensure labels are binary (0 and 1 only) - filter out any invalid labels like 2 (unknown)
    if isinstance(y_smote, torch.Tensor):
        valid_labels = torch.isin(y_smote, torch.tensor([0, 1], device=y_smote.device))
        valid_mask = train_mask_smote & valid_labels
    else:
        valid_labels = np.isin(y_smote, [0, 1])
        valid_mask = train_mask_smote & torch.from_numpy(valid_labels).bool()
    
    if not valid_mask.any():
        print("Warning: No valid training samples after filtering. Falling back to no sampling.")
        # Fall back to regular GNN without SMOTE
        return GNN_features(ntw_torch, model, lr, n_epochs, train_loader, test_loader, train_mask, test_mask, use_intrinsic)

    # Create a copy of the graph data with SMOTE-augmented data
    ntw_torch_smote = ntw_torch.clone()
    ntw_torch_smote.x = x_smote.to(device)
    ntw_torch_smote.y = y_smote.long().to(device)
    ntw_torch_smote.edge_index = edge_index_smote.long().to(device)
    train_mask_smote = valid_mask.bool().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    def train_GNN_graphsmote():
        model.train()
        optimizer.zero_grad()
        y_hat, h = model(ntw_torch_smote.x, ntw_torch_smote.edge_index)
        y = ntw_torch_smote.y
        # Double-check: only use samples with valid labels (0 or 1)
        valid_mask_batch = torch.isin(y, torch.tensor([0, 1], device=y.device))
        if valid_mask_batch.any():
            y_hat_filtered = y_hat[train_mask_smote & valid_mask_batch]
            y_filtered = y[train_mask_smote & valid_mask_batch]
            if len(y_filtered) > 0:
                loss = criterion(y_hat_filtered, y_filtered)
                loss.backward()
                optimizer.step()
        else:
            print("Warning: No valid batches for training. Skipping batch.")

    for _ in range(n_epochs):
        train_GNN_graphsmote()

    model.eval()
    # Evaluate on original test set (without SMOTE augmentation)
    y_hat, h = model(ntw_torch.x.to(device), ntw_torch.edge_index.to(device))
    y = ntw_torch.y.to(device)
    y_pred = y_hat[test_mask].softmax(dim=1)

    try:
        ap_score = average_precision_score(y[test_mask].cpu().detach().numpy(), y_pred.cpu().detach().numpy()[:,1])
    except:
        pred = y_hat[test_mask].argmax(dim=1)
        test_correct = pred == y[test_mask]
        ap_score = int(test_correct.sum()) / int(test_mask.sum())

    return ap_score 



    return(ap_score)