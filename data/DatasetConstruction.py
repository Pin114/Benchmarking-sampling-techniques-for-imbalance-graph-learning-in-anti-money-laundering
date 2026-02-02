import pandas as pd
import torch
import os
import sys
# Ensure the `src` directory is on sys.path so imports like `utils.Network`
# work when running this script directly.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_PATH = os.path.join(ROOT, 'src')
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)
from utils.Network import network_AML
from tqdm import tqdm


#### Elliptic dataset ####
def load_elliptic():
    data_dir = os.path.join(ROOT, 'data', 'data', 'elliptic_bitcoin_dataset')
    raw_paths = [
        os.path.join(data_dir, 'elliptic_txs_features.csv'),
        os.path.join(data_dir, 'elliptic_txs_edgelist.csv'),
        os.path.join(data_dir, 'elliptic_txs_classes.csv'),
    ]
    feat_df = pd.read_csv(raw_paths[0], header=None)
    edge_df = pd.read_csv(raw_paths[1])
    class_df = pd.read_csv(raw_paths[2])

    columns = {0: 'txId', 1: 'time_step'}
    feat_df = feat_df.rename(columns=columns)

    # Feature matrix x is composed of all columns starting from the time_step column, converted to PyTorch float Tensor
    x = torch.from_numpy(feat_df.loc[:, 'time_step':].values).to(torch.float)

    # There exists 3 different classes in the dataset:
    # 0=licit,  1=illicit, 2=unknown
    mapping = {'unknown': 2, '1': 1, '2': 0}
    class_df['class'] = class_df['class'].map(mapping)
    y = torch.from_numpy(class_df['class'].values)
    feat_df["class"] = y

    # Timestamp based split: Use time_step column for time-series split, exclude unlabeled transactions (class=2)
    # Train Mask: time_step < 30 and labeled
    # Val Mask: 30 <= time_step < 40 and labeled
    # Test Mask: time_step >= 40 and labeled

    time_step = torch.from_numpy(feat_df['time_step'].values)
    train_mask = (time_step < 30) & (y != 2)
    val_mask = (time_step >= 30) & (time_step < 40) & (y != 2) 
    test_mask = (time_step >= 40) & (y != 2)
    
    ntw = network_AML(feat_df, edge_df, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask, name='elliptic')

    return(ntw)

#### IBM dataset ####
from datetime import timedelta
import os

def preprocess_ibm(num_obs):
    date_format = '%Y/%m/%d %H:%M'

    data_path = os.path.join(ROOT, 'data', 'data', 'IBM', 'HI-Small_Trans.csv')
    data_df = pd.read_csv(data_path)
    data_df['Timestamp'] = pd.to_datetime(data_df['Timestamp'], format=date_format)
    data_df.sort_values('Timestamp', inplace=True)
    data_df = data_df[data_df['Account']!= data_df['Account.1']]  # Load raw transaction data, sort by time, and remove self-transactions (Account != Account.1)
    start_index = int(len(data_df)-num_obs)
    data_df = data_df.iloc[start_index:]
    data_df.reset_index(drop=True, inplace=True)
    data_df.reset_index(inplace=True)

    # Build graph using sliding window method: construct a directed graph where nodes are transactions (txId)
    # and edges are consecutive transactions with correct money flow

    # If the time difference between transaction A (Account.1) and transaction B (Account) is <= delta (default 4 hours),
    # create an edge from A to B
    data_df_accounts = data_df[['index', 'Account', 'Account.1', 'Timestamp']]
    delta = 4*60  # 4 hours

    print('Number of observations: ', len(data_df_accounts))
    pieces = 100  # Number of chunks. Since data volume is large (500K transactions), divide into 100 chunks for efficiency

    source = []
    target = []
    
    # Build edge list for IBM transaction graph (define relationships between transactions).
    # Use sliding window join to identify transactions with successor relationships,
    # converting independent transaction records into a directed graph
    for i in tqdm(range(pieces)):
        start = i*num_obs//pieces
        end = (i+1)*num_obs//pieces
        data_df_right = data_df_accounts[start:end]  # Define right window (current transactions to find predecessors for)
        min_timestamp = data_df_right['Timestamp'].iloc[0]
        max_timestamp = data_df_right['Timestamp'].iloc[-1]

        # Define left window (candidate predecessor transactions)
        # Contains: 1) transactions slightly before min_timestamp (at most delta time earlier, i.e., 4 hours)
        #           2) all transactions up to max_timestamp
        data_df_left = data_df_accounts[(data_df_accounts['Timestamp']>=min_timestamp-timedelta(minutes=delta)) & (data_df_accounts['Timestamp']<=max_timestamp)]
        
        # Identify potential money flow paths from transaction 1 to transaction 2
        # Find: receiving account of transaction 1 (Account.1_1) == sending account of transaction 2 (Account_2)
        data_df_join = data_df_left.merge(data_df_right, left_on='Account.1', right_on='Account', suffixes=('_1', '_2'))

        for j in range(len(data_df_join)):
            row = data_df_join.iloc[j]
            delta_trans = row['Timestamp_2']-row['Timestamp_1']
            if (delta_trans.days*24*60+delta_trans.seconds/60 <= delta) & (delta_trans.days*24*60+delta_trans.seconds/60 >= 0):
                source.append(row['index_1']) #將符合時間和帳戶流動條件的交易索引（即 txId）分別存入 source 和 target 列表，構成了圖的邊緣列表
                target.append(row['index_2']) #將所有找到的邊緣關係寫入 edges.csv 檔案

    edges_out = os.path.join(ROOT, 'data', 'data', 'IBM', 'edges.csv')
    pd.DataFrame({'txId1': source, 'txId2': target}).to_csv(edges_out, index=False)

def load_ibm():
    path = os.path.join(ROOT, 'data', 'data', 'IBM')

    df_features = pd.read_csv(os.path.join(path, 'HI-Small_Trans.csv')) 
    df_features['Timestamp'] = pd.to_datetime(df_features['Timestamp'], dayfirst=True, errors='coerce') #IBM 時間格式
#     df_features['Timestamp'] = pd.to_datetime(df_features['Timestamp'], format='%Y/%m/%d %H:%M') #Elliptic 時間格式
    df_features.sort_values('Timestamp', inplace=True)
    df_features = df_features[df_features['Account']!= df_features['Account.1']]

    num_obs = 500000
    start_index = int(len(df_features)-500000)
    df_features = df_features.iloc[start_index:]

    df_features.reset_index(drop=True, inplace=True)
    df_features.reset_index(inplace=True)

    df_features.columns = ['txId', 'Timestamp', 'From Bank', 'Account', 'To Bank', 'Account.1', 'Amount Received', 'Receiving Currency', 'Amount Paid', 'Payment Currency', 'Payment Format', 'class']
    df_features = df_features[['txId', 'Timestamp', 'Amount Received', 'Receiving Currency', 'Amount Paid', 'Payment Currency', 'Payment Format', 'class']]

    print('Number of observations: ', len(df_features))

    if not os.path.exists(os.path.join(path, 'edges.csv')):
        preprocess_ibm(num_obs=num_obs)
    df_edges = pd.read_csv(os.path.join(path, 'edges.csv'))

    list_day = []
    list_hour = []
    list_minute = []
    for date in list(df_features['Timestamp']):
        list_day.append(date.day)
        list_hour.append(date.hour)
        list_minute.append(date.minute)
    df_features['Day'] = list_day
    df_features['Hour'] = list_hour
    df_features['Minute'] = list_minute

    df_features = df_features.drop(columns=['Timestamp'])
    df_features = pd.get_dummies(df_features, columns=['Receiving Currency', 'Payment Currency', 'Payment Format'], dtype=float)

    # Timestamp based split:
    mask = torch.tensor([False]*df_features.shape[0])
    train_size = int(0.6*df_features.shape[0])
    val_size = int(0.2*df_features.shape[0])

    train_mask = mask.clone()
    train_mask[:train_size] = True
    val_mask = mask.clone()
    val_mask[train_size:train_size+val_size] = True
    test_mask = mask.clone()
    test_mask[train_size+val_size:] = True

    ntw = network_AML(df_features, df_edges, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask, name='ibm')

    return(ntw)
