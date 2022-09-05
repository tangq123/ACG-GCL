import torch
import numpy as np
import scipy.sparse as sp
from torch.utils.data import Dataset

import networkx as nx
import opt
from munkres import Munkres, print_matrix
from sklearn import metrics

class clustering_metrics():
    "from https://github.com/Ruiqi-Hu/ARGA"
    def __init__(self, true_label, predict_label):
        self.true_label = true_label
        self.pred_label = predict_label


    def clusteringAcc(self):
        # best mapping between true_label and predict label
        l1 = list(set(self.true_label))
        numclass1 = len(l1)

        l2 = list(set(self.pred_label))
        numclass2 = len(l2)
        if numclass1 != numclass2:
            print('Class Not equal, Error!!!!')
            return 0

        cost = np.zeros((numclass1, numclass2), dtype=int)
        for i, c1 in enumerate(l1):
            mps = [i1 for i1, e1 in enumerate(self.true_label) if e1 == c1]
            for j, c2 in enumerate(l2):
                mps_d = [i1 for i1 in mps if self.pred_label[i1] == c2]

                cost[i][j] = len(mps_d)

        # match two clustering results by Munkres algorithm
        m = Munkres()
        cost = cost.__neg__().tolist()

        indexes = m.compute(cost)

        # get the match results
        new_predict = np.zeros(len(self.pred_label))
        for i, c in enumerate(l1):
            # correponding label in l2:
            c2 = l2[indexes[i][1]]

            # ai is the index with label==c2 in the pred_label list
            ai = [ind for ind, elm in enumerate(self.pred_label) if elm == c2]
            new_predict[ai] = c

        acc = metrics.accuracy_score(self.true_label, new_predict)
        f1_macro = metrics.f1_score(self.true_label, new_predict, average='macro')
        precision_macro = metrics.precision_score(self.true_label, new_predict, average='macro')
        recall_macro = metrics.recall_score(self.true_label, new_predict, average='macro')
        f1_micro = metrics.f1_score(self.true_label, new_predict, average='micro')
        precision_micro = metrics.precision_score(self.true_label, new_predict, average='micro')
        recall_micro = metrics.recall_score(self.true_label, new_predict, average='micro')
        return acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro

    def evaluationClusterModelFromLabel(self):
        nmi = metrics.normalized_mutual_info_score(self.true_label, self.pred_label)
        adjscore = metrics.adjusted_rand_score(self.true_label, self.pred_label)
        acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro = self.clusteringAcc()

        
        return acc, nmi, adjscore
def wrong_edge(num):
    v1 = np.random.randint(100,size = num)
    v2 = np.random.randint(100,size = num)
    random_edge = np.zeros((2*num,2),dtype=np.int32)
    for i in range(num):
       e1 = v1[i]
       e2 = v2[i]
       random_edge[2 * i][0] = e1
       random_edge[2 * i][1] = e2
       random_edge[2 * i + 1][0] = e2
       random_edge[2 * i + 1][1] = e1

    #print(random_edge)
    return random_edge


def new_graph(edge_index,weight,n,device):
    indices = torch.from_numpy(np.vstack((edge_index[0], edge_index[1])).astype(np.int64)).to(device) # 二维数组
    values = weight
    shape = torch.Size((n,n))
    return torch.sparse.FloatTensor(indices, values, shape)

def load_graph(k, graph_k_save_path, graph_save_path, data_path,walk_length,num_walk):
    if k:
        path = graph_k_save_path
    else:
        path = graph_save_path

    print("Loading path:", path)

    data = np.loadtxt(data_path, dtype=float)

    n, _ = data.shape

    idx = np.array([i for i in range(n)], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(path, dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)

  #  random_edge = wrong_edge(100)
  #  edges = np.vstack((edges,random_edge))
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(n, n), dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    G = nx.DiGraph()

    # add edges
    for i in range(len(edges)):
        src = str(edges[i][0])
        dst = str(edges[i][1])
        G.add_edge(src, dst)
        G[src][dst]['weight'] = 1.0
        #  print("88888888888888",G.edges)






    # g = Graph(G)

    model = Node2vec_onlywalk(num = n,graph=G, path_length=walk_length, num_paths=num_walk, dim=4, workers=8,
                              window=5, p=2, q=0.5, dw=False)

    return adj,model.walker#,random_edge


def normalize(mx):
    adj = sp.coo_matrix(adj)
    
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx).tocoo().todense()
    return mx

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1)) # 按行求和
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features

def norm_X(X):
    X_abs = X.norm(dim=1).unsqueeze(1)
    X_norm = X / torch.max(X_abs, 1e-8 * torch.ones_like(X_abs))
    return X_norm

def normalize_adj(adj, self_loop=False):
    """Symmetrically normalize adjacency matrix."""
    if self_loop:
        adj = adj + sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo().todense()

def normalize_adj_torch(adj,device,self_loop=False):
    """Symmetrically normalize adjacency matrix."""
    if self_loop:
        adj = adj + torch.eye(adj.shape[0]).to(device)
    rowsum = adj.sum(1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt).to(device)
    return torch.mm(d_mat_inv_sqrt,torch.mm(adj,d_mat_inv_sqrt))
    # return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)

def normal(x):
    rowmu = (np.mean(x,axis=1)).reshape((x.shape[0],1)).repeat(x.shape[1],1)
    rowstd = (np.std(x,axis=1)).reshape((x.shape[0],1)).repeat(x.shape[1],1)

    return (x-rowmu)/rowstd



def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


class LoadDataset(Dataset):

    def __init__(self, data):
        self.x = data

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])).float(), torch.from_numpy(np.array(idx))