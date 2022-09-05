from opt import args
from GAE import MVGRL
from utils import setup_seed
from train_step import Train_gae
from load_data import *
import warnings
from view_learner import ViewLearner
import pandas as pd
warnings.filterwarnings('ignore')


setup_seed(args.seed)

pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)

print("use cuda: {}".format(args.cuda))
device = torch.device("cuda:0" if opt.args.cuda else "cpu")
args.device = device
opt.args.name = 'cora'

x = np.load('data/{}/feat.npy'.format(opt.args.name)).astype(float)
y = np.load('data/{}/label.npy'.format(opt.args.name))
adj = np.load('data/{}/adj.npy'.format(opt.args.name))
idx_train = np.load('data/{}/idx_train.npy'.format(opt.args.name))
idx_test = np.load('data/{}/idx_test.npy'.format(opt.args.name))

tmp_coo = sp.coo_matrix(adj)
edge_index1 = np.vstack((tmp_coo.row, tmp_coo.col))

adj = torch.Tensor(normalize_adj(adj + np.eye(adj.shape[0])))

opt.args.n_clusters = len(np.unique(y))

data = torch.FloatTensor(x).to(device)

model_gae = MVGRL(data.shape[1], args.hid_units, data.shape[0], args.n_clusters, args.gama).to(device)
print(model_gae.state_dict().keys())
view_learner = ViewLearner(data.shape[1], args.hid_units).to(device)
print(view_learner.state_dict().keys())

Train_gae(model_gae, view_learner, data, adj.to(device), y, edge_index1, device, args.seed, idx_train, idx_test)
