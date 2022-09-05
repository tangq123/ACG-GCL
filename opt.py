import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--name', type=str, default='cora')
parser.add_argument('--hid_units', type=int, default=512,help='embedding dimension')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--view_lr', type=float, default=1e-3,help='View Learning rate.')
parser.add_argument('--gama',type=int,default=1000)
parser.add_argument('--beta',type=int,default=2)
parser.add_argument('--lambda_1',type=float,default=1)
parser.add_argument('--lambda_2',type=float,default=1)
parser.add_argument('--alpha',type=int,default=1)
parser.add_argument('--n_clusters', type=int, default=7)
parser.add_argument('--epoch', type=int, default=3000)
parser.add_argument('--patience',type=int,default=100)
parser.add_argument('--cuda', type=bool, default=True)
parser.add_argument('--seed', type=int, default=666)

args = parser.parse_args()