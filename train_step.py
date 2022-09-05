import torch
from opt import args
from utils import eva, target_distribution
from torch.optim import Adam
import torch.nn.functional as F
from load_data import *
import dgl
from sklearn.cluster import KMeans
import torch.nn as nn
import warnings

warnings.filterwarnings('ignore')
from sklearn.metrics.pairwise import cosine_similarity
from scipy.linalg import fractional_matrix_power, inv
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import seaborn as sns
import matplotlib

matplotlib.use('svg')
import matplotlib.pyplot as plt


# matplotlib.use('svg')
# plt.switch_backend('agg')
class LogReg(nn.Module):
    def __init__(self, hid_dim, n_classes):
        super(LogReg, self).__init__()

        self.fc = nn.Linear(hid_dim, n_classes)

    def forward(self, x):
        ret = self.fc(x)
        return ret


# class LogReg(nn.Module):
#     def __init__(self, ft_in, nb_classes):
#         super(LogReg, self).__init__()
#         self.fc = nn.Linear(ft_in, nb_classes)
#         self.sigm = nn.Sigmoid()
#
#         for m in self.modules():
#             self.weights_init(m)
#
#     def weights_init(self, m):
#         if isinstance(m, nn.Linear):
#             torch.nn.init.xavier_uniform_(m.weight.data)
#             if m.bias is not None:
#                 m.bias.data.fill_(0.0)
#
#     def forward(self, seq):
#         ret = torch.log_softmax(self.fc(seq), dim=-1)
#         return ret

acc_reuslt = []
acc_reuslt.append(0)
nmi_result = []
ari_result = []
f1_result = []
use_adjust_lr = []


def Train_gae(model, view_learner, data, adj, label, edge_index, device, seed, idx_train, idx_test):
    acc_reuslt = []
    acc_reuslt.append(0)
    nmi_result = []
    ari_result = []
    f1_result = []
    view_optimizer = Adam(view_learner.parameters(), lr=args.view_lr)  #

    model_optimizer = Adam(model.parameters(), lr=args.lr)

    n = data.shape[0]

    best_loss = 1e9
    best_t = 0
    cnt_wait = 0

    b_xent = nn.BCEWithLogitsLoss()
    for epoch in range(args.epoch):

        # --------------------------------------------min step under InfoMin principle with regularization terms-------------------------------------
        model.eval()
        model.zero_grad()
        view_learner.train()
        view_learner.zero_grad()
        # -----feature masking augmenter
        edge_logits, fea_logits = view_learner(model.encoder, data, adj, edge_index)  # shape (M,1)
        aug_data_weight = torch.sigmoid(torch.mean(fea_logits, 0))
        aug_data_weight2 = aug_data_weight.expand_as(data).contiguous()
        aug_data = aug_data_weight2 * data

        # -----edge weight augmenter
        batch_aug_edge_weight = torch.sigmoid(edge_logits).squeeze()  # p
        aug_adj = new_graph(edge_index, batch_aug_edge_weight, n, device)
        aug_adj = aug_adj.to_dense()
        aug_adj = aug_adj * adj  #
        aug_adj = aug_adj + torch.eye(n).to(device)
        aug_adj = normalize_adj_torch(aug_adj, device)

        z_igae, _, _, _, aug_z_igae, _, _, _ = model.embed(data, aug_data, adj, aug_adj)

        view_loss = -model.calc_loss(z_igae, aug_z_igae,
                                     temperature=0.2) + args.lambda_1 * batch_aug_edge_weight.mean() + args.lambda_2 *aug_data_weight.mean()

        view_loss.backward()
        view_optimizer.step()

        # --------------------------------------------max step under InfoMax principle with respective to node-global MI and node-cluster MI--
        view_learner.eval()
        view_learner.zero_grad()
        model.train()
        model.zero_grad()

        edge_logits, fea_logits = view_learner(model.encoder, data, adj, edge_index)
        aug_data_weight = torch.sigmoid(torch.mean(fea_logits, 0))
        aug_data_weight2 = aug_data_weight.expand_as(data).contiguous()
        aug_data = aug_data_weight2 * data

        batch_aug_edge_weight = torch.sigmoid(edge_logits).squeeze()
        aug_adj = new_graph(edge_index, batch_aug_edge_weight, n, device)
        aug_adj = aug_adj.to_dense()
        aug_adj = aug_adj * adj
        aug_adj = aug_adj + torch.eye(n).to(device)
        aug_adj = normalize_adj_torch(aug_adj, device)

        idx = np.random.permutation(n)
        shuf_data = data[idx, :].to(device)
        shuf_aug_data = aug_data[idx, :].to(device)

        logits, logits2, z_igae, aug_z_igae = model(data, aug_data, shuf_data, shuf_aug_data, adj, aug_adj)
        lbl_1 = torch.ones(n * 2)
        lbl_2 = torch.zeros(n * 2)
        lbl = torch.cat((lbl_1, lbl_2)).to(device)

        model_loss = args.beta * b_xent(logits, lbl) + args.alpha * b_xent(logits2, lbl)
        model_loss.backward()
        model_optimizer.step()

        # if epoch % 100 == 0:
        #     print('Epoch: {0}, Loss: {1:0.4f}'.format(epoch, model_loss.item()))
        #     z_igae, _, c1, Z1, aug_z_igae, _, c2, Z2 = model.embed(data, aug_data, adj, aug_adj)
        #
        #     embs = z_igae + aug_z_igae
        #     embs2 = torch.cat((z_igae, aug_z_igae), dim=1)
        #     kmean = KMeans(n_clusters=int(args.n_clusters), n_init=20, random_state=seed)
        #     for z in [z_igae, aug_z_igae, embs, embs2]:
        #         kmeans = kmean.fit(z.data.cpu().numpy())
        #         acc, nmi, ari, f1 = eva(label, kmeans.labels_, epoch)

        if model_loss < best_loss:
            best_loss = model_loss
            best_t = epoch
            cnt_wait = 0
            model = model.eval()
            view_learner = view_learner.eval()
            torch.save(view_learner.state_dict(), args.name + '_view_learner.pkl')
            torch.save(model.state_dict(), args.name + '_model.pkl')

        else:
            cnt_wait += 1

        if cnt_wait == args.patience:
            print('Early stopping!')
            break

    # --------------------------------------------eve step-------------------------------------
    print('loss{},Loading {}th epoch'.format(best_loss,best_t))
    view_learner.load_state_dict(torch.load(args.name + '_view_learner.pkl'))
    model.load_state_dict(torch.load(args.name + '_model.pkl'))

    model = model.eval()
    view_learner = view_learner.eval()

    edge_logits, fea_logits = view_learner(model.encoder, data, adj, edge_index)
    aug_data_weight = torch.sigmoid(torch.mean(fea_logits, 0))
    aug_data_weight2 = aug_data_weight.expand_as(data).contiguous()
    aug_data = aug_data_weight2 * data

    batch_aug_edge_weight = torch.sigmoid(edge_logits).squeeze()  # p
    aug_adj = new_graph(edge_index, batch_aug_edge_weight, n, device)
    aug_adj = aug_adj.to_dense()
    aug_adj = aug_adj * adj
    aug_adj = aug_adj + torch.eye(n).to(device)
    aug_adj = normalize_adj_torch(aug_adj, device)

    z_igae, _, c1, Z1, aug_z_igae, _, c2, Z2 = model.embed(data, aug_data, adj, aug_adj)

    embs = z_igae + aug_z_igae

    # ----clustering
    kmean = KMeans(n_clusters=int(args.n_clusters), n_init=20, random_state=seed)
    kmeans = kmean.fit(embs.data.cpu().numpy())
    acc, nmi, ari, f1 = eva(label, kmeans.labels_, best_t)

    # ------classification
    if 1 == 1:
        # embs = embs2.data
        embs = embs.data
        train_embs = embs[idx_train]
        test_embs = embs[idx_test]
        label = torch.LongTensor(label).to(device)
        train_lbls = label[idx_train]
        test_lbls = label[idx_test]

        accs = []
        wd = 0.1 if args.name == 'citeseer' else 0.0
        xent = nn.CrossEntropyLoss()
        for _ in range(50):
            log = LogReg(args.hid_units, args.n_clusters)
            opt = torch.optim.Adam(log.parameters(), lr=1e-2, weight_decay=wd)
            log = log.to(device)
            for _ in range(300):
                log.train()
                opt.zero_grad()
                logits = log(train_embs)
                loss = xent(logits, train_lbls)
                loss.backward()
                opt.step()

            log.eval()
            logits = log(test_embs)
            preds = torch.argmax(logits, dim=1)
            acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
            accs.append(acc * 100)

        accs = torch.stack(accs)
        print(accs.mean().item(), accs.std().item())
