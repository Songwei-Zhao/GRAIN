import random
import os
from scipy.sparse import csr_matrix
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj
from torch_geometric.nn import GCNConv, ChebConv  # noqa
from gym import spaces
import torch.nn as nn
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
from utils import *
from torch_geometric.utils import add_self_loops, degree

class Net(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.5):
        super(Net, self).__init__()
        self.fcn1 = nn.Linear(nfeat, nhid)
        self.fcn2 = nn.Linear(nhid, nclass)
        self.dropout = dropout

    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.fcn1(x))

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fcn2(x)
        return F.log_softmax(x, dim=1)

class gcn_env(object):
    def __init__(self, dataset='Cora', lr=0.01, weight_decay=5e-4, max_layer=10, batch_size=128, policy=""):
        # device = 'cpu'
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dataset = get_dataset(dataset)
        data = dataset.data

        data.edge_index, _ = add_self_loops(data.edge_index, num_nodes=data.x.size(0))

        adj = to_dense_adj(data.edge_index).numpy()[0]

        norm = np.array([np.sum(row) for row in adj])

        self.adj = (adj/norm).T
        self.init_k_hop(6)
        n_class = data.y.max()+1
        N = dataset.data.x.shape[0]
        num_per_class = int((N * 0.4) / n_class)
        num_development = int(N * 0.2)

        data = set_train_val_test_split(
            0,
            data,
            num_development,
            num_per_class
        ).to(device)

        self.model, self.data = Net(data.x.shape[1], 64, n_class).to(device), data.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr, weight_decay=weight_decay)

        train_mask = self.data.train_mask.to('cpu').numpy()
        self.train_indexes = np.where(train_mask==True)[0]

        self.batch_size = len(self.train_indexes) - 1
        self.i = 0
        self.val_acc = 0.0
        self._set_action_space(max_layer)
        obs = self.reset()
        self._set_observation_space(obs)
        self.policy = policy
        self.max_layer = max_layer

        # For Experiment #

        self.baseline_experience = 50

        self.buffers = []
        self.past_performance = [0]
        self.val_acc_dict = [5,1]

    def agg(self, action, indx):

        edge_index = self.data.edge_index

        row, col = edge_index
        deg = degree(col, self.data.x.size(0), dtype=self.data.x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        feature_list = []
        feature_list.append(self.data.x)
        djmat = torch.sparse.FloatTensor(edge_index, norm)
        for i in range(1, 8):
            feature_list.append(torch.spmm(djmat, feature_list[-1]))
        input_feature = []

        for k, i in enumerate(indx):
            hop = round(action[k][0])
            if hop == 0:
                fea = (action[k][0] - hop) * feature_list[hop][i].unsqueeze(0) + (hop + 1 - action[k][0]) * feature_list[hop + 1][i].unsqueeze(0)
            else:
                fea = 0
                alpha = 0.2
                for j in range(hop):
                    fea += (1 - alpha) * feature_list[j][i].unsqueeze(0) + alpha * feature_list[0][i].unsqueeze(0)
                fea = fea / hop
                fea += (action[k][0] - hop) * feature_list[hop][i].unsqueeze(0) + (hop + 1 - action[k][0]) * feature_list[hop + 1][i].unsqueeze(0)
            input_feature.append(fea)
        input_feature = torch.cat(input_feature, dim=0)

        return input_feature

    def seed(self, random_seed):
        torch.manual_seed(random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)

    def init_k_hop(self, max_hop):
        sp_adj = csr_matrix(self.adj)
        dd = sp_adj
        self.adjs = [dd]
        for i in range(max_hop):
            dd *= sp_adj
            self.adjs.append(dd)

    def reset(self):
        index = self.train_indexes[self.i]
        state = self.data.x[index].to('cpu').numpy()
        self.optimizer.zero_grad()
        return state

    def _set_action_space(self, _max):
        self.action_num = _max

    def action_space(self, s, action_num):
        action = np.random.normal(2, 1, (s, action_num))
        return action

    def _set_observation_space(self, observation):
        low = np.full(observation.shape, -float('inf'))
        high = np.full(observation.shape, float('inf'))
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def reset2(self):
        start = self.i
        end = (self.i + self.batch_size) % len(self.train_indexes)
        index = self.train_indexes[start:end]
        state = self.data.x[index].to('cpu').numpy()
        self.optimizer.zero_grad()
        return state

    def step2(self, actions):
        start = self.i
        end = (self.i + self.batch_size) % len(self.train_indexes)
        index = self.train_indexes[start:end]

        feture = self.agg(actions.clip(0, 5), index)
        self.train(feture, index)

        done = True

        index = self.stochastic_k_hop(actions.clip(0, 5), index)
        next_state = self.data.x[index].to('cpu').numpy()

        val_acc_dict = self.eval_batch()

        self.val_acc_dict.append(val_acc_dict)

        avg = np.mean(val_acc_dict - np.array(self.val_acc_dict))

        reward = avg


        r = val_acc_dict

        return next_state, reward, [done]*self.batch_size, r

    def stochastic_k_hop(self, actions, index):
        next_batch = []
        for idx, act in zip(index, actions):
            prob = self.adjs[int(act)].getrow(idx).toarray().flatten()
            prob /= prob.sum()
            cand = np.array([i for i in range(len(prob))])
            next_cand = np.random.choice(cand, p=prob)
            next_batch.append(next_cand)
        return next_batch

    def train(self, feature, indexes):
        self.model.train()
        self.optimizer.zero_grad()
        pred = self.model(feature)
        y = self.data.y[indexes]
        F.nll_loss(pred, y).backward()
        self.optimizer.step()
        
    def eval_batch(self):
        self.model.eval()
        val_index = np.where(self.data.val_mask.to('cpu').numpy()==True)[0]
        val_states = self.data.x[val_index].to('cpu').numpy()

        val_acts= self.policy.select_action(val_states).clip(0, 5)
        val_acts = val_acts.reshape(-1, 1)

        feature = self.agg(val_acts, val_index)
        logits = self.model(feature)
        pred = logits.max(1)[1]

        acc = pred.eq(self.data.y[val_index]).sum().item() / len(val_index)
        return acc

    def test_batch(self):
        self.model.eval()
        batch_dict = {}
        test_index = np.where(self.data.test_mask.to('cpu').numpy()==True)[0]
        test_states = self.data.x[test_index].to('cpu').numpy()

        test_acts = self.policy.select_action(test_states).clip(0, 5)
        test_acts = test_acts.reshape(-1, 1)
        feature = self.agg(test_acts, test_index)
        logits = self.model(feature)

        pred = logits.max(1)[1]

        acc = pred.eq(self.data.y[test_index]).sum().item() / len(test_index)

        return acc

    def check(self):
        self.model.eval()
        train_index = np.where(self.data.train_mask.to('cpu').numpy()==True)[0]
        tr_states = self.data.x[train_index].to('cpu').numpy()
        tr_acts = self.policy.eval_step(tr_states)

        val_index = np.where(self.data.val_mask.to('cpu').numpy()==True)[0]
        val_states = self.data.x[val_index].to('cpu').numpy()
        val_acts = self.policy.eval_step(val_states)

        test_index = np.where(self.data.test_mask.to('cpu').numpy()==True)[0]
        test_states = self.data.x[test_index].to('cpu').numpy()
        test_acts = self.policy.eval_step(test_states)

        return (train_index, tr_states, tr_acts), (val_index, val_states, val_acts), (test_index, test_states, test_acts)


