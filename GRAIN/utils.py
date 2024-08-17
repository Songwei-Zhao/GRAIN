import numpy as np
import torch
import os
from torch_geometric.datasets import Planetoid, Amazon, Coauthor, Actor
import scipy.io
from torch_geometric.data import Data, InMemoryDataset
DATA_PATH = 'data'

def evaluate_policy(env, model, render, turns=5):
    scores = 0
    for j in range(turns):
        s, done, ep_r, steps = env.reset2(), False, 0, 0
        while not done:
            steps += 1
            # Take deterministic actions at test time
            a= model.select_action(s)
            a = a.reshape(-1, 1)

            s_prime, r, done, info = env.step2(a)
            if (done and steps != 5):
                done = False
            else:
                done = True

            ep_r += r
            s = s_prime
            if render: env.render()

        scores += ep_r
    return scores / turns


def str2bool(v):
    '''transfer str to bool for argparse'''
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'True','true','TRUE', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'False','false','FALSE', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


#reward engineering for better training
def Reward_adapter(r, EnvIdex):
    # For Pendulum-v0
    if EnvIdex == 0:
        r = (r + 8) / 8

    # For LunarLander
    elif EnvIdex == 1:
        if r <= -100: r = -10

    # For BipedalWalker
    elif EnvIdex == 4 or EnvIdex == 5:
        if r <= -100: r = -1
    return r

def get_dataset(name: str, use_lcc: bool = True) -> InMemoryDataset:
    path = os.path.join(DATA_PATH, name)
    if name in ['Cora', 'Citeseer', 'pubmed']:
        dataset = Planetoid(path, name)
        use_lcc = False
    elif name in ['Computers', 'Photo']:
        dataset = Amazon(path, name)
        use_lcc = True
    elif name == 'CoauthorCS':
        dataset = Coauthor(path, 'CS')
    elif name == 'actor':
        dataset = Actor(path, 'name')
    elif name in ('chameleon', 'cornell', 'film', 'squirrel', 'texas', 'wisconsin'):
        dataset = load_geom_gcn_dataset(name)
    else:
        raise Exception('Unknown dataset.')

    if use_lcc:

        x_new = dataset.data.x
        y_new = dataset.data.y

        edges = dataset.data.edge_index

        data = Data(
            x=x_new,
            edge_index=torch.LongTensor(edges),
            y=y_new,
            train_mask=torch.zeros(y_new.size()[0], dtype=torch.bool),
            test_mask=torch.zeros(y_new.size()[0], dtype=torch.bool),
            val_mask=torch.zeros(y_new.size()[0], dtype=torch.bool)
        )
        dataset.data = data

    return dataset

def get_component(dataset: InMemoryDataset, start: int = 0) -> set:
    visited_nodes = set()
    queued_nodes = set([start])
    row, col = dataset.data.edge_index.numpy()
    while queued_nodes:
        current_node = queued_nodes.pop()
        visited_nodes.update([current_node])
        neighbors = col[np.where(row == current_node)[0]]
        neighbors = [n for n in neighbors if n not in visited_nodes and n not in queued_nodes]
        queued_nodes.update(neighbors)
    return visited_nodes

def get_largest_connected_component(dataset: InMemoryDataset) -> np.ndarray:
    remaining_nodes = set(range(dataset.data.x.shape[0]))
    comps = []
    while remaining_nodes:
        start = min(remaining_nodes)
        comp = get_component(dataset, start)
        comps.append(comp)
        remaining_nodes = remaining_nodes.difference(comp)
    return np.array(list(comps[np.argmax(list(map(len, comps)))]))

def remap_edges(edges: list, mapper: dict) -> list:
    row = [e[0] for e in edges]
    col = [e[1] for e in edges]
    row = list(map(lambda x: mapper[x], row))
    col = list(map(lambda x: mapper[x], col))
    return [row, col]

def get_node_mapper(lcc: np.ndarray) -> dict:
    mapper = {}
    counter = 0
    for node in lcc:
        mapper[node] = counter
        counter += 1
    return mapper

def get_adj_matrix(dataset: InMemoryDataset) -> np.ndarray:
    num_nodes = dataset.data.x.shape[0]
    adj_matrix = np.zeros(shape=(num_nodes, num_nodes))
    for i, j in zip(dataset.data.edge_index[0], dataset.data.edge_index[1]):
        adj_matrix[i, j] = 1.
    return adj_matrix


def load_geom_gcn_dataset(name):
    fulldata = scipy.io.loadmat(f'{DATA_PATH}/{name}.mat')
    edge_index = fulldata['edge_index']
    node_feat = fulldata['node_feat']
    label = np.array(fulldata['label'], dtype=np.int).flatten()
    num_nodes = node_feat.shape[0]

    dataset = NCDataset(name)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    node_feat = torch.tensor(node_feat, dtype=torch.float)
    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes}
    label = torch.tensor(label, dtype=torch.long)
    dataset.label = label
    return dataset

class NCDataset(object):
    def __init__(self, name, root=f'{DATA_PATH}'):
        """
        based off of ogb NodePropPredDataset
        https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/dataset.py
        Gives torch tensors instead of numpy arrays
            - name (str): name of the dataset
            - root (str): root directory to store the dataset folder
            - meta_dict: dictionary that stores all the meta-information about data. Default is None,
                    but when something is passed, it uses its information. Useful for debugging for external contributers.

        Usage after construction:

        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph, label = dataset[0]

        Where the graph is a dictionary of the following form:
        dataset.graph = {'edge_index': edge_index,
                         'edge_feat': None,
                         'node_feat': node_feat,
                         'num_nodes': num_nodes}
        For additional documentation, see OGB Library-Agnostic Loader https://ogb.stanford.edu/docs/nodeprop/
        """

        self.name = name  # original name, e.g., ogbn-proteins
        self.graph = {}
        self.label = None

    def get_idx_split(self, split_type='random', train_prop=.5, valid_prop=.25):
        """
        train_prop: The proportion of dataset for train split. Between 0 and 1.
        valid_prop: The proportion of dataset for validation split. Between 0 and 1.
        """

        if split_type == 'random':
            ignore_negative = False if self.name == 'ogbn-proteins' else True
            train_idx, valid_idx, test_idx = rand_train_test_idx(
                self.label, train_prop=train_prop, valid_prop=valid_prop, ignore_negative=ignore_negative)
            split_idx = {'train': train_idx,
                         'valid': valid_idx,
                         'test': test_idx}
        # data.train_mask = split_idx["train"]
        # data.val_mask = split_idx["valid"]
        # data.test_mask = split_idx["test"]
        return split_idx

    def __getitem__(self, idx):
        assert idx == 0, 'This dataset has only one graph'
        return self.graph, self.label

    def __len__(self):
        return 1

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, len(self))

def rand_train_test_idx(label, train_prop=.5, valid_prop=.25, ignore_negative=True):
    """ randomly splits label into train/valid/test splits """
    if ignore_negative:
        labeled_nodes = torch.where(label != -1)[0]
    else:
        labeled_nodes = label

    n = labeled_nodes.shape[0]
    train_num = int(n * train_prop)
    valid_num = int(n * valid_prop)

    perm = torch.as_tensor(np.random.permutation(n))

    train_indices = perm[:train_num].type(torch.long)
    val_indices = perm[train_num:train_num + valid_num].type(torch.long)
    test_indices = perm[train_num + valid_num:].type(torch.long)

    if not ignore_negative:
        return train_indices, val_indices, test_indices

    train_idx = labeled_nodes[train_indices]
    valid_idx = labeled_nodes[val_indices]
    test_idx = labeled_nodes[test_indices]
    return train_idx, valid_idx, test_idx

def set_train_val_test_split(
        seed: int,
        data: Data,
        num_development: int = 1500,
        num_per_class: int = 20) -> Data:
    #rnd_state = np.random.RandomState(development_seed)
    num_nodes = data.y.shape[0]
    all_idx = np.arange(num_nodes)

    train_idx = []
    rnd_state = np.random.RandomState(seed)
    for c in range(data.y.max() + 1):
        # print(all_idx[np.where(data.y.cpu() == c)[0]])
        # exit()
        class_idx = all_idx[np.where(data.y.cpu() == c)[0]]
        if len(class_idx)<num_per_class:
            train_idx.extend(class_idx)
        else:
            train_idx.extend(rnd_state.choice(class_idx, num_per_class, replace=False))

    ctrain_idx = np.array([i for i in np.arange(num_nodes) if i not in train_idx])
    ctrain_idx_val = rnd_state.choice(ctrain_idx, num_development, replace=False)
    val_idx = rnd_state.choice(ctrain_idx, num_development, replace=False)
    test_idx = ctrain_idx[[i for i in np.arange(num_nodes - len(train_idx)) if i not in ctrain_idx_val]]


    def get_mask(idx):
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        mask[idx] = 1
        return mask

    data.train_mask = get_mask(train_idx)
    data.val_mask = get_mask(val_idx)
    data.test_mask = get_mask(test_idx)

    return data