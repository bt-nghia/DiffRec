import jax.numpy as jnp
import numpy as np
import pandas as pd
import scipy.sparse as sp
from jax.experimental import sparse
from torch.utils.data import Dataset

from config import *

TOTAL_TIMESTEP = conf["timestep"]


def get_pairs(file_path):
    xy = pd.read_csv(file_path, sep="\t", names=["x", "y"])
    xy = xy.to_numpy()
    return xy


def get_size(file_path):
    nu, nb, ni = pd.read_csv(file_path, sep="\t", names=["u", "b", "i"]).to_numpy()[0]
    return nu, nb, ni


def list2jax_sp_graph(list_index, shape):
    values = np.ones(list_index.shape[0])
    sp_graph = sp.coo_matrix(
        (values, (list_index[:, 0], list_index[:, 1])),
        shape=shape
    )
    jax_sp_graph = sparse.BCOO.from_scipy_sparse(sp_graph)
    return jax_sp_graph


def list2graph(list_index, shape):
    graph = np.zeros(shape)
    for i in list_index:
        graph[i[0], i[1]] = 1
        # important
        # graph[i[0], i[1]] += 1 # load repeat or not
    return graph


def list2csr_sp_graph(list_index, shape):
    """
    list indices to scipy.sparse csr
    """
    sp_graph = sp.coo_matrix(
        (np.ones(list_index.shape[0]), (list_index[:, 0], list_index[:, 1])),
        shape=shape
    ).tocsr()
    return sp_graph > 0


def graph2list(graph):
    idx = np.stack(graph.nonzero(), axis=0)
    idx = idx.T  # [[row, col], ...]
    return idx


def jax_sp_graph2list(graph):
    idx = graph.indices
    idx = idx.T
    return idx


def csr_sp_graph2list(graph):
    graph = graph.tocoo()
    indices = np.array([graph.row, graph.col]).T
    return indices


def make_sp_diag_mat(n):
    ids = np.arange(0, n)
    vals = np.ones(n, dtype=float)
    diag_mat = sp.coo_matrix(
        (vals, (ids, ids)),
        shape=(n, n)
    )
    jax_sp_diag_mat = sparse.BCOO.from_scipy_sparse(diag_mat)
    return jax_sp_diag_mat


class DiffusionScheduler:
    '''
    replicate & simplified code from diffusers.DDPMScheduler
    '''

    def __init__(
            self,
            num_train_timestep=TOTAL_TIMESTEP,
            beta_start=0,
            beta_end=1
    ):
        super().__init__()
        self.betas = jnp.linspace(beta_start, beta_end, num_train_timestep)
        self.sqrt_one_minus_betas = jnp.sqrt(1 - self.betas)
        self.alphas = 1 - self.betas
        self.alphas_cum_prod = jnp.cumprod(self.alphas, axis=0)
        self.sqrt_one_minus_alphas_cum_prod = jnp.sqrt(1 - self.alphas_cum_prod)
        self.sqrt_alphas_cum_prod = jnp.sqrt(self.alphas_cum_prod)
        self.timestep = jnp.arange(0, num_train_timestep)[::-1] + 1

    def add_noise(
            self,
            x_t_1,
            epsilon,
            t
    ):
        x_t = (x_t_1 * self.sqrt_alphas_cum_prod[t].reshape(-1, 1) +
               epsilon * self.sqrt_one_minus_alphas_cum_prod[t].reshape(-1, 1))
        return x_t

    # def step(
    #         self,
    #         x_t,
    #         t,
    #         x_t_1,
    # ):
    #     x_t_out = x_t * (1 / t) + x_t_1 * (1 - 1 / t)
    #     return x_t_out

    def polyak_update(
            self,
            x_t,
            x_t_1,
            t,
    ):
        x_t_1_out = x_t_1 * (1 / t) + x_t * (1 - 1 / t)
        return x_t_1_out


'''
Generation Dataloader
'''


class TestData():
    def __init__(self, conf, task="test"):
        super().__init__()
        self.conf = conf
        self.task = task
        self.num_user = self.conf["n_user"]
        self.num_item = self.conf["n_item"]
        self.num_bundle = self.conf["n_bundle"]

        self.ui_pairs = get_pairs(f"{self.conf['data_path']}/{self.conf['dataset']}/user_item.txt")
        self.ub_pairs_test = get_pairs(f"{self.conf['data_path']}/{self.conf['dataset']}/user_bundle_{self.task}.txt")
        self.bi_pairs = get_pairs(f"{self.conf['data_path']}/{self.conf['dataset']}/bundle_item.txt")
        self.ub_pairs_train = get_pairs(f"{self.conf['data_path']}/{self.conf['dataset']}/user_bundle_train.txt")

        self.ui_graph = list2csr_sp_graph(self.ui_pairs, (self.num_user, self.num_item))
        self.ub_graph_test = list2csr_sp_graph(self.ub_pairs_test, (self.num_user, self.num_bundle))
        self.bi_graph = list2csr_sp_graph(self.bi_pairs, (self.num_bundle, self.num_item))
        self.test_uid = self.ub_graph_test.sum(axis=1).nonzero()[0]
        self.ub_graph_train = list2csr_sp_graph(self.ub_pairs_train, (self.num_user, self.num_bundle))

        self.ubi_graph = self.ub_graph_train @ self.bi_graph

    def __getitem__(self, index):
        uid = self.test_uid[index]
        prob_iids = np.array(self.ui_graph[uid].todense()).reshape(-1)
        # prob_iids = np.array(self.ubi_graph[uid].todense()).reshape(-1)
        return uid, prob_iids

    def __len__(self):
        return len(self.test_uid)


class TrainData(Dataset):
    """
    return 
    user id -> for personalize
    item prob -> for guidance
    item (bundle) -> for denoised
    """

    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        # we use bundle id to easily link bundle to user for train and test purpose 
        self.num_user = self.conf["n_user"]
        self.num_item = self.conf["n_item"]
        self.num_bundle = self.conf["n_bundle"]

        self.ui_pairs = get_pairs(f"{self.conf['data_path']}/{self.conf['dataset']}/user_item.txt")
        self.ub_pairs = get_pairs(f"{self.conf['data_path']}/{self.conf['dataset']}/user_bundle_train.txt")
        self.bi_pairs = get_pairs(f"{self.conf['data_path']}/{self.conf['dataset']}/bundle_item.txt")

        self.ui_graph = list2csr_sp_graph(self.ui_pairs, (self.num_user, self.num_item))
        self.ub_graph = list2csr_sp_graph(self.ub_pairs, (self.num_user, self.num_bundle))
        self.bi_graph = list2csr_sp_graph(self.bi_pairs, (self.num_bundle, self.num_item))

        self.ubi_graph = self.ub_graph @ self.bi_graph
        self.uibi_graph = self.ui_graph + self.ub_graph @ self.bi_graph
        self.zeros_prob_iids = np.zeros((self.num_item,))

    def __getitem__(self, index):
        uid = index
        # prob_iids = np.array(self.ui_graph[index].todense()).reshape(-1)
        prob_iids = np.array(self.ui_graph[uid].todense()).reshape(-1)
        bun_idx = self.ub_graph[index].nonzero()[1]
        if len(bun_idx) > 0:
            rand_bun_id = np.random.choice(bun_idx)
            prob_iids_bundle = np.array(self.bi_graph[rand_bun_id].todense()).reshape(-1)
        else:
            prob_iids_bundle = self.zeros_prob_iids
        return uid, prob_iids, prob_iids_bundle

    def __len__(self):
        return self.num_user


class TrainDataVer2(Dataset):
    """
    return 
    user id -> for personalize
    item prob -> for guidance
    item (bundle) -> for denoised
    """

    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        # we use bundle id to easily link bundle to user for train and test purpose 
        self.num_user = self.conf["n_user"]
        self.num_item = self.conf["n_item"]
        self.num_bundle = self.conf["n_bundle"]

        self.ui_pairs = get_pairs(f"{self.conf['data_path']}/{self.conf['dataset']}/user_item.txt")
        self.ub_pairs = get_pairs(f"{self.conf['data_path']}/{self.conf['dataset']}/user_bundle_train.txt")
        self.bi_pairs = get_pairs(f"{self.conf['data_path']}/{self.conf['dataset']}/bundle_item.txt")

        self.ui_graph = list2csr_sp_graph(self.ui_pairs, (self.num_user, self.num_item))
        self.ub_graph = list2csr_sp_graph(self.ub_pairs, (self.num_user, self.num_bundle))
        self.bi_graph = list2csr_sp_graph(self.bi_pairs, (self.num_bundle, self.num_item))

        self.ubi_graph = self.ub_graph @ self.bi_graph
        self.uibi_graph = self.ui_graph + self.ub_graph @ self.bi_graph
        self.zeros_prob_iids = np.zeros((self.num_item,))

    def __getitem__(self, index):
        uid, bid = self.ub_pairs[index]
        prob_iids = np.array(self.ui_graph[uid].todense()).reshape(-1)
        prob_iids_bundle = np.array(self.bi_graph[bid].todense()).reshape(-1)
        return uid, prob_iids, prob_iids_bundle

    def __len__(self):
        return len(self.ub_pairs)

# meal_cold
# class TrainData(Dataset):
#     """
#     return 
#     user id -> for personalize
#     item prob -> for guidance
#     item (bundle) -> for denoised
#     """
#     def __init__(self, conf):
#         super().__init__()
#         self.conf = conf
#         # we use bundle id to easily link bundle to user for train and test purpose 
#         self.num_user = self.conf["n_user"]
#         self.num_item = self.conf["n_item"]
#         self.num_bundle = self.conf["n_bundle"]

#         self.ui_pairs = get_pairs(f"{self.conf['data_path']}/{self.conf['dataset']}/user_item.txt")
#         self.ub_pairs = get_pairs(f"{self.conf['data_path']}/{self.conf['dataset']}/user_bundle_train.txt")
#         self.bi_pairs = get_pairs(f"{self.conf['data_path']}/{self.conf['dataset']}/bundle_item.txt")

#         self.ui_graph = list2csr_sp_graph(self.ui_pairs, (self.num_user, self.num_item))
#         self.ub_graph = list2csr_sp_graph(self.ub_pairs, (self.num_user, self.num_bundle))
#         self.bi_graph = list2csr_sp_graph(self.bi_pairs, (self.num_bundle, self.num_item))

#         self.ubi_graph = self.ub_graph @ self.bi_graph
#         self.uibi_graph = self.ui_graph + self.ub_graph @ self.bi_graph
#         self.zeros_prob_iids = np.zeros((self.num_item,))

#     def __getitem__(self, index):
#         uid = index
#         # prob_iids = np.array(self.ui_graph[index].todense()).reshape(-1)
#         prob_iids = np.array(self.ui_graph[uid].todense()).reshape(-1)
#         bun_idx = self.ub_graph[index].nonzero()[1]
#         if len(bun_idx) > 0:
#             rand_bun_id = np.random.choice(bun_idx)
#             prob_iids_bundle = np.array(self.bi_graph[rand_bun_id].todense()).reshape(-1)
#         else:
#             prob_iids_bundle = self.zeros_prob_iids
#         return uid, prob_iids, prob_iids_bundle

#     def __len__(self):
#         return self.num_user


# class TrainDataVer2(Dataset):
#     """
#     return 
#     user id -> for personalize
#     item prob -> for guidance
#     item (bundle) -> for denoised
#     """
#     def __init__(self, conf):
#         super().__init__()
#         self.conf = conf
#         # we use bundle id to easily link bundle to user for train and test purpose 
#         self.num_user = self.conf["n_user"]
#         self.num_item = self.conf["n_item"]
#         self.num_bundle = self.conf["n_bundle"]

#         self.ui_pairs = get_pairs(f"{self.conf['data_path']}/{self.conf['dataset']}/user_item.txt")
#         self.ub_pairs = get_pairs(f"{self.conf['data_path']}/{self.conf['dataset']}/user_bundle_train.txt")
#         self.bi_pairs = get_pairs(f"{self.conf['data_path']}/{self.conf['dataset']}/bundle_item.txt")

#         self.ui_graph = list2csr_sp_graph(self.ui_pairs, (self.num_user, self.num_item))
#         self.ub_graph = list2csr_sp_graph(self.ub_pairs, (self.num_user, self.num_bundle))
#         self.bi_graph = list2csr_sp_graph(self.bi_pairs, (self.num_bundle, self.num_item))

#         self.ubi_graph = self.ub_graph @ self.bi_graph
#         self.uibi_graph = self.ui_graph + self.ub_graph @ self.bi_graph
#         self.zeros_prob_iids = np.zeros((self.num_item,))

#     def __getitem__(self, index):
#         uid, bid = self.ub_pairs[index]
#         prob_iids = np.array(self.ubi_graph[uid].todense()).reshape(-1)
#         prob_iids_bundle = np.array(self.bi_graph[bid].todense()).reshape(-1)
#         return uid, prob_iids, prob_iids_bundle

#     def __len__(self):
#         return len(self.ub_pairs)
