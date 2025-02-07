import jax.numpy as jnp
import pandas as pd
import numpy as np
from config import *
from torch.utils.data import Dataset, DataLoader
from diffusers import DDPMScheduler
import scipy.sparse as sp
from jax.experimental import sparse


TOTAL_TIMESTEPS = conf["timesteps"]


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
    jax_sp_diag_mat =  sparse.BCOO.from_scipy_sparse(diag_mat)
    return jax_sp_diag_mat


class DiffusionScheduler:
    '''
    replicate & simplified code from diffusers.DDPMScheduler
    '''
    def __init__(
            self,
            num_train_timesteps=TOTAL_TIMESTEPS,
            beta_start=0,
            beta_end=1
    ):
        super().__init__()
        self.betas = jnp.linspace(beta_start, beta_end, num_train_timesteps)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = jnp.cumprod(self.alphas, axis=0)
        self.timesteps = jnp.arange(0, num_train_timesteps)[::-1] + 1

    def add_noise(
            self,
            original_samples,
            noise,
            timesteps,
    ):
        noisy_input = original_samples * (1-self.betas[timesteps].reshape(-1, 1)) \
            + noise * self.betas[timesteps].reshape(-1, 1)
        return noisy_input

    def step(
            self,
            model_output,
            time_step,
            post_output,      
    ):
        prev_pred = post_output * (1-1/time_step) + model_output * (1/time_step)
        return prev_pred


'''
Generation Dataloader
'''
class TestData():
    def __init__(self, conf, task="test"):
        super().__init__()
        self.conf = conf
        self.num_user = self.conf["n_user"]
        self.num_item = self.conf["n_item"]
        self.num_bundle = self.conf["n_bundle"]

        self.ui_pairs = get_pairs(f"{self.conf['data_path']}/{self.conf['dataset']}/user_item.txt")
        self.ub_pairs = get_pairs(f"{self.conf['data_path']}/{self.conf['dataset']}/user_bundle_{task}.txt")
        self.bi_pairs = get_pairs(f"{self.conf['data_path']}/{self.conf['dataset']}/bundle_item.txt")
        self.ub_mask_pairs = get_pairs(f"{self.conf['data_path']}/{self.conf['dataset']}/user_bundle_train.txt")

        self.ui_graph = list2csr_sp_graph(self.ui_pairs, (self.num_user, self.num_item))
        self.ub_graph = list2csr_sp_graph(self.ub_pairs, (self.num_user, self.num_bundle))
        self.bi_graph = list2csr_sp_graph(self.bi_pairs, (self.num_bundle, self.num_item))
        self.test_uid = self.ub_graph.sum(axis=1).nonzero()[0]
        self.ub_mask_graph = list2csr_sp_graph(self.ub_mask_pairs, (self.num_user, self.num_bundle))

    def __getitem__(self, index):
        uid = self.test_uid[index]
        prob_iids = np.array(self.ui_graph[uid].todense()).reshape(-1)
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

        self.uibi_graph = self.ui_graph + self.ub_graph @ self.bi_graph
        self.zeros_prob_iids = np.zeros((self.num_item,))

    def __getitem__(self, index):
        uid = index
        prob_iids = np.array(self.ui_graph[index].todense()).reshape(-1)
        bun_idx = self.ub_graph[index].nonzero()[1]
        if len(bun_idx) > 0:
            rand_bun_id = np.random.choice(bun_idx)
            prob_iids_bundle = np.array(self.bi_graph[rand_bun_id].todense()).reshape(-1)
        else:
            prob_iids_bundle = self.zeros_prob_iids
        return uid, prob_iids, prob_iids_bundle

    def __len__(self):
        return self.num_user
    