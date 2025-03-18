import jax.experimental.sparse
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
from flax import linen as nn

INF = 1e8


def normalize(x, p=2, dim=1, eps=1e-12):
    """JAX equivalent of torch.nn.functional.normalize

    Args:
        x: Input tensor
        p: Power for the normalization (default: 2)
        dim: Dimension to normalize over (default: 1)
        eps: Small value to avoid division by zero (default: 1e-12)
    """
    norm = jnp.linalg.norm(x, ord=p, axis=dim, keepdims=True)
    norm = jnp.maximum(norm, eps)
    return x / norm


def laplace_norm(mat):
    # mat: sp.coo_matrix
    norm1 = sp.diags(1 / (np.sqrt(mat.sum(axis=1).A.ravel()) + 1e-8))
    norm2 = sp.diags(1 / (np.sqrt(mat.sum(axis=0).A.ravel()) + 1e-8))
    norm_mat = norm1 @ mat @ norm2
    return norm_mat


def scaled_dot_product(q, k, v):
    dim = q.shape[-1]
    attn = jnp.matmul(q, k.swapaxes(-1, -2)) / dim ** -0.5
    attn = nn.softmax(attn, axis=-1)
    out = jnp.matmul(attn, v)
    return out, attn


class LinNorm(nn.Module):
    n_dim: int

    def setup(self):
        self.lin1 = nn.Dense(self.n_dim * 4,
                             kernel_init=nn.initializers.xavier_uniform(),
                             bias_init=nn.initializers.zeros)
        self.lin2 = nn.Dense(self.n_dim,
                             kernel_init=nn.initializers.xavier_uniform(),
                             bias_init=nn.initializers.zeros)
        self.layer_norm = nn.LayerNorm()

    def __call__(self, x):
        out = self.lin1(x)
        out = nn.relu(out)
        out = self.lin2(out) + x
        out = self.layer_norm(out)
        return out


class MultiHeadAttention(nn.Module):
    n_dim: int
    n_head: int

    def setup(self):
        self.qkv_proj = nn.Dense(self.n_dim * self.n_head * 3,
                                 kernel_init=nn.initializers.xavier_uniform(),
                                 bias_init=nn.initializers.zeros)
        self.o_proj = nn.Dense(self.n_dim,
                               kernel_init=nn.initializers.xavier_uniform(),
                               bias_init=nn.initializers.zeros)
        self.layer_norm = nn.LayerNorm()

    def __call__(self, x):
        """
        X: [bs, seq_len, n_dim]
        """
        bs, seq_len, n_dim = x.shape  # [n_dim * n_aspect == hidden_dim]
        qkv = self.qkv_proj(x)  # [bs, seq_len, n_dim * n_head * 3]
        # [bs, seq_len, n_head, n_dim]
        q, k, v = jnp.array_split(qkv, 3, axis=-1)

        q = q.reshape((bs, seq_len, self.n_head, n_dim)).transpose(
            0, 2, 1, 3)  # [bs, n_head, seq_len, n_dim]
        k = k.reshape((bs, seq_len, self.n_head, n_dim)).transpose(0, 2, 1, 3)
        v = v.reshape((bs, seq_len, self.n_head, n_dim)).transpose(0, 2, 1, 3)

        out, attn = scaled_dot_product(q, k, v)  # [bs, n_head, seq_len, n_dim]
        # [bs, seq_len, n_head * n_dim]
        out = out.swapaxes(1, 2).reshape(bs, seq_len, self.n_head * n_dim)
        out = x + self.o_proj(out)  # [bs, seq_len, n_dim]
        out = self.layer_norm(out)
        return out


class EncoderLayer(nn.Module):
    conf: dict

    def setup(self):
        self.attn = MultiHeadAttention(
            self.conf["n_dim"] // self.conf["n_aspect"], self.conf["n_head"])
        self.lin_norm = LinNorm(self.conf["n_dim"] // self.conf["n_aspect"])

    def __call__(self, x):
        out = self.attn(x)
        out = self.lin_norm(out)
        return out


class PredLayer(nn.Module):
    conf: dict

    def setup(self):
        self.n_item = self.conf["n_item"]
        self.lin = nn.Dense(self.n_item,
                            kernel_init=nn.initializers.xavier_uniform(),
                            bias_init=nn.initializers.zeros)

    def __call__(
            self,
            x,
            residual_feat
    ):
        out = self.lin(x) + residual_feat
        logits = nn.sigmoid(out)
        # logits = nn.tanh(out)
        return logits


class Merge(nn.Module):
    conf: dict
    ui_graph: sp.coo_matrix
    ub_graph: sp.coo_matrix
    bi_graph: sp.coo_matrix

    def setup(self):
        self.n_users = self.conf["n_user"]
        self.n_items = self.conf["n_item"]
        self.n_bundles = self.conf["n_bundle"]
        self.hidden_dim = self.conf["n_dim"]
        self.n_aspect = self.conf["n_aspect"]
        self.num_layers = 1
        self.encoder = [EncoderLayer(self.conf)
                        for _ in range(self.conf["n_layer"])]
        self.mlp = PredLayer(self.conf)
        self.enc = nn.Dense(self.hidden_dim,
                            kernel_init=nn.initializers.xavier_uniform(),
                            bias_init=nn.initializers.zeros)
        self.users_feature = self.param("users_feature", nn.initializers.xavier_normal(),
                                        (self.n_users, self.hidden_dim))
        self.items_feature = self.param("items_feature", nn.initializers.xavier_normal(),
                                        (self.n_items, self.hidden_dim))
        self.bundles_feature = self.param("bundles_feature", nn.initializers.xavier_normal(),
                                          (self.n_bundles, self.hidden_dim))
        self.construct_graph_kernel()

    def construct_graph_kernel(self):
        ui_graph = self.ui_graph
        ub_graph = self.ub_graph
        bi_graph = self.bi_graph

        item_level_graph = sp.bmat([[sp.csr_matrix((ui_graph.shape[0], ui_graph.shape[0])), ui_graph],
                                    [ui_graph.T, sp.csr_matrix((ui_graph.shape[1], ui_graph.shape[1]))]])
        self.item_level_graph = jax.experimental.sparse.BCOO.from_scipy_sparse(
            item_level_graph)
        bundle_level_graph = sp.bmat([[sp.csr_matrix((ub_graph.shape[0], ub_graph.shape[0])), ub_graph],
                                      [ub_graph.T, sp.csr_matrix((ub_graph.shape[1], ub_graph.shape[1]))]])
        self.bundle_level_graph = jax.experimental.sparse.BCOO.from_scipy_sparse(
            bundle_level_graph)
        bi_level_graph = sp.bmat([[sp.csr_matrix((bi_graph.shape[0], bi_graph.shape[0])), bi_graph],
                                  [bi_graph.T, sp.csr_matrix((bi_graph.shape[1], bi_graph.shape[1]))]])
        self.bi_level_graph = jax.experimental.sparse.BCOO.from_scipy_sparse(bi_level_graph)

        bundle_size = bi_graph.sum(axis=1) + 1e-8
        bi_graph = sp.diags(1 / bundle_size.A.ravel()) @ bi_graph
        self.bundle_agg_graph = jax.experimental.sparse.BCOO.from_scipy_sparse(bi_graph)

        user_size = ui_graph.sum(axis=1) + 1e-8
        ui_graph = sp.diags(1 / user_size.A.ravel()) @ ui_graph
        self.users_agg_graph = jax.experimental.sparse.BCOO.from_scipy_sparse(ui_graph)

    def one_propagate(self, graph, A_feature, B_feature):
        features = jnp.concat((A_feature, B_feature), axis=0)
        all_features = [features]
        for i in range(self.num_layers):
            features = graph @ features
            features = features / (i + 2)
            all_features.append(normalize(features, p=2, dim=1))
        all_features = jnp.stack(all_features, 1)
        all_features = jnp.sum(all_features, axis=1)
        A_feature, B_feature = jnp.split(all_features, [A_feature.shape[0]], 0)
        return A_feature, B_feature

    def get_IL_bundle_rep(self, IL_items_feature):
        IL_bundles_feature = self.bundle_agg_graph @ IL_items_feature
        return IL_bundles_feature

    def get_BI_user_rep(self, BI_items_feature):
        BI_users_feature = self.users_agg_graph @ BI_items_feature
        return BI_users_feature

    def propagate(self):
        IL_users_feature, IL_items_feature = self.one_propagate(self.item_level_graph, self.users_feature,
                                                                self.items_feature)
        IL_bundles_feature = self.get_IL_bundle_rep(IL_items_feature)
        BL_users_feature, BL_bundles_feature = self.one_propagate(self.bundle_level_graph, self.users_feature,
                                                                  self.bundles_feature)
        BI_bundles_feature, BI_items_feature = self.one_propagate(self.bi_level_graph, self.bundles_feature,
                                                                  self.items_feature)
        BI_users_feature = self.get_BI_user_rep(BI_items_feature)

        users_feature = [IL_users_feature, BL_users_feature, BI_users_feature]
        bundles_feature = [IL_bundles_feature, BL_bundles_feature, BI_bundles_feature]
        return users_feature, bundles_feature

    def __call__(
            self,
            uids,
            pbids,
            nbids,
            prob_iids,
            prob_iids_bundle
    ):
        """
        uids: user ids
        prob_iids: user's item probability
        prob_iids_bundle: sampled item in interacted bundle probability (noise while inference)
        """
        users_feat, bundles_feat = self.propagate()
        users_feat_0_uids = users_feat[0][uids]
        users_feat0 = users_feat_0_uids.copy()
        users_feat0 = jax.lax.stop_gradient(users_feat0)

        users_feat0 = users_feat0.reshape(-1, self.n_aspect,
                                          self.hidden_dim // self.n_aspect)
        for l in self.encoder:
            users_feat0 = l(users_feat0)
        users_feat0 = users_feat0.reshape(-1, self.hidden_dim)

        # probabilistic
        prob_enc = self.enc(prob_iids_bundle)
        in_feat = jnp.concat([users_feat0, prob_enc], axis=1)
        out_distri = self.mlp(in_feat, prob_iids)

        pos_score = jnp.sum(users_feat[0][uids] * bundles_feat[0][pbids], axis=1)
        neg_score = jnp.sum(users_feat[0][uids] * bundles_feat[0][nbids], axis=1)

        pos_score += jnp.sum(users_feat[1][uids] * bundles_feat[1][pbids], axis=1)
        neg_score += jnp.sum(users_feat[1][uids] * bundles_feat[1][nbids], axis=1)

        pos_score += jnp.sum(users_feat[2][uids] * bundles_feat[2][pbids], axis=1)
        neg_score += jnp.sum(users_feat[2][uids] * bundles_feat[2][nbids], axis=1)
        return out_distri, pos_score, neg_score

    def infer(
            self,
            uids,
            prob_iids,
            prob_iids_bundle
    ):
        users_feat, bundles_feat = self.propagate()
        users_feat0 = users_feat[0][uids]

        users_feat0 = users_feat0.reshape(-1, self.n_aspect,
                                          self.hidden_dim // self.n_aspect)
        for l in self.encoder:
            users_feat0 = l(users_feat0)
        users_feat0 = users_feat0.reshape(-1, self.hidden_dim)

        # probabilistic
        prob_enc = self.enc(prob_iids_bundle)
        in_feat = jnp.concat([users_feat0, prob_enc], axis=1)
        out_distri = self.mlp(in_feat, prob_iids)
        return out_distri

    def eval(self, users):
        users_feature, bundles_feature = self.propagate()
        users_feature_atom, users_feature_non_atom, users_feature_non_atom2 = [
            i[users] for i in users_feature]
        bundles_feature_atom, bundles_feature_non_atom, bundles_feature_non_atom2 = bundles_feature

        scores = (users_feature_atom @ bundles_feature_atom.T +
                  users_feature_non_atom @ bundles_feature_non_atom.T +
                  users_feature_non_atom2 @ bundles_feature_non_atom2.T)
        return scores
