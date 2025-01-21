from flax import linen as nn
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
from jax.experimental import sparse


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


def scaled_dot_product(q, k, v, mask=None):
    dim = q.shape[-1]
    attn = jnp.matmul(q, k.swapaxes(-1, -2)) / dim ** -0.5
    if mask is not None:
        attn = jnp.where(mask == 0, -INF, attn)

    attn = nn.softmax(attn, axis=-1)
    out = jnp.matmul(attn, v)
    return out, attn


class LinNorm(nn.Module):
    n_dim: int

    def setup(self):
        self.lin1 = nn.Dense(self.n_dim * 4,
                             kernel_init=nn.initializers.xavier_uniform(),
                             bias_init=nn.initializers.zeros)

        self.act = nn.relu
        self.lin2 = nn.Dense(self.n_dim,
                             kernel_init=nn.initializers.xavier_uniform(),
                             bias_init=nn.initializers.zeros)

        self.layer_norm = nn.LayerNorm()

    def __call__(self, X):
        out = self.lin1(X)
        out = self.act(out)
        out = self.lin2(out) + X
        out = self.layer_norm(out)
        return out


class MultiHeadAttention(nn.Module):
    n_dim: int
    n_head: int
    enc_out: bool

    def setup(self):
        if self.enc_out:
            self.q_proj = nn.Dense(self.n_dim * self.n_head,
                                   kernel_init=nn.initializers.xavier_uniform(),
                                   bias_init=nn.initializers.zeros)

            self.kv_proj = nn.Dense(self.n_dim * self.n_head * 2,
                                    kernel_init=nn.initializers.xavier_uniform(),
                                    bias_init=nn.initializers.zeros)
        else:
            self.qkv_proj = nn.Dense(self.n_dim * self.n_head * 3,
                                     kernel_init=nn.initializers.xavier_uniform(),
                                     bias_init=nn.initializers.zeros)

        self.o_proj = nn.Dense(self.n_dim,
                               kernel_init=nn.initializers.xavier_uniform(),
                               bias_init=nn.initializers.zeros)

        self.layer_norm = nn.LayerNorm()

    def __call__(self, X, enc_out=None, mask=None):
        """
        X: [bs, seq_len, n_dim]
        """
        bs, seq_len, n_dim = X.shape
        if self.enc_out:
            q = self.q_proj(X)
            kv = self.kv_proj(enc_out)
            k, v = jnp.array_split(kv, 2, axis=-1)
        else:
            qkv = self.qkv_proj(X)
            q, k, v = jnp.array_split(qkv, 3, axis=-1)  # [bs, seq_len, n_head, n_dim]

        q = q.reshape((bs, seq_len, self.n_head, n_dim)).transpose(0, 2, 1, 3)  # [bs, n_head, seq_len, n_dim]
        k = k.reshape((bs, seq_len, self.n_head, n_dim)).transpose(0, 2, 1, 3)
        v = v.reshape((bs, seq_len, self.n_head, n_dim)).transpose(0, 2, 1, 3)

        out, attn = scaled_dot_product(q, k, v, mask)
        out = out.swapaxes(1, 2).reshape(bs, seq_len, self.n_head * n_dim)
        out = X + self.o_proj(out)
        out = self.layer_norm(out)
        return out
    

class EncoderLayer(nn.Module):
    conf: dict

    def setup(self):
        self.attn = MultiHeadAttention(self.conf["n_dim"], self.conf["n_head"], False)
        self.lin_norm = LinNorm(self.conf["n_dim"])

    def __call__(self, X):
        out = self.attn(X)
        out = self.lin_norm(out)
        return out
    

class PredLayer(nn.Module):
    conf: dict

    def setup(self):
        self.n_item = self.conf["n_item"]
        self.lin = nn.Dense(self.n_item,
                            kernel_init=nn.initializers.xavier_uniform(),
                            bias_init=nn.initializers.zeros)

    def __call__(self, X, residual_feat):
        out = self.lin(X) + residual_feat
        logits = nn.sigmoid(out)
        return logits


class Net(nn.Module):
    conf: dict
    ui_graph: sp.coo_matrix


    def setup(self):
        self.n_users = self.conf["n_user"]
        self.n_items = self.conf["n_item"]
        self.n_bundles = self.conf["n_bundle"]
        self.hidden_dim = self.conf["n_dim"]

        self.user_emb = self.param("user_emb", 
                                   nn.initializers.xavier_uniform(),
                                   (self.n_users, self.hidden_dim))

        self.item_emb = self.param("item_emb",
                                   nn.initializers.xavier_uniform(),
                                   (self.n_items, self.hidden_dim))
        
        self.encoder = [EncoderLayer(self.conf) for _ in range(self.conf["n_layer"])]
        self.mlp = PredLayer(self.conf)
        self.enc = nn.Dense(self.hidden_dim,
                            kernel_init=nn.initializers.xavier_uniform(),
                            bias_init=nn.initializers.zeros)
        
        self.ui_propagate_graph = self.get_propagate_graph()
        
    def get_propagate_graph(self):
        ui_propagate_graph = sp.bmat([[sp.coo_matrix((self.ui_graph.shape[0], self.ui_graph.shape[0])), self.ui_graph],
                                      [self.ui_graph.T, sp.coo_matrix((self.ui_graph.shape[1], self.ui_graph.shape[1]))]])
        ui_propagate_graph = sparse.BCOO.from_scipy_sparse(laplace_norm(ui_propagate_graph))
        return ui_propagate_graph
    
    def propagate(self, num_layers=2):
        features = jnp.concatenate([self.user_emb, self.item_emb], axis=0)
        all_features = [features]
        for i in range(0, num_layers):
            features = self.ui_propagate_graph @ features
            features = features / (i+2)
            features = normalize(features)
            all_features.append(features)
        all_features = jnp.stack(all_features, axis=1)
        all_features = jnp.mean(all_features, axis=1)
        u_feat, i_feat = jnp.split(all_features, [self.n_users], axis=0)
        return u_feat, i_feat

    def __call__(self, uids, prob_iids, prob_iids_bundle):
        """
        uids: user ids
        prob_iids: user's item probability
        prob_iids_bundle: sampled item in interacted bundle probability (noise while inference)
        """
        # print(uids)
        u_feat, i_feat = self.propagate()
        # users_feat = self.user_emb[uids]
        users_feat = u_feat[uids]
        prob_enc = self.enc(prob_iids_bundle)
        in_feat = jnp.concat([users_feat, prob_enc], axis=1)
        out_feat = self.mlp(in_feat, prob_iids)
        return out_feat
        # return prob_iids
    
