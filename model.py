from flax import linen as nn
import jax.numpy as jnp
import numpy as np

INF = 1e8


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

    def __call__(self, X):
        out = self.lin(X)
        logits = nn.sigmoid(out)
        return logits


class Net(nn.Module):
    conf: dict

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

    def __call__(self, uids, prob_iids, prob_iids_bundle):
        """
        uids: user ids
        prob_iids: user's item probability
        prob_iids_bundle: sampled item in interacted bundle probability (noise while inference)
        """
        # print(uids)
        # users_feat = self.user_emb[uids]
        # in_feat = jnp.concat([users_feat, prob_iids], axis=1)
        # print(in_feat.shape)
        # out_feat = self.mlp(in_feat)
        # print(X)
        # exit()
        # print(X.nonzero())
        # print(uids, iids)
        # return iids
        # return out_feat
        # print(prob_iids.dtype)
        return prob_iids
    
# Youshu
# Recall@1: 0.0024061751452623434
# Precision@1: 0.007028504490433424
# NDCG@1: 0.0070285043
# Recall@2: 0.004415460768592321
# Precision@2: 0.007223740726278798
# NDCG@2: 0.0075769755
# Recall@3: 0.005186490064993961
# Precision@3: 0.005857087075361187
# NDCG@3: 0.007292046
# Recall@5: 0.008498846825820354
# Precision@5: 0.006013276064037485
# NDCG@5: 0.0085358
# Recall@10: 0.015737290618651003
# Precision@10: 0.005661850839515814
# NDCG@10: 0.011152425
# Recall@20: 0.022220992778953324
# Precision@20: 0.003982819211245607
# NDCG@20: 0.012984116
# Recall@40: 0.03319864650706107
# Precision@40: 0.0032897305739945336
# NDCG@40: 0.016294852
# Recall@50: 0.038119055747622324
# Precision@50: 0.003170636470128856
# NDCG@50: 0.017740801


# iFashion
