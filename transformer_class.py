# transformer_class.py
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

# ---------------- util hook ----------------
class HookPoint(nn.Module):
    """Captures activations via sow so they appear in the
    intermediates collection of model.apply.

    Args
    ----
    key : str | None
        Optional explicit name.  If None we fall back to the module’s
        scope path, e.g. "blocks_0/mlp/hook_pre".
    """
    key: str | None = None

    @nn.compact
    def __call__(self, x):
        # Derive a unique name if none was given
        name = self.key or "/".join(self.scope.path)
        # Store the tensor every forward pass
        self.sow("intermediates", name, x, reduce_fn=lambda _, v: v)
        return x


class Embed(nn.Module):
    d_vocab: int
    d_model: int

    @nn.compact
    def __call__(self, x):
        """x: [batch, seq_len], returns [batch, seq_len, d_model]."""
        embedding = nn.Embed(
            num_embeddings=self.d_vocab,
            features=self.d_model,
            embedding_init=nn.initializers.normal(stddev=1.0 / np.sqrt(self.d_model))
        )
        return embedding(x)

class PosEmbed(nn.Module):
    max_ctx: int
    d_model: int

    def setup(self):
        self.W_pos = self.param(
            "W_pos",
            nn.initializers.normal(stddev=1.0 / np.sqrt(self.d_model)),
            (self.max_ctx, self.d_model),
        )

    def __call__(self, x):
        """
        x: [batch, seq_len, d_model]
        Add learned position embeddings for the first seq_len positions.
        """
        seq_len = x.shape[1]
        pos_emb = self.W_pos[:seq_len]  # [seq_len, d_model]
        return x + pos_emb[jnp.newaxis, :, :]

class Attention(nn.Module):
    d_model: int
    num_heads: int
    d_head: int
    n_ctx: int
    attn_coeff: float

    def setup(self):
        self.W_K = self.param(
            "W_K",
            nn.initializers.normal(stddev=1.0 / np.sqrt(self.d_model)),
            (self.num_heads, self.d_head, self.d_model),
        )
        self.W_Q = self.param(
            "W_Q",
            nn.initializers.normal(stddev=1.0 / np.sqrt(self.d_model)),
            (self.num_heads, self.d_head, self.d_model),
        )
        self.W_V = self.param(
            "W_V",
            nn.initializers.normal(stddev=1.0 / np.sqrt(self.d_model)),
            (self.num_heads, self.d_head, self.d_model),
        )
        # Final linear after concatenating heads
        self.W_O = self.param(
            "W_O",
            nn.initializers.normal(stddev=1.0 / np.sqrt(self.d_model)),
            (self.d_model, self.num_heads * self.d_head),
        )
        # Causal mask of shape (n_ctx, n_ctx), typically (2,2) for this example
        causal_mask = np.tril(np.ones((self.n_ctx, self.n_ctx), dtype=np.float32))
        self.causal_mask = jnp.array(causal_mask)

        self.hook_k = HookPoint()
        self.hook_q = HookPoint()
        self.hook_v = HookPoint()
        self.hook_z = HookPoint()
        self.hook_attn = HookPoint()
        self.hook_attn_pre = HookPoint()

    def __call__(self, x):
        batch_size, seq_len, _ = x.shape
        def project(W, x_):
            return jnp.einsum("ihd,bpd->biph", W, x_)

        k = self.hook_k(project(self.W_K, x))
        q = self.hook_q(project(self.W_Q, x))
        v = self.hook_v(project(self.W_V, x))

        attn_scores_pre = jnp.einsum("biph,biqh->biqp", k, q)
        attn_scores_pre = self.hook_attn_pre(attn_scores_pre / np.sqrt(self.d_head))

        full_mask = self.causal_mask[:seq_len, :seq_len]
        mask = (1.0 - full_mask) * -1e10
        attn_scores_masked = attn_scores_pre #+ mask

        attn_matrix = nn.softmax(attn_scores_masked, axis=-1)
        attn_matrix = attn_matrix * self.attn_coeff + (1.0 - self.attn_coeff)
        attn_matrix = self.hook_attn(attn_matrix)

        z = jnp.einsum("biph,biqp->biqh", v, attn_matrix)
        z = self.hook_z(z)

        z_trans = jnp.transpose(z, (0, 2, 1, 3))  # (b, seq_len, heads, d_head)
        z_flat = jnp.reshape(z_trans, (batch_size, seq_len, self.num_heads * self.d_head))

        out = jnp.einsum("df,bpf->bpd", self.W_O, z_flat)
        return out

class MLP(nn.Module):
    d_model: int
    d_mlp: int
    num_layers: int
    act_type: str = "ReLU"

    # ---------- parameters & hooks ----------
    def setup(self):
        # For each hidden layer i, stash its params and hooks under unique names.
        for i in range(self.num_layers):
            in_dim  = self.d_model if i == 0 else self.d_mlp
            out_dim = self.d_mlp

            # weight + bias
            setattr(
                self, f"W_{i}",
                self.param(f"W_{i}",
                           nn.initializers.normal(stddev=1/np.sqrt(out_dim)),
                           (out_dim, in_dim))
            )
            setattr(
                self, f"b_{i}",
                self.param(f"b_{i}", nn.initializers.zeros, (out_dim,))
            )

            # hooks before & after activation
            setattr(self, f"hook_pre{i+1}",
                    HookPoint(key=f"blocks_0/mlp/hook_pre{i+1}"))
            setattr(self, f"hook_post{i+1}",
                    HookPoint(key=f"blocks_0/mlp/hook_post{i+1}"))

        # final projection back to d_model
        self.W_out = self.param(
            "W_out",
            nn.initializers.normal(stddev=1/np.sqrt(self.d_model)),
            (self.d_model, self.d_mlp)
        )
        self.b_out = self.param("b_out", nn.initializers.zeros, (self.d_model,))

    # ---------- forward ----------
    def _act(self, x):
        if self.act_type == "ReLU":
            return nn.relu(x)
        raise ValueError(f"Unsupported activation {self.act_type!r}")

    def __call__(self, x):
        h = x
        for i in range(self.num_layers):
            W     = getattr(self, f"W_{i}")
            b     = getattr(self, f"b_{i}")
            pre_h = getattr(self, f"hook_pre{i+1}")
            post_h= getattr(self, f"hook_post{i+1}")

            pre = pre_h(jnp.einsum("md,bpd->bpm", W, h) + b)
            h   = post_h(self._act(pre))

        # final output projection
        return jnp.einsum("dm,bpm->bpd", self.W_out, h) + self.b_out



class TransformerBlock(nn.Module):
    d_model: int
    d_head: int
    num_heads: int
    n_ctx: int
    act_type: str
    attn_coeff: float
    num_mlp_layers: int 
    nn_multiplier: int

    def setup(self):
        self.attn = Attention(
            d_model=self.d_model,
            num_heads=self.num_heads,
            d_head=self.d_head,
            n_ctx=self.n_ctx,
            attn_coeff=self.attn_coeff
        )
        self.mlp = MLP(
            d_model=self.d_model,
            d_mlp=self.d_model * self.nn_multiplier,
            num_layers=self.num_mlp_layers,
            act_type=self.act_type,
        )
        self.hook_attn_out = HookPoint()
        self.hook_mlp_out = HookPoint()
        self.hook_resid_pre = HookPoint()
        self.hook_resid_mid = HookPoint()
        self.hook_resid_post = HookPoint()

    def __call__(self, x):
        resid_pre = self.hook_resid_pre(x)
        attn_out = self.attn(resid_pre)
        attn_out = self.hook_attn_out(attn_out)
        x_mid = self.hook_resid_mid(resid_pre + attn_out)

        mlp_out = self.mlp(x_mid)
        mlp_out = self.hook_mlp_out(mlp_out)
        x_post = self.hook_resid_post(x_mid + mlp_out)
        return x_post

# ---------------- shared tool ----------------
def _find_embed_weight(subtree: dict):
    if "embedding" in subtree:
        return subtree["embedding"]
    for v in subtree.values():
        leaves = jax.tree_util.tree_leaves(v)
        if leaves:
            return leaves[0]
    raise KeyError("Could not locate 'embedding' param weight")

# ======================================================================
# One-embed
# ======================================================================
class TransformerOneEmbed(nn.Module):
    num_layers: int
    num_mlp_layers: int
    d_vocab: int
    d_model: int
    d_head: int
    num_heads: int
    n_ctx: int
    act_type: str
    attn_coeff: float
    nn_multiplier: int

    def setup(self):
        self.embed = Embed(self.d_vocab, self.d_model)
        self.pos_embed = PosEmbed(self.n_ctx, self.d_model)
        self.blocks = [TransformerBlock(
            d_model=self.d_model,
            d_head=self.d_head,
            num_heads=self.num_heads,
            n_ctx=self.n_ctx,
            act_type=self.act_type,
            attn_coeff=self.attn_coeff,
            num_mlp_layers=self.num_mlp_layers,
            nn_multiplier=self.nn_multiplier
        ) for _ in range(self.num_layers)]
        self.blocks = nn.Sequential(self.blocks)
        self.W_U = self.param(
            "W_U",
            nn.initializers.normal(stddev=1.0 / np.sqrt(self.d_vocab)),
            (self.d_model, self.d_vocab),
        )

    def __call__(self, x, training=False):
        x_emb = self.embed(x)           # [batch, seq_len, d_model]
        x_emb = self.pos_embed(x_emb)   # [batch, seq_len, d_model]
        x_out = self.blocks(x_emb)      # [batch, seq_len, d_model]
        logits = jnp.einsum("dm,bpd->bpm", self.W_U, x_out)
        return logits



    # ---- analysis ----
    def call_from_embedding(self, x_emb, params):
        if x_emb.ndim == 2:
            seq_emb = x_emb[None, ...]
        elif x_emb.ndim == 3:
            seq_emb = x_emb
        else:
            raise ValueError(f"x_emb must be (2,D) or (1,2,D), got {x_emb.shape}")
        return self.call_from_embedding_sequence(seq_emb, params)[0, -1]

    def extract_embeddings_ab(self, params):
        W_E = _find_embed_weight(params["embed"])
        return W_E, W_E
    
    @nn.nowrap
    def call_from_embedding_sequence(self, seq_emb, params):
        # seq_emb: (1,2,d_model)
        # 1) add pos‑emb
        seq_len = seq_emb.shape[1]
        x = seq_emb + params["pos_embed"]["W_pos"][:seq_len]

        # 2) run each TransformerBlock manually (here just blocks_0 since num_layers=1)
        x = TransformerBlock(
            d_model=self.d_model,
            d_head=self.d_head,
            num_heads=self.num_heads,
            n_ctx=self.n_ctx,
            act_type=self.act_type,
            attn_coeff=self.attn_coeff,
            num_mlp_layers=self.num_mlp_layers,
            nn_multiplier=self.nn_multiplier
        ).apply({"params": params["blocks_0"]}, x)    # → (1,2,d_model)

        # 3) unembed
        logits = jnp.einsum("dm,bpd->bpm", params["W_U"], x)  # (1,2,p)
        return logits
    
    def all_p_squared_embeddings(self, params):
        W_E, _ = self.extract_embeddings_ab(params)
        pos0, pos1 = params["pos_embed"]["W_pos"][:2]
        p = W_E.shape[0]
        ii = np.repeat(np.arange(p), p); jj = np.tile(np.arange(p), p)
        A = np.asarray(W_E)[ii] + np.asarray(pos0)
        B = np.asarray(W_E)[jj] + np.asarray(pos1)
        return np.concatenate([A, B], axis=1)

# ======================================================================
# Two-embed
# ======================================================================
class TransformerTwoEmbed(nn.Module):
    num_layers: int
    num_mlp_layers: int
    d_vocab: int
    d_model: int
    d_head: int
    num_heads: int
    n_ctx: int
    act_type: str
    attn_coeff: float
    nn_multiplier: int  # NEW
    def setup(self):
        self.embed_a = Embed(self.d_vocab, self.d_model)
        self.embed_b = Embed(self.d_vocab, self.d_model)
        self.pos_embed = PosEmbed(self.n_ctx, self.d_model)
        self.blocks = [TransformerBlock(
            d_model=self.d_model,
            d_head=self.d_head,
            num_heads=self.num_heads,
            n_ctx=self.n_ctx,
            act_type=self.act_type,
            attn_coeff=self.attn_coeff,
            num_mlp_layers=self.num_mlp_layers,
            nn_multiplier=self.nn_multiplier
        ) for _ in range(self.num_layers)]
        self.blocks = nn.Sequential(self.blocks)
        self.W_U = self.param(
            "W_U",
            nn.initializers.normal(stddev=1.0 / np.sqrt(self.d_vocab)),
            (self.d_model, self.d_vocab),
        )
    def __call__(self, x, training: bool = False):
        a, b = x[:, 0], x[:, 1]
        ea = self.embed_a(a)    # [B, D]
        eb = self.embed_b(b)    # [B, D]
        seq = jnp.stack([ea, eb], axis=1)       # [B, 2, D]
        seq = self.pos_embed(seq)
        h = self.blocks(seq)
        return jnp.einsum("dm,bpd->bpm", self.W_U, h)

    # ---- analysis ----
    def call_from_embedding(self, x_emb, params):
        if x_emb.ndim == 2:
            seq_emb = x_emb[None, ...]
        elif x_emb.ndim == 3:
            seq_emb = x_emb
        else:
            raise ValueError(f"x_emb must be (2,D) or (1,2,D), got {x_emb.shape}")
        return self.call_from_embedding_sequence(seq_emb, params)[0, -1]

    def extract_embeddings_ab(self, params):
        Wa = _find_embed_weight(params["embed_a"])
        Wb = _find_embed_weight(params["embed_b"])
        return Wa, Wb
    
    @nn.nowrap
    def call_from_embedding_sequence(self, seq_emb, params):
        # seq_emb: (1,2,d_model)
        # 1) add pos‑emb
        seq_len = seq_emb.shape[1]
        x = seq_emb + params["pos_embed"]["W_pos"][:seq_len]

        # 2) run each TransformerBlock manually (here just blocks_0 since num_layers=1)
        x = TransformerBlock(
            d_model=self.d_model,
            d_head=self.d_head,
            num_heads=self.num_heads,
            n_ctx=self.n_ctx,
            act_type=self.act_type,
            attn_coeff=self.attn_coeff,
            num_mlp_layers=self.num_mlp_layers,
            nn_multiplier=self.nn_multiplier
        ).apply({"params": params["blocks_0"]}, x)    # → (1,2,d_model)

        # 3) unembed
        logits = jnp.einsum("dm,bpd->bpm", params["W_U"], x)  # (1,2,p)
        return logits
    
    def all_p_squared_embeddings(self, params):
        Wa, Wb = self.extract_embeddings_ab(params)
        pos0, pos1 = params["pos_embed"]["W_pos"][:2]
        p = Wa.shape[0]
        ii = np.repeat(np.arange(p), p); jj = np.tile(np.arange(p), p)
        A = np.asarray(Wa)[ii] + np.asarray(pos0)
        B = np.asarray(Wb)[jj] + np.asarray(pos1)
        return np.concatenate([A, B], axis=1)


__all__ = [
    "HookPoint", "Embed", "PosEmbed", "Attention", "MLP", "TransformerBlock",
    "TransformerOneEmbed", "TransformerTwoEmbed"
]
