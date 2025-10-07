import einops
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from equinox import nn
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from proteinmpnn import constants


def pairwise_distances(
    x: Float[Array, "n d"],
    y: Float[Array, "n d"],
    epsilon: float = constants.EPS,
) -> Float[Array, "n n"]:
    """Compute pairwise Euclidean distances between points."""
    if x.shape != y.shape:
        raise ValueError(
            f"x and y must have the same shape, got {x.shape} != {y.shape}"
        )

    diff = x[:, None, :] - y[None, :, :]
    return jnp.sqrt(jnp.sum(diff**2, axis=-1) + epsilon)


def mink_neighbors(
    x: Float[Array, "n n"], k: int
) -> tuple[Float[Array, "n k"], Int[Array, "n k"]]:
    """Compute a mask for the top-k values in each row of a matrix."""
    if k <= 0:
        raise ValueError("k must be positive")
    n, _ = x.shape
    distances, indices = jax.lax.top_k(-x, min(k, n))
    return -distances, indices


def radial_basis_function(
    x: Float[Array, "n k"], dmin: float, dmax: float, dim: int
) -> Float[Array, "n k dim"]:
    mu = jnp.linspace(dmin, dmax, dim, device=x.device)
    beta = (dim / (dmax - dmin)) ** 2
    return jnp.exp(-beta * (einops.rearrange(x, "n k -> n k ()") - mu) ** 2)


def backbone_features(
    backbone_pos: Float[Array, "n 4 3"],
    dmin: float = 2.0,
    dmax: float = 22.0,
    rbf_dim: int = 16,
    k_neighbors: int = 30,
) -> Float[Array, "n 3"]:
    """Compute backbone features for a protein structure."""
    ca = backbone_pos[:, constants.ATOM_INDICES["CA"], :]
    ca_distances = pairwise_distances(ca, ca)
    ca_distances, edge_index = mink_neighbors(ca_distances, k=k_neighbors)

    b = ca - backbone_pos[:, constants.ATOM_INDICES["N"], :]
    c = backbone_pos[:, constants.ATOM_INDICES["C"], :] - ca
    a = jnp.cross(b, c)
    cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + ca

    backbone_pos = jnp.concat([backbone_pos, cb[:, None, :]], axis=1)

    radial_basis_ = [
        radial_basis_function(ca_distances, dmin=dmin, dmax=dmax, dim=rbf_dim)
    ]
    for atom1, atom2 in constants.ATOM_PAIR_RBFS:
        idx1 = constants.ATOM_INDICES[atom1]
        idx2 = constants.ATOM_INDICES[atom2]
        d = pairwise_distances(backbone_pos[:, idx1, :], backbone_pos[:, idx2, :])
        d = gather_edges(d, edge_index)
        rbf = radial_basis_function(d, dmin=dmin, dmax=dmax, dim=rbf_dim)
        radial_basis_.append(rbf)

    radial_basis = einops.rearrange(radial_basis_, "b n1 n2 d -> n1 n2 (b d)")

    return radial_basis, edge_index


def gather_edges(
    edges: Float[Array, "n n dim"], edge_index: Int[Array, "n k"]
) -> Float[Array, "n k dim"]:
    """Gather edge features at neighbor indices."""
    n, k = edge_index.shape
    i_idx = jnp.arange(n)[:, None]
    i_idx = jnp.broadcast_to(i_idx, (n, k))
    return edges[i_idx, edge_index, ...]


def gather_nodes(
    nodes: Float[Array, "n dim"], edge_index: Int[Array, "n k"]
) -> Float[Array, "n k dim"]:
    n, _ = nodes.shape
    edge_index = einops.rearrange(edge_index, "n k -> (n k)")
    neighbors = nodes[edge_index]
    neighbors = einops.rearrange(neighbors, "(n k) d -> n k d", n=n)
    return neighbors


class PositionalEmbedding(eqx.Module):
    linear: nn.Linear

    def __init__(
        self, dim: int, max_relative_offset: int = 32, *, key: PRNGKeyArray
    ) -> None:
        self.max_offset = max_relative_offset
        self.linear = nn.Linear(2 * max_relative_offset + 1 + 1, dim, key=key)

    def __call__(
        self,
        residue_index: Int[Array, " n"],
        edge_index: Int[Array, " n k"],
    ) -> Float[Array, " n ne dim"]:
        offset = residue_index[:, None] - residue_index[None, :]
        offset = gather_edges(offset.astype(jnp.int32), edge_index)

        clipped = jnp.clip(offset - self.max_offset, 0, 2 * self.max_offset + 1)
        one_hot = jax.nn.one_hot(
            clipped, 2 * self.max_offset + 1 + 1, dtype=jnp.float32
        )
        return self.linear(one_hot)


class FeedForward(eqx.Module):
    linear_in: nn.Linear
    linear_out: nn.Linear

    def __init__(self, dim: int, factor: int = 4, *, key: PRNGKeyArray) -> None:
        key1, key2 = jr.split(key, 2)

        self.linear_in = nn.Linear(dim, dim * factor, use_bias=False, key=key1)
        self.linear_out = nn.Linear(dim * factor, dim, use_bias=False, key=key2)

    def __call__(self, x: Float[Array, " ... dim"]) -> Float[Array, " ... dim"]:
        x = self.linear_in(x)
        x = jax.nn.gelu(x)
        return self.linear_out(x)


class SubBlock(eqx.Module):
    block: nn.Sequential

    def __init__(self, in_dim: int, dim: int, *, key: PRNGKeyArray) -> None:
        key1, key2, key3 = jr.split(key, 3)

        self.in_block = nn.Sequential(
            nn.Linear(in_dim, dim, key=key1),
            nn.Lambda(jax.nn.gelu),
            nn.Linear(dim, dim, key=key2),
            nn.Lambda(jax.nn.gelu),
            nn.Linear(dim, dim, key=key3),
        )

    def __call__(self, x: Float[Array, "... in_dim"]) -> Float[Array, "... dim"]:
        return self.block(x)


class EncoderBlock(eqx.Module):
    in_block: SubBlock
    out_block: SubBlock

    dropout1: nn.Dropout
    dropout2: nn.Dropout
    dropout3: nn.Dropout

    norm1: nn.LayerNorm
    norm2: nn.LayerNorm
    norm3: nn.LayerNorm

    feedforward: FeedForward

    def __init__(
        self,
        dim: int,
        dropout_rate: float,
        scale: float = 30,
        *,
        key: PRNGKeyArray,
    ) -> None:
        key1, key2, key3 = jr.split(key, 3)

        self.scale = scale

        self.in_block = SubBlock(dim * 3, dim, key=key1)
        self.out_block = SubBlock(dim * 3, dim, key=key2)

        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.dropout3 = nn.Dropout(dropout_rate)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

        self.feedforward = FeedForward(dim=dim, factor=4, key=key3)

    def __call__(
        self,
        nodes: Float[Array, "n dim"],
        edges: Float[Array, "n k dim"],
        edge_index: Int[Array, "n k"],
        mask_nodes: Bool[Array, " n"],
        mask_edges: Bool[Array, "n k"],
        enable_dropout: bool,
        key: PRNGKeyArray,
    ) -> tuple[Float[Array, "n dim"], Float[Array, "n k dim"]]:
        _, k, _ = edges.shape
        message = jnp.concat(
            [
                einops.repeat(nodes, "n d -> n k d", k=k),
                edges,
                gather_nodes(nodes, edge_index),
            ],
            axis=-1,
        )
        message = self.in_block(message)
        message = einops.reduce(message, "n k d -> n d", "sum") / self.scale
        message = self.dropout1(message, key=key, inference=not enable_dropout)
        out_nodes = self.norm1(nodes + message)

        message = self.feedforward(out_nodes)
        message = self.dropout1(message, key=key, inference=not enable_dropout)
        message = message * einops.rearrange(mask_edges, "n k -> n k ()")
        out_nodes = self.norm2(out_nodes + message)
        out_nodes = out_nodes * einops.rearrange(mask_nodes, "n -> n ()")

        message = jnp.concat(
            [
                einops.repeat(out_nodes, "n d -> n k d", k=k),
                edges,
                gather_nodes(out_nodes, edge_index),
            ],
            axis=-1,
        )
        message = self.out_block(message)
        message = self.dropout3(message, key=key, inference=not enable_dropout)
        out_edges = self.norm2(edges + message)
        return out_nodes, out_edges


class DecoderBlock(eqx.Module):
    block: nn.Sequential

    dropout1: nn.Dropout
    dropout2: nn.Dropout

    norm1: nn.LayerNorm
    norm2: nn.LayerNorm

    feedforward: FeedForward

    def __init__(
        self,
        dim: int,
        dropout_rate: float,
        scale: int,
        *,
        key: PRNGKeyArray,
    ) -> None:
        key1, key2 = jr.split(key, 2)

        self.scale = scale

        self.block = SubBlock(dim * 4, dim, key=key1)

        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.feedforward = FeedForward(dim=dim, factor=4, key=key2)

    def __call__(
        self,
        nodes: Float[Array, "n d"],
        edges: Float[Array, "n k dim"],
        mask_nodes: Bool[Array, " n"],
        mask_edges: Bool[Array, "n k"],
        enable_dropout: bool,
        key: PRNGKeyArray,
    ) -> Float[Array, "n dim"]:
        _, k, _ = edges.shape
        message = jnp.concat(
            [einops.repeat(nodes, "n d -> n k d", k=k), edges],
            axis=-1,
        )
        message = self.block(message)

        message = einops.reduce(message, "n k d -> n d", "sum") / self.scale
        message = self.dropout1(message, key=key, inference=not enable_dropout)
        message = message * einops.rearrange(mask_edges, "n k -> n k ()")
        out_nodes = self.norm1(nodes + message)

        message = self.feedforward(out_nodes)
        message = self.dropout2(message, key=key, inference=not enable_dropout)
        out_nodes = self.norm2(out_nodes + message)

        return out_nodes * einops.rearrange(mask_nodes, "n -> n ()")


class ProteinMPNN(eqx.Module):
    edge_linear: nn.Linear
    encoder_blocks: list[EncoderBlock]

    def __init__(
        self,
        in_edge_dim: int,
        dim: int,
        k: int,
        num_encoder_blocks: int,
        vocab: int = 21,
        dropout_rate: float = 0.1,
        *,
        key: Array,
    ) -> None:
        key1, key2, key3 = jr.split(key, 2)

        self.k = k

        # encoder blocks
        self.edge_linear = nn.Linear(in_edge_dim, dim, key=key1)
        self.encoder_blocks = [
            EncoderBlock(dim=dim, dropout_rate=dropout_rate, key=key)
            for key in jr.split(key2, num_encoder_blocks)
        ]

        # decoder blocks
        self.sequence_embedding = nn.Embedding(21, vocab, key=key3)

    def encode(
        self,
        pos: Float[Array, "n 4 3"],
        edges: Float[Array, "n k eidim"],
        edge_index: Int[Array, "n k"],
        enable_dropout: bool,
        key: PRNGKeyArray,
    ) -> tuple[Float[Array, "n dim"], Float[Array, "n k dim"]]:
        edges, edge_index = backbone_features(pos, k_neighbors=self.k)

        n, _, in_edge_dim = edges.shape
        nodes = jnp.zeros((n, in_edge_dim), device=pos.device)

        edges = self.edge_linear(edges)

        for block in self.encoder_blocks:
            key, subkey = jr.split(key)
            nodes, edges = block(
                nodes=nodes,
                edges=edges,
                edge_index=edge_index,
                enable_dropout=enable_dropout,
                key=subkey,
            )

        return nodes, edges

    def __call__(
        self,
        pos: Float[Array, "n 4 3"],
        edges: Float[Array, "n k eidim"],
        edge_index: Int[Array, "n k"],
        enable_dropout: bool,
        key: PRNGKeyArray,
    ) -> Float[Array, "n vocab"]:
        key1, key2 = jr.split(key, 2)

        nodes, edges = self.encode(
            pos=pos,
            edges=edges,
            edge_index=edge_index,
            enable_dropout=enable_dropout,
            key=key1,
        )
