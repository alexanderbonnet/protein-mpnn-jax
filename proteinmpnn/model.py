"""Protein MPNN model implementation with JAX/Equinox."""

import einops
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import tqdm
from equinox import nn
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from proteinmpnn import constants


def gelu(x: Float[Array, " ..."]) -> Float[Array, " ..."]:
    """Matches the default PyTorch gelu implementation."""
    return jax.nn.gelu(x, approximate=False)


def pairwise_distances(
    x: Float[Array, "n d"],
    y: Float[Array, "n d"],
    epsilon: float = 1e-6,
) -> Float[Array, "n n"]:
    """Compute pairwise Euclidean distances between points."""
    diff = x[:, None, :] - y[None, :, :]
    return jnp.sqrt(jnp.sum(diff**2, axis=-1) + epsilon)


def mink_neighbors(
    x: Float[Array, "n n"],
    k: int,
    mask: Bool[Array, " n"],
) -> tuple[Float[Array, "n k"], Int[Array, "n k"]]:
    """Compute distances and edges indices for the k-nearest using a
    distance matrix and a mask.
    """
    mask = mask[None, :] * mask[:, None]
    x = x * mask
    x = x + (1 - mask) * jnp.max(x, axis=-1, keepdims=True)
    distances, indices = jax.lax.top_k(-x, k)
    return -distances, indices


def radial_basis_function(
    x: Float[Array, "n k"], dmin: float, dmax: float, dim: int
) -> Float[Array, "n k dim"]:
    """Compute Gaussian radial basis functions."""
    mu = jnp.linspace(dmin, dmax, dim, device=x.device)
    beta = (dim / (dmax - dmin)) ** 2
    return jnp.exp(-beta * (einops.rearrange(x, "n k -> n k ()") - mu) ** 2)


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
    """Gather node features at neighbor indices.

    Used to construct messages for edges.
    """
    edge_index = einops.rearrange(edge_index, "n k -> (n k)")
    neighbors = nodes[edge_index]
    neighbors = einops.rearrange(neighbors, "(n k) d -> n k d", n=nodes.shape[0])
    return neighbors


def gather_chain_mask(
    chain_labels: Int[Array, " n"], edge_index: Int[Array, "n k"]
) -> Bool[Array, "n k"]:
    """Compute a mask indicating if edges are within the same chain."""
    difference = (chain_labels[:, None] - chain_labels[None, :]) == 0
    return gather_edges(difference, edge_index)


class PositionalEncodings(eqx.Module):
    """Positional encodings for residue indices and across different chains."""

    max_offset: int
    linear: nn.Linear

    def __init__(
        self, dim: int, max_relative_offset: int = 32, *, key: PRNGKeyArray
    ) -> None:
        self.max_offset = max_relative_offset
        self.linear = nn.Linear(2 * max_relative_offset + 1 + 1, dim, key=key)

    def __call__(
        self,
        residue_index: Int[Array, " n"],
        edge_index: Int[Array, "n k"],
        chain_mask: Bool[Array, "n k"],
    ) -> Float[Array, " n ne dim"]:
        offset: Int[Array, "n k"] = gather_edges(
            residue_index[:, None] - residue_index[None, :], edge_index
        )
        clipped = jnp.clip(
            offset + self.max_offset, min=0, max=2 * self.max_offset
        ) * chain_mask + (1 - chain_mask) * (2 * self.max_offset + 1)
        one_hot = jax.nn.one_hot(clipped, 2 * self.max_offset + 1 + 1)
        return jax.vmap(jax.vmap(self.linear))(one_hot)


def virtual_cb_pos(backbone_pos: Float[Array, "n 4 3"]) -> Float[Array, "n 3"]:
    """Compute ideal c-beta positions from backbone atoms."""
    ca = backbone_pos[:, constants.ATOM_INDICES["CA"], :]
    b = ca - backbone_pos[:, constants.ATOM_INDICES["N"], :]
    c = backbone_pos[:, constants.ATOM_INDICES["C"], :] - ca
    a = jnp.cross(b, c)
    cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + ca
    return cb


def backbone_features(
    backbone_pos: Float[Array, "n 4 3"],
    mask: Bool[Array, " n"],
    dmin: float = 2.0,
    dmax: float = 22.0,
    rbf_dim: int = 16,
    k_neighbors: int = 30,
) -> tuple[Float[Array, "n n dim"], Int[Array, "n k"]]:
    """Compute backbone features for a protein structure."""
    ca = backbone_pos[:, constants.ATOM_INDICES["CA"], :]
    ca_distances = pairwise_distances(ca, ca)
    ca_distances, edge_index = mink_neighbors(ca_distances, k=k_neighbors, mask=mask)

    cb = virtual_cb_pos(backbone_pos)
    pos: Float[Array, "n 5 3"] = jnp.concat([backbone_pos, cb[:, None, :]], axis=1)

    radial_bases = [
        radial_basis_function(ca_distances, dmin=dmin, dmax=dmax, dim=rbf_dim)
    ]
    for atom1, atom2 in constants.ATOM_PAIR_RBFS:
        idx1 = constants.ATOM_INDICES[atom1]
        idx2 = constants.ATOM_INDICES[atom2]
        d = pairwise_distances(pos[:, idx1, :], pos[:, idx2, :])
        d = gather_edges(d, edge_index)
        rbf = radial_basis_function(d, dmin=dmin, dmax=dmax, dim=rbf_dim)
        radial_bases.append(rbf)

    return jnp.concat(radial_bases, axis=-1), edge_index


class FeatureEmbedding(eqx.Module):
    """Embedding of structural features for the MPNN."""

    rbf_dim: int
    rbf_dmin: float
    rbf_dmax: float
    k: int

    positional_encoding: PositionalEncodings
    edge_embedding: nn.Linear
    edge_norm: nn.LayerNorm

    def __init__(
        self,
        dim: int,
        rbf_dmin: float = 2.0,
        rbf_dmax: float = 22.0,
        rbf_dim: int = 16,
        num_pos_emb: int = 16,
        k: int = 30,
        *,
        key: PRNGKeyArray,
    ) -> None:
        key1, key2 = jr.split(key)
        self.rbf_dim = rbf_dim
        self.rbf_dmin = rbf_dmin
        self.rbf_dmax = rbf_dmax
        self.k = k

        self.edge_embedding = nn.Linear(
            num_pos_emb + rbf_dim * 25, dim, use_bias=False, key=key1
        )
        self.edge_norm = nn.LayerNorm(dim)
        self.positional_encoding = PositionalEncodings(num_pos_emb, key=key2)

    def __call__(
        self,
        backbone_pos: Float[Array, "n 4 3"],
        residue_index: Int[Array, " n"],
        chain_labels: Float[Array, " n"],
        mask: Bool[Array, " n"],
    ) -> tuple[Float[Array, "n k dim"], Int[Array, "n k"]]:
        radial_basis, edge_index = backbone_features(
            backbone_pos,
            dmin=self.rbf_dmin,
            dmax=self.rbf_dmax,
            rbf_dim=self.rbf_dim,
            k_neighbors=self.k,
            mask=mask,
        )

        chain_mask = gather_chain_mask(chain_labels, edge_index)

        positional_emb = self.positional_encoding(
            residue_index=residue_index,
            edge_index=edge_index,
            chain_mask=chain_mask,
        )

        edges = jnp.concat([positional_emb, radial_basis], axis=-1)
        edges = jax.vmap(jax.vmap(self.edge_embedding))(edges)
        edges = jax.vmap(jax.vmap(self.edge_norm))(edges)

        return edges, edge_index


class FeedForward(eqx.Module):
    """Simple feedforward layer used in the encoder and decoder blocks."""

    linear_in: nn.Linear
    linear_out: nn.Linear

    def __init__(self, dim: int, factor: int = 4, *, key: PRNGKeyArray) -> None:
        key1, key2 = jr.split(key, 2)

        self.linear_in = nn.Linear(dim, dim * factor, key=key1)
        self.linear_out = nn.Linear(dim * factor, dim, key=key2)

    def __call__(self, x: Float[Array, " n dim"]) -> Float[Array, " n dim"]:
        x = jax.vmap(self.linear_in)(x)
        x = gelu(x)
        return jax.vmap(self.linear_out)(x)


class SubBlock(eqx.Module):
    """A 3 layer MLP used to update edge messages."""

    linear_in: nn.Linear
    hidden: nn.Linear
    linear_out: nn.Linear

    def __init__(self, in_dim: int, dim: int, *, key: PRNGKeyArray) -> None:
        key1, key2, key3 = jr.split(key, 3)

        self.linear_in = nn.Linear(in_dim, dim, key=key1)
        self.hidden = nn.Linear(dim, dim, key=key2)
        self.linear_out = nn.Linear(dim, dim, key=key3)

    def __call__(self, x: Float[Array, "n k in_dim"]) -> Float[Array, "n k dim"]:
        x = gelu(jax.vmap(jax.vmap(self.linear_in))(x))
        x = gelu(jax.vmap(jax.vmap(self.hidden))(x))
        return jax.vmap(jax.vmap(self.linear_out))(x)


class EncoderBlock(eqx.Module):
    """Encoder block for the MPNN. Updates node and edge representations
    using structural features.
    """

    scale: float

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
        enable_dropout: bool,
        mask_nodes: Bool[Array, " n"],
        mask_edges: Bool[Array, "n k"],
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

        # attention mask
        message = message * einops.rearrange(mask_edges, "n k -> n k ()")

        # aggregate message
        message = einops.reduce(message, "n k d -> n d", "sum") / self.scale

        # dropout and norm
        message = self.dropout1(message, key=key, inference=not enable_dropout)
        out_nodes = jax.vmap(self.norm1)(nodes + message)

        # update representation
        message = self.feedforward(out_nodes)

        # dropout and norm
        message = self.dropout2(message, key=key, inference=not enable_dropout)
        out_nodes = jax.vmap(self.norm2)(out_nodes + message)

        # node mask
        out_nodes = out_nodes * einops.rearrange(mask_nodes, "n -> n ()")

        # update edge representation
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
        out_edges = jax.vmap(jax.vmap(self.norm3))(edges + message)
        return out_nodes, out_edges


class DecoderBlock(eqx.Module):
    """Decoder block for the MPNN. Updates node representations using
    structural and sequence features.
    """

    scale: float

    block: SubBlock

    dropout1: nn.Dropout
    dropout2: nn.Dropout

    norm1: nn.LayerNorm
    norm2: nn.LayerNorm

    feedforward: FeedForward

    def __init__(
        self,
        dim: int,
        dropout_rate: float,
        scale: float = 30,
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
        enable_dropout: bool,
        mask_nodes: Bool[Array, " n"],
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
        out_nodes = jax.vmap(self.norm1)(nodes + message)

        message = self.feedforward(out_nodes)
        message = self.dropout2(message, key=key, inference=not enable_dropout)
        out_nodes = jax.vmap(self.norm2)(out_nodes + message)

        return out_nodes * einops.rearrange(mask_nodes, "n -> n ()")


def build_decoding_attention_mask(
    decoding_order: Int[Array, " n"], edge_index: Int[Array, "n k"]
) -> Float[Array, "n k"]:
    """Helper function used to build the forward and backward attention masks."""
    n, device = decoding_order.shape[0], decoding_order.device
    arr = jnp.zeros((n, n), device=device)
    current = jnp.zeros((n,), device=device)
    for k in range(1, n):
        current = current + jax.nn.one_hot(decoding_order[k - 1], n)
        arr = arr.at[decoding_order[k]].set(current)
    return jnp.take_along_axis(arr, edge_index, axis=1)


def build_forward_backward_mask(
    decoding_order: Int[Array, " n"],
    edge_index: Int[Array, "n k"],
    mask: Bool[Array, " n"],
) -> tuple[Float[Array, "n k"], Float[Array, "n k"]]:
    """Build forward and backward attention masks for the decoder."""
    decoding_attention_mask = build_decoding_attention_mask(decoding_order, edge_index)

    mask = einops.rearrange(mask, "n -> n ()")
    mask_backward = decoding_attention_mask * mask
    mask_forward = (1.0 - decoding_attention_mask) * mask

    return mask_backward, mask_forward


class ProteinMPNN(eqx.Module):
    """The ProteinMPNN model."""

    k: int

    features: FeatureEmbedding

    edge_linear: nn.Linear
    encoder_blocks: list[EncoderBlock]

    sequence_embedding: nn.Embedding
    decoder_blocks: list[DecoderBlock]

    linear_out: nn.Linear

    def __init__(
        self,
        dim: int,
        k: int,
        num_encoder_blocks: int,
        num_decoder_blocks: int,
        vocab: int = 21,
        dropout_rate: float = 0.1,
        *,
        key: Array,
    ) -> None:
        key1, key2, key3, key4, key5, key6 = jr.split(key, 6)

        self.k = k

        # feature embedding
        self.features = FeatureEmbedding(dim=dim, k=k, key=key1)

        # encoder blocks
        self.edge_linear = nn.Linear(dim, dim, key=key2)
        self.encoder_blocks = [
            EncoderBlock(dim=dim, dropout_rate=dropout_rate, key=key)
            for key in jr.split(key3, num_encoder_blocks)
        ]

        # decoder blocks
        self.sequence_embedding = nn.Embedding(vocab, dim, key=key4)
        self.decoder_blocks = [
            DecoderBlock(dim=dim, dropout_rate=dropout_rate, key=key)
            for key in jr.split(key5, num_decoder_blocks)
        ]

        # output
        self.linear_out = nn.Linear(dim, vocab, key=key6)

    def encode(
        self,
        pos: Float[Array, "n 4 3"],
        residue_index: Int[Array, " n"],
        chain_labels: Int[Array, " n"],
        enable_dropout: bool,
        mask_nodes: Bool[Array, " n"],
        key: PRNGKeyArray,
    ) -> tuple[Float[Array, "n dim"], Float[Array, "n k dim"]]:
        edges, edge_index = self.features(
            backbone_pos=pos,
            residue_index=residue_index,
            chain_labels=chain_labels,
            mask=mask_nodes,
        )

        n, _, in_edge_dim = edges.shape
        nodes = jnp.zeros((n, in_edge_dim), device=pos.device)

        edges = jax.vmap(jax.vmap(self.edge_linear))(edges)

        mask_edges = gather_nodes(einops.rearrange(mask_nodes, "n -> n ()"), edge_index)
        mask_edges = einops.rearrange(mask_edges, "n k () -> n k") * einops.rearrange(
            mask_nodes, "n -> n ()"
        )

        keys = jr.split(key, len(self.encoder_blocks))
        for i, block in enumerate(self.encoder_blocks):
            nodes, edges = block(
                nodes=nodes,
                edges=edges,
                edge_index=edge_index,
                mask_nodes=mask_nodes,
                mask_edges=mask_edges,
                enable_dropout=enable_dropout,
                key=keys[i],
            )

        return nodes, edges, edge_index

    def decode(
        self,
        sequence: Int[Array, " n"],
        nodes: Float[Array, "n dim"],
        edges: Float[Array, "n k dim"],
        edge_index: Int[Array, "n k"],
        mask_nodes: Bool[Array, " n"],
        decoding_order: Int[Array, " n"],
        enable_dropout: bool,
        key: PRNGKeyArray,
    ) -> Float[Array, "n dim"]:
        sequence_emb = jax.vmap(self.sequence_embedding)(sequence)
        sequence_emb = jnp.concat(
            [edges, gather_nodes(sequence_emb, edge_index)], axis=-1
        )

        mask_backward, mask_forward = build_forward_backward_mask(
            decoding_order=decoding_order,
            edge_index=edge_index,
            mask=mask_nodes,
        )
        mask_backward = einops.rearrange(mask_backward, "n k -> n k ()")
        mask_forward = einops.rearrange(mask_forward, "n k -> n k ()")

        encoder_edge_emb = jnp.concat(
            [
                edges,
                gather_nodes(jnp.zeros_like(nodes), edge_index=edge_index),
                gather_nodes(nodes, edge_index=edge_index),
            ],
            axis=-1,
        )
        encoder_edge_emb = encoder_edge_emb * mask_forward

        keys = jr.split(key, len(self.decoder_blocks))
        for i, block in enumerate(self.decoder_blocks):
            sequence_edge_emb = jnp.concat(
                [sequence_emb, gather_nodes(nodes, edge_index=edge_index)],
                axis=-1,
            )
            sequence_edge_emb = sequence_edge_emb * mask_backward + encoder_edge_emb
            nodes = block(
                nodes=nodes,
                edges=sequence_edge_emb,
                mask_nodes=mask_nodes,
                enable_dropout=enable_dropout,
                key=keys[i],
            )

        logits = jax.vmap(self.linear_out)(nodes)

        # return jax.nn.log_softmax(logits)
        return logits

    def sample(
        self,
        pos: Float[Array, "n 4 3"],
        residue_index: Int[Array, " n"],
        chain_labels: Int[Array, " n"],
        mask_nodes: Bool[Array, " n"],
        decoding_order: Int[Array, " n"],
        temperature: float = 1.0,
        top_k: int = 1,
        *,
        key: PRNGKeyArray,
    ) -> Int[Array, " n"]:
        nodes, edges, edge_index = self.encode(
            pos=pos,
            residue_index=residue_index,
            chain_labels=chain_labels,
            mask_nodes=mask_nodes,
            enable_dropout=False,
            key=key,
        )
        n = nodes.shape[0]
        sequence = jnp.zeros((n,), dtype=jnp.int32, device=pos.device)

        keys = jr.split(key, n)
        for i in tqdm.tqdm(range(n)):
            key1, key2 = jr.split(keys[i])
            logits = self.decode(
                sequence=sequence,
                nodes=nodes,
                edges=edges,
                edge_index=edge_index,
                mask_nodes=mask_nodes,
                decoding_order=decoding_order,
                enable_dropout=False,
                key=key1,
            )
            logits = logits / temperature

            if top_k == 1:
                sampled = jnp.argmax(logits[decoding_order[i]])
            else:
                probabilities = jax.nn.softmax(logits[decoding_order[i]] / temperature)
                sampled = jr.choice(
                    key=key2, a=logits.shape[-1], p=probabilities, shape=()
                )
            sequence = sequence.at[decoding_order[i]].set(sampled)

        return sequence

    def __call__(
        self,
        pos: Float[Array, "n 4 3"],
        residue_index: Int[Array, " n"],
        chain_labels: Int[Array, " n"],
        mask_nodes: Bool[Array, " n"],
        sequence: Int[Array, " n"],
        decoding_order: Int[Array, " n"],
        enable_dropout: bool,
        key: PRNGKeyArray,
    ) -> Float[Array, "n vocab"]:
        key1, key2 = jr.split(key, 2)

        nodes, edges, edge_index = self.encode(
            pos=pos,
            residue_index=residue_index,
            chain_labels=chain_labels,
            mask_nodes=mask_nodes,
            enable_dropout=enable_dropout,
            key=key1,
        )

        nodes = self.decode(
            sequence=sequence,
            nodes=nodes,
            edges=edges,
            mask_nodes=mask_nodes,
            edge_index=edge_index,
            decoding_order=decoding_order,
            enable_dropout=enable_dropout,
            key=key2,
        )

        return nodes
