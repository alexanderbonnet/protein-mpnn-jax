"""Protein MPNN model implementation with JAX/Equinox."""

from pathlib import Path
from typing import Literal

import einops
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import loguru
import torch
import tqdm
from equinox import nn
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from proteinmpnn import constants
from proteinmpnn import mpnn as model_


def gelu(x: Float[Array, " ..."]) -> Float[Array, " ..."]:
    """Matches the default PyTorch implementation."""
    return jax.nn.gelu(x, approximate=False)


def pairwise_distances(
    x: Float[Array, "n d"], y: Float[Array, "n d"], epsilon: float = 1e-6
) -> Float[Array, "n n"]:
    """Compute pairwise Euclidean distances between points."""
    diff = x[:, None, :] - y[None, :, :]
    return jnp.sqrt(jnp.sum(diff**2, axis=-1) + epsilon)


def mink_neighbors(
    x: Float[Array, "n n"], k: int, mask: Bool[Array, " n"]
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
    mu = jnp.linspace(dmin, dmax, dim)
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

    max_relative_offset: int
    linear: nn.Linear

    def __init__(self, dim: int, max_relative_offset: int = 32, *, key: PRNGKeyArray) -> None:
        self.max_relative_offset = max_relative_offset
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
            offset + self.max_relative_offset, min=0, max=2 * self.max_relative_offset
        ) * chain_mask + (1 - chain_mask) * (2 * self.max_relative_offset + 1)
        one_hot = jax.nn.one_hot(clipped, 2 * self.max_relative_offset + 1 + 1)
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

    radial_bases = [radial_basis_function(ca_distances, dmin=dmin, dmax=dmax, dim=rbf_dim)]
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

        self.edge_embedding = nn.Linear(num_pos_emb + rbf_dim * 25, dim, use_bias=False, key=key1)
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
            residue_index=residue_index, edge_index=edge_index, chain_mask=chain_mask
        )

        edges = jnp.concat([positional_emb, radial_basis], axis=-1)
        edges = jax.vmap(jax.vmap(self.edge_embedding))(edges)
        edges = jax.vmap(jax.vmap(self.edge_norm))(edges)

        return edges, edge_index


def make_feedforward(dim: int, factor: int, *, key: PRNGKeyArray) -> nn.Sequential:
    """Helper function to create a feedforward layer."""
    key1, key2 = jr.split(key)
    return nn.Sequential(
        [
            nn.Linear(dim, dim * factor, key=key1),
            nn.Lambda(gelu),
            nn.Linear(dim * factor, dim, key=key2),
        ]
    )


def make_subblock(in_dim: int, dim: int, *, key: PRNGKeyArray) -> nn.Sequential:
    """Helper function to create a 3-layer MLP used to update edge messages."""
    key1, key2, key3 = jr.split(key, 3)
    return nn.Sequential(
        [
            nn.Linear(in_dim, dim, key=key1),
            nn.Lambda(gelu),
            nn.Linear(dim, dim, key=key2),
            nn.Lambda(gelu),
            nn.Linear(dim, dim, key=key3),
        ]
    )


class EncoderBlock(eqx.Module):
    """Encoder block for the MPNN. Updates node and edge representations
    using structural features.
    """

    scale: float

    in_block: nn.Sequential
    out_block: nn.Sequential

    dropout1: nn.Dropout
    dropout2: nn.Dropout
    dropout3: nn.Dropout

    norm1: nn.LayerNorm
    norm2: nn.LayerNorm
    norm3: nn.LayerNorm

    feedforward: nn.Sequential

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

        self.in_block = make_subblock(in_dim=dim * 3, dim=dim, key=key1)
        self.out_block = make_subblock(in_dim=dim * 3, dim=dim, key=key2)

        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.dropout3 = nn.Dropout(dropout_rate)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

        self.feedforward = make_feedforward(dim=dim, factor=4, key=key3)

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
        key1, key2, key3 = jr.split(key, 3)

        message = jnp.concat(
            [
                einops.repeat(nodes, "n d -> n k d", k=edges.shape[1]),
                edges,
                gather_nodes(nodes, edge_index),
            ],
            axis=-1,
        )
        message = jax.vmap(jax.vmap(self.in_block))(message)

        # attention mask
        message = message * einops.rearrange(mask_edges, "n k -> n k ()")

        # aggregate message
        message = einops.reduce(message, "n k d -> n d", "sum") / self.scale

        # dropout and norm
        message = self.dropout1(message, key=key1, inference=not enable_dropout)
        out_nodes = jax.vmap(self.norm1)(nodes + message)

        # update representation
        message = jax.vmap(self.feedforward)(out_nodes)

        # dropout and norm
        message = self.dropout2(message, key=key2, inference=not enable_dropout)
        out_nodes = jax.vmap(self.norm2)(out_nodes + message)

        # node mask
        out_nodes = out_nodes * einops.rearrange(mask_nodes, "n -> n ()")

        # update edge representation
        message = jnp.concat(
            [
                einops.repeat(out_nodes, "n d -> n k d", k=edges.shape[1]),
                edges,
                gather_nodes(out_nodes, edge_index),
            ],
            axis=-1,
        )

        message = jax.vmap(jax.vmap(self.out_block))(message)
        message = self.dropout3(message, key=key3, inference=not enable_dropout)
        out_edges = jax.vmap(jax.vmap(self.norm3))(edges + message)
        return out_nodes, out_edges


class DecoderBlock(eqx.Module):
    """Decoder block for the MPNN. Updates node representations using
    structural and sequence features.
    """

    scale: float

    block: nn.Sequential

    dropout1: nn.Dropout
    dropout2: nn.Dropout

    norm1: nn.LayerNorm
    norm2: nn.LayerNorm

    feedforward: nn.Sequential

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

        self.block = make_subblock(in_dim=dim * 4, dim=dim, key=key1)

        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.feedforward = make_feedforward(dim=dim, factor=4, key=key2)

    def __call__(
        self,
        nodes: Float[Array, "n d"],
        edges: Float[Array, "n k dim"],
        enable_dropout: bool,
        mask_nodes: Bool[Array, " n"],
        key: PRNGKeyArray,
    ) -> Float[Array, "n dim"]:
        key1, key2 = jr.split(key, 2)

        message = jnp.concat(
            [einops.repeat(nodes, "n d -> n k d", k=edges.shape[1]), edges],
            axis=-1,
        )
        message = jax.vmap(jax.vmap(self.block))(message)

        message = einops.reduce(message, "n k d -> n d", "sum") / self.scale
        message = self.dropout1(message, key=key1, inference=not enable_dropout)
        out_nodes = jax.vmap(self.norm1)(nodes + message)

        message = jax.vmap(self.feedforward)(out_nodes)
        message = self.dropout2(message, key=key2, inference=not enable_dropout)
        out_nodes = jax.vmap(self.norm2)(out_nodes + message)

        return out_nodes * einops.rearrange(mask_nodes, "n -> n ()")


def build_decoding_attention_mask(
    decoding_order: Int[Array, " n"], edge_index: Int[Array, "n k"]
) -> Float[Array, "n k"]:
    """Helper function used to build the forward and backward attention masks."""
    n = decoding_order.shape[0]
    arr, current = jnp.zeros((n, n)), jnp.zeros((n,))
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


def keep_top_k(x: Float[Array, " n"], k: int) -> Float[Array, " n"]:
    """Keep only the top-k elements, set others to -inf."""
    _, indices = jax.lax.top_k(x, k=k)
    mask = jnp.zeros_like(x, dtype=bool)
    mask = mask.at[indices].set(True)
    return x * mask + (1 - mask) * -1e9


class ProteinMPNN(eqx.Module):
    """The ProteinMPNN model."""

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
        vocab: int,
        dropout_rate: float,
        *,
        key: Array,
    ) -> None:
        key1, key2, key3, key4, key5, key6 = jr.split(key, 6)

        # feature embedding
        self.features = FeatureEmbedding(dim=dim, k=k, key=key1)

        # encoder blocks
        self.edge_linear = nn.Linear(dim, dim, key=key2)

        keys = jr.split(key3, num_encoder_blocks)
        self.encoder_blocks = [
            EncoderBlock(dim=dim, dropout_rate=dropout_rate, key=key) for key in keys
        ]

        # decoder blocks
        self.sequence_embedding = nn.Embedding(vocab, dim, key=key4)

        keys = jr.split(key5, num_decoder_blocks)
        self.decoder_blocks = [
            DecoderBlock(dim=dim, dropout_rate=dropout_rate, key=key) for key in keys
        ]

        # output
        self.linear_out = nn.Linear(dim, vocab, key=key6)

    @eqx.filter_jit
    def encode(
        self,
        pos: Float[Array, "n 4 3"],
        residue_index: Int[Array, " n"],
        chain_labels: Int[Array, " n"],
        enable_dropout: bool,
        mask_nodes: Bool[Array, " n"],
        key: PRNGKeyArray,
    ) -> tuple[
        Float[Array, "n dim"],
        Float[Array, "n k dim"],
        Int[Array, "n k"],
    ]:
        edges, edge_index = self.features(
            backbone_pos=pos,
            residue_index=residue_index,
            chain_labels=chain_labels,
            mask=mask_nodes,
        )

        n, _, in_edge_dim = edges.shape
        nodes = jnp.zeros((n, in_edge_dim))

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

    @eqx.filter_jit
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
        sequence_emb = jnp.concat([edges, gather_nodes(sequence_emb, edge_index)], axis=-1)

        mask_backward, mask_forward = build_forward_backward_mask(
            decoding_order=decoding_order, edge_index=edge_index, mask=mask_nodes
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
                [sequence_emb, gather_nodes(nodes, edge_index=edge_index)], axis=-1
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

        # the original implementation returns log softmax outputs
        # return jax.nn.log_softmax(logits)
        return logits

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

    @classmethod
    def from_pretrained(cls, name: str, *, key: PRNGKeyArray) -> "ProteinMPNN":
        """Load a pretrained ProteinMPNN model."""
        weights_path = get_weights_path(name)
        if not weights_path.is_file():
            loguru.logger.info(
                f"Weights file not found: {weights_path}, converting weights from torch."
            )
            convert_torch_to_equinox(name)
        model = ProteinMPNN(**constants.DEFAULT_HYPERPARAMS, key=key)
        model = eqx.tree_deserialise_leaves(path_or_file=weights_path, like=model)
        return model

    def sample(
        self,
        sequence: Int[Array, " n"],
        pos: Float[Array, "n 4 3"],
        residue_index: Int[Array, " n"],
        chain_labels: Int[Array, " n"],
        mask_nodes: Bool[Array, " n"],
        # decoding order is used when fixing residues and/or chains
        decoding_order: Int[Array, " n"],
        decoding_start_index: int = 0,
        temperature: float = 1.0,
        top_k: int = 1,
        *,
        key: PRNGKeyArray,
        progress_bar: bool = False,
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

        keys = jr.split(key, n)
        for idx in tqdm.tqdm(range(decoding_start_index, n), disable=not progress_bar):
            key1, key2 = jr.split(keys[idx])
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
                sampled = jnp.argmax(logits[decoding_order[idx]])
            else:
                top_k_logits = keep_top_k(logits[decoding_order[idx]], k=top_k)
                probs = jax.nn.softmax(top_k_logits / temperature)
                sampled = jr.choice(key=key2, a=logits.shape[-1], p=probs, shape=())

            sequence = sequence.at[decoding_order[idx]].set(sampled)

        return sequence


def update_eqx_with_state_dict(
    module: eqx.Module, state_dict: dict[str, torch.Tensor], conversion_map: dict[str, str]
) -> eqx.Module:
    path_vals, treedef = jax.tree.flatten_with_path(module)
    updated_path_vals = []
    count = 0
    for names, array in path_vals:
        key = ".".join(str(x).strip(".") for x in names)
        if key in conversion_map:
            weights = state_dict[conversion_map[key]]
            assert array.shape == weights.shape, f"{array.shape} != {weights.shape} for {key=}"
            updated_path_vals.append((names, jnp.asarray(weights)))
            count += 1
        else:
            updated_path_vals.append((names, array))

    updated_leaves = [v for _, v in updated_path_vals]
    updated_module = jax.tree.unflatten(treedef, updated_leaves)

    if not count == len(conversion_map):
        raise ValueError("Did not find all keys in conversion map.")
    return updated_module


def get_weights_path(name: str, framework: Literal["torch", "jax"] = "jax") -> Path:
    """Build local path to stored model weights."""
    weights_dir = Path(__file__).parent.parent / "data" / "weights"
    extensions = {"torch": "pt", "jax": "eqx"}
    return weights_dir / framework / f"{name}.{extensions[framework]}"


def convert_torch_to_equinox(name: str) -> None:
    """Convert torch model weights to equinox and save to file."""
    torch_weights_path = get_weights_path(name, framework="torch")
    state_dict = torch.load(torch_weights_path, map_location="cpu")["model_state_dict"]

    model = model_.ProteinMPNN(**constants.DEFAULT_HYPERPARAMS, key=jr.PRNGKey(0))
    updated_model = update_eqx_with_state_dict(model, state_dict, load_conversion_map())

    weights_path = get_weights_path(name, framework="jax")
    if not weights_path.parent.is_dir():
        weights_path.parent.mkdir(parents=True, exist_ok=True)

    eqx.tree_serialise_leaves(weights_path, updated_model)

    if not weights_path.is_file():
        raise Exception(f"Failed to save converted weights to {weights_path}.")

    loguru.logger.info(f"Converted weights saved to {weights_path}.")


def load_conversion_map() -> dict[str, str]:
    """Load the conversion map from a TOML file."""

    conversion_map = {
        "features.positional_encoding.linear.weight": "features.embeddings.linear.weight",
        "features.positional_encoding.linear.bias": "features.embeddings.linear.bias",
        "features.edge_embedding.weight": "features.edge_embedding.weight",
        "features.edge_norm.weight": "features.norm_edges.weight",
        "features.edge_norm.bias": "features.norm_edges.bias",
        "edge_linear.weight": "W_e.weight",
        "edge_linear.bias": "W_e.bias",
        "linear_out.weight": "W_out.weight",
        "linear_out.bias": "W_out.bias",
        "sequence_embedding.weight": "W_s.weight",
    }

    for k in range(constants.DEFAULT_HYPERPARAMS["num_encoder_blocks"]):
        conversion_map.update(
            {
                f"encoder_blocks.[{k}].in_block.layers.[0].weight": f"encoder_layers.{k}.W1.weight",
                f"encoder_blocks.[{k}].in_block.layers.[0].bias": f"encoder_layers.{k}.W1.bias",
                f"encoder_blocks.[{k}].in_block.layers.[2].weight": f"encoder_layers.{k}.W2.weight",
                f"encoder_blocks.[{k}].in_block.layers.[2].bias": f"encoder_layers.{k}.W2.bias",
                f"encoder_blocks.[{k}].in_block.layers.[4].weight": f"encoder_layers.{k}.W3.weight",
                f"encoder_blocks.[{k}].in_block.layers.[4].bias": f"encoder_layers.{k}.W3.bias",
                f"encoder_blocks.[{k}].out_block.layers.[0].weight": f"encoder_layers.{k}.W11.weight",
                f"encoder_blocks.[{k}].out_block.layers.[0].bias": f"encoder_layers.{k}.W11.bias",
                f"encoder_blocks.[{k}].out_block.layers.[2].weight": f"encoder_layers.{k}.W12.weight",
                f"encoder_blocks.[{k}].out_block.layers.[2].bias": f"encoder_layers.{k}.W12.bias",
                f"encoder_blocks.[{k}].out_block.layers.[4].weight": f"encoder_layers.{k}.W13.weight",
                f"encoder_blocks.[{k}].out_block.layers.[4].bias": f"encoder_layers.{k}.W13.bias",
                f"encoder_blocks.[{k}].norm1.weight": f"encoder_layers.{k}.norm1.weight",
                f"encoder_blocks.[{k}].norm1.bias": f"encoder_layers.{k}.norm1.bias",
                f"encoder_blocks.[{k}].norm2.weight": f"encoder_layers.{k}.norm2.weight",
                f"encoder_blocks.[{k}].norm2.bias": f"encoder_layers.{k}.norm2.bias",
                f"encoder_blocks.[{k}].norm3.weight": f"encoder_layers.{k}.norm3.weight",
                f"encoder_blocks.[{k}].norm3.bias": f"encoder_layers.{k}.norm3.bias",
                f"encoder_blocks.[{k}].feedforward.layers.[0].weight": f"encoder_layers.{k}.dense.W_in.weight",
                f"encoder_blocks.[{k}].feedforward.layers.[0].bias": f"encoder_layers.{k}.dense.W_in.bias",
                f"encoder_blocks.[{k}].feedforward.layers.[2].weight": f"encoder_layers.{k}.dense.W_out.weight",
                f"encoder_blocks.[{k}].feedforward.layers.[2].bias": f"encoder_layers.{k}.dense.W_out.bias",
            }
        )

    for k in range(constants.DEFAULT_HYPERPARAMS["num_decoder_blocks"]):
        conversion_map.update(
            {
                f"decoder_blocks.[{k}].block.layers.[0].weight": f"decoder_layers.{k}.W1.weight",
                f"decoder_blocks.[{k}].block.layers.[0].bias": f"decoder_layers.{k}.W1.bias",
                f"decoder_blocks.[{k}].block.layers.[2].weight": f"decoder_layers.{k}.W2.weight",
                f"decoder_blocks.[{k}].block.layers.[2].bias": f"decoder_layers.{k}.W2.bias",
                f"decoder_blocks.[{k}].block.layers.[4].weight": f"decoder_layers.{k}.W3.weight",
                f"decoder_blocks.[{k}].block.layers.[4].bias": f"decoder_layers.{k}.W3.bias",
                f"decoder_blocks.[{k}].norm1.weight": f"decoder_layers.{k}.norm1.weight",
                f"decoder_blocks.[{k}].norm1.bias": f"decoder_layers.{k}.norm1.bias",
                f"decoder_blocks.[{k}].norm2.weight": f"decoder_layers.{k}.norm2.weight",
                f"decoder_blocks.[{k}].norm2.bias": f"decoder_layers.{k}.norm2.bias",
                f"decoder_blocks.[{k}].feedforward.layers.[0].weight": f"decoder_layers.{k}.dense.W_in.weight",
                f"decoder_blocks.[{k}].feedforward.layers.[0].bias": f"decoder_layers.{k}.dense.W_in.bias",
                f"decoder_blocks.[{k}].feedforward.layers.[2].weight": f"decoder_layers.{k}.dense.W_out.weight",
                f"decoder_blocks.[{k}].feedforward.layers.[2].bias": f"decoder_layers.{k}.dense.W_out.bias",
            }
        )

    return conversion_map
