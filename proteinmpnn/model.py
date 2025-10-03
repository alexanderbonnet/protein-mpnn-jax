import einops
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from equinox import nn
from jax import Array
from jaxtyping import Float, Int

from proteinmpnn import constants


def pairwise_distances(
    x: Float[jnp.ndarray, "n d"],
    y: Float[jnp.ndarray, "n d"],
    epsilon: float = constants.EPS,
) -> Float[jnp.ndarray, "n n"]:
    """Compute pairwise Euclidean distances between points."""
    if x.shape != y.shape:
        raise ValueError(
            f"x and y must have the same shape, got {x.shape} != {y.shape}"
        )

    diff = x[:, None, :] - y[None, :, :]
    return jnp.sqrt(jnp.sum(diff**2, axis=-1) + epsilon)


# def mink_neighbors(
#     x: Float[jnp.ndarray, "n n"], k: int
# ) -> tuple[Float[jnp.ndarray, "n k"], Int[jnp.ndarray, "n k"]]:
#     """Compute a mask for the top-k values in each row of a matrix."""
#     if k <= 0:
#         raise ValueError("k must be positive")
#     n, _ = x.shape
#     distances, indices = jax.lax.top_k(-x, min(k, n))
#     return -distances, indices


def radial_basis_function(
    x: Float[jnp.ndarray, "n k"], dmin: float, dmax: float, dim: int
) -> Float[jnp.ndarray, "n k dim"]:
    mu = jnp.linspace(dmin, dmax, dim, device=x.device)
    beta = (dim / (dmax - dmin)) ** 2
    return jnp.exp(-beta * (einops.rearrange(x, "n k -> n k ()") - mu) ** 2)


def backbone_features(
    backbone_pos: Float[jnp.ndarray, "n 5 3"],
    dmin: float = 2.0,
    dmax: float = 22.0,
    rbf_dim: int = 16,
) -> Float[jnp.ndarray, "n 3"]:
    """Compute backbone features for a protein structure."""
    ca = backbone_pos[:, constants.ATOM_INDICES["CA"], :]
    ca_distances = pairwise_distances(ca, ca)

    radial_basis_ = [
        radial_basis_function(ca_distances, dmin=dmin, dmax=dmax, dim=rbf_dim)
    ]
    for atom1, atom2 in constants.ATOM_PAIR_RBFS:
        idx1 = constants.ATOM_INDICES[atom1]
        idx2 = constants.ATOM_INDICES[atom2]
        d = pairwise_distances(backbone_pos[:, idx1, :], backbone_pos[:, idx2, :])
        rbf = radial_basis_function(d, dmin=dmin, dmax=dmax, dim=rbf_dim)
        radial_basis_.append(rbf)

    radial_basis = einops.rearrange(radial_basis_, "b n1 n2 d -> n1 n2 (b d)")

    return radial_basis


def gather_edges(
    edges: Float[jnp.ndarray, "n n ..."], edge_index: Int[jnp.ndarray, "n k"]
) -> Float[jnp.ndarray, "n k ..."]:
    """Gather edge features at neighbor indices."""
    # Get dimensions
    n, k = edge_index.shape
    i_idx = jnp.arange(n)[:, None]
    i_idx = jnp.broadcast_to(i_idx, (n, k))
    return edges[i_idx, edge_index, ...]


class PositionalEmbedding(eqx.Module):
    def __init__(self, dim: int, max_relative_offset: int = 32, *, key: Array) -> None:
        self.max_offset = max_relative_offset
        self.linear = nn.Linear(2 * max_relative_offset + 1 + 1, dim, key=key)

    def __call__(
        self,
        residue_index: Int[jnp.ndarray, " n"],
        edge_index: Int[jnp.ndarray, " n k"],
    ) -> Float[jnp.ndarray, " n ne dim"]:
        offset = residue_index[:, None] - residue_index[None, :]
        offset = gather_edges(offset.astype(jnp.int32), edge_index)

        clipped = jnp.clip(offset - self.max_offset, 0, 2 * self.max_offset + 1)
        one_hot = jax.nn.one_hot(
            clipped, 2 * self.max_offset + 1 + 1, dtype=jnp.float32
        )
        return self.linear(one_hot)


class FeedForward(eqx.Module):
    def __init__(self, dim: int, factor: int = 4, *, key: Array) -> None:
        key1, key2 = jr.split(key, 2)
        self.linear_in = nn.Linear(dim, dim * factor, use_bias=False, key=key1)
        self.linear_out = nn.Linear(dim * factor, dim, use_bias=False, key=key2)

    def __call__(
        self, x: Float[jnp.ndarray, " ... dim"]
    ) -> Float[jnp.ndarray, " ... dim"]:
        x = self.linear_in(x)
        x = jax.nn.gelu(x)
        return self.linear_out(x)
