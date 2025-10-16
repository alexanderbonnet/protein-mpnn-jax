import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Int, PRNGKeyArray


def build_decoding_order(
    backbone_chains: list[str],
    fixed_positions: list[int] | None = None,
    fixed_chains: list[str] | None = None,
    *,
    # used to generate the random decoding order for unfixed residues
    key: PRNGKeyArray,
) -> tuple[Int[Array, " n"], int]:
    """Build a decoding order for the residues taking into account fixed
    positions and chains.

    Places the fixed positions first in the decoding order, followed by the
    unfixed positions in random order. A decoding start index, indicating the
    index in the decoding order where the unfixed positions start, is also
    returned.

    Used during the sampling process to appropriately mask the input sequence.
    """
    if fixed_positions is None:
        fixed_positions = []
    if fixed_chains is None:
        fixed_chains = []

    fixed_positions_set = set(fixed_positions)
    fixed_chains_set = set(fixed_chains)
    for idx, chain in enumerate(backbone_chains):
        if chain in fixed_chains_set:
            fixed_positions_set.add(idx)

    # order does not matter for the fixed positions
    fixed_positions = jnp.array(list(fixed_positions_set))
    decoding_start_index = fixed_positions.shape[0]

    free_positions = jnp.array(
        [idx for idx in range(len(backbone_chains)) if idx not in fixed_positions_set]
    )

    # randomize the decoding order for the free positions
    free_positions = jr.permutation(key=key, x=free_positions)

    decoding_order = jnp.concatenate(
        [fixed_positions, free_positions], axis=0, dtype=jnp.int32
    )
    return decoding_order, decoding_start_index
