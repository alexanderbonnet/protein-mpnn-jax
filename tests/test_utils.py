import jax.numpy as jnp
import jax.random as jr
import pytest
from jaxtyping import Array, Int

from proteinmpnn import utils


@pytest.mark.parametrize(
    "backbone_chains, fixed_positions, fixed_chains, expected_order, expected_start",
    [
        (
            ["A", "A", "A", "A", "B", "B", "C"],
            [0, 2, 6],
            ["B"],
            jnp.array([0, 2, 4, 5, 6, 1, 3]),
            5,
        ),
        (
            ["A", "A", "A"],
            [],
            [],
            jnp.array([0, 1, 2]),
            0,
        ),
    ],
)
def test_build_decoding_order(
    backbone_chains: list[str],
    fixed_positions: list[int],
    fixed_chains: list[str],
    expected_order: Int[Array, " n"],
    expected_start: int,
) -> None:
    decoding_order, decoding_start_index = utils.build_decoding_order(
        backbone_chains=backbone_chains,
        fixed_positions=fixed_positions,
        fixed_chains=fixed_chains,
        key=jr.PRNGKey(0),
    )

    assert jnp.all(decoding_order == expected_order)
    assert jnp.all(decoding_start_index == expected_start)
