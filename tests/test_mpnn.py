import jax.numpy as jnp
import jax.random as jr
import pytest

import proteinmpnn
from proteinmpnn import mpnn, parse


@pytest.mark.parametrize(
    "backbone_path, expected_sequence, expected_probabilities",
    [
        (
            "examples/inputs/monomers/5L33.cif.gz",
            "SIDEDELVALKFIEALEKADPELMKKVISPDTELEWNGKKFKGEEIVEFVKEVAKEGTKYTLVSYEKEGDEFVFNVKVEKDGETRNATVRITVKDGKVAKVEITCE",
            [0.79, 0.48, 1.0, 0.97, 0.71, 1.0, 0.82, 1.0, 1.0, 1.0, 0.53, 1.0, 0.99, 0.6, 1.0],
        ),
    ],
)
def test_sampling(
    backbone_path: str, expected_sequence: str, expected_probabilities: list[float]
) -> None:
    sampled = proteinmpnn.sample(
        model_name="soluble/v_48_002",
        backbone_path=backbone_path,
        top_k=1,
        temperature=0.1,
        fixed_positions=None,
        fixed_chains=None,
        seed=0,
        progress_bar=False,
    )

    assert sampled[0]["sequence"] == expected_sequence

    rounded_probabilities = [round(p, 2) for p in sampled[0]["probabilities"]]
    assert rounded_probabilities[:15] == expected_probabilities


@pytest.mark.parametrize(
    "backbone_path",
    ["examples/inputs/monomers/5L33.cif.gz"],
)
def test_masking(backbone_path: str) -> None:
    to_pad = 10

    backbone = parse.read_structure(str(backbone_path), use_assembly=True)
    residues = parse.parse_backbone(backbone)
    inputs = parse.prepare_tensors(residues)

    padded_inputs = inputs.pad(n=inputs.pos.shape[0] + to_pad)

    key1, key2, key3 = jr.split(jr.PRNGKey(74), num=3)

    decoding_order = jnp.arange(inputs.pos.shape[0], dtype=jnp.int32)

    decoding_order_padding = jnp.ones(shape=(to_pad,), dtype=jnp.int32) + jnp.max(decoding_order)
    padded_decoding_order = jnp.concatenate([decoding_order, decoding_order_padding], axis=0)

    model: mpnn.ProteinMPNN
    model = mpnn.ProteinMPNN.from_pretrained("soluble/v_48_002", key=key2)

    outputs = model(
        sequence=inputs.restypes,
        pos=inputs.pos,
        residue_index=inputs.residue_index,
        chain_labels=inputs.chain_labels,
        mask_nodes=inputs.mask,
        decoding_order=decoding_order,
        enable_dropout=False,
        key=key3,
    )

    outputs_padded = model(
        sequence=padded_inputs.restypes,
        pos=padded_inputs.pos,
        residue_index=padded_inputs.residue_index,
        chain_labels=padded_inputs.chain_labels,
        mask_nodes=padded_inputs.mask,
        decoding_order=padded_decoding_order,
        enable_dropout=False,
        key=key3,
    )

    assert jnp.allclose(outputs, outputs_padded[: outputs.shape[0], :])
