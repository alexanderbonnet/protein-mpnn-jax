import pytest

import proteinmpnn


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
        fixed_positions=[],
        fixed_chains=[],
        seed=0,
        progress_bar=False,
    )

    assert sampled[0]["sequence"] == expected_sequence

    rounded_probabilities = [round(p, 2) for p in sampled[0]["probabilities"]]
    assert rounded_probabilities[:15] == expected_probabilities
