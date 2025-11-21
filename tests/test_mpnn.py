import pytest

import proteinmpnn


@pytest.mark.parametrize(
    "backbone_path, expected_sequence",
    [
        (
            "examples/inputs/monomers/5L33.cif.gz",
            "SIDEDELVALKFIEALEKADPELMKKVISPDTELEWNGKKFKGEEIVEFVKEVAKEGTKYTLVSYEKEGDEFVFNVKVEKDGETRNATVRITVKDGKVAKVEITCE",
        ),
    ],
)
def test_sampling(backbone_path: str, expected_sequence: str) -> None:
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
